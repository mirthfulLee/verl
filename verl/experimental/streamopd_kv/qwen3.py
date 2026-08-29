# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import types
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from .attention import BatchedReverseChunkState, LayerKVTrace, ReverseChunkState


def capture_qwen3_kv_trace(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[LayerKVTrace, ...]:
    """Build the reference trace used to validate a serving-time KV handoff."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("the Qwen3 correctness backend currently supports one trajectory at a time")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    if was_training:
        model.train()

    cache = output.past_key_values
    layers = getattr(cache, "layers", None)
    if layers is None:
        try:
            legacy = cache.to_legacy_cache()
        except AttributeError as exc:
            raise TypeError("unsupported Transformers KV cache representation") from exc
        return tuple(LayerKVTrace(key.detach(), value.detach()) for key, value in legacy)
    return tuple(LayerKVTrace(layer.keys.detach(), layer.values.detach()) for layer in layers)


@contextmanager
def use_qwen3_reverse_attention(model: torch.nn.Module, state: ReverseChunkState) -> Iterator[None]:
    """Replace Qwen3 attention with exact prefix-KV reverse attention for one traversal."""

    try:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    except ImportError as exc:
        raise RuntimeError("Qwen3 reverse training requires transformers with Qwen3 support") from exc

    unwrapped = getattr(model, "module", model)
    layers = getattr(getattr(unwrapped, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a Qwen3ForCausalLM-compatible model")
    if len(layers) != len(state.layers):
        raise ValueError(f"model has {len(layers)} layers but KV trace has {len(state.layers)}")

    originals: list[tuple[torch.nn.Module, Any]] = []

    def make_forward(layer_idx: int):
        def reverse_forward(
            attention: torch.nn.Module,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values: Any = None,
            cache_position: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            del attention_mask, past_key_values, cache_position, kwargs
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attention.head_dim)
            query = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            output = state.attention(layer_idx, query, key, value, scale=attention.scaling)
            output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            return attention.o_proj(output), None

        return reverse_forward

    try:
        for layer_idx, decoder_layer in enumerate(layers):
            attention = decoder_layer.self_attn
            originals.append((attention, attention.forward))
            attention.forward = types.MethodType(make_forward(layer_idx), attention)
        yield
    finally:
        for attention, original in originals:
            attention.forward = original


@dataclass(frozen=True)
class ReverseTrainingResult:
    loss_sum: torch.Tensor
    valid_tokens: int
    chunks: int
    backward_calls: int = 0


class Qwen3ReverseTrainer:
    """Correctness-first dense OOMB traversal over a sealed Qwen3 KV trace.

    The caller owns loss normalization and the optimizer step. ``loss_fn`` must
    return an unnormalized token-loss sum and the number of valid tokens for the
    global positions in ``[start, end)``.
    """

    def __init__(self, model: torch.nn.Module, chunk_size: int) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if model.training:
            raise ValueError("Qwen3 reverse training requires model.eval() so rollout and recomputation match")
        self.model = model
        self.chunk_size = chunk_size

    def backward(
        self,
        input_ids: torch.Tensor,
        layers: Sequence[LayerKVTrace],
        loss_fn: Callable[[torch.Tensor, int, int], tuple[torch.Tensor, int]],
        *,
        release_processed_suffix: bool = True,
        backward_context: Callable[[int, int], AbstractContextManager] | None = None,
    ) -> ReverseTrainingResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the Qwen3 correctness backend currently supports one trajectory at a time")
        state = ReverseChunkState(list(layers))
        sequence_length = input_ids.shape[1]
        if state.sequence_length != sequence_length:
            raise ValueError(f"input/KV token length mismatch: input={sequence_length}, KV={state.sequence_length}")

        loss_sum = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        valid_tokens = 0
        chunks = 0
        total_chunks = (sequence_length + self.chunk_size - 1) // self.chunk_size
        with use_qwen3_reverse_attention(self.model, state):
            for end in range(sequence_length, 0, -self.chunk_size):
                start = max(0, end - self.chunk_size)
                state.begin(start, end)
                position_ids = torch.arange(start, end, device=input_ids.device).unsqueeze(0)
                output = self.model(
                    input_ids=input_ids[:, start:end],
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                chunk_loss, chunk_valid_tokens = loss_fn(output.logits, start, end)
                if chunk_loss.ndim != 0 or chunk_valid_tokens < 0:
                    raise ValueError("loss_fn must return a scalar loss sum and a non-negative token count")
                sync_context = backward_context(chunks, total_chunks) if backward_context else nullcontext()
                with sync_context:
                    (chunk_loss + state.gradient_injection()).backward()
                state.commit_prefix_gradients(release_processed_suffix=release_processed_suffix)
                loss_sum = loss_sum + chunk_loss.detach().float()
                valid_tokens += chunk_valid_tokens
                chunks += 1
        return ReverseTrainingResult(loss_sum=loss_sum, valid_tokens=valid_tokens, chunks=chunks)

    def backward_batched(
        self,
        input_ids: Sequence[torch.Tensor],
        layers: Sequence[Sequence[LayerKVTrace]],
        loss_fns: Sequence[Callable[[torch.Tensor, int, int], tuple[torch.Tensor, int]]],
        *,
        release_processed_suffix: bool = True,
        backward_context: Callable[[int, int], AbstractContextManager] | None = None,
    ) -> ReverseTrainingResult:
        """Reverse a ragged trajectory microbatch in batched suffix waves."""

        if not input_ids or not (len(input_ids) == len(layers) == len(loss_fns)):
            raise ValueError("input trajectories, KV traces, and loss functions must be non-empty and aligned")
        sequences = []
        lengths = []
        for sequence, trajectory_layers in zip(input_ids, layers, strict=True):
            if sequence.ndim != 2 or sequence.shape[0] != 1:
                raise ValueError("each Qwen3 reverse trajectory must have shape [1, tokens]")
            if not trajectory_layers:
                raise ValueError("each Qwen3 reverse trajectory requires KV layers")
            trace_lengths = {layer.length for layer in trajectory_layers}
            if trace_lengths != {sequence.shape[1]}:
                raise ValueError("input/KV token length mismatch in reverse microbatch")
            sequences.append(sequence)
            lengths.append(sequence.shape[1])

        schedule: list[tuple[list[int], list[int], list[int]]] = []
        ends = list(lengths)
        while active := [idx for idx, end in enumerate(ends) if end]:
            starts = [max(0, ends[idx] - self.chunk_size) for idx in active]
            active_ends = [ends[idx] for idx in active]
            schedule.append((active, starts, active_ends))
            for idx, start in zip(active, starts, strict=True):
                ends[idx] = start

        state = BatchedReverseChunkState(layers)
        loss_sum = torch.zeros((), device=sequences[0].device, dtype=torch.float32)
        valid_tokens = 0
        total_chunks = sum((length + self.chunk_size - 1) // self.chunk_size for length in lengths)
        with use_qwen3_reverse_attention(self.model, state):
            for call_idx, (active, starts, ends) in enumerate(schedule):
                state.begin(active, starts, ends)
                chunk_lengths = [end - start for start, end in zip(starts, ends, strict=True)]
                max_chunk_length = max(chunk_lengths)
                padded_chunks = []
                for idx, start, end, length in zip(active, starts, ends, chunk_lengths, strict=True):
                    chunk = sequences[idx][:, start:end]
                    if length < max_chunk_length:
                        padding = torch.zeros((1, max_chunk_length - length), dtype=chunk.dtype, device=chunk.device)
                        chunk = torch.cat((chunk, padding), dim=1)
                    padded_chunks.append(chunk)
                chunk_ids = torch.cat(padded_chunks)
                position_ids = torch.stack(
                    [torch.arange(start, start + max_chunk_length, device=chunk_ids.device) for start in starts]
                )
                output = self.model(
                    input_ids=chunk_ids,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                chunk_loss = output.logits.sum() * 0.0
                chunk_valid_tokens = 0
                for row, (idx, start, end) in enumerate(zip(active, starts, ends, strict=True)):
                    length = end - start
                    sample_loss, sample_valid_tokens = loss_fns[idx](output.logits[row : row + 1, :length], start, end)
                    if sample_loss.ndim != 0 or sample_valid_tokens < 0:
                        raise ValueError("loss_fn must return a scalar loss sum and a non-negative token count")
                    chunk_loss = chunk_loss + sample_loss
                    chunk_valid_tokens += sample_valid_tokens
                sync_context = backward_context(call_idx, len(schedule)) if backward_context else nullcontext()
                with sync_context:
                    (chunk_loss + state.gradient_injection()).backward()
                state.commit_prefix_gradients(release_processed_suffix=release_processed_suffix)
                loss_sum = loss_sum + chunk_loss.detach().float()
                valid_tokens += chunk_valid_tokens
        return ReverseTrainingResult(
            loss_sum=loss_sum,
            valid_tokens=valid_tokens,
            chunks=total_chunks,
            backward_calls=len(schedule),
        )
