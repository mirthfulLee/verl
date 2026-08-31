# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import math
import types
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from .attention import LayerKVTrace


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
def use_qwen3_reverse_attention(model: torch.nn.Module, state: Any) -> Iterator[None]:
    """Replace Qwen3 attention with paged reverse attention for one traversal."""

    try:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    except ImportError as exc:
        raise RuntimeError("Qwen3 reverse training requires transformers with Qwen3 support") from exc

    unwrapped = getattr(model, "module", model)
    layers = getattr(getattr(unwrapped, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a Qwen3ForCausalLM-compatible model")
    state_layer_count = getattr(state, "num_layers", None)
    if state_layer_count is None:
        state_layer_count = len(state.layers)
    if len(layers) != state_layer_count:
        raise ValueError(f"model has {len(layers)} layers but KV trace has {state_layer_count}")

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
    max_parallel_trajectories: int = 1


def _build_reverse_wavefront(sequence_lengths: Sequence[int], chunk_size: int) -> list[tuple[int, list[int]]]:
    if not sequence_lengths or chunk_size < 1 or any(length < 1 for length in sequence_lengths):
        raise ValueError("wavefront scheduling requires positive sequence lengths and chunk size")
    chunk_counts = [math.ceil(length / chunk_size) for length in sequence_lengths]
    return [
        (depth, [idx for idx, count in enumerate(chunk_counts) if count >= depth])
        for depth in range(max(chunk_counts), 0, -1)
    ]


class Qwen3ReverseTrainer:
    """Dense OOMB paged traversal over a sealed Qwen3 KV trace.

    The caller owns loss normalization and the optimizer step. ``loss_fn`` must
    return an unnormalized token-loss sum and the number of valid tokens for the
    global positions in ``[start, end)``.
    """

    def __init__(self, model: torch.nn.Module, chunk_size: int, *, page_size: int = 64) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if page_size < 1 or chunk_size % page_size:
            raise ValueError("chunk_size must be divisible by page_size")
        if model.training:
            raise ValueError("Qwen3 reverse training requires model.eval() so rollout and recomputation match")
        self.model = model
        self.chunk_size = chunk_size
        self.page_size = page_size

    def backward(
        self,
        input_ids: torch.Tensor,
        layers: Sequence[LayerKVTrace],
        loss_fn: Callable[[torch.Tensor, int, int], tuple[torch.Tensor, int]],
        *,
        release_processed_suffix: bool = True,
        backward_context: Callable[[int, int], AbstractContextManager] | None = None,
        stage1_release: Callable[[], None] | None = None,
    ) -> ReverseTrainingResult:
        if input_ids.ndim != 2 or input_ids.shape[0] < 1:
            raise ValueError("Qwen3 reverse training requires a non-empty [batch, sequence] input")
        from .oomb_paged_attention import OOMBPagedReverseState

        state = OOMBPagedReverseState(list(layers), chunk_size=self.chunk_size, page_size=self.page_size)
        if stage1_release is not None:
            stage1_release()
        sequence_length = input_ids.shape[1]
        if state.sequence_length != sequence_length:
            raise ValueError(f"input/KV token length mismatch: input={sequence_length}, KV={state.sequence_length}")

        loss_sum = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        valid_tokens = 0
        chunks = 0
        bounds = [
            (start, min(start + self.chunk_size, sequence_length))
            for start in range(0, sequence_length, self.chunk_size)
        ]
        total_chunks = len(bounds)
        with use_qwen3_reverse_attention(self.model, state):
            for start, end in reversed(bounds):
                state.begin(start, end)
                position_ids = torch.arange(start, end, device=input_ids.device).unsqueeze(0)
                position_ids = position_ids.expand(input_ids.shape[0], -1)
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
        stage1_release: Callable[[], None] | None = None,
    ) -> ReverseTrainingResult:
        """Run ragged trajectories in reverse-depth wavefront batches."""

        if not input_ids or not (len(input_ids) == len(layers) == len(loss_fns)):
            raise ValueError("input trajectories, KV traces, and loss functions must be non-empty and aligned")
        if any(sequence.ndim != 2 or sequence.shape[0] != 1 for sequence in input_ids):
            raise ValueError("OOMB paged reverse batches require [1, sequence] inputs")
        sequence_lengths = [sequence.shape[1] for sequence in input_ids]
        layer_counts = {len(trace) for trace in layers}
        if len(layer_counts) != 1:
            raise ValueError("OOMB paged reverse batches require the same layer count")
        for sequence_length, trace in zip(sequence_lengths, layers, strict=True):
            if any(layer.length != sequence_length for layer in trace):
                raise ValueError("each OOMB trajectory KV trace must match its input length")

        padded_lengths = [math.ceil(length / self.chunk_size) * self.chunk_size for length in sequence_lengths]
        padded_input_ids = [
            torch.nn.functional.pad(sequence, (0, padded_length - sequence.shape[1]))
            for sequence, padded_length in zip(input_ids, padded_lengths, strict=True)
        ]
        padded_trajectories = [
            [
                LayerKVTrace(
                    torch.nn.functional.pad(layer.key, (0, 0, 0, padded_length - sequence_length)),
                    torch.nn.functional.pad(layer.value, (0, 0, 0, padded_length - sequence_length)),
                )
                for layer in trace
            ]
            for sequence_length, padded_length, trace in zip(sequence_lengths, padded_lengths, layers, strict=True)
        ]
        from .oomb_paged_attention import OOMBPagedWavefrontState

        state = OOMBPagedWavefrontState(
            padded_trajectories,
            chunk_size=self.chunk_size,
            page_size=self.page_size,
        )
        if stage1_release is not None:
            stage1_release()

        schedule = _build_reverse_wavefront(sequence_lengths, self.chunk_size)
        loss_sum = torch.zeros((), device=input_ids[0].device, dtype=torch.float32)
        valid_tokens = 0
        total_chunks = sum(math.ceil(length / self.chunk_size) for length in sequence_lengths)
        with use_qwen3_reverse_attention(self.model, state):
            for call_idx, (depth, active) in enumerate(schedule):
                start = (depth - 1) * self.chunk_size
                end = depth * self.chunk_size
                state.begin(active, start, end)
                chunk_ids = torch.cat([padded_input_ids[idx][:, start:end] for idx in active], dim=0)
                position_ids = torch.arange(start, end, device=chunk_ids.device).unsqueeze(0)
                output = self.model(
                    input_ids=chunk_ids,
                    position_ids=position_ids.expand(len(active), -1),
                    use_cache=False,
                    return_dict=True,
                )
                chunk_loss = output.logits.sum() * 0.0
                chunk_valid_tokens = 0
                for row, trajectory_idx in enumerate(active):
                    sample_end = min(end, sequence_lengths[trajectory_idx])
                    if start >= sample_end:
                        continue
                    sample_loss, sample_valid_tokens = loss_fns[trajectory_idx](
                        output.logits[row : row + 1, : sample_end - start],
                        start,
                        sample_end,
                    )
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
            max_parallel_trajectories=max(len(active) for _, active in schedule),
        )
