# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right


def exact_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_start: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Exact dense GQA attention for suffix queries and prefix-inclusive KV.

    Tensors are ``[batch, heads, tokens, head_dim]``.  A custom mask is
    required because ``is_causal=True`` uses local query indices when Q and KV
    lengths differ, which is wrong for suffix chunks.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 tensors")
    if key.shape != value.shape:
        raise ValueError("key and value shapes differ")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and KV batch/head dimensions are incompatible")
    if query_start < 0 or query_start + query.shape[2] != key.shape[2]:
        raise ValueError("KV must end at the end of the current query chunk")

    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    enable_gqa = query.shape[1] != key.shape[1]
    # Preserve the model's native SDPA path for the first chunk. Suffix chunks
    # need a lower-right causal bias because their query positions do not start
    # at zero while the retained KV does.
    attention_bias = None if query_start == 0 else causal_lower_right(query.shape[2], key.shape[2])
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_bias,
        dropout_p=0.0,
        is_causal=query_start == 0,
        scale=scale,
        enable_gqa=enable_gqa,
    )


@dataclass
class LayerKVTrace:
    """One sealed no-grad KV layer plus accumulated suffix gradients."""

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if self.key.shape != self.value.shape or self.key.ndim != 4:
            raise ValueError("layer KV must have matching [B, H, T, D] tensors")
        if self.key.requires_grad or self.value.requires_grad:
            raise ValueError("rollout KV trace must be detached")
        self.key = self.key.detach()
        self.value = self.value.detach()
        self.key_grad: torch.Tensor | None = None
        self.value_grad: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return self.key.shape[2]

    def release_suffix(self, start: int) -> None:
        if not 0 <= start <= self.length:
            raise ValueError("release start is outside the KV trace")
        if start == self.length:
            return
        self.key = self.key[:, :, :start].contiguous()
        self.value = self.value[:, :, :start].contiguous()
        if self.key_grad is not None:
            self.key_grad = self.key_grad[:, :, :start].contiguous()
        if self.value_grad is not None:
            self.value_grad = self.value_grad[:, :, :start].contiguous()


class ReverseChunkState:
    """Coordinates K/V replacement and gradient injection for reverse chunks."""

    def __init__(self, layers: list[LayerKVTrace]) -> None:
        if not layers:
            raise ValueError("reverse state requires at least one layer")
        lengths = {layer.length for layer in layers}
        if len(lengths) != 1:
            raise ValueError("all KV layers must cover the same token range")
        self.layers = layers
        self.start = 0
        self.end = 0
        self._prefix_leaves: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._current: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def sequence_length(self) -> int:
        return self.layers[0].length

    def begin(self, start: int, end: int) -> None:
        if not 0 <= start < end <= self.sequence_length:
            raise ValueError(f"invalid reverse chunk [{start}, {end})")
        self.start, self.end = start, end
        self._prefix_leaves.clear()
        self._current.clear()

    def attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        if layer_idx in self._current:
            raise RuntimeError(f"layer {layer_idx} was visited twice in one chunk")
        layer = self.layers[layer_idx]
        expected_tokens = self.end - self.start
        if current_key.shape[2] != expected_tokens or current_value.shape != current_key.shape:
            raise ValueError("current KV does not match the active reverse chunk")

        if self.start:
            prefix_key = layer.key[:, :, : self.start].detach().requires_grad_(True)
            prefix_value = layer.value[:, :, : self.start].detach().requires_grad_(True)
            full_key = torch.cat((prefix_key, current_key), dim=2)
            full_value = torch.cat((prefix_value, current_value), dim=2)
            self._prefix_leaves[layer_idx] = (prefix_key, prefix_value)
        else:
            full_key, full_value = current_key, current_value
        self._current[layer_idx] = (current_key, current_value)
        return exact_causal_attention(query, full_key, full_value, query_start=self.start, scale=scale)

    def gradient_injection(self) -> torch.Tensor:
        """Scalar VJP that injects dK/dV accumulated from later chunks."""

        if len(self._current) != len(self.layers):
            raise RuntimeError(f"expected {len(self.layers)} visited layers, got {len(self._current)}")
        if all(layer.key_grad is None for layer in self.layers):
            first_key, _ = next(iter(self._current.values()))
            return first_key.new_zeros(())
        injection = None
        for layer_idx, (current_key, current_value) in self._current.items():
            layer = self.layers[layer_idx]
            if layer.key_grad is None or layer.value_grad is None:
                raise RuntimeError(f"incomplete accumulated KV gradients for layer {layer_idx}")
            key_grad = layer.key_grad[:, :, self.start : self.end].to(current_key.dtype)
            value_grad = layer.value_grad[:, :, self.start : self.end].to(current_value.dtype)
            term = (current_key * key_grad).sum() + (current_value * value_grad).sum()
            injection = term if injection is None else injection + term
        assert injection is not None
        return injection

    def commit_prefix_gradients(self, *, release_processed_suffix: bool = False) -> None:
        """Accumulate this chunk's attention VJPs into earlier sealed KV."""

        for layer_idx, (prefix_key, prefix_value) in self._prefix_leaves.items():
            if prefix_key.grad is None or prefix_value.grad is None:
                raise RuntimeError(f"attention backward did not produce prefix KV gradients for layer {layer_idx}")
            layer = self.layers[layer_idx]
            if layer.key_grad is None:
                layer.key_grad = torch.zeros_like(layer.key, dtype=torch.float32)
                layer.value_grad = torch.zeros_like(layer.value, dtype=torch.float32)
            layer.key_grad[:, :, : self.start].add_(prefix_key.grad.float())
            assert layer.value_grad is not None
            layer.value_grad[:, :, : self.start].add_(prefix_value.grad.float())
        if release_processed_suffix:
            for layer in self.layers:
                layer.release_suffix(self.start)
        self._prefix_leaves.clear()
        self._current.clear()


class BatchedReverseChunkState:
    """Exact reverse attention for a ragged batch of trajectory KV traces.

    Active trajectories always recompute the same number of current tokens,
    while their retained prefixes may have different lengths. Prefix pages are
    padded only for the batched SDPA call and excluded by an explicit mask.
    Gradients are scattered back into each trajectory's independent dK/dV
    buffers after backward.
    """

    def __init__(self, trajectories: Sequence[Sequence[LayerKVTrace]]) -> None:
        if not trajectories or not trajectories[0]:
            raise ValueError("batched reverse state requires trajectories with KV layers")
        num_layers = len(trajectories[0])
        if any(len(trajectory) != num_layers for trajectory in trajectories):
            raise ValueError("all trajectories must have the same number of KV layers")
        for trajectory in trajectories:
            if len({layer.length for layer in trajectory}) != 1:
                raise ValueError("all KV layers in a trajectory must cover the same token range")
        self.layers = [[trajectory[layer_idx] for trajectory in trajectories] for layer_idx in range(num_layers)]
        self.num_trajectories = len(trajectories)
        self.active_indices: tuple[int, ...] = ()
        self.starts: tuple[int, ...] = ()
        self.ends: tuple[int, ...] = ()
        self._prefix_leaves: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._current: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def begin(self, active_indices: Sequence[int], starts: Sequence[int], ends: Sequence[int]) -> None:
        if not active_indices or not (len(active_indices) == len(starts) == len(ends)):
            raise ValueError("active trajectory indices and chunk bounds must be non-empty and aligned")
        chunk_lengths = [end - start for start, end in zip(starts, ends, strict=True)]
        if any(length < 1 for length in chunk_lengths):
            raise ValueError("a batched reverse step requires positive current chunk lengths")
        for trajectory_idx, start, end in zip(active_indices, starts, ends, strict=True):
            if not 0 <= trajectory_idx < self.num_trajectories:
                raise ValueError("active trajectory index is outside the reverse batch")
            available = self.layers[0][trajectory_idx].length
            if not 0 <= start < end <= available:
                raise ValueError(f"invalid reverse chunk [{start}, {end}) for trajectory {trajectory_idx}")
        self.active_indices = tuple(active_indices)
        self.starts = tuple(starts)
        self.ends = tuple(ends)
        self._prefix_leaves.clear()
        self._current.clear()

    def attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        if layer_idx in self._current:
            raise RuntimeError(f"layer {layer_idx} was visited twice in one chunk")
        batch_size = len(self.active_indices)
        chunk_lengths = [end - start for start, end in zip(self.starts, self.ends, strict=True)]
        max_chunk_length = max(chunk_lengths)
        if current_key.shape[0] != batch_size or current_key.shape[2] != max_chunk_length:
            raise ValueError("current KV does not match the active reverse batch")
        if current_value.shape != current_key.shape or query.shape[0] != batch_size:
            raise ValueError("query, key, and value batch dimensions are incompatible")

        traces = [self.layers[layer_idx][idx] for idx in self.active_indices]
        max_prefix = max(self.starts)
        if max_prefix:
            padded_keys = []
            padded_values = []
            for trace, start in zip(traces, self.starts, strict=True):
                pad_tokens = max_prefix - start
                padded_keys.append(F.pad(trace.key[:, :, :start], (0, 0, 0, pad_tokens)).squeeze(0))
                padded_values.append(F.pad(trace.value[:, :, :start], (0, 0, 0, pad_tokens)).squeeze(0))
            prefix_key = torch.stack(padded_keys).detach().requires_grad_(True)
            prefix_value = torch.stack(padded_values).detach().requires_grad_(True)
            full_key = torch.cat((prefix_key, current_key), dim=2)
            full_value = torch.cat((prefix_value, current_value), dim=2)
            self._prefix_leaves[layer_idx] = (prefix_key, prefix_value)
        else:
            full_key, full_value = current_key, current_value

        uniform_bounds = len(set(self.starts)) == 1 and len(set(chunk_lengths)) == 1
        if uniform_bounds:
            # A lower-right causal bias preserves the global token positions
            # of suffix queries and lets SDPA select its optimized causal
            # kernel.  Materialize a boolean mask only for genuinely ragged
            # reverse waves.
            attention_mask = (
                None if max_prefix == 0 else causal_lower_right(max_chunk_length, max_prefix + max_chunk_length)
            )
        else:
            prefix_columns = torch.arange(max_prefix, device=query.device)
            prefix_mask = prefix_columns.unsqueeze(0) < torch.tensor(self.starts, device=query.device).unsqueeze(1)
            current_columns = torch.arange(max_chunk_length, device=query.device)
            current_mask = current_columns.unsqueeze(0) <= current_columns.unsqueeze(1)
            current_mask = current_mask.unsqueeze(0).expand(batch_size, -1, -1)
            valid_current_keys = current_columns.unsqueeze(0) < torch.tensor(
                chunk_lengths, device=query.device
            ).unsqueeze(1)
            current_mask = current_mask & valid_current_keys.unsqueeze(1)
            attention_mask = torch.cat(
                (prefix_mask.unsqueeze(1).expand(-1, max_chunk_length, -1), current_mask), dim=-1
            ).unsqueeze(1)

        if query.shape[1] % full_key.shape[1] != 0:
            raise ValueError("query heads must be divisible by KV heads")
        self._current[layer_idx] = (current_key, current_value)
        return F.scaled_dot_product_attention(
            query,
            full_key,
            full_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=uniform_bounds and max_prefix == 0,
            scale=scale,
            enable_gqa=query.shape[1] != full_key.shape[1],
        )

    def gradient_injection(self) -> torch.Tensor:
        if len(self._current) != len(self.layers):
            raise RuntimeError(f"expected {len(self.layers)} visited layers, got {len(self._current)}")
        injection = None
        for layer_idx, (current_key, current_value) in self._current.items():
            for row, (trajectory_idx, start, end) in enumerate(
                zip(self.active_indices, self.starts, self.ends, strict=True)
            ):
                trace = self.layers[layer_idx][trajectory_idx]
                if trace.key_grad is None:
                    continue
                if trace.value_grad is None:
                    raise RuntimeError(f"incomplete accumulated KV gradients for layer {layer_idx}")
                length = end - start
                term = (current_key[row, :, :length] * trace.key_grad[0, :, start:end].to(current_key.dtype)).sum()
                term = (
                    term
                    + (current_value[row, :, :length] * trace.value_grad[0, :, start:end].to(current_value.dtype)).sum()
                )
                injection = term if injection is None else injection + term
        if injection is not None:
            return injection
        first_key, _ = next(iter(self._current.values()))
        return first_key.new_zeros(())

    def commit_prefix_gradients(self, *, release_processed_suffix: bool = False) -> None:
        for layer_idx, (prefix_key, prefix_value) in self._prefix_leaves.items():
            if prefix_key.grad is None or prefix_value.grad is None:
                raise RuntimeError(f"attention backward did not produce prefix KV gradients for layer {layer_idx}")
            for row, (trajectory_idx, start) in enumerate(zip(self.active_indices, self.starts, strict=True)):
                if start == 0:
                    continue
                trace = self.layers[layer_idx][trajectory_idx]
                if trace.key_grad is None:
                    trace.key_grad = torch.zeros_like(trace.key, dtype=torch.float32)
                    trace.value_grad = torch.zeros_like(trace.value, dtype=torch.float32)
                trace.key_grad[:, :, :start].add_(prefix_key.grad[row : row + 1, :, :start].float())
                assert trace.value_grad is not None
                trace.value_grad[:, :, :start].add_(prefix_value.grad[row : row + 1, :, :start].float())
        if release_processed_suffix:
            for trajectory_idx, start in zip(self.active_indices, self.starts, strict=True):
                for layer in self.layers:
                    layer[trajectory_idx].release_suffix(start)
        self._prefix_leaves.clear()
        self._current.clear()
