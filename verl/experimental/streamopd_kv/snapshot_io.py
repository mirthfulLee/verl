# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from .host_slot_pool import HostKVSlotPool
from .protocol import TrajectoryKey

_POOLS: dict[str, HostKVSlotPool] = {}
_POOLS_LOCK = threading.Lock()


if _TRITON_AVAILABLE:

    @triton.jit
    def _cross_layers_nhd_gather_kernel(
        cache,
        block_ids,
        layer_order,
        output,
        layer_count,
        token_count,
        token_offset,
        block_size,
        head_count,
        head_dim,
        cache_stride_block,
        cache_stride_layer,
        cache_stride_kv,
        cache_stride_token,
        cache_stride_head,
        cache_stride_dim,
        output_stride_layer,
        output_stride_token,
        output_stride_kv,
        output_stride_head,
        output_stride_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        feature_count = head_count * head_dim
        values_per_token = 2 * feature_count
        total = layer_count * token_count * values_per_token
        mask = offsets < total
        remainder = offsets
        dim = remainder % head_dim
        remainder //= head_dim
        head = remainder % head_count
        remainder //= head_count
        kv = remainder % 2
        remainder //= 2
        token = remainder % token_count
        layer = remainder // token_count
        absolute_token = token_offset + token
        logical_block = absolute_token // block_size
        block_token = absolute_token % block_size
        physical_block = tl.load(block_ids + logical_block, mask=mask, other=0)
        source_layer = tl.load(layer_order + layer, mask=mask, other=0)
        source_offset = (
            physical_block * cache_stride_block
            + source_layer * cache_stride_layer
            + kv * cache_stride_kv
            + block_token * cache_stride_token
            + head * cache_stride_head
            + dim * cache_stride_dim
        )
        output_offset = (
            layer * output_stride_layer
            + token * output_stride_token
            + kv * output_stride_kv
            + head * output_stride_head
            + dim * output_stride_dim
        )
        value = tl.load(cache + source_offset, mask=mask)
        tl.store(output + output_offset, value, mask=mask)


@dataclass(frozen=True)
class HostSlotLayerKV:
    """Token-major rollout KV view backed by a shared Host slot."""

    key: torch.Tensor
    value: torch.Tensor

    @property
    def length(self) -> int:
        return self.key.shape[0]


@dataclass(frozen=True)
class HostSlotKVSnapshot:
    layers: tuple[HostSlotLayerKV, ...]
    handoff_seconds: float
    streamed_tokens_before_eos: int
    streamed_chunks_before_eos: int


def _pool_for(slot_path: str) -> HostKVSlotPool:
    root, _, _ = HostKVSlotPool.parse_slot_path(slot_path)
    with _POOLS_LOCK:
        pool = _POOLS.get(root)
        if pool is None:
            pool = HostKVSlotPool.open_for_slot(slot_path)
            _POOLS[root] = pool
        return pool


def load_vllm_snapshot(
    slot_path: str,
    *,
    key: TrajectoryKey,
    tp_rank: int,
    expected_tp_size: int,
    expected_token_ids: Sequence[int],
    expected_prompt_length: int,
) -> HostSlotKVSnapshot:
    """Validate and expose one sealed vLLM Host-slot shard."""

    started = time.perf_counter()
    root, _, _ = HostKVSlotPool.parse_slot_path(slot_path)
    if expected_tp_size != 1 or not root.endswith(f".tp{tp_rank}"):
        raise RuntimeError("shared Host KV slot TP layout mismatch")
    pool = _pool_for(slot_path)
    metadata = pool.metadata(
        slot_path,
        trajectory_id=key.trajectory_id,
        policy_version=key.policy_version,
        prompt_length=expected_prompt_length,
        token_ids=expected_token_ids,
        wait_timeout_seconds=600.0,
    )
    if len(expected_token_ids) != metadata["token_count"]:
        raise RuntimeError("shared Host KV token count does not match the training trajectory")
    layers = []
    for layer_index in range(pool.num_layers):
        key_view, value_view = pool.layer(metadata["slot"], layer_index)
        layers.append(
            HostSlotLayerKV(
                key_view[: metadata["token_count"]],
                value_view[: metadata["token_count"]],
            )
        )
    return HostSlotKVSnapshot(
        layers=tuple(layers),
        handoff_seconds=time.perf_counter() - started,
        streamed_tokens_before_eos=metadata["streamed_tokens_before_eos"],
        streamed_chunks_before_eos=metadata["streamed_chunks_before_eos"],
    )


def release_vllm_snapshot(slot_path: str) -> None:
    _pool_for(slot_path).release(slot_path)


def extract_vllm_nhd_tokens(
    kv_cache: torch.Tensor,
    block_ids: Sequence[int],
    block_size: int,
    num_tokens: int,
    *,
    kv_axis: int | None = None,
) -> torch.Tensor:
    """Gather logical tokens from supported vLLM NHD cache layouts."""

    if kv_cache.ndim != 5 or kv_cache.shape[2] != block_size:
        raise ValueError(f"expected a 5-D vLLM NHD KV cache with block size {block_size}, got {tuple(kv_cache.shape)}")
    if kv_axis is None:
        candidates = [axis for axis in (0, 1) if kv_cache.shape[axis] == 2]
        if len(candidates) != 1:
            raise ValueError(f"cannot infer the K/V axis for vLLM cache shape {tuple(kv_cache.shape)}")
        kv_axis = candidates[0]
    if kv_axis not in (0, 1) or kv_cache.shape[kv_axis] != 2:
        raise ValueError(f"invalid K/V axis {kv_axis} for vLLM cache shape {tuple(kv_cache.shape)}")
    if not 0 <= num_tokens <= len(block_ids) * block_size:
        raise ValueError("requested token count is outside the supplied KV blocks")
    blocks = torch.as_tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    selected = kv_cache.index_select(1 - kv_axis, blocks)
    logical = selected.permute(1, 2, 0, 3, 4) if kv_axis == 0 else selected.permute(0, 2, 1, 3, 4)
    return logical.flatten(0, 1)[:num_tokens].contiguous()


def extract_vllm_nhd_token_range(
    kv_cache: torch.Tensor,
    block_ids: Sequence[int],
    block_size: int,
    start: int,
    end: int,
) -> torch.Tensor:
    """Gather only logical KV blocks intersecting the requested token range."""

    capacity = len(block_ids) * block_size
    if not 0 <= start < end <= capacity:
        raise ValueError(f"invalid KV token range [{start}, {end}) for capacity {capacity}")
    first_block = start // block_size
    last_block = (end + block_size - 1) // block_size
    selected_blocks = block_ids[first_block:last_block]
    logical = extract_vllm_nhd_tokens(
        kv_cache,
        selected_blocks,
        block_size,
        len(selected_blocks) * block_size,
    )
    offset = start - first_block * block_size
    return logical[offset : offset + end - start].contiguous()


def extract_vllm_cross_layers_nhd_token_range(
    kv_cache: torch.Tensor,
    block_ids: Sequence[int],
    block_size: int,
    start: int,
    end: int,
    *,
    layer_order: Sequence[int] | None = None,
) -> torch.Tensor:
    """Gather one logical token range from vLLM's block-major layer cache."""

    if kv_cache.ndim != 6 or kv_cache.shape[2] != 2 or kv_cache.shape[3] != block_size:
        raise ValueError(
            "expected a 6-D vLLM cross-layer NHD cache shaped "
            f"[block, layer, 2, block_size, head, dim], got {tuple(kv_cache.shape)}"
        )
    capacity = len(block_ids) * block_size
    if not 0 <= start < end <= capacity:
        raise ValueError(f"invalid cross-layer KV token range [{start}, {end}) for capacity {capacity}")
    first_block = start // block_size
    last_block = (end + block_size - 1) // block_size
    blocks = torch.as_tensor(block_ids[first_block:last_block], dtype=torch.long, device=kv_cache.device)
    selected = kv_cache.index_select(0, blocks)
    if layer_order is not None:
        order = tuple(int(index) for index in layer_order)
        if order != tuple(range(kv_cache.shape[1])):
            indices = torch.as_tensor(order, dtype=torch.long, device=kv_cache.device)
            selected = selected.index_select(1, indices)
    # [block, layer, 2, token, head, dim] -> [layer, token, 2, head, dim]
    logical = selected.permute(1, 0, 3, 2, 4, 5).flatten(1, 2)
    offset = start - first_block * block_size
    return logical[:, offset : offset + end - start].contiguous()


def gather_vllm_cross_layers_nhd_into(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    layer_order: torch.Tensor,
    output: torch.Tensor,
    *,
    token_offset: int,
    token_count: int,
) -> None:
    """Gather block-major vLLM KV into a preallocated layer-major buffer."""

    if kv_cache.ndim != 6 or kv_cache.shape[2] != 2:
        raise ValueError(f"invalid cross-layer KV cache shape: {tuple(kv_cache.shape)}")
    expected_tail = (2, kv_cache.shape[4], kv_cache.shape[5])
    if output.ndim != 5 or output.shape[0] != kv_cache.shape[1] or output.shape[2:] != expected_tail:
        raise ValueError(f"invalid cross-layer KV output shape: {tuple(output.shape)}")
    block_size = int(kv_cache.shape[3])
    if not 0 <= token_offset < block_size or not 0 < token_count <= output.shape[1]:
        raise ValueError("invalid cross-layer KV output token range")
    required_blocks = (token_offset + token_count + block_size - 1) // block_size
    if block_ids.ndim != 1 or block_ids.numel() < required_blocks or block_ids.device != kv_cache.device:
        raise ValueError("cross-layer KV block ids do not cover the requested output")
    if layer_order.shape != (kv_cache.shape[1],) or layer_order.device != kv_cache.device:
        raise ValueError("cross-layer KV layer order is invalid")

    if not _TRITON_AVAILABLE or not kv_cache.is_cuda:
        source = extract_vllm_cross_layers_nhd_token_range(
            kv_cache,
            block_ids[:required_blocks].tolist(),
            block_size,
            token_offset,
            token_offset + token_count,
            layer_order=layer_order.tolist(),
        )
        output[:, :token_count].copy_(source)
        return

    total = output.shape[0] * token_count * 2 * output.shape[3] * output.shape[4]
    _cross_layers_nhd_gather_kernel[(triton.cdiv(total, 1024),)](
        kv_cache,
        block_ids,
        layer_order,
        output,
        output.shape[0],
        token_count,
        token_offset,
        block_size,
        output.shape[3],
        output.shape[4],
        *kv_cache.stride(),
        *output.stride(),
        BLOCK_SIZE=1024,
    )
