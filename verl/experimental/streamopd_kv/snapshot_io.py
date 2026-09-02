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

from .host_slot_pool import HostKVSlotPool
from .protocol import TrajectoryKey

_POOLS: dict[str, HostKVSlotPool] = {}
_POOLS_LOCK = threading.Lock()


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
