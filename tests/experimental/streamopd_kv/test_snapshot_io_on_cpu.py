# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import os
import threading
import time

import pytest
import torch

from verl.experimental.streamopd_kv import (
    TrajectoryKey,
)
from verl.experimental.streamopd_kv.host_slot_pool import HostKVSlotPool, cleanup_host_kv_pools
from verl.experimental.streamopd_kv.snapshot_io import (
    extract_vllm_cross_layers_nhd_token_range,
    extract_vllm_nhd_token_range,
    extract_vllm_nhd_tokens,
    load_vllm_snapshot,
    release_vllm_snapshot,
)


def test_shared_host_kv_slot_pool_uses_fixed_backing_and_reuses_rows(tmp_path) -> None:
    pool = HostKVSlotPool.create_or_open(
        str(tmp_path),
        tp_rank=0,
        slot_count=2,
        token_capacity=4,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        page_size=16,
        dtype=torch.float32,
    )
    first = pool.acquire(request_id="backend-a", trajectory_id="trajectory-a", policy_version=3, prompt_length=1)
    second = pool.acquire(request_id="backend-b", trajectory_id="trajectory-b", policy_version=3, prompt_length=1)
    with pytest.raises(RuntimeError, match="pool is full"):
        pool.acquire(request_id="backend-c", trajectory_id="trajectory-c", policy_version=3, prompt_length=1)

    for layer_index in range(2):
        key, value = pool.layer(0, layer_index)
        key[:3].fill_(layer_index + 1)
        value[:3].fill_(-(layer_index + 1))
    pool.seal(
        first,
        request_id="backend-a",
        trajectory_id="trajectory-a",
        policy_version=3,
        prompt_length=1,
        token_ids=[4, 5, 6],
        token_count=3,
        streamed_tokens_before_eos=2,
        streamed_chunks_before_eos=1,
    )
    snapshot = load_vllm_snapshot(
        first,
        key=TrajectoryKey(3, "trajectory-a"),
        tp_rank=0,
        expected_tp_size=1,
        expected_token_ids=[4, 5, 6],
        expected_prompt_length=1,
    )
    assert snapshot.layers[0].key.is_contiguous()
    assert snapshot.streamed_tokens_before_eos == 2
    torch.testing.assert_close(snapshot.layers[1].key, torch.full((3, 1, 2), 2.0))
    torch.testing.assert_close(snapshot.layers[1].value, torch.full((3, 1, 2), -2.0))
    with pytest.raises(RuntimeError, match="token identity"):
        load_vllm_snapshot(
            first,
            key=TrajectoryKey(3, "trajectory-a"),
            tp_rank=0,
            expected_tp_size=1,
            expected_token_ids=[4, 5, 7],
            expected_prompt_length=1,
        )

    data_path = HostKVSlotPool.data_path(pool.root)
    fixed_size = os.path.getsize(data_path)
    release_vllm_snapshot(first)
    pool.release(second)
    reused = pool.acquire(
        request_id="backend-c",
        trajectory_id="trajectory-c",
        policy_version=4,
        prompt_length=1,
    )
    assert HostKVSlotPool.parse_slot_path(reused)[1] == 0
    assert reused != first
    assert os.path.getsize(data_path) == fixed_size
    assert not list(tmp_path.glob("*.safetensors"))
    with pytest.raises(RuntimeError, match="stale"):
        pool.release(first)
    assert pool.state_counts() == {"free": 1, "writing": 1, "sealed": 0}
    with pytest.raises(RuntimeError, match="cannot clean active"):
        cleanup_host_kv_pools(str(tmp_path))
    pool.release(reused)
    assert pool.state_counts() == {"free": 2, "writing": 0, "sealed": 0}
    assert cleanup_host_kv_pools(str(tmp_path)) > fixed_size
    assert not list(tmp_path.glob("host_kv_pool.tp0.*"))


def test_shared_host_kv_metadata_waits_for_async_seal(tmp_path) -> None:
    pool = HostKVSlotPool.create_or_open(
        str(tmp_path),
        tp_rank=0,
        slot_count=1,
        token_capacity=4,
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        page_size=16,
        dtype=torch.float32,
    )
    slot_path = pool.acquire(request_id="backend", trajectory_id="trajectory", policy_version=3, prompt_length=1)

    def seal() -> None:
        time.sleep(0.02)
        pool.seal(
            slot_path,
            request_id="backend",
            trajectory_id="trajectory",
            policy_version=3,
            prompt_length=1,
            token_ids=[4, 5, 6],
            token_count=3,
            streamed_tokens_before_eos=0,
            streamed_chunks_before_eos=0,
        )

    thread = threading.Thread(target=seal)
    thread.start()
    metadata = pool.metadata(
        slot_path,
        trajectory_id="trajectory",
        policy_version=3,
        prompt_length=1,
        token_ids=[4, 5, 6],
        wait_timeout_seconds=1.0,
    )
    thread.join()
    assert metadata["token_count"] == 3
    pool.release(slot_path)
    pool.close()


def test_vllm_range_extraction_copies_only_intersecting_blocks() -> None:
    cache = torch.arange(4 * 2 * 4 * 1 * 2).reshape(4, 2, 4, 1, 2)
    full = extract_vllm_nhd_tokens(cache, [3, 1, 2], block_size=4, num_tokens=12)
    selected = extract_vllm_nhd_token_range(cache, [3, 1, 2], block_size=4, start=3, end=10)
    torch.testing.assert_close(selected, full[3:10])


def test_vllm_cross_layer_range_extraction_matches_individual_layers() -> None:
    layers = [torch.arange(4 * 2 * 4 * 1 * 2).reshape(4, 2, 4, 1, 2) + 1000 * layer for layer in range(3)]
    cross_layers = torch.stack(layers, dim=1)
    expected = torch.stack([extract_vllm_nhd_token_range(layer, [3, 1, 2], 4, 3, 10) for layer in layers])

    actual = extract_vllm_cross_layers_nhd_token_range(cross_layers, [3, 1, 2], 4, 3, 10)
    torch.testing.assert_close(actual, expected)
    reordered = extract_vllm_cross_layers_nhd_token_range(
        cross_layers,
        [3, 1, 2],
        4,
        3,
        10,
        layer_order=[2, 0, 1],
    )
    torch.testing.assert_close(reordered, expected[[2, 0, 1]])


def test_vllm_nhd_page_extraction_preserves_logical_token_order() -> None:
    cache = torch.arange(3 * 2 * 4 * 2 * 3).reshape(3, 2, 4, 2, 3)
    extracted = extract_vllm_nhd_tokens(cache, block_ids=[2, 0], block_size=4, num_tokens=6)
    expected = torch.cat((cache[2].transpose(0, 1), cache[0].transpose(0, 1)), dim=0)[:6]
    torch.testing.assert_close(extracted, expected)

    legacy_cache = cache.permute(1, 0, 2, 3, 4).contiguous()
    legacy_extracted = extract_vllm_nhd_tokens(legacy_cache, block_ids=[2, 0], block_size=4, num_tokens=6)
    torch.testing.assert_close(legacy_extracted, expected)
