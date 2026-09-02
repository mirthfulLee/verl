# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn

from verl.experimental.streamopd_kv import (
    CommittedChunkPublisher,
    CommittedTokenChunk,
    StreamingTeacherCoordinator,
    TrajectoryKey,
    prepare_streamopd_kv_config,
)
from verl.experimental.streamopd_kv.config import _auto_streamopd_runtime_profile
from verl.experimental.streamopd_kv.fsdp_worker import (
    StreamOPDKVTrainingWorker,
    _deferred_training_state_bytes,
    _fixed_reverse_slot_plan,
    _forward_kl_topk_sum,
    _has_valid_response,
    _partition_reverse_microbatches,
    _reverse_backward_calls,
)
from verl.experimental.streamopd_kv.host_slot_pool import HostKVSlotPool, cleanup_host_kv_pools
from verl.experimental.streamopd_kv.qwen3 import _build_reverse_wavefront
from verl.experimental.streamopd_kv.scheduler import StreamOPDTaskScheduler
from verl.experimental.streamopd_kv.snapshot_io import (
    extract_vllm_nhd_token_range,
    extract_vllm_nhd_tokens,
    load_vllm_snapshot,
    release_vllm_snapshot,
)
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.workers.config.distillation import DistillationConfig, StreamOPDKVConfig


@pytest.mark.asyncio
async def test_committed_chunk_publisher_emits_only_accepted_contiguous_tokens() -> None:
    emitted = []

    async def submit(chunk) -> None:
        emitted.append(chunk)

    publisher = CommittedChunkPublisher(TrajectoryKey(4, "req"), [1, 2], chunk_size=4, submit=submit)
    await publisher.observe([10])
    await publisher.observe([10, 11, 12])
    await publisher.observe([10, 11, 12, 13], terminal=True)

    assert [(chunk.start, chunk.token_ids, chunk.terminal) for chunk in emitted] == [
        (0, (10, 11), False),
        (2, (12, 13), True),
    ]
    assert emitted[0].prompt_ids == (1, 2)
    assert emitted[1].prompt_ids == ()
    with pytest.raises(RuntimeError, match="retracted or replaced"):
        other = CommittedChunkPublisher(TrajectoryKey(4, "other"), [], 2, submit)
        await other.observe([8, 9])
        await other.observe([8, 7])


@pytest.mark.asyncio
async def test_committed_chunk_publisher_streams_every_complete_chunk_before_eos() -> None:
    emitted = []

    async def submit(chunk) -> None:
        emitted.append(chunk)

    publisher = CommittedChunkPublisher(TrajectoryKey(4, "every-chunk"), [1], chunk_size=2, submit=submit)
    await publisher.observe([10, 11])
    await publisher.observe([10, 11, 12, 13])
    await publisher.observe([10, 11, 12, 13, 14, 15])

    assert [(chunk.start, chunk.end, chunk.terminal) for chunk in emitted] == [
        (0, 1, False),
        (1, 3, False),
        (3, 5, False),
    ]


@pytest.mark.asyncio
async def test_committed_chunk_publisher_sends_longest_page_aligned_progress() -> None:
    emitted = []

    async def submit(chunk) -> None:
        emitted.append(chunk)

    publisher = CommittedChunkPublisher(
        TrajectoryKey(4, "page-aligned"),
        [1],
        chunk_size=8,
        page_size=4,
        submit=submit,
    )
    await publisher.observe(list(range(7)))
    await publisher.observe(list(range(12)))
    await publisher.observe(list(range(13)), terminal=True)

    assert [(chunk.start, chunk.end, chunk.terminal) for chunk in emitted] == [
        (0, 8, False),
        (8, 13, True),
    ]


@pytest.mark.asyncio
async def test_committed_publisher_long_prompt_starts_after_one_aligned_page() -> None:
    emitted = []

    async def submit(chunk: CommittedTokenChunk) -> None:
        emitted.append(chunk)

    publisher = CommittedChunkPublisher(
        TrajectoryKey(4, "long-prompt"),
        list(range(12)),
        chunk_size=8,
        page_size=4,
        submit=submit,
    )
    await publisher.observe([10, 11, 12])
    assert emitted == []
    await publisher.observe([10, 11, 12, 13])
    assert [(chunk.start, chunk.end) for chunk in emitted] == [(0, 4)]


@pytest.mark.asyncio
async def test_teacher_streaming_coalesces_queued_progress_before_prefill() -> None:
    calls = []

    async def score(fragment: list[int], request_id: str, terminal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((list(fragment), request_id, terminal))
        ids = torch.tensor(fragment).unsqueeze(-1)
        return ids, -ids.float()

    coordinator = StreamingTeacherCoordinator(score, max_pending_chunks=2)
    key = TrajectoryKey(9, "streamed")
    await coordinator.submit(
        CommittedTokenChunk(key=key, start=0, token_ids=(10, 11), prompt_ids=(1, 2), terminal=False)
    )
    await coordinator.submit(CommittedTokenChunk(key=key, start=2, token_ids=(12, 13), terminal=False))
    await coordinator.submit(CommittedTokenChunk(key=key, start=4, token_ids=(), terminal=True))
    ids, logprobs = await coordinator.result(key, required_completion_tokens=4)
    assert [call[0] for call in calls] == [[1, 2, 10, 11, 12, 13]]
    assert [call[2] for call in calls] == [True]
    torch.testing.assert_close(ids[:, 0], torch.tensor([1, 2, 10, 11, 12, 13]))
    torch.testing.assert_close(logprobs[:, 0], -ids[:, 0].float())


@pytest.mark.asyncio
async def test_teacher_streaming_serializes_chunks_for_one_trajectory() -> None:
    calls = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    active = 0
    max_active = 0

    async def score(fragment: list[int], request_id: str, terminal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        calls.append((list(fragment), request_id, terminal))
        if len(calls) == 1:
            first_started.set()
            await release_first.wait()
        ids = torch.tensor(fragment).unsqueeze(-1)
        active -= 1
        return ids, -ids.float()

    coordinator = StreamingTeacherCoordinator(score, max_pending_chunks=8, kv_page_size=1)
    key = TrajectoryKey(9, "serial")
    await coordinator.submit(
        CommittedTokenChunk(key=key, start=0, token_ids=(10, 11), prompt_ids=(1, 2), terminal=False)
    )
    await first_started.wait()
    await coordinator.submit(CommittedTokenChunk(key=key, start=2, token_ids=(12, 13), terminal=True))
    await asyncio.sleep(0)
    assert len(calls) == 1
    release_first.set()
    ids, _ = await coordinator.result(key, required_completion_tokens=4)

    assert max_active == 1
    assert [call[0] for call in calls] == [[1, 2, 10, 11], [12, 13]]
    assert calls[0][1] == calls[1][1]
    assert [call[2] for call in calls] == [False, True]
    torch.testing.assert_close(ids[:, 0], torch.tensor([1, 2, 10, 11, 12, 13]))


@pytest.mark.asyncio
async def test_teacher_streaming_closes_session_on_empty_terminal_marker() -> None:
    calls = []
    first_done = asyncio.Event()

    async def score(fragment: list[int], request_id: str, terminal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((fragment, terminal))
        values = torch.tensor(fragment, dtype=torch.int32).unsqueeze(-1)
        if len(calls) == 1:
            first_done.set()
        return values, values.float()

    coordinator = StreamingTeacherCoordinator(score, max_pending_chunks=8, kv_page_size=1)
    key = TrajectoryKey(9, "empty-terminal")
    await coordinator.submit(CommittedTokenChunk(key=key, start=0, token_ids=(10, 11), prompt_ids=(1,), terminal=False))
    await first_done.wait()
    await coordinator.submit(CommittedTokenChunk(key=key, start=2, token_ids=(), terminal=True))
    ids, _ = await coordinator.result(key, required_completion_tokens=2)

    assert calls == [([1, 10, 11], False), ([], True)]
    torch.testing.assert_close(ids[:, 0], torch.tensor([1, 10, 11], dtype=torch.int32))


@pytest.mark.asyncio
async def test_teacher_streaming_limits_active_trajectories_and_kv_tokens() -> None:
    started = asyncio.Queue()
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def score(fragment: list[int], request_id: str, terminal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await started.put(request_id)
        await release.wait()
        ids = torch.tensor(fragment).unsqueeze(-1)
        active -= 1
        return ids, -ids.float()

    coordinator = StreamingTeacherCoordinator(
        score,
        max_pending_chunks=8,
        max_active_trajectories=2,
        max_active_kv_tokens=7,
        kv_page_size=1,
    )
    keys = [TrajectoryKey(4, f"budget-{idx}") for idx in range(3)]
    for idx, key in enumerate(keys):
        await coordinator.submit(
            CommittedTokenChunk(
                key=key,
                start=0,
                token_ids=(10 + idx, 20 + idx, 30 + idx),
                prompt_ids=(1,),
                terminal=True,
            )
        )

    await started.get()
    await asyncio.sleep(0.01)
    # Each active request owns four prefix tokens, so the seven-token budget
    # admits only one trajectory even though the count limit is two.
    assert active == 1
    release.set()
    await asyncio.gather(*(coordinator.result(key, required_completion_tokens=3) for key in keys))
    assert max_active == 1


def test_vllm_prompt_logprobs_can_extract_incremental_tensor_suffix() -> None:
    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs

    def row(first: int):
        return {
            first: SimpleNamespace(rank=1, logprob=-0.1 * first),
            first + 1: SimpleNamespace(rank=2, logprob=-0.1 * (first + 1)),
        }

    output = SimpleNamespace(prompt_logprobs=[None, row(10), row(20), row(30)])
    result = {}
    extract_prompt_logprobs(output, 2, result, start=1, as_tensors=True)

    assert result["prompt_ids"].dtype == torch.int32
    assert result["prompt_logprobs"].dtype == torch.float32
    torch.testing.assert_close(result["prompt_ids"], torch.tensor([[20, 21], [30, 31], [0, 0]], dtype=torch.int32))


def test_vllm_worker_reports_and_resets_device_memory_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.workers.rollout.vllm_rollout import utils

    class DeviceModule:
        reset_device = None
        free_bytes = 31

        def reset_peak_memory_stats(self, device) -> None:
            self.reset_device = device

        @staticmethod
        def memory_allocated(device) -> int:
            assert device == 2
            return 11

        @staticmethod
        def memory_reserved(device) -> int:
            assert device == 2
            return 12

        @staticmethod
        def max_memory_allocated(device) -> int:
            assert device == 2
            return 21

        @staticmethod
        def max_memory_reserved(device) -> int:
            assert device == 2
            return 22

        def mem_get_info(self, device) -> tuple[int, int]:
            assert device == 2
            return self.free_bytes, 32

        def empty_cache(self) -> None:
            self.free_bytes = 32

    device_module = DeviceModule()
    monkeypatch.setattr(utils, "get_torch_device", lambda: device_module)
    worker = SimpleNamespace(device=2)

    utils.vLLMColocateWorkerExtension.reset_device_memory_stats(worker)
    stats = utils.vLLMColocateWorkerExtension.get_device_memory_stats(worker)

    assert device_module.reset_device == 2
    assert stats == {
        "allocated_bytes": 11,
        "reserved_bytes": 12,
        "max_allocated_bytes": 21,
        "max_reserved_bytes": 22,
        "free_bytes": 31,
        "total_bytes": 32,
    }
    assert utils.vLLMColocateWorkerExtension.trim_device_memory(worker, minimum_free_bytes=32) == {
        "freed_bytes": 1,
        "free_before_bytes": 31,
        "free_after_bytes": 32,
    }


def test_rollout_manager_aggregates_vllm_worker_memory_stats() -> None:
    from verl.workers.rollout.llm_server import LLMServerManager

    class RemoteMethod:
        def __init__(self, stats):
            self.stats = stats

        async def remote(self, method):
            assert method == "get_device_memory_stats"
            return self.stats

    class Server:
        def __init__(self, stats):
            self.collective_rpc = RemoteMethod(stats)

    manager = LLMServerManager.__new__(LLMServerManager)
    manager.server_handles = [
        Server(
            [
                {
                    "allocated_bytes": 1,
                    "reserved_bytes": 3,
                    "max_allocated_bytes": 5,
                    "max_reserved_bytes": 7,
                    "free_bytes": 11,
                    "total_bytes": 16,
                }
            ]
        ),
        Server(
            [
                {
                    "allocated_bytes": 2,
                    "reserved_bytes": 4,
                    "max_allocated_bytes": 6,
                    "max_reserved_bytes": 8,
                    "free_bytes": 9,
                    "total_bytes": 16,
                }
            ]
        ),
    ]

    assert manager.collect_device_memory_stats() == {
        "allocated_bytes": 2,
        "reserved_bytes": 4,
        "max_allocated_bytes": 6,
        "max_reserved_bytes": 8,
        "free_bytes": 9,
        "total_bytes": 16,
    }


def test_vllm_worker_reports_profiled_kv_capacity() -> None:
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    worker = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks=17, block_size=32))
    assert vLLMColocateWorkerExtension.get_kv_cache_capacity(worker) == {
        "num_gpu_blocks": 17,
        "block_size": 32,
        "capacity_tokens": 544,
    }
    worker.cache_config.num_gpu_blocks = None
    with pytest.raises(RuntimeError, match="unavailable"):
        vLLMColocateWorkerExtension.get_kv_cache_capacity(worker)


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


def test_vllm_range_extraction_copies_only_intersecting_blocks() -> None:
    cache = torch.arange(4 * 2 * 4 * 1 * 2).reshape(4, 2, 4, 1, 2)
    full = extract_vllm_nhd_tokens(cache, [3, 1, 2], block_size=4, num_tokens=12)
    selected = extract_vllm_nhd_token_range(cache, [3, 1, 2], block_size=4, start=3, end=10)
    torch.testing.assert_close(selected, full[3:10])


def test_vllm_connector_queues_kv_before_eos_and_only_seals_the_tail() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector, _SchedulerSaveState

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._chunk_size = 4
    connector._block_size = 2
    connector._pending = []
    connector._claimed_requests = set()
    state = _SchedulerSaveState(
        req_id="backend",
        trajectory_id="trajectory",
        slot_path="/tmp/trajectory",
        block_ids_by_group=[[0, 1, 2, 3]],
        policy_version=6,
        prompt_length=2,
    )

    connector._queue_committed(state, 3)
    assert not connector._pending
    connector._queue_committed(state, 5)
    assert [(item.start, item.end, item.terminal) for item in connector._pending] == [(0, 4, False)]
    connector._queue_committed(state, 7, terminal=True, token_ids=torch.arange(7))
    assert [(item.start, item.end, item.terminal) for item in connector._pending] == [
        (0, 4, False),
        (4, 7, True),
    ]
    assert state.published_tokens == 7
    assert state.next_chunk_index == 2


def test_fixed_slot_preflight_jointly_selects_stable_batch_and_chunk() -> None:
    config = SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=28,
        num_key_value_heads=8,
        num_attention_heads=16,
        head_dim=128,
        vocab_size=151936,
    )
    model = SimpleNamespace(config=config, model=SimpleNamespace(layers=[object()] * 28))
    memory = int(58.5 * 1024**3)

    capped = _fixed_reverse_slot_plan(
        model,
        configured_batch_size=16,
        token_capacity=4096,
        max_batch_tokens=32768,
        max_chunk_size=1024,
        min_chunk_size=256,
        page_size=64,
        dtype=torch.bfloat16,
        available_memory_bytes=memory,
    )
    assert (capped.batch_size, capped.token_capacity, capped.chunk_size) == (8, 4096, 1024)

    wide = _fixed_reverse_slot_plan(
        model,
        configured_batch_size=16,
        token_capacity=4096,
        max_batch_tokens=65536,
        max_chunk_size=1024,
        min_chunk_size=256,
        page_size=64,
        dtype=torch.bfloat16,
        available_memory_bytes=memory,
    )
    assert (wide.batch_size, wide.token_capacity, wide.chunk_size) == (8, 4096, 1024)

    long = _fixed_reverse_slot_plan(
        model,
        configured_batch_size=16,
        token_capacity=8192,
        max_batch_tokens=65536,
        max_chunk_size=1024,
        min_chunk_size=256,
        page_size=64,
        dtype=torch.bfloat16,
        available_memory_bytes=memory,
    )
    assert (long.batch_size, long.token_capacity, long.chunk_size) == (4, 8192, 1024)


def test_fixed_slot_preflight_reports_memory_budget_on_failure() -> None:
    config = SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=28,
        num_key_value_heads=8,
        num_attention_heads=16,
        head_dim=128,
        vocab_size=151936,
    )
    model = SimpleNamespace(config=config, model=SimpleNamespace(layers=[object()] * 28))

    with pytest.raises(RuntimeError, match=r"available=.*reserve=.*chunk_limit=.*minimum_chunk=64"):
        _fixed_reverse_slot_plan(
            model,
            configured_batch_size=1,
            token_capacity=4096,
            max_batch_tokens=4096,
            max_chunk_size=1024,
            min_chunk_size=64,
            page_size=64,
            dtype=torch.bfloat16,
            available_memory_bytes=1024**3,
        )


def test_reverse_preflight_reserves_lazy_adam_state_and_gradients() -> None:
    model = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters())
    parameter_bytes = model.weight.numel() * model.weight.element_size()

    assert _deferred_training_state_bytes(model, optimizer) == 3 * parameter_bytes
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.state[model.weight]["exp_avg"] = torch.zeros_like(model.weight)
    assert _deferred_training_state_bytes(model, optimizer) == parameter_bytes


def test_trainer_rejects_a_second_gpu_kv_lease() -> None:
    worker = StreamOPDKVTrainingWorker.__new__(StreamOPDKVTrainingWorker)
    worker._gpu_kv_lease_active = True
    with pytest.raises(RuntimeError, match="already holds a GPU KV lease"):
        worker.train_mini_batch(TensorDict({}, batch_size=[]))


def test_reverse_preflight_constraints_are_installed_before_slot_allocation() -> None:
    worker = StreamOPDKVTrainingWorker.__new__(StreamOPDKVTrainingWorker)
    worker._reverse_slot_pool = None
    worker._preflight_batch_cap = None
    worker._preflight_additional_reserve_bytes = 0
    worker.configure_reverse_preflight(batch_cap=4, additional_reserve_gib=1.25)
    assert worker._preflight_batch_cap == 4
    assert worker._preflight_additional_reserve_bytes == int(1.25 * 1024**3)
    worker._reverse_slot_pool = object()
    with pytest.raises(RuntimeError, match="after slot allocation"):
        worker.configure_reverse_preflight(batch_cap=2)


def test_vllm_nhd_page_extraction_preserves_logical_token_order() -> None:
    cache = torch.arange(3 * 2 * 4 * 2 * 3).reshape(3, 2, 4, 2, 3)
    extracted = extract_vllm_nhd_tokens(cache, block_ids=[2, 0], block_size=4, num_tokens=6)
    expected = torch.cat((cache[2].transpose(0, 1), cache[0].transpose(0, 1)), dim=0)[:6]
    torch.testing.assert_close(extracted, expected)

    legacy_cache = cache.permute(1, 0, 2, 3, 4).contiguous()
    legacy_extracted = extract_vllm_nhd_tokens(legacy_cache, block_ids=[2, 0], block_size=4, num_tokens=6)
    torch.testing.assert_close(legacy_extracted, expected)


def test_reverse_microbatch_partition_and_call_count() -> None:
    lengths = [9, 8, 12, 4]
    assert _partition_reverse_microbatches(lengths, max_batch_size=3, max_batch_tokens=20) == [
        [2],
        [0, 1],
        [3],
    ]
    assert _reverse_backward_calls([8, 13], chunk_size=5) == 3
    assert _reverse_backward_calls([8, 12], chunk_size=5) == 3


def test_reverse_wavefront_only_batches_active_trajectories() -> None:
    assert _build_reverse_wavefront([4, 4, 6, 6], chunk_size=1) == [
        (6, [2, 3]),
        (5, [2, 3]),
        (4, [0, 1, 2, 3]),
        (3, [0, 1, 2, 3]),
        (2, [0, 1, 2, 3]),
        (1, [0, 1, 2, 3]),
    ]


def test_zero_loss_synthetic_padding_is_not_trainable() -> None:
    real = torch.tensor([1, 1, 0], dtype=torch.int64)
    padding = torch.zeros(1, dtype=torch.int64)

    assert _has_valid_response(TensorDict({"response_mask": real}, batch_size=[])) is True
    assert _has_valid_response(TensorDict({"response_mask": padding}, batch_size=[])) is False


def test_prepare_config_installs_connector_only_for_streamopd() -> None:
    config = OmegaConf.create(
        {
            "trainer": {
                "use_v1": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "v1": {
                    "trainer_mode": "streamopd",
                    "streamopd": {},
                    "sampler": {"max_off_policy_threshold": 8, "max_off_policy_strategy": "drop"},
                },
            },
            "data": {"train_batch_size": 128},
            "algorithm": {"filter_groups": {"enable": True, "metric": "acc"}},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "nnodes": 1,
                    "n_gpus_per_node": 2,
                    "checkpoint_engine": {"backend": "nccl"},
                    "engine_kwargs": {},
                },
                "actor": {
                    "strategy": "fsdp",
                    "ppo_epochs": 1,
                    "loss_agg_mode": "token-mean",
                    "use_torch_compile": False,
                },
            },
            "distillation": {
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "streamopd_kv": {
                    "enabled": True,
                    "runtime_profile": "manual",
                    "kv_handoff_dir": "/tmp/test-streamopd",
                },
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk",
                    "use_policy_gradient": False,
                    "use_task_rewards": False,
                },
            },
        }
    )
    prepare_streamopd_kv_config(config)
    connector = config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config
    assert connector.kv_connector == "StreamOPDKVConnector"
    assert connector.kv_connector_extra_config.streamopd_kv_handoff_dir == "/tmp/test-streamopd"
    assert connector.kv_connector_extra_config.streamopd_host_slot_count == 128
    assert connector.kv_connector_extra_config.streamopd_host_slot_tokens == 4096
    assert OmegaConf.to_container(config.trainer.v1.streamopd) == {}
    assert config.trainer.v1.sampler.max_off_policy_threshold == 1
    assert config.trainer.v1.sampler.max_off_policy_strategy == "drop"
    assert config.algorithm.filter_groups.enable is False
    assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "nccl"
    assert config.distillation.streamopd_kv.reverse_slot_max_tokens == 4096
    assert config.distillation.streamopd_kv.reverse_batch_size == 128
    assert config.distillation.streamopd_kv.reverse_batch_max_tokens == 128 * 4096
    assert config.distillation.streamopd_kv.reverse_chunk_size == 1024
    assert config.distillation.streamopd_kv.reverse_chunk_min_size == 64

    stale_config = copy.deepcopy(config)
    stale_config.trainer.v1.trainer_mode = "sync"
    with pytest.raises(ValueError, match="trainer.v1.trainer_mode=streamopd"):
        prepare_streamopd_kv_config(stale_config)

    multi_sample_config = copy.deepcopy(config)
    multi_sample_config.actor_rollout_ref.rollout.n = 2
    with pytest.raises(NotImplementedError, match="rollout.n=1"):
        prepare_streamopd_kv_config(multi_sample_config)

    mismatched_shared = copy.deepcopy(config)
    mismatched_shared.distillation.n_gpus_per_node = 3
    with pytest.raises(ValueError, match="cover every Teacher GPU"):
        prepare_streamopd_kv_config(mismatched_shared)

    dedicated = copy.deepcopy(config)
    dedicated.distillation.streamopd_kv.trainer_placement = "dedicated"
    dedicated.distillation.n_gpus_per_node = 3
    prepare_streamopd_kv_config(dedicated)

    rollout_shared = copy.deepcopy(config)
    rollout_shared.distillation.streamopd_kv.trainer_placement = "rollout"
    prepare_streamopd_kv_config(rollout_shared)

    oversized_rollout = copy.deepcopy(rollout_shared)
    oversized_rollout.actor_rollout_ref.rollout.n_gpus_per_node = 3
    with pytest.raises(ValueError, match="cover every Rollout GPU"):
        prepare_streamopd_kv_config(oversized_rollout)

    union = copy.deepcopy(config)
    union.trainer.n_gpus_per_node = 4
    union.distillation.streamopd_kv.trainer_placement = "union"
    prepare_streamopd_kv_config(union)

    undersized_union = copy.deepcopy(union)
    undersized_union.trainer.n_gpus_per_node = 3
    undersized_union.distillation.n_gpus_per_node = 2
    undersized_union.actor_rollout_ref.rollout.n_gpus_per_node = 2
    with pytest.raises(ValueError, match="disjoint Teacher and Rollout"):
        prepare_streamopd_kv_config(undersized_union)

    baseline_placement_flag = copy.deepcopy(config)
    baseline_placement_flag.distillation.colocate_teacher_with_student = True
    with pytest.raises(ValueError, match="sync-baseline option"):
        prepare_streamopd_kv_config(baseline_placement_flag)


def test_auto_runtime_profile_needs_only_resource_allocation() -> None:
    config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 128,
                "max_prompt_length": 1024,
                "max_response_length": 3072,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 2,
                    "data_parallel_size": 1,
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n": 1,
                    "max_model_len": None,
                    "max_num_seqs": 1024,
                    "gpu_memory_utilization": 0.3,
                    "checkpoint_engine": {"backend": "naive"},
                }
            },
            "distillation": {
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "teacher_models": {
                    "teacher_model": {
                        "inference": {
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": None,
                            "max_num_seqs": 1024,
                            "max_num_batched_tokens": 8192,
                            "gpu_memory_utilization": 0.5,
                        }
                    }
                },
                "streamopd_kv": {"trainer_placement": "teacher"},
            },
        }
    )

    resource_config = (
        config.actor_rollout_ref.rollout.n_gpus_per_node,
        config.actor_rollout_ref.rollout.nnodes,
        config.actor_rollout_ref.rollout.tensor_model_parallel_size,
        config.actor_rollout_ref.rollout.pipeline_model_parallel_size,
        config.distillation.n_gpus_per_node,
        config.distillation.nnodes,
    )
    plan = _auto_streamopd_runtime_profile(config)
    assert plan == {
        "profile": "auto",
        "trajectory_tokens": 4096,
        "token_chunk_size": 384,
        "teacher_max_batched_tokens": 4096,
        "teacher_gpu_memory_utilization": 0.25,
        "teacher_max_num_seqs": 64,
        "rollout_gpu_memory_utilization_ceiling": 0.9,
        "rollout_max_num_seqs": 64,
    }
    assert config.actor_rollout_ref.rollout.nnodes == 1
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert config.actor_rollout_ref.rollout.pipeline_model_parallel_size == 1
    assert config.actor_rollout_ref.rollout.n == 1
    assert resource_config == (
        config.actor_rollout_ref.rollout.n_gpus_per_node,
        config.actor_rollout_ref.rollout.nnodes,
        config.actor_rollout_ref.rollout.tensor_model_parallel_size,
        config.actor_rollout_ref.rollout.pipeline_model_parallel_size,
        config.distillation.n_gpus_per_node,
        config.distillation.nnodes,
    )
    assert config.actor_rollout_ref.rollout.max_model_len == 4097
    assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "host"
    assert config.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes == 512
    assert config.distillation.streamopd_kv.reverse_batch_size == 0
    assert config.distillation.streamopd_kv.reverse_chunk_size == 0
    assert config.distillation.streamopd_kv.teacher_prefill_max_active_kv_tokens == 0
    assert config.distillation.streamopd_kv.kv_handoff_dir.startswith("/dev/shm/verl-streamopd-kv-")

    config.distillation.streamopd_kv.trainer_placement = "rollout"
    shared_rollout_plan = _auto_streamopd_runtime_profile(config)
    assert shared_rollout_plan["teacher_gpu_memory_utilization"] == 0.9
    assert shared_rollout_plan["rollout_gpu_memory_utilization_ceiling"] == 0.9
    assert config.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes == 128

    config.data.train_batch_size = 256
    config.data.max_response_length = 7168
    config.actor_rollout_ref.rollout.n_gpus_per_node = 4
    config.distillation.n_gpus_per_node = 4
    config.distillation.streamopd_kv.trainer_placement = "dedicated"
    long_context_plan = _auto_streamopd_runtime_profile(config)
    assert long_context_plan["trajectory_tokens"] == 8192
    assert long_context_plan["token_chunk_size"] == 896
    assert long_context_plan["teacher_max_batched_tokens"] == 8192
    assert long_context_plan["teacher_max_num_seqs"] == 64
    assert long_context_plan["rollout_max_num_seqs"] == 64
    assert long_context_plan["teacher_gpu_memory_utilization"] == 0.9
    assert long_context_plan["rollout_gpu_memory_utilization_ceiling"] == 0.9


def test_auto_runtime_profile_preserves_explicit_existing_options() -> None:
    config = OmegaConf.create(
        {
            "data": {"train_batch_size": 128, "max_prompt_length": 1024, "max_response_length": 3072},
            "actor_rollout_ref": {
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 1,
                    "data_parallel_size": 1,
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n": 1,
                    "max_model_len": 6000,
                    "max_num_seqs": 48,
                    "gpu_memory_utilization": 0.7,
                    "checkpoint_engine": {"backend": "host", "update_weights_bucket_megabytes": 256},
                }
            },
            "distillation": {
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "teacher_models": {
                    "teacher_model": {
                        "inference": {
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": 6000,
                            "max_num_seqs": 24,
                            "max_num_batched_tokens": 3072,
                            "gpu_memory_utilization": 0.3,
                        }
                    }
                },
                "streamopd_kv": {"trainer_placement": "teacher", "token_chunk_size": 512},
            },
        }
    )
    explicit = {
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        "actor_rollout_ref.rollout.max_num_seqs",
        "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens",
        "distillation.streamopd_kv.token_chunk_size",
    }

    _auto_streamopd_runtime_profile(config, explicit_paths=explicit)

    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.7
    assert config.actor_rollout_ref.rollout.max_num_seqs == 48
    assert config.distillation.teacher_models.teacher_model.inference.max_num_batched_tokens == 3072
    assert config.distillation.streamopd_kv.token_chunk_size == 512
    assert config.actor_rollout_ref.rollout.max_model_len == 4097
    assert config.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization == 0.25


def test_prepare_config_applies_auto_runtime_profile_before_validation() -> None:
    config = OmegaConf.create(
        {
            "trainer": {
                "use_v1": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "v1": {
                    "trainer_mode": "streamopd",
                    "streamopd": {},
                    "sampler": {"max_off_policy_threshold": 8, "max_off_policy_strategy": "drop"},
                },
            },
            "data": {"train_batch_size": 128, "max_prompt_length": 1024, "max_response_length": 3072},
            "algorithm": {"filter_groups": None},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n": 1,
                    "nnodes": 1,
                    "n_gpus_per_node": 2,
                    "max_model_len": None,
                    "max_num_seqs": 1024,
                    "gpu_memory_utilization": 0.5,
                    "checkpoint_engine": {"backend": "naive", "update_weights_bucket_megabytes": 2048},
                    "engine_kwargs": {},
                },
                "actor": {
                    "strategy": "fsdp",
                    "ppo_epochs": 1,
                    "loss_agg_mode": "token-mean",
                    "use_torch_compile": False,
                },
            },
            "distillation": {
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "teacher_models": {
                    "teacher_model": {
                        "inference": {
                            "tensor_model_parallel_size": 1,
                            "data_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": None,
                            "max_num_seqs": 1024,
                            "max_num_batched_tokens": 8192,
                            "gpu_memory_utilization": 0.5,
                        }
                    }
                },
                "streamopd_kv": {
                    "enabled": True,
                    "runtime_profile": "auto",
                    "trainer_placement": "teacher",
                    "kv_handoff_dir": "/tmp/test-streamopd-auto",
                },
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk",
                    "use_policy_gradient": False,
                    "use_task_rewards": False,
                },
            },
        }
    )

    prepare_streamopd_kv_config(config)

    assert config.distillation.streamopd_kv.runtime_profile == "auto"
    assert config.distillation.streamopd_kv.token_chunk_size == 384
    assert config.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization == 0.25
    assert config.actor_rollout_ref.rollout.max_num_seqs == 64
    assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "host"
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.9
    assert config.distillation.teacher_models.teacher_model.inference.max_num_batched_tokens == 4096
    assert config.distillation.streamopd_kv.reverse_chunk_size == 1024
    assert config.distillation.streamopd_kv.reverse_batch_size == 128


def test_prepare_config_does_not_mutate_sync_baseline() -> None:
    config = OmegaConf.create(
        {
            "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync", "sync": {}}},
            "distillation": {
                "colocate_teacher_with_student": True,
                "streamopd_kv": {"enabled": False, "kv_handoff_dir": "/tmp/unused"},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "engine_kwargs": {"vllm": {"enforce_eager": True}},
                    "checkpoint_engine": {"backend": "naive"},
                }
            },
        }
    )
    before = OmegaConf.to_container(config, resolve=False)

    prepare_streamopd_kv_config(config)

    assert OmegaConf.to_container(config, resolve=False) == before
    assert "kv_transfer_config" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm


def test_teacher_priority_scheduler_enforces_version_barrier() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(7)
    scheduler.teacher_notified(7)
    scheduler.teacher_enqueued(7)
    scheduler.teacher_started(7)
    state = scheduler.snapshot()
    assert state["teacher_pending"] == 1
    with pytest.raises(RuntimeError, match="unfinished work"):
        scheduler.end_policy(7)

    scheduler.teacher_finished(7)
    scheduler.training_started(7)
    with pytest.raises(RuntimeError, match="unfinished work"):
        scheduler.end_policy(7)
    scheduler.training_finished(7)
    metrics = scheduler.end_policy(7)
    assert metrics["streamopd/scheduler_teacher_chunks"] == 1
    assert metrics["streamopd/scheduler_teacher_notifications"] == 1
    assert metrics["streamopd/scheduler_teacher_coalesced_fragments"] == 0
    assert metrics["streamopd/scheduler_training_units"] == 1
    assert metrics["streamopd/scheduler_pool_busy_seconds"] >= 0
    with pytest.raises(RuntimeError, match="no active policy"):
        scheduler.teacher_enqueued(8)


def test_streamopd_resource_planners(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.experimental.streamopd_kv.planning import (
        checkpoint_weight_bytes,
        kv_bytes_per_token,
        minimum_device_total_bytes,
        partition_training_units,
        plan_host_kv,
        plan_teacher_admission,
        plan_vllm_memory,
        planned_reverse_width,
        shared_vllm_utilization_limit,
    )

    assert partition_training_units(128, 16) == [16] * 8
    assert partition_training_units(130, 16) == [16] * 8 + [2]
    assert (
        planned_reverse_width(
            [{"slot_batch_size": 8.0}, [{"slot_batch_size": 4.0}]],
            fallback=16,
        )
        == 4
    )
    assert plan_teacher_admission(
        expected_trajectories=32,
        trajectory_tokens=1000,
        vllm_capacity_tokens=40000,
        page_size=64,
        max_batched_tokens=2048,
        initial_chunk_tokens=384,
        train_launch_width=8,
    ) == {
        "active_trajectories": 8,
        "active_kv_tokens": 8192,
        "vllm_capacity_tokens": 40000,
        "safe_capacity_tokens": 29952,
        "trajectory_tokens": 1024,
        "prefill_wave": 5,
    }
    capped = plan_teacher_admission(
        expected_trajectories=32,
        trajectory_tokens=4096,
        vllm_capacity_tokens=20000,
        page_size=64,
        max_batched_tokens=2048,
        initial_chunk_tokens=1024,
        train_launch_width=8,
        trajectory_cap=16,
        token_cap=12000,
    )
    assert capped["active_trajectories"] == 2
    assert capped["active_kv_tokens"] == 8192
    with pytest.raises(ValueError, match="cannot fit one trajectory"):
        plan_teacher_admission(
            expected_trajectories=8,
            trajectory_tokens=4096,
            vllm_capacity_tokens=20000,
            page_size=64,
            max_batched_tokens=2048,
            initial_chunk_tokens=1024,
            train_launch_width=4,
            token_cap=2048,
        )

    assert minimum_device_total_bytes([{"total_bytes": 80 * 1024**3}, [{"total_bytes": 79 * 1024**3}]]) == 79 * 1024**3
    shared_limit = shared_vllm_utilization_limit(
        [
            {"free_bytes": 30 * 1024**3, "total_bytes": 80 * 1024**3},
            {"free_bytes": 68 * 1024**3, "total_bytes": 80 * 1024**3},
        ],
        rank_offset=1,
        world_size=1,
        required_free_bytes=28 * 1024**3,
    )
    assert shared_limit == {
        "utilization_limit": 0.5,
        "free_gib": 68.0,
        "total_gib": 80.0,
        "reverse_reserve_gib": 28.0,
    }
    with pytest.raises(ValueError, match="insufficient memory"):
        shared_vllm_utilization_limit(
            [{"free_bytes": 30 * 1024**3, "total_bytes": 80 * 1024**3}],
            rank_offset=0,
            world_size=1,
            required_free_bytes=20 * 1024**3,
        )
    inferred_4k = plan_vllm_memory(
        total_memory_bytes=80 * 1024**3,
        weight_bytes=4063479808,
        kv_bytes_per_token=112 * 1024,
        requested_max_num_seqs=128,
        max_model_len=4096,
        utilization_limit=0.9,
        max_num_seqs_explicit=False,
        utilization_explicit=False,
    )
    assert inferred_4k["max_num_seqs"] == 128
    assert inferred_4k["gpu_memory_utilization"] == 0.9

    inferred_8k = plan_vllm_memory(
        total_memory_bytes=80 * 1024**3,
        weight_bytes=4063479808,
        kv_bytes_per_token=112 * 1024,
        requested_max_num_seqs=128,
        max_model_len=8192,
        utilization_limit=0.9,
        max_num_seqs_explicit=False,
        utilization_explicit=False,
    )
    assert inferred_8k["max_num_seqs"] == 64
    assert inferred_8k["gpu_memory_utilization"] == 0.9

    memory_capped_8k = plan_vllm_memory(
        total_memory_bytes=80 * 1024**3,
        weight_bytes=4063479808,
        kv_bytes_per_token=112 * 1024,
        requested_max_num_seqs=128,
        max_model_len=8192,
        utilization_limit=0.5,
        max_num_seqs_explicit=False,
        utilization_explicit=True,
    )
    assert memory_capped_8k["max_num_seqs"] == 32
    assert memory_capped_8k["gpu_memory_utilization"] == 0.5

    with pytest.raises(ValueError, match="explicitly requested"):
        plan_vllm_memory(
            total_memory_bytes=80 * 1024**3,
            weight_bytes=4063479808,
            kv_bytes_per_token=112 * 1024,
            requested_max_num_seqs=128,
            max_model_len=8192,
            utilization_limit=0.5,
            max_num_seqs_explicit=True,
            utilization_explicit=True,
        )

    monkeypatch.setattr(
        "verl.experimental.streamopd_kv.planning.shutil.disk_usage",
        lambda path: SimpleNamespace(total=200 * 1024**3, used=0, free=200 * 1024**3),
    )
    host_plan = plan_host_kv(
        handoff_dir=str(tmp_path / "handoff"),
        global_batch_size=128,
        max_model_len=4096,
        kv_bytes_per_token=112 * 1024,
    )
    assert host_plan["host_kv_required_gib"] == 56.0
    with pytest.raises(ValueError, match="Host KV backing"):
        plan_host_kv(
            handoff_dir=str(tmp_path / "handoff"),
            global_batch_size=512,
            max_model_len=8192,
            kv_bytes_per_token=112 * 1024,
        )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text('{"metadata": {"total_size": 12345}}')
    (model_dir / "config.json").write_text(
        '{"num_hidden_layers": 4, "num_key_value_heads": 2, '
        '"num_attention_heads": 8, "hidden_size": 512, "head_dim": 64}'
    )
    assert checkpoint_weight_bytes(str(model_dir)) == 12345
    assert kv_bytes_per_token(str(model_dir), "bfloat16") == 4 * 2 * 64 * 2 * 2


def test_teacher_priority_scheduler_rejects_policy_staleness() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(3)
    with pytest.raises(RuntimeError, match="policy mismatch"):
        scheduler.teacher_enqueued(2)


def test_teacher_and_training_admission_are_mutually_exclusive() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(11)
    scheduler.teacher_enqueued(11)
    assert scheduler.try_training_started(11, teacher_queue_threshold=0) is False
    assert scheduler.try_teacher_started(11) is True
    assert scheduler.try_training_started(11, teacher_queue_threshold=1) is False
    scheduler.teacher_enqueued(11)
    scheduler.teacher_finished(11)
    assert scheduler.try_training_started(11, teacher_queue_threshold=1) is True
    assert scheduler.try_teacher_started(11) is False
    scheduler.training_finished(11)
    assert scheduler.try_teacher_started(11) is True
    scheduler.teacher_finished(11)
    scheduler.end_policy(11)


def test_teacher_admission_waits_for_asynchronous_wake() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(12, teacher_available=False)
    scheduler.teacher_enqueued(12)

    assert scheduler.try_teacher_started(12) is False
    assert scheduler.snapshot()["teacher_available"] is False
    scheduler.teacher_wake_completed(12)
    assert scheduler.try_teacher_started(12) is True
    scheduler.teacher_finished(12)
    scheduler.end_policy(12)


def test_ready_training_waiter_wins_tie_with_teacher_queue() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(14)
    scheduler.training_waiting(14, teacher_queue_threshold=1)
    scheduler.teacher_enqueued(14)
    assert scheduler.try_teacher_started(14) is False
    assert scheduler.try_training_started(14, teacher_queue_threshold=1) is True
    assert scheduler.snapshot()["training_waiters"] == 0
    scheduler.training_finished(14)
    scheduler.teacher_cancelled(14)
    scheduler.end_policy(14)


def test_adaptive_shared_scheduler_uses_ready_backlog_hysteresis() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(20, expected_trajectories=12, scheduler_policy="adaptive", train_launch_width=4)
    scheduler.training_waiting(20, teacher_queue_threshold=0, trajectory_count=4)
    scheduler.teacher_enqueued(20)
    for _ in range(4):
        scheduler.teacher_trajectory_completed(20)

    # A genuinely ready cohort starts immediately; no second cohort gate.
    assert scheduler.try_teacher_started(20) is False
    assert scheduler.try_training_started(20, teacher_queue_threshold=0) is True
    scheduler.training_finished(20)

    for _ in range(4):
        scheduler.teacher_trajectory_completed(20)
    scheduler.training_waiting(20, teacher_queue_threshold=0, trajectory_count=4)
    # The completed-but-unlaunched cohort keeps ownership on Trainer.
    assert scheduler.try_teacher_started(20) is False
    assert scheduler.try_training_started(20, teacher_queue_threshold=0) is True
    scheduler.training_finished(20)

    # Once Trainer catches up, pending Teacher work gets the next turn.
    assert scheduler.try_teacher_started(20) is True
    scheduler.teacher_finished(20)
    assert scheduler.snapshot()["forced_teacher_turns"] == 1
    assert scheduler.snapshot()["max_training_burst"] == 2


def test_adaptive_shared_scheduler_bounds_next_training_handoff() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(25, expected_trajectories=12, scheduler_policy="adaptive", train_launch_width=4)
    for _ in range(8):
        scheduler.teacher_trajectory_completed(25)
    scheduler.training_waiting(25, teacher_queue_threshold=0, trajectory_count=4)
    assert scheduler.try_training_started(25, teacher_queue_threshold=0) is True
    scheduler.training_finished(25)
    scheduler.teacher_enqueued(25)

    # Reserve a short controller handoff window for the next ready cohort.
    assert scheduler.try_teacher_started(25) is False
    scheduler.last_training_finished_at -= scheduler.teacher_turn_grace_seconds
    assert scheduler.try_teacher_started(25) is True
    scheduler.teacher_finished(25)


def test_adaptive_shared_scheduler_resumes_training_after_teacher_grace() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(24, scheduler_policy="adaptive", train_launch_width=4)
    scheduler.training_waiting(24, teacher_queue_threshold=0, trajectory_count=4)
    assert scheduler.try_training_started(24, teacher_queue_threshold=0) is True
    scheduler.training_finished(24)
    scheduler.training_waiting(24, teacher_queue_threshold=0, trajectory_count=4)
    assert scheduler.try_training_started(24, teacher_queue_threshold=0) is False
    scheduler.last_training_finished_at -= scheduler.teacher_turn_grace_seconds
    assert scheduler.try_training_started(24, teacher_queue_threshold=0) is True
    scheduler.training_finished(24)


def test_dedicated_teacher_and_trainer_resources_can_run_concurrently() -> None:
    scheduler = StreamOPDTaskScheduler(teacher_resources=("teacher",), trainer_resources=("trainer",))
    scheduler.begin_policy(21)
    scheduler.teacher_enqueued(21)
    assert scheduler.try_teacher_started(21) is True
    scheduler.training_waiting(21, teacher_queue_threshold=0, trajectory_count=4)
    assert scheduler.try_training_started(21, teacher_queue_threshold=0) is True
    state = scheduler.snapshot()
    assert state["teacher_active"] == 1
    assert state["training_active"] == 1
    assert state["resources_overlap"] is False
    scheduler.training_finished(21)
    scheduler.teacher_finished(21)
    metrics = scheduler.end_policy(21)
    assert metrics["streamopd/scheduler_resources_overlap"] == 0
    assert metrics["streamopd/scheduler_concurrent_busy_seconds"] >= 0


def test_teacher_then_train_policy_waits_for_streaming_teacher_drain() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(
        22,
        expected_trajectories=2,
        scheduler_policy="teacher_then_train",
        train_launch_width=2,
    )
    scheduler.training_waiting(22, teacher_queue_threshold=0, trajectory_count=2)
    assert scheduler.try_training_started(22, teacher_queue_threshold=0) is False
    for _ in range(2):
        scheduler.teacher_trajectory_terminal_submitted(22)
        scheduler.teacher_trajectory_completed(22)
    assert scheduler.snapshot()["teacher_drained"] is True
    assert scheduler.try_training_started(22, teacher_queue_threshold=0) is True
    scheduler.training_finished(22)
    scheduler.end_policy(22)


def test_adaptive_scheduler_drains_a_single_teacher_tail_quantum() -> None:
    scheduler = StreamOPDTaskScheduler(teacher_resources=("teacher",), trainer_resources=("trainer",))
    scheduler.begin_policy(23, expected_trajectories=8, scheduler_policy="adaptive", train_launch_width=4)
    for _ in range(8):
        scheduler.teacher_trajectory_terminal_submitted(23)
    for _ in range(4):
        scheduler.teacher_trajectory_completed(23)
    scheduler.training_waiting(23, teacher_queue_threshold=0, trajectory_count=4)
    assert scheduler.snapshot()["teacher_drained"] is False
    assert scheduler.try_training_started(23, teacher_queue_threshold=0) is False
    for _ in range(4):
        scheduler.teacher_trajectory_completed(23)
    assert scheduler.try_training_started(23, teacher_queue_threshold=0) is True
    scheduler.training_finished(23)
    scheduler.end_policy(23)


def test_teacher_admission_limits_active_trajectory_count_and_kv_tokens() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(12)
    for _ in range(3):
        scheduler.teacher_enqueued(12)

    assert scheduler.try_teacher_started(12, 4, max_active_trajectories=2, max_active_kv_tokens=7) is True
    assert scheduler.try_teacher_started(12, 4, max_active_trajectories=2, max_active_kv_tokens=7) is False
    assert scheduler.snapshot()["teacher_active_kv_tokens"] == 4
    scheduler.teacher_finished(12, 4)
    assert scheduler.try_teacher_started(12, 4, max_active_trajectories=2, max_active_kv_tokens=7) is True
    scheduler.teacher_finished(12, 4)
    scheduler.teacher_cancelled(12)
    scheduler.end_policy(12)


def test_teacher_session_reservation_is_held_until_eos() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(13)
    assert scheduler.try_teacher_session_admitted(13, "a", 4096, 16, 8192) is True
    assert scheduler.try_teacher_session_admitted(13, "b", 4096, 16, 8192) is True
    assert scheduler.try_teacher_session_admitted(13, "c", 4096, 16, 8192) is False
    state = scheduler.snapshot()
    assert state["teacher_sessions"] == 2
    assert state["teacher_session_kv_tokens"] == 8192
    with pytest.raises(RuntimeError, match="teacher_sessions=2"):
        scheduler.end_policy(13)
    scheduler.teacher_session_released(13, "a")
    assert scheduler.try_teacher_session_admitted(13, "c", 4096, 16, 8192) is True
    scheduler.teacher_session_released(13, "b")
    scheduler.teacher_session_released(13, "c")
    scheduler.end_policy(13)


def test_teacher_session_slot_refills_after_eos() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(15)
    assert scheduler.try_teacher_session_admitted(15, "a", 10, 2, 20) is True
    assert scheduler.try_teacher_session_admitted(15, "b", 10, 2, 20) is True
    assert scheduler.try_teacher_session_admitted(15, "c", 10, 2, 20) is False
    scheduler.teacher_session_released(15, "a")
    assert scheduler.try_teacher_session_admitted(15, "c", 10, 2, 20) is True
    scheduler.teacher_session_released(15, "b")
    scheduler.teacher_session_released(15, "c")
    scheduler.end_policy(15)


def test_policy_barrier_rejects_missing_teacher_trajectory() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(18, expected_trajectories=2)
    scheduler.teacher_trajectory_terminal_submitted(18)
    scheduler.teacher_trajectory_completed(18)
    with pytest.raises(RuntimeError, match="incomplete trajectories"):
        scheduler.end_policy(18)


@pytest.mark.asyncio
async def test_teacher_client_holds_load_balancer_lease_until_stream_eos() -> None:
    from verl.workers.rollout.llm_server import LLMServerClient

    class RemoteMethod:
        async def remote(self, **kwargs):
            return {"terminal": kwargs["terminal"]}

    class Server:
        stream_teacher_chunk = RemoteMethod()

    class Client(LLMServerClient):
        def __init__(self):
            super().__init__(config=object())
            self.acquire_count = 0
            self.releases = []
            self.server = Server()

        async def _acquire_server(self, request_id):
            self.acquire_count += 1
            return "teacher-0", self.server

        def _release_server(self, server_id):
            self.releases.append(server_id)

    client = Client()
    params = {"prompt_logprobs": 2}
    await client.stream_teacher_chunk("trajectory", token_ids=[1, 2], sampling_params=params, terminal=False)
    await client.stream_teacher_chunk("trajectory", token_ids=[3, 4], sampling_params=params, terminal=False)
    assert client.acquire_count == 1
    assert client.releases == []
    await client.stream_teacher_chunk("trajectory", token_ids=[], sampling_params=params, terminal=True)
    assert client.acquire_count == 1
    assert client.releases == ["teacher-0"]


@pytest.mark.parametrize("use_chunked_topk", [False, True])
def test_reverse_forward_kl_topk_matches_native_numerical_path(use_chunked_topk: bool) -> None:
    torch.manual_seed(31)
    logits = torch.randn(7, 29, dtype=torch.bfloat16, requires_grad=True)
    native_logits = logits.detach().clone().requires_grad_(True)
    teacher_ids = torch.randint(0, logits.shape[-1], (7, 5))
    teacher_logprobs = torch.log_softmax(torch.randn(7, 5), dim=-1)
    temperature = 0.73

    reverse_loss = _forward_kl_topk_sum(
        logits,
        teacher_ids,
        teacher_logprobs,
        temperature=temperature,
        use_chunked_topk=use_chunked_topk,
        log_prob_min_clamp=-10.0,
        loss_max_clamp=10.0,
    )
    scaled_native_logits = native_logits / torch.as_tensor(temperature, dtype=native_logits.dtype)
    if use_chunked_topk:
        native_student = _chunked_topk_log_probs(
            scaled_native_logits.unsqueeze(0), teacher_ids.unsqueeze(0), chunk_size=3
        ).squeeze(0)
    else:
        native_student = F.log_softmax(scaled_native_logits, dim=-1).gather(-1, teacher_ids)
    native_student = native_student.clamp_min(-10.0).float()
    native_teacher = teacher_logprobs.clamp_min(-10.0).float()
    native_loss = (native_teacher.exp() * (native_teacher - native_student)).sum(-1).clamp_min(0.0)
    native_loss = native_loss.clamp_max(10.0).sum()

    reverse_loss.backward()
    native_loss.backward()
    torch.testing.assert_close(reverse_loss, native_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(logits.grad, native_logits.grad, rtol=0.0, atol=0.0)


def test_reverse_forward_kl_topk_temperature_one_avoids_logits_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = torch.randn(3, 11, requires_grad=True)
    teacher_ids = torch.randint(0, logits.shape[-1], (3, 2))
    teacher_logprobs = torch.log_softmax(torch.randn(3, 2), dim=-1)
    original_log_softmax = torch.log_softmax

    def assert_input_aliases_logits(value: torch.Tensor, *args, **kwargs):
        assert value.data_ptr() == logits.data_ptr()
        return original_log_softmax(value, *args, **kwargs)

    monkeypatch.setattr(torch, "log_softmax", assert_input_aliases_logits)
    loss = _forward_kl_topk_sum(
        logits,
        teacher_ids,
        teacher_logprobs,
        temperature=1.0,
        use_chunked_topk=False,
        log_prob_min_clamp=-10.0,
        loss_max_clamp=10.0,
    )
    loss.backward()
    assert logits.grad is not None


def test_distillation_config_coerces_streamopd_mapping_without_mutating_frozen_field() -> None:
    config = DistillationConfig(
        enabled=False,
        streamopd_kv={
            "enabled": False,
            "reverse_chunk_size": 32,
            "reverse_chunk_min_size": 16,
            "reverse_page_size": 16,
        },
    )
    assert isinstance(config.streamopd_kv, StreamOPDKVConfig)
    assert config.streamopd_kv.reverse_chunk_size == 32


def test_streamopd_runtime_profile_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="runtime_profile"):
        StreamOPDKVConfig(runtime_profile="guess")


def test_dapo_adapter_wraps_plain_prompt_as_chat_messages(monkeypatch) -> None:
    from benchmarks.streamopd_kv.dapo_math_dataset import DAPOMathDataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    dataset = DAPOMathDataset.__new__(DAPOMathDataset)
    assert dataset._build_messages({"prompt": "2 + 2?"}, "prompt") == [{"role": "user", "content": "2 + 2?"}]
    monkeypatch.setattr(RLHFDataset, "__getitem__", lambda _self, item: {"item": item})
    monkeypatch.setenv("STREAMOPD_RAGGED_RESPONSE_LENGTHS", "256,768")
    assert dataset[100]["max_response_tokens"] == 256
    assert dataset[7]["max_response_tokens"] == 768


def test_streamopd_trainer_does_not_expose_unused_reward_handles() -> None:
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    assert trainer.get_reward_handles() is None
    assert trainer._get_required_batch_multiple(dp_size=3) == 3
    assert trainer._optimizer_updates_per_global_step() == 1
    metric_data = TensorDict({"responses": torch.ones(2, 3, dtype=torch.long)}, batch_size=[2])
    prepared = trainer._prepare_metric_tensors(metric_data)
    assert torch.equal(prepared["rm_scores"], torch.zeros(2, 3))


def test_union_topology_falls_back_to_drain_first_policy() -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.config = OmegaConf.create(
        {"distillation": {"streamopd_kv": {"trainer_placement": "union", "scheduler_policy": "adaptive"}}}
    )
    trainer.placement = TrainerPlacement.UNION
    assert trainer._effective_scheduler_policy() == "teacher_then_train"
    trainer.config.distillation.streamopd_kv.trainer_placement = "rollout"
    trainer.placement = TrainerPlacement.ROLLOUT
    assert trainer._effective_scheduler_policy() == "adaptive"


def test_vllm_connector_waits_for_terminal_seal_and_ignores_unclaimed_request() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector

    class SealFuture:
        complete = False

        def done(self) -> bool:
            return self.complete

        def result(self) -> None:
            assert self.complete

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._connector_metadata = None
    connector._finished_requests = set()
    connector._claimed_requests = {"claimed"}
    connector._seal_futures = {}
    assert connector.get_finished({"claimed", "ordinary"}) == (None, None)

    future = SealFuture()
    connector._seal_futures["claimed"] = future
    assert connector.get_finished(set()) == (None, None)
    future.complete = True
    assert connector.get_finished(set()) == ({"claimed"}, None)
    assert not connector._claimed_requests


def test_vllm_connector_finish_is_idempotent() -> None:
    from types import SimpleNamespace

    from verl.experimental.streamopd_kv.vllm_connector import (
        StreamOPDKVConnector,
        _SchedulerSaveState,
    )

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._scheduler_paths = {"backend-id": "/tmp/streamopd/backend-id"}
    connector._scheduler_states = {
        "backend-id": _SchedulerSaveState(
            req_id="backend-id",
            trajectory_id="trajectory-id",
            slot_path="/tmp/streamopd/backend-id",
            block_ids_by_group=[[0, 1]],
            policy_version=3,
            prompt_length=1,
        )
    }
    connector._pending = []
    connector._claimed_requests = set()
    connector._block_size = 2
    connector._chunk_size = 4
    connector._tp_size = 1
    request = SimpleNamespace(
        request_id="backend-id",
        kv_transfer_params={"streamopd_kv": True, "trajectory_id": "trajectory-id"},
        status="FINISHED_STOPPED",
        all_token_ids=[10, 11, 12],
    )

    saved, params = connector.request_finished_all_groups(request, ([0, 1],))
    assert saved is True
    assert params["streamopd_kv_path"] == "/tmp/streamopd/backend-id"
    assert len(connector._pending) == 1

    # A second vLLM finish notification must not raise or enqueue another
    # terminal save after the scheduler state has been consumed. It retains
    # page ownership until the worker reports finished_sending.
    saved, params = connector.request_finished_all_groups(request, ([0, 1],))
    assert saved is True
    assert params is None
    assert len(connector._pending) == 1

    connector.update_connector_output(SimpleNamespace(finished_sending={"backend-id"}))
    assert not connector._claimed_requests
    with pytest.raises(RuntimeError, match="no scheduler state"):
        connector.request_finished_all_groups(request, ([0, 1],))
