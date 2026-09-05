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
import threading
import time
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
    _unsharded_gradient_reserve_bytes,
)
from verl.experimental.streamopd_kv.host_slot_pool import HostKVSlotPool, cleanup_host_kv_pools
from verl.experimental.streamopd_kv.qwen3 import _build_reverse_wavefront, _wavefront_compute_end
from verl.experimental.streamopd_kv.reverse_attention import (
    FixedSlotPageState,
    ReverseKVSlotPool,
    _ContiguousKVBatchView,
)
from verl.experimental.streamopd_kv.scheduler import StreamOPDTaskScheduler
from verl.experimental.streamopd_kv.snapshot_io import (
    extract_vllm_cross_layers_nhd_token_range,
    extract_vllm_nhd_token_range,
    extract_vllm_nhd_tokens,
    load_vllm_snapshot,
    release_vllm_snapshot,
)
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.workers.config.distillation import DistillationConfig, StreamOPDKVConfig


def test_contiguous_kv_view_keeps_native_gqa_heads_and_accumulates_gradients() -> None:
    layer = SimpleNamespace(
        key=torch.arange(3 * 5 * 2 * 3, dtype=torch.float32).reshape(3, 5, 2, 3),
        value=torch.ones(3, 5, 2, 3),
        key_grad=torch.zeros(3, 5, 2, 3),
        value_grad=torch.zeros(3, 5, 2, 3),
        num_kv_heads=2,
        head_dim=3,
    )
    view = _ContiguousKVBatchView(layer, active=[0, 2], start=2, end=5)

    key, value = view.key_value(query_heads=4)
    assert key.shape == value.shape == (2, 5, 2, 3)
    torch.testing.assert_close(key[1], layer.key[2])

    key_gradient = torch.ones_like(key)
    value_gradient = torch.full_like(value, 2)
    view.accumulate_gradients(key_gradient, value_gradient)
    torch.testing.assert_close(layer.key_grad[0], torch.ones_like(layer.key_grad[0]))
    torch.testing.assert_close(layer.key_grad[1], torch.zeros_like(layer.key_grad[1]))
    torch.testing.assert_close(layer.key_grad[2], torch.ones_like(layer.key_grad[2]))
    torch.testing.assert_close(layer.value_grad[2], torch.full_like(layer.value_grad[2], 2))
    current_key_grad, current_value_grad = view.grad
    assert current_key_grad.shape == current_value_grad.shape == (2, 3, 2, 3)

    with pytest.raises(ValueError, match="query heads"):
        view.key_value(query_heads=3)


def test_reverse_slot_abort_clears_failed_group_metadata() -> None:
    pool = ReverseKVSlotPool.__new__(ReverseKVSlotPool)
    pool.batch_size = 2
    pool.num_pages = 3
    pool._current_lengths = [192]
    pool._next_sources = [[object()]]
    pool._next_lengths = [64]
    pool._next_padded_lengths = [64]
    pool._pending_enqueues = []
    pool.page_states = [[FixedSlotPageState.CURRENT_ACTIVE] * 3 for _ in range(2)]
    pool.load_events = [[object()] * 3 for _ in range(2)]
    pool.free_events = [[object()] * 3 for _ in range(2)]

    pool.abort_groups()

    assert pool._current_lengths == []
    assert pool._next_sources is None
    assert pool._next_lengths == []
    assert pool._next_padded_lengths == []
    assert pool.page_states == [[FixedSlotPageState.FREE] * 3 for _ in range(2)]
    assert pool.load_events == [[None] * 3 for _ in range(2)]
    assert pool.free_events == [[None] * 3 for _ in range(2)]


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
async def test_teacher_streaming_reports_one_scheduler_summary_per_trajectory() -> None:
    calls: list[tuple[str, tuple]] = []

    class RemoteMethod:
        def __init__(self, name: str, result=None) -> None:
            self.name = name
            self.result = result

        async def remote(self, *args):
            calls.append((self.name, args))
            return self.result

    class Scheduler:
        wait_teacher_session_admitted = RemoteMethod("admit", True)
        teacher_trajectory_terminal_submitted = RemoteMethod("terminal")
        teacher_session_released = RemoteMethod("release")
        teacher_session_admission_cancelled = RemoteMethod("cancel")
        teacher_trajectory_completed = RemoteMethod("complete")

    async def score(fragment: list[int], request_id: str, terminal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        del request_id, terminal
        ids = torch.tensor(fragment).unsqueeze(-1)
        return ids, -ids.float()

    coordinator = StreamingTeacherCoordinator(score, max_pending_chunks=4, scheduler=Scheduler())
    key = TrajectoryKey(3, "summary")
    await coordinator.submit(CommittedTokenChunk(key=key, start=0, token_ids=(10, 11), prompt_ids=(1,), terminal=False))
    await coordinator.submit(CommittedTokenChunk(key=key, start=2, token_ids=(12, 13), terminal=True))
    await coordinator.result(key, required_completion_tokens=4)

    assert [name for name, _ in calls] == ["terminal", "admit", "release", "complete"]
    assert calls[0][1] == (3, 2)
    assert len(calls[-1][1][1]) == 1


@pytest.mark.asyncio
async def test_teacher_admission_waits_once_for_scheduler_notification() -> None:
    attempts = 0
    releases = []

    class WaitMethod:
        async def remote(self, *args):
            nonlocal attempts
            attempts += 1
            return True

    class ReleaseMethod:
        async def remote(self, *args):
            releases.append(args)

    class Scheduler:
        wait_teacher_session_admitted = WaitMethod()
        teacher_session_released = ReleaseMethod()
        teacher_session_admission_cancelled = ReleaseMethod()

    async def score(fragment: list[int], request_id: str, terminal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        del request_id, terminal
        ids = torch.tensor(fragment).unsqueeze(-1)
        return ids, -ids.float()

    coordinator = StreamingTeacherCoordinator(
        score,
        max_pending_chunks=4,
        scheduler=Scheduler(),
        max_active_trajectories=1,
        max_active_kv_tokens=16,
        kv_page_size=1,
        kv_reservation_tokens=4,
    )
    key = TrajectoryKey(3, "backoff")

    await coordinator._admit_session(key, 4)
    await coordinator._release_session(key, 4)

    assert attempts == 1
    assert releases == [(3, "v3-backoff")]


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
async def test_teacher_streaming_concatenates_fragments_once_at_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.experimental.streamopd_kv import streaming_teacher

    completed = asyncio.Queue()

    async def score(fragment: list[int], request_id: str, terminal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        del request_id, terminal
        ids = torch.tensor(fragment).unsqueeze(-1)
        await completed.put(None)
        return ids, -ids.float()

    cat_calls = []
    original_cat = torch.cat

    def counted_cat(tensors, *args, **kwargs):
        cat_calls.append(len(tensors))
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(streaming_teacher.torch, "cat", counted_cat)
    coordinator = StreamingTeacherCoordinator(score, max_pending_chunks=8, kv_page_size=1)
    key = TrajectoryKey(9, "deferred-concat")
    chunks = [
        CommittedTokenChunk(key=key, start=0, token_ids=(10, 11), prompt_ids=(1, 2), terminal=False),
        CommittedTokenChunk(key=key, start=2, token_ids=(12, 13), terminal=False),
        CommittedTokenChunk(key=key, start=4, token_ids=(14, 15), terminal=True),
    ]
    for chunk in chunks:
        await coordinator.submit(chunk)
        await completed.get()
        while coordinator._sessions[key].scored_response_tokens < chunk.end:
            await asyncio.sleep(0)

    assert cat_calls == []
    ids, logprobs = await coordinator.result(key, required_completion_tokens=6)
    assert cat_calls == [3, 3]
    torch.testing.assert_close(ids[:, 0], torch.tensor([1, 2, 10, 11, 12, 13, 14, 15]))
    torch.testing.assert_close(logprobs[:, 0], -ids[:, 0].float())


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


def test_vllm_prompt_logprobs_chunked_gather_matches_full_normalization() -> None:
    from vllm.v1.sample.sampler import Sampler

    from verl.workers.rollout.vllm_rollout.utils import _gather_prompt_logprobs_in_chunks

    torch.manual_seed(17)
    logits = torch.randn(11, 31, dtype=torch.bfloat16)
    token_ids = torch.randint(0, logits.shape[-1], (logits.shape[0],))
    sampler = Sampler()
    expected = sampler.gather_logprobs(sampler.compute_logprobs(logits), 4, token_ids)
    actual = _gather_prompt_logprobs_in_chunks(sampler, logits, 4, token_ids, chunk_size=3)

    torch.testing.assert_close(actual.logprob_token_ids, expected.logprob_token_ids, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.selected_token_ranks, expected.selected_token_ranks, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.logprobs, expected.logprobs, rtol=0.0, atol=0.0)


def test_vllm_prompt_logprobs_chunks_lm_head_and_normalization() -> None:
    from vllm.v1.sample.sampler import Sampler

    from verl.workers.rollout.vllm_rollout.utils import _compute_prompt_logprobs_in_chunks

    torch.manual_seed(19)
    hidden_states = torch.randn(11, 31, dtype=torch.bfloat16)
    target_ids = torch.randint(0, hidden_states.shape[-1], (hidden_states.shape[0],))
    calls = []

    class Model:
        @staticmethod
        def compute_logits(chunk):
            calls.append(chunk.shape[0])
            return chunk

    sampler = Sampler()
    expected = sampler.gather_logprobs(sampler.compute_logprobs(hidden_states), 4, target_ids)
    actual = _compute_prompt_logprobs_in_chunks(Model(), sampler, hidden_states, 4, target_ids, chunk_size=3)

    assert calls == [3, 3, 3, 2]
    torch.testing.assert_close(actual.logprob_token_ids, expected.logprob_token_ids, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.selected_token_ranks, expected.selected_token_ranks, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.logprobs, expected.logprobs, rtol=0.0, atol=0.0)


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


def test_vllm_worker_exposes_streamopd_connector_transfer_stats(monkeypatch) -> None:
    import vllm.distributed.kv_transfer as kv_transfer

    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    resets = []
    connector = SimpleNamespace(
        reset_transfer_stats=lambda: resets.append(True),
        get_transfer_stats=lambda: {"copy_chunks": 3.0},
        wait_for_all_exports=lambda: 1.25,
    )
    monkeypatch.setattr(kv_transfer, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer, "get_kv_transfer_group", lambda: connector)

    worker = SimpleNamespace()
    assert vLLMColocateWorkerExtension.reset_streamopd_kv_transfer_stats(worker) is True
    assert resets == [True]
    assert vLLMColocateWorkerExtension.get_streamopd_kv_transfer_stats(worker) == {"copy_chunks": 3.0}
    assert vLLMColocateWorkerExtension.wait_for_streamopd_kv_transfers(worker) == 1.25

    monkeypatch.setattr(kv_transfer, "has_kv_transfer_group", lambda: False)
    assert vLLMColocateWorkerExtension.reset_streamopd_kv_transfer_stats(worker) is False
    assert vLLMColocateWorkerExtension.get_streamopd_kv_transfer_stats(worker) == {}
    assert vLLMColocateWorkerExtension.wait_for_streamopd_kv_transfers(worker) == 0.0


def test_rollout_manager_aggregates_streamopd_transfer_stats() -> None:
    from verl.workers.rollout.llm_server import LLMServerManager

    keys = (
        "copy_chunks",
        "copy_bytes",
        "copy_calls",
        "block_runs",
        "staging_wait_seconds",
        "copy_enqueue_seconds",
        "gpu_gather_seconds",
        "gpu_d2h_seconds",
        "gpu_copy_seconds",
        "d2h_wait_seconds",
        "host_commit_seconds",
        "terminal_wait_seconds",
        "max_staging_wait_seconds",
        "max_outstanding_writes",
    )

    class RemoteMethod:
        def __init__(self, value):
            self.value = value

        async def remote(self, method):
            assert method == "get_streamopd_kv_transfer_stats"
            return [self.value]

    manager = LLMServerManager.__new__(LLMServerManager)
    manager.server_handles = [
        SimpleNamespace(collective_rpc=RemoteMethod({key: float(index + 1) for index, key in enumerate(keys)})),
        SimpleNamespace(collective_rpc=RemoteMethod({key: float(2 * (index + 1)) for index, key in enumerate(keys)})),
    ]

    stats = manager.collect_streamopd_kv_transfer_stats()
    for index, key in enumerate(keys[:-2]):
        assert stats[key] == 3.0 * (index + 1)
    assert stats["max_staging_wait_seconds"] == 2.0 * (keys.index("max_staging_wait_seconds") + 1)
    assert stats["max_outstanding_writes"] == 2.0 * (keys.index("max_outstanding_writes") + 1)


def test_rollout_manager_aggregates_profiled_kv_capacity() -> None:
    from verl.workers.rollout.llm_server import LLMServerManager

    class RemoteMethod:
        def __init__(self, value):
            self.value = value

        async def remote(self, method):
            assert method == "get_kv_cache_capacity"
            return self.value

    manager = LLMServerManager.__new__(LLMServerManager)
    manager.server_handles = [
        SimpleNamespace(collective_rpc=RemoteMethod([{"capacity_tokens": 120}, {"capacity_tokens": 120}])),
        SimpleNamespace(collective_rpc=RemoteMethod([{"capacity_tokens": 80}, {"capacity_tokens": 80}])),
    ]

    assert manager.collect_kv_cache_capacity_tokens() == 200


def test_vllm_connector_transfer_stats_accumulate_maxima_and_reset() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._futures = {}
    connector._record_transfer_stats(copy_chunks=2, copy_bytes=16, maxima={"max_outstanding_writes": 3})
    connector._record_transfer_stats(copy_chunks=1, copy_bytes=8, maxima={"max_outstanding_writes": 2})

    stats = connector.get_transfer_stats()
    assert stats["copy_chunks"] == 3
    assert stats["copy_bytes"] == 24
    assert stats["max_outstanding_writes"] == 3

    connector.reset_transfer_stats()
    assert all(value == 0 for value in connector.get_transfer_stats().values())


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


def test_vllm_cross_layer_registration_uses_physical_layout_not_backend_identity(monkeypatch) -> None:
    from verl.experimental.streamopd_kv import vllm_connector

    connector = vllm_connector.StreamOPDKVConnector.__new__(vllm_connector.StreamOPDKVConnector)
    connector._block_size = 4
    connector._chunk_size = 8
    connector._export_strategy = "eos_host"
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(shared_by=["model.layers.1.attn"]),
            SimpleNamespace(shared_by=["model.layers.0.attn"]),
        ]
    )
    connector._staging_token_capacity = connector._chunk_size
    connector._cross_output_buffers = []
    connector._cross_block_id_buffers = []
    connector._cross_block_id_host_buffers = []
    connector._cross_block_id_host_views = []
    initialized = {}
    connector._initialize_host_storage = lambda *args: initialized.setdefault("shape", args)
    monkeypatch.setattr(vllm_connector, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vllm_connector, "get_tensor_model_parallel_world_size", lambda: 1)

    cache = torch.empty(3, 2, 2, 4, 1, 8)
    connector.register_cross_layers_kv_cache(cache, attn_backend=object)

    assert connector._cross_layer_order == (1, 0)
    assert connector._staging_token_capacity == 12
    assert initialized["shape"] == (2, 1, 8, cache.dtype)


def test_vllm_connector_queues_kv_before_eos_and_only_seals_the_tail() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector, _SchedulerSaveState

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._chunk_size = 4
    connector._block_size = 2
    connector._export_strategy = "incremental_triton"
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


def test_vllm_connector_eos_export_queues_the_complete_trajectory() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector, _SchedulerSaveState

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._chunk_size = 4
    connector._block_size = 2
    connector._export_strategy = "eos_host"
    connector._pending = []
    state = _SchedulerSaveState(
        req_id="backend",
        trajectory_id="trajectory",
        slot_path="/tmp/trajectory",
        block_ids_by_group=[[0, 1, 2, 3]],
        policy_version=6,
        prompt_length=2,
    )

    connector._queue_committed(state, 6)
    assert not connector._pending
    assert state.published_tokens == 0

    connector._queue_committed(state, 7, terminal=True, token_ids=torch.arange(7))
    assert [(item.start, item.end, item.terminal) for item in connector._pending] == [
        (0, 4, False),
        (4, 7, True),
    ]
    assert connector._pending[-1].streamed_tokens_before_eos == 0
    assert connector._pending[-1].streamed_chunks_before_eos == 0


def test_vllm_connector_preemption_restarts_export_from_recomputed_prefix() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector, _SchedulerSaveState

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    state = _SchedulerSaveState(
        req_id="backend",
        trajectory_id="trajectory",
        slot_path="/tmp/trajectory",
        block_ids_by_group=[[1, 2]],
        policy_version=6,
        prompt_length=2,
        published_tokens=4,
        next_chunk_index=1,
    )
    connector._scheduler_states = {state.req_id: state}
    connector._scheduler_paths = {state.req_id: state.slot_path}
    connector._pending = []
    cached = SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set())
    scheduler_output = SimpleNamespace(
        preempted_req_ids={state.req_id},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={},
    )

    metadata = connector.build_connector_meta(scheduler_output)

    assert not metadata.pending_saves
    assert state.published_tokens == 0
    assert state.next_chunk_index == 1
    assert connector._scheduler_paths[state.req_id] == state.slot_path


def test_vllm_connector_raw_block_runs_and_host_reorder() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector

    assert StreamOPDKVConnector._block_runs([8, 9, 10, 3, 4, 12]) == [
        (0, 8, 1, 3),
        (3, 3, 1, 2),
        (5, 12, 1, 1),
    ]
    assert StreamOPDKVConnector._block_runs([2, 66, 130, 194]) == [(0, 2, 64, 4)]

    blocks = torch.arange(4 * 3 * 2, dtype=torch.float32).view(4, 3, 2)
    destination = torch.empty(8, 2)
    StreamOPDKVConnector._copy_raw_token_range(
        destination,
        blocks,
        token_offset=1,
        token_count=8,
    )
    torch.testing.assert_close(destination, blocks.flatten(0, 1)[1:9])


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
    assert (long.batch_size, long.token_capacity, long.chunk_size) == (8, 8192, 1024)

    non_power_cap = _fixed_reverse_slot_plan(
        model,
        configured_batch_size=6,
        token_capacity=4096,
        max_batch_tokens=6 * 4096,
        max_chunk_size=1024,
        min_chunk_size=256,
        page_size=64,
        dtype=torch.bfloat16,
        available_memory_bytes=128 * 1024**3,
    )
    assert (non_power_cap.batch_size, non_power_cap.token_capacity, non_power_cap.chunk_size) == (4, 4096, 1024)


def test_fixed_slot_preflight_maximizes_token_tile_without_preferring_singleton_batch() -> None:
    config = SimpleNamespace(
        hidden_size=2560,
        num_hidden_layers=36,
        num_key_value_heads=8,
        num_attention_heads=32,
        head_dim=128,
        vocab_size=151936,
    )
    model = SimpleNamespace(config=config, model=SimpleNamespace(layers=[object()] * 36))

    balanced = _fixed_reverse_slot_plan(
        model,
        configured_batch_size=16,
        token_capacity=8192,
        max_batch_tokens=16 * 8192,
        max_chunk_size=8192,
        min_chunk_size=64,
        page_size=64,
        dtype=torch.bfloat16,
        available_memory_bytes=int(78.5 * 1024**3),
        reserve_bytes=int(30.2 * 1024**3),
    )
    assert (balanced.batch_size, balanced.chunk_size) == (2, 2048)
    assert balanced.prefetch_kv is False

    small_config = SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=28,
        num_key_value_heads=8,
        num_attention_heads=16,
        head_dim=128,
        vocab_size=151936,
    )
    small_model = SimpleNamespace(config=small_config, model=SimpleNamespace(layers=[object()] * 28))
    one_chunk = _fixed_reverse_slot_plan(
        small_model,
        configured_batch_size=16,
        token_capacity=4096,
        max_batch_tokens=16 * 4096,
        max_chunk_size=4096,
        min_chunk_size=64,
        page_size=64,
        dtype=torch.bfloat16,
        available_memory_bytes=int(78.5 * 1024**3),
        reserve_bytes=int(15.3 * 1024**3),
    )
    assert (one_chunk.batch_size, one_chunk.chunk_size) == (2, 4096)
    assert one_chunk.prefetch_kv is True

    memory_fallback = _fixed_reverse_slot_plan(
        small_model,
        configured_batch_size=16,
        token_capacity=4096,
        max_batch_tokens=16 * 4096,
        max_chunk_size=4096,
        min_chunk_size=64,
        page_size=64,
        dtype=torch.bfloat16,
        available_memory_bytes=int(52 * 1024**3),
        reserve_bytes=int(15.3 * 1024**3),
    )
    assert (memory_fallback.batch_size, memory_fallback.chunk_size) == (2, 2048)
    assert memory_fallback.prefetch_kv is False

    singleton_fallback = _fixed_reverse_slot_plan(
        model,
        configured_batch_size=1,
        token_capacity=8192,
        max_batch_tokens=8192,
        max_chunk_size=8192,
        min_chunk_size=64,
        page_size=64,
        dtype=torch.bfloat16,
        available_memory_bytes=int(78.5 * 1024**3),
        reserve_bytes=int(30.2 * 1024**3),
    )
    assert (singleton_fallback.batch_size, singleton_fallback.chunk_size) == (1, 4096)
    assert singleton_fallback.prefetch_kv is False


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
    assert _unsharded_gradient_reserve_bytes(model, data_parallel_size=4) == 3 * parameter_bytes
    with pytest.raises(ValueError, match="data_parallel_size"):
        _unsharded_gradient_reserve_bytes(model, data_parallel_size=0)
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.state[model.weight]["exp_avg"] = torch.zeros_like(model.weight)
    assert _deferred_training_state_bytes(model, optimizer) == 3 * parameter_bytes


def test_trainer_rejects_a_second_gpu_kv_lease() -> None:
    worker = StreamOPDKVTrainingWorker.__new__(StreamOPDKVTrainingWorker)
    worker._gpu_kv_lease_active = True
    with pytest.raises(RuntimeError, match="already holds a GPU KV lease"):
        worker.train_mini_batch(TensorDict({}, batch_size=[]))


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


def test_reverse_wavefront_trims_only_trailing_page_padding() -> None:
    lengths = [7010, 6577, 6100]

    assert _wavefront_compute_end(lengths, [0, 1], 6144, 8192, page_size=64) == 7040
    assert _wavefront_compute_end(lengths, [0, 1, 2], 4096, 6144, page_size=64) == 6144


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
                    "fsdp_config": {"param_offload": True, "optimizer_offload": True},
                },
            },
            "distillation": {
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "streamopd_kv": {
                    "enabled": True,
                    "runtime_profile": "manual",
                    "trainer_placement": "teacher",
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
    assert config.distillation.streamopd_kv.reverse_chunk_size == 4096
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
                "actor": {
                    "fsdp_config": {
                        "param_offload": False,
                        "optimizer_offload": False,
                        "use_no_sync_for_gradient_accumulation": False,
                    }
                },
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
                },
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
        "token_chunk_size": 768,
        "teacher_max_batched_tokens": 4096,
        "vllm_memory_policy": "exclusive_free",
        "teacher_max_num_seqs": 32,
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
    assert config.actor_rollout_ref.actor.fsdp_config.param_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation is False
    assert config.distillation.streamopd_kv.reverse_batch_size == 0
    assert config.distillation.streamopd_kv.reverse_chunk_size == 0
    assert config.distillation.streamopd_kv.teacher_prefill_max_active_kv_tokens == 0
    assert config.distillation.streamopd_kv.kv_handoff_dir.startswith("/dev/shm/verl-streamopd-kv-")
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.3
    assert config.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization == 0.5
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.verl_exclusive_gpu_memory
    assert config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.get(
        "verl_exclusive_gpu_memory"
    )
    assert config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.get(
        "verl_streaming_teacher_logprobs"
    )

    config.distillation.streamopd_kv.trainer_placement = "rollout"
    shared_rollout_plan = _auto_streamopd_runtime_profile(config)
    assert shared_rollout_plan["vllm_memory_policy"] == "exclusive_free"
    assert config.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes == 128

    config.data.train_batch_size = 256
    config.data.max_response_length = 7168
    config.actor_rollout_ref.rollout.n_gpus_per_node = 4
    config.distillation.n_gpus_per_node = 4
    config.distillation.streamopd_kv.trainer_placement = "dedicated"
    long_context_plan = _auto_streamopd_runtime_profile(config)
    assert long_context_plan["trajectory_tokens"] == 8192
    assert long_context_plan["token_chunk_size"] == 1024
    assert config.distillation.streamopd_kv.rollout_kv_export_strategy == "eos_host"
    assert config.distillation.streamopd_kv.rollout_kv_export_chunk_size == 2048
    assert config.distillation.streamopd_kv.rollout_kv_writer_threads == 4
    assert long_context_plan["teacher_max_batched_tokens"] == 8192
    assert long_context_plan["teacher_max_num_seqs"] == 32
    assert long_context_plan["rollout_max_num_seqs"] == 64
    assert long_context_plan["vllm_memory_policy"] == "exclusive_free"


def test_auto_runtime_profile_preserves_explicit_existing_options() -> None:
    config = OmegaConf.create(
        {
            "data": {"train_batch_size": 128, "max_prompt_length": 1024, "max_response_length": 3072},
            "actor_rollout_ref": {
                "actor": {"fsdp_config": {"param_offload": False, "optimizer_offload": False}},
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
                },
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
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
        "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens",
        "distillation.streamopd_kv.token_chunk_size",
    }

    plan = _auto_streamopd_runtime_profile(config, explicit_paths=explicit)

    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.7
    assert config.actor_rollout_ref.rollout.max_num_seqs == 48
    assert config.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes == 256
    assert config.distillation.teacher_models.teacher_model.inference.max_num_batched_tokens == 3072
    assert config.distillation.streamopd_kv.token_chunk_size == 512
    assert plan["teacher_max_batched_tokens"] == 3072
    assert plan["token_chunk_size"] == 512
    assert config.actor_rollout_ref.rollout.max_model_len == 4097
    assert config.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization == 0.3
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.verl_exclusive_gpu_memory
    assert config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.get(
        "verl_exclusive_gpu_memory"
    )


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
                    "fsdp_config": {
                        "param_offload": False,
                        "optimizer_offload": False,
                        "use_no_sync_for_gradient_accumulation": False,
                    },
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
    assert config.distillation.streamopd_kv.token_chunk_size == 768
    assert config.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization == 0.5
    assert config.actor_rollout_ref.rollout.max_num_seqs == 64
    assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "host"
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.5
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.verl_exclusive_gpu_memory
    assert config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.get(
        "verl_exclusive_gpu_memory"
    )
    assert config.actor_rollout_ref.actor.fsdp_config.param_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation is False
    assert config.distillation.teacher_models.teacher_model.inference.max_num_batched_tokens == 4096
    assert config.distillation.streamopd_kv.reverse_chunk_size == 4096
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
    scheduler.begin_policy(7, expected_trajectories=1)
    assert scheduler.try_teacher_session_admitted(7, "teacher-0", 32, 1, 32)
    scheduler.teacher_trajectory_terminal_submitted(7, notifications=2)
    state = scheduler.snapshot()
    assert state["teacher_pending"] == 1
    with pytest.raises(RuntimeError, match="unfinished work"):
        scheduler.end_policy(7)

    score_started = time.perf_counter()
    time.sleep(0.001)
    scheduler.teacher_session_released(7, "teacher-0")
    scheduler.teacher_trajectory_completed(7, [(score_started, time.perf_counter())])
    scheduler.training_started(7)
    with pytest.raises(RuntimeError, match="unfinished work"):
        scheduler.end_policy(7)
    scheduler.training_finished(7)
    metrics = scheduler.end_policy(7)
    assert metrics["streamopd/scheduler_teacher_chunks"] == 1
    assert metrics["streamopd/scheduler_teacher_notifications"] == 2
    assert metrics["streamopd/scheduler_teacher_coalesced_fragments"] == 1
    assert metrics["streamopd/scheduler_training_units"] == 1
    assert metrics["streamopd/scheduler_pool_busy_seconds"] >= 0
    with pytest.raises(RuntimeError, match="no active policy"):
        scheduler.teacher_trajectory_terminal_submitted(8)


def test_streamopd_resource_planners(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.experimental.streamopd_kv.planning import (
        kv_bytes_per_token,
        partition_training_units,
        plan_host_kv,
        plan_teacher_admission,
        plan_training_unit_size,
        planned_reverse_width,
    )

    assert partition_training_units(128, 16) == [16] * 8
    assert partition_training_units(130, 16) == [16] * 8 + [2]
    assert (
        plan_training_unit_size(
            train_batch_size=128,
            reverse_wave_size=16,
            resources_overlap=True,
            kv_prefetch_depth=1,
        )
        == 128
    )
    assert (
        plan_training_unit_size(
            train_batch_size=256,
            reverse_wave_size=16,
            resources_overlap=False,
            kv_prefetch_depth=1,
        )
        == 32
    )
    assert (
        plan_training_unit_size(
            train_batch_size=256,
            reverse_wave_size=16,
            resources_overlap=False,
            kv_prefetch_depth=3,
        )
        == 64
    )
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
    ) == {
        "active_trajectories": 32,
        "active_kv_tokens": 32768,
        "vllm_capacity_tokens": 40000,
        "safe_capacity_tokens": 40000,
        "trajectory_tokens": 1024,
        "teacher_replicas": 1,
        "prefill_wave_per_replica": 5,
        "prefill_wave": 5,
    }
    replicated = plan_teacher_admission(
        expected_trajectories=32,
        trajectory_tokens=1000,
        vllm_capacity_tokens=40000,
        page_size=64,
        max_batched_tokens=2048,
        initial_chunk_tokens=384,
        teacher_replicas=2,
    )
    assert replicated["active_trajectories"] == 32
    assert replicated["prefill_wave_per_replica"] == 5
    assert replicated["prefill_wave"] == 10
    capped = plan_teacher_admission(
        expected_trajectories=32,
        trajectory_tokens=4096,
        vllm_capacity_tokens=20000,
        page_size=64,
        max_batched_tokens=2048,
        initial_chunk_tokens=1024,
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
            token_cap=2048,
        )
    with pytest.raises(ValueError, match="vLLM capacity"):
        plan_teacher_admission(
            expected_trajectories=8,
            trajectory_tokens=4096,
            vllm_capacity_tokens=2048,
            page_size=64,
            max_batched_tokens=2048,
            initial_chunk_tokens=1024,
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
    (model_dir / "config.json").write_text(
        '{"num_hidden_layers": 4, "num_key_value_heads": 2, '
        '"num_attention_heads": 8, "hidden_size": 512, "head_dim": 64, "vocab_size": 100}'
    )
    assert kv_bytes_per_token(str(model_dir), "bfloat16") == 4 * 2 * 64 * 2 * 2


def test_teacher_priority_scheduler_rejects_policy_staleness() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(3)
    with pytest.raises(RuntimeError, match="policy mismatch"):
        scheduler.teacher_trajectory_terminal_submitted(2)


def test_shared_trainer_waits_for_complete_teacher_drain() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(11, expected_trajectories=2, train_launch_width=2)
    scheduler.training_waiting(11, trajectory_count=2)
    assert scheduler.try_teacher_session_admitted(11, "a", 32, 2, 64)
    assert scheduler.try_teacher_session_admitted(11, "b", 32, 2, 64)
    assert scheduler.try_training_started(11) is False
    for session_id in ("a", "b"):
        scheduler.teacher_trajectory_terminal_submitted(11)
        scheduler.teacher_session_released(11, session_id)
        scheduler.teacher_trajectory_completed(11)
    assert scheduler.try_training_started(11) is True
    scheduler.training_finished(11)
    metrics = scheduler.end_policy(11)
    assert metrics["streamopd/scheduler_teacher_completed_at_first_training"] == 2
    assert metrics["streamopd/scheduler_rollouts_terminal_at_first_training"] == 2
    assert metrics["streamopd/scheduler_teacher_pending_at_first_training"] == 0


def test_teacher_admission_waits_for_asynchronous_wake() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(12, teacher_available=False)

    assert scheduler.try_teacher_session_admitted(12, "a", 32, 1, 32) is False
    assert scheduler.snapshot()["teacher_available"] is False
    scheduler.teacher_wake_completed(12)
    assert scheduler.try_teacher_session_admitted(12, "a", 32, 1, 32) is True
    scheduler.teacher_session_released(12, "a")
    metrics = scheduler.end_policy(12)
    assert metrics["streamopd/scheduler_teacher_admission_attempts"] == 2
    assert metrics["streamopd/scheduler_teacher_admission_rejections"] == 1
    assert metrics["streamopd/scheduler_teacher_admission_unavailable_rejections"] == 1
    assert metrics["streamopd/scheduler_teacher_admission_trajectory_rejections"] == 0
    assert metrics["streamopd/scheduler_teacher_admission_kv_rejections"] == 0
    assert metrics["streamopd/scheduler_teacher_admission_waited_sessions"] == 1
    assert metrics["streamopd/scheduler_teacher_admission_wait_seconds"] >= 0


@pytest.mark.asyncio
async def test_teacher_admission_notification_wakes_on_session_release() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(18)

    assert await scheduler.wait_teacher_session_admitted(18, "a", 32, 1, 32)
    pending = asyncio.create_task(scheduler.wait_teacher_session_admitted(18, "b", 32, 1, 32))
    await asyncio.sleep(0)
    assert not pending.done()
    assert scheduler.snapshot()["teacher_admission_waiters"] == 1

    scheduler.teacher_session_released(18, "a")
    assert await pending
    scheduler.teacher_session_released(18, "b")
    metrics = scheduler.end_policy(18)
    assert metrics["streamopd/scheduler_teacher_admission_attempts"] == 2
    assert metrics["streamopd/scheduler_teacher_admission_rejections"] == 1
    assert metrics["streamopd/scheduler_teacher_admission_waited_sessions"] == 1


@pytest.mark.asyncio
async def test_teacher_admission_cancellation_does_not_leak_reservation() -> None:
    scheduler = StreamOPDTaskScheduler()
    scheduler.begin_policy(19)

    assert await scheduler.wait_teacher_session_admitted(19, "a", 32, 1, 32)
    pending = asyncio.create_task(scheduler.wait_teacher_session_admitted(19, "b", 32, 1, 32))
    await asyncio.sleep(0)
    scheduler.teacher_session_admission_cancelled(19, "b")
    with pytest.raises(RuntimeError, match="cancelled"):
        await pending
    scheduler.teacher_session_released(19, "a")
    scheduler.end_policy(19)


def test_dedicated_teacher_and_trainer_resources_can_run_concurrently() -> None:
    scheduler = StreamOPDTaskScheduler(teacher_resources=("teacher",), trainer_resources=("trainer",))
    scheduler.begin_policy(21, expected_trajectories=1)
    assert scheduler.try_teacher_session_admitted(21, "a", 32, 1, 32)
    scheduler.teacher_trajectory_terminal_submitted(21)
    score_started = time.perf_counter()
    scheduler.training_waiting(21, trajectory_count=4)
    assert scheduler.try_training_started(21) is True
    state = scheduler.snapshot()
    assert state["teacher_sessions"] == 1
    assert state["training_active"] == 1
    assert state["resources_overlap"] is False
    time.sleep(0.001)
    score_finished = time.perf_counter()
    scheduler.training_finished(21)
    scheduler.teacher_session_released(21, "a")
    scheduler.teacher_trajectory_completed(21, [(score_started, score_finished)])
    metrics = scheduler.end_policy(21)
    assert metrics["streamopd/scheduler_resources_overlap"] == 0
    assert metrics["streamopd/scheduler_concurrent_busy_seconds"] > 0


def test_dedicated_trainer_starts_when_one_complete_unit_is_ready() -> None:
    scheduler = StreamOPDTaskScheduler(teacher_resources=("teacher",), trainer_resources=("trainer",))
    scheduler.begin_policy(23, expected_trajectories=8, train_launch_width=4)
    for _ in range(8):
        scheduler.teacher_trajectory_terminal_submitted(23)
    for _ in range(4):
        scheduler.teacher_trajectory_completed(23)
    scheduler.training_waiting(23, trajectory_count=4)
    assert scheduler.snapshot()["teacher_drained"] is False
    assert scheduler.try_training_started(23) is True
    scheduler.training_finished(23)
    for _ in range(4):
        scheduler.teacher_trajectory_completed(23)
    scheduler.training_waiting(23, trajectory_count=4)
    assert scheduler.try_training_started(23) is True
    scheduler.training_finished(23)
    scheduler.end_policy(23)


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
    metrics = scheduler.end_policy(13)
    assert metrics["streamopd/scheduler_teacher_admission_attempts"] == 4
    assert metrics["streamopd/scheduler_teacher_admission_rejections"] == 1
    assert metrics["streamopd/scheduler_teacher_admission_trajectory_rejections"] == 0
    assert metrics["streamopd/scheduler_teacher_admission_kv_rejections"] == 1


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


def test_streamopd_default_trainer_placement_uses_teacher_and_rollout_pools() -> None:
    assert StreamOPDKVConfig().trainer_placement == "union"


def test_sync_stage_timing_aggregates_rollout_teacher_and_training() -> None:
    from verl.trainer.ppo.v1.trainer_sync import _stage_timing_metrics_from_tags

    metrics = _stage_timing_metrics_from_tags(
        [
            {
                "_stage_rollout_started_at": 101.0,
                "_stage_rollout_completed_at": 105.0,
                "_stage_rollout_request_seconds": 4.0,
                "_stage_teacher_started_at": 105.0,
                "_stage_teacher_completed_at": 108.0,
                "_stage_teacher_request_seconds": 3.0,
            },
            {
                "_stage_rollout_started_at": 102.0,
                "_stage_rollout_completed_at": 107.0,
                "_stage_rollout_request_seconds": 5.0,
                "_stage_teacher_started_at": 107.0,
                "_stage_teacher_completed_at": 109.0,
                "_stage_teacher_request_seconds": 2.0,
            },
            {"is_padding": True},
        ],
        policy_started_at=100.0,
        training_seconds=6.5,
    )

    assert metrics == {
        "stage/training_seconds": 6.5,
        "stage/rollout_span_seconds": 6.0,
        "stage/rollout_makespan_seconds": 7.0,
        "stage/rollout_request_seconds/mean": 4.5,
        "stage/rollout_request_seconds/max": 5.0,
        "stage/teacher_span_seconds": 4.0,
        "stage/teacher_makespan_seconds": 9.0,
        "stage/teacher_request_seconds/mean": 2.5,
        "stage/teacher_request_seconds/max": 3.0,
        "stage/teacher_tail_seconds": 2.0,
    }


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


@pytest.mark.parametrize(
    ("placement", "expected_pools", "teacher_pool"),
    [
        ("teacher", {"global_pool": [2], "rollout_pool": [3]}, "global_pool"),
        ("rollout", {"global_pool": [2], "teacher_pool": [1]}, "teacher_pool"),
        ("union", {"global_pool": [2]}, "global_pool"),
        (
            "dedicated",
            {"global_pool": [2], "teacher_pool": [1], "rollout_pool": [3]},
            "teacher_pool",
        ),
    ],
)
def test_streamopd_resource_pools_follow_trainer_placement(
    monkeypatch: pytest.MonkeyPatch,
    placement: str,
    expected_pools: dict[str, list[int]],
    teacher_pool: str,
) -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.utils import Role
    from verl.trainer.ppo.v1 import trainer_streamopd
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    def fake_base_init(self) -> None:
        self.role_worker_mapping = {Role.ActorRollout: object(), Role.TeacherModel: object()}
        self.mapping = {Role.ActorRollout: "global_pool", Role.TeacherModel: "teacher_pool"}
        self.resource_pool_manager = SimpleNamespace(resource_pool_spec={"global_pool": [2], "teacher_pool": [1]})

    monkeypatch.setattr(PPOTrainer, "_init_resource_pool_mgr", fake_base_init)
    monkeypatch.setattr(trainer_streamopd.ray, "remote", lambda cls: cls)
    monkeypatch.setattr(trainer_streamopd, "need_reference_policy", lambda _config: False)
    trainer = trainer_streamopd.PPOTrainerStreamOPD.__new__(trainer_streamopd.PPOTrainerStreamOPD)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n_gpus_per_node": 3, "nnodes": 1}},
        }
    )
    trainer.placement = TrainerPlacement(placement)

    trainer._init_resource_pool_mgr()

    assert trainer.resource_pool_manager.resource_pool_spec == expected_pools
    assert trainer.mapping[Role.Actor] == "global_pool"
    assert trainer.mapping[Role.TeacherModel] == teacher_pool


def test_auto_shared_teacher_runtime_defers_reverse_plan_until_pool_sleep() -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1 import trainer_streamopd

    trainer = trainer_streamopd.PPOTrainerStreamOPD.__new__(trainer_streamopd.PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement.TEACHER
    trainer._reverse_plan_result = None
    reverse_plans = []

    def prepare_reverse_plan():
        reverse_plans.append(True)
        return [{"slot_batch_size": 8.0}]

    trainer.actor_rollout_wg = SimpleNamespace(
        get_streamopd_device_memory_stats=lambda: [
            {"free_bytes": 79 * 1024**3, "total_bytes": 80 * 1024**3},
            {"free_bytes": 79 * 1024**3, "total_bytes": 80 * 1024**3},
        ],
        prepare_streamopd_reverse_plan=prepare_reverse_plan,
    )
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 128},
            "distillation": {
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "streamopd_kv": {
                    "runtime_profile": "auto",
                    "planner_explicit_options": [],
                    "reverse_slot_max_tokens": 8192,
                    "reverse_batch_max_tokens": 1048576,
                    "reverse_slot_reserve_gib": 4.0,
                },
                "teacher_models": {
                    "teacher_model": {
                        "model_path": "/teacher",
                        "inference": {
                            "tensor_model_parallel_size": 2,
                            "data_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_num_seqs": 128,
                            "max_num_batched_tokens": 8192,
                            "max_model_len": 8193,
                            "dtype": "bfloat16",
                            "gpu_memory_utilization": 0.9,
                        },
                    }
                },
            },
        }
    )

    trainer._prepare_teacher_runtime()

    inference = trainer.config.distillation.teacher_models.teacher_model.inference
    assert inference.gpu_memory_utilization == 0.9
    assert inference.max_num_seqs == 128
    assert trainer._teacher_memory_plan == {"max_num_seqs": 128.0, "exclusive_pool_memory": 1.0}
    assert trainer._reverse_plan_result is None
    assert reverse_plans == []


@pytest.mark.parametrize("placement", ["teacher", "rollout", "union"])
def test_shared_trainer_state_has_one_load_offload_pair(placement: str) -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    transitions = []
    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement(placement)
    trainer._trainer_state_offloaded = True
    trainer._reverse_plan_result = [{"slot_batch_size": 8.0}]
    trainer._teacher_sleeping = trainer.placement.shares_teacher
    trainer._shared_rollout_sleeping = trainer.placement.shares_rollout
    trainer.actor_rollout_wg = SimpleNamespace(
        offload_streamopd_trainer_state=lambda: transitions.append("offload"),
        load_streamopd_trainer_state=lambda: transitions.append("load"),
    )

    assert trainer._load_trainer_state() >= 0.0
    assert trainer._load_trainer_state() == 0.0
    assert trainer._offload_trainer_state() >= 0.0
    assert trainer._offload_trainer_state() == 0.0
    assert transitions == ["load", "offload"]


def test_shared_trainer_plans_against_sleeping_pool_before_loading() -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    transitions = []
    plan_result = [{"slot_batch_size": 2.0}]
    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement.UNION
    trainer._trainer_state_offloaded = True
    trainer._teacher_sleeping = True
    trainer._shared_rollout_sleeping = True
    trainer._reverse_plan_result = None
    trainer.actor_rollout_wg = SimpleNamespace(
        prepare_streamopd_reverse_plan=lambda: transitions.append("plan") or plan_result,
        load_streamopd_trainer_state=lambda: transitions.append("load"),
    )

    def configure_reverse_plan(result) -> None:
        assert result is plan_result
        transitions.append("configure")
        trainer._reverse_plan_result = result

    trainer._configure_reverse_plan = configure_reverse_plan

    assert trainer._load_trainer_state() >= 0.0
    assert transitions == ["plan", "configure", "load"]


def test_shared_trainer_cannot_load_before_inference_pool_sleeps() -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement.UNION
    trainer._trainer_state_offloaded = True
    trainer._teacher_sleeping = True
    trainer._shared_rollout_sleeping = False
    trainer.actor_rollout_wg = SimpleNamespace(load_streamopd_trainer_state=lambda: None)

    with pytest.raises(RuntimeError, match="shared Rollout"):
        trainer._load_trainer_state()


@pytest.mark.parametrize(
    ("placement", "expected"),
    [("union", ["sleep", "sync", "cleanup"]), ("dedicated", ["sync", "cleanup"])],
)
def test_initial_weight_sync_releases_shared_teacher_first(placement: str, expected: list[str]) -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    transitions = []
    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement(placement)
    trainer.global_steps = 0
    trainer._maybe_sleep_teacher = lambda _state: transitions.append("sleep")
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda _step, **_kwargs: transitions.append("sync"))
    trainer.actor_rollout_wg = SimpleNamespace(release_streamopd_allocator_cache=lambda: transitions.append("cleanup"))

    trainer._publish_initial_weights()

    assert transitions == expected


def test_phase_exclusive_host_weight_sync_serializes_trainer_and_rollout(monkeypatch) -> None:
    from verl.checkpoint_engine import base as checkpoint_base

    events = []

    class Replica:
        workers = [object()]

        async def sleep(self, level=None):
            events.append(("rollout-sleep", level))

        async def release_kv_cache(self):
            events.append("weights-wake")

        async def resume_kv_cache(self):
            events.append("kv-wake")

        async def abort_all_requests(self):
            events.append("abort")

        async def resume_generation(self):
            events.append("resume-generation")

    class ActorGroup:
        world_size = 1

        def update_weights(self, **kwargs):
            events.append(("trainer-publish", kwargs["mode"]))
            return [{"sender_metric": 1.0}]

        def execute_checkpoint_engine(self, methods):
            assert methods == ["finalize"]
            events.append("trainer-finalize")
            return [None]

        def release_streamopd_allocator_cache(self):
            events.append("trainer-release")

    class RolloutGroup:
        world_size = 1

        def __init__(self, **_kwargs):
            pass

        def update_weights(self, **_kwargs):
            events.append("rollout-receive")
            return [None]

        def execute_checkpoint_engine(self, methods):
            assert methods == ["finalize"]
            events.append("rollout-finalize")
            return [None]

    monkeypatch.setattr(checkpoint_base, "RayWorkerGroup", RolloutGroup)
    monkeypatch.setattr(checkpoint_base.ray, "get", lambda values: values)
    manager = checkpoint_base.CheckpointEngineManager.__new__(checkpoint_base.CheckpointEngineManager)
    manager.backend = "host"
    manager.actor_wg = ActorGroup()
    manager.replicas = [Replica()]
    manager.build_process_group = lambda _rollout: events.append("build")

    metrics = asyncio.run(
        checkpoint_base.CheckpointEngineManager.update_weights.__wrapped__(
            manager,
            global_steps=3,
            phase_exclusive=True,
        )
    )

    assert events == [
        ("rollout-sleep", 2),
        "build",
        ("trainer-publish", "host"),
        "trainer-release",
        "weights-wake",
        "rollout-receive",
        "trainer-finalize",
        "rollout-finalize",
        "kv-wake",
    ]
    assert metrics["sender_metric"] == 1.0
    assert "abort" not in events
    assert "resume-generation" not in events


def test_shared_rollout_sleeps_once_after_all_trajectories_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1 import trainer_streamopd

    class Snapshot:
        @staticmethod
        def remote():
            return object()

    sleep_levels = []
    trainer = trainer_streamopd.PPOTrainerStreamOPD.__new__(trainer_streamopd.PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement.ROLLOUT
    trainer._shared_rollout_sleeping = False
    trainer.config = OmegaConf.create(
        {"distillation": {"streamopd_kv": {"scheduler_poll_interval_ms": 1, "scheduler_timeout_seconds": 1}}}
    )
    trainer._scheduler = SimpleNamespace(snapshot=Snapshot())
    transfer_drains = []
    trainer.llm_server_manager = SimpleNamespace(wait_for_streamopd_kv_transfers=lambda: transfer_drains.append(True))
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda *, level: sleep_levels.append(level))
    monkeypatch.setattr(
        trainer_streamopd.ray,
        "get",
        lambda _value: {"terminal_trajectories": 128, "expected_trajectories": 128},
    )

    trainer._wait_for_shared_rollout_idle()
    trainer._wait_for_shared_rollout_idle()

    assert sleep_levels == [2]
    assert transfer_drains == [True]
    assert trainer._shared_rollout_sleeping


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


def test_vllm_connector_pipelines_terminal_exports_across_trajectories() -> None:
    from concurrent.futures import Future, ThreadPoolExecutor

    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector, _PendingSave

    sealed = []

    class Pool:
        @staticmethod
        def validate_writer(*args, **kwargs):
            return 0

        @staticmethod
        def seal(*args, **kwargs):
            sealed.append(kwargs["request_id"])

    pool = Pool()
    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._executor = ThreadPoolExecutor(max_workers=2)
    connector._get_host_pool = lambda: pool
    blockers: dict[str, Future] = {}
    launched = []

    def launch(pending, _pool, _slot, request_futures, *, ready_event=None):
        del ready_event
        launched.append(pending.req_id)
        blocker = Future()
        blockers[pending.req_id] = blocker
        request_futures.append(blocker)

    connector._launch_host_slot_chunk = launch

    def terminal(req_id: str) -> _PendingSave:
        return _PendingSave(
            req_id=req_id,
            trajectory_id=req_id,
            slot_path=f"/tmp/{req_id}",
            block_ids_by_group=([0],),
            policy_version=1,
            prompt_length=1,
            start=0,
            end=1,
            chunk_index=0,
            terminal=True,
            token_ids=torch.tensor([1]),
        )

    first_seal = connector._export_terminal_saves([terminal("first")], ready_event=object())
    second_seal = connector._export_terminal_saves([terminal("second")], ready_event=object())
    assert launched == ["first", "second"]
    assert not first_seal.done()
    assert not second_seal.done()

    blockers["first"].set_result(None)
    blockers["second"].set_result(None)
    first_seal.result(timeout=1)
    second_seal.result(timeout=1)
    assert sorted(sealed) == ["first", "second"]
    connector._executor.shutdown(wait=True)


def test_vllm_connector_completion_tracks_inner_seal_future() -> None:
    from concurrent.futures import Future

    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector

    submission = Future()
    seal = Future()
    completion = Future()
    submission.set_result(seal)
    StreamOPDKVConnector._terminal_export_submitted(completion, submission)
    assert not completion.done()

    seal.set_result(None)
    assert completion.result(timeout=1) is None

    failed_submission = Future()
    failed_completion = Future()
    failed_submission.set_exception(RuntimeError("submit failed"))
    StreamOPDKVConnector._terminal_export_submitted(failed_completion, failed_submission)
    with pytest.raises(RuntimeError, match="submit failed"):
        failed_completion.result(timeout=1)


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
