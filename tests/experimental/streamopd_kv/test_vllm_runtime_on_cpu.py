# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verl.experimental.streamopd_kv.vllm_patch import (
    _configure_exclusive_gpu_memory,
    _determine_exclusive_available_memory,
    _prompt_logprob_chunk_rows,
    _request_exclusive_gpu_memory,
    _streamopd_unprofiled_workspace_bytes,
)


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

    from verl.experimental.streamopd_kv.vllm_patch import _gather_prompt_logprobs_in_chunks

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

    from verl.experimental.streamopd_kv.vllm_patch import _compute_prompt_logprobs_in_chunks

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


def test_rollout_manager_aggregates_vllm_worker_memory_stats() -> None:
    from verl.experimental.streamopd_kv.replica_group import VLLMReplicaGroup

    class RemoteMethod:
        def __init__(self, stats):
            self.stats = stats

        async def remote(self, method):
            assert method == "get_device_memory_stats"
            return self.stats

    class Server:
        def __init__(self, stats):
            self.collective_rpc = RemoteMethod(stats)

    manager = VLLMReplicaGroup([])
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


def test_rollout_manager_aggregates_streamopd_transfer_stats() -> None:
    from verl.experimental.streamopd_kv.replica_group import VLLMReplicaGroup

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

    manager = VLLMReplicaGroup([])
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
    from verl.experimental.streamopd_kv.replica_group import VLLMReplicaGroup

    class RemoteMethod:
        def __init__(self, value):
            self.value = value

        async def remote(self, method):
            assert method == "get_kv_cache_capacity"
            return self.value

    manager = VLLMReplicaGroup([])
    manager.server_handles = [
        SimpleNamespace(collective_rpc=RemoteMethod([{"capacity_tokens": 120}, {"capacity_tokens": 120}])),
        SimpleNamespace(collective_rpc=RemoteMethod([{"capacity_tokens": 80}, {"capacity_tokens": 80}])),
    ]

    assert manager.collect_kv_cache_capacity_tokens() == 200


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


def test_exclusive_memory_patch_preserves_unmarked_workers(monkeypatch):
    from vllm.v1.worker import gpu_worker

    from verl.experimental.streamopd_kv.vllm_patch import _configure_exclusive_gpu_memory

    monkeypatch.setattr(gpu_worker, "request_memory", lambda snapshot, cache: 17)
    monkeypatch.setattr(gpu_worker.Worker, "determine_available_memory", lambda worker: 23)
    config = SimpleNamespace(additional_config={"verl_exclusive_gpu_memory": True}, cache_config=SimpleNamespace())
    _configure_exclusive_gpu_memory(config)
    request_memory = gpu_worker.request_memory
    _configure_exclusive_gpu_memory(config)
    assert gpu_worker.request_memory is request_memory
    snapshot = SimpleNamespace(free_memory=100, total_memory=200)
    assert request_memory(snapshot, config.cache_config) == 100
    assert request_memory(snapshot, SimpleNamespace()) == 17
    baseline = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))
    assert gpu_worker.Worker.determine_available_memory(baseline) == 23


def test_exclusive_gpu_memory_uses_post_nccl_free_bytes_without_deriving_fraction() -> None:
    cache_config = SimpleNamespace(gpu_memory_utilization=0.25)
    requested = _request_exclusive_gpu_memory(
        SimpleNamespace(free_memory=75 * 1024**3, total_memory=80 * 1024**3),
        cache_config,
    )
    assert requested == 75 * 1024**3
    assert cache_config.gpu_memory_utilization == 0.25


def test_exclusive_gpu_memory_marker_installs_worker_request(monkeypatch) -> None:
    from vllm.v1.worker import gpu_worker

    original = gpu_worker.request_memory
    original_determine = gpu_worker.Worker.determine_available_memory
    monkeypatch.setattr(gpu_worker, "request_memory", original)
    monkeypatch.setattr(gpu_worker.Worker, "determine_available_memory", original_determine)
    assert not _configure_exclusive_gpu_memory(SimpleNamespace(additional_config={}))
    assert gpu_worker.request_memory is original
    assert _configure_exclusive_gpu_memory(
        SimpleNamespace(additional_config={"verl_exclusive_gpu_memory": True}, cache_config=SimpleNamespace())
    )
    assert gpu_worker.request_memory._verl_exclusive_gpu_memory
    assert gpu_worker.Worker.determine_available_memory._verl_exclusive_gpu_memory


def test_exclusive_gpu_memory_reserves_measured_runtime_and_graph_peaks() -> None:
    mib = 1024**2
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        model_runner=SimpleNamespace(
            cudagraph_dispatcher=SimpleNamespace(get_capture_descs=lambda: [("piecewise", [1, 2]), ("full", [1])])
        ),
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        cache_config=SimpleNamespace(),
        peak_activation_memory=12 * mib,
    )

    assert _determine_exclusive_available_memory(worker, lambda _worker: 256 * mib) == 70 * mib
    assert worker.cache_config._verl_exclusive_activation_reserve_bytes == 36 * mib
    assert worker.cache_config._verl_exclusive_activation_reserve_count == 3
    assert worker.cache_config._verl_exclusive_graph_pool_count == 2
    assert worker.cache_config._verl_exclusive_connector_reserve_bytes == 0
    assert worker.cache_config._verl_exclusive_redundancy_reserve_bytes == 150 * mib


def test_exclusive_gpu_memory_accounts_for_native_sampler_workspace() -> None:
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_vocab_size=lambda: 151936,
    )
    vllm_config = SimpleNamespace(
        additional_config={"verl_exclusive_gpu_memory": True},
        kv_transfer_config=None,
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=8192),
    )
    segment = 2 * 1024**2

    def rounded(size: int) -> int:
        return (size + segment - 1) // segment * segment

    rows = 64
    vocab = 151936
    expected = (
        rounded(rows * vocab * 2)
        + 5 * rounded(rows * vocab * 4)
        + rounded(rows * vocab * 8)
        + 2 * rounded(rows * vocab)
        + segment
    )
    assert _streamopd_unprofiled_workspace_bytes(vllm_config) == expected


def test_streaming_teacher_workspace_accounts_for_chunked_tp_logits_and_normalization() -> None:
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        max_model_len=4096,
        get_vocab_size=lambda: 151936,
    )
    vllm_config = SimpleNamespace(
        additional_config={"verl_streaming_teacher_logprobs": True},
        kv_transfer_config=None,
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )
    segment = 2 * 1024**2

    def rounded(size: int) -> int:
        return (size + segment - 1) // segment * segment

    expected = rounded(1024 * 151936 * 2) + rounded(1024 * (151936 // 2) * 2) + 2 * rounded(1024 * 151936 * 4) + segment
    assert _streamopd_unprofiled_workspace_bytes(vllm_config) == expected


def test_streaming_teacher_normalization_rows_follow_runtime_free_memory() -> None:
    vocab_size = 151936
    bytes_per_row = vocab_size * 5
    free_bytes = 2 * 2 * 1024**2 + 300 * bytes_per_row

    assert _prompt_logprob_chunk_rows(free_bytes=free_bytes, vocab_size=vocab_size, requested_rows=512) == 300
    assert _prompt_logprob_chunk_rows(free_bytes=free_bytes * 4, vocab_size=vocab_size, requested_rows=512) == 512


@pytest.mark.parametrize("export_strategy", ["eos_triton", "incremental_triton"])
def test_streamopd_gpu_gather_workspace_follows_export_strategy(export_strategy: str) -> None:
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_num_layers=lambda _parallel: 12,
        get_num_kv_heads=lambda _parallel: 4,
        get_head_size=lambda: 64,
    )
    extra = {
        "streamopd_kv_export_strategy": export_strategy,
        "streamopd_writer_threads": 3,
        "streamopd_kv_chunk_size": 256,
    }
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(
            kv_connector="StreamOPDKVConnector",
            kv_connector_extra_config=extra,
        ),
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    segment = 2 * 1024**2
    output_bytes = 12 * 256 * 2 * 4 * 64 * 2
    rounded_output_bytes = (output_bytes + segment - 1) // segment * segment

    assert _streamopd_unprofiled_workspace_bytes(vllm_config) == 3 * rounded_output_bytes + segment


def test_streamopd_eos_host_reserves_no_gpu_gather_workspace() -> None:
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(
            kv_connector="StreamOPDKVConnector",
            kv_connector_extra_config={
                "streamopd_kv_export_strategy": "eos_host",
                "streamopd_writer_threads": 64,
                "streamopd_kv_chunk_size": 8192,
            },
        ),
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    assert _streamopd_unprofiled_workspace_bytes(vllm_config) == 0


@pytest.mark.parametrize("available", [300 * 1024**2, 8 * 1024**3])
@pytest.mark.parametrize("reservation_tokens", [0, 256])
def test_exclusive_kv_budget_never_exceeds_whole_policy_demand(available, reservation_tokens):
    from verl.experimental.streamopd_kv.vllm_patch import _determine_exclusive_available_memory

    model = SimpleNamespace(
        enforce_eager=True,
        max_model_len=97,
        dtype=torch.bfloat16,
        get_num_layers=lambda parallel: 28,
        get_num_kv_heads=lambda parallel: 8,
        get_head_size=lambda: 128,
    )
    config = SimpleNamespace(
        model_config=model,
        parallel_config=SimpleNamespace(),
        kv_transfer_config=None,
        additional_config={
            "verl_streamopd_max_live_trajectories": 128,
            "verl_streamopd_reserved_trajectory_tokens": reservation_tokens,
        },
    )
    worker = SimpleNamespace(
        model_config=model,
        vllm_config=config,
        peak_activation_memory=0,
        cache_config=SimpleNamespace(block_size=64),
    )
    budget = _determine_exclusive_available_memory(worker, lambda _: available)
    # All 128 trajectories can route to this worker; never cap by max_num_seqs
    # or by an assumed even split across replicas.
    demand = 128 * max(128, reservation_tokens) * 28 * 8 * 128 * 2 * 2
    assert budget == min(available - 150 * 1024**2, demand)


def test_teacher_lifecycle_preserves_no_argument_replica_sleep():
    from verl.experimental.teacher_loop.teacher_model import MultiTeacherModelManager

    events = []

    class Replica:
        async def sleep(self):
            events.append("sleep")

        async def wake_up(self):
            events.append("wake")

    manager = MultiTeacherModelManager.__new__(MultiTeacherModelManager)
    manager.teacher_model_managers = {"teacher": SimpleNamespace(rollout_replicas=[Replica()])}
    manager.sleep()
    manager.wake_up()
    assert events == ["sleep", "wake"]
