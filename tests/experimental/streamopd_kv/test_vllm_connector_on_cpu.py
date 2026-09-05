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
    connector._export_strategy = "eos_host"
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


def test_eos_host_rejects_unbudgeted_gpu_gather_fallback():
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._export_strategy = "eos_host"
    with pytest.raises(RuntimeError, match="select eos_triton explicitly"):
        connector.register_kv_caches({})
    assert connector._export_strategy == "eos_host"


@pytest.mark.parametrize("legacy", [False, True])
def test_preemption_waits_for_prior_exports_with_both_vllm_contracts(legacy):
    from concurrent.futures import Future

    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector, StreamOPDKVConnectorMetadata

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    completed = Future()
    completed.set_result(None)
    failed = Future()
    failed.set_exception(RuntimeError("export failed before page reuse"))
    connector._request_futures = {"done": [completed], "failed": [failed]}
    payload = {"done"} if legacy else StreamOPDKVConnectorMetadata(preempted_req_ids={"done"})
    connector.handle_preemptions(payload)
    payload = {"failed"} if legacy else StreamOPDKVConnectorMetadata(preempted_req_ids={"failed"})
    with pytest.raises(RuntimeError, match="before page reuse"):
        connector.handle_preemptions(payload)
