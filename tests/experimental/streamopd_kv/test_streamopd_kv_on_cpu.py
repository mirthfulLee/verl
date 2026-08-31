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
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import save_file
from tensordict import TensorDict
from torch import nn

from verl.experimental.streamopd_kv import (
    BatchedReverseChunkState,
    CommittedChunkPublisher,
    CommittedTokenChunk,
    KVLayout,
    KVSnapshotStore,
    LayerKVTrace,
    PolicyVersionBarrier,
    Qwen3ReverseTrainer,
    ReverseChunkState,
    SealedKVSnapshot,
    SnapshotState,
    StreamingTeacherCoordinator,
    TeacherArtifactBuffer,
    TrajectoryKey,
    exact_causal_attention,
    extract_vllm_nhd_token_range,
    extract_vllm_nhd_tokens,
    load_vllm_snapshot,
    move_vllm_snapshot,
    prepare_streamopd_kv_config,
)
from verl.experimental.streamopd_kv.fsdp_worker import (
    StreamOPDKVTrainingWorker,
    _dynamic_reverse_chunk_size,
    _forward_kl_topk_sum,
    _has_valid_response,
    _memory_limited_reverse_batch_size,
    _partition_reverse_microbatches,
    _reverse_backward_calls,
)
from verl.experimental.streamopd_kv.qwen3 import _build_reverse_wavefront
from verl.experimental.streamopd_kv.scheduler import StreamOPDTaskScheduler
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.workers.config.distillation import DistillationConfig, StreamOPDKVConfig


class _ToyBlock(nn.Module):
    def __init__(self, hidden_size: int, query_heads: int, kv_heads: int) -> None:
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = hidden_size // query_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        layer_idx: int,
        reverse_state: ReverseChunkState | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, tokens, _ = hidden.shape
        normed = self.norm(hidden)
        query = self.q_proj(normed).view(batch, tokens, self.query_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(normed).view(batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(normed).view(batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)
        if reverse_state is None:
            attention = exact_causal_attention(query, key, value, query_start=0)
        else:
            attention = reverse_state.attention(layer_idx, query, key, value)
        attention = attention.transpose(1, 2).reshape(batch, tokens, -1)
        hidden = hidden + self.o_proj(attention)
        hidden = hidden + self.ff(self.ff_norm(hidden))
        return hidden, (key, value)


class _ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(31, 16)
        self.layers = nn.ModuleList([_ToyBlock(16, 4, 2), _ToyBlock(16, 4, 2)])
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 31, bias=False)

    def forward(
        self, input_ids: torch.Tensor, reverse_state: ReverseChunkState | None = None
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden = self.embedding(input_ids)
        cache = []
        for layer_idx, layer in enumerate(self.layers):
            hidden, kv = layer(hidden, layer_idx=layer_idx, reverse_state=reverse_state)
            cache.append(kv)
        return self.head(self.norm(hidden)), tuple(cache)


@pytest.mark.asyncio
async def test_committed_chunk_publisher_emits_only_accepted_contiguous_tokens() -> None:
    emitted = []

    async def submit(chunk) -> None:
        emitted.append(chunk)

    publisher = CommittedChunkPublisher(TrajectoryKey(4, "req"), [1, 2], chunk_size=2, submit=submit)
    await publisher.observe([10])
    await publisher.observe([10, 11, 12])
    await publisher.observe([10, 11, 12, 13], terminal=True)

    assert [(chunk.start, chunk.token_ids, chunk.terminal) for chunk in emitted] == [
        (0, (10, 11, 12), False),
        (3, (13,), True),
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
        (0, 2, False),
        (2, 4, False),
        (4, 6, False),
    ]


@pytest.mark.asyncio
async def test_committed_chunk_publisher_sends_longest_page_aligned_progress() -> None:
    emitted = []

    async def submit(chunk) -> None:
        emitted.append(chunk)

    publisher = CommittedChunkPublisher(
        TrajectoryKey(4, "page-aligned"),
        [1],
        chunk_size=2,
        page_size=4,
        submit=submit,
    )
    await publisher.observe(list(range(7)))
    await publisher.observe(list(range(10)))
    await publisher.observe(list(range(11)), terminal=True)

    assert [(chunk.start, chunk.end, chunk.terminal) for chunk in emitted] == [
        (0, 4, False),
        (4, 8, False),
        (8, 11, True),
    ]


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


@pytest.mark.asyncio
async def test_committed_publisher_can_emit_one_early_chunk_then_terminal_catch_up() -> None:
    emitted = []

    async def submit(chunk: CommittedTokenChunk) -> None:
        emitted.append(chunk)

    publisher = CommittedChunkPublisher(
        TrajectoryKey(5, "terminal-catch-up"),
        [1, 2],
        2,
        submit,
        terminal_only_after_initial=True,
    )
    await publisher.observe([10, 11])
    await publisher.observe([10, 11, 12, 13])
    await publisher.observe([10, 11, 12, 13, 14], terminal=True)

    assert [(chunk.start, chunk.token_ids, chunk.terminal) for chunk in emitted] == [
        (0, (10, 11), False),
        (2, (12, 13, 14), True),
    ]


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


def test_snapshot_lifecycle_teacher_coverage_and_version_barrier() -> None:
    key = TrajectoryKey(3, "trajectory")
    layout = KVLayout(num_layers=1, num_kv_heads=2, head_dim=4, dtype="float32", page_size=16)
    snapshot = SealedKVSnapshot(
        key=key,
        token_ids=(1, 2, 3),
        prompt_length=2,
        layout=layout,
        layers=((torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),),
    )
    store = KVSnapshotStore()
    store.seal(snapshot)
    lease = store.acquire(key)
    assert lease.state is SnapshotState.LEASED
    with pytest.raises(RuntimeError, match="leased"):
        store.invalidate_version(3)
    lease.release()
    assert lease.state is SnapshotState.RELEASED
    assert store.invalidate_version(3) == 1

    artifacts = TeacherArtifactBuffer()
    artifacts.append(key, 0, torch.tensor([[1], [2]]), torch.tensor([[-1.0], [-2.0]]))
    artifacts.append(key, 2, torch.tensor([[3]]), torch.tensor([[-3.0]]), terminal=True)
    assert artifacts.is_complete(key, 3)
    ids, logprobs = artifacts.consume_reverse(key, 1, 3)
    torch.testing.assert_close(ids[:, 0], torch.tensor([2, 3]))
    torch.testing.assert_close(logprobs[:, 0], torch.tensor([-2.0, -3.0]))

    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([6.0])
    barrier = PolicyVersionBarrier(3, ["a", "b"], valid_token_count=3)
    barrier.mark_backward_complete(TrajectoryKey(3, "a"))
    with pytest.raises(RuntimeError, match="not ready"):
        barrier.step([parameter], lambda: None)
    barrier.mark_backward_complete(TrajectoryKey(3, "b"))
    barrier.step([parameter], lambda: parameter.data.add_(-parameter.grad))
    torch.testing.assert_close(parameter, torch.tensor([-1.0]))


def test_vllm_snapshot_loader_validates_identity_and_restores_layout(tmp_path) -> None:
    base_path = str(tmp_path / "sealed")
    filename = base_path + ".tp0.safetensors"
    packed = torch.arange(3 * 2 * 2 * 4, dtype=torch.float32).reshape(3, 2, 2, 4)
    save_file(
        {"token_ids": torch.tensor([4, 5, 6]), "layer_00000": packed},
        filename,
        metadata={
            "format": "verl-streamopd-kv-v1",
            "request_id": "request-backend-suffix",
            "trajectory_id": "request",
            "policy_version": "2",
            "prompt_length": "2",
            "tp_rank": "0",
            "tp_size": "1",
            "page_size": "16",
            "axis_order": "token_kv_head_dim",
            "rope_convention": "post_rope_key",
            "layer_names": '["model.layers.0.self_attn"]',
        },
    )
    (tmp_path / "sealed.tp0.safetensors.lock").touch()
    snapshot = load_vllm_snapshot(
        base_path,
        key=TrajectoryKey(2, "request"),
        tp_rank=0,
        expected_tp_size=1,
        expected_token_ids=[4, 5, 6],
        expected_prompt_length=2,
        pin_memory=True,
    )
    assert snapshot.layout.num_layers == 1
    assert snapshot.layout.num_kv_heads == 2
    assert snapshot.layers[0][0].shape == (1, 2, 3, 4)
    torch.testing.assert_close(snapshot.layers[0][0][0, :, 1], packed[1, 0])
    with pytest.raises(RuntimeError, match="token identity"):
        load_vllm_snapshot(
            base_path,
            key=TrajectoryKey(2, "request"),
            tp_rank=0,
            expected_tp_size=1,
            expected_token_ids=[4, 5, 7],
        )
    with pytest.raises(RuntimeError, match="request identity"):
        load_vllm_snapshot(
            base_path,
            key=TrajectoryKey(2, "other-request"),
            tp_rank=0,
            expected_tp_size=1,
        )


def test_move_vllm_snapshot_preserves_ownership_metadata() -> None:
    key = TrajectoryKey(5, "prefetch")
    layout = KVLayout(num_layers=1, num_kv_heads=2, head_dim=4, dtype="float32", page_size=16)
    source = SealedKVSnapshot(
        key=key,
        token_ids=(1, 2, 3),
        prompt_length=1,
        layout=layout,
        layers=((torch.ones(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),),
        source="host-prefetch",
        handoff_seconds=0.25,
        streamed_tokens_before_eos=2,
        streamed_chunks_before_eos=1,
    )
    staged = move_vllm_snapshot(source, "meta", non_blocking=False)
    assert staged is not source
    assert staged.state is SnapshotState.SEALED
    assert staged.key == key
    assert staged.source == source.source
    assert staged.handoff_seconds == source.handoff_seconds
    assert staged.layers[0][0].device.type == "meta"
    assert source.layers[0][0].device.type == "cpu"


def test_streamed_vllm_snapshot_loader_assembles_contiguous_chunks(tmp_path) -> None:
    base_path = str(tmp_path / "streamed")
    packed = torch.arange(5 * 2 * 2 * 4, dtype=torch.bfloat16).reshape(5, 2, 2, 4)
    for chunk_index, (start, end) in enumerate(((0, 2), (2, 5))):
        save_file(
            {"layer_00000": packed[start:end]},
            f"{base_path}.tp0.chunk{chunk_index:05d}.safetensors",
            metadata={
                "format": "verl-streamopd-kv-v2-chunk",
                "request_id": "backend-request",
                "trajectory_id": "trajectory",
                "policy_version": "4",
                "tp_rank": "0",
                "tp_size": "1",
                "chunk_index": str(chunk_index),
                "start": str(start),
                "end": str(end),
            },
        )
    manifest = f"{base_path}.tp0.manifest.safetensors"
    save_file(
        {"token_ids": torch.tensor([2, 3, 5, 7, 11])},
        manifest,
        metadata={
            "format": "verl-streamopd-kv-v2",
            "request_id": "backend-request",
            "trajectory_id": "trajectory",
            "policy_version": "4",
            "prompt_length": "2",
            "tp_rank": "0",
            "tp_size": "1",
            "page_size": "16",
            "axis_order": "token_kv_head_dim",
            "rope_convention": "post_rope_key",
            "layer_names": '["model.layers.0.self_attn"]',
            "num_chunks": "2",
            "streamed_tokens_before_eos": "2",
            "streamed_chunks_before_eos": "1",
        },
    )
    (tmp_path / "streamed.tp0.manifest.safetensors.lock").touch()

    snapshot = load_vllm_snapshot(
        base_path,
        key=TrajectoryKey(4, "trajectory"),
        tp_rank=0,
        expected_tp_size=1,
        expected_token_ids=[2, 3, 5, 7, 11],
        expected_prompt_length=2,
    )
    assert snapshot.source == "vllm-stream-v2"
    assert snapshot.streamed_tokens_before_eos == 2
    assert snapshot.streamed_chunks_before_eos == 1
    assert snapshot.layers[0][0].shape == (1, 2, 5, 4)
    torch.testing.assert_close(snapshot.layers[0][0][0].transpose(0, 1), packed[:, 0])


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
        base_path="/tmp/trajectory",
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


@pytest.mark.parametrize(
    ("max_chunk", "min_chunk", "microbatch_size", "trace_length", "target_length", "expected"),
    [
        (2048, 256, 1, 4096, 4096, 2048),
        (2048, 256, 16, 4096, 4096, 2048),
        (2048, 256, 32, 4096, 4096, 1024),
        (2048, 256, 1, 8192, 4096, 1024),
        (2048, 256, 64, 2048, 4096, 512),
        (2048, 256, 128, 2048, 4096, 256),
    ],
)
def test_dynamic_reverse_chunk_size_bounds_token_working_set(
    max_chunk: int,
    min_chunk: int,
    microbatch_size: int,
    trace_length: int,
    target_length: int,
    expected: int,
) -> None:
    assert (
        _dynamic_reverse_chunk_size(
            max_chunk,
            min_chunk,
            microbatch_size,
            max_trace_length=trace_length,
            target_trace_length=target_length,
            alignment=64,
        )
        == expected
    )


def test_dynamic_reverse_chunk_size_prefers_maximum_with_memory_headroom() -> None:
    assert (
        _dynamic_reverse_chunk_size(
            2048,
            256,
            32,
            max_trace_length=8192,
            target_trace_length=4096,
            alignment=64,
            available_memory_bytes=64 * 1024**3,
            estimated_base_bytes=4 * 1024**3,
            estimated_bytes_per_token=8 * 1024**2,
        )
        == 2048
    )


def test_reverse_batch_planner_preserves_chunk_size_and_uses_stable_power_of_two() -> None:
    config = SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=28,
        num_key_value_heads=8,
        num_attention_heads=16,
        head_dim=128,
        vocab_size=151936,
    )
    model = SimpleNamespace(config=config, model=SimpleNamespace(layers=[object()] * 28))

    assert (
        _memory_limited_reverse_batch_size(
            model,
            configured_batch_size=16,
            trace_length=4096,
            chunk_size=1024,
            dtype=torch.bfloat16,
            available_memory_bytes=int(58.5 * 1024**3),
        )
        == 8
    )
    assert (
        _memory_limited_reverse_batch_size(
            model,
            configured_batch_size=16,
            trace_length=8192,
            chunk_size=1024,
            dtype=torch.bfloat16,
            available_memory_bytes=int(58.5 * 1024**3),
        )
        == 4
    )
    assert (
        _memory_limited_reverse_batch_size(
            model,
            configured_batch_size=32,
            trace_length=4096,
            chunk_size=1024,
            dtype=torch.bfloat16,
            available_memory_bytes=256 * 1024**3,
        )
        == 32
    )


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


def test_zero_loss_synthetic_padding_is_not_trainable() -> None:
    real = torch.tensor([1, 1, 0], dtype=torch.int64)
    padding = torch.zeros(1, dtype=torch.int64)

    assert _has_valid_response(TensorDict({"response_mask": real}, batch_size=[])) is True
    assert _has_valid_response(TensorDict({"response_mask": padding}, batch_size=[])) is False


def test_prepare_config_installs_connector_and_rejects_stale_trainer() -> None:
    config = OmegaConf.create(
        {
            "trainer": {
                "use_v1": True,
                "n_gpus_per_node": 4,
                "nnodes": 1,
                "v1": {"trainer_mode": "sync", "sync": {}},
            },
            "data": {"train_batch_size": 128},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
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
                "streamopd_kv": {"enabled": True, "kv_handoff_dir": "/tmp/test-streamopd"},
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

    overlap_config = copy.deepcopy(config)
    overlap_config.distillation.streamopd_kv.overlap_rollout_training = True
    overlap_config.distillation.streamopd_kv.rollout_micro_batch_size = 32
    prepare_streamopd_kv_config(overlap_config)
    assert overlap_config.trainer.v1.sync.parameter_sync_step == 4

    colocated_config = copy.deepcopy(config)
    colocated_config.distillation.streamopd_kv.colocate_teacher_with_student = True
    prepare_streamopd_kv_config(colocated_config)

    dedicated_config = copy.deepcopy(config)
    dedicated_config.trainer.n_gpus_per_node = 2
    dedicated_config.trainer.v1.trainer_mode = "streamopd_colocate"
    dedicated_config.trainer.v1.streamopd_colocate = {"micro_batch_size": 1, "parameter_sync_step": 1}
    dedicated_config.trainer.v1.sampler = {
        "max_off_policy_threshold": 8,
        "max_off_policy_strategy": "drop",
    }
    dedicated_config.actor_rollout_ref.rollout.nnodes = 1
    dedicated_config.actor_rollout_ref.rollout.n_gpus_per_node = 2
    dedicated_config.actor_rollout_ref.rollout.checkpoint_engine = {"backend": "nccl"}
    dedicated_config.distillation.streamopd_kv.colocate_teacher_with_student = True
    dedicated_config.distillation.streamopd_kv.micro_batch_size = 16
    prepare_streamopd_kv_config(dedicated_config)
    assert dedicated_config.trainer.v1.streamopd_colocate.micro_batch_size == 16
    assert dedicated_config.trainer.v1.streamopd_colocate.parameter_sync_step == 8
    assert dedicated_config.trainer.v1.sampler.max_off_policy_threshold == 1
    assert dedicated_config.trainer.v1.sampler.max_off_policy_strategy == "drop"

    reverse_trigger_config = copy.deepcopy(config)
    reverse_trigger_config.trainer.v1.trainer_mode = "streamopd_colocate"
    reverse_trigger_config.trainer.v1.streamopd_colocate = {"micro_batch_size": 1, "parameter_sync_step": 1}
    reverse_trigger_config.distillation.streamopd_kv.colocate_teacher_with_student = True
    reverse_trigger_config.distillation.streamopd_kv.micro_batch_size = 32
    reverse_trigger_config.distillation.streamopd_kv.reverse_batch_size = 16
    reverse_trigger_config.trainer.n_gpus_per_node = 2
    reverse_trigger_config.actor_rollout_ref.rollout.nnodes = 1
    reverse_trigger_config.actor_rollout_ref.rollout.n_gpus_per_node = 2
    reverse_trigger_config.actor_rollout_ref.rollout.checkpoint_engine = {"backend": "nccl"}
    reverse_trigger_config.trainer.v1.sampler = {"max_off_policy_threshold": 8, "max_off_policy_strategy": "drop"}
    prepare_streamopd_kv_config(reverse_trigger_config)
    assert reverse_trigger_config.trainer.v1.streamopd_colocate.parameter_sync_step == 4

    config.trainer.v1.trainer_mode = "separate_async"
    with pytest.raises(ValueError, match="sync or streamopd_colocate"):
        prepare_streamopd_kv_config(config)


def test_streamopd_colocate_starts_with_reverse_capacity_cohort() -> None:
    from verl.trainer.ppo.v1.trainer_streamopd_colocate import _streamopd_batch_sizes

    assert _streamopd_batch_sizes(128, 32, 16) == [16, 32, 32, 32, 16]
    assert _streamopd_batch_sizes(128, 16, 16) == [16] * 8
    assert _streamopd_batch_sizes(128, 32, 64) == [32] * 4


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


def test_dapo_adapter_wraps_plain_prompt_as_chat_messages() -> None:
    from benchmarks.streamopd_kv.dapo_math_dataset import DAPOMathDataset

    dataset = DAPOMathDataset.__new__(DAPOMathDataset)
    assert dataset._build_messages({"prompt": "2 + 2?"}, "prompt") == [{"role": "user", "content": "2 + 2?"}]


def test_vllm_connector_waits_for_claimed_copy_event_and_ignores_unclaimed_request() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector

    class Event:
        complete = False

        def query(self) -> bool:
            return self.complete

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._connector_metadata = None
    connector._finished_requests = set()
    connector._claimed_requests = {"claimed"}
    connector._copy_events = {}
    connector._lock_fds = {}
    assert connector.get_finished({"claimed", "ordinary"}) == (None, None)

    event = Event()
    connector._copy_events["claimed"] = event
    assert connector.get_finished(set()) == (None, None)
    event.complete = True
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
            base_path="/tmp/streamopd/backend-id",
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


@pytest.mark.parametrize(("sequence_length", "chunk_size"), [(5, 8), (7, 3), (9, 4)])
def test_reverse_chunk_gradients_match_full_sequence_opd(sequence_length: int, chunk_size: int) -> None:
    torch.manual_seed(7)
    baseline = _ToyCausalLM().eval()
    reverse = copy.deepcopy(baseline).eval()
    full_tokens = torch.tensor([[2, 5, 7, 11, 13, 17, 19, 23, 29, 3]])[:, : sequence_length + 1]
    input_ids, targets = full_tokens[:, :-1], full_tokens[:, 1:]
    loss_mask = torch.arange(sequence_length).unsqueeze(0) >= 2

    baseline_logits, _ = baseline(input_ids)
    baseline_loss = F.cross_entropy(baseline_logits[loss_mask], targets[loss_mask], reduction="sum")
    baseline_loss.backward()

    with torch.no_grad():
        _, rollout_cache = reverse(input_ids)
    state = ReverseChunkState([LayerKVTrace(key, value) for key, value in rollout_cache])
    reverse_loss = torch.zeros(())
    for end in range(input_ids.shape[1], 0, -chunk_size):
        start = max(0, end - chunk_size)
        state.begin(start, end)
        logits, _ = reverse(input_ids[:, start:end], reverse_state=state)
        local_mask = loss_mask[:, start:end]
        chunk_loss = (
            F.cross_entropy(logits[local_mask], targets[:, start:end][local_mask], reduction="sum")
            if local_mask.any()
            else logits.sum() * 0.0
        )
        (chunk_loss + state.gradient_injection()).backward()
        state.commit_prefix_gradients(release_processed_suffix=True)
        reverse_loss += chunk_loss.detach()

    torch.testing.assert_close(reverse_loss, baseline_loss.detach(), rtol=1e-6, atol=1e-6)
    for (baseline_name, baseline_parameter), (reverse_name, reverse_parameter) in zip(
        baseline.named_parameters(), reverse.named_parameters(), strict=True
    ):
        assert baseline_name == reverse_name
        assert baseline_parameter.grad is not None, baseline_name
        assert reverse_parameter.grad is not None, reverse_name
        torch.testing.assert_close(
            reverse_parameter.grad,
            baseline_parameter.grad,
            rtol=2e-4,
            atol=2e-5,
            msg=lambda message, name=baseline_name: f"{name}: {message}",
        )


def test_batched_ragged_reverse_gradients_match_full_sequence() -> None:
    torch.manual_seed(43)
    baseline = _ToyCausalLM().eval()
    reverse = copy.deepcopy(baseline).eval()
    sequences = [
        torch.tensor([[2, 5, 7, 11, 13, 17, 19]]),
        torch.tensor([[3, 7, 11, 17, 23, 29, 2, 5, 13]]),
    ]
    targets = [torch.roll(sequence, shifts=-1, dims=1) for sequence in sequences]
    valid_masks = [torch.arange(sequence.shape[1]).unsqueeze(0) >= 2 for sequence in sequences]

    baseline_loss = torch.zeros(())
    for sequence, target, valid in zip(sequences, targets, valid_masks, strict=True):
        logits, _ = baseline(sequence)
        baseline_loss = baseline_loss + F.cross_entropy(logits[valid], target[valid], reduction="sum")
    baseline_loss.backward()

    traces = []
    with torch.no_grad():
        for sequence in sequences:
            _, cache = reverse(sequence)
            traces.append([LayerKVTrace(key, value) for key, value in cache])
    state = BatchedReverseChunkState(traces)
    chunk_size = 3
    ends = [sequence.shape[1] for sequence in sequences]
    reverse_loss = torch.zeros(())
    while active := [idx for idx, end in enumerate(ends) if end >= chunk_size]:
        starts = [ends[idx] - chunk_size for idx in active]
        active_ends = [ends[idx] for idx in active]
        state.begin(active, starts, active_ends)
        chunk_ids = torch.cat(
            [sequences[idx][:, start:end] for idx, start, end in zip(active, starts, active_ends, strict=True)]
        )
        logits, _ = reverse(chunk_ids, reverse_state=state)
        chunk_loss = logits.sum() * 0.0
        for row, (idx, start, end) in enumerate(zip(active, starts, active_ends, strict=True)):
            mask = valid_masks[idx][:, start:end]
            if mask.any():
                chunk_loss = chunk_loss + F.cross_entropy(
                    logits[row : row + 1][mask], targets[idx][:, start:end][mask], reduction="sum"
                )
        (chunk_loss + state.gradient_injection()).backward()
        state.commit_prefix_gradients(release_processed_suffix=True)
        reverse_loss += chunk_loss.detach()
        for idx, start in zip(active, starts, strict=True):
            ends[idx] = start

    if active := [idx for idx, end in enumerate(ends) if end]:
        starts = [0] * len(active)
        active_ends = [ends[idx] for idx in active]
        state.begin(active, starts, active_ends)
        max_residual = max(active_ends)
        chunk_ids = torch.cat(
            [
                F.pad(sequences[idx][:, :end], (0, max_residual - end))
                for idx, end in zip(active, active_ends, strict=True)
            ]
        )
        logits, _ = reverse(chunk_ids, reverse_state=state)
        chunk_loss = logits.sum() * 0.0
        for row, (idx, end) in enumerate(zip(active, active_ends, strict=True)):
            mask = valid_masks[idx][:, :end]
            if mask.any():
                chunk_loss = chunk_loss + F.cross_entropy(
                    logits[row : row + 1, :end][mask], targets[idx][:, :end][mask], reduction="sum"
                )
        (chunk_loss + state.gradient_injection()).backward()
        state.commit_prefix_gradients(release_processed_suffix=True)
        reverse_loss += chunk_loss.detach()

    torch.testing.assert_close(reverse_loss, baseline_loss.detach(), rtol=1e-6, atol=1e-6)
    for (baseline_name, baseline_parameter), (reverse_name, reverse_parameter) in zip(
        baseline.named_parameters(), reverse.named_parameters(), strict=True
    ):
        assert baseline_name == reverse_name
        torch.testing.assert_close(
            reverse_parameter.grad,
            baseline_parameter.grad,
            rtol=3e-4,
            atol=3e-5,
            msg=lambda message, name=baseline_name: f"{name}: {message}",
        )


def test_qwen3_paged_reverse_rejects_trace_length_mismatch() -> None:
    trainer = Qwen3ReverseTrainer(torch.nn.Linear(2, 2).eval(), chunk_size=16, page_size=16)
    sequences = [torch.ones((1, 4), dtype=torch.long), torch.ones((1, 5), dtype=torch.long)]
    loss_fns = [lambda *_: (torch.zeros(()), 0), lambda *_: (torch.zeros(()), 0)]
    layer = LayerKVTrace(torch.zeros((1, 1, 4, 2)), torch.zeros((1, 1, 4, 2)))
    with pytest.raises(ValueError, match="trace must match"):
        trainer.backward_batched(sequences, [[layer], [layer]], loss_fns)
