# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.streamopd_kv import (
    CommittedChunkPublisher,
    CommittedTokenChunk,
    StreamingTeacherCoordinator,
    TrajectoryKey,
)
from verl.experimental.streamopd_kv.fsdp_worker import (
    _fixed_reverse_slot_plan,
)


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


@pytest.mark.asyncio
async def test_teacher_client_holds_load_balancer_lease_until_stream_eos() -> None:
    from verl.experimental.streamopd_kv.teacher_client import StreamingTeacherClient

    class RemoteMethod:
        async def remote(self, **kwargs):
            return {"terminal": kwargs["terminal"]}

    class Server:
        stream_teacher_chunk = RemoteMethod()

    class Client(StreamingTeacherClient):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", [None, "ids", "logprobs"])
async def test_teacher_reference_validation_ignores_only_unsupervised_final_row(mismatch):
    from verl.experimental.streamopd_kv.agent import StreamOPDAgentSession

    ids = torch.tensor([[4, 5], [7, 8], [10, 11]], dtype=torch.int32)
    logprobs = torch.tensor([[-0.4, -1.2], [-0.6, -1.3], [-0.2, -1.1]])
    expected_ids, expected_logprobs = ids.clone(), logprobs.clone()
    expected_ids[-1] = 0
    expected_logprobs[-1] = 0
    if mismatch == "ids":
        expected_ids[1, 0] = 9
    elif mismatch == "logprobs":
        expected_logprobs[1, 0] -= 1

    async def streamed(*args, **kwargs):
        return ids, logprobs

    async def full(*args, **kwargs):
        return expected_ids, expected_logprobs

    session = StreamOPDAgentSession.__new__(StreamOPDAgentSession)
    session.config = SimpleNamespace(validate_teacher_artifacts=True, validation_atol=1e-4)
    session.coordinator = SimpleNamespace(result=streamed)
    session.teacher_manager = SimpleNamespace(compute_teacher_logprobs_single=full)
    output = SimpleNamespace(
        multi_modal_data=None,
        extra_fields={
            "streamopd_trajectory_id": "trajectory",
            "streamopd_policy_version": 0,
        },
    )
    if mismatch:
        with pytest.raises(RuntimeError, match="differ from reference"):
            await session.result(output, prompt_ids=[1], response_ids=[2, 3], routing_key=None)
    else:
        result = await session.result(output, prompt_ids=[1], response_ids=[2, 3], routing_key=None)
        torch.testing.assert_close(result[0], ids)
        torch.testing.assert_close(result[1], logprobs)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
async def test_ordinary_teacher_scoring_preserves_backend_request_contract(backend):
    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

    class SGLangClient:
        async def generate(
            self, request_id, prompt_ids, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        ):
            assert len(prompt_ids) == 2
            assert sampling_params["prompt_logprobs"] == 2
            return SimpleNamespace(extra_fields={"prompt_ids": [[1, 2], [3, 4]], "prompt_logprobs": [[-1.0, -2.0]] * 2})

    class VLLMClient(SGLangClient):
        async def generate(self, *, prompt_logprobs_as_tensors=False, **kwargs):
            assert prompt_logprobs_as_tensors
            return await super().generate(**kwargs)

    manager = AsyncTeacherLLMServerManager.__new__(AsyncTeacherLLMServerManager)
    manager.teacher_model_configs = {
        "teacher": SimpleNamespace(inference=SimpleNamespace(name=backend, temperature=1.0))
    }
    manager.distillation_loss_config = SimpleNamespace(topk=2, loss_settings=SimpleNamespace(use_topk=True))
    manager.teacher_client = {"teacher": VLLMClient() if backend == "vllm" else SGLangClient()}
    ids, logprobs = await manager.compute_teacher_logprobs_single([5, 6])
    assert ids.dtype == torch.int32
    assert ids.tolist() == [[1, 2], [3, 4]]
    assert logprobs.shape == (2, 2)
