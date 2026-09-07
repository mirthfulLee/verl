# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import asyncio

import pytest
import torch

from verl.experimental.streamopd_kv.planning import plan_teacher_admission
from verl.experimental.streamopd_kv.protocol import CommittedTokenChunk, TrajectoryKey
from verl.experimental.streamopd_kv.streaming_teacher import StreamingTeacherCoordinator


def test_variable_prompt_plan_preserves_memory_budget_and_explicit_session_cap():
    kwargs = dict(
        expected_trajectories=128,
        trajectory_tokens=4097,
        vllm_capacity_tokens=430080,
        page_size=64,
        max_batched_tokens=4096,
        initial_chunk_tokens=1024,
    )
    fixed = plan_teacher_admission(**kwargs)
    dynamic = plan_teacher_admission(**kwargs, variable_reservations=True)
    assert fixed["active_trajectories"] == 103
    assert dynamic["active_trajectories"] == 128
    assert dynamic["active_kv_tokens"] == 430080
    capped = plan_teacher_admission(**kwargs, variable_reservations=True, trajectory_cap=8)
    assert capped["active_trajectories"] == 8
    assert capped["active_kv_tokens"] == 8 * 4160


@pytest.mark.asyncio
async def test_actual_prompt_reservations_admit_fitting_sessions_and_block_the_next():
    started, finish = {key: asyncio.Event() for key in "abc"}, {key: asyncio.Event() for key in "abc"}

    async def score(tokens, request_id, terminal):
        key = request_id.rsplit("-", 1)[-1]
        started[key].set()
        await finish[key].wait()
        return torch.zeros(len(tokens), 2, dtype=torch.long), torch.zeros(len(tokens), 2)

    coordinator = StreamingTeacherCoordinator(
        score,
        max_pending_chunks=3,
        max_active_trajectories=3,
        max_active_kv_tokens=64,
        kv_page_size=8,
        kv_reservation_tokens=128,
        max_response_tokens=16,
    )
    for key, prompt_length in (("a", 1), ("b", 9), ("c", 9)):
        await coordinator.submit(CommittedTokenChunk(TrajectoryKey(0, key), 0, (4,), True, (1,) * prompt_length))
    await asyncio.wait_for(asyncio.gather(started["a"].wait(), started["b"].wait()), timeout=1)
    assert not started["c"].is_set()
    assert coordinator._local_admission._active_kv_tokens == 24 + 32
    finish["a"].set()
    await asyncio.wait_for(started["c"].wait(), timeout=1)
    assert coordinator._local_admission._active_kv_tokens == 64
    finish["b"].set()
    finish["c"].set()
    await asyncio.gather(*(coordinator.result(TrajectoryKey(0, key), 1) for key in "abc"))
    assert coordinator._local_admission._active_kv_tokens == 0
