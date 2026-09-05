# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
import time

import pytest
import torch

from verl.experimental.streamopd_kv import (
    CommittedTokenChunk,
    StreamingTeacherCoordinator,
    TrajectoryKey,
)
from verl.experimental.streamopd_kv.scheduler import StreamOPDTaskScheduler


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
