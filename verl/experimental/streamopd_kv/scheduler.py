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
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class _TeacherAdmissionWaiter:
    reservation_tokens: int
    max_active_trajectories: int
    max_active_kv_tokens: int
    started_at: float = field(default_factory=time.perf_counter)
    ready: asyncio.Event = field(default_factory=asyncio.Event)


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("StreamOPD timing interval ends before it starts")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _interval_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _interval_overlap_seconds(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    left_idx = 0
    right_idx = 0
    while left_idx < len(left) and right_idx < len(right):
        left_start, left_end = left[left_idx]
        right_start, right_end = right[right_idx]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_idx += 1
        else:
            right_idx += 1
    return total


def _max_interval_concurrency(intervals: Sequence[tuple[float, float]]) -> int:
    events = [
        (timestamp, delta) for start, end in intervals if end > start for timestamp, delta in ((start, 1), (end, -1))
    ]
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


class StreamOPDTaskScheduler:
    """Central accounting and atomic admission for StreamOPD work queues.

    Teacher inference remains asynchronous. Teacher and Trainer may execute
    concurrently only when their declared resource sets are disjoint.
    """

    def __init__(
        self,
        teacher_resources: tuple[str, ...] = ("shared",),
        trainer_resources: tuple[str, ...] = ("shared",),
    ) -> None:
        if not teacher_resources or not trainer_resources:
            raise ValueError("StreamOPD scheduler resource sets must be non-empty")
        self.teacher_resources = frozenset(teacher_resources)
        self.trainer_resources = frozenset(trainer_resources)
        self.resources_overlap = bool(self.teacher_resources & self.trainer_resources)
        self.policy_version: int | None = None
        self.train_launch_width = 1
        self.teacher_available = True
        self.teacher_sessions: dict[str, int] = {}
        self.teacher_session_kv_tokens = 0
        self.expected_trajectories = 0
        self.terminal_trajectories = 0
        self.completed_teacher_trajectories = 0
        self.all_rollouts_terminal_at = 0.0
        self.all_teacher_completed_at = 0.0
        self.training_active = 0
        self.training_waiters: list[int] = []
        self.ready_training_trajectories = 0
        self.max_training_waiters = 0
        self.teacher_chunks = 0
        self.teacher_notifications = 0
        self.training_units = 0
        self.max_teacher_sessions = 0
        self.max_teacher_session_kv_tokens = 0
        self.teacher_admission_attempts = 0
        self.teacher_admission_rejections = 0
        self.teacher_admission_unavailable_rejections = 0
        self.teacher_admission_trajectory_rejections = 0
        self.teacher_admission_kv_rejections = 0
        self.teacher_admission_waiters: dict[str, _TeacherAdmissionWaiter] = {}
        self.teacher_admission_wait_seconds = 0.0
        self.teacher_admission_max_wait_seconds = 0.0
        self.teacher_admission_waited_sessions = 0
        self._training_busy_started_at = 0.0
        self._teacher_score_intervals: list[tuple[float, float]] = []
        self._training_intervals: list[tuple[float, float]] = []
        self.first_training_ready_at = 0.0
        self.first_training_started_at = 0.0
        self.teacher_completed_at_first_training = 0
        self.rollouts_terminal_at_first_training = 0
        self.teacher_pending_at_first_training = 0
        self.training_trajectories_started = 0
        self.policy_started_at = 0.0

    def begin_policy(
        self,
        policy_version: int,
        expected_trajectories: int = 0,
        train_launch_width: int = 1,
        teacher_available: bool = True,
    ) -> None:
        if self.training_active or self.teacher_sessions or self.teacher_admission_waiters or self.training_waiters:
            raise RuntimeError(
                "cannot begin a StreamOPD policy version while work is active: "
                f"teacher_sessions={len(self.teacher_sessions)}, training_active={self.training_active}, "
                f"teacher_admission_waiters={len(self.teacher_admission_waiters)}, "
                f"training_waiters={len(self.training_waiters)}"
            )
        self.policy_version = int(policy_version)
        if expected_trajectories < 0:
            raise ValueError("expected_trajectories must be non-negative")
        if train_launch_width < 1:
            raise ValueError("train_launch_width must be positive")
        self.teacher_available = bool(teacher_available)
        self.train_launch_width = int(train_launch_width)
        self.expected_trajectories = int(expected_trajectories)
        self.terminal_trajectories = 0
        self.completed_teacher_trajectories = 0
        self.all_rollouts_terminal_at = 0.0
        self.all_teacher_completed_at = 0.0
        self.teacher_session_kv_tokens = 0
        self.training_waiters = []
        self.ready_training_trajectories = 0
        self.max_training_waiters = 0
        self.teacher_chunks = 0
        self.teacher_notifications = 0
        self.training_units = 0
        self.max_teacher_sessions = 0
        self.max_teacher_session_kv_tokens = 0
        self.teacher_admission_attempts = 0
        self.teacher_admission_rejections = 0
        self.teacher_admission_unavailable_rejections = 0
        self.teacher_admission_trajectory_rejections = 0
        self.teacher_admission_kv_rejections = 0
        self.teacher_admission_waiters = {}
        self.teacher_admission_wait_seconds = 0.0
        self.teacher_admission_max_wait_seconds = 0.0
        self.teacher_admission_waited_sessions = 0
        self._training_busy_started_at = 0.0
        self._teacher_score_intervals = []
        self._training_intervals = []
        self.first_training_ready_at = 0.0
        self.first_training_started_at = 0.0
        self.teacher_completed_at_first_training = 0
        self.rollouts_terminal_at_first_training = 0
        self.teacher_pending_at_first_training = 0
        self.training_trajectories_started = 0
        self.policy_started_at = time.perf_counter()

    def teacher_trajectory_terminal_submitted(self, policy_version: int, notifications: int = 1) -> None:
        self._check_version(policy_version)
        if notifications < 1:
            raise ValueError("a terminal Teacher trajectory must contain at least one notification")
        self.teacher_notifications += notifications
        self.terminal_trajectories += 1
        if self.expected_trajectories and self.terminal_trajectories > self.expected_trajectories:
            raise RuntimeError("received more terminal Teacher trajectories than expected")
        if self.terminal_trajectories == self.expected_trajectories:
            self.all_rollouts_terminal_at = time.perf_counter()

    def teacher_trajectory_completed(
        self,
        policy_version: int,
        score_intervals: Sequence[tuple[float, float]] = (),
    ) -> None:
        self._check_version(policy_version)
        intervals = [(float(start), float(end)) for start, end in score_intervals]
        _merge_intervals(intervals)
        self._teacher_score_intervals.extend(intervals)
        self.teacher_chunks += len(intervals)
        self.completed_teacher_trajectories += 1
        if self.expected_trajectories and self.completed_teacher_trajectories > self.expected_trajectories:
            raise RuntimeError("completed more Teacher trajectories than expected")
        if self.completed_teacher_trajectories == self.expected_trajectories:
            self.all_teacher_completed_at = time.perf_counter()

    def _teacher_admission_rejection(
        self,
        kv_reservation_tokens: int,
        max_active_trajectories: int,
        max_active_kv_tokens: int,
    ) -> str | None:
        if not self.teacher_available:
            return "unavailable"
        if len(self.teacher_sessions) >= max_active_trajectories:
            return "trajectory"
        if self.teacher_sessions and self.teacher_session_kv_tokens + kv_reservation_tokens > max_active_kv_tokens:
            return "kv"
        return None

    def _reserve_teacher_session(self, session_id: str, kv_reservation_tokens: int) -> None:
        waiter = self.teacher_admission_waiters.pop(session_id, None)
        if waiter is not None:
            wait_seconds = time.perf_counter() - waiter.started_at
            self.teacher_admission_wait_seconds += wait_seconds
            self.teacher_admission_max_wait_seconds = max(self.teacher_admission_max_wait_seconds, wait_seconds)
            self.teacher_admission_waited_sessions += 1
        self.teacher_sessions[session_id] = kv_reservation_tokens
        self.teacher_session_kv_tokens += kv_reservation_tokens
        self.max_teacher_sessions = max(self.max_teacher_sessions, len(self.teacher_sessions))
        self.max_teacher_session_kv_tokens = max(
            self.max_teacher_session_kv_tokens,
            self.teacher_session_kv_tokens,
        )
        if waiter is not None:
            waiter.ready.set()

    def _grant_teacher_admission_waiters(self) -> None:
        for session_id, waiter in tuple(self.teacher_admission_waiters.items()):
            if (
                self._teacher_admission_rejection(
                    waiter.reservation_tokens,
                    waiter.max_active_trajectories,
                    waiter.max_active_kv_tokens,
                )
                is not None
            ):
                break
            self._reserve_teacher_session(session_id, waiter.reservation_tokens)

    def try_teacher_session_admitted(
        self,
        policy_version: int,
        session_id: str,
        kv_reservation_tokens: int,
        max_active_trajectories: int,
        max_active_kv_tokens: int,
    ) -> bool:
        """Atomically reserve one live resumable teacher KV session."""
        self._check_version(policy_version)
        self.teacher_admission_attempts += 1
        if session_id in self.teacher_sessions:
            return True
        waiter = self.teacher_admission_waiters.get(session_id)
        if waiter is not None and (
            waiter.reservation_tokens != kv_reservation_tokens
            or waiter.max_active_trajectories != max_active_trajectories
            or waiter.max_active_kv_tokens != max_active_kv_tokens
        ):
            raise RuntimeError(f"teacher session {session_id!r} changed its pending admission request")
        rejection = self._teacher_admission_rejection(
            kv_reservation_tokens,
            max_active_trajectories,
            max_active_kv_tokens,
        )
        if rejection is not None:
            self.teacher_admission_rejections += 1
            if rejection == "unavailable":
                self.teacher_admission_unavailable_rejections += 1
            elif rejection == "trajectory":
                self.teacher_admission_trajectory_rejections += 1
            else:
                self.teacher_admission_kv_rejections += 1
            self.teacher_admission_waiters.setdefault(
                session_id,
                _TeacherAdmissionWaiter(
                    reservation_tokens=kv_reservation_tokens,
                    max_active_trajectories=max_active_trajectories,
                    max_active_kv_tokens=max_active_kv_tokens,
                ),
            )
            return False
        self._reserve_teacher_session(session_id, kv_reservation_tokens)
        return True

    async def wait_teacher_session_admitted(
        self,
        policy_version: int,
        session_id: str,
        kv_reservation_tokens: int,
        max_active_trajectories: int,
        max_active_kv_tokens: int,
    ) -> bool:
        """Wait for one admission notification without polling the Ray actor."""

        if self.try_teacher_session_admitted(
            policy_version,
            session_id,
            kv_reservation_tokens,
            max_active_trajectories,
            max_active_kv_tokens,
        ):
            return True
        waiter = self.teacher_admission_waiters[session_id]
        await waiter.ready.wait()
        if session_id not in self.teacher_sessions:
            raise RuntimeError(f"teacher session {session_id!r} admission was cancelled")
        return True

    def teacher_session_released(self, policy_version: int, session_id: str) -> None:
        self._check_version(policy_version)
        try:
            reservation = self.teacher_sessions.pop(session_id)
        except KeyError as error:
            raise RuntimeError(f"teacher session {session_id!r} was not admitted") from error
        self.teacher_session_kv_tokens -= reservation
        if self.teacher_session_kv_tokens < 0:
            raise RuntimeError("teacher session KV accounting became negative")
        self._grant_teacher_admission_waiters()

    def teacher_session_admission_cancelled(self, policy_version: int, session_id: str) -> None:
        """Release a pending or granted reservation after coordinator cancellation."""

        self._check_version(policy_version)
        waiter = self.teacher_admission_waiters.pop(session_id, None)
        if waiter is not None:
            waiter.ready.set()
        reservation = self.teacher_sessions.pop(session_id, None)
        if reservation is not None:
            self.teacher_session_kv_tokens -= reservation
            if self.teacher_session_kv_tokens < 0:
                raise RuntimeError("teacher session KV accounting became negative")
        if waiter is not None or reservation is not None:
            self._grant_teacher_admission_waiters()

    def teacher_wake_completed(self, policy_version: int) -> None:
        """Open Teacher admission after an asynchronous wake."""

        self._check_version(policy_version)
        self.teacher_available = True
        self._grant_teacher_admission_waiters()

    def training_started(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if self.resources_overlap and not self.teacher_drained:
            raise RuntimeError("shared StreamOPD reverse training requires complete Teacher drain")
        if self.training_active:
            raise RuntimeError("StreamOPD permits only one controller-level reverse-training unit at a time")
        self.training_active = 1
        self.training_units += 1
        self._training_busy_started_at = time.perf_counter()

    def training_waiting(
        self,
        policy_version: int,
        trajectory_count: int = 1,
    ) -> None:
        """Register a ready reverse unit while it waits for pool ownership."""

        self._check_version(policy_version)
        if trajectory_count < 1:
            raise ValueError("ready training trajectory count must be positive")
        if not self.first_training_ready_at:
            self.first_training_ready_at = time.perf_counter()
        self.training_waiters.append(int(trajectory_count))
        self.ready_training_trajectories += int(trajectory_count)
        self.max_training_waiters = max(self.max_training_waiters, len(self.training_waiters))

    def training_waiting_cancelled(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if not self.training_waiters:
            raise RuntimeError("training_waiting_cancelled without a waiting reverse unit")
        trajectory_count = self.training_waiters.pop(0)
        self.ready_training_trajectories -= trajectory_count

    def try_training_started(self, policy_version: int) -> bool:
        self._check_version(policy_version)
        # A shared pool changes active ownership once per policy: all Teacher
        # work drains, Teacher sleeps, then Trainer runs its full phase.
        if self.resources_overlap and not self.teacher_drained:
            return False
        if self.training_waiters:
            trajectory_count = self.training_waiters.pop(0)
            self.ready_training_trajectories -= trajectory_count
            self.training_trajectories_started += trajectory_count
        if not self.first_training_started_at:
            self.first_training_started_at = time.perf_counter()
            self.teacher_completed_at_first_training = self.completed_teacher_trajectories
            self.rollouts_terminal_at_first_training = self.terminal_trajectories
            self.teacher_pending_at_first_training = len(self.teacher_sessions)
        self.training_started(policy_version)
        return True

    def training_finished(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if not self.training_active:
            raise RuntimeError("training_finished without an active StreamOPD training unit")
        self._training_intervals.append((self._training_busy_started_at, time.perf_counter()))
        self._training_busy_started_at = 0.0
        self.training_active = 0

    def snapshot(self) -> dict[str, int | float | str | bool | None]:
        return {
            "policy_version": self.policy_version,
            "teacher_sessions": len(self.teacher_sessions),
            "teacher_session_kv_tokens": self.teacher_session_kv_tokens,
            "expected_trajectories": self.expected_trajectories,
            "terminal_trajectories": self.terminal_trajectories,
            "completed_teacher_trajectories": self.completed_teacher_trajectories,
            "teacher_pending": self.teacher_pending,
            "teacher_available": self.teacher_available,
            "training_active": self.training_active,
            "training_waiters": len(self.training_waiters),
            "ready_training_trajectories": self.ready_training_trajectories,
            "train_launch_width": self.train_launch_width,
            "resources_overlap": self.resources_overlap,
            "teacher_drained": self.teacher_drained,
            "max_training_waiters": self.max_training_waiters,
            "teacher_chunks": self.teacher_chunks,
            "teacher_notifications": self.teacher_notifications,
            "training_units": self.training_units,
            "max_teacher_sessions": self.max_teacher_sessions,
            "max_teacher_session_kv_tokens": self.max_teacher_session_kv_tokens,
            "teacher_admission_attempts": self.teacher_admission_attempts,
            "teacher_admission_rejections": self.teacher_admission_rejections,
            "teacher_admission_waiters": len(self.teacher_admission_waiters),
            "training_trajectories_started": self.training_trajectories_started,
        }

    def end_policy(self, policy_version: int) -> dict[str, float]:
        self._check_version(policy_version)
        if self.training_active or self.teacher_sessions or self.teacher_admission_waiters or self.training_waiters:
            raise RuntimeError(
                "StreamOPD policy barrier reached with unfinished work: "
                f"teacher_sessions={len(self.teacher_sessions)}, training_active={self.training_active}, "
                f"teacher_admission_waiters={len(self.teacher_admission_waiters)}, "
                f"training_waiters={len(self.training_waiters)}"
            )
        if self.expected_trajectories and (
            self.terminal_trajectories != self.expected_trajectories
            or self.completed_teacher_trajectories != self.expected_trajectories
        ):
            raise RuntimeError(
                "StreamOPD policy barrier reached with incomplete trajectories: "
                f"terminal={self.terminal_trajectories}, completed={self.completed_teacher_trajectories}, "
                f"expected={self.expected_trajectories}"
            )
        rollout_terminal_seconds = (
            self.all_rollouts_terminal_at - self.policy_started_at if self.all_rollouts_terminal_at else 0.0
        )
        teacher_complete_seconds = (
            self.all_teacher_completed_at - self.policy_started_at if self.all_teacher_completed_at else 0.0
        )
        teacher_drain_seconds = (
            max(0.0, self.all_teacher_completed_at - self.all_rollouts_terminal_at)
            if self.all_rollouts_terminal_at and self.all_teacher_completed_at
            else 0.0
        )
        teacher_intervals = _merge_intervals(self._teacher_score_intervals)
        training_intervals = _merge_intervals(self._training_intervals)
        teacher_busy_seconds = _interval_seconds(teacher_intervals)
        training_busy_seconds = _interval_seconds(training_intervals)
        concurrent_busy_seconds = _interval_overlap_seconds(teacher_intervals, training_intervals)
        first_teacher_started_seconds = teacher_intervals[0][0] - self.policy_started_at if teacher_intervals else 0.0
        first_training_ready_seconds = (
            self.first_training_ready_at - self.policy_started_at if self.first_training_ready_at else 0.0
        )
        first_training_started_seconds = (
            self.first_training_started_at - self.policy_started_at if self.first_training_started_at else 0.0
        )
        policy_seconds = time.perf_counter() - self.policy_started_at
        pool_busy_seconds = teacher_busy_seconds + training_busy_seconds
        busy_union_seconds = pool_busy_seconds - concurrent_busy_seconds
        teacher_busy_before_rollout_terminal = 0.0
        if self.all_rollouts_terminal_at:
            teacher_busy_before_rollout_terminal = sum(
                max(0.0, min(end, self.all_rollouts_terminal_at) - start) for start, end in teacher_intervals
            )
        metrics = {
            "streamopd/scheduler_teacher_chunks": float(self.teacher_chunks),
            "streamopd/scheduler_teacher_notifications": float(self.teacher_notifications),
            "streamopd/scheduler_teacher_coalesced_fragments": float(self.teacher_notifications - self.teacher_chunks),
            "streamopd/scheduler_training_units": float(self.training_units),
            "streamopd/scheduler_max_training_waiters": float(self.max_training_waiters),
            "streamopd/scheduler_max_teacher_pending": float(self.max_teacher_sessions),
            "streamopd/scheduler_max_teacher_active": float(_max_interval_concurrency(self._teacher_score_intervals)),
            "streamopd/scheduler_max_teacher_sessions": float(self.max_teacher_sessions),
            "streamopd/scheduler_max_teacher_session_kv_tokens": float(self.max_teacher_session_kv_tokens),
            "streamopd/scheduler_teacher_admission_attempts": float(self.teacher_admission_attempts),
            "streamopd/scheduler_teacher_admission_rejections": float(self.teacher_admission_rejections),
            "streamopd/scheduler_teacher_admission_unavailable_rejections": float(
                self.teacher_admission_unavailable_rejections
            ),
            "streamopd/scheduler_teacher_admission_trajectory_rejections": float(
                self.teacher_admission_trajectory_rejections
            ),
            "streamopd/scheduler_teacher_admission_kv_rejections": float(self.teacher_admission_kv_rejections),
            "streamopd/scheduler_teacher_admission_wait_seconds": self.teacher_admission_wait_seconds,
            "streamopd/scheduler_teacher_admission_max_wait_seconds": self.teacher_admission_max_wait_seconds,
            "streamopd/scheduler_teacher_admission_waited_sessions": float(self.teacher_admission_waited_sessions),
            "streamopd/scheduler_terminal_trajectories": float(self.terminal_trajectories),
            "streamopd/scheduler_completed_teacher_trajectories": float(self.completed_teacher_trajectories),
            "streamopd/scheduler_all_rollouts_terminal_seconds": rollout_terminal_seconds,
            "streamopd/scheduler_all_teacher_complete_seconds": teacher_complete_seconds,
            "streamopd/scheduler_teacher_drain_after_rollout_seconds": teacher_drain_seconds,
            "streamopd/scheduler_first_teacher_started_seconds": first_teacher_started_seconds,
            "streamopd/scheduler_first_training_ready_seconds": first_training_ready_seconds,
            "streamopd/scheduler_first_training_started_seconds": first_training_started_seconds,
            "streamopd/scheduler_teacher_completed_at_first_training": float(self.teacher_completed_at_first_training),
            "streamopd/scheduler_rollouts_terminal_at_first_training": float(self.rollouts_terminal_at_first_training),
            "streamopd/scheduler_teacher_pending_at_first_training": float(self.teacher_pending_at_first_training),
            "streamopd/scheduler_teacher_busy_before_all_rollouts_terminal_seconds": (
                teacher_busy_before_rollout_terminal
            ),
            "streamopd/scheduler_teacher_busy_after_all_rollouts_terminal_seconds": max(
                0.0, teacher_busy_seconds - teacher_busy_before_rollout_terminal
            ),
            "streamopd/scheduler_teacher_busy_seconds": teacher_busy_seconds,
            "streamopd/scheduler_training_busy_seconds": training_busy_seconds,
            "streamopd/scheduler_pool_busy_seconds": pool_busy_seconds,
            "streamopd/scheduler_concurrent_busy_seconds": concurrent_busy_seconds,
            "streamopd/scheduler_busy_union_seconds": busy_union_seconds,
            "streamopd/scheduler_pool_idle_seconds": max(0.0, policy_seconds - busy_union_seconds),
            "streamopd/scheduler_pool_utilization": busy_union_seconds / max(policy_seconds, 1e-9),
            "streamopd/scheduler_aggregate_role_utilization": pool_busy_seconds / max(policy_seconds, 1e-9),
            "streamopd/scheduler_resources_overlap": float(self.resources_overlap),
            "streamopd/scheduler_train_launch_width": float(self.train_launch_width),
            "streamopd/scheduler_training_trajectories_started": float(self.training_trajectories_started),
            "streamopd/scheduler_lifecycle_seconds": policy_seconds,
        }
        self.policy_version = None
        return metrics

    @property
    def teacher_pending(self) -> int:
        return len(self.teacher_sessions)

    @property
    def teacher_drained(self) -> bool:
        return bool(
            self.terminal_trajectories == self.expected_trajectories
            and self.completed_teacher_trajectories == self.expected_trajectories
            and not self.teacher_sessions
            and not self.teacher_admission_waiters
        )

    def _check_version(self, policy_version: int) -> None:
        if self.policy_version is None:
            raise RuntimeError("StreamOPD scheduler has no active policy version")
        if int(policy_version) != self.policy_version:
            raise RuntimeError(
                f"StreamOPD scheduler policy mismatch: active={self.policy_version}, received={policy_version}"
            )
