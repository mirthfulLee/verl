# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import time


class StreamOPDTaskScheduler:
    """Central accounting and admission state for a shared teacher/trainer pool.

    Teacher inference remains asynchronous, but teacher forward and student
    backward are mutually exclusive on the shared teacher/trainer GPU pool.
    Rollout runs in its independent pool. Admission is atomic
    so polling clients cannot race between observing and starting work.
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
        self.scheduler_policy = "adaptive"
        self.train_launch_width = 1
        self.shared_launch_target = 1
        self.teacher_turn_grace_seconds = 0.05
        self.teacher_queued = 0
        self.teacher_active = 0
        self.teacher_active_kv_tokens = 0
        self.teacher_sessions: dict[str, int] = {}
        self.teacher_session_kv_tokens = 0
        self.expected_trajectories = 0
        self.terminal_trajectories = 0
        self.completed_teacher_trajectories = 0
        self.all_rollouts_terminal_at = 0.0
        self.all_teacher_completed_at = 0.0
        self.first_teacher_started_at = 0.0
        self.teacher_busy_before_all_rollouts_terminal = 0.0
        self.training_active = 0
        self.training_waiters: list[tuple[int, int]] = []
        self.ready_training_trajectories = 0
        self.max_training_waiters = 0
        self.teacher_chunks = 0
        self.teacher_notifications = 0
        self.training_units = 0
        self.max_teacher_pending = 0
        self.max_teacher_active = 0
        self.max_teacher_sessions = 0
        self.max_teacher_session_kv_tokens = 0
        self.teacher_busy_seconds = 0.0
        self.training_busy_seconds = 0.0
        self._teacher_busy_started_at = 0.0
        self._training_busy_started_at = 0.0
        self.first_training_ready_at = 0.0
        self.first_training_started_at = 0.0
        self.teacher_completed_at_first_training = 0
        self.rollouts_terminal_at_first_training = 0
        self.teacher_pending_at_first_training = 0
        self.concurrent_busy_seconds = 0.0
        self._concurrent_busy_started_at = 0.0
        self.training_units_since_teacher = 0
        self.last_training_finished_at = 0.0
        self.forced_teacher_turns = 0
        self.policy_started_at = 0.0

    def begin_policy(
        self,
        policy_version: int,
        expected_trajectories: int = 0,
        scheduler_policy: str = "adaptive",
        train_launch_width: int = 1,
    ) -> None:
        if self.teacher_pending or self.training_active or self.teacher_sessions or self.training_waiters:
            raise RuntimeError(
                "cannot begin a StreamOPD policy version while work is active: "
                f"teacher_pending={self.teacher_pending}, training_active={self.training_active}, "
                f"training_waiters={len(self.training_waiters)}"
            )
        self.policy_version = int(policy_version)
        if expected_trajectories < 0:
            raise ValueError("expected_trajectories must be non-negative")
        if scheduler_policy not in {"adaptive", "teacher_then_train"}:
            raise ValueError("invalid StreamOPD scheduler policy")
        if train_launch_width < 1:
            raise ValueError("train_launch_width must be positive")
        self.scheduler_policy = scheduler_policy
        self.train_launch_width = int(train_launch_width)
        self.expected_trajectories = int(expected_trajectories)
        self.shared_launch_target = min(
            self.expected_trajectories or 2 * self.train_launch_width,
            2 * self.train_launch_width,
        )
        self.terminal_trajectories = 0
        self.completed_teacher_trajectories = 0
        self.all_rollouts_terminal_at = 0.0
        self.all_teacher_completed_at = 0.0
        self.first_teacher_started_at = 0.0
        self.teacher_busy_before_all_rollouts_terminal = 0.0
        self.teacher_active_kv_tokens = 0
        self.teacher_session_kv_tokens = 0
        self.training_waiters = []
        self.ready_training_trajectories = 0
        self.max_training_waiters = 0
        self.teacher_chunks = 0
        self.teacher_notifications = 0
        self.training_units = 0
        self.max_teacher_pending = 0
        self.max_teacher_active = 0
        self.max_teacher_sessions = 0
        self.max_teacher_session_kv_tokens = 0
        self.teacher_busy_seconds = 0.0
        self.training_busy_seconds = 0.0
        self._teacher_busy_started_at = 0.0
        self._training_busy_started_at = 0.0
        self.first_training_ready_at = 0.0
        self.first_training_started_at = 0.0
        self.teacher_completed_at_first_training = 0
        self.rollouts_terminal_at_first_training = 0
        self.teacher_pending_at_first_training = 0
        self.concurrent_busy_seconds = 0.0
        self._concurrent_busy_started_at = 0.0
        self.training_units_since_teacher = 0
        self.last_training_finished_at = 0.0
        self.forced_teacher_turns = 0
        self.policy_started_at = time.perf_counter()

    def _transition_concurrency(self, was_concurrent: bool) -> None:
        is_concurrent = bool(self.teacher_active and self.training_active)
        if not was_concurrent and is_concurrent:
            self._concurrent_busy_started_at = time.perf_counter()
        elif was_concurrent and not is_concurrent:
            self.concurrent_busy_seconds += time.perf_counter() - self._concurrent_busy_started_at
            self._concurrent_busy_started_at = 0.0

    def teacher_notified(self, policy_version: int) -> None:
        self._check_version(policy_version)
        self.teacher_notifications += 1

    def teacher_trajectory_terminal_submitted(self, policy_version: int) -> None:
        self._check_version(policy_version)
        self.terminal_trajectories += 1
        if self.expected_trajectories and self.terminal_trajectories > self.expected_trajectories:
            raise RuntimeError("received more terminal Teacher trajectories than expected")
        if self.terminal_trajectories == self.expected_trajectories:
            self.all_rollouts_terminal_at = time.perf_counter()
            self.teacher_busy_before_all_rollouts_terminal = self.teacher_busy_seconds
            if self.teacher_active:
                self.teacher_busy_before_all_rollouts_terminal += (
                    self.all_rollouts_terminal_at - self._teacher_busy_started_at
                )

    def teacher_trajectory_completed(self, policy_version: int) -> None:
        self._check_version(policy_version)
        self.completed_teacher_trajectories += 1
        if self.expected_trajectories and self.completed_teacher_trajectories > self.expected_trajectories:
            raise RuntimeError("completed more Teacher trajectories than expected")
        if self.completed_teacher_trajectories == self.expected_trajectories:
            self.all_teacher_completed_at = time.perf_counter()

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
        if session_id in self.teacher_sessions:
            return True
        if len(self.teacher_sessions) >= max_active_trajectories:
            return False
        if self.teacher_sessions and self.teacher_session_kv_tokens + kv_reservation_tokens > max_active_kv_tokens:
            return False
        self.teacher_sessions[session_id] = kv_reservation_tokens
        self.teacher_session_kv_tokens += kv_reservation_tokens
        self.max_teacher_sessions = max(self.max_teacher_sessions, len(self.teacher_sessions))
        self.max_teacher_session_kv_tokens = max(
            self.max_teacher_session_kv_tokens,
            self.teacher_session_kv_tokens,
        )
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

    def teacher_enqueued(self, policy_version: int) -> None:
        self._check_version(policy_version)
        self.teacher_queued += 1
        self.teacher_chunks += 1
        self.max_teacher_pending = max(self.max_teacher_pending, self.teacher_pending)

    def teacher_started(self, policy_version: int, kv_tokens: int = 0) -> None:
        self._check_version(policy_version)
        if self.resources_overlap and self.training_active:
            raise RuntimeError("teacher forward cannot overlap StreamOPD reverse training")
        if self.teacher_queued < 1:
            raise RuntimeError("teacher_started without a queued StreamOPD chunk")
        self.teacher_queued -= 1
        if self.teacher_active == 0:
            self._teacher_busy_started_at = time.perf_counter()
            if not self.first_teacher_started_at:
                self.first_teacher_started_at = self._teacher_busy_started_at
        if self.resources_overlap and self.training_units_since_teacher:
            self.forced_teacher_turns += 1
            self.training_units_since_teacher = 0
        was_concurrent = bool(self.teacher_active and self.training_active)
        self.teacher_active += 1
        self.teacher_active_kv_tokens += kv_tokens
        self.max_teacher_active = max(self.max_teacher_active, self.teacher_active)
        self._transition_concurrency(was_concurrent)

    def try_teacher_started(
        self,
        policy_version: int,
        kv_tokens: int = 0,
        max_active_trajectories: int = 2**31 - 1,
        max_active_kv_tokens: int = 2**63 - 1,
    ) -> bool:
        self._check_version(policy_version)
        shared_ready = bool(
            not self.expected_trajectories
            or self.completed_teacher_trajectories >= self.shared_launch_target
            or self.terminal_trajectories == self.expected_trajectories
        )
        teacher_turn_due = bool(
            self.resources_overlap and self.training_units_since_teacher and not self.teacher_drained
        )
        adaptive_yield = bool(
            self.resources_overlap
            and self.scheduler_policy == "adaptive"
            and self.training_waiters
            and shared_ready
            and not teacher_turn_due
            and not self.should_drain_teacher_tail
            and (
                self.ready_training_trajectories >= self.train_launch_width
                or self.terminal_trajectories == self.expected_trajectories
            )
        )
        if (
            (self.resources_overlap and self.training_active)
            or adaptive_yield
            or self.teacher_active >= max_active_trajectories
            or (self.teacher_active and self.teacher_active_kv_tokens + kv_tokens > max_active_kv_tokens)
        ):
            return False
        self.teacher_started(policy_version, kv_tokens)
        return True

    def teacher_finished(self, policy_version: int, kv_tokens: int = 0) -> None:
        self._check_version(policy_version)
        if self.teacher_active < 1:
            raise RuntimeError("teacher_finished without an active StreamOPD chunk")
        was_concurrent = bool(self.teacher_active and self.training_active)
        self.teacher_active -= 1
        self.teacher_active_kv_tokens -= kv_tokens
        if self.teacher_active == 0:
            self.teacher_busy_seconds += time.perf_counter() - self._teacher_busy_started_at
            self._teacher_busy_started_at = 0.0
        if self.teacher_active_kv_tokens < 0:
            raise RuntimeError("teacher active KV accounting became negative")
        self._transition_concurrency(was_concurrent)

    def teacher_cancelled(self, policy_version: int, kv_tokens: int = 0) -> None:
        del kv_tokens
        self._check_version(policy_version)
        if self.teacher_queued < 1:
            raise RuntimeError("teacher_cancelled without a queued StreamOPD chunk")
        self.teacher_queued -= 1

    def training_started(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if self.resources_overlap and self.teacher_active:
            raise RuntimeError("StreamOPD reverse training cannot overlap teacher forward")
        if self.training_active:
            raise RuntimeError("StreamOPD permits only one controller-level reverse-training unit at a time")
        was_concurrent = bool(self.teacher_active and self.training_active)
        self.training_active = 1
        self.training_units += 1
        self._training_busy_started_at = time.perf_counter()
        self._transition_concurrency(was_concurrent)

    def training_waiting(
        self,
        policy_version: int,
        teacher_queue_threshold: int,
        trajectory_count: int = 1,
    ) -> None:
        """Register a ready reverse unit while it waits for pool ownership."""

        self._check_version(policy_version)
        if teacher_queue_threshold < 0:
            raise ValueError("teacher_queue_threshold must be non-negative")
        if trajectory_count < 1:
            raise ValueError("ready training trajectory count must be positive")
        if not self.first_training_ready_at:
            self.first_training_ready_at = time.perf_counter()
        self.training_waiters.append((int(teacher_queue_threshold), int(trajectory_count)))
        self.ready_training_trajectories += int(trajectory_count)
        self.max_training_waiters = max(self.max_training_waiters, len(self.training_waiters))

    def training_waiting_cancelled(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if not self.training_waiters:
            raise RuntimeError("training_waiting_cancelled without a waiting reverse unit")
        _, trajectory_count = self.training_waiters.pop(0)
        self.ready_training_trajectories -= trajectory_count

    def try_training_started(self, policy_version: int, teacher_queue_threshold: int) -> bool:
        self._check_version(policy_version)
        if teacher_queue_threshold < 0:
            raise ValueError("teacher_queue_threshold must be non-negative")
        if self.scheduler_policy == "adaptive" and self.should_drain_teacher_tail and not self.teacher_drained:
            return False
        if self.resources_overlap:
            if self.teacher_active:
                return False
            if self.scheduler_policy == "teacher_then_train":
                if not self.teacher_drained:
                    return False
            elif self.scheduler_policy == "adaptive":
                all_rollouts_terminal = bool(
                    self.expected_trajectories and self.terminal_trajectories == self.expected_trajectories
                )
                if (
                    self.expected_trajectories
                    and self.completed_teacher_trajectories < self.shared_launch_target
                    and not all_rollouts_terminal
                ):
                    return False
                teacher_turn_due = self.training_units_since_teacher and not self.teacher_drained
                if teacher_turn_due:
                    if self.teacher_queued > teacher_queue_threshold:
                        return False
                    if time.perf_counter() - self.last_training_finished_at < self.teacher_turn_grace_seconds:
                        return False
                if (
                    self.ready_training_trajectories < self.train_launch_width
                    and not all_rollouts_terminal
                    and self.teacher_queued > teacher_queue_threshold
                ):
                    return False
            elif self.teacher_queued > teacher_queue_threshold:
                return False
        if self.training_waiters:
            _, trajectory_count = self.training_waiters.pop(0)
            self.ready_training_trajectories -= trajectory_count
        if not self.first_training_started_at:
            self.first_training_started_at = time.perf_counter()
            self.teacher_completed_at_first_training = self.completed_teacher_trajectories
            self.rollouts_terminal_at_first_training = self.terminal_trajectories
            self.teacher_pending_at_first_training = self.teacher_pending
        self.training_started(policy_version)
        return True

    def training_finished(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if not self.training_active:
            raise RuntimeError("training_finished without an active StreamOPD training unit")
        was_concurrent = bool(self.teacher_active and self.training_active)
        self.training_busy_seconds += time.perf_counter() - self._training_busy_started_at
        self._training_busy_started_at = 0.0
        self.training_active = 0
        if self.resources_overlap:
            self.training_units_since_teacher += 1
            self.last_training_finished_at = time.perf_counter()
        self._transition_concurrency(was_concurrent)

    def snapshot(self) -> dict[str, int | float | str | bool | None]:
        return {
            "policy_version": self.policy_version,
            "teacher_queued": self.teacher_queued,
            "teacher_active": self.teacher_active,
            "teacher_active_kv_tokens": self.teacher_active_kv_tokens,
            "teacher_sessions": len(self.teacher_sessions),
            "teacher_session_kv_tokens": self.teacher_session_kv_tokens,
            "expected_trajectories": self.expected_trajectories,
            "terminal_trajectories": self.terminal_trajectories,
            "completed_teacher_trajectories": self.completed_teacher_trajectories,
            "posthoc_ready": self.posthoc_ready,
            "teacher_pending": self.teacher_pending,
            "training_active": self.training_active,
            "training_waiters": len(self.training_waiters),
            "ready_training_trajectories": self.ready_training_trajectories,
            "scheduler_policy": self.scheduler_policy,
            "train_launch_width": self.train_launch_width,
            "shared_launch_target": self.shared_launch_target,
            "resources_overlap": self.resources_overlap,
            "teacher_drained": self.teacher_drained,
            "max_training_waiters": self.max_training_waiters,
            "teacher_chunks": self.teacher_chunks,
            "teacher_notifications": self.teacher_notifications,
            "training_units": self.training_units,
            "max_teacher_pending": self.max_teacher_pending,
            "max_teacher_active": self.max_teacher_active,
            "max_teacher_sessions": self.max_teacher_sessions,
            "max_teacher_session_kv_tokens": self.max_teacher_session_kv_tokens,
            "teacher_busy_seconds": self.teacher_busy_seconds,
            "training_busy_seconds": self.training_busy_seconds,
            "training_units_since_teacher": self.training_units_since_teacher,
            "forced_teacher_turns": self.forced_teacher_turns,
        }

    def end_policy(self, policy_version: int) -> dict[str, float]:
        self._check_version(policy_version)
        if self.teacher_pending or self.training_active or self.teacher_sessions or self.training_waiters:
            raise RuntimeError(
                "StreamOPD policy barrier reached with unfinished work: "
                f"teacher_pending={self.teacher_pending}, teacher_sessions={len(self.teacher_sessions)}, "
                f"training_active={self.training_active}, training_waiters={len(self.training_waiters)}"
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
        first_teacher_started_seconds = (
            self.first_teacher_started_at - self.policy_started_at if self.first_teacher_started_at else 0.0
        )
        first_training_ready_seconds = (
            self.first_training_ready_at - self.policy_started_at if self.first_training_ready_at else 0.0
        )
        first_training_started_seconds = (
            self.first_training_started_at - self.policy_started_at if self.first_training_started_at else 0.0
        )
        policy_seconds = time.perf_counter() - self.policy_started_at
        pool_busy_seconds = self.teacher_busy_seconds + self.training_busy_seconds
        busy_union_seconds = pool_busy_seconds - self.concurrent_busy_seconds
        metrics = {
            "streamopd/scheduler_teacher_chunks": float(self.teacher_chunks),
            "streamopd/scheduler_teacher_notifications": float(self.teacher_notifications),
            "streamopd/scheduler_teacher_coalesced_fragments": float(self.teacher_notifications - self.teacher_chunks),
            "streamopd/scheduler_training_units": float(self.training_units),
            "streamopd/scheduler_max_training_waiters": float(self.max_training_waiters),
            "streamopd/scheduler_max_teacher_pending": float(self.max_teacher_pending),
            "streamopd/scheduler_max_teacher_active": float(self.max_teacher_active),
            "streamopd/scheduler_max_teacher_sessions": float(self.max_teacher_sessions),
            "streamopd/scheduler_max_teacher_session_kv_tokens": float(self.max_teacher_session_kv_tokens),
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
                self.teacher_busy_before_all_rollouts_terminal
            ),
            "streamopd/scheduler_teacher_busy_after_all_rollouts_terminal_seconds": max(
                0.0, self.teacher_busy_seconds - self.teacher_busy_before_all_rollouts_terminal
            ),
            "streamopd/scheduler_teacher_busy_seconds": self.teacher_busy_seconds,
            "streamopd/scheduler_training_busy_seconds": self.training_busy_seconds,
            "streamopd/scheduler_pool_busy_seconds": pool_busy_seconds,
            "streamopd/scheduler_concurrent_busy_seconds": self.concurrent_busy_seconds,
            "streamopd/scheduler_busy_union_seconds": busy_union_seconds,
            "streamopd/scheduler_pool_idle_seconds": max(0.0, policy_seconds - busy_union_seconds),
            "streamopd/scheduler_pool_utilization": busy_union_seconds / max(policy_seconds, 1e-9),
            "streamopd/scheduler_aggregate_role_utilization": pool_busy_seconds / max(policy_seconds, 1e-9),
            "streamopd/scheduler_resources_overlap": float(self.resources_overlap),
            "streamopd/scheduler_train_launch_width": float(self.train_launch_width),
            "streamopd/scheduler_shared_launch_target": float(self.shared_launch_target),
            "streamopd/scheduler_forced_teacher_turns": float(self.forced_teacher_turns),
            "streamopd/scheduler_policy_seconds": policy_seconds,
        }
        self.policy_version = None
        return metrics

    @property
    def teacher_pending(self) -> int:
        return self.teacher_queued + self.teacher_active

    @property
    def posthoc_ready(self) -> bool:
        return self.teacher_drained

    @property
    def teacher_drained(self) -> bool:
        return bool(
            self.expected_trajectories
            and self.terminal_trajectories == self.expected_trajectories
            and self.completed_teacher_trajectories == self.expected_trajectories
            and not self.teacher_pending
            and not self.teacher_sessions
        )

    @property
    def should_drain_teacher_tail(self) -> bool:
        if not self.expected_trajectories or self.terminal_trajectories != self.expected_trajectories:
            return False
        remaining = self.expected_trajectories - self.completed_teacher_trajectories
        return 0 < remaining <= self.train_launch_width

    def _check_version(self, policy_version: int) -> None:
        if self.policy_version is None:
            raise RuntimeError("StreamOPD scheduler has no active policy version")
        if int(policy_version) != self.policy_version:
            raise RuntimeError(
                f"StreamOPD scheduler policy mismatch: active={self.policy_version}, received={policy_version}"
            )
