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

    def __init__(self) -> None:
        self.policy_version: int | None = None
        self.teacher_queued = 0
        self.teacher_active = 0
        self.teacher_active_kv_tokens = 0
        self.teacher_sessions: dict[str, int] = {}
        self.teacher_session_kv_tokens = 0
        self.training_active = 0
        self.training_waiters: list[int] = []
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
        self.policy_started_at = 0.0

    def begin_policy(self, policy_version: int) -> None:
        if self.teacher_pending or self.training_active or self.teacher_sessions or self.training_waiters:
            raise RuntimeError(
                "cannot begin a StreamOPD policy version while work is active: "
                f"teacher_pending={self.teacher_pending}, training_active={self.training_active}, "
                f"training_waiters={len(self.training_waiters)}"
            )
        self.policy_version = int(policy_version)
        self.teacher_active_kv_tokens = 0
        self.teacher_session_kv_tokens = 0
        self.training_waiters = []
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
        self.policy_started_at = time.perf_counter()

    def teacher_notified(self, policy_version: int) -> None:
        self._check_version(policy_version)
        self.teacher_notifications += 1

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
        if self.training_active:
            raise RuntimeError("teacher forward cannot overlap StreamOPD reverse training")
        if self.teacher_queued < 1:
            raise RuntimeError("teacher_started without a queued StreamOPD chunk")
        self.teacher_queued -= 1
        if self.teacher_active == 0:
            self._teacher_busy_started_at = time.perf_counter()
        self.teacher_active += 1
        self.teacher_active_kv_tokens += kv_tokens
        self.max_teacher_active = max(self.max_teacher_active, self.teacher_active)

    def try_teacher_started(
        self,
        policy_version: int,
        kv_tokens: int = 0,
        max_active_trajectories: int = 2**31 - 1,
        max_active_kv_tokens: int = 2**63 - 1,
    ) -> bool:
        self._check_version(policy_version)
        if (
            self.training_active
            or (self.training_waiters and self.teacher_queued <= min(self.training_waiters))
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
        self.teacher_active -= 1
        self.teacher_active_kv_tokens -= kv_tokens
        if self.teacher_active == 0:
            self.teacher_busy_seconds += time.perf_counter() - self._teacher_busy_started_at
            self._teacher_busy_started_at = 0.0
        if self.teacher_active_kv_tokens < 0:
            raise RuntimeError("teacher active KV accounting became negative")

    def teacher_cancelled(self, policy_version: int, kv_tokens: int = 0) -> None:
        del kv_tokens
        self._check_version(policy_version)
        if self.teacher_queued < 1:
            raise RuntimeError("teacher_cancelled without a queued StreamOPD chunk")
        self.teacher_queued -= 1

    def training_started(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if self.teacher_active:
            raise RuntimeError("StreamOPD reverse training cannot overlap teacher forward")
        if self.training_active:
            raise RuntimeError("StreamOPD permits only one controller-level reverse-training unit at a time")
        self.training_active = 1
        self.training_units += 1
        self._training_busy_started_at = time.perf_counter()

    def training_waiting(self, policy_version: int, teacher_queue_threshold: int) -> None:
        """Register a ready reverse unit while it waits for pool ownership."""

        self._check_version(policy_version)
        if teacher_queue_threshold < 0:
            raise ValueError("teacher_queue_threshold must be non-negative")
        self.training_waiters.append(int(teacher_queue_threshold))
        self.max_training_waiters = max(self.max_training_waiters, len(self.training_waiters))

    def training_waiting_cancelled(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if not self.training_waiters:
            raise RuntimeError("training_waiting_cancelled without a waiting reverse unit")
        self.training_waiters.pop(0)

    def try_training_started(self, policy_version: int, teacher_queue_threshold: int) -> bool:
        self._check_version(policy_version)
        if teacher_queue_threshold < 0:
            raise ValueError("teacher_queue_threshold must be non-negative")
        if self.teacher_active or self.teacher_queued > teacher_queue_threshold:
            return False
        if self.training_waiters:
            self.training_waiters.pop(0)
        self.training_started(policy_version)
        return True

    def training_finished(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if not self.training_active:
            raise RuntimeError("training_finished without an active StreamOPD training unit")
        self.training_busy_seconds += time.perf_counter() - self._training_busy_started_at
        self._training_busy_started_at = 0.0
        self.training_active = 0

    def snapshot(self) -> dict[str, int | float | None]:
        return {
            "policy_version": self.policy_version,
            "teacher_queued": self.teacher_queued,
            "teacher_active": self.teacher_active,
            "teacher_active_kv_tokens": self.teacher_active_kv_tokens,
            "teacher_sessions": len(self.teacher_sessions),
            "teacher_session_kv_tokens": self.teacher_session_kv_tokens,
            "teacher_pending": self.teacher_pending,
            "training_active": self.training_active,
            "training_waiters": len(self.training_waiters),
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
        }

    def end_policy(self, policy_version: int) -> dict[str, float]:
        self._check_version(policy_version)
        if self.teacher_pending or self.training_active or self.teacher_sessions or self.training_waiters:
            raise RuntimeError(
                "StreamOPD policy barrier reached with unfinished work: "
                f"teacher_pending={self.teacher_pending}, teacher_sessions={len(self.teacher_sessions)}, "
                f"training_active={self.training_active}, training_waiters={len(self.training_waiters)}"
            )
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
            "streamopd/scheduler_teacher_busy_seconds": self.teacher_busy_seconds,
            "streamopd/scheduler_training_busy_seconds": self.training_busy_seconds,
            "streamopd/scheduler_pool_busy_seconds": self.teacher_busy_seconds + self.training_busy_seconds,
            "streamopd/scheduler_policy_seconds": time.perf_counter() - self.policy_started_at,
        }
        self.policy_version = None
        return metrics

    @property
    def teacher_pending(self) -> int:
        return self.teacher_queued + self.teacher_active

    def _check_version(self, policy_version: int) -> None:
        if self.policy_version is None:
            raise RuntimeError("StreamOPD scheduler has no active policy version")
        if int(policy_version) != self.policy_version:
            raise RuntimeError(
                f"StreamOPD scheduler policy mismatch: active={self.policy_version}, received={policy_version}"
            )
