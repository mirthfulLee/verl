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
        self.training_active = 0
        self.teacher_chunks = 0
        self.training_units = 0
        self.max_teacher_pending = 0
        self.policy_started_at = 0.0

    def begin_policy(self, policy_version: int) -> None:
        if self.teacher_pending or self.training_active:
            raise RuntimeError(
                "cannot begin a StreamOPD policy version while work is active: "
                f"teacher_pending={self.teacher_pending}, training_active={self.training_active}"
            )
        self.policy_version = int(policy_version)
        self.teacher_chunks = 0
        self.training_units = 0
        self.max_teacher_pending = 0
        self.policy_started_at = time.perf_counter()

    def teacher_enqueued(self, policy_version: int) -> None:
        self._check_version(policy_version)
        self.teacher_queued += 1
        self.teacher_chunks += 1
        self.max_teacher_pending = max(self.max_teacher_pending, self.teacher_pending)

    def teacher_started(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if self.training_active:
            raise RuntimeError("teacher forward cannot overlap StreamOPD reverse training")
        if self.teacher_queued < 1:
            raise RuntimeError("teacher_started without a queued StreamOPD chunk")
        self.teacher_queued -= 1
        self.teacher_active += 1

    def try_teacher_started(self, policy_version: int) -> bool:
        self._check_version(policy_version)
        if self.training_active:
            return False
        self.teacher_started(policy_version)
        return True

    def teacher_finished(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if self.teacher_active < 1:
            raise RuntimeError("teacher_finished without an active StreamOPD chunk")
        self.teacher_active -= 1

    def teacher_cancelled(self, policy_version: int) -> None:
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

    def try_training_started(self, policy_version: int, teacher_queue_threshold: int) -> bool:
        self._check_version(policy_version)
        if teacher_queue_threshold < 0:
            raise ValueError("teacher_queue_threshold must be non-negative")
        if self.teacher_active or self.teacher_queued > teacher_queue_threshold:
            return False
        self.training_started(policy_version)
        return True

    def training_finished(self, policy_version: int) -> None:
        self._check_version(policy_version)
        if not self.training_active:
            raise RuntimeError("training_finished without an active StreamOPD training unit")
        self.training_active = 0

    def snapshot(self) -> dict[str, int | float | None]:
        return {
            "policy_version": self.policy_version,
            "teacher_queued": self.teacher_queued,
            "teacher_active": self.teacher_active,
            "teacher_pending": self.teacher_pending,
            "training_active": self.training_active,
            "teacher_chunks": self.teacher_chunks,
            "training_units": self.training_units,
            "max_teacher_pending": self.max_teacher_pending,
        }

    def end_policy(self, policy_version: int) -> dict[str, float]:
        self._check_version(policy_version)
        if self.teacher_pending or self.training_active:
            raise RuntimeError(
                "StreamOPD policy barrier reached with unfinished work: "
                f"teacher_pending={self.teacher_pending}, training_active={self.training_active}"
            )
        metrics = {
            "streamopd/scheduler_teacher_chunks": float(self.teacher_chunks),
            "streamopd/scheduler_training_units": float(self.training_units),
            "streamopd/scheduler_max_teacher_pending": float(self.max_teacher_pending),
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
