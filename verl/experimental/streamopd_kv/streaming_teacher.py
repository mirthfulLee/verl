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
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import torch

from .protocol import CommittedTokenChunk, TrajectoryKey

TeacherScoreFn = Callable[[list[int], str, bool], Awaitable[tuple[torch.Tensor, torch.Tensor]]]


@dataclass
class _TeacherSession:
    prompt_ids: tuple[int, ...]
    response_ids: list[int] = field(default_factory=list)
    tail: asyncio.Task[None] | None = None
    terminal: bool = False
    id_fragments: list[torch.Tensor] = field(default_factory=list)
    logprob_fragments: list[torch.Tensor] = field(default_factory=list)
    scored_response_tokens: int = 0
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    admitted: bool = False
    stream_closed: bool = False
    pending: deque[tuple[int, bool]] = field(default_factory=deque)
    notifications: int = 0
    score_intervals: list[tuple[float, float]] = field(default_factory=list)


class _LocalTeacherAdmission:
    """Fallback admission control for tests without the global scheduler."""

    def __init__(self, max_trajectories: int, max_kv_tokens: int) -> None:
        self._slots = asyncio.Semaphore(max_trajectories)
        self._max_kv_tokens = max_kv_tokens
        self._active_kv_tokens = 0
        self._condition = asyncio.Condition()

    async def acquire(self, kv_tokens: int) -> None:
        await self._slots.acquire()
        try:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: (
                        self._active_kv_tokens == 0
                        if kv_tokens > self._max_kv_tokens
                        else self._active_kv_tokens + kv_tokens <= self._max_kv_tokens
                    )
                )
                self._active_kv_tokens += kv_tokens
        except BaseException:
            self._slots.release()
            raise

    async def release(self, kv_tokens: int) -> None:
        async with self._condition:
            self._active_kv_tokens -= kv_tokens
            if self._active_kv_tokens < 0:
                raise RuntimeError("teacher KV admission accounting became negative")
            self._condition.notify_all()
        self._slots.release()


class StreamingTeacherCoordinator:
    """Score increasing committed prefixes without delaying rollout to teacher completion."""

    def __init__(
        self,
        score: TeacherScoreFn,
        max_pending_chunks: int,
        scheduler=None,
        max_active_trajectories: int | None = None,
        max_active_kv_tokens: int = 65536,
        kv_page_size: int = 64,
        kv_reservation_tokens: int | None = None,
    ) -> None:
        if max_pending_chunks < 1:
            raise ValueError("max_pending_chunks must be positive")
        max_active_trajectories = max_active_trajectories or max_pending_chunks
        if max_active_trajectories < 1 or max_active_kv_tokens < 1 or kv_page_size < 1:
            raise ValueError("teacher trajectory, KV-token, and page limits must be positive")
        self._score = score
        self._sessions: dict[TrajectoryKey, _TeacherSession] = {}
        self._scheduler = scheduler
        self._max_active_trajectories = max_active_trajectories
        self._max_active_kv_tokens = max_active_kv_tokens
        self._kv_page_size = kv_page_size
        self._kv_reservation_tokens = kv_reservation_tokens
        if kv_reservation_tokens is not None and kv_reservation_tokens < 1:
            raise ValueError("teacher KV reservation must be positive")
        self._local_admission = _LocalTeacherAdmission(max_active_trajectories, max_active_kv_tokens)

    async def _schedule(self, method: str, policy_version: int, *args) -> None:
        if self._scheduler is not None:
            await getattr(self._scheduler, method).remote(policy_version, *args)

    async def _admit_session(self, key: TrajectoryKey, reservation: int) -> None:
        session_id = f"v{key.policy_version}-{key.trajectory_id}"
        try:
            if self._scheduler is not None:
                await self._scheduler.wait_teacher_session_admitted.remote(
                    key.policy_version,
                    session_id,
                    reservation,
                    self._max_active_trajectories,
                    self._max_active_kv_tokens,
                )
            await self._local_admission.acquire(reservation)
        except BaseException:
            if self._scheduler is not None:
                await self._schedule("teacher_session_admission_cancelled", key.policy_version, session_id)
            raise

    async def _release_session(self, key: TrajectoryKey, reservation: int) -> None:
        session_id = f"v{key.policy_version}-{key.trajectory_id}"
        try:
            if self._scheduler is not None:
                await self._schedule("teacher_session_released", key.policy_version, session_id)
        finally:
            await self._local_admission.release(reservation)

    async def submit(self, chunk: CommittedTokenChunk) -> None:
        session = self._sessions.get(chunk.key)
        if session is None:
            if chunk.start != 0 or not chunk.prompt_ids:
                raise RuntimeError("the first teacher chunk must include the prompt and start at completion token 0")
            session = _TeacherSession(prompt_ids=chunk.prompt_ids)
            self._sessions[chunk.key] = session
        elif chunk.prompt_ids:
            raise RuntimeError("only the first teacher chunk may include prompt ids")
        if session.terminal:
            raise RuntimeError(f"teacher session {chunk.key} is already terminal")
        if chunk.start != len(session.response_ids):
            raise RuntimeError(
                f"non-contiguous completion chunks for {chunk.key}: "
                f"expected {len(session.response_ids)}, got {chunk.start}"
            )

        session.response_ids.extend(chunk.token_ids)
        session.terminal = chunk.terminal
        session.notifications += 1
        if chunk.terminal:
            await self._schedule(
                "teacher_trajectory_terminal_submitted",
                chunk.key.policy_version,
                session.notifications,
            )
        session.pending.append((len(session.response_ids), chunk.terminal))
        session.updated.set()
        if session.tail is None:
            session.tail = asyncio.create_task(
                self._run_session(chunk.key, session),
                name=f"streamopd-teacher-{chunk.key.trajectory_id}",
            )

    async def _run_session(self, key: TrajectoryKey, session: _TeacherSession) -> None:
        reservation = self._kv_reservation_tokens or self._max_active_kv_tokens
        try:
            while True:
                await session.updated.wait()
                session.updated.clear()
                while True:
                    if not session.pending:
                        break
                    target_response_tokens, terminal = session.pending.popleft()
                    requested_kv_tokens = len(session.prompt_ids) + target_response_tokens
                    if requested_kv_tokens > reservation:
                        raise RuntimeError(
                            f"teacher session {key} exceeded its {reservation}-token KV reservation: "
                            f"{requested_kv_tokens}"
                        )
                    if not session.admitted:
                        await self._admit_session(key, reservation)
                        session.admitted = True
                    # The live-session reservation is a stable upper bound, so
                    # no second, per-fragment admission loop is needed.
                    while session.pending:
                        target_response_tokens, terminal = session.pending.popleft()
                    if target_response_tokens < session.scored_response_tokens:
                        raise RuntimeError(
                            f"teacher session {key} received an already-scored chunk ending at {target_response_tokens}"
                        )
                    request_id = f"streamopd-teacher-v{key.policy_version}-{key.trajectory_id}"
                    if target_response_tokens == session.scored_response_tokens:
                        if terminal and not session.stream_closed:
                            score_started = time.perf_counter()
                            await self._score([], request_id, True)
                            session.score_intervals.append((score_started, time.perf_counter()))
                            session.stream_closed = True
                            return
                        continue
                    scored_response_tokens = session.scored_response_tokens
                    if scored_response_tokens == 0:
                        fragment = list(session.prompt_ids) + list(session.response_ids[:target_response_tokens])
                    else:
                        fragment = list(session.response_ids[scored_response_tokens:target_response_tokens])
                    score_started = time.perf_counter()
                    try:
                        teacher_ids, teacher_logprobs = await self._score(fragment, request_id, terminal)
                    finally:
                        session.score_intervals.append((score_started, time.perf_counter()))
                    expected_rows = len(fragment)
                    if teacher_ids.shape[0] != expected_rows or teacher_logprobs.shape[0] != expected_rows:
                        raise RuntimeError(
                            f"teacher returned incomplete fragment for {key}: "
                            f"ids={teacher_ids.shape[0]}, logprobs={teacher_logprobs.shape[0]}, "
                            f"expected={expected_rows}"
                        )
                    session.id_fragments.append(teacher_ids.detach().cpu())
                    session.logprob_fragments.append(teacher_logprobs.detach().cpu())
                    session.scored_response_tokens = target_response_tokens
                    if terminal:
                        session.stream_closed = True
                        return
        finally:
            if session.admitted:
                await self._release_session(key, reservation)
                session.admitted = False
            if session.stream_closed:
                await self._schedule("teacher_trajectory_completed", key.policy_version, session.score_intervals)

    async def result(self, key: TrajectoryKey, required_completion_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        session = self._sessions.get(key)
        if session is None or not session.terminal or session.tail is None:
            raise RuntimeError(f"teacher session {key} has not reached a terminal chunk")
        if len(session.response_ids) != required_completion_tokens:
            raise RuntimeError(
                f"teacher/student completion length mismatch for {key}: "
                f"streamed={len(session.response_ids)}, required={required_completion_tokens}"
            )
        await session.tail
        required_tokens = len(session.prompt_ids) + required_completion_tokens
        if not session.id_fragments or not session.logprob_fragments:
            raise RuntimeError(f"teacher session {key} produced no artifacts")
        teacher_ids = torch.cat(session.id_fragments)
        teacher_logprobs = torch.cat(session.logprob_fragments)
        if teacher_ids.shape[0] != required_tokens or teacher_logprobs.shape[0] != required_tokens:
            raise RuntimeError(f"terminal teacher prefix length does not match {key}")
        result = teacher_ids, teacher_logprobs
        del self._sessions[key]
        return result

    def invalidate_version(self, policy_version: int) -> int:
        keys = [key for key in self._sessions if key.policy_version == policy_version]
        for key in keys:
            task = self._sessions[key].tail
            if task is not None and not task.done():
                task.cancel()
            del self._sessions[key]
        return len(keys)
