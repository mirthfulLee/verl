# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
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
    latest_ids: torch.Tensor | None = None
    latest_logprobs: torch.Tensor | None = None
    scored_response_tokens: int = 0
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    admitted: bool = False
    stream_closed: bool = False


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
        scheduler_poll_interval_ms: int = 10,
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
        self._scheduler_poll_seconds = scheduler_poll_interval_ms / 1000.0
        if self._scheduler_poll_seconds <= 0:
            raise ValueError("scheduler_poll_interval_ms must be positive")

    async def _schedule(self, method: str, policy_version: int, *args) -> None:
        if self._scheduler is not None:
            await getattr(self._scheduler, method).remote(policy_version, *args)

    async def _acquire_teacher(self, policy_version: int, kv_tokens: int) -> None:
        if self._scheduler is None:
            return
        while not await self._scheduler.try_teacher_started.remote(
            policy_version,
            kv_tokens,
            self._max_active_trajectories,
            self._max_active_kv_tokens,
        ):
            await asyncio.sleep(self._scheduler_poll_seconds)

    async def _release_teacher(self, policy_version: int, kv_tokens: int, *, started: bool) -> None:
        if self._scheduler is not None:
            await self._schedule("teacher_finished" if started else "teacher_cancelled", policy_version, kv_tokens)

    async def _admit_session(self, key: TrajectoryKey, reservation: int) -> None:
        globally_admitted = False
        session_id = f"v{key.policy_version}-{key.trajectory_id}"
        try:
            if self._scheduler is not None:
                while not await self._scheduler.try_teacher_session_admitted.remote(
                    key.policy_version,
                    session_id,
                    reservation,
                    self._max_active_trajectories,
                    self._max_active_kv_tokens,
                ):
                    await asyncio.sleep(self._scheduler_poll_seconds)
                globally_admitted = True
            await self._local_admission.acquire(reservation)
        except BaseException:
            if globally_admitted:
                await self._schedule("teacher_session_released", key.policy_version, session_id)
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
        session.updated.set()
        if session.tail is None:
            session.tail = asyncio.create_task(
                self._run_session(chunk.key, session),
                name=f"streamopd-teacher-{chunk.key.trajectory_id}",
            )

    def _target_response_tokens(self, session: _TeacherSession) -> int:
        available = len(session.response_ids)
        if session.terminal:
            return available
        aligned_total = (len(session.prompt_ids) + available) // self._kv_page_size * self._kv_page_size
        return max(0, aligned_total - len(session.prompt_ids))

    async def _run_session(self, key: TrajectoryKey, session: _TeacherSession) -> None:
        reservation = self._kv_reservation_tokens or self._max_active_kv_tokens
        try:
            while True:
                await session.updated.wait()
                session.updated.clear()
                while True:
                    target_response_tokens = self._target_response_tokens(session)
                    if target_response_tokens <= session.scored_response_tokens:
                        if session.terminal and session.scored_response_tokens == len(session.response_ids):
                            if not session.stream_closed:
                                request_id = f"streamopd-teacher-v{key.policy_version}-{key.trajectory_id}"
                                await self._score([], request_id, True)
                                session.stream_closed = True
                            return
                        break
                    scored_response_tokens = session.scored_response_tokens
                    if scored_response_tokens == 0:
                        fragment = list(session.prompt_ids) + list(session.response_ids[:target_response_tokens])
                    else:
                        fragment = list(session.response_ids[scored_response_tokens:target_response_tokens])
                    terminal = session.terminal and target_response_tokens == len(session.response_ids)
                    kv_tokens = len(session.prompt_ids) + target_response_tokens
                    if kv_tokens > reservation:
                        raise RuntimeError(
                            f"teacher session {key} exceeded its {reservation}-token KV reservation: {kv_tokens}"
                        )
                    if not session.admitted:
                        await self._admit_session(key, reservation)
                        session.admitted = True
                    await self._schedule("teacher_enqueued", key.policy_version)
                    started = False
                    acquired = False
                    try:
                        await self._acquire_teacher(key.policy_version, kv_tokens)
                        acquired = True
                        started = True
                        request_id = f"streamopd-teacher-v{key.policy_version}-{key.trajectory_id}"
                        teacher_ids, teacher_logprobs = await self._score(fragment, request_id, terminal)
                        expected_rows = len(fragment)
                        if teacher_ids.shape[0] != expected_rows or teacher_logprobs.shape[0] != expected_rows:
                            raise RuntimeError(
                                f"teacher returned incomplete fragment for {key}: "
                                f"ids={teacher_ids.shape[0]}, logprobs={teacher_logprobs.shape[0]}, "
                                f"expected={expected_rows}"
                            )
                        if session.latest_ids is not None and session.latest_logprobs is not None:
                            session.latest_ids = torch.cat((session.latest_ids, teacher_ids.detach().cpu()))
                            session.latest_logprobs = torch.cat(
                                (session.latest_logprobs, teacher_logprobs.detach().cpu())
                            )
                        else:
                            session.latest_ids = teacher_ids.detach().cpu()
                            session.latest_logprobs = teacher_logprobs.detach().cpu()
                        session.scored_response_tokens = target_response_tokens
                        if terminal:
                            session.stream_closed = True
                    finally:
                        if acquired:
                            await self._release_teacher(key.policy_version, kv_tokens, started=started)
                        elif self._scheduler is not None:
                            await self._schedule("teacher_cancelled", key.policy_version, kv_tokens)
        finally:
            if session.admitted:
                await self._release_session(key, reservation)
                session.admitted = False

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
        if session.latest_ids is None or session.latest_logprobs is None:
            raise RuntimeError(f"teacher session {key} produced no artifacts")
        if session.latest_ids.shape[0] != required_tokens or session.latest_logprobs.shape[0] != required_tokens:
            raise RuntimeError(f"terminal teacher prefix length does not match {key}")
        result = session.latest_ids, session.latest_logprobs
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
