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

TeacherScoreFn = Callable[[list[int], str, int], Awaitable[tuple[torch.Tensor, torch.Tensor]]]


@dataclass
class _TeacherSession:
    prompt_ids: tuple[int, ...]
    response_ids: list[int] = field(default_factory=list)
    tail: asyncio.Task[None] | None = None
    terminal: bool = False
    latest_ids: torch.Tensor | None = None
    latest_logprobs: torch.Tensor | None = None


class StreamingTeacherCoordinator:
    """Score increasing committed prefixes without delaying rollout to teacher completion."""

    def __init__(
        self,
        score: TeacherScoreFn,
        max_pending_chunks: int,
        scheduler=None,
        scheduler_poll_interval_ms: int = 10,
    ) -> None:
        if max_pending_chunks < 1:
            raise ValueError("max_pending_chunks must be positive")
        self._score = score
        self._slots = asyncio.Semaphore(max_pending_chunks)
        self._sessions: dict[TrajectoryKey, _TeacherSession] = {}
        self._scheduler = scheduler
        self._scheduler_poll_seconds = scheduler_poll_interval_ms / 1000.0
        if self._scheduler_poll_seconds <= 0:
            raise ValueError("scheduler_poll_interval_ms must be positive")

    async def _schedule(self, method: str, policy_version: int) -> None:
        if self._scheduler is not None:
            await getattr(self._scheduler, method).remote(policy_version)

    async def _acquire_teacher(self, policy_version: int) -> None:
        if self._scheduler is None:
            return
        while not await self._scheduler.try_teacher_started.remote(policy_version):
            await asyncio.sleep(self._scheduler_poll_seconds)

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

        prior = session.tail
        session.response_ids.extend(chunk.token_ids)
        sequence = list(session.prompt_ids) + list(session.response_ids)
        # The prior prefix ends in a dummy row because there is no next token
        # yet. Re-fetch that boundary row when a later chunk makes it valid.
        artifact_start = 0 if chunk.start == 0 else len(session.prompt_ids) + chunk.start - 1
        session.terminal = chunk.terminal
        await self._schedule("teacher_enqueued", chunk.key.policy_version)
        try:
            await self._slots.acquire()
        except BaseException:
            await self._schedule("teacher_cancelled", chunk.key.policy_version)
            raise

        async def score_chunk() -> None:
            started = False
            try:
                if prior is not None:
                    await prior
                await self._acquire_teacher(chunk.key.policy_version)
                started = True
                if not chunk.token_ids and artifact_start:
                    return
                request_id = f"streamopd-teacher-v{chunk.key.policy_version}-{chunk.key.trajectory_id}"
                teacher_ids, teacher_logprobs = await self._score(sequence, request_id, artifact_start)
                expected_rows = len(sequence) - artifact_start
                if teacher_ids.shape[0] != expected_rows or teacher_logprobs.shape[0] != expected_rows:
                    raise RuntimeError(
                        f"teacher returned incomplete prefix for {chunk.key}: "
                        f"ids={teacher_ids.shape[0]}, logprobs={teacher_logprobs.shape[0]}, expected={expected_rows}"
                    )
                # The last prompt-logprob row is a dummy because it has no
                # next-token target yet. A later prefix replaces that row, so
                # retaining only the latest full result is required for exact
                # next-token alignment across chunk boundaries.
                if artifact_start:
                    if session.latest_ids is None or session.latest_logprobs is None:
                        raise RuntimeError(f"teacher session {chunk.key} is missing its prior artifact prefix")
                    session.latest_ids = torch.cat((session.latest_ids[:artifact_start], teacher_ids.detach().cpu()))
                    session.latest_logprobs = torch.cat(
                        (session.latest_logprobs[:artifact_start], teacher_logprobs.detach().cpu())
                    )
                else:
                    session.latest_ids = teacher_ids.detach().cpu()
                    session.latest_logprobs = teacher_logprobs.detach().cpu()
            finally:
                if started:
                    await self._schedule("teacher_finished", chunk.key.policy_version)
                else:
                    await self._schedule("teacher_cancelled", chunk.key.policy_version)
                self._slots.release()

        session.tail = asyncio.create_task(score_chunk(), name=f"streamopd-teacher-{chunk.key.trajectory_id}")

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
