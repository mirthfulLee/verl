# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from .protocol import CommittedTokenChunk, TrajectoryKey


class CommittedChunkPublisher:
    """Convert cumulative backend outputs into accepted token chunks.

    vLLM and SGLang expose cumulative accepted output IDs while streaming.
    Rejected speculative tokens never appear in that list.  A non-prefix update
    therefore indicates a backend contract violation and fails closed. In the
    normal mode every complete chunk is submitted immediately; terminal-only
    catch-up is retained as an explicit ablation for older workloads.
    """

    def __init__(
        self,
        key: TrajectoryKey,
        prompt_ids: Sequence[int],
        chunk_size: int,
        submit: Callable[[CommittedTokenChunk], Awaitable[None]],
        terminal_only_after_initial: bool = False,
        page_size: int = 1,
    ) -> None:
        if chunk_size < 1 or page_size < 1:
            raise ValueError("chunk_size and page_size must be positive")
        self.key = key
        self.prompt_ids = tuple(int(token_id) for token_id in prompt_ids)
        self.chunk_size = chunk_size
        self.submit = submit
        self.terminal_only_after_initial = terminal_only_after_initial
        self.page_size = page_size
        self._accepted: tuple[int, ...] = ()
        self._emitted = 0
        self._terminal = False

    async def observe(self, accepted_token_ids: Sequence[int], *, terminal: bool = False) -> None:
        if self._terminal:
            raise RuntimeError("received tokens after terminal rollout output")
        accepted = tuple(int(token_id) for token_id in accepted_token_ids)
        if len(accepted) < len(self._accepted) or accepted[: len(self._accepted)] != self._accepted:
            raise RuntimeError("rollout backend retracted or replaced committed token ids")
        self._accepted = accepted

        terminal_sent = False
        while len(self._accepted) - self._emitted >= self.chunk_size and (
            not self.terminal_only_after_initial or self._emitted == 0
        ):
            end = len(self._accepted)
            if not terminal:
                end = end // self.page_size * self.page_size
            if end <= self._emitted:
                break
            chunk_is_terminal = terminal and end == len(self._accepted)
            await self._emit(end, terminal=chunk_is_terminal)
            terminal_sent = terminal_sent or chunk_is_terminal

        if terminal and self._emitted < len(self._accepted):
            await self._emit(len(self._accepted), terminal=True)
        elif terminal and not terminal_sent:
            await self.submit(
                CommittedTokenChunk(
                    key=self.key,
                    start=self._emitted,
                    token_ids=(),
                    terminal=True,
                    prompt_ids=self.prompt_ids if self._emitted == 0 else (),
                )
            )
        self._terminal = terminal

    async def _emit(self, end: int, *, terminal: bool) -> None:
        start = self._emitted
        await self.submit(
            CommittedTokenChunk(
                key=self.key,
                start=start,
                token_ids=self._accepted[start:end],
                terminal=terminal,
                prompt_ids=self.prompt_ids if start == 0 else (),
            )
        )
        self._emitted = end
