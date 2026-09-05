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
    therefore indicates a backend contract violation and fails closed. Every
    complete Teacher-input chunk is submitted immediately;
    the first chunk budget includes the prompt, while later resumable chunks
    contain only new response tokens.
    """

    def __init__(
        self,
        key: TrajectoryKey,
        prompt_ids: Sequence[int],
        chunk_size: int,
        submit: Callable[[CommittedTokenChunk], Awaitable[None]],
        page_size: int = 1,
    ) -> None:
        if chunk_size < 1 or page_size < 1:
            raise ValueError("chunk_size and page_size must be positive")
        self.key = key
        self.prompt_ids = tuple(int(token_id) for token_id in prompt_ids)
        self.chunk_size = chunk_size
        self.submit = submit
        self.page_size = page_size
        self._accepted: tuple[int, ...] = ()
        self._emitted = 0
        self._terminal = False

    def _next_response_threshold(self) -> int:
        """Return the response tokens needed for the next teacher prefill.

        The first request contains both the prompt and response prefix, so
        its configured input budget includes ``len(prompt_ids)``. Resumable
        requests contain only new response tokens and therefore use the full
        chunk size as their delta budget.
        """

        if self._emitted == 0:
            return max(1, self.chunk_size - len(self.prompt_ids))
        return self.chunk_size

    async def observe(self, accepted_token_ids: Sequence[int], *, terminal: bool = False) -> None:
        if self._terminal:
            raise RuntimeError("received tokens after terminal rollout output")
        accepted = tuple(int(token_id) for token_id in accepted_token_ids)
        if len(accepted) < len(self._accepted) or accepted[: len(self._accepted)] != self._accepted:
            raise RuntimeError("rollout backend retracted or replaced committed token ids")
        self._accepted = accepted

        terminal_sent = False
        while True:
            threshold = self._next_response_threshold()
            available = len(self._accepted) - self._emitted
            aligned_threshold = ((threshold + self.page_size - 1) // self.page_size) * self.page_size
            required = threshold if terminal else aligned_threshold
            if available < required:
                break
            end = len(self._accepted) if terminal else self._emitted + aligned_threshold
            chunk_is_terminal = terminal and end == len(self._accepted)
            await self._emit(end, terminal=chunk_is_terminal)
            terminal_sent = terminal_sent or chunk_is_terminal
            if terminal:
                break

        if terminal and not terminal_sent:
            await self._emit(len(self._accepted), terminal=True)
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
