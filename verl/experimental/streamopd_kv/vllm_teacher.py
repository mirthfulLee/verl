# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Resumable Teacher sessions, instantiated only by the streaming RPC."""

import asyncio
from dataclasses import dataclass
from typing import Any

import torch
import vllm
from packaging import version
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind

from verl.utils.tokenizer import normalize_token_ids

try:
    from vllm.v1.engine.async_llm import StreamingInput
except ImportError:
    StreamingInput = None

_VLLM_VERSION = version.parse(vllm.__version__)


_TEACHER_STREAM_END = object()


@dataclass
class _TeacherInputStream:
    inputs: asyncio.Queue
    outputs: asyncio.Queue
    task: asyncio.Task
    lock: asyncio.Lock
    num_logprobs: int
    prompt_length: int = 0
    first_chunk: bool = True


def _ranked_logprob_row(logprobs: dict, num_logprobs: int) -> tuple[list[int], list[float]]:
    token_ids = [0] * num_logprobs
    values = [0.0] * num_logprobs
    populated = [False] * num_logprobs
    for token_id, item in logprobs.items():
        rank = int(item.rank)
        if 1 <= rank <= num_logprobs:
            token_ids[rank - 1] = int(token_id)
            values[rank - 1] = float(item.logprob)
            populated[rank - 1] = True
    if not all(populated):
        raise RuntimeError("vLLM returned an incomplete teacher top-k logprob row")
    return token_ids, values


class StreamingTeacherServer:
    """Own input queues, output validation and KV-session lifetime for one engine."""

    def __init__(self, engine):
        self.engine = engine
        self._teacher_input_streams: dict[str, _TeacherInputStream] = {}

    async def _run_teacher_input_stream(
        self,
        request_id: str,
        inputs: asyncio.Queue,
        outputs: asyncio.Queue,
        sampling_params: SamplingParams,
    ) -> None:
        async def input_stream():
            while True:
                value = await inputs.get()
                if value is _TEACHER_STREAM_END:
                    return
                yield value

        try:
            async for output in self.engine.generate(
                prompt=input_stream(),
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                await outputs.put(output)
        except BaseException as error:
            await outputs.put(error)
        finally:
            await outputs.put(_TEACHER_STREAM_END)

    async def _close_teacher_input_stream(self, request_id: str, session: _TeacherInputStream) -> None:
        await session.inputs.put(_TEACHER_STREAM_END)
        try:
            await session.task
        finally:
            self._teacher_input_streams.pop(request_id, None)

    async def stream_teacher_chunk(
        self,
        token_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        terminal: bool,
    ) -> dict[str, Any]:
        """Append one teacher input fragment while retaining its vLLM KV session."""
        if _VLLM_VERSION < version.parse("0.15.1") or StreamingInput is None:
            raise RuntimeError("teacher StreamingInput requires vLLM 0.15.1 or newer")
        token_ids = normalize_token_ids(token_ids)

        session = self._teacher_input_streams.get(request_id)
        if not token_ids:
            if not terminal or session is None:
                raise ValueError("an empty teacher input fragment is only valid as a terminal marker")
            async with session.lock:
                await self._close_teacher_input_stream(request_id, session)
            return {
                "prompt_ids": torch.empty((0, session.num_logprobs), dtype=torch.int32),
                "prompt_logprobs": torch.empty((0, session.num_logprobs), dtype=torch.float32),
            }
        if session is None:
            params = dict(sampling_params)
            num_logprobs = params.get("prompt_logprobs")
            if not isinstance(num_logprobs, int) or num_logprobs < 1:
                raise ValueError("resumable teacher scoring requires a positive top-k prompt_logprobs value")
            params.update(
                max_tokens=1,
                logprobs=num_logprobs,
                output_kind=RequestOutputKind.DELTA,
                detokenize=False,
                ignore_eos=True,
            )
            vllm_sampling_params = SamplingParams(**params)
            inputs: asyncio.Queue = asyncio.Queue()
            outputs: asyncio.Queue = asyncio.Queue()
            task = asyncio.create_task(
                self._run_teacher_input_stream(request_id, inputs, outputs, vllm_sampling_params),
                name=f"teacher-input-stream-{request_id}",
            )
            session = _TeacherInputStream(
                inputs=inputs,
                outputs=outputs,
                task=task,
                lock=asyncio.Lock(),
                num_logprobs=num_logprobs,
            )
            self._teacher_input_streams[request_id] = session

        async with session.lock:
            if terminal and request_id not in self._teacher_input_streams:
                raise RuntimeError(f"teacher input stream {request_id!r} is already closed")
            try:
                result = await self._score_fragment(request_id, session, token_ids)
            except BaseException:
                await self._close_teacher_input_stream(request_id, session)
                raise
            if terminal:
                await self._close_teacher_input_stream(request_id, session)
            return result

    async def _score_fragment(
        self, request_id: str, session: _TeacherInputStream, token_ids: list[int]
    ) -> dict[str, torch.Tensor]:
        expected_prompt_length = session.prompt_length + len(token_ids)
        await session.inputs.put(StreamingInput(prompt=TokensPrompt(prompt_token_ids=token_ids)))
        while True:
            output = await session.outputs.get()
            if output is _TEACHER_STREAM_END:
                raise RuntimeError(f"teacher input stream {request_id!r} ended before producing a chunk")
            if isinstance(output, BaseException):
                raise output
            output_prompt_length = len(output.prompt_token_ids or ())
            if output_prompt_length < expected_prompt_length:
                # DELTA output may contain an earlier chunk when several
                # resumable sessions are advanced in one scheduler tick.
                continue
            if output_prompt_length > expected_prompt_length:
                raise RuntimeError(
                    f"teacher StreamingInput advanced to {output_prompt_length} prompt tokens, "
                    f"expected {expected_prompt_length}"
                )
            break

        prompt_rows = list(output.prompt_logprobs or ())
        if session.first_chunk:
            if not prompt_rows or prompt_rows[0] is not None:
                raise RuntimeError("the first teacher StreamingInput output is missing its initial logprob row")
            prompt_rows = prompt_rows[1:]
        elif prompt_rows and prompt_rows[0] is None:
            raise RuntimeError("a resumed teacher StreamingInput unexpectedly restarted prompt logprobs")

        if not output.outputs or not output.outputs[0].logprobs:
            raise RuntimeError("teacher StreamingInput output is missing sampled top-k logprobs")
        sample_row = output.outputs[0].logprobs[-1]
        rows = [_ranked_logprob_row(row, session.num_logprobs) for row in prompt_rows]
        rows.append(_ranked_logprob_row(sample_row, session.num_logprobs))
        if len(rows) != len(token_ids):
            raise RuntimeError(f"teacher StreamingInput returned {len(rows)} artifact rows for {len(token_ids)} tokens")

        session.first_chunk = False
        session.prompt_length = expected_prompt_length
        return {
            "prompt_ids": torch.tensor([row[0] for row in rows], dtype=torch.int32),
            "prompt_logprobs": torch.tensor([row[1] for row in rows], dtype=torch.float32),
        }
