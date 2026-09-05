# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Exercise the streaming service through a fake engine's async input/output API."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.streamopd_kv.vllm_teacher import StreamingTeacherServer


class FakeEngine:
    def __init__(self, fault=None):
        self.fault = fault
        self.active = set()
        self.started = asyncio.Event()

    async def generate(self, prompt, sampling_params, request_id):
        self.active.add(request_id)
        self.started.set()
        tokens = []
        try:
            async for fragment in prompt:
                if self.fault == "engine":
                    raise RuntimeError("engine failed")
                if self.fault == "cancel":
                    # Yield a stale DELTA then await the next input. Cancelling
                    # the scoring RPC must close the input stream to free KV.
                    yield SimpleNamespace(prompt_token_ids=[])
                    continue
                new_tokens = fragment.prompt["prompt_token_ids"]
                rows = [{token: SimpleNamespace(rank=1, logprob=-0.5)} for token in new_tokens]
                if self.fault == "rank":
                    rows[-1] = {}
                if not tokens:
                    prompt_rows = [None] + rows[:-1]
                else:
                    prompt_rows = rows[:-1]
                tokens.extend(new_tokens)
                yield SimpleNamespace(
                    prompt_token_ids=list(tokens),
                    prompt_logprobs=prompt_rows,
                    outputs=[SimpleNamespace(logprobs=[rows[-1]])],
                )
        finally:
            self.active.remove(request_id)


@pytest.mark.asyncio
async def test_teacher_service_retains_independent_sessions_and_closes_at_eos():
    engine = FakeEngine()
    server = StreamingTeacherServer(engine)
    params = {"prompt_logprobs": 1}
    for request_id, tokens in [("a", [1, 2]), ("b", [7, 8])]:
        result = await server.stream_teacher_chunk(tokens, params, request_id, False)
        assert result["prompt_ids"].tolist() == [[token] for token in tokens]
        assert result["prompt_logprobs"].dtype == torch.float32
    assert engine.active == {"a", "b"}
    result = await server.stream_teacher_chunk([3], params, "a", True)
    assert result["prompt_ids"].tolist() == [[3]]
    assert engine.active == {"b"}
    result = await server.stream_teacher_chunk([], params, "b", True)
    assert result["prompt_ids"].shape == (0, 1)
    assert not engine.active
    assert not server._teacher_input_streams


@pytest.mark.asyncio
@pytest.mark.parametrize(("fault", "message"), [("engine", "engine failed"), ("rank", "incomplete teacher top-k")])
async def test_teacher_service_releases_kv_when_engine_or_artifact_validation_fails(fault, message):
    engine = FakeEngine(fault)
    server = StreamingTeacherServer(engine)
    with pytest.raises(RuntimeError, match=message):
        await server.stream_teacher_chunk([1, 2], {"prompt_logprobs": 1}, "a", False)
    assert not engine.active
    assert not server._teacher_input_streams


@pytest.mark.asyncio
async def test_teacher_service_releases_kv_when_scoring_is_cancelled():
    engine = FakeEngine("cancel")
    server = StreamingTeacherServer(engine)
    task = asyncio.create_task(server.stream_teacher_chunk([1, 2], {"prompt_logprobs": 1}, "a", False))
    await engine.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not engine.active
    assert not server._teacher_input_streams
