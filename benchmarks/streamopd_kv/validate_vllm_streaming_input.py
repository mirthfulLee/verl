# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate StreamOPD teacher artifacts produced by vLLM StreamingInput."""

import argparse
import asyncio
import time

import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs


async def validate(args: argparse.Namespace) -> None:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            dtype=args.dtype,
            enforce_eager=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            disable_log_stats=True,
            skip_tokenizer_init=args.skip_tokenizer_init,
            worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        )
    )
    patch_result = await engine.collective_rpc(method="enable_streaming_prompt_logprobs")

    # Importing the server after engine construction avoids initializing CUDA
    # in the parent before vLLM selects its worker multiprocessing method.
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.engine = engine
    server._is_teacher_model = True
    server._teacher_input_streams = {}
    sequence = list(range(100, 100 + args.sequence_length))
    prompt_length = args.prompt_length
    boundaries = [prompt_length + args.response_chunk, prompt_length + 2 * args.response_chunk, len(sequence)]
    params = {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": args.topk}

    async def one_session(session_id: int):
        streamed = []
        start = 0
        for index, end in enumerate(boundaries):
            if index == 0:
                start = 0
            streamed.append(
                await server.stream_teacher_chunk(
                    sequence[start:end],
                    params,
                    request_id=f"streamopd-validation-{session_id}",
                    terminal=end == len(sequence),
                )
            )
            start = end
        return streamed

    wall_start = time.perf_counter()
    all_streamed = await asyncio.gather(*(one_session(i) for i in range(args.num_sessions)))
    wall_seconds = time.perf_counter() - wall_start
    streamed = all_streamed[0]
    streamed_ids = torch.cat([item["prompt_ids"] for item in streamed])
    streamed_logprobs = torch.cat([item["prompt_logprobs"] for item in streamed])

    full_output = None
    async for output in engine.generate(
        TokensPrompt(prompt_token_ids=sequence),
        SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=args.topk),
        "full-validation",
    ):
        full_output = output
    assert full_output is not None
    full = {}
    extract_prompt_logprobs(full_output, args.topk, full, as_tensors=True)
    full_ids = full["prompt_ids"]
    full_logprobs = full["prompt_logprobs"]

    comparable = streamed_ids[:-1] == full_ids[:-1]
    shared_error = (
        float((streamed_logprobs[:-1][comparable] - full_logprobs[:-1][comparable]).abs().max())
        if comparable.any()
        else float("nan")
    )
    print(
        {
            "patch_result": patch_result,
            "num_sessions": args.num_sessions,
            "stream_wall_seconds": wall_seconds,
            "chunk_rows": [item["prompt_ids"].shape[0] for item in streamed],
            "session_count_after_eos": len(server._teacher_input_streams),
            "topk_id_mismatches_except_last": int((~comparable).sum()),
            "shared_topk_logprob_max_abs_error": shared_error,
        }
    )
    engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=3200)
    parser.add_argument("--prompt-length", type=int, default=100)
    parser.add_argument("--response-chunk", type=int, default=1066)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-sessions", type=int, default=1)
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    parsed = parser.parse_args()
    if parsed.max_num_batched_tokens is None:
        parsed.max_num_batched_tokens = parsed.max_model_len
    asyncio.run(validate(parsed))


if __name__ == "__main__":
    main()
