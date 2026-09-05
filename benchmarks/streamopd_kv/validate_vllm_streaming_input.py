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
import os
import time

import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.engine.async_llm import AsyncLLM

from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs


async def validate(args: argparse.Namespace) -> None:
    additional_config = {}
    if args.exclusive_gpu_memory:
        additional_config = {
            "verl_exclusive_gpu_memory": True,
            "verl_streaming_teacher_logprobs": True,
        }
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            dtype=args.dtype,
            enforce_eager=not args.cuda_graphs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend="mp",
            enable_sleep_mode=args.exclusive_gpu_memory,
            additional_config=additional_config,
            logprobs_mode="processed_logprobs",
            max_logprobs=args.topk,
            disable_log_stats=True,
            skip_tokenizer_init=args.skip_tokenizer_init,
            attention_backend=(
                AttentionBackendEnum[args.attention_backend] if args.attention_backend is not None else None
            ),
            worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        )
    )
    patch_result = await engine.collective_rpc(method="enable_streaming_prompt_logprobs")

    from verl.experimental.streamopd_kv.vllm_teacher import StreamingTeacherServer

    server = StreamingTeacherServer(engine)
    sequence = list(range(100, 100 + args.sequence_length))
    prompt_length = args.prompt_length
    boundaries = list(range(prompt_length + args.response_chunk, len(sequence), args.response_chunk))
    boundaries.append(len(sequence))
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

    streamed_ids = streamed_ids[:-1]
    streamed_logprobs = streamed_logprobs[:-1]
    full_ids = full_ids[:-1]
    full_logprobs = full_logprobs[:-1]
    comparable = streamed_ids == full_ids
    pairwise_id_match = streamed_ids.unsqueeze(-1) == full_ids.unsqueeze(-2)
    shared_rows, streamed_ranks, full_ranks = pairwise_id_match.nonzero(as_tuple=True)
    shared_error = (
        float((streamed_logprobs[shared_rows, streamed_ranks] - full_logprobs[shared_rows, full_ranks]).abs().max())
        if shared_rows.numel()
        else float("nan")
    )

    def range_metrics(start: int, end: int) -> dict[str, object]:
        end = min(end, streamed_ids.shape[0], full_ids.shape[0])
        if start >= end:
            return {"start": start, "end": end, "rows": 0}
        streamed_range_ids = streamed_ids[start:end]
        full_range_ids = full_ids[start:end]
        streamed_range_logprobs = streamed_logprobs[start:end]
        full_range_logprobs = full_logprobs[start:end]
        pairwise = streamed_range_ids.unsqueeze(-1) == full_range_ids.unsqueeze(-2)
        rows, streamed_rank, full_rank = pairwise.nonzero(as_tuple=True)
        return {
            "start": start,
            "end": end,
            "rows": end - start,
            "top1_match_ratio": float((streamed_range_ids[:, 0] == full_range_ids[:, 0]).float().mean()),
            "topk_set_overlap_ratio": float(pairwise.any(dim=-1).float().mean()),
            "shared_logprob_max_abs_error": (
                float((streamed_range_logprobs[rows, streamed_rank] - full_range_logprobs[rows, full_rank]).abs().max())
                if rows.numel()
                else float("nan")
            ),
        }

    segment_metrics = []
    segment_start = 0
    for rows in [item["prompt_ids"].shape[0] for item in streamed]:
        segment_end = segment_start + rows
        segment = range_metrics(segment_start, segment_end)
        offset_matches = {}
        for offset in (-1, 0, 1):
            shifted_start = max(0, segment_start + offset)
            shifted_end = min(full_ids.shape[0], segment_end + offset)
            comparable_rows = min(
                segment_end - segment_start,
                streamed_ids.shape[0] - segment_start,
                shifted_end - shifted_start,
            )
            if comparable_rows:
                offset_matches[str(offset)] = float(
                    (
                        streamed_ids[segment_start : segment_start + comparable_rows, 0]
                        == full_ids[shifted_start : shifted_start + comparable_rows, 0]
                    )
                    .float()
                    .mean()
                )
        segment["top1_offset_match_ratio"] = offset_matches
        segment_metrics.append(segment)
        segment_start = segment_end

    boundary_metrics = [
        range_metrics(max(0, boundary - 4), boundary + 4)
        for boundary in torch.tensor([item["prompt_ids"].shape[0] for item in streamed]).cumsum(0).tolist()[:-1]
    ]
    mismatch_positions = (streamed_ids[:, 0] != full_ids[:, 0]).nonzero(as_tuple=False).flatten()
    sleep_memory = None
    if args.report_sleep_memory:
        if not args.exclusive_gpu_memory:
            raise ValueError("--report-sleep-memory requires --exclusive-gpu-memory")
        before = await engine.collective_rpc(method="get_device_memory_stats")
        sleep_started = time.perf_counter()
        await engine.sleep(level=2)
        sleep_seconds = time.perf_counter() - sleep_started
        after = await engine.collective_rpc(method="get_device_memory_stats")
        gib = 1024**3
        sleep_memory = {
            "sleep_seconds": sleep_seconds,
            "free_before_gib": [item["free_bytes"] / gib for item in before],
            "free_after_gib": [item["free_bytes"] / gib for item in after],
            "allocated_after_gib": [item["allocated_bytes"] / gib for item in after],
            "reserved_after_gib": [item["reserved_bytes"] / gib for item in after],
        }
    print(
        {
            "patch_result": patch_result,
            "num_sessions": args.num_sessions,
            "stream_wall_seconds": wall_seconds,
            "stream_tokens_per_second": args.num_sessions * args.sequence_length / wall_seconds,
            "batch_invariant": os.getenv("VLLM_BATCH_INVARIANT", "0") != "0",
            "attention_backend": args.attention_backend,
            "chunk_rows": [item["prompt_ids"].shape[0] for item in streamed],
            "session_count_after_eos": len(server._teacher_input_streams),
            "top1_id_match_ratio": float((streamed_ids[:, 0] == full_ids[:, 0]).float().mean()),
            "topk_set_overlap_ratio": float(pairwise_id_match.any(dim=-1).float().mean()),
            "topk_id_mismatches_except_last": int((~comparable).sum()),
            "shared_topk_logprob_max_abs_error": shared_error,
            "segment_metrics": segment_metrics,
            "boundary_metrics": boundary_metrics,
            "first_top1_mismatch_positions": mismatch_positions[:32].tolist(),
            "sleep_memory": sleep_memory,
        }
    )
    engine.shutdown()
    if args.assert_match:
        if server._teacher_input_streams:
            raise AssertionError("Teacher sessions remain open after EOS")
        for fragments in all_streamed:
            # The native teacher API has a dummy final row without a target.
            ids = torch.cat([item["prompt_ids"] for item in fragments])[:-1]
            logprobs = torch.cat([item["prompt_logprobs"] for item in fragments])[:-1]
            torch.testing.assert_close(ids, full_ids, rtol=0, atol=0)
            torch.testing.assert_close(logprobs, full_logprobs, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=4097)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=3200)
    parser.add_argument("--prompt-length", type=int, default=100)
    parser.add_argument("--response-chunk", type=int, default=1066)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-sessions", type=int, default=1)
    parser.add_argument("--assert-match", action="store_true", help="Fail on any supervised top-k/logprob mismatch")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--exclusive-gpu-memory", action="store_true")
    parser.add_argument("--report-sleep-memory", action="store_true")
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    parser.add_argument("--attention-backend", choices=sorted(AttentionBackendEnum.__members__))
    parsed = parser.parse_args()
    if parsed.max_num_batched_tokens is None:
        parsed.max_num_batched_tokens = parsed.max_model_len
    asyncio.run(validate(parsed))


if __name__ == "__main__":
    main()
