# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Isolated vLLM decode/KV-export benchmark for StreamOPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.experimental.streamopd_kv.host_slot_pool import cleanup_host_kv_pools
from verl.experimental.streamopd_kv.protocol import TrajectoryKey
from verl.experimental.streamopd_kv.snapshot_io import load_vllm_snapshot, release_vllm_snapshot

EXPORT_MODES = ("no_export", "eos_host", "eos_triton", "incremental_triton")


def _load_prompts(
    dataset_path: str,
    tokenizer: Any,
    *,
    batch_size: int,
    max_prompt_length: int,
) -> list[list[int]]:
    table = pq.read_table(dataset_path, columns=["prompt"])
    prompts = table.column("prompt").to_pylist()
    if not prompts:
        raise RuntimeError(f"benchmark dataset has no prompts: {dataset_path}")
    encoded = []
    for index in range(batch_size):
        value = prompts[index % len(prompts)]
        messages = value if isinstance(value, list) else [{"role": "user", "content": str(value)}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        encoded.append(list(token_ids)[-max_prompt_length:])
    return encoded


def _aggregate_worker_stats(worker_stats: list[dict[str, float]]) -> dict[str, float]:
    keys = set().union(*(stats.keys() for stats in worker_stats))
    maxima = {"max_staging_wait_seconds", "max_outstanding_writes"}
    return {
        key: (max(float(stats.get(key, 0.0)) for stats in worker_stats) if key in maxima else 0.0) for key in keys
    } | {key: sum(float(stats.get(key, 0.0)) for stats in worker_stats) for key in keys - maxima}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/nasdata/Model/Qwen3-4B")
    parser.add_argument(
        "--dataset",
        default="/nasdata/Model/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet",
    )
    parser.add_argument("--mode", choices=EXPORT_MODES, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=896)
    parser.add_argument("--writer-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=0)
    parser.add_argument("--handoff-dir", default="/dev/shm/verl-streamopd-kv-export-benchmark")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_num_seqs < 1:
        raise ValueError("batch size and max_num_seqs must be positive")
    if args.max_total_tokens <= args.max_prompt_length:
        raise ValueError("max_total_tokens must exceed max_prompt_length")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = _load_prompts(
        args.dataset,
        tokenizer,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
    )
    run_id = uuid.uuid4().hex
    handoff_dir = Path(args.handoff_dir) / run_id
    engine_kwargs: dict[str, Any] = {}
    if args.mode != "no_export":
        handoff_dir.mkdir(parents=True, exist_ok=False)
        engine_kwargs["kv_transfer_config"] = {
            "kv_connector": "StreamOPDKVConnector",
            "kv_role": "kv_producer",
            "kv_connector_module_path": "verl.experimental.streamopd_kv.vllm_connector",
            "kv_connector_extra_config": {
                "streamopd_kv_handoff_dir": str(handoff_dir),
                "streamopd_kv_chunk_size": args.chunk_size,
                "streamopd_kv_export_strategy": args.mode,
                "streamopd_host_slot_count": args.batch_size,
                "streamopd_host_slot_tokens": args.max_total_tokens,
                "streamopd_writer_threads": args.writer_threads,
            },
        }
        engine_kwargs["worker_extension_cls"] = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_total_tokens + 1,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        **engine_kwargs,
    )

    def run_batch(policy_version: int) -> dict[str, Any]:
        request_ids = [f"{run_id}-{policy_version}-{index}" for index in range(args.batch_size)]
        sampling_params = []
        for request_id, prompt in zip(request_ids, prompts, strict=True):
            extra_args = None
            if args.mode != "no_export":
                extra_args = {
                    "kv_transfer_params": {
                        "streamopd_kv": True,
                        "trajectory_id": request_id,
                        "policy_version": policy_version,
                        "prompt_length": len(prompt),
                    }
                }
            sampling_params.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=args.max_total_tokens - len(prompt) + 1,
                    ignore_eos=args.ignore_eos,
                    extra_args=extra_args,
                )
            )

        started = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts],
            sampling_params,
            use_tqdm=False,
        )
        generation_seconds = time.perf_counter() - started
        total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        digester = hashlib.sha256()
        for output in outputs:
            for token_id in output.outputs[0].token_ids:
                digester.update(int(token_id).to_bytes(4, "little", signed=False))

        validation_seconds = 0.0
        if args.mode != "no_export":
            validation_started = time.perf_counter()
            for request_id, prompt, output in zip(request_ids, prompts, outputs, strict=True):
                transfer = output.kv_transfer_params
                if transfer is None:
                    raise RuntimeError(f"vLLM returned no KV transfer metadata for {request_id}")
                token_ids = (prompt + list(output.outputs[0].token_ids))[:-1]
                snapshot = load_vllm_snapshot(
                    transfer["streamopd_kv_path"],
                    key=TrajectoryKey(policy_version, request_id),
                    tp_rank=0,
                    expected_tp_size=1,
                    expected_token_ids=token_ids,
                    expected_prompt_length=len(prompt),
                )
                if snapshot.layers[0].length != len(token_ids):
                    raise RuntimeError("sealed StreamOPD KV snapshot has the wrong length")
                release_vllm_snapshot(transfer["streamopd_kv_path"])
            validation_seconds = time.perf_counter() - validation_started
        return {
            "generation_seconds": generation_seconds,
            "output_tokens": total_output_tokens,
            "output_digest": digester.hexdigest(),
            "validation_seconds": validation_seconds,
        }

    try:
        warmups = [run_batch(index) for index in range(args.warmup_batches)]
        if args.mode != "no_export" and warmups:
            llm.collective_rpc("reset_streamopd_kv_transfer_stats")
        measured = run_batch(args.warmup_batches)
        transfer_stats = {}
        if args.mode != "no_export":
            transfer_stats = _aggregate_worker_stats(llm.collective_rpc("get_streamopd_kv_transfer_stats"))
    finally:
        llm.llm_engine.engine_core.shutdown()
    released_host_bytes = 0
    if args.mode != "no_export":
        released_host_bytes = cleanup_host_kv_pools(str(handoff_dir))
        handoff_dir.rmdir()
    result = {
        "mode": args.mode,
        "model": args.model,
        "batch_size": args.batch_size,
        "max_num_seqs": args.max_num_seqs,
        "max_total_tokens": args.max_total_tokens,
        "prompt_tokens": sum(map(len, prompts)),
        "output_tokens": measured["output_tokens"],
        "generation_seconds": measured["generation_seconds"],
        "output_tokens_per_second": measured["output_tokens"] / measured["generation_seconds"],
        "output_digest": measured["output_digest"],
        "validation_seconds_excluded": measured["validation_seconds"],
        "warmup_generation_seconds": [item["generation_seconds"] for item in warmups],
        "released_host_gib": released_host_bytes / 2**30,
        "transfer": transfer_stats,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
