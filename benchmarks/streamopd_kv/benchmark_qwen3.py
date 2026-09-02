# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.experimental.streamopd_kv import Qwen3ReverseTrainer
from verl.experimental.streamopd_kv.reverse_attention import ReverseKVSlotPool
from verl.experimental.streamopd_kv.snapshot_io import HostSlotLayerKV

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def synchronize(*devices: torch.device) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def topk_distribution(logits: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    values, ids = torch.topk(logprobs, k=topk, dim=-1)
    return ids.cpu(), values.cpu()


@torch.no_grad()
def rollout(model, prompt_ids: torch.Tensor, response_tokens: int, chunk_size: int, submit=None):
    output = model(input_ids=prompt_ids, use_cache=True)
    cache = output.past_key_values
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    pending = [next_token]
    for _ in range(1, response_tokens):
        output = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        pending.append(next_token)
        if submit is not None and len(pending) == chunk_size:
            submit(torch.cat(pending, dim=1).cpu())
            pending.clear()
    if submit is not None and pending:
        submit(torch.cat(pending, dim=1).cpu())
    return torch.cat(generated, dim=1), cache


def _teacher_worker(
    model_path: str,
    dtype_name: str,
    device_name: str,
    topk: int,
    commands: mp.Queue,
    results: mp.Queue,
) -> None:
    try:
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=DTYPES[dtype_name]).to(device).eval()
        results.put(("ready", None))
        prompt_ids = None
        cache = None
        outputs = []
        while True:
            operation, payload = commands.get()
            if operation == "stop":
                return
            if operation == "full":
                input_ids = torch.tensor(payload, dtype=torch.long, device=device).unsqueeze(0)
                with torch.no_grad():
                    result = topk_distribution(model(input_ids=input_ids, use_cache=False).logits, topk)
                results.put(("result", (result[0].numpy(), result[1].numpy())))
            elif operation == "stream_start":
                prompt_ids = torch.tensor(payload, dtype=torch.long, device=device).unsqueeze(0)
                cache = None
                outputs = []
            elif operation == "stream_chunk":
                if prompt_ids is None:
                    raise RuntimeError("teacher stream was not initialized")
                chunk = torch.tensor(payload, dtype=torch.long, device=device).unsqueeze(0)
                model_input = torch.cat((prompt_ids, chunk), dim=1) if cache is None else chunk
                with torch.no_grad():
                    output = model(input_ids=model_input, past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    outputs.append(topk_distribution(output.logits, topk))
            elif operation == "stream_close":
                ids = torch.cat([item[0] for item in outputs], dim=1)
                logprobs = torch.cat([item[1] for item in outputs], dim=1)
                results.put(("result", (ids.numpy(), logprobs.numpy())))
                prompt_ids = None
                cache = None
                outputs = []
            else:
                raise RuntimeError(f"unknown teacher operation: {operation}")
    except BaseException as exc:
        results.put(("error", repr(exc)))


class TeacherProcess:
    """A separate CUDA process mirroring verl's independent Teacher Pool."""

    def __init__(self, model_path: str, dtype_name: str, device_name: str, topk: int, max_pending: int = 4) -> None:
        context = mp.get_context("spawn")
        self.commands = context.Queue(maxsize=max_pending)
        self.results = context.Queue(maxsize=1)
        self.process = context.Process(
            target=_teacher_worker,
            args=(model_path, dtype_name, device_name, topk, self.commands, self.results),
            daemon=True,
        )
        self.process.start()
        self._receive("ready")

    def _receive(self, expected: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            kind, payload = self.results.get(timeout=300)
        except queue.Empty as exc:
            raise RuntimeError("teacher process timed out") from exc
        if kind == "error":
            raise RuntimeError(f"teacher process failed: {payload}")
        if kind != expected:
            raise RuntimeError(f"unexpected teacher response: {kind}")
        if payload is None:
            return None
        return torch.from_numpy(payload[0]), torch.from_numpy(payload[1])

    def score_full(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.commands.put(("full", token_ids.flatten().cpu().tolist()))
        result = self._receive("result")
        assert result is not None
        return result

    def start_stream(self, prompt_ids: torch.Tensor) -> None:
        self.commands.put(("stream_start", prompt_ids.flatten().cpu().tolist()))

    def submit(self, tokens: torch.Tensor) -> None:
        self.commands.put(("stream_chunk", tokens.flatten().cpu().tolist()))

    def close_stream(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.commands.put(("stream_close", None))
        result = self._receive("result")
        assert result is not None
        return result

    def shutdown(self) -> None:
        self.commands.put(("stop", None))
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


def forward_kl_sum(
    logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    logits = logits[0, valid]
    host_valid = valid.cpu() if valid.device.type != "cpu" else valid
    ids = teacher_ids[0, host_valid].long().to(logits.device)
    teacher = teacher_logprobs[0, host_valid].to(logits.device).float()
    student = torch.log_softmax(logits, dim=-1).gather(-1, ids).float()
    return (teacher.exp() * (teacher - student)).sum(dim=-1).clamp_min(0.0).sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            "/data1/models/hf/datasets--open-r1--DAPO-Math-17k-Processed/snapshots/"
            "31dd309567e3da778038cc87d868b6097a3ccf68/en/train-00000-of-00001.parquet"
        ),
    )
    parser.add_argument("--student", default="/models/store/Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher", default="/models/store/Qwen/Qwen3-4B")
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--student-dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--teacher-dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--response-tokens", type=int, default=64)
    parser.add_argument("--token-chunk-size", type=int, default=16)
    parser.add_argument("--reverse-chunk-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--output", default="benchmarks/streamopd_kv/results/qwen3_stage.json")
    args = parser.parse_args()

    student_device = torch.device(args.student_device)
    dataset = Dataset.from_parquet(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    prompt = dataset[args.dataset_index]["prompt"]
    messages = [{"role": "user", "content": prompt}]
    encoded_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    )
    prompt_ids = (encoded_prompt["input_ids"] if hasattr(encoded_prompt, "keys") else encoded_prompt)[
        :, -args.prompt_tokens :
    ].to(student_device)

    teacher = TeacherProcess(args.teacher, args.teacher_dtype, args.teacher_device, args.topk)
    student = (
        AutoModelForCausalLM.from_pretrained(args.student, dtype=DTYPES[args.student_dtype]).to(student_device).eval()
    )
    synchronize(student_device)

    # Warm up generation and both teacher execution shapes without changing weights.
    with torch.no_grad():
        _, warm_cache = rollout(student, prompt_ids, args.response_tokens, args.token_chunk_size)
        teacher.score_full(prompt_ids)
        teacher.start_stream(prompt_ids)
        teacher.submit(torch.zeros((1, 1), dtype=torch.long))
        teacher.close_stream()
        del warm_cache
    synchronize(student_device)
    torch.cuda.empty_cache()

    start = time.perf_counter()
    baseline_tokens, baseline_cache = rollout(student, prompt_ids, args.response_tokens, args.token_chunk_size)
    synchronize(student_device)
    baseline_rollout_seconds = time.perf_counter() - start
    del baseline_cache
    full_sequence = torch.cat((prompt_ids, baseline_tokens), dim=1)
    trace_ids = full_sequence[:, :-1]
    valid = torch.arange(trace_ids.shape[1], device=student_device) >= prompt_ids.shape[1] - 1

    start = time.perf_counter()
    baseline_teacher_ids, baseline_teacher_logprobs = teacher.score_full(full_sequence)
    baseline_teacher_seconds = time.perf_counter() - start

    # Initialize the conventional backward kernels before measuring them.
    warm_logits = student(input_ids=trace_ids, use_cache=False).logits
    warm_loss = forward_kl_sum(
        warm_logits,
        baseline_teacher_ids[:, : trace_ids.shape[1]],
        baseline_teacher_logprobs[:, : trace_ids.shape[1]],
        valid,
    )
    warm_loss.backward()
    synchronize(student_device)
    student.zero_grad(set_to_none=True)
    del warm_logits, warm_loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(student_device)
    start = time.perf_counter()
    baseline_logits = student(input_ids=trace_ids, use_cache=False).logits
    baseline_loss = forward_kl_sum(
        baseline_logits,
        baseline_teacher_ids[:, : trace_ids.shape[1]],
        baseline_teacher_logprobs[:, : trace_ids.shape[1]],
        valid,
    )
    baseline_loss.backward()
    synchronize(student_device)
    baseline_train_seconds = time.perf_counter() - start
    baseline_train_peak_gb = torch.cuda.max_memory_allocated(student_device) / 2**30
    student.zero_grad(set_to_none=True)
    del baseline_logits
    torch.cuda.empty_cache()

    teacher.start_stream(prompt_ids)
    start = time.perf_counter()
    stream_tokens, rollout_cache = rollout(
        student,
        prompt_ids,
        args.response_tokens,
        args.token_chunk_size,
        submit=teacher.submit,
    )
    streamed_teacher_ids, streamed_teacher_logprobs = teacher.close_stream()
    synchronize(student_device)
    overlapped_rollout_teacher_seconds = time.perf_counter() - start
    if not torch.equal(stream_tokens, baseline_tokens):
        raise RuntimeError("baseline and streaming rollout produced different greedy trajectories")

    with torch.no_grad():
        streamed_full_logits = student(input_ids=trace_ids, use_cache=False).logits
        streamed_full_loss = forward_kl_sum(
            streamed_full_logits,
            streamed_teacher_ids[:, : trace_ids.shape[1]],
            streamed_teacher_logprobs[:, : trace_ids.shape[1]],
            valid,
        )
    del streamed_full_logits

    def reverse_loss(logits: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, int]:
        local_valid = valid[start:end]
        if not local_valid.any():
            return logits.sum() * 0.0, 0
        return (
            forward_kl_sum(
                logits,
                streamed_teacher_ids[:, start:end],
                streamed_teacher_logprobs[:, start:end],
                local_valid,
            ),
            int(local_valid.sum()),
        )

    page_size = min(64, args.reverse_chunk_size)
    token_capacity = (trace_ids.shape[1] + args.reverse_chunk_size - 1) // args.reverse_chunk_size
    token_capacity *= args.reverse_chunk_size
    layers = [HostSlotLayerKV(layer.keys, layer.values) for layer in rollout_cache.layers]
    first = layers[0]
    slots = ReverseKVSlotPool(
        batch_size=1,
        token_capacity=token_capacity,
        num_layers=len(layers),
        num_kv_heads=first.key.shape[1],
        head_dim=first.key.shape[-1],
        page_size=page_size,
        dtype=first.key.dtype,
        device=student_device,
    )

    def reverse_backward():
        slots.prepare_next([layers], [trace_ids.shape[1]], [token_capacity])
        slots.activate_next()
        result = Qwen3ReverseTrainer(student, args.reverse_chunk_size, page_size=page_size).backward(
            [trace_ids],
            [reverse_loss],
            state=slots.state(),
            on_depth_committed=slots.release_current_range,
        )
        slots.finish_current()
        return result

    # Reverse attention has a distinct backward graph, so warm it independently.
    reverse_backward()
    synchronize(student_device)
    student.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(student_device)

    start = time.perf_counter()
    reverse_result = reverse_backward()
    synchronize(student_device)
    reverse_train_seconds = time.perf_counter() - start
    reverse_train_peak_gb = torch.cuda.max_memory_allocated(student_device) / 2**30

    baseline_total = baseline_rollout_seconds + baseline_teacher_seconds + baseline_train_seconds
    stream_total = overlapped_rollout_teacher_seconds + reverse_train_seconds
    result = {
        "dataset": "open-r1/DAPO-Math-17k-Processed",
        "dataset_index": args.dataset_index,
        "student": args.student,
        "teacher": args.teacher,
        "student_dtype": args.student_dtype,
        "teacher_dtype": args.teacher_dtype,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "response_tokens": args.response_tokens,
        "topk": args.topk,
        "token_chunk_size": args.token_chunk_size,
        "reverse_chunk_size": args.reverse_chunk_size,
        "verl_sync_opd_stage": {
            "rollout_seconds": baseline_rollout_seconds,
            "teacher_seconds": baseline_teacher_seconds,
            "training_seconds": baseline_train_seconds,
            "total_seconds": baseline_total,
            "training_peak_gb": baseline_train_peak_gb,
            "loss_sum": baseline_loss.detach().item(),
        },
        "streamopd_kv": {
            "overlapped_rollout_teacher_seconds": overlapped_rollout_teacher_seconds,
            "reverse_training_seconds": reverse_train_seconds,
            "total_seconds": stream_total,
            "training_peak_gb": reverse_train_peak_gb,
            "loss_sum": reverse_result.loss_sum.item(),
        },
        "speedup": baseline_total / stream_total,
        "loss_abs_error": abs(baseline_loss.detach().item() - reverse_result.loss_sum.item()),
        "teacher_streaming_objective_abs_error": abs(
            baseline_loss.detach().item() - streamed_full_loss.detach().item()
        ),
        "reverse_abs_error_same_teacher": abs(streamed_full_loss.detach().item() - reverse_result.loss_sum.item()),
        "notes": (
            "The baseline is the stage-equivalent of verl V1 Native OPD with trainer_mode=sync, not an external "
            "implementation. Teacher scoring runs in a separate CUDA process; rollout KV remains in the student "
            "process and therefore has zero serialization handoff cost in this stage benchmark. "
            "Correctness-only full forward with streamed teacher artifacts is excluded from timing."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    teacher.shutdown()


if __name__ == "__main__":
    main()
