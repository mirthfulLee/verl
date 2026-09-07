# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sequential eight-GPU allocation screening and matched OPD pilot matrix."""

import argparse
import csv
import importlib.metadata
import io
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.streamopd_cf.summarize import summarize_run


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def wait_for_idle(devices):
    deadline = time.monotonic() + 120
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={devices}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        memory = [float(row[0]) for row in csv.reader(io.StringIO(output))]
        if len(memory) == len(devices.split(",")) and max(memory) < 1024:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPUs remain occupied; refusing to overlap benchmarks: {memory}")
        time.sleep(5)


def allocation_cases():
    for method in ("streamopd-cf", "streamopd-kv-dedicated"):
        for trainer, rollout, teacher in ((4, 2, 2), (2, 4, 2), (2, 2, 4)):
            yield dict(method=method, trainer=trainer, rollout=rollout, teacher=teacher, tp=1)
    for tp in (1, 2):
        for rollout, teacher in ((6, 2), (4, 4), (2, 6)):
            yield dict(method="streamopd-kv-union", trainer=8, rollout=rollout, teacher=teacher, tp=tp)


def name(case):
    base = f"{case['method']}-t{case['trainer']}-r{case['rollout']}-h{case['teacher']}-p{case['tp']}"
    return base + (f"-{case['tuning']}" if case.get("tuning") else "")


def chunk_cases():
    for batch, chunk in ((2, 2048), (1, 4096)):
        yield dict(
            method="streamopd-kv-union",
            trainer=8,
            rollout=6,
            teacher=2,
            tp=1,
            tuning=f"b{batch}-c{chunk}-reserve0",
            overrides=[
                f"distillation.streamopd_kv.reverse_batch_size={batch}",
                f"distillation.streamopd_kv.reverse_chunk_size={chunk}",
                f"distillation.streamopd_kv.reverse_chunk_min_size={chunk}",
                "distillation.streamopd_kv.reverse_slot_reserve_gib=0",
            ],
        )


def choose_kv_modes(results, minimum_gain=0.1):
    best = {}
    for method in ("streamopd-kv-dedicated", "streamopd-kv-union"):
        candidates = [r for r in results if r["status"] == "complete" and r["case"]["method"] == method]
        if not candidates:
            raise ValueError(f"KV placement screening has no complete result for {method}")
        best[method] = min(candidates, key=lambda r: r["step_seconds"])
    dedicated, shared = best["streamopd-kv-dedicated"], best["streamopd-kv-union"]
    step_gain = 1 - dedicated["step_seconds"] / shared["step_seconds"]
    token_gain = dedicated["response_tokens_per_second"] / shared["response_tokens_per_second"] - 1
    retained = dedicated["step_seconds"] <= shared["step_seconds"] * (1 - minimum_gain) and dedicated[
        "response_tokens_per_second"
    ] >= shared["response_tokens_per_second"] * (1 + minimum_gain)
    decision = {
        "minimum_gain": minimum_gain,
        "dedicated_step_reduction": step_gain,
        "dedicated_token_throughput_gain": token_gain,
        "retain_dedicated": retained,
        "default_kv": "streamopd-kv-dedicated" if retained else "streamopd-kv-union",
        "dedicated_case": dedicated["case"],
        "shared_case": shared["case"],
        "interpretation": "Practical single-step screening threshold, not statistical significance",
    }
    methods = ["streamopd-cf", "streamopd-kv-union"]
    if retained:
        methods.append("streamopd-kv-dedicated")
    return methods, decision


def comparison_variants(selected):
    variants = [(method, method) for method in selected]
    variants += [("verl-sync-opd-union", "streamopd-kv-union"), ("verl-async-opd", "streamopd-cf")]
    if "streamopd-kv-dedicated" in selected:
        variants.append(("verl-sync-opd-separate", "streamopd-kv-dedicated"))
    return variants


def qualified_kv_decision(reference_results, largest_results, minimum_gain):
    _, reference = choose_kv_modes(reference_results, minimum_gain)
    _, largest = choose_kv_modes(largest_results, minimum_gain)
    retained = reference["retain_dedicated"] or largest["retain_dedicated"]
    return {
        "reference_workload": reference,
        "largest_workload": largest,
        "retain_dedicated": retained,
        "default_kv": "streamopd-kv-dedicated" if retained else "streamopd-kv-union",
        "rule": "Retain dedicated KV and its sync control if either tested workload meets the practical gain threshold",
    }


def teacher_tp_for(allocation, teacher_model):
    if teacher_model == "Qwen3-14B":
        return allocation["tp"]
    # A shared Trainer also needs room beside sleeping Teacher processes.
    # Four-way TP is the conservative large-Teacher choice when the pool fits it.
    if allocation["method"] == "streamopd-kv-union" and allocation["teacher"] % 4 == 0:
        return 4
    return 2


def qualification_order(candidates):
    # Shared 8B training benefits from four-way Teacher sharding.
    # Prefer that topology before attempting tighter two-way shared pools.
    return sorted(
        candidates,
        key=lambda result: (
            -teacher_tp_for(result["case"], "Qwen3-32B")
            if result["case"]["method"] == "streamopd-kv-union"
            else -result["case"]["trainer"],
            result["step_seconds"],
        ),
    )


def comparison_case(allocation, method, student, teacher, tokens):
    case = dict(allocation, method=method, student=student, teacher_model=teacher, tokens=tokens)
    case["tp"] = teacher_tp_for(allocation, teacher)
    if method != allocation["method"]:
        case.pop("tuning", None)
        case.pop("overrides", None)
    return case


def comparison_settings(root):
    path = root / "comparison_scope.json"
    lengths = json.loads(path.read_text())["max_tokens"] if path.exists() else [4096]
    return itertools.product(("Qwen3-4B", "Qwen3-8B"), ("Qwen3-14B", "Qwen3-32B", "Qwen3-30B-A3B"), lengths)


def environment(args, case, directory):
    native_8b = case["method"].startswith("verl-sync-opd") and case["student"] == "Qwen3-8B"
    return {
        "METHOD": case["method"],
        "STUDENT_MODEL": str(args.models / case["student"]),
        "TEACHER_MODEL": str(args.models / case["teacher_model"]),
        "DATASET": str(args.dataset),
        "STUDENT_GPUS": str(case["trainer"]),
        "ROLLOUT_GPUS": str(case["rollout"]),
        "TEACHER_GPUS": str(case["teacher"]),
        "TEACHER_TP_SIZE": str(case["tp"]),
        "RESULT_DIR": str(directory),
        "BATCH_SIZE": str(args.batch),
        "TOTAL_TRAJECTORY_LENGTH": str(case["tokens"]),
        "TOTAL_TRAINING_STEPS": str(args.warmup + 1),
        "CUDA_VISIBLE_DEVICES": args.devices,
        "TRAIN_MAX_TOKENS_PER_GPU": "8192" if native_8b else "0",
        "ASYNC_TRAIN_MAX_TOKENS_PER_GPU": "8192",
        "CHECKPOINT_BUCKET_MB": "128",
    }


def completed_summary(args, directory):
    logs = [path for path in directory.glob("*.log") if path.name != "launcher.log"]
    if len(logs) != 1:
        raise ValueError(f"Expected one method log, got {logs}")
    try:
        return summarize_run(logs[0], expected_steps=args.warmup + 1, warmup_steps=args.warmup)
    except ValueError as console_error:
        runtime = directory / "runtime.json"
        if not runtime.exists():
            raise
        ray_root = Path(json.loads(runtime.read_text())["ray_tmpdir"])
        candidates = []
        for path in (ray_root / "ray" / "session_latest" / "logs").glob("worker-*.out"):
            with path.open(errors="replace") as source:
                if ":actor_name:TaskRunnerV1\n" in source.read(256):
                    candidates.append(path)
        if len(candidates) != 1:
            raise ValueError(f"{console_error}; expected one original Trainer log, got {candidates}") from console_error
        destination = directory / "trainer_stdout.txt"
        shutil.copyfile(candidates[0], destination)
        summary = summarize_run(destination, expected_steps=args.warmup + 1, warmup_steps=args.warmup)
        summary["console_log_error"] = str(console_error)
        summary["original_worker_log"] = str(candidates[0].resolve())
        return summary


def validate_completed_record(args, case, directory, record):
    if record["returncode"] != 0:
        raise ValueError(f"exit={record['returncode']}")
    summary = completed_summary(args, directory)
    stable = summary["stable_step"]
    staleness = stable.get("training/off_policy/trajectory_staleness/max")
    if case["method"] != "verl-async-opd" and staleness != 0:
        raise ValueError(f"strict on-policy run has missing/nonzero staleness: {staleness}")
    if case["method"].startswith("streamopd-"):
        prefix = "streamopd_cf" if case["method"] == "streamopd-cf" else "streamopd"
        for step in summary["steps"]:
            for metric in ("terminal_trajectories", "completed_teacher_trajectories", "training_trajectories_started"):
                if step.get(f"{prefix}/scheduler_{metric}") != args.batch:
                    raise ValueError(f"step {step['step']}: incomplete {metric}")
    record.update(status="complete", summary=summary)
    record.pop("error", None)
    record["step_seconds"] = stable["timing_s/step"]
    record["response_tokens_per_second"] = args.batch * stable["response_length/mean"] / record["step_seconds"]


def run_case(args, case, directory):
    env = environment(args, case, directory)
    request = {"case": case, "environment": env, "warmup": args.warmup}
    manifest = directory / "request.json"
    if manifest.exists() and json.loads(manifest.read_text()) != request:
        raise ValueError(f"Refusing to mix different settings in {directory}")
    write_json(manifest, request)
    if (directory / "result.json").exists():
        prior = json.loads((directory / "result.json").read_text())
        if prior["status"] == "complete":
            return prior
        if not args.retry_failed:
            return prior
        raise ValueError("Failed runs must be retried in a fresh result root to preserve diagnostics")
    if (directory / "launcher.log").exists():
        raise ValueError(f"Incomplete run must be archived before retrying: {directory}")
    command = [
        sys.executable,
        "-m",
        "benchmarks.streamopd_cf.profile_run",
        "--output",
        str(directory / "gpu.csv"),
        "--devices",
        args.devices,
        "--require-idle",
        "--",
        "bash",
        "benchmarks/streamopd_kv/run_8gpu_case.sh",
        *case.get("overrides", []),
    ]
    source_patch = args.root / "source.patch"
    if source_patch.exists():
        shutil.copyfile(source_patch, directory / "source.patch")
    write_json(
        directory / "launch.json",
        {
            "command": command,
            "start_utc": time.time(),
            "timeout_seconds": args.timeout,
            "runtime": json.loads((args.root / "environment.json").read_text()),
        },
    )
    wait_for_idle(args.devices)
    print(f"START {directory.name}", flush=True)
    stop_reason = None
    log_path = directory / "launcher.log"
    with log_path.open("w") as output, log_path.open(errors="replace") as monitor:
        result = subprocess.Popen(command, env={**os.environ, **env}, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + args.timeout
        tail = ""
        try:
            while result.poll() is None:
                latest = tail + monitor.read()
                tail = latest[-256:]
                if "EngineDeadError: EngineCore encountered an issue" in latest:
                    stop_reason = "vLLM engine failed; stopped the waiting trainer"
                elif time.monotonic() > deadline:
                    stop_reason = "case timeout"
                if stop_reason:
                    break
                time.sleep(2)
        finally:
            if result.poll() is None:
                # profile_run handles SIGINT in its finally block and terminates
                # its separately grouped training processes before returning.
                result.send_signal(signal.SIGINT)
                try:
                    result.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    result.kill()
                    result.wait()
    record = {"case": case, "status": "failed", "returncode": result.returncode, "end_utc": time.time()}
    if stop_reason:
        record["stop_reason"] = stop_reason
    try:
        validate_completed_record(args, case, directory, record)
    except (ValueError, KeyError) as error:
        record["error"] = str(error)
    write_json(directory / "result.json", record)
    print(f"END {directory.name}: {record['status']} {record.get('step_seconds', '')}", flush=True)
    return record


def report(root):
    results = [
        json.loads(p.read_text())
        for p in sorted(root.rglob("result.json"))
        if "diagnostics" not in p.relative_to(root).parts
    ]
    write_json(root / "summary.json", results)
    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "student",
                "teacher",
                "max_total_tokens",
                "method",
                "trainer_gpus",
                "rollout_gpus",
                "teacher_gpus",
                "teacher_tp",
                "status",
                "step_seconds",
                "trained_response_tokens_per_second",
                "tuning",
                "hydra_overrides",
                "launcher_returncode",
                "training_returncode",
                "post_training_launcher_error",
            ]
        )
        for result in results:
            case = result["case"]
            writer.writerow(
                [
                    case["student"],
                    case["teacher_model"],
                    case["tokens"],
                    case["method"],
                    case["trainer"],
                    case["rollout"],
                    case["teacher"],
                    case["tp"],
                    result["status"],
                    result.get("step_seconds"),
                    result.get("response_tokens_per_second"),
                    case.get("tuning", "auto"),
                    json.dumps(case.get("overrides", [])),
                    result["returncode"],
                    result.get("training_returncode", result["returncode"]),
                    bool(result.get("post_training_launcher_error")),
                ]
            )
    lines = [
        "# Eight-GPU OPD pilot",
        "",
        "One measured step after warmup; no variance estimate.",
        "",
        "| Student | Teacher | Tokens | Method / T:R:H / TP | Status | Step s | Response tokens/s |",
        "| --- | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for result in results:
        case = result["case"]
        status = result["status"]
        if result.get("post_training_launcher_error"):
            status += "; launcher error after training (see result.json)"
        lines.append(
            f"| {case['student']} | {case['teacher_model']} | {case['tokens']} | {name(case)} | "
            f"{status} | {result.get('step_seconds', '-')} | "
            f"{result.get('response_tokens_per_second', '-')} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    if (root / "selected_allocations.json").exists():
        from benchmarks.streamopd_kv.summarize_pilot_comparison import build_report

        build_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["allocations", "chunks", "single", "recover", "qualify", "comparison", "report"]
    )
    parser.add_argument("--case-json", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("/data1/models/store/Qwen"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, nargs="+", choices=[4096, 8192], default=[4096])
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--dedicated-min-gain", type=float, default=0.1)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--filter", default="")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        report(args.root)
        return
    device_count = len(args.devices.split(","))
    if args.warmup < 1 or args.timeout <= 0 or device_count < 1:
        parser.error("devices, a positive timeout and at least one warmup step are required")
    if args.stage not in ("single", "recover") and device_count != 8:
        parser.error("allocation and comparison matrices require eight devices; use single for smaller experiments")
    if not 0 <= args.dedicated_min_gain < 1:
        parser.error("dedicated-min-gain must be in [0, 1)")
    if args.stage == "comparison":
        write_json(args.root / "comparison_scope.json", {"max_tokens": sorted(set(args.max_tokens))})
    packages = {
        key: importlib.metadata.version(key)
        for key in (
            "torch",
            "vllm",
            "transformers",
            "flash-attn",
            "ray",
            "liger-kernel",
            "transferqueue",
            "cuda-bindings",
        )
    }
    provenance = {
        "packages": packages,
        "python": sys.executable,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(args.root / "environment.json", provenance)
    (args.root / "source.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], text=True))
    if args.stage in ("single", "recover"):
        if args.case_json is None:
            parser.error("single requires --case-json")
        case = json.loads(args.case_json.read_text())
        setting = f"{case['student']}_{case['teacher_model']}_{case['tokens']}"
        directory = args.root / "comparison" / setting / name(case)
        if args.stage == "recover":
            record = json.loads((directory / "result.json").read_text())
            request = json.loads((directory / "request.json").read_text())
            if request != {"case": case, "environment": environment(args, case, directory), "warmup": args.warmup}:
                raise ValueError("Recovery must use the original benchmark request")
            backup = directory / "result.before_log_recovery.json"
            if not backup.exists():
                shutil.copyfile(directory / "result.json", backup)
            validate_completed_record(args, case, directory, record)
            write_json(directory / "result.json", record)
        else:
            run_case(args, case, directory)
        report(args.root)
        return
    if args.stage in ("allocations", "chunks"):
        candidates = allocation_cases() if args.stage == "allocations" else chunk_cases()
        section = "allocation" if args.stage == "allocations" else "chunks"
        for allocation in candidates:
            case = dict(allocation, student="Qwen3-4B", teacher_model="Qwen3-14B", tokens=4096)
            if args.filter and args.filter not in name(case):
                continue
            run_case(args, case, args.root / section / name(case))
            report(args.root)
        return
    selected_path = args.root / "selected_allocations.json"
    if args.stage == "qualify":
        selected = {}
        qualified_results = {}
        screening = [
            json.loads(p.read_text())
            for section in ("allocation", "chunks")
            for p in (args.root / section).glob("*/result.json")
        ]
        methods, decision = choose_kv_modes(screening, args.dedicated_min_gain)
        write_json(args.root / "kv_placement_decision.json", decision)
        print(f"KV placement decision: {decision}", flush=True)
        for method in methods:
            candidates = [r for r in screening if r["status"] == "complete" and r["case"]["method"] == method]
            for candidate in qualification_order(candidates):
                case = dict(
                    candidate["case"],
                    student="Qwen3-8B",
                    teacher_model="Qwen3-32B",
                    tokens=8192,
                    tp=teacher_tp_for(candidate["case"], "Qwen3-32B"),
                )
                setting = "Qwen3-8B_Qwen3-32B_8192"
                result = run_case(args, case, args.root / "comparison" / setting / name(case))
                report(args.root)
                selected_case = candidate["case"]
                log = args.root / "comparison" / setting / name(case) / "launcher.log"
                memory_rejected = (
                    result["status"] != "complete"
                    and log.exists()
                    and ("auto microbatch estimate cannot fit one trajectory" in log.read_text(errors="replace"))
                )
                if method == "streamopd-cf" and case["trainer"] == 4 and memory_rejected:
                    case = dict(case, tuning="memory90", overrides=["distillation.batching.memory_fraction=0.9"])
                    result = run_case(args, case, args.root / "comparison" / setting / name(case))
                    report(args.root)
                    selected_case = dict(selected_case, tuning=case["tuning"], overrides=case["overrides"])
                if result["status"] == "complete":
                    selected[method] = selected_case
                    qualified_results[method] = result
                    break
            if method not in selected:
                if method == "streamopd-kv-dedicated":
                    decision.update(
                        retain_dedicated=False,
                        default_kv="streamopd-kv-union",
                        qualification_note="No dedicated allocation qualified; retain shared KV only",
                    )
                    write_json(args.root / "kv_placement_decision.json", decision)
                    continue
                raise ValueError(f"No allocation qualifies for the largest dense workload: {method}")
        if "streamopd-kv-dedicated" in selected:
            qualified_screening = [r for r in screening if r["case"] in selected.values()]
            qualified_decision = qualified_kv_decision(
                qualified_screening, list(qualified_results.values()), args.dedicated_min_gain
            )
            if not qualified_decision["retain_dedicated"]:
                selected.pop("streamopd-kv-dedicated")
            decision["after_qualification"] = qualified_decision
            decision["retain_dedicated"] = qualified_decision["retain_dedicated"]
            decision["default_kv"] = qualified_decision["default_kv"]
            write_json(args.root / "kv_placement_decision.json", decision)
        write_json(selected_path, selected)
        report(args.root)
        return
    if not selected_path.exists():
        raise ValueError("Run the qualify stage before the comparison stage")
    selected = json.loads(selected_path.read_text())
    report(args.root)
    for student, teacher, tokens in comparison_settings(args.root):
        for method, allocation in comparison_variants(selected):
            case = comparison_case(selected[allocation], method, student, teacher, tokens)
            setting = f"{student}_{teacher}_{tokens}"
            if args.filter and args.filter not in setting + "/" + name(case):
                continue
            run_case(args, case, args.root / "comparison" / setting / name(case))
            report(args.root)


if __name__ == "__main__":
    main()
