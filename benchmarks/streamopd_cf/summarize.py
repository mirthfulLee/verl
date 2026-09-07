# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Summarize complete matched StreamOPD-CF and native OPD logs."""

import argparse
import json
import math
from pathlib import Path

from benchmarks.streamopd_kv.summarize_colocate_matrix import METRICS, parse_steps, summarize_step_metrics

SYNC_METRICS = (
    "actor/loss",
    "separate_sync/supervised_trajectories_at_training",
    "training/off_policy/trajectory_staleness/max",
    "training/off_policy/trajectory_spans/max",
    "training/off_policy/trajectory_staleness/mean",
    "training/off_policy/trajectory_staleness_worst/max",
    "response_length/clip_ratio",
    "response/aborted_ratio",
    "perf/total_num_tokens",
    "timing_s/switch_to_trainer",
    "timing_s/train_stream",
    "actor/grad_norm",
    "streamopd_cf/scheduler_terminal_trajectories",
    "streamopd_cf/scheduler_completed_teacher_trajectories",
    "streamopd_cf/scheduler_training_trajectories_started",
    "actor/streamopd_cf/training_seconds",
    "actor/streamopd_cf/forward_gpu_seconds",
    "actor/streamopd_cf/loss_gpu_seconds",
    "actor/streamopd_cf/backward_gpu_seconds",
    "actor/streamopd_cf/forward_chunks",
    "actor/streamopd_cf/backward_calls",
    "actor/streamopd_cf/optimizer_steps",
    "actor/streamopd_cf/stream_wait_seconds",
    "actor/batching/stream_window_size",
    "streamopd_cf/stream/forward_chunks_before_eos",
    "streamopd_cf/stream/forward_tokens_before_eos",
    "streamopd_cf/stream/first_forward_seconds",
    "streamopd_cf/stream/all_rollouts_terminal_seconds",
    "streamopd_cf/stream/trained_trajectories",
    "streamopd_cf/stream/backward_calls",
    "actor/streamopd_cf/valid_tokens",
    "actor/streamopd_cf/micro_batch_size_min",
    "actor/streamopd_cf/micro_batch_size_max",
    "actor/batching/dynamic",
    "actor/batching/max_tokens_per_gpu",
    "actor/batching/fixed_micro_batch_size",
    "actor/batching/capacity_at_max_length",
    "actor/batching/memory_budget_gib",
    "actor/batching/estimated_workspace_gib",
    "streamopd_cf/scheduler_all_rollouts_terminal_seconds",
    "streamopd_cf/scheduler_all_teacher_complete_seconds",
    "streamopd_cf/scheduler_teacher_drain_after_rollout_seconds",
    "streamopd_cf/scheduler_teacher_completed_at_first_training",
    "streamopd_cf/scheduler_rollouts_terminal_at_first_training",
    "streamopd_cf/scheduler_teacher_pending_at_first_training",
    "streamopd_cf/scheduler_teacher_busy_seconds",
    "streamopd_cf/scheduler_teacher_busy_before_all_rollouts_terminal_seconds",
    "streamopd_cf/scheduler_teacher_busy_after_all_rollouts_terminal_seconds",
    "streamopd_cf/scheduler_teacher_admission_wait_seconds",
    "streamopd_cf/scheduler_teacher_admission_max_wait_seconds",
    "streamopd_cf/overlap/training_rollout_fraction",
    "streamopd_cf/overlap/training_triple_fraction",
    *(
        f"streamopd_cf/overlap/{phase}_{metric}"
        for phase in ("forward", "loss", "backward", "training")
        for metric in (
            "gpu_seconds",
            "during_rollout_seconds",
            "during_teacher_service_seconds",
            "during_both_seconds",
            "after_teacher_seconds",
        )
    ),
)


def summarize_run(path, expected_steps=3, warmup_steps=1):
    """Fail on partial logs instead of reporting a speedup from incomplete runs."""
    steps = parse_steps(path, (*METRICS, *SYNC_METRICS))
    numbers = [int(step["step"]) for step in steps]
    if numbers != list(range(1, expected_steps + 1)):
        raise ValueError(f"{path}: expected steps 1..{expected_steps}, got {numbers}")
    if "Final validation metrics:" not in path.read_text(errors="replace"):
        raise ValueError(f"{path}: training completion marker is missing")
    for step in steps:
        if not math.isfinite(step.get("timing_s/step", float("nan"))) or step["timing_s/step"] <= 0:
            raise ValueError(f"{path}: invalid step duration")
        if any(not math.isfinite(step[key]) for key in ("actor/loss", "actor/grad_norm") if key in step):
            raise ValueError(f"{path}: non-finite training loss or gradient norm")
        if "streamopd_cf/stream/trained_trajectories" in step:
            for key in ("completed_teacher_trajectories", "terminal_trajectories"):
                if step[f"streamopd_cf/scheduler_{key}"] != step["streamopd_cf/stream/trained_trajectories"]:
                    raise ValueError(f"{path}: policy ended before all streaming windows finished")
            if step.get("actor/streamopd_cf/optimizer_steps") != 1:
                raise ValueError(f"{path}: CF requires one optimizer step per complete policy batch")
            if step.get("training/off_policy/trajectory_staleness/max") != 0:
                raise ValueError(f"{path}: CF consumed a stale trajectory")
    result = summarize_step_metrics(steps, warmup_steps)
    if result["measured_steps"] < 1 or "timing_s/step" not in result["stable_step"]:
        raise ValueError(f"{path}: no measured steps after warmup")
    return {"log": str(path), "steps": steps, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("streamopd_cf", type=Path)
    parser.add_argument("--baseline-label", default="verl-sync-opd")
    parser.add_argument("--expected-steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        name: summarize_run(path, args.expected_steps, args.warmup_steps)
        for name, path in ((args.baseline_label, args.baseline), ("StreamOPD-CF", args.streamopd_cf))
    }
    speedup = (
        runs[args.baseline_label]["stable_step"]["timing_s/step"] / runs["StreamOPD-CF"]["stable_step"]["timing_s/step"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps({"runs": runs, "speedup": speedup}, indent=2) + "\n")
    lines = [
        "# StreamOPD-CF comparison",
        "",
        "| Mode | Mean step (s) | Sample SD (s) | Response tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, run in runs.items():
        stable = run["stable_step"]
        sd = run["step_time_stddev"]
        sd_text = "n/a" if sd is None else f"{sd:.3f}"
        lines.append(
            f"| {name} | {stable['timing_s/step']:.3f} | {sd_text} | {stable.get('response_length/mean', 0):.2f} |"
        )
    lines.extend(
        [
            "",
            f"Speedup: **{speedup:.3f}x**. First {args.warmup_steps} step(s) excluded as warmup.",
            "",
            "Only compare logs produced with the same total GPU budget, models, data, objective and sampling policy. "
            "Use the per-step details to assess response-length differences and variance.",
        ]
    )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"StreamOPD-CF speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
