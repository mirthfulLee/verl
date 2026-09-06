# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compare consumed training work, stage times and policy staleness."""

import argparse
import json
from pathlib import Path

from .summarize import summarize_run


def summarize_case(path, *, steps, warmup, policy_batch):
    result = summarize_run(path, expected_steps=steps, warmup_steps=warmup)
    measured = result["steps"][warmup:]
    seconds = sum(row["timing_s/step"] for row in measured)
    tokens = sum(row["response_length/mean"] * policy_batch for row in measured)
    staleness = [row["training/off_policy/trajectory_staleness/max"] for row in measured]
    result.update(
        response_tokens_per_second=tokens / seconds,
        consumed_trajectories_per_second=len(measured) * policy_batch / seconds,
        max_staleness=max(staleness),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", default=["default", "model", "tokens", "batch"])
    args = parser.parse_args()
    results = {}
    rows = []
    overlap_rows = []
    for setting in args.settings:
        results[setting] = {}
        batch = 256 if setting == "batch" else 128
        for method in ("streamopd-cf", "verl-sync-opd-separate", "verl-async-opd"):
            logs = list((args.root / setting / method).glob("*.log"))
            if len(logs) != 1:
                raise ValueError(f"{setting}/{method}: expected one complete log, found {logs}")
            result = summarize_case(logs[0], steps=args.steps, warmup=args.warmup, policy_batch=batch)
            if method != "verl-async-opd" and result["max_staleness"] != 0:
                raise ValueError(f"{method} must remain strictly on-policy")
            results[setting][method] = result
            stable = result["stable_step"]
            cf_compute = sum(
                stable.get(f"actor/streamopd_cf/{stage}_gpu_seconds", 0) for stage in ("forward", "loss", "backward")
            )
            rows.append(
                f"| {setting} | {method} | {stable['timing_s/step']:.2f} | "
                f"{result['response_tokens_per_second']:.1f} | {result['max_staleness']:.0f} | "
                f"{stable.get('timing_s/gen', 0):.2f} | {stable.get('timing_s/old_log_prob', 0):.2f} | "
                f"{stable.get('timing_s/update_actor', 0):.2f} | "
                f"{stable.get('timing_s/train_stream', 0):.2f} | {cf_compute:.2f} | "
                f"{stable.get('actor/streamopd_cf/stream_wait_seconds', 0):.2f} | "
                f"{stable.get('timing_s/update_weights', 0):.2f} |"
            )
        cf = results[setting]["streamopd-cf"]
        stable = cf["stable_step"]
        overlap_keys = (
            "streamopd_cf/scheduler_all_rollouts_terminal_seconds",
            "streamopd_cf/scheduler_teacher_drain_after_rollout_seconds",
            "streamopd_cf/overlap/training_during_rollout_seconds",
            "streamopd_cf/overlap/training_during_teacher_service_seconds",
            "streamopd_cf/overlap/training_during_both_seconds",
            "streamopd_cf/overlap/training_after_teacher_seconds",
        )
        overlap_rows.append(
            f"| {setting} | "
            + " | ".join(f"{stable[key]:.2f}" if key in stable else "n/a" for key in overlap_keys)
            + " |"
        )
        for label, baseline in (("async", "verl-async-opd"), ("sync", "verl-sync-opd-separate")):
            results[setting][f"cf_response_throughput_over_{label}"] = (
                cf["response_tokens_per_second"] / results[setting][baseline]["response_tokens_per_second"]
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output / "comparison.md").write_text(
        "# CF, Matched Sync and Native Async OPD\n\n"
        "| Setting | Method | Step (s) | Trained response tokens/s | Max stale versions | "
        "Sample wait (s) | Old logprob (s) | Native update (s) | CF stream wall (s) | "
        "CF GPU F+loss+B (s) | CF input wait (s) | Weight sync (s) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\nAsync sample wait includes only time exposed to Trainer; inference overlaps training. "
        "CF stream wall time includes waiting for rollout/Teacher; GPU stage times exclude input waits. "
        "Stage times overlap and are not additive across roles. "
        "Throughput counts consumed trajectories, excluding prefetched work left at shutdown. "
        "Async allows stale and mixed-version trajectories, so throughput does not establish equal convergence.\n"
        "\n## CF Compute Overlap\n\n"
        "| Setting | All EOS (s) | Teacher tail (s) | CUDA during rollout (s) | "
        "CUDA during Teacher service (s) | CUDA during both (s) | CUDA after Teacher (s) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(overlap_rows)
        + "\n\nCUDA durations average Trainer ranks. Teacher service includes queue/RPC time; "
        "rollout lifetime is not a kernel trace. These overlap columns are not additive. "
        "GPU utilization samples and per-step timelines are stored in each case directory.\n"
    )


if __name__ == "__main__":
    main()
