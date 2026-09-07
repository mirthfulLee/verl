# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compare reverse shapes with whole-device memory during measured training."""

import argparse
import ast
import csv
import json
import re
from pathlib import Path

from benchmarks.streamopd_kv.pilot_8gpu import write_json


def summarize_case(directory):
    result = json.loads((directory / "result.json").read_text())
    record = {"directory": str(directory), "case": result["case"], "status": result["status"]}
    plans = []
    for line in (directory / "launcher.log").read_text(errors="replace").splitlines():
        if "StreamOPD reverse preflight: " in line:
            value = re.sub(r"\x1b\[[0-9;]*m", "", line.split("StreamOPD reverse preflight: ", 1)[1])
            plans = ast.literal_eval(value)
    record["rank_plans"] = plans
    if result["status"] != "complete":
        record["error"] = result.get("error", result.get("stop_reason"))
        return record
    request = json.loads((directory / "request.json").read_text())
    timeline = json.loads((directory / "timelines" / f"step-{request['warmup'] + 1}.json").read_text())
    intervals = timeline["training"]
    samples = {}
    with (directory / "gpu.csv").open() as source:
        for row in csv.DictReader(source):
            timestamp = float(row["monotonic_seconds"])
            if any(start <= timestamp <= end for start, end in intervals):
                samples.setdefault(row["gpu_index"], []).append(row)
    record["training_device_samples"] = {
        gpu: {
            "sample_count": len(rows),
            "peak_used_gib": max(float(row["memory_used_mib"]) for row in rows) / 1024,
            "mean_utilization_percent": sum(float(row["gpu_utilization_percent"]) for row in rows) / len(rows),
        }
        for gpu, rows in samples.items()
    }
    metrics = result["summary"]["stable_step"]
    record.update(
        step_seconds=result["step_seconds"],
        response_tokens_per_second=result["response_tokens_per_second"],
        training_seconds=sum(end - start for start, end in intervals),
        training_intervals=intervals,
        training_load_seconds=metrics.get("streamopd/trainer_load_seconds"),
        training_offload_seconds=metrics.get("streamopd/trainer_offload_seconds"),
        weight_sync_seconds=metrics.get("timing_s/update_weights"),
        teacher_complete_seconds=metrics.get("streamopd/scheduler_all_teacher_complete_seconds"),
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    directories = [root / "allocation" / "streamopd-kv-union-t8-r6-h2-p1"]
    directories.extend(sorted((root / "chunks").glob("*")))
    directories.append(root / "allocation" / "streamopd-kv-union-t8-r6-h2-p2")
    records = [summarize_case(path) for path in directories if (path / "result.json").exists()]
    write_json(root / "chunk_comparison.json", records)
    lines = [
        "# Shared KV reverse chunk pilot",
        "",
        "One measured step after one warmup. Memory is sampled whole-device usage during training;",
        "it includes inference processes and allocator caches, and may miss brief peaks.",
        "Rank plans are preserved in chunk_comparison.json; controller metrics alone do not describe every rank.",
        "",
        "| Case | Status | Training s | Step s | Response tokens/s | Peak device GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        peak = max((row["peak_used_gib"] for row in record.get("training_device_samples", {}).values()), default=0)
        lines.append(
            f"| {Path(record['directory']).name} | {record['status']} | "
            f"{record.get('training_seconds', '-')} | {record.get('step_seconds', '-')} | "
            f"{record.get('response_tokens_per_second', '-')} | {peak or '-'} |"
        )
    (root / "chunk_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.plot:
        plot_comparison(records, root / "chunk_comparison.png")


def plot_comparison(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [record for record in records if record["status"] == "complete"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [record["case"].get("tuning", f"auto: Teacher TP={record['case']['tp']}") for record in complete]
    rows = list(range(len(complete)))
    training = [record["training_seconds"] for record in complete]
    other = [record["step_seconds"] - record["training_seconds"] for record in complete]
    axes[0].barh(rows, training, label="Training service", color="#218c74")
    axes[0].barh(rows, other, left=training, label="Other step time", color="#6989ad")
    for row, record in enumerate(complete):
        axes[0].text(record["step_seconds"] + 2, row, f"{record['step_seconds']:.2f}s", va="center", fontsize=9)
    axes[0].set_yticks(rows, labels)
    axes[0].set_xlim(0, max(record["step_seconds"] for record in complete) * 1.25)
    axes[0].set_xlabel("Measured step seconds")
    axes[0].legend(loc="lower right", fontsize=8)
    for row, record in enumerate(complete):
        peaks = [values["peak_used_gib"] for values in record["training_device_samples"].values()]
        axes[1].scatter(peaks, [row] * len(peaks), s=28, alpha=0.6, color="#ae4964")
    axes[1].axvline(79.253662109375, color="#555555", linestyle="--", linewidth=1, label="Device capacity")
    axes[1].set_yticks(rows, [])
    axes[1].set_xlabel("Per-GPU sampled peak usage during training (GiB)")
    axes[1].set_xlim(0, 82)
    axes[1].legend(loc="upper left", fontsize=8)
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Shared KV: Qwen3-4B / Qwen3-14B, 4096 tokens, Rollout 6 + Teacher 2")
    figure.text(
        0.01, 0.01, "One measured step after warmup. Tuned cases remove the extra 4 GiB planner reserve.", fontsize=8
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
