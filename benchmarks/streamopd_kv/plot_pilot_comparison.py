# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Export compact tables and a static figure from the matched pilot results."""

import argparse
import csv
import json
from pathlib import Path

METHODS = {
    "streamopd-cf": ("CF", "#278064"),
    "verl-async-opd": ("Async", "#83b5a4"),
    "streamopd-kv-union": ("KV shared", "#356fab"),
    "verl-sync-opd-union": ("Sync shared", "#8ab5d8"),
    "streamopd-kv-dedicated": ("KV dedicated", "#ad4a63"),
    "verl-sync-opd-separate": ("Sync dedicated", "#d69caa"),
}


def export(root):
    pairs = json.loads((root / "matched_comparison.json").read_text())
    settings = {}
    records = []
    for pair in pairs:
        key = (pair["student"], pair["teacher"], pair["tokens"])
        cases = settings.setdefault(key, {})
        for role in ("method", "baseline"):
            path = Path(pair[f"{role}_result"])
            result = json.loads(path.read_text()) if path.exists() else {"status": "pending"}
            metrics = result.get("summary", {}).get("stable_step", {})
            record = {
                "student": key[0],
                "teacher": key[1],
                "max_total_tokens": key[2],
                "method": pair[role],
                "status": result["status"],
                "step_seconds": result.get("step_seconds"),
                "response_tokens_per_second": result.get("response_tokens_per_second"),
                "staleness_max": metrics.get("training/off_policy/trajectory_staleness/max"),
                "training_service_seconds": metrics.get("streamopd/scheduler_training_busy_seconds"),
                "native_update_actor_seconds": metrics.get("timing_s/update_actor"),
                "old_log_prob_seconds": metrics.get("timing_s/old_log_prob"),
                "result": str(path),
            }
            cases[pair[role]] = record
            records.append(record)
    with (root / "overview.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    methods = [method for method in METHODS if any(method in cases for cases in settings.values())]
    lines = [
        "# Eight-GPU OPD pilot",
        "",
        "One measured step after one warmup; these are individual observations, not averages.",
        "All token limits below include the prompt (prompt cap 1024).",
        "Native async retains PPO-loop preparation; its observed staleness is recorded in overview.csv.",
        "See matched_comparison.md for baseline-specific notes and recorded host-pressure events.",
        "Training service intervals and native update_actor timers have different boundaries and must not be equated.",
        "",
    ]
    for title, field in (
        ("Step time (seconds; lower is better)", "step_seconds"),
        ("Consumed response tokens/second (higher is better)", "response_tokens_per_second"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Student / Teacher / Max total tokens | " + " | ".join(METHODS[m][0] for m in methods) + " |",
                "| --- | " + " | ".join("---:" for _ in methods) + " |",
            ]
        )
        for (student, teacher, tokens), cases in settings.items():
            values = []
            for method in methods:
                case = cases.get(method, {"status": "pending"})
                value = case.get(field)
                values.append(f"{value:.2f}" if case["status"] == "complete" and value is not None else case["status"])
            lines.append(f"| {student} / {teacher} / {tokens} | " + " | ".join(values) + " |")
        lines.append("")
    (root / "overview.md").write_text("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 3
    rows = (len(settings) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows), squeeze=False)
    maximum_seconds = max((r["step_seconds"] for r in records if r["status"] == "complete"), default=1)
    for axis, ((student, teacher, tokens), cases) in zip(axes.flat, settings.items(), strict=False):
        for index, method in enumerate(methods):
            case = cases.get(method, {"status": "pending"})
            value = case.get("step_seconds")
            if case["status"] == "complete" and value is not None:
                axis.barh(index, value, color=METHODS[method][1], height=0.65)
                axis.text(value, index, f"  {value:.1f}", va="center", fontsize=9)
            else:
                axis.text(0.02, index, case["status"], transform=axis.get_yaxis_transform(), va="center", fontsize=9)
        axis.set_yticks(range(len(methods)), [METHODS[m][0] for m in methods])
        axis.set_ylim(len(methods) - 0.3, -0.7)
        axis.set_title(f"{student} / {teacher}\nMax total tokens: {tokens}", fontsize=11)
        axis.set_xlabel("Seconds per step (lower is better)")
        axis.set_xlim(0, maximum_seconds * 1.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.grid(True, alpha=0.15)
        axis.set_axisbelow(True)
    for axis in list(axes.flat)[len(settings) :]:
        axis.set_visible(False)
    figure.suptitle("8 x A100 80GB | One measured step after one warmup", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(root / "overview.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    export(parser.parse_args().root.resolve())
