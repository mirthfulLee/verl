# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Summarize only matched, completed method/baseline pairs from the pilot."""

import argparse
import csv
import json
from pathlib import Path

from benchmarks.streamopd_kv.pilot_8gpu import comparison_case, comparison_settings, name, write_json


def read_case(root, case):
    setting = f"{case['student']}_{case['teacher_model']}_{case['tokens']}"
    path = root / "comparison" / setting / name(case) / "result.json"
    if not path.exists():
        return {"status": "pending"}, path
    result = json.loads(path.read_text())
    if result["case"] != case:
        raise ValueError(f"Result does not match selected allocation: {path}")
    return result, path


def build_report(root):
    selected = json.loads((root / "selected_allocations.json").read_text())
    pairs = [("streamopd-kv-union", "verl-sync-opd-union"), ("streamopd-cf", "verl-async-opd")]
    if "streamopd-kv-dedicated" in selected:
        pairs.append(("streamopd-kv-dedicated", "verl-sync-opd-separate"))
    rows = []
    for student, teacher, tokens in comparison_settings(root):
        for method, baseline in pairs:
            candidate = comparison_case(selected[method], method, student, teacher, tokens)
            control = comparison_case(selected[method], baseline, student, teacher, tokens)
            measured, measured_path = read_case(root, candidate)
            reference, reference_path = read_case(root, control)
            row = dict(
                student=student,
                teacher=teacher,
                tokens=tokens,
                method=method,
                baseline=baseline,
                trainer_gpus=candidate["trainer"],
                rollout_gpus=candidate["rollout"],
                teacher_gpus=candidate["teacher"],
                teacher_tp=candidate["tp"],
                method_status=measured["status"],
                baseline_status=reference["status"],
                method_seconds=measured.get("step_seconds"),
                baseline_seconds=reference.get("step_seconds"),
                step_speedup=None,
                response_throughput_ratio=None,
                baseline_staleness_max=None,
                baseline_old_log_prob_seconds=None,
                notes=[],
                method_result=str(measured_path),
                baseline_result=str(reference_path),
            )
            for label, result in (("method", measured), ("baseline", reference)):
                case_path = measured_path if label == "method" else reference_path
                audit_path = case_path.parent / "host_pressure_audit.json"
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    pressure = audit["system_memory_pressure_seconds"]["some"]
                    row["notes"].append(
                        f"{label}: memory.high events near weight sync; {pressure:.3f}s system memory PSI in "
                        "bracketing samples, not a per-job latency estimate"
                    )
                if result.get("post_training_launcher_error"):
                    row["notes"].append(f"{label}: verified training completed before a launcher error")
                if result["status"] == "complete":
                    warmup = result["summary"]["steps"][0]["timing_s/step"]
                    if result["step_seconds"] >= warmup * 1.5:
                        row["notes"].append(f"{label}: measured step >= 1.5x warmup; stability needs review")
            if reference["status"] == "complete":
                row["baseline_old_log_prob_seconds"] = reference["summary"]["stable_step"].get(
                    "timing_s/old_log_prob", 0
                )
                if row["baseline_old_log_prob_seconds"]:
                    row["notes"].append(
                        f"native baseline includes {row['baseline_old_log_prob_seconds']:.2f}s old-log-prob preparation"
                    )
            if measured["status"] == reference["status"] == "complete":
                row["step_speedup"] = reference["step_seconds"] / measured["step_seconds"]
                row["response_throughput_ratio"] = (
                    measured["response_tokens_per_second"] / reference["response_tokens_per_second"]
                )
                row["baseline_staleness_max"] = reference["summary"]["stable_step"].get(
                    "training/off_policy/trajectory_staleness/max"
                )
            rows.append(row)
    write_json(root / "matched_comparison.json", rows)
    with (root / "matched_comparison.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, "notes": "; ".join(row["notes"])} for row in rows)
    lines = [
        "# Matched OPD pilot comparisons",
        "",
        "One measured step after one warmup. Ratios above 1 favor the streaming method.",
        "No speedup is reported until both matching runs complete. CSV/JSON retain allocation and result paths.",
        "",
        "| Student / Teacher / Tokens | Method | Method s | Baseline s | Step speedup | Token ratio | "
        "Baseline staleness | Status / Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def number(value):
        return "-" if value is None else f"{value:.2f}"

    for row in rows:
        status = f"{row['method_status']} / {row['baseline_status']}"
        notes = "; ".join(row["notes"])
        lines.append(
            f"| {row['student']} / {row['teacher']} / {row['tokens']} | {row['method']} | "
            f"{number(row['method_seconds'])} | {number(row['baseline_seconds'])} | "
            f"{number(row['step_speedup'])} | {number(row['response_throughput_ratio'])} | "
            f"{number(row['baseline_staleness_max'])} | {status}{'; ' + notes if notes else ''} |"
        )
    (root / "matched_comparison.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build_report(parser.parse_args().root.resolve())
