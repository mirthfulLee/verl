# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Collect stage bounds for the streaming cases in the selected pilot matrix."""

import argparse
import csv
import json
from pathlib import Path


def export(root):
    pairs = json.loads((root / "matched_comparison.json").read_text())
    rows = []
    for pair in pairs:
        if pair["method_status"] != "complete":
            continue
        directory = Path(pair["method_result"]).parent
        request = json.loads((directory / "request.json").read_text())
        for path in sorted((directory / "timelines").glob("step-*.json")):
            step = int(path.stem.rsplit("-", 1)[1])
            timeline = json.loads(path.read_text())
            service = timeline.get("service", timeline)
            origin = service["started"]
            stages = [("rollout_lifetime", "service", origin, service["rollout_end"])]
            teacher = service["teacher"]
            stages.append(("teacher", "service", min(s for s, _ in teacher), service["teacher_end"]))
            training = service.get("training", [])
            kind = "service"
            if not training:
                kind = "cuda_all_ranks"
                training = [
                    span for rank in timeline.get("trainer_gpu", []) for spans in rank.values() for span in spans
                ]
            if not training:
                raise ValueError(f"Missing training intervals in completed streaming case: {path}")
            stages.append(("training", kind, min(s for s, _ in training), max(e for _, e in training)))
            for phase, kind, start, end in stages:
                if start is None or end is None or end < start:
                    raise ValueError(f"Invalid completed stage in {path}: {phase}")
                rows.append(
                    {
                        "student": pair["student"],
                        "teacher": pair["teacher"],
                        "max_total_tokens": pair["tokens"],
                        "method": pair["method"],
                        "step": step,
                        "warmup": step <= request["warmup"],
                        "phase": phase,
                        "kind": kind,
                        "start_monotonic_seconds": start,
                        "end_monotonic_seconds": end,
                        "start_from_policy_seconds": start - origin,
                        "end_from_policy_seconds": end - origin,
                        "span_seconds": end - start,
                        "timeline": str(path),
                    }
                )
    with (root / "stage_bounds.csv").open("w", newline="") as destination:
        if rows:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "stage_bounds.md").write_text(
        "# Streaming stage bounds\n\n"
        "stage_bounds.csv covers only completed streaming cases in the selected matrix, including warmup.\n"
        "Times are monotonic host-clock seconds, with a separate offset from the policy start.\n"
        "Rollout lifetime starts at scheduler policy start and includes dispatch. Teacher service includes waits.\n"
        "KV training uses scheduler service bounds. CF training uses the first and last recorded CUDA events "
        "across all Trainer ranks. These boundaries have different meanings.\n"
        "A span includes gaps between intervals; it is not the sum of GPU compute time. "
        "Overlapping stages must not be added to estimate whole-step latency.\n"
        "The source timeline retains individual intervals and, for CF, per-rank forward/loss/backward events.\n"
    )
    print(f"Exported {len(rows)} stage bounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    export(parser.parse_args().root.resolve())
