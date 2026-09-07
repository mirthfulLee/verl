# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Export recorded service and GPU intervals without conflating their clocks or meaning."""

import argparse
import csv
import json
from pathlib import Path


def export_case(directory):
    request = json.loads((directory / "request.json").read_text())
    case = request["case"]
    rows = []
    for source in sorted((directory / "timelines").glob("step-*.json")):
        step = int(source.stem.split("-")[-1])
        timeline = json.loads(source.read_text())
        service = timeline.get("service", timeline)
        origin = service["started"]
        intervals = (
            [("rollout_lifetime", "service", "", [(origin, service["rollout_end"])])]
            if service.get("rollout_end") is not None
            else []
        )
        intervals.extend((phase, "service", "", service.get(phase, [])) for phase in ("teacher", "training"))
        for rank, phases in enumerate(timeline.get("trainer_gpu", [])):
            intervals.extend((phase, "gpu", rank, spans) for phase, spans in phases.items())
        for phase, kind, rank, spans in intervals:
            for index, (start, end) in enumerate(spans):
                if end < start:
                    raise ValueError(f"Negative interval in {source}: {phase}, rank={rank}")
                rows.append(
                    dict(
                        method=case["method"],
                        student=case["student"],
                        teacher=case["teacher_model"],
                        max_total_tokens=case["tokens"],
                        step=step,
                        warmup=step <= request["warmup"],
                        kind=kind,
                        phase=phase,
                        rank=rank,
                        interval=index,
                        start_monotonic_seconds=start,
                        end_monotonic_seconds=end,
                        start_from_policy_seconds=start - origin,
                        end_from_policy_seconds=end - origin,
                        duration_seconds=end - start,
                    )
                )
    if rows:
        with (directory / "phase_intervals.csv").open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    count = cases = 0
    for request in args.root.rglob("request.json"):
        if "diagnostics" in request.relative_to(args.root).parts:
            continue
        directory = request.parent
        if not (directory / "result.json").exists():
            continue
        added = export_case(directory)
        count += added
        cases += bool(added)
    print(f"Exported {count} intervals across {cases} cases")


if __name__ == "__main__":
    main()
