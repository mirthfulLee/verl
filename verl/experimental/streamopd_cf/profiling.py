# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Single-host CUDA compute overlap with rollout and Teacher service intervals."""

from verl.experimental.streamopd_kv.scheduler import _interval_overlap_seconds, _interval_seconds, _merge_intervals


def summarize_overlap(timelines, service):
    if not timelines:
        return {}
    rollout = [(service["started"], service["rollout_end"])]
    teacher = _merge_intervals(service["teacher"])
    triple = [(max(a, c), min(b, d)) for a, b in rollout for c, d in teacher if max(a, c) < min(b, d)]
    values = {}
    for timeline in timelines:
        phases = dict(timeline)
        phases["training"] = [interval for intervals in timeline.values() for interval in intervals]
        for phase, intervals in phases.items():
            intervals = _merge_intervals(intervals)
            duration = _interval_seconds(intervals)
            prefix = f"streamopd_cf/overlap/{phase}"
            current = {
                f"{prefix}_gpu_seconds": duration,
                f"{prefix}_during_rollout_seconds": _interval_overlap_seconds(intervals, rollout),
                f"{prefix}_during_teacher_service_seconds": _interval_overlap_seconds(intervals, teacher),
                f"{prefix}_during_both_seconds": _interval_overlap_seconds(intervals, triple),
                f"{prefix}_after_teacher_seconds": sum(
                    max(0.0, end - max(start, service["teacher_end"])) for start, end in intervals
                ),
            }
            for key, value in current.items():
                values[key] = values.get(key, 0.0) + value / len(timelines)
    prefix = "streamopd_cf/overlap/training"
    total = max(values[f"{prefix}_gpu_seconds"], 1e-9)
    values[f"{prefix}_rollout_fraction"] = values[f"{prefix}_during_rollout_seconds"] / total
    values[f"{prefix}_triple_fraction"] = values[f"{prefix}_during_both_seconds"] / total
    return values
