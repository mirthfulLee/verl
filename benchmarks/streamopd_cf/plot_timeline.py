# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Plot measured CF CUDA intervals, inference service spans and GPU samples."""

import argparse
import csv
import json
from pathlib import Path


def plot_timeline(timeline, output, gpu_samples=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    service, ranks = timeline["service"], timeline["trainer_gpu"]
    origin = service["started"]
    end = max(end for rank in ranks for intervals in rank.values() for _, end in intervals)
    colors = {"forward": "#218c74", "loss": "#c29628", "backward": "#b34057"}
    figure, axes = plt.subplots(2 if gpu_samples else 1, 1, figsize=(12, 6 if gpu_samples else 3.5), squeeze=False)
    axis = axes[0, 0]

    def bar(intervals, row, color):
        axis.broken_barh(
            [(start - origin, finish - start) for start, finish in intervals], (row - 0.3, 0.6), facecolors=color
        )

    bar([(origin, service["rollout_end"])], 0, "#9aa5ae")
    bar(service["teacher"], 1, "#567bb5")
    for index, rank in enumerate(ranks):
        for phase, intervals in rank.items():
            bar(intervals, index + 2, colors[phase])
    axis.set_yticks(
        range(len(ranks) + 2),
        ["Rollout lifetime", "Teacher service"] + [f"Trainer rank {i}" for i in range(len(ranks))],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, end - origin)
    axis.set_xlabel("Seconds from policy start")
    axis.set_title("StreamOPD-CF: CUDA compute and inference service intervals")
    axis.legend(handles=[Patch(color=color, label=phase) for phase, color in colors.items()], loc="upper right")
    axis.grid(axis="x", alpha=0.2)
    if gpu_samples:
        axis = axes[1, 0]
        grouped = {}
        for row in gpu_samples:
            timestamp = float(row["monotonic_seconds"])
            if origin <= timestamp <= end:
                grouped.setdefault(row["gpu_index"], []).append(
                    (timestamp - origin, float(row["gpu_utilization_percent"]))
                )
        for gpu, samples in sorted(grouped.items()):
            axis.plot(*zip(*samples, strict=True), label=f"GPU {gpu}", linewidth=1)
        axis.set_xlim(0, end - origin)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Sampled GPU utilization (%)")
        axis.set_xlabel("Seconds from policy start")
        axis.legend(ncol=4, loc="lower left")
        axis.grid(alpha=0.2)
    figure.text(
        0.01, 0.01, "Teacher spans include queue/RPC time; NVML utilization is sampled, not a kernel trace.", fontsize=8
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--gpu-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = None
    if args.gpu_csv:
        with args.gpu_csv.open(newline="") as source:
            samples = list(csv.DictReader(source))
    plot_timeline(json.loads(args.timeline.read_text()), args.output, samples)


if __name__ == "__main__":
    main()
