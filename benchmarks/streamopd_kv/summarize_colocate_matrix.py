# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRICS = (
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/policy_barrier",
    "perf/throughput",
    "response_length/mean",
    "actor/perf/max_memory_allocated_gb",
    "actor/streamopd/kv_streamed_tokens_before_eos",
    "actor/streamopd/kv_streamed_chunks_before_eos",
    "actor/streamopd/reverse_chunk_size_min",
    "actor/streamopd/reverse_chunk_size_max",
    "actor/streamopd/reverse_memory_budget_gib",
    "actor/streamopd/reverse_page_size",
    "actor/streamopd/reverse_backward_calls",
    "actor/streamopd/reverse_microbatches",
    "actor/streamopd/reverse_planned_batch_size",
    "actor/streamopd/reverse_max_parallel_trajectories",
    "actor/streamopd/training_seconds",
    "actor/streamopd/kv_prefetch_host_seconds",
    "actor/streamopd/kv_prefetch_wait_seconds",
    "actor/streamopd/kv_prefetch_transfer_seconds",
    "actor/streamopd/kv_prefetched_snapshots",
    "actor/streamopd/optimizer_finalized",
    "timing/checkpoint_host_seconds",
    "checkpoint/host_gib_per_second",
    "streamopd/scheduler_wait_seconds",
    "streamopd/scheduler_teacher_chunks",
    "streamopd/scheduler_training_units",
    "streamopd/scheduler_max_training_waiters",
    "streamopd/scheduler_max_teacher_pending",
    "streamopd/scheduler_policy_seconds",
    "training/off_policy/trajectory_staleness/max",
    "training/off_policy/trajectory_spans/max",
)


def _format(value: float | None, precision: int = 2) -> str:
    return "-" if value is None else f"{value:.{precision}f}"


def write_markdown(path: Path, runs: dict[str, dict]) -> None:
    rows = [
        "# StreamOPD-colocate benchmark",
        "",
        "| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(
        runs.values(),
        key=lambda run: (run["total_trajectory_tokens"], run["mode"], run["micro_batch_size"] or 0),
    )
    for run in ordered:
        stable = run["stable_step"]
        rows.append(
            "| "
            + " | ".join(
                (
                    str(run["total_trajectory_tokens"]),
                    f"`{run['mode']}`",
                    str(run["micro_batch_size"]) if run["micro_batch_size"] is not None else "-",
                    _format(stable.get("timing_s/step")),
                    _format(stable.get("perf/throughput")),
                    _format(stable.get("actor/perf/max_memory_allocated_gb")),
                    _format(run.get("speedup_vs_verl_sync_opd"), 3) + "x"
                    if run.get("speedup_vs_verl_sync_opd") is not None
                    else "-",
                    _format(run.get("speedup_vs_verl_colocate_opd"), 3) + "x"
                    if run.get("speedup_vs_verl_colocate_opd") is not None
                    else "-",
                )
            )
            + " |"
        )
    path.write_text("\n".join(rows) + "\n")


def parse_steps(path: Path) -> list[dict[str, float]]:
    steps: list[dict[str, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"\bstep:(\d+)\s+-", line)
        if match is None:
            continue
        values: dict[str, float] = {"step": float(match.group(1))}
        for name in METRICS:
            metric = re.search(rf"(?:^|\s)-\s+{re.escape(name)}:([^\s]+)", line)
            if metric is None:
                continue
            raw = re.sub(r"^np\.(?:float|int)\d*\((.*)\)$", r"\1", metric.group(1))
            try:
                values[name] = float(raw)
            except ValueError:
                pass
        steps.append(values)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional result directories containing baseline .log files.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs: dict[str, dict] = {}
    pattern = re.compile(r"(?P<mode>.+)_total(?P<tokens>\d+)(?:_mb(?P<microbatch>\d+))?\.log$")
    # Load baseline directories first so the requested result directory wins
    # if a run with the same filename is present in both places.
    for directory in [*args.baseline_dir, args.result_dir]:
        for path in sorted(directory.glob("*.log")):
            match = pattern.match(path.name)
            if match is None:
                continue
            steps = parse_steps(path)
            mode = match.group("mode")
            microbatch = match.group("microbatch")
            runs[path.stem] = {
                "mode": mode,
                "total_trajectory_tokens": int(match.group("tokens")),
                # Legacy baseline files carried an mb32 suffix even though
                # sync paths never consumed the StreamOPD trainer microbatch.
                "micro_batch_size": int(microbatch) if mode == "streamopd-colocate" and microbatch else None,
                "status": "ok" if steps else "failed",
                "steps": steps,
                "stable_step": steps[-1] if steps else {},
                "source_dir": str(directory),
            }

    for run in runs.values():
        if run["status"] != "ok":
            continue
        step = run["stable_step"].get("timing_s/step")
        for baseline_mode in ("verl-sync-opd", "verl-colocate-opd"):
            baseline_candidates = (
                candidate
                for candidate in runs.values()
                if candidate["mode"] == baseline_mode
                and candidate["total_trajectory_tokens"] == run["total_trajectory_tokens"]
                and candidate["status"] == "ok"
            )
            baseline = min(
                baseline_candidates,
                key=lambda candidate: candidate["stable_step"].get("timing_s/step", float("inf")),
                default=None,
            )
            baseline_step = baseline["stable_step"].get("timing_s/step") if baseline else None
            if step and baseline_step:
                run[f"speedup_vs_{baseline_mode.replace('-', '_')}"] = baseline_step / step

    output = args.output or args.result_dir / "summary.json"
    output.write_text(json.dumps({"runs": runs}, indent=2, sort_keys=True) + "\n")
    write_markdown(output.with_suffix(".md"), runs)
    print(output)


if __name__ == "__main__":
    main()
