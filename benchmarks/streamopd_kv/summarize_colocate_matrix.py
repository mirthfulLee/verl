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
    "stage/rollout_span_seconds",
    "stage/rollout_makespan_seconds",
    "stage/teacher_span_seconds",
    "stage/teacher_makespan_seconds",
    "stage/teacher_tail_seconds",
    "stage/training_seconds",
    "stage/rollout_request_seconds/mean",
    "stage/rollout_request_seconds/max",
    "stage/teacher_request_seconds/mean",
    "stage/teacher_request_seconds/max",
    "perf/throughput",
    "response_length/mean",
    "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb",
    "actor/streamopd/kv_streamed_tokens_before_eos",
    "actor/streamopd/kv_streamed_chunks_before_eos",
    "actor/streamopd/reverse_chunk_size_min",
    "actor/streamopd/reverse_chunk_size_max",
    "actor/streamopd/reverse_memory_budget_gib",
    "actor/streamopd/reverse_slot_batch_size",
    "actor/streamopd/reverse_slot_token_capacity",
    "actor/streamopd/reverse_slot_prefetch_kv",
    "actor/streamopd/reverse_slot_backing_gib",
    "actor/streamopd/reverse_slot_loaded_gib",
    "actor/streamopd/reverse_slot_copy_cuda_seconds",
    "actor/streamopd/reverse_slot_next_copy_cuda_seconds",
    "actor/streamopd/reverse_slot_copy_enqueue_seconds",
    "actor/streamopd/reverse_slot_next_copy_enqueue_seconds",
    "actor/streamopd/reverse_slot_initial_wait_seconds",
    "actor/streamopd/reverse_slot_next_wait_seconds",
    "actor/streamopd/reverse_slot_overlap_seconds",
    "actor/streamopd/reverse_slot_next_loaded_pages",
    "actor/streamopd/reverse_page_size",
    "actor/streamopd/reverse_backward_calls",
    "actor/streamopd/reverse_real_backward_calls",
    "actor/streamopd/reverse_dummy_backward_calls",
    "actor/streamopd/reverse_microbatches",
    "actor/streamopd/reverse_planned_batch_size",
    "actor/streamopd/reverse_max_parallel_trajectories",
    "actor/streamopd/reverse_model_tokens",
    "actor/streamopd/reverse_model_token_fraction",
    "actor/streamopd/reverse_padding_tokens_trimmed",
    "actor/streamopd/training_seconds",
    "actor/streamopd/kv_prefetch_host_seconds",
    "actor/streamopd/kv_prefetch_wait_seconds",
    "actor/streamopd/kv_prefetch_transfer_seconds",
    "actor/streamopd/kv_prefetched_snapshots",
    "actor/streamopd/optimizer_finalized",
    "actor/streamopd/gradient_syncs",
    "actor/streamopd/gradient_syncs_total",
    "actor/streamopd/defer_gradient_sync",
    "timing/checkpoint_host_seconds",
    "checkpoint/host_gib_per_second",
    "streamopd/scheduler_wait_seconds",
    "streamopd/scheduler_teacher_chunks",
    "streamopd/scheduler_teacher_notifications",
    "streamopd/scheduler_teacher_coalesced_fragments",
    "streamopd/scheduler_teacher_admission_attempts",
    "streamopd/scheduler_teacher_admission_rejections",
    "streamopd/scheduler_teacher_admission_unavailable_rejections",
    "streamopd/scheduler_teacher_admission_trajectory_rejections",
    "streamopd/scheduler_teacher_admission_kv_rejections",
    "streamopd/scheduler_teacher_admission_wait_seconds",
    "streamopd/scheduler_teacher_admission_max_wait_seconds",
    "streamopd/scheduler_teacher_admission_waited_sessions",
    "streamopd/scheduler_training_units",
    "streamopd/scheduler_max_training_waiters",
    "streamopd/scheduler_max_teacher_pending",
    "streamopd/scheduler_max_teacher_active",
    "streamopd/scheduler_max_teacher_sessions",
    "streamopd/scheduler_max_teacher_session_kv_tokens",
    "streamopd/scheduler_teacher_busy_seconds",
    "streamopd/scheduler_training_busy_seconds",
    "streamopd/scheduler_concurrent_busy_seconds",
    "streamopd/scheduler_busy_union_seconds",
    "streamopd/scheduler_pool_busy_seconds",
    "streamopd/scheduler_pool_idle_seconds",
    "streamopd/scheduler_pool_utilization",
    "streamopd/scheduler_aggregate_role_utilization",
    "streamopd/scheduler_resources_overlap",
    "streamopd/scheduler_train_launch_width",
    "streamopd/scheduler_training_trajectories_started",
    "streamopd/scheduler_lifecycle_seconds",
    "streamopd/scheduler_terminal_trajectories",
    "streamopd/scheduler_completed_teacher_trajectories",
    "streamopd/scheduler_all_rollouts_terminal_seconds",
    "streamopd/scheduler_all_teacher_complete_seconds",
    "streamopd/scheduler_teacher_drain_after_rollout_seconds",
    "streamopd/scheduler_first_teacher_started_seconds",
    "streamopd/scheduler_first_training_ready_seconds",
    "streamopd/scheduler_first_training_started_seconds",
    "streamopd/scheduler_teacher_completed_at_first_training",
    "streamopd/scheduler_rollouts_terminal_at_first_training",
    "streamopd/scheduler_teacher_pending_at_first_training",
    "streamopd/scheduler_teacher_busy_before_all_rollouts_terminal_seconds",
    "streamopd/scheduler_teacher_busy_after_all_rollouts_terminal_seconds",
    "streamopd/teacher_drain_wait_seconds",
    "streamopd/teacher_sleep_seconds",
    "streamopd/teacher_wake_seconds",
    "streamopd/rollout_dispatch_seconds",
    "streamopd/rollout_pool_wait_seconds",
    "streamopd/trainer_load_seconds",
    "streamopd/trainer_offload_seconds",
    "streamopd/rollout_wake_seconds",
    "checkpoint/phase_exclusive_rollout_sleep_seconds",
    "checkpoint/phase_exclusive_publish_seconds",
    "checkpoint/phase_exclusive_trainer_release_seconds",
    "checkpoint/phase_exclusive_weights_wake_seconds",
    "checkpoint/phase_exclusive_receive_seconds",
    "checkpoint/phase_exclusive_kv_wake_seconds",
    "streamopd/host_slot_pool_released_gib",
    "streamopd/host_slot_pool_cleanup_seconds",
    "streamopd/memory_stats_reset_seconds",
    "streamopd/teacher_plan_active_trajectories",
    "streamopd/teacher_plan_active_kv_tokens",
    "streamopd/teacher_plan_vllm_capacity_tokens",
    "streamopd/teacher_plan_safe_capacity_tokens",
    "streamopd/teacher_plan_trajectory_tokens",
    "streamopd/teacher_plan_teacher_replicas",
    "streamopd/teacher_plan_prefill_wave_per_replica",
    "streamopd/teacher_plan_prefill_wave",
    "streamopd/training_unit_size",
    "streamopd/reverse_wave_size",
    "streamopd/reverse_waves_per_training_unit",
    "streamopd/rollout_plan_exclusive_pool_memory",
    "streamopd/rollout_plan_vllm_capacity_tokens",
    "streamopd/rollout_plan_max_num_seqs",
    "streamopd/rollout_plan_configured_max_num_seqs",
    "streamopd/rollout_plan_max_model_len",
    "streamopd/rollout_plan_checkpoint_bucket_mb",
    "streamopd/teacher_runtime_plan_exclusive_pool_memory",
    "streamopd/teacher_runtime_plan_vllm_capacity_tokens",
    "streamopd/teacher_runtime_plan_max_num_seqs",
    "streamopd/runtime_profile_auto",
    "streamopd/runtime_profile_trajectory_tokens",
    "streamopd/runtime_profile_token_chunk_size",
    "streamopd/runtime_profile_teacher_max_batched_tokens",
    "streamopd/runtime_profile_teacher_exclusive_pool_memory",
    "streamopd/runtime_profile_teacher_vllm_capacity_tokens",
    "streamopd/runtime_profile_teacher_max_num_seqs",
    "streamopd/runtime_profile_rollout_exclusive_pool_memory",
    "streamopd/runtime_profile_rollout_vllm_capacity_tokens",
    "streamopd/runtime_profile_rollout_max_num_seqs",
    "streamopd/teacher_memory/allocated_gib",
    "streamopd/teacher_memory/reserved_gib",
    "streamopd/teacher_memory/max_allocated_gib",
    "streamopd/teacher_memory/max_reserved_gib",
    "streamopd/teacher_memory/collect_seconds",
    "streamopd/rollout_memory/allocated_gib",
    "streamopd/rollout_memory/reserved_gib",
    "streamopd/rollout_memory/max_allocated_gib",
    "streamopd/rollout_memory/max_reserved_gib",
    "streamopd/rollout_kv_transfer/copy_chunks",
    "streamopd/rollout_kv_transfer/copy_gib",
    "streamopd/rollout_kv_transfer/copy_calls",
    "streamopd/rollout_kv_transfer/block_runs",
    "streamopd/rollout_kv_transfer/staging_wait_seconds",
    "streamopd/rollout_kv_transfer/max_staging_wait_seconds",
    "streamopd/rollout_kv_transfer/copy_enqueue_seconds",
    "streamopd/rollout_kv_transfer/gpu_gather_seconds",
    "streamopd/rollout_kv_transfer/gpu_d2h_seconds",
    "streamopd/rollout_kv_transfer/gpu_copy_seconds",
    "streamopd/rollout_kv_transfer/d2h_wait_seconds",
    "streamopd/rollout_kv_transfer/host_commit_seconds",
    "streamopd/rollout_kv_transfer/terminal_wait_seconds",
    "streamopd/rollout_kv_transfer/max_outstanding_writes",
    "training/off_policy/trajectory_staleness/max",
    "training/off_policy/trajectory_spans/max",
)


def _format(value: float | None, precision: int = 2) -> str:
    return "-" if value is None else f"{value:.{precision}f}"


def _stage_value(stable: dict[str, float], baseline_name: str, streamopd_name: str) -> float | None:
    return stable.get(streamopd_name, stable.get(baseline_name))


def write_markdown(path: Path, runs: dict[str, dict]) -> None:
    rows = [
        "# StreamOPD benchmark",
        "",
        "| Total tokens | Batch | Mode | Status | Microbatch | Step (s) | Gen (s) | "
        "Rollout EOS (s) | Teacher done (s) | Train busy (s) | Tokens/s | "
        "Actor peak (GiB) | vs sync |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(
        runs.values(),
        key=lambda run: (
            run["total_trajectory_tokens"],
            run["batch_size"] or 0,
            run["mode"],
            run["micro_batch_size"] or 0,
        ),
    )
    for run in ordered:
        stable = run["stable_step"]
        rows.append(
            "| "
            + " | ".join(
                (
                    str(run["total_trajectory_tokens"]),
                    str(run["batch_size"]) if run["batch_size"] is not None else "-",
                    f"`{run['mode']}`",
                    run["status"],
                    str(run["micro_batch_size"]) if run["micro_batch_size"] is not None else "-",
                    _format(stable.get("timing_s/step")),
                    _format(stable.get("timing_s/gen")),
                    _format(
                        _stage_value(
                            stable,
                            "stage/rollout_makespan_seconds",
                            "streamopd/scheduler_all_rollouts_terminal_seconds",
                        )
                    ),
                    _format(
                        _stage_value(
                            stable,
                            "stage/teacher_makespan_seconds",
                            "streamopd/scheduler_all_teacher_complete_seconds",
                        )
                    ),
                    _format(
                        _stage_value(
                            stable,
                            "stage/training_seconds",
                            "streamopd/scheduler_training_busy_seconds",
                        )
                    ),
                    _format(stable.get("perf/throughput")),
                    _format(stable.get("actor/perf/max_memory_allocated_gb")),
                    _format(run.get("speedup_vs_verl_sync_opd"), 3) + "x"
                    if run.get("speedup_vs_verl_sync_opd") is not None
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
    pattern = re.compile(r"(?P<mode>.+)_total(?P<tokens>\d+)(?:_bs(?P<batch>\d+))?(?:_mb(?P<microbatch>\d+))?\.log$")
    # Load baseline directories first so the requested result directory wins
    # if a run with the same filename is present in both places.
    for directory in [*args.baseline_dir, args.result_dir]:
        for path in sorted(directory.glob("*.log")):
            match = pattern.match(path.name)
            if match is None:
                continue
            steps = parse_steps(path)
            text = path.read_text(errors="replace")
            failed = "Error executing job with overrides:" in text or "torch.OutOfMemoryError:" in text
            mode = match.group("mode")
            batch = match.group("batch")
            microbatch = match.group("microbatch")
            runs[path.stem] = {
                "mode": mode,
                "total_trajectory_tokens": int(match.group("tokens")),
                "batch_size": int(batch) if batch else None,
                # Legacy baseline files carried an mb32 suffix even though
                # sync paths never consumed the StreamOPD trainer microbatch.
                "micro_batch_size": (int(microbatch) if mode.startswith("streamopd") and microbatch else None),
                "status": "partial" if steps and failed else "ok" if steps else "failed",
                "steps": steps,
                "stable_step": steps[-1] if steps else {},
                "source_dir": str(directory),
            }

    for run in runs.values():
        if run["status"] == "failed":
            continue
        step = run["stable_step"].get("timing_s/step")
        baseline_candidates = (
            candidate
            for candidate in runs.values()
            if candidate["mode"] == "verl-sync-opd"
            and candidate["total_trajectory_tokens"] == run["total_trajectory_tokens"]
            and candidate["batch_size"] == run["batch_size"]
            and candidate["status"] in {"ok", "partial"}
        )
        baseline = min(
            baseline_candidates,
            key=lambda candidate: candidate["stable_step"].get("timing_s/step", float("inf")),
            default=None,
        )
        baseline_step = baseline["stable_step"].get("timing_s/step") if baseline else None
        if step and baseline_step:
            run["speedup_vs_verl_sync_opd"] = baseline_step / step

    output = args.output or args.result_dir / "summary.json"
    output.write_text(json.dumps({"runs": runs}, indent=2, sort_keys=True) + "\n")
    write_markdown(output.with_suffix(".md"), runs)
    print(output)


if __name__ == "__main__":
    main()
