# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Failed or timed-out GPU jobs must never become performance measurements."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.streamopd_kv import pilot_8gpu
from benchmarks.streamopd_kv.summarize_pilot_comparison import build_report


def test_all_methods_record_the_same_checkpoint_bucket(tmp_path):
    args = SimpleNamespace(
        models=tmp_path, dataset=tmp_path / "train.parquet", batch=128, warmup=1, devices="0,1,2,3,4,5,6,7"
    )
    methods = (
        "streamopd-kv-union",
        "streamopd-kv-dedicated",
        "streamopd-cf",
        "verl-sync-opd-union",
        "verl-sync-opd-separate",
        "verl-async-opd",
    )
    for method in methods:
        shared = method.endswith("union")
        case = dict(
            method=method,
            student="Qwen3-8B",
            teacher_model="Qwen3-14B",
            tokens=4096,
            trainer=8 if shared else 4,
            rollout=4 if shared else 2,
            teacher=4 if shared else 2,
            tp=2,
        )
        environment = pilot_8gpu.environment(args, case, tmp_path / method)
        assert environment["CHECKPOINT_BUCKET_MB"] == "128"


def test_comparison_and_reports_follow_the_reduced_token_scope(tmp_path):
    settings = list(pilot_8gpu.comparison_settings(tmp_path))
    assert len(settings) == 6 and {tokens for _, _, tokens in settings} == {4096}
    pilot_8gpu.write_json(tmp_path / "comparison_scope.json", {"max_tokens": [4096, 8192]})
    assert len(list(pilot_8gpu.comparison_settings(tmp_path))) == 12
    pilot_8gpu.write_json(tmp_path / "comparison_scope.json", {"max_tokens": [4096]})
    assert list(pilot_8gpu.comparison_settings(tmp_path)) == settings


def test_cf_summary_exposes_all_completion_counts_to_the_pilot(tmp_path):
    lines = []
    for step in (1, 2):
        lines.append(
            f"step:{step} - timing_s/step:10 - response_length/mean:100 - actor/loss:0.1 - actor/grad_norm:1 "
            "- training/off_policy/trajectory_staleness/max:0"
            " - streamopd_cf/scheduler_terminal_trajectories:128"
            " - streamopd_cf/scheduler_completed_teacher_trajectories:128"
            " - streamopd_cf/scheduler_training_trajectories_started:128"
            " - streamopd_cf/stream/trained_trajectories:128 - actor/streamopd_cf/optimizer_steps:1"
        )
    lines.append("Final validation metrics: None")
    (tmp_path / "method.log").write_text("\n".join(lines))
    record = dict(returncode=0, status="failed")
    pilot_8gpu.validate_completed_record(
        SimpleNamespace(warmup=1, batch=128), {"method": "streamopd-cf"}, tmp_path, record
    )
    assert record["status"] == "complete"
    assert record["summary"]["stable_step"]["streamopd_cf/scheduler_training_trajectories_started"] == 128


def test_comparison_never_uses_missing_or_mismatched_baselines(tmp_path):
    selected = {
        "streamopd-kv-union": dict(method="streamopd-kv-union", trainer=8, rollout=4, teacher=4, tp=2),
        "streamopd-cf": dict(method="streamopd-cf", trainer=4, rollout=2, teacher=2, tp=1),
    }
    pilot_8gpu.write_json(tmp_path / "selected_allocations.json", selected)
    assert all(row["step_speedup"] is None for row in build_report(tmp_path))
    case = pilot_8gpu.comparison_case(selected["streamopd-cf"], "streamopd-cf", "Qwen3-4B", "Qwen3-14B", 4096)
    setting = tmp_path / "comparison" / "Qwen3-4B_Qwen3-14B_4096"
    record = dict(
        case=case,
        status="complete",
        step_seconds=4,
        response_tokens_per_second=100,
        summary={"steps": [{"timing_s/step": 5}], "stable_step": {"training/off_policy/trajectory_staleness/max": 0}},
    )
    pilot_8gpu.write_json(setting / pilot_8gpu.name(case) / "result.json", record)
    assert all(row["step_speedup"] is None for row in build_report(tmp_path))
    baseline = pilot_8gpu.comparison_case(selected["streamopd-cf"], "verl-async-opd", "Qwen3-4B", "Qwen3-14B", 4096)
    baseline_path = setting / pilot_8gpu.name(baseline) / "result.json"
    pilot_8gpu.write_json(baseline_path, dict(record, case=dict(baseline, trainer=2)))
    with pytest.raises(ValueError, match="does not match selected allocation"):
        build_report(tmp_path)
    pilot_8gpu.write_json(baseline_path, dict(record, case=baseline, step_seconds=5, response_tokens_per_second=80))
    completed = [row for row in build_report(tmp_path) if row["step_speedup"] is not None]
    assert len(completed) == 1
    assert completed[0]["step_speedup"] == 1.25


@pytest.mark.parametrize("complete", [False, True])
def test_original_worker_log_recovery_still_requires_a_complete_run(tmp_path, complete):
    (tmp_path / "method.log").write_text("console forwarding omitted metrics\n")
    ray_root = tmp_path / "runtime"
    logs = ray_root / "ray" / "session_latest" / "logs"
    logs.mkdir(parents=True)
    (tmp_path / "runtime.json").write_text(json.dumps({"ray_tmpdir": str(ray_root)}))
    lines = [":job_id:01000000", ":actor_name:TaskRunnerV1"]
    for step in range(1, 3 if complete else 2):
        lines.append(
            f"step:{step} - timing_s/step:10 - response_length/mean:100 - actor/loss:0.1 - actor/grad_norm:1 "
            "- training/off_policy/trajectory_staleness/max:0"
            " - streamopd/scheduler_terminal_trajectories:128"
            " - streamopd/scheduler_completed_teacher_trajectories:128"
            " - streamopd/scheduler_training_trajectories_started:128"
        )
    lines.append("Final validation metrics: None")
    (logs / "worker-owner.out").write_text("\n".join(lines) + "\n")
    args = SimpleNamespace(warmup=1, batch=128)
    record = dict(returncode=0, status="failed")
    case = {"method": "streamopd-kv-union"}
    if complete:
        pilot_8gpu.validate_completed_record(args, case, tmp_path, record)
        assert record["status"] == "complete"
        assert record["step_seconds"] == 10
        assert record["response_tokens_per_second"] == 1280
        assert record["summary"]["console_log_error"]
        assert (tmp_path / "trainer_stdout.txt").exists()
    else:
        with pytest.raises(ValueError, match="expected steps"):
            pilot_8gpu.validate_completed_record(args, case, tmp_path, record)
        assert record["status"] == "failed"


@pytest.mark.parametrize(
    "seconds,tokens,retained", [(95, 1050, False), (90, 1100, True), (80, 1300, True), (80, 900, False)]
)
def test_dedicated_kv_and_its_sync_control_are_pruned_together(seconds, tokens, retained):
    results = [
        dict(
            status="complete", case={"method": "streamopd-kv-union"}, step_seconds=100, response_tokens_per_second=1000
        ),
        dict(
            status="complete",
            case={"method": "streamopd-kv-dedicated"},
            step_seconds=seconds,
            response_tokens_per_second=tokens,
        ),
    ]
    methods, decision = pilot_8gpu.choose_kv_modes(results)
    assert decision["retain_dedicated"] is retained
    variants = dict(pilot_8gpu.comparison_variants(methods))
    assert ("streamopd-kv-dedicated" in variants) is retained
    assert ("verl-sync-opd-separate" in variants) is retained
    assert variants["verl-sync-opd-union"] == "streamopd-kv-union"
    assert variants["verl-async-opd"] == "streamopd-cf"


@pytest.mark.parametrize("large_dedicated_seconds,retained", [(50, True), (95, False)])
def test_large_workload_gain_prevents_global_dedicated_pruning(large_dedicated_seconds, retained):
    shared = dict(
        case={"method": "streamopd-kv-union"}, status="complete", step_seconds=100, response_tokens_per_second=100
    )
    dedicated = dict(
        case={"method": "streamopd-kv-dedicated"}, status="complete", step_seconds=95, response_tokens_per_second=105
    )
    largest = dict(
        dedicated, step_seconds=large_dedicated_seconds, response_tokens_per_second=10000 / large_dedicated_seconds
    )
    decision = pilot_8gpu.qualified_kv_decision([shared, dedicated], [shared, largest], 0.1)
    assert decision["reference_workload"]["retain_dedicated"] is False
    assert decision["retain_dedicated"] is retained


def test_teacher_tp_preserves_screening_and_matches_large_shared_pools():
    allocation = dict(method="streamopd-kv-union", teacher=4, tp=2)
    assert pilot_8gpu.teacher_tp_for(allocation, "Qwen3-14B") == 2
    assert pilot_8gpu.teacher_tp_for(allocation, "Qwen3-32B") == 4
    assert pilot_8gpu.teacher_tp_for(allocation, "Qwen3-30B-A3B") == 4
    allocation["teacher"] = 2
    assert pilot_8gpu.teacher_tp_for(allocation, "Qwen3-32B") == 2


def test_sync_baseline_matches_pool_without_inheriting_reverse_overrides():
    allocation = dict(
        method="streamopd-kv-union",
        trainer=8,
        rollout=4,
        teacher=4,
        tp=2,
        tuning="chunk-check",
        overrides=["distillation.streamopd_kv.reverse_chunk_size=4096"],
    )
    case = pilot_8gpu.comparison_case(allocation, "verl-sync-opd-union", "Qwen3-8B", "Qwen3-32B", 8192)
    assert (case["trainer"], case["rollout"], case["teacher"], case["tp"]) == (8, 4, 4, 4)
    assert "overrides" not in case and "tuning" not in case
    assert allocation["overrides"] == ["distillation.streamopd_kv.reverse_chunk_size=4096"]


def test_largest_shared_workload_prefers_teacher_sharding_before_screening_speed():
    fastest = dict(case=dict(method="streamopd-kv-union", teacher=2, trainer=2, tp=1), step_seconds=100)
    more_headroom = dict(case=dict(method="streamopd-kv-union", teacher=4, trainer=4, tp=2), step_seconds=110)
    assert pilot_8gpu.qualification_order([fastest, more_headroom]) == [more_headroom, fastest]
    fastest["case"]["method"] = "streamopd-kv-dedicated"
    more_headroom["case"]["method"] = "streamopd-kv-dedicated"
    assert pilot_8gpu.qualification_order([fastest, more_headroom]) == [more_headroom, fastest]


@pytest.mark.parametrize(
    "message,timeout,reason",
    [("EngineDeadError: EngineCore encountered an issue", 30, "vLLM engine failed"), ("", 0.01, "case timeout")],
)
def test_failed_job_is_stopped_without_performance_metrics(tmp_path, monkeypatch, message, timeout, reason):
    (tmp_path / "environment.json").write_text("{}")
    args = SimpleNamespace(
        root=tmp_path,
        models=tmp_path,
        dataset=tmp_path / "data.parquet",
        warmup=1,
        batch=128,
        devices="0,1,2,3,4,5,6,7",
        retry_failed=False,
        timeout=timeout,
    )
    case = dict(
        method="streamopd-cf",
        student="student",
        teacher_model="teacher",
        trainer=4,
        rollout=2,
        teacher=2,
        tp=1,
        tokens=4096,
    )
    monkeypatch.setattr(pilot_8gpu, "wait_for_idle", lambda devices: None)
    popen = subprocess.Popen

    def synthetic_job(command, **kwargs):
        return popen([sys.executable, "-c", f"import time; print({message!r}, flush=True); time.sleep(60)"], **kwargs)

    monkeypatch.setattr(pilot_8gpu.subprocess, "Popen", synthetic_job)
    result = pilot_8gpu.run_case(args, case, tmp_path / "case")
    assert result["status"] == "failed"
    assert result["returncode"] != 0
    assert result["stop_reason"].startswith(reason)
    assert "step_seconds" not in result
    assert "response_tokens_per_second" not in result
