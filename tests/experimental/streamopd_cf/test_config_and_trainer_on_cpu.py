# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.experimental.streamopd_cf.config import prepare_streamopd_cf_config
from verl.experimental.streamopd_cf.worker import pack_training_group


def config():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "verl/trainer/config"), version_base=None
    ):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "trainer.use_v1=True",
                "data.train_batch_size=8",
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.use_dynamic_bsz=True",
                "trainer.v1.trainer_mode=streamopd_cf",
                "trainer.n_gpus_per_node=2",
                "distillation.enabled=True",
                "distillation.n_gpus_per_node=1",
                "distillation.nnodes=1",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.nnodes=1",
                "actor_rollout_ref.rollout.n_gpus_per_node=1",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                "actor_rollout_ref.rollout.checkpoint_engine.backend=host",
                "distillation.teacher_models.teacher_model.model_path=teacher",
                "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
                "distillation.distillation_loss.loss_mode=forward_kl_topk",
                "distillation.distillation_loss.use_policy_gradient=False",
                "distillation.distillation_loss.use_task_rewards=False",
            ],
        )


def test_streamopd_cf_config_keeps_reverse_connector_disabled():
    cfg = config()
    prepare_streamopd_cf_config(cfg)
    assert cfg.distillation.streamopd_cf.enabled
    assert not cfg.distillation.streamopd_kv.enabled
    assert not cfg.actor_rollout_ref.rollout.engine_kwargs.get("vllm", {}).get("kv_transfer_config")
    additional = cfg.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config
    assert additional.verl_streaming_teacher_logprobs
    assert cfg.trainer.v1.streamopd_cf.parameter_sync_step == 1


@pytest.mark.parametrize("mode", ["streamopd_cf", "separate_sync", "union_sync"])
def test_inference_limits_are_independent_of_training(mode):
    cfg = config()
    cfg.trainer.v1.trainer_mode = mode
    cfg.actor_rollout_ref.actor.use_dynamic_bsz = False
    cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 2
    rollout = cfg.actor_rollout_ref.rollout
    teacher = cfg.distillation.teacher_models.teacher_model.inference
    rollout.max_num_seqs, rollout.max_num_batched_tokens = 0, 4096
    teacher.max_num_seqs, teacher.max_num_batched_tokens = 3, 0
    prepare_streamopd_cf_config(cfg)
    assert rollout.max_num_seqs == cfg.data.train_batch_size
    assert rollout.max_num_batched_tokens == 4096
    assert teacher.max_num_seqs == 3
    assert teacher.max_num_batched_tokens > 0
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 2


@pytest.mark.parametrize(
    "path", ["actor_rollout_ref.model.use_remove_padding", "distillation.distillation_loss.use_chunked_topk"]
)
def test_baseline_auto_budget_rejects_unprofiled_memory_paths(path):
    cfg = config()
    cfg.trainer.v1.trainer_mode = "separate_sync"
    cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = 0
    cfg.actor_rollout_ref.model.use_remove_padding = True
    cfg.distillation.distillation_loss.use_chunked_topk = True
    OmegaConf.update(cfg, path, False)
    with pytest.raises(ValueError, match="baseline auto planning requires"):
        prepare_streamopd_cf_config(cfg)


def test_union_sync_uses_strict_replay_without_streaming():
    from verl.trainer.ppo.v1.trainer_separate_sync import PPOTrainerUnionSync

    cfg = config()
    cfg.trainer.v1.trainer_mode = "union_sync"
    prepare_streamopd_cf_config(cfg)
    assert cfg.trainer.v1.union_sync.parameter_sync_step == 1
    assert not cfg.distillation.streamopd_cf.enabled
    assert not cfg.distillation.streamopd_kv.enabled
    trainer = PPOTrainerUnionSync.__new__(PPOTrainerUnionSync)
    trainer.config = cfg
    trainer._add_prompts_to_generate = lambda *args, **kwargs: None
    buffer = trainer._build_replay_buffer()
    assert buffer.trainer_mode == "sync"
    assert buffer.max_off_policy_threshold == 1


def test_union_sync_reserves_eight_physical_gpus(monkeypatch):
    from verl.experimental.streamopd_cf.topology import DedicatedOPDPools
    from verl.single_controller.ray import ResourcePoolManager
    from verl.trainer.ppo.utils import Role
    from verl.trainer.ppo.v1.trainer_separate_sync import PPOTrainerUnionSync

    cfg = config()
    cfg.trainer.n_gpus_per_node = 8
    cfg.distillation.n_gpus_per_node = 2
    cfg.actor_rollout_ref.rollout.n_gpus_per_node = 6
    cfg.actor_rollout_ref.actor.fsdp_config.param_offload = True
    cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
    trainer = PPOTrainerUnionSync.__new__(PPOTrainerUnionSync)
    trainer.config = cfg

    def dedicated_pools(self):
        self.mapping = {Role.Actor: "global_pool", Role.TeacherModel: "teacher_pool"}
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={"global_pool": [8], "rollout_pool": [6], "teacher_pool": [2]},
            mapping=self.mapping,
        )

    monkeypatch.setattr(DedicatedOPDPools, "_init_resource_pool_mgr", dedicated_pools)
    trainer._init_resource_pool_mgr()
    assert trainer.resource_pool_manager.resource_pool_spec == {"global_pool": [8]}
    assert trainer.mapping[Role.TeacherModel] == trainer.mapping[Role.Actor]
    cfg.actor_rollout_ref.rollout.n_gpus_per_node = 8
    with pytest.raises(ValueError, match="span the disjoint"):
        trainer._init_resource_pool_mgr()


@pytest.mark.parametrize(
    "path,value",
    [
        ("distillation.streamopd_kv.enabled", True),
        ("actor_rollout_ref.rollout.nnodes", 0),
        ("actor_rollout_ref.rollout.checkpoint_engine.backend", "naive"),
        ("distillation.colocate_teacher_with_student", True),
        ("distillation.distillation_loss.use_policy_gradient", True),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", 4),
        ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", -1),
        ("actor_rollout_ref.actor.use_dynamic_bsz", False),
    ],
)
def test_streamopd_cf_rejects_incompatible_config(path, value):
    cfg = config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        prepare_streamopd_cf_config(cfg)


def test_chunk_training_targets_align_across_prompt_boundary_and_padding():
    samples = []
    for tokens, prompt_len, response_mask in [([1, 2, 3, 4, 5], 2, [1, 0, 1]), ([6, 7, 8], 1, [1, 1])]:
        samples.append(
            {
                "input_ids": torch.tensor(tokens),
                "prompts": torch.tensor(tokens[:prompt_len]),
                "response_mask": torch.tensor(response_mask),
                "teacher_ids": torch.arange(len(tokens) * 2).view(len(tokens), 2),
                "teacher_logprobs": torch.zeros(len(tokens), 2),
            }
        )
    inputs, valid, ids, _ = pack_training_group(samples, 2, "cpu", min_length=6)
    assert inputs.tolist() == [[1, 2, 3, 4, 0, 0], [6, 7, 0, 0, 0, 0]]
    assert valid.tolist() == [[False, True, False, True, False, False], [True, True, False, False, False, False]]
    assert ids[0, valid[0]].tolist() == [[2, 3], [6, 7]]


def test_trainer_launches_stream_consumer_before_rollout_and_drains_before_policy_end(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_streamopd_cf

    events = []
    snapshots = iter([False, True])

    class Remote:
        def __init__(self, name):
            self.name = name

        def remote(self, *args):
            events.append(self.name)
            if self.name == "snapshot":
                return {"teacher_drained": next(snapshots)}
            if self.name == "try_training_started":
                return True
            return {}

    trainer = trainer_streamopd_cf.PPOTrainerStreamOPDCF.__new__(trainer_streamopd_cf.PPOTrainerStreamOPDCF)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 8},
            "actor_rollout_ref": {"rollout": {"temperature": 1.0}},
            "distillation": {"streamopd_cf": {"scheduler_timeout_seconds": 1, "training_stream_actor_name": "stream"}},
        }
    )
    trainer.global_steps = 1
    methods = (
        "begin_policy",
        "snapshot",
        "training_waiting",
        "try_training_started",
        "training_finished",
        "end_policy",
    )
    trainer._scheduler = SimpleNamespace(**{name: Remote(name) for name in methods})
    trainer._training_stream = SimpleNamespace(
        begin=Remote("stream_begin"), snapshot=Remote("stream_snapshot"), abort=Remote("stream_abort")
    )
    outputs = [{"metrics": {"loss": 0.2, "mfu": 0}}]
    trainer.actor_rollout_wg = SimpleNamespace(train_stream=lambda *args: events.append("launch") or outputs)
    batch = object()
    trainer.prepare_step = lambda: events.append("rollout")
    trainer.replay_buffer = SimpleNamespace(
        sample=lambda **kwargs: events.append("supervision_published") or (batch, {})
    )

    def get(result):
        if result is outputs:
            events.append("worker_complete")
        return result

    monkeypatch.setattr(trainer_streamopd_cf.ray, "get", get)
    monkeypatch.setattr(trainer_streamopd_cf.time, "sleep", lambda seconds: None)
    assert trainer.step({}, {}) is batch
    assert events == [
        "begin_policy",
        "stream_begin",
        "training_waiting",
        "try_training_started",
        "launch",
        "rollout",
        "worker_complete",
        "supervision_published",
        "snapshot",
        "snapshot",
        "stream_snapshot",
        "training_finished",
        "end_policy",
    ]


def test_benchmark_summary_rejects_incomplete_runs(tmp_path):
    from benchmarks.streamopd_cf.summarize import summarize_run

    log = tmp_path / "partial.log"
    log.write_text("step:1 - timing_s/step:10.0\nFinal validation metrics: None\n")
    with pytest.raises(ValueError, match="expected steps"):
        summarize_run(log)
    log.write_text("\n".join(f"step:{i} - timing_s/step:{seconds}" for i, seconds in [(1, 30), (2, 10), (3, 12)]))
    with pytest.raises(ValueError, match="completion marker"):
        summarize_run(log)
    log.write_text(log.read_text() + "\nFinal validation metrics: None\n")
    result = summarize_run(log)
    assert result["stable_step"]["timing_s/step"] == 11
    assert result["measured_steps"] == 2


def test_async_summary_counts_consumed_work_and_staleness(tmp_path):
    from benchmarks.streamopd_cf.summarize_async_matrix import summarize_case

    log = tmp_path / "async.log"
    log.write_text(
        "\n".join(
            f"step:{step} - timing_s/step:{seconds} - response_length/mean:{tokens} - "
            f"training/off_policy/trajectory_staleness/max:{stale}"
            for step, seconds, tokens, stale in [(1, 100, 99, 0), (2, 20, 10, 1), (3, 10, 20, 2)]
        )
        + "\nFinal validation metrics: None\n"
    )
    result = summarize_case(log, steps=3, warmup=1, policy_batch=8)
    assert result["response_tokens_per_second"] == 8
    assert result["consumed_trajectories_per_second"] == 16 / 30
    assert result["max_staleness"] == 2


def test_matrix_report_includes_matched_sync_and_cf_overlap(tmp_path, monkeypatch):
    import json

    from benchmarks.streamopd_cf.summarize_async_matrix import main

    for method, seconds, stale in (
        ("streamopd-cf", 10, 0),
        ("verl-sync-opd-separate", 20, 0),
        ("verl-async-opd", 8, 1),
    ):
        directory = tmp_path / "default" / method
        directory.mkdir(parents=True)
        (directory / "case.log").write_text(
            f"step:1 - timing_s/step:{seconds} - response_length/mean:100 - "
            f"training/off_policy/trajectory_staleness/max:{stale} - "
            "streamopd_cf/overlap/training_rollout_fraction:0.25\nFinal validation metrics: None\n"
        )
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        ["summarize", str(tmp_path), "--steps", "1", "--warmup", "0", "--settings", "default", "--output", str(output)],
    )
    main()
    result = json.loads((output / "comparison.json").read_text())["default"]
    assert result["cf_response_throughput_over_sync"] == 2
    assert result["cf_response_throughput_over_async"] == 0.8
    assert result["streamopd-cf"]["stable_step"]["streamopd_cf/overlap/training_rollout_fraction"] == 0.25
    assert "verl-sync-opd-separate" in (output / "comparison.md").read_text()


@pytest.mark.parametrize(
    "trained,updates,stale,error",
    [(8, 1, 0, None), (6, 1, 0, "windows finished"), (8, 2, 0, "one optimizer step"), (8, 1, 1, "stale")],
)
def test_stream_summary_checks_complete_policy_update(tmp_path, trained, updates, stale, error):
    from benchmarks.streamopd_cf.summarize import summarize_run

    log = tmp_path / "stream.log"
    log.write_text(
        "step:1 - timing_s/step:10 - streamopd_cf/scheduler_terminal_trajectories:8 - "
        "streamopd_cf/scheduler_completed_teacher_trajectories:8 - "
        f"streamopd_cf/stream/trained_trajectories:{trained} - actor/streamopd_cf/optimizer_steps:{updates} - "
        f"training/off_policy/trajectory_staleness/max:{stale}\nFinal validation metrics: None\n"
    )
    if error:
        with pytest.raises(ValueError, match=error):
            summarize_run(log, expected_steps=1, warmup_steps=0)
    else:
        assert summarize_run(log, expected_steps=1, warmup_steps=0)["measured_steps"] == 1


def test_separate_sync_uses_native_trainer_and_sync_sampling():
    from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
    from verl.trainer.ppo.v1.trainer_separate_sync import PPOTrainerSeparateSync
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

    cfg = config()
    cfg.trainer.v1.trainer_mode = "separate_sync"
    prepare_streamopd_cf_config(cfg)
    assert not cfg.distillation.streamopd_cf.enabled
    assert not cfg.distillation.streamopd_kv.enabled
    additional = cfg.distillation.teacher_models.teacher_model.inference.engine_kwargs.get("vllm", {})
    assert not additional.get("additional_config", {}).get("verl_streaming_teacher_logprobs", False)
    trainer = PPOTrainerSeparateSync(cfg)
    assert type(trainer.replay_buffer) is ReplayBuffer
    assert not isinstance(trainer.replay_buffer, ReplayBufferAsync)
    assert trainer.parameter_sync_step == 1
    assert trainer.actor_worker_cls is ActorRolloutRefWorker
    assert trainer.actor_worker_cls.actor_worker_cls.train_mini_batch is TrainingWorker.train_mini_batch
    assert trainer._uses_external_checkpoint_engine()


def test_sync_placement_adapters_inherit_native_training_pipeline():
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.trainer.ppo.v1.trainer_separate_sync import PPOTrainerSeparateSync, PPOTrainerUnionSync

    for cls in (PPOTrainerSeparateSync, PPOTrainerUnionSync):
        assert cls._step_once is PPOTrainer._step_once
        assert cls.get_reward_handles is PPOTrainer.get_reward_handles
        assert cls._prepare_metric_tensors is PPOTrainer._prepare_metric_tensors


def test_union_sync_uses_native_weight_manager_after_inference_sleep():
    from verl.trainer.ppo.v1.trainer_separate_sync import PPOTrainerUnionSync

    trainer = PPOTrainerUnionSync.__new__(PPOTrainerUnionSync)
    events = []
    trainer.global_steps = 2
    trainer.timing_raw = {}
    trainer.teacher_model_manager = SimpleNamespace(
        sleep=lambda **kwargs: events.append(("teacher_sleep", kwargs)),
        wake_up=lambda: events.append("teacher_wake"),
    )
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda **kwargs: events.append(("rollout_sleep", kwargs)),
        update_weights=lambda step: events.append(("update_weights", step)) or {},
    )
    trainer.on_sample_end()
    trainer.on_step_end()
    assert events == [
        ("teacher_sleep", {"level": 1}),
        ("rollout_sleep", {"level": 2}),
        ("update_weights", 2),
        "teacher_wake",
    ]
