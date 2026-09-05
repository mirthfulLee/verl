# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


def test_sync_stage_timing_aggregates_rollout_teacher_and_training() -> None:
    from verl.trainer.ppo.v1.trainer_sync import _stage_timing_metrics_from_tags

    metrics = _stage_timing_metrics_from_tags(
        [
            {
                "_stage_rollout_started_at": 101.0,
                "_stage_rollout_completed_at": 105.0,
                "_stage_rollout_request_seconds": 4.0,
                "_stage_teacher_started_at": 105.0,
                "_stage_teacher_completed_at": 108.0,
                "_stage_teacher_request_seconds": 3.0,
            },
            {
                "_stage_rollout_started_at": 102.0,
                "_stage_rollout_completed_at": 107.0,
                "_stage_rollout_request_seconds": 5.0,
                "_stage_teacher_started_at": 107.0,
                "_stage_teacher_completed_at": 109.0,
                "_stage_teacher_request_seconds": 2.0,
            },
            {"is_padding": True},
        ],
        policy_started_at=100.0,
        training_seconds=6.5,
    )

    assert metrics == {
        "stage/training_seconds": 6.5,
        "stage/rollout_span_seconds": 6.0,
        "stage/rollout_makespan_seconds": 7.0,
        "stage/rollout_request_seconds/mean": 4.5,
        "stage/rollout_request_seconds/max": 5.0,
        "stage/teacher_span_seconds": 4.0,
        "stage/teacher_makespan_seconds": 9.0,
        "stage/teacher_request_seconds/mean": 2.5,
        "stage/teacher_request_seconds/max": 3.0,
        "stage/teacher_tail_seconds": 2.0,
    }


def test_streamopd_trainer_does_not_expose_unused_reward_handles() -> None:
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    assert trainer.get_reward_handles() is None
    assert trainer._get_required_batch_multiple(dp_size=3) == 3
    assert trainer._optimizer_updates_per_global_step() == 1
    metric_data = TensorDict({"responses": torch.ones(2, 3, dtype=torch.long)}, batch_size=[2])
    prepared = trainer._prepare_metric_tensors(metric_data)
    assert torch.equal(prepared["rm_scores"], torch.zeros(2, 3))


@pytest.mark.parametrize(
    ("placement", "expected_pools", "teacher_pool"),
    [
        ("teacher", {"global_pool": [2], "rollout_pool": [3]}, "global_pool"),
        ("rollout", {"global_pool": [2], "teacher_pool": [1]}, "teacher_pool"),
        ("union", {"global_pool": [2]}, "global_pool"),
        (
            "dedicated",
            {"global_pool": [2], "teacher_pool": [1], "rollout_pool": [3]},
            "teacher_pool",
        ),
    ],
)
def test_streamopd_resource_pools_follow_trainer_placement(
    monkeypatch: pytest.MonkeyPatch,
    placement: str,
    expected_pools: dict[str, list[int]],
    teacher_pool: str,
) -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.utils import Role
    from verl.trainer.ppo.v1 import trainer_streamopd
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    def fake_base_init(self) -> None:
        self.role_worker_mapping = {Role.ActorRollout: object(), Role.TeacherModel: object()}
        self.mapping = {Role.ActorRollout: "global_pool", Role.TeacherModel: "teacher_pool"}
        self.resource_pool_manager = SimpleNamespace(resource_pool_spec={"global_pool": [2], "teacher_pool": [1]})

    monkeypatch.setattr(PPOTrainer, "_init_resource_pool_mgr", fake_base_init)
    monkeypatch.setattr(trainer_streamopd.ray, "remote", lambda cls: cls)
    monkeypatch.setattr(trainer_streamopd, "need_reference_policy", lambda _config: False)
    trainer = trainer_streamopd.PPOTrainerStreamOPD.__new__(trainer_streamopd.PPOTrainerStreamOPD)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n_gpus_per_node": 3, "nnodes": 1}},
        }
    )
    trainer.placement = TrainerPlacement(placement)

    trainer._init_resource_pool_mgr()

    assert trainer.resource_pool_manager.resource_pool_spec == expected_pools
    assert trainer.mapping[Role.Actor] == "global_pool"
    assert trainer.mapping[Role.TeacherModel] == teacher_pool


@pytest.mark.parametrize("placement", ["teacher", "rollout", "union"])
def test_shared_trainer_state_has_one_load_offload_pair(placement: str) -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    transitions = []
    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement(placement)
    trainer._trainer_state_offloaded = True
    trainer._reverse_plan_result = [{"slot_batch_size": 8.0}]
    trainer._teacher_sleeping = trainer.placement.shares_teacher
    trainer._shared_rollout_sleeping = trainer.placement.shares_rollout
    trainer.actor_rollout_wg = SimpleNamespace(
        offload_streamopd_trainer_state=lambda: transitions.append("offload"),
        load_streamopd_trainer_state=lambda: transitions.append("load"),
    )

    assert trainer._load_trainer_state() >= 0.0
    assert trainer._load_trainer_state() == 0.0
    assert trainer._offload_trainer_state() >= 0.0
    assert trainer._offload_trainer_state() == 0.0
    assert transitions == ["load", "offload"]


def test_shared_trainer_plans_against_sleeping_pool_before_loading() -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    transitions = []
    plan_result = [{"slot_batch_size": 2.0}]
    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement.UNION
    trainer._trainer_state_offloaded = True
    trainer._teacher_sleeping = True
    trainer._shared_rollout_sleeping = True
    trainer._reverse_plan_result = None
    trainer.actor_rollout_wg = SimpleNamespace(
        prepare_streamopd_reverse_plan=lambda: transitions.append("plan") or plan_result,
        load_streamopd_trainer_state=lambda: transitions.append("load"),
    )

    def configure_reverse_plan(result) -> None:
        assert result is plan_result
        transitions.append("configure")
        trainer._reverse_plan_result = result

    trainer._configure_reverse_plan = configure_reverse_plan

    assert trainer._load_trainer_state() >= 0.0
    assert transitions == ["plan", "configure", "load"]


def test_shared_trainer_cannot_load_before_inference_pool_sleeps() -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement.UNION
    trainer._trainer_state_offloaded = True
    trainer._teacher_sleeping = True
    trainer._shared_rollout_sleeping = False
    trainer.actor_rollout_wg = SimpleNamespace(load_streamopd_trainer_state=lambda: None)

    with pytest.raises(RuntimeError, match="shared Rollout"):
        trainer._load_trainer_state()


@pytest.mark.parametrize(
    ("placement", "expected"),
    [("union", ["sleep", "sync", "cleanup"]), ("dedicated", ["sync", "cleanup"])],
)
def test_initial_weight_sync_releases_shared_teacher_first(placement: str, expected: list[str], monkeypatch) -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1.trainer_streamopd import PPOTrainerStreamOPD

    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trainer_streamopd.update_streamopd_weights",
        lambda manager, step, **kwargs: manager.update_weights(step),
    )
    transitions = []
    trainer = PPOTrainerStreamOPD.__new__(PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement(placement)
    trainer.global_steps = 0
    trainer._maybe_sleep_teacher = lambda _state: transitions.append("sleep")
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda _step, **_kwargs: transitions.append("sync"))
    trainer.actor_rollout_wg = SimpleNamespace(release_streamopd_allocator_cache=lambda: transitions.append("cleanup"))

    trainer._publish_initial_weights()

    assert transitions == expected


@pytest.mark.parametrize(
    ("backend", "shares_rollout"), [("host", True), ("host", False), ("nccl", False), ("naive", False)]
)
def test_phase_exclusive_host_weight_sync_serializes_trainer_and_rollout(monkeypatch, backend, shares_rollout) -> None:
    from verl.checkpoint_engine import base as checkpoint_base
    from verl.experimental.streamopd_kv.checkpoint import update_streamopd_weights

    events = []

    class Replica:
        workers = [object()]

        async def sleep(self, level=None):
            events.append(("rollout-sleep", level))

        async def release_kv_cache(self):
            events.append("weights-wake")

        async def resume_kv_cache(self):
            events.append("kv-wake")

        async def abort_all_requests(self):
            events.append("abort")

        async def resume_generation(self):
            events.append("resume-generation")

    class ActorGroup:
        world_size = 1

        def update_weights(self, **kwargs):
            events.append(("trainer-publish", kwargs["mode"]))
            return [{"sender_metric": 1.0}]

        def execute_checkpoint_engine(self, methods):
            assert methods == ["finalize"]
            events.append("trainer-finalize")
            return [None]

        def release_streamopd_allocator_cache(self):
            events.append("trainer-release")

    class RolloutGroup:
        world_size = 1

        def __init__(self, **_kwargs):
            pass

        def update_weights(self, **_kwargs):
            events.append("rollout-receive")
            return [None]

        def execute_checkpoint_engine(self, methods):
            assert methods == ["finalize"]
            events.append("rollout-finalize")
            return [None]

    monkeypatch.setattr(checkpoint_base, "RayWorkerGroup", RolloutGroup)
    monkeypatch.setattr(checkpoint_base.ray, "get", lambda values: values)
    manager = checkpoint_base.CheckpointEngineManager.__new__(checkpoint_base.CheckpointEngineManager)
    manager.backend = backend
    manager.actor_wg = ActorGroup()
    manager.replicas = [Replica()]
    manager.build_process_group = lambda _rollout: events.append("build")

    metrics = asyncio.run(
        update_streamopd_weights.__wrapped__(
            manager,
            global_steps=3,
            shares_rollout=shares_rollout,
        )
    )

    if not shares_rollout:
        if backend == "naive":
            assert events == [("trainer-publish", "naive")]
            assert metrics == {}
        else:
            assert events == [
                "abort",
                "weights-wake",
                "build",
                ("trainer-publish", backend),
                "rollout-receive",
                "trainer-finalize",
                "rollout-finalize",
                "kv-wake",
                "resume-generation",
            ]
            assert metrics == {"sender_metric": 1.0}
        return
    assert events == [
        ("rollout-sleep", 2),
        "build",
        ("trainer-publish", "host"),
        "trainer-release",
        "weights-wake",
        "rollout-receive",
        "trainer-finalize",
        "rollout-finalize",
        "kv-wake",
    ]
    assert metrics["sender_metric"] == 1.0
    assert "abort" not in events
    assert "resume-generation" not in events


def test_shared_rollout_sleeps_once_after_all_trajectories_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1 import trainer_streamopd

    class Snapshot:
        @staticmethod
        def remote():
            return object()

    sleep_levels = []
    trainer = trainer_streamopd.PPOTrainerStreamOPD.__new__(trainer_streamopd.PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement.ROLLOUT
    trainer._shared_rollout_sleeping = False
    trainer.config = OmegaConf.create(
        {"distillation": {"streamopd_kv": {"scheduler_poll_interval_ms": 1, "scheduler_timeout_seconds": 1}}}
    )
    trainer._scheduler = SimpleNamespace(snapshot=Snapshot())
    transfer_drains = []
    trainer._rollout_runtime = SimpleNamespace(wait_for_streamopd_kv_transfers=lambda: transfer_drains.append(True))
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda *, level: sleep_levels.append(level))
    monkeypatch.setattr(
        trainer_streamopd.ray,
        "get",
        lambda _value: {"terminal_trajectories": 128, "expected_trajectories": 128},
    )

    trainer._wait_for_shared_rollout_idle()
    trainer._wait_for_shared_rollout_idle()

    assert sleep_levels == [2]
    assert transfer_drains == [True]
    assert trainer._shared_rollout_sleeping


def test_auto_shared_teacher_runtime_defers_reverse_plan_until_pool_sleep() -> None:
    from verl.experimental.streamopd_kv.placement import TrainerPlacement
    from verl.trainer.ppo.v1 import trainer_streamopd

    trainer = trainer_streamopd.PPOTrainerStreamOPD.__new__(trainer_streamopd.PPOTrainerStreamOPD)
    trainer.placement = TrainerPlacement.TEACHER
    trainer._reverse_plan_result = None
    reverse_plans = []

    def prepare_reverse_plan():
        reverse_plans.append(True)
        return [{"slot_batch_size": 8.0}]

    trainer.actor_rollout_wg = SimpleNamespace(
        prepare_streamopd_reverse_plan=prepare_reverse_plan,
    )
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 128},
            "distillation": {
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "streamopd_kv": {
                    "runtime_profile": "auto",
                    "planner_explicit_options": [],
                    "reverse_slot_max_tokens": 8192,
                    "reverse_batch_max_tokens": 1048576,
                    "reverse_slot_reserve_gib": 4.0,
                },
                "teacher_models": {
                    "teacher_model": {
                        "model_path": "/teacher",
                        "inference": {
                            "tensor_model_parallel_size": 2,
                            "data_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_num_seqs": 128,
                            "max_num_batched_tokens": 8192,
                            "max_model_len": 8193,
                            "dtype": "bfloat16",
                            "gpu_memory_utilization": 0.9,
                        },
                    }
                },
            },
        }
    )

    trainer._prepare_teacher_runtime()

    inference = trainer.config.distillation.teacher_models.teacher_model.inference
    assert inference.gpu_memory_utilization == 0.9
    assert inference.max_num_seqs == 128
    assert trainer._teacher_memory_plan == {"max_num_seqs": 128.0, "exclusive_pool_memory": 1.0}
    assert trainer._reverse_plan_result is None
    assert reverse_plans == []
