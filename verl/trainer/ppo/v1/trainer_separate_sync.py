# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from verl.experimental.streamopd_cf.topology import DedicatedOPDPools
from verl.single_controller.ray import ResourcePoolManager
from verl.single_controller.ray.base import split_resource_pool
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.debug import marked_timer
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerManager


@register_trainer("separate_sync")
class PPOTrainerSeparateSync(DedicatedOPDPools, PPOTrainerSync):
    """Native synchronous OPD pipeline with independent inference GPU pools."""

    actor_worker_cls = ActorRolloutRefWorker

    def _build_replay_buffer(self):
        sampler = self.config.trainer.v1.sampler
        custom = sampler.get("custom_sampler")
        if custom and custom.get("path"):
            raise ValueError("separate_sync requires the strict synchronous replay buffer")
        return ReplayBuffer(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=1,
            max_off_policy_strategy="drop",
            sampler_kwargs=sampler.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
            train_batch_size=self.config.data.train_batch_size,
            gen_batch_size=self.config.data.get("gen_batch_size") or self.config.data.train_batch_size,
        )

    def on_init_end(self):
        # Actor-only external replicas require an explicit sleep before the
        # native weights-only wake and transfer sequence.
        self.checkpoint_manager.sleep_replicas(level=2)
        super().on_init_end()

    def on_sample_end(self):
        self.checkpoint_manager.sleep_replicas(level=2)


@register_trainer("union_sync")
class PPOTrainerUnionSync(PPOTrainerSeparateSync):
    """Native full-trajectory OPD with Trainer on the inference GPU union."""

    def _init_resource_pool_mgr(self):
        teacher = int(self.config.distillation.n_gpus_per_node)
        rollout = int(self.config.actor_rollout_ref.rollout.n_gpus_per_node)
        if teacher + rollout != int(self.config.trainer.n_gpus_per_node):
            raise ValueError("union_sync Trainer must span the disjoint Teacher and Rollout pools")
        actor = self.config.actor_rollout_ref.actor
        if not actor.fsdp_config.param_offload or not actor.fsdp_config.optimizer_offload:
            raise ValueError("union_sync requires Trainer parameter and optimizer offload")
        super()._init_resource_pool_mgr()
        pools = dict(self.resource_pool_manager.resource_pool_spec)
        pools.pop("rollout_pool")
        pools.pop("teacher_pool")
        self.mapping[Role.TeacherModel] = "global_pool"
        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=pools, mapping=self.mapping)

    def _is_teacher_colocated(self):
        return True

    def _create_llm_server_manager(self, actor_rollout_resource_pool):
        pool = split_resource_pool(
            actor_rollout_resource_pool,
            split_size=[
                self.config.distillation.n_gpus_per_node,
                self.config.actor_rollout_ref.rollout.n_gpus_per_node,
            ],
        )[1]
        return LLMServerManager.create(
            config=self.config, rollout_resource_pool=pool, colocate_without_worker_group=True
        )

    def on_init_end(self):
        self.teacher_model_manager.sleep(level=1)
        super().on_init_end()
        self.teacher_model_manager.wake_up()

    def on_sample_end(self):
        self.teacher_model_manager.sleep(level=1)
        super().on_sample_end()

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw):
            self._pending_sync_metrics = self.checkpoint_manager.update_weights(self.global_steps)
            self.teacher_model_manager.wake_up()
