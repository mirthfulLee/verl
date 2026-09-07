# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch

from verl.experimental.streamopd_cf.batching import AutoBatchActorWorker
from verl.experimental.streamopd_cf.topology import DedicatedOPDPools
from verl.experimental.streamopd_kv.checkpoint import update_streamopd_weights
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import ResourcePoolManager
from verl.single_controller.ray.base import split_resource_pool
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.debug import marked_timer
from verl.utils.device import get_torch_device
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.rollout.llm_server import LLMServerManager


@register_trainer("separate_sync")
class PPOTrainerSeparateSync(DedicatedOPDPools, PPOTrainerSync):
    """Full-trajectory Teacher and native Trainer F+B on independent GPUs.

    Exactly one complete policy batch is submitted and consumed before each
    optimizer update and blocking weight synchronization. No async prefetch or
    partially stale trajectory is allowed. Direct OPD needs no PPO-only forward.
    """

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

    def _step_once(self, metrics, timing_raw, sample_batch_size):
        with marked_timer("gen", timing_raw):
            batch, off_policy = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="train", batch_size=sample_batch_size
            )
            metrics.update(off_policy)
        # AgentLoopWorkerTQ publishes samples only after full-trajectory Teacher
        # scoring. Check the behavior version as well as the synchronous barrier.
        real_tags = [tag for tag in batch.tags if not tag.get("is_padding", False)]
        if any(
            tag["min_global_steps"] != self.global_steps - 1 or tag["max_global_steps"] != self.global_steps - 1
            for tag in real_tags
        ):
            raise RuntimeError("separate_sync received a stale rollout")
        metrics["separate_sync/supervised_trajectories_at_training"] = len(real_tags)
        batch = self._balance_batch(batch, metrics=metrics)
        with marked_timer("update_actor", timing_raw):
            return self._update_actor(batch, metrics)

    def on_sample_end(self):
        # Independent inference GPUs remain resident between policy batches.
        return

    def get_reward_handles(self):
        return None

    def _prepare_metric_tensors(self, data):
        if "rm_scores" not in data:
            data["rm_scores"] = torch.zeros_like(data["responses"], dtype=torch.float32)
        return data


class PhaseSharedNativeActorWorker(AutoBatchActorWorker):
    """Release native Trainer state before shared inference mappings wake."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_streamopd_allocator_cache(self):
        get_torch_device().synchronize()
        optimizer = self.actor.engine.optimizer
        if optimizer is not None:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                        state[key] = value.to("cpu", non_blocking=False)
        self.actor.engine.to(device="cpu", model=True, optimizer=False, grad=True)
        aggressive_empty_cache(force_sync=True)


@register_trainer("union_sync")
class PPOTrainerUnionSync(PPOTrainerSeparateSync):
    """Native full-trajectory OPD with Trainer on the inference GPU union."""

    actor_worker_cls = PhaseSharedNativeActorWorker

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
        self.teacher_model_manager.sleep(level=2)
        update_streamopd_weights(self.checkpoint_manager, self.global_steps, shares_rollout=True)
        self.teacher_model_manager.wake_up()

    def _update_actor(self, batch, metrics):
        with marked_timer("inference_sleep", self.timing_raw):
            self.teacher_model_manager.sleep(level=2)
            self.checkpoint_manager.sleep_replicas(level=2)
        return super()._update_actor(batch, metrics)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw):
            self._pending_sync_metrics = update_streamopd_weights(
                self.checkpoint_manager, self.global_steps, shares_rollout=True
            )
            self.teacher_model_manager.wake_up()
