# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch

from verl.experimental.streamopd_cf.topology import DedicatedOPDPools
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.debug import marked_timer


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
