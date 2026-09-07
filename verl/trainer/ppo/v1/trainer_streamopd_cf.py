# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import time
import uuid
from pathlib import Path

import ray
import torch
from omegaconf import open_dict

from verl.experimental.streamopd_cf.profiling import summarize_overlap
from verl.experimental.streamopd_cf.stream import CFTrainingStream
from verl.experimental.streamopd_cf.topology import DedicatedOPDPools
from verl.experimental.streamopd_cf.worker import StreamOPDCFActorWorker
from verl.experimental.streamopd_kv.config import get_streamopd_teacher
from verl.experimental.streamopd_kv.planning import plan_teacher_admission
from verl.experimental.streamopd_kv.replica_group import VLLMReplicaGroup
from verl.experimental.streamopd_kv.scheduler import StreamOPDTaskScheduler
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


@register_trainer("streamopd_cf")
class PPOTrainerStreamOPDCF(DedicatedOPDPools, PPOTrainer):
    """StreamOPD-CF: ascending chunked forward and one backward per microbatch."""

    actor_worker_cls = StreamOPDCFActorWorker

    def _setup(self):
        name = f"verl-streamopd-cf-{uuid.uuid4().hex}"
        with open_dict(self.config.distillation.streamopd_cf):
            self.config.distillation.streamopd_cf.scheduler_actor_name = name
            self.config.distillation.streamopd_cf.training_stream_actor_name = name + "-training"
        self._scheduler = ray.remote(StreamOPDTaskScheduler).options(name=name).remote(("teacher",), ("trainer",))
        rollout = self.config.actor_rollout_ref.rollout
        self._training_stream = (
            ray.remote(CFTrainingStream)
            .options(name=name + "-training")
            .remote(
                int(rollout.prompt_length + rollout.response_length),
                self.config.distillation.streamopd_cf.scheduler_timeout_seconds,
            )
        )
        super()._setup()

    def on_init_end(self):
        settings = self.config.distillation.streamopd_cf
        teacher = get_streamopd_teacher(self.distillation_config)[1]
        runtime = VLLMReplicaGroup(
            server for servers in self.teacher_model_manager.server_handles.values() for server in servers
        )
        rollout = self.config.actor_rollout_ref.rollout
        teacher_world = int(self.config.distillation.nnodes) * int(self.config.distillation.n_gpus_per_node)
        replica_size = (
            teacher.inference.tensor_model_parallel_size
            * teacher.inference.pipeline_model_parallel_size
            * teacher.inference.data_parallel_size
        )
        plan = plan_teacher_admission(
            expected_trajectories=int(self.config.data.train_batch_size),
            trajectory_tokens=int(rollout.prompt_length + rollout.response_length + 1),
            vllm_capacity_tokens=runtime.collect_kv_cache_capacity_tokens(),
            page_size=settings.teacher_prefill_kv_page_size,
            max_batched_tokens=int(teacher.inference.max_num_batched_tokens),
            initial_chunk_tokens=settings.token_chunk_size,
            teacher_replicas=teacher_world // replica_size,
            trajectory_cap=settings.teacher_prefill_max_active_trajectories,
            token_cap=settings.teacher_prefill_max_active_kv_tokens,
            variable_reservations=True,
        )
        with open_dict(settings):
            settings.teacher_prefill_max_active_trajectories = plan["active_trajectories"]
            settings.teacher_prefill_max_active_kv_tokens = plan["active_kv_tokens"]
        self.checkpoint_manager.update_weights(self.global_steps)

    def step(self, metrics, timing_raw):
        policy_version = self.global_steps - 1
        batch_size = int(self.config.data.train_batch_size)
        ray.get(self._scheduler.begin_policy.remote(policy_version, batch_size, batch_size, True))
        ray.get(self._training_stream.begin.remote(policy_version, batch_size))
        ray.get(self._scheduler.training_waiting.remote(policy_version, batch_size))
        if not ray.get(self._scheduler.try_training_started.remote(policy_version)):
            raise RuntimeError("independent CF Trainer could not acquire its pool")
        futures = self.actor_rollout_wg.train_stream(
            policy_version,
            batch_size,
            self.config.actor_rollout_ref.rollout.temperature,
            self.config.distillation.streamopd_cf.training_stream_actor_name,
        )
        try:
            self.prepare_step()
            with marked_timer("train_stream", timing_raw):
                outputs = ray.get(futures)
            batch, off_policy = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=batch_size,
            )
            metrics.update(off_policy)
            # All window backwards and the single update have finished. Drain
            # publication/release RPCs before advancing the behavior policy.
            deadline = time.monotonic() + self.config.distillation.streamopd_cf.scheduler_timeout_seconds
            while not ray.get(self._scheduler.snapshot.remote())["teacher_drained"]:
                if time.monotonic() > deadline:
                    raise TimeoutError("streamopd-cf timed out waiting for complete Teacher supervision")
                time.sleep(0.05)
            collected = {}
            for output in outputs:
                for key, value in tu.get_non_tensor_data(output, "metrics", {}).items():
                    collected.setdefault(f"actor/{key}", []).append(value)
            metrics.update(reduce_metrics(collected))
            metrics["perf/mfu/actor"] = metrics.pop("actor/mfu", 0.0)
            metrics.update(ray.get(self._training_stream.snapshot.remote(policy_version)))
            timelines = [tu.get_non_tensor_data(output, "gpu_timeline", {}) for output in outputs]
            timelines = [timeline for timeline in timelines if timeline]
            if timelines:
                service = ray.get(self._scheduler.timeline.remote(policy_version))
                metrics.update(summarize_overlap(timelines, service))
                directory = self.config.distillation.streamopd_cf.timeline_dir
                if directory:
                    destination = Path(directory)
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / f"step-{self.global_steps}.json").write_text(
                        json.dumps({"service": service, "trainer_gpu": timelines}, indent=2) + "\n"
                    )
        except Exception as exc:
            ray.get(self._training_stream.abort.remote(policy_version, str(exc)))
            raise
        finally:
            ray.get(self._scheduler.training_finished.remote(policy_version))
        self._scheduler_metrics = ray.get(self._scheduler.end_policy.remote(policy_version))
        return batch

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw):
            self._pending_sync_metrics = self.checkpoint_manager.update_weights(self.global_steps) or {}
        self._pending_sync_metrics.update(
            {key.replace("streamopd/", "streamopd_cf/"): value for key, value in self._scheduler_metrics.items()}
        )

    def on_sample_end(self):
        return

    def get_reward_handles(self):
        return None

    def _get_required_batch_multiple(self, dp_size):
        return dp_size

    def _optimizer_updates_per_global_step(self):
        return 1

    def _prepare_metric_tensors(self, data):
        if "rm_scores" not in data:
            data["rm_scores"] = torch.zeros_like(data["responses"], dtype=torch.float32)
        return data
