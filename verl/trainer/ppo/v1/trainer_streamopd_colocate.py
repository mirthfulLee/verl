# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging
import os
import time
import uuid

import ray
from omegaconf import DictConfig, open_dict
from transfer_queue import KVBatchMeta

from verl.experimental.streamopd_kv.scheduler import StreamOPDTaskScheduler
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _streamopd_batch_sizes(train_batch_size: int, micro_batch_size: int, reverse_batch_size: int) -> list[int]:
    """Return accumulation units with an early reverse-capacity trigger."""
    if train_batch_size < 1 or micro_batch_size < 1 or reverse_batch_size < 1:
        raise ValueError("StreamOPD batch sizes must be positive")
    if train_batch_size % micro_batch_size:
        raise ValueError("StreamOPD global batch must be divisible by micro_batch_size")
    first = min(micro_batch_size, reverse_batch_size)
    sizes = [first]
    remaining = train_batch_size - first
    while remaining:
        size = min(micro_batch_size, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


@register_trainer("streamopd_colocate")
class PPOTrainerStreamOPDColocate(PPOTrainer):
    """Strict two-pool StreamOPD trainer.

    The actor worker is trainer-only. Rollout is a standalone pool, while the
    frozen teacher and sharded actor trainer share the global pool. Raw
    gradients accumulate across rollout microbatches and weights are published
    only after the final policy-version barrier.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._scheduler = None
        self._policy_version: int | None = None

    def _setup(self):
        scheduler_name = f"verl-streamopd-scheduler-{uuid.uuid4().hex}"
        with open_dict(self.config.distillation.streamopd_kv):
            self.config.distillation.streamopd_kv.scheduler_actor_name = scheduler_name
        self._scheduler = (
            ray.remote(StreamOPDTaskScheduler).options(name=scheduler_name, lifetime="non_detached").remote()
        )
        super()._setup()

    def on_init_end(self):
        # The standalone rollout pool starts asleep so the initial checkpoint
        # load cannot race with weight publication.
        if self.streamopd_kv_enabled:
            self.actor_rollout_wg.prepare_streamopd_reverse_plan()
            # Warm the cross-process metrics RPC before the first timed policy
            # step, then reset again at each policy boundary.
            self.actor_rollout_wg.reset_streamopd_memory_stats()
            self.teacher_model_manager.reset_device_memory_stats()
            self.llm_server_manager.reset_device_memory_stats()
        self.checkpoint_manager.update_weights(self.global_steps)

    def prepare_step(self) -> dict:
        self._policy_version = self.global_steps - 1
        reset_started = time.perf_counter()
        self.actor_rollout_wg.reset_streamopd_memory_stats()
        self.teacher_model_manager.reset_device_memory_stats()
        self.llm_server_manager.reset_device_memory_stats()
        reset_seconds = time.perf_counter() - reset_started
        ray.get(
            self._scheduler.begin_policy.remote(
                self._policy_version,
                int(self.config.data.train_batch_size),
            )
        )
        dispatch_started = time.perf_counter()
        metrics = super().prepare_step()
        metrics["streamopd/rollout_dispatch_seconds"] = time.perf_counter() - dispatch_started
        metrics["streamopd/memory_stats_reset_seconds"] = reset_seconds
        return metrics

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        """Consume an early reverse cohort, then normal Teacher/Trainer microbatches."""
        train_batch_size = int(self.config.data.train_batch_size)
        stream_config = self.config.distillation.streamopd_kv
        batch_sizes = _streamopd_batch_sizes(
            train_batch_size,
            int(stream_config.micro_batch_size),
            int(stream_config.reverse_batch_size),
        )
        prepare_metrics = self.prepare_step()
        metrics_aggregator = MetricsAggregator()
        if prepare_metrics:
            metrics_aggregator.add_step_metrics(prepare_metrics)
        if bool(stream_config.posthoc_ablation):
            with marked_timer("gen", timing_raw, color="red"):
                posthoc_wait = self._wait_for_posthoc_ready()
            metrics_aggregator.add_step_metrics({"streamopd/posthoc_ready_wait_seconds": posthoc_wait})
        combined_keys: list = []
        combined_tags: list = []
        self._streamopd_runtime_accumulation_steps = len(batch_sizes)
        try:
            for trigger_idx, sample_batch_size in enumerate(batch_sizes):
                self.local_trigger_step = trigger_idx
                iter_metrics: dict = {}
                batch = self._step_once(iter_metrics, timing_raw, sample_batch_size)
                sample_count = sum(not tag.get("is_padding", False) for tag in batch.tags)
                metrics_aggregator.add_step_metrics(iter_metrics, sample_count=sample_count)
                combined_keys.extend(batch.keys)
                combined_tags.extend(batch.tags)
            metrics.update(metrics_aggregator.get_aggregated_metrics())
            return KVBatchMeta(partition_id="train", keys=combined_keys, tags=combined_tags)
        finally:
            del self._streamopd_runtime_accumulation_steps

    def _wait_for_posthoc_ready(self) -> float:
        stream_config = self.config.distillation.streamopd_kv
        # This is a coarse global-batch barrier, not latency-sensitive queue
        # arbitration. Avoid flooding the Ray scheduler while rollout workers
        # and terminal Teacher prefills are still active.
        poll_seconds = max(int(stream_config.scheduler_poll_interval_ms) / 1000.0, 0.1)
        timeout = float(stream_config.scheduler_timeout_seconds)
        started = time.perf_counter()
        while True:
            state = ray.get(self._scheduler.snapshot.remote())
            if bool(state["posthoc_ready"]):
                return time.perf_counter() - started
            if time.perf_counter() - started > timeout:
                raise TimeoutError(f"timed out waiting for post-hoc StreamOPD barrier: {state}")
            time.sleep(poll_seconds)

    def on_sample_end(self):
        # Rollout owns a separate pool and remains resident while this
        # microbatch enters reverse training.
        return

    def _update_actor(self, batch, metrics: dict):
        if self._policy_version is None:
            raise RuntimeError("StreamOPD training started outside an active policy step")
        wait_seconds = self._acquire_training()
        metrics["streamopd/scheduler_wait_seconds"] = wait_seconds
        try:
            return super()._update_actor(batch, metrics)
        finally:
            ray.get(self._scheduler.training_finished.remote(self._policy_version))

    def _acquire_training(self) -> float:
        stream_config = self.config.distillation.streamopd_kv
        threshold = int(stream_config.teacher_priority_threshold)
        poll_seconds = int(stream_config.scheduler_poll_interval_ms) / 1000.0
        timeout = float(stream_config.scheduler_timeout_seconds)
        started = time.perf_counter()
        registered = False
        ray.get(self._scheduler.training_waiting.remote(self._policy_version, threshold))
        registered = True
        try:
            while True:
                granted = ray.get(self._scheduler.try_training_started.remote(self._policy_version, threshold))
                if granted:
                    registered = False
                    return time.perf_counter() - started
                if time.perf_counter() - started > timeout:
                    state = ray.get(self._scheduler.snapshot.remote())
                    raise TimeoutError(f"timed out waiting for StreamOPD teacher priority queue: {state}")
                time.sleep(poll_seconds)
        finally:
            if registered:
                ray.get(self._scheduler.training_waiting_cancelled.remote(self._policy_version))

    def on_step_end(self):
        if self._policy_version is None:
            raise RuntimeError("StreamOPD policy barrier has no active version")
        with marked_timer("policy_barrier", self.timing_raw, color="cyan"):
            scheduler_metrics = ray.get(self._scheduler.end_policy.remote(self._policy_version))
        memory_started = time.perf_counter()
        teacher_memory = self.teacher_model_manager.collect_device_memory_stats()
        teacher_memory_metrics = {
            f"streamopd/teacher_memory/{name.removesuffix('_bytes')}_gib": value / (1024**3)
            for name, value in teacher_memory.items()
        }
        teacher_memory_metrics["streamopd/teacher_memory/collect_seconds"] = time.perf_counter() - memory_started
        rollout_memory = self.llm_server_manager.collect_device_memory_stats()
        rollout_memory_metrics = {
            f"streamopd/rollout_memory/{name.removesuffix('_bytes')}_gib": value / (1024**3)
            for name, value in rollout_memory.items()
        }
        with marked_timer("update_weights", self.timing_raw, color="red"):
            sync_metrics = dict(self.checkpoint_manager.update_weights(self.global_steps) or {})
        sync_metrics.update(scheduler_metrics)
        sync_metrics.update(teacher_memory_metrics)
        sync_metrics.update(rollout_memory_metrics)
        self._pending_sync_metrics = sync_metrics
        self._policy_version = None

    def _get_n_gpus_for_throughput(self) -> int:
        trainer_gpus = self.resource_pool_manager.get_n_gpus()
        rollout = self.config.actor_rollout_ref.rollout
        return trainer_gpus + rollout.n_gpus_per_node * rollout.nnodes
