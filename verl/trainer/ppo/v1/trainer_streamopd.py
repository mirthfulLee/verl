# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import ray
import torch
from omegaconf import DictConfig, open_dict
from transfer_queue import KVBatchMeta

from verl.experimental.streamopd_kv.host_slot_pool import cleanup_host_kv_pools
from verl.experimental.streamopd_kv.placement import TrainerPlacement
from verl.experimental.streamopd_kv.planning import (
    checkpoint_weight_bytes,
    kv_bytes_per_token,
    minimum_device_total_bytes,
    partition_training_units,
    plan_host_kv,
    plan_teacher_admission,
    plan_vllm_memory,
    planned_reverse_required_free_gib,
    planned_reverse_width,
    shared_vllm_utilization_limit,
)
from verl.experimental.streamopd_kv.ray_worker import StreamOPDActorWorker
from verl.experimental.streamopd_kv.scheduler import StreamOPDTaskScheduler
from verl.single_controller.ray import ResourcePoolManager
from verl.single_controller.ray.base import split_resource_pool
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import LLMServerManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("streamopd")
class PPOTrainerStreamOPD(PPOTrainer):
    """Strict placement-aware StreamOPD trainer.

    The actor worker is trainer-only. Teacher and Rollout remain independent
    model processes whose GPU resource sets may intersect Trainer. Raw
    gradients accumulate across preflight-sized units and weights are published
    only after the final policy-version barrier.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.placement = TrainerPlacement(str(config.distillation.streamopd_kv.trainer_placement))
        self._scheduler = None
        self._policy_version: int | None = None
        self._training_unit_size = 1
        self._teacher_admission_plan: dict[str, int] = {}
        self._teacher_memory_plan: dict[str, float] = {}
        self._rollout_memory_plan: dict[str, float] = {}
        self._shared_reverse_batch_cap: int | None = None
        self._shared_reverse_reserve_gib = 0.0
        self._early_reverse_plan_result = None
        self._reverse_runtime_required_free_gib = 0.0
        self._teacher_sleeping = False
        self._teacher_wake_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamopd-teacher-wake")
        self._teacher_wake_future: Future | None = None
        self._policy_lifecycle_metrics: dict[str, float] = {}
        self._last_trimmed_teacher_chunks = -1

    def _effective_scheduler_policy(self) -> str:
        configured = str(self.config.distillation.streamopd_kv.scheduler_policy)
        # Union Trainer work cannot begin before Rollout EOS and then shares
        # the remaining Teacher resource. No compute overlap is possible, so
        # interleaving only adds switches to an otherwise serial tail.
        if configured == "adaptive" and self.placement is TrainerPlacement.UNION:
            return "teacher_then_train"
        return configured

    def _init_resource_pool_mgr(self) -> None:
        """Replace only the actor and placement mappings built by PPOTrainer."""

        if need_reference_policy(self.config):
            raise NotImplementedError("StreamOPD does not support a reference policy")
        super()._init_resource_pool_mgr()
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        self.role_worker_mapping.pop(actor_role)
        self.mapping.pop(actor_role)
        self.role_worker_mapping[Role.Actor] = ray.remote(StreamOPDActorWorker)
        self.mapping[Role.Actor] = "global_pool"

        resource_pool_spec = dict(self.resource_pool_manager.resource_pool_spec)
        if self.placement.shares_teacher:
            self.mapping[Role.TeacherModel] = "global_pool"
            resource_pool_spec.pop("teacher_pool", None)
        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def _is_teacher_colocated(self) -> bool:
        return self.placement.shares_teacher

    def _uses_external_checkpoint_engine(self) -> bool:
        return True

    def _setup(self):
        scheduler_name = f"verl-streamopd-scheduler-{uuid.uuid4().hex}"
        with open_dict(self.config.distillation.streamopd_kv):
            self.config.distillation.streamopd_kv.scheduler_actor_name = scheduler_name
        teacher_resources, trainer_resources = self.placement.resource_sets
        self._scheduler = (
            ray.remote(StreamOPDTaskScheduler)
            .options(
                name=scheduler_name,
                lifetime="non_detached",
            )
            .remote(teacher_resources, trainer_resources)
        )
        super()._setup()

    def _create_llm_server_manager(self, actor_rollout_resource_pool) -> LLMServerManager:
        shared_rollout = self.placement.shares_rollout
        stream_config = self.config.distillation.streamopd_kv
        rollout = self.config.actor_rollout_ref.rollout
        rollout_world_size = int(rollout.n_gpus_per_node) * int(rollout.nnodes)
        replica_world_size = (
            int(rollout.tensor_model_parallel_size)
            * int(rollout.data_parallel_size)
            * int(rollout.pipeline_model_parallel_size)
        )
        if rollout_world_size < replica_world_size or rollout_world_size % replica_world_size:
            raise ValueError("Rollout GPU allocation must contain a whole number of inference replicas")
        num_replicas = rollout_world_size // replica_world_size
        configured_max_num_seqs = int(rollout.max_num_seqs)
        effective_max_num_seqs = min(
            configured_max_num_seqs,
            math.ceil(int(self.config.data.train_batch_size) / num_replicas),
        )
        auto_profile = str(self.config.distillation.streamopd_kv.runtime_profile) == "auto"
        explicit_options = set(stream_config.get("planner_explicit_options", []))
        configured_utilization = float(rollout.gpu_memory_utilization)
        rollout_offset = (
            int(self.config.distillation.n_gpus_per_node) if self.placement is TrainerPlacement.UNION else 0
        )

        if shared_rollout:
            checkpoint_config = rollout.checkpoint_engine
            original_bucket_mb = int(checkpoint_config.update_weights_bucket_megabytes)
            shared_bucket_mb = min(original_bucket_mb, 128)
            sync_reserve_gib = max(1.0, 2 * shared_bucket_mb / 1024)
            with open_dict(rollout):
                checkpoint_config.update_weights_bucket_megabytes = shared_bucket_mb
            slot_tokens = int(stream_config.reverse_slot_max_tokens)
            max_group_tokens = min(8192, int(stream_config.reverse_batch_max_tokens))
            max_rows = max(1, max_group_tokens // slot_tokens)
            self._shared_reverse_batch_cap = 1 << (max_rows.bit_length() - 1)
            self._shared_reverse_reserve_gib = sync_reserve_gib
            self.actor_rollout_wg.configure_streamopd_reverse_preflight(
                batch_cap=self._shared_reverse_batch_cap,
                additional_reserve_gib=self._shared_reverse_reserve_gib,
            )
            self._early_reverse_plan_result = self.actor_rollout_wg.prepare_streamopd_reverse_plan()
            self._reverse_runtime_required_free_gib = planned_reverse_required_free_gib(self._early_reverse_plan_result)

        device_memory = self.actor_rollout_wg.get_streamopd_device_memory_stats()
        total_memory = minimum_device_total_bytes(device_memory)
        utilization_limit = configured_utilization
        shared_limit: dict[str, float] = {}
        if shared_rollout:
            shared_limit = shared_vllm_utilization_limit(
                device_memory,
                rank_offset=rollout_offset,
                world_size=rollout_world_size,
                required_free_bytes=math.ceil(self._reverse_runtime_required_free_gib * 1024**3),
            )
            utilization_limit = min(utilization_limit, shared_limit["utilization_limit"])

        model_path = str(self.actor_model_config.local_path or self.actor_model_config.path)
        max_model_len = int(
            rollout.max_model_len or (self.config.data.max_prompt_length + self.config.data.max_response_length + 1)
        )
        if auto_profile or shared_rollout:
            utilization_path = "actor_rollout_ref.rollout.gpu_memory_utilization"
            max_num_seqs_path = "actor_rollout_ref.rollout.max_num_seqs"
            utilization_explicit = auto_profile and utilization_path in explicit_options
            max_num_seqs_explicit = auto_profile and max_num_seqs_path in explicit_options
            self._rollout_memory_plan = plan_vllm_memory(
                total_memory_bytes=total_memory,
                weight_bytes=checkpoint_weight_bytes(model_path),
                kv_bytes_per_token=kv_bytes_per_token(model_path, str(rollout.dtype)),
                requested_max_num_seqs=effective_max_num_seqs,
                max_model_len=max_model_len,
                utilization_limit=utilization_limit,
                max_num_seqs_explicit=max_num_seqs_explicit,
                utilization_explicit=utilization_explicit,
            )
            effective_max_num_seqs = int(self._rollout_memory_plan["max_num_seqs"])
            self._rollout_memory_plan.update(
                {
                    "configured_max_num_seqs": float(configured_max_num_seqs),
                    "configured_gpu_memory_utilization": configured_utilization,
                    **{f"shared_{name}": value for name, value in shared_limit.items()},
                }
            )
            with open_dict(rollout):
                rollout.gpu_memory_utilization = self._rollout_memory_plan["gpu_memory_utilization"]
                rollout.max_num_seqs = effective_max_num_seqs

        self._rollout_memory_plan.update(
            plan_host_kv(
                handoff_dir=str(self.config.distillation.streamopd_kv.kv_handoff_dir),
                global_batch_size=int(self.config.data.train_batch_size),
                max_model_len=max_model_len,
                kv_bytes_per_token=kv_bytes_per_token(model_path, str(rollout.dtype)),
            )
        )

        if not shared_rollout:
            if self._rollout_memory_plan:
                logger.info("StreamOPD dedicated Rollout memory preflight: %s", self._rollout_memory_plan)
            return super()._create_llm_server_manager(actor_rollout_resource_pool)

        self._rollout_memory_plan.update(
            {
                "checkpoint_bucket_mb": float(shared_bucket_mb),
                "checkpoint_sync_reserve_gib": sync_reserve_gib,
                "reverse_batch_cap": float(self._shared_reverse_batch_cap),
            }
        )
        if rollout_offset or rollout_world_size < actor_rollout_resource_pool.world_size:
            split_sizes = []
            if rollout_offset:
                split_sizes.append(rollout_offset)
            split_sizes.append(rollout_world_size)
            remainder = actor_rollout_resource_pool.world_size - rollout_offset - rollout_world_size
            if remainder:
                split_sizes.append(remainder)
            pools = split_resource_pool(actor_rollout_resource_pool, split_size=split_sizes)
            rollout_resource_pool = pools[1 if rollout_offset else 0]
        else:
            rollout_resource_pool = actor_rollout_resource_pool
        logger.info("StreamOPD shared Rollout memory preflight: %s", self._rollout_memory_plan)
        return LLMServerManager.create(
            config=self.config,
            worker_group=None,
            rollout_resource_pool=rollout_resource_pool,
            colocate_without_worker_group=True,
        )

    def _prepare_teacher_runtime(self) -> None:
        if self.placement.shares_teacher or str(self.config.distillation.streamopd_kv.runtime_profile) != "auto":
            return
        teacher_model = next(iter(self.config.distillation.teacher_models.values()))
        inference = teacher_model.inference
        total_memory = minimum_device_total_bytes(self.actor_rollout_wg.get_streamopd_device_memory_stats())
        tp_size = int(inference.tensor_model_parallel_size)
        replica_size = (
            tp_size
            * int(inference.get("data_parallel_size", 1))
            * int(inference.get("pipeline_model_parallel_size", 1))
        )
        teacher_world_size = int(self.config.distillation.n_gpus_per_node) * int(self.config.distillation.nnodes)
        replicas = max(1, teacher_world_size // replica_size)
        requested_max_num_seqs = min(
            int(inference.max_num_seqs),
            math.ceil(int(self.config.data.train_batch_size) / replicas),
        )
        explicit_options = set(self.config.distillation.streamopd_kv.get("planner_explicit_options", []))
        utilization_path = "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization"
        max_num_seqs_path = "distillation.teacher_models.teacher_model.inference.max_num_seqs"
        model_path = str(teacher_model.model_path)
        self._teacher_memory_plan = plan_vllm_memory(
            total_memory_bytes=total_memory,
            weight_bytes=math.ceil(checkpoint_weight_bytes(model_path) / tp_size),
            kv_bytes_per_token=kv_bytes_per_token(model_path, str(inference.dtype), tp_size),
            requested_max_num_seqs=requested_max_num_seqs,
            max_model_len=int(inference.max_model_len),
            utilization_limit=float(inference.gpu_memory_utilization),
            max_num_seqs_explicit=max_num_seqs_path in explicit_options,
            utilization_explicit=utilization_path in explicit_options,
        )
        self._teacher_memory_plan.update(
            {
                "configured_max_num_seqs": float(inference.max_num_seqs),
                "configured_gpu_memory_utilization": float(inference.gpu_memory_utilization),
            }
        )
        with open_dict(inference):
            inference.max_num_seqs = int(self._teacher_memory_plan["max_num_seqs"])
            inference.gpu_memory_utilization = self._teacher_memory_plan["gpu_memory_utilization"]
        logger.info("StreamOPD dedicated Teacher memory preflight: %s", self._teacher_memory_plan)

    def on_init_end(self):
        # All shape and memory plans are frozen before the first policy version.
        if self._early_reverse_plan_result is None:
            if self._shared_reverse_batch_cap is not None:
                self.actor_rollout_wg.configure_streamopd_reverse_preflight(
                    batch_cap=self._shared_reverse_batch_cap,
                    additional_reserve_gib=self._shared_reverse_reserve_gib,
                )
            plan_result = self.actor_rollout_wg.prepare_streamopd_reverse_plan()
        else:
            plan_result = self._early_reverse_plan_result
        logger.info("StreamOPD reverse preflight: %s", plan_result)
        self._reverse_runtime_required_free_gib = planned_reverse_required_free_gib(plan_result)
        fallback = int(self.config.distillation.streamopd_kv.reverse_batch_size)
        local_width = planned_reverse_width(plan_result, fallback)
        try:
            dp_mapping = self.actor_rollout_wg._query_dispatch_info("actor")
            dp_size = max(dp_mapping) + 1
        except (AttributeError, ValueError):
            dp_size = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
        self._training_unit_size = min(int(self.config.data.train_batch_size), local_width * dp_size)
        self.parameter_sync_step = len(
            partition_training_units(int(self.config.data.train_batch_size), self._training_unit_size)
        )
        stream_config = self.config.distillation.streamopd_kv
        teacher_config = next(iter(self.config.distillation.teacher_models.values())).inference
        self._teacher_admission_plan = plan_teacher_admission(
            expected_trajectories=int(self.config.data.train_batch_size),
            trajectory_tokens=int(stream_config.reverse_slot_max_tokens),
            vllm_capacity_tokens=int(self.teacher_model_manager.collect_kv_cache_capacity_tokens()),
            page_size=int(stream_config.teacher_prefill_kv_page_size),
            max_batched_tokens=int(teacher_config.max_num_batched_tokens or teacher_config.max_model_len),
            initial_chunk_tokens=int(stream_config.token_chunk_size),
            train_launch_width=self._training_unit_size,
            trajectory_cap=int(stream_config.teacher_prefill_max_active_trajectories),
            token_cap=int(stream_config.teacher_prefill_max_active_kv_tokens),
        )
        with open_dict(stream_config):
            stream_config.teacher_prefill_max_active_trajectories = self._teacher_admission_plan["active_trajectories"]
            stream_config.teacher_prefill_max_active_kv_tokens = self._teacher_admission_plan["active_kv_tokens"]
        logger.info("StreamOPD Teacher admission preflight: %s", self._teacher_admission_plan)
        # Warm the cross-process metrics RPC before the first timed policy
        # step, then reset again at each policy boundary.
        self.actor_rollout_wg.reset_streamopd_memory_stats()
        self.teacher_model_manager.reset_device_memory_stats()
        self.llm_server_manager.reset_device_memory_stats()
        self.checkpoint_manager.update_weights(self.global_steps)

    def prepare_step(self) -> dict:
        self._policy_version = self.global_steps - 1
        self._policy_lifecycle_metrics = {}
        self._last_trimmed_teacher_chunks = -1
        reset_started = time.perf_counter()
        self.actor_rollout_wg.reset_streamopd_memory_stats()
        self.teacher_model_manager.reset_device_memory_stats()
        self.llm_server_manager.reset_device_memory_stats()
        reset_seconds = time.perf_counter() - reset_started
        ray.get(
            self._scheduler.begin_policy.remote(
                self._policy_version,
                int(self.config.data.train_batch_size),
                self._effective_scheduler_policy(),
                self._training_unit_size,
                not self._teacher_sleeping,
            )
        )
        if self._teacher_sleeping:
            policy_version = self._policy_version

            def wake_teacher() -> float:
                started = time.perf_counter()
                self.teacher_model_manager.wake_up()
                ray.get(self._scheduler.teacher_wake_completed.remote(policy_version))
                self._teacher_sleeping = False
                return time.perf_counter() - started

            self._teacher_wake_future = self._teacher_wake_executor.submit(wake_teacher)
        dispatch_started = time.perf_counter()
        metrics = super().prepare_step()
        metrics["streamopd/rollout_dispatch_seconds"] = time.perf_counter() - dispatch_started
        metrics["streamopd/memory_stats_reset_seconds"] = reset_seconds
        metrics.update(
            {f"streamopd/teacher_plan_{name}": float(value) for name, value in self._teacher_admission_plan.items()}
        )
        metrics.update(
            {f"streamopd/rollout_plan_{name}": float(value) for name, value in self._rollout_memory_plan.items()}
        )
        metrics.update(
            {
                f"streamopd/teacher_runtime_plan_{name}": float(value)
                for name, value in self._teacher_memory_plan.items()
            }
        )
        teacher_runtime = next(iter(self.config.distillation.teacher_models.values())).inference
        runtime_profile = {
            "trajectory_tokens": int(self.config.distillation.streamopd_kv.reverse_slot_max_tokens),
            "token_chunk_size": int(self.config.distillation.streamopd_kv.token_chunk_size),
            "teacher_max_batched_tokens": int(teacher_runtime.max_num_batched_tokens),
            "teacher_gpu_memory_utilization": float(teacher_runtime.gpu_memory_utilization),
            "teacher_max_num_seqs": int(teacher_runtime.max_num_seqs),
            "rollout_gpu_memory_utilization": float(self.config.actor_rollout_ref.rollout.gpu_memory_utilization),
            "rollout_max_num_seqs": int(self.config.actor_rollout_ref.rollout.max_num_seqs),
        }
        metrics["streamopd/runtime_profile_auto"] = float(
            str(self.config.distillation.streamopd_kv.runtime_profile) == "auto"
        )
        metrics.update({f"streamopd/runtime_profile_{name}": float(value) for name, value in runtime_profile.items()})
        metrics["streamopd/scheduler_topology_fallback"] = float(
            self._effective_scheduler_policy() != str(self.config.distillation.streamopd_kv.scheduler_policy)
        )
        return metrics

    def _check_teacher_wake(self, *, wait: bool = False) -> None:
        future = self._teacher_wake_future
        if future is None or (not wait and not future.done()):
            return
        self._policy_lifecycle_metrics["streamopd/teacher_wake_seconds"] = future.result()
        self._teacher_wake_future = None

    def _maybe_sleep_teacher(self, state: dict) -> float:
        if not self.placement.shares_teacher or self._teacher_sleeping or not bool(state["teacher_drained"]):
            return 0.0
        self._check_teacher_wake(wait=True)
        started = time.perf_counter()
        self.teacher_model_manager.sleep()
        elapsed = time.perf_counter() - started
        self._teacher_sleeping = True
        self._policy_lifecycle_metrics["streamopd/teacher_sleep_seconds"] = elapsed
        return elapsed

    def _trim_teacher_before_training(self, state: dict) -> tuple[float, float]:
        if (
            not self.placement.shares_teacher
            or self._teacher_sleeping
            or bool(state["teacher_drained"])
            or int(state["teacher_chunks"]) == self._last_trimmed_teacher_chunks
        ):
            return 0.0, 0.0
        started = time.perf_counter()
        trim_result = self.teacher_model_manager.trim_device_memory(
            int(self._reverse_runtime_required_free_gib * 1024**3)
        )
        self._last_trimmed_teacher_chunks = int(state["teacher_chunks"])
        self._policy_lifecycle_metrics["streamopd/teacher_trim_free_before_gib"] = int(
            trim_result["free_before_bytes"]
        ) / (1024**3)
        self._policy_lifecycle_metrics["streamopd/teacher_trim_free_after_gib"] = int(
            trim_result["free_after_bytes"]
        ) / (1024**3)
        return time.perf_counter() - started, int(trim_result["freed_bytes"]) / (1024**3)

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        """Consume an early reverse cohort, then normal Teacher/Trainer microbatches."""
        train_batch_size = int(self.config.data.train_batch_size)
        batch_sizes = partition_training_units(train_batch_size, self._training_unit_size)
        prepare_metrics = self.prepare_step()
        metrics_aggregator = MetricsAggregator()
        if prepare_metrics:
            metrics_aggregator.add_step_metrics(prepare_metrics)
        if self._effective_scheduler_policy() == "teacher_then_train":
            with marked_timer("gen", timing_raw, color="red"):
                teacher_drain_wait = self._wait_for_teacher_drain()
            metrics_aggregator.add_step_metrics({"streamopd/teacher_drain_wait_seconds": teacher_drain_wait})
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

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        """Run the direct-distillation pipeline without PPO-only forwards."""

        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()
        batch = self._balance_batch(batch, metrics=metrics)
        with marked_timer("update_actor", timing_raw, color="red"):
            return self._update_actor(batch, metrics=metrics)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        return dp_size

    def _optimizer_updates_per_global_step(self) -> int:
        return 1

    def _actor_update_extra_info(self) -> dict[str, int]:
        return {
            "streamopd_accumulation_step": self.local_trigger_step,
            "streamopd_accumulation_steps": self._streamopd_runtime_accumulation_steps,
        }

    def _prepare_metric_tensors(self, data):
        if "rm_scores" not in data:
            data["rm_scores"] = torch.zeros_like(data["responses"], dtype=torch.float32)
        return data

    def _wait_for_teacher_drain(self) -> float:
        stream_config = self.config.distillation.streamopd_kv
        # This is a coarse global-batch barrier, not latency-sensitive queue
        # arbitration. Avoid flooding the Ray scheduler while rollout workers
        # and terminal Teacher prefills are still active.
        poll_seconds = max(int(stream_config.scheduler_poll_interval_ms) / 1000.0, 0.1)
        timeout = float(stream_config.scheduler_timeout_seconds)
        started = time.perf_counter()
        while True:
            state = ray.get(self._scheduler.snapshot.remote())
            if bool(state["teacher_drained"]):
                self._maybe_sleep_teacher(state)
                return time.perf_counter() - started
            self._check_teacher_wake()
            if time.perf_counter() - started > timeout:
                raise TimeoutError(f"timed out waiting for StreamOPD Teacher drain: {state}")
            time.sleep(poll_seconds)

    def on_sample_end(self):
        # Placement-aware admission happens immediately before actor update.
        return

    def get_reward_handles(self):
        # Direct StreamOPD is validated to use neither task rewards nor policy
        # gradients. Passing reward workers into AgentLoop would add a
        # per-trajectory RPC whose output is discarded by the reverse loss.
        return None

    def _update_actor(self, batch, metrics: dict):
        if self._policy_version is None:
            raise RuntimeError("StreamOPD training started outside an active policy step")
        trajectory_count = sum(not tag.get("is_padding", False) for tag in batch.tags)
        metrics["streamopd/rollout_pool_wait_seconds"] = self._wait_for_shared_rollout_idle()
        wait_seconds = self._acquire_training(trajectory_count)
        metrics["streamopd/scheduler_wait_seconds"] = wait_seconds
        state = ray.get(self._scheduler.snapshot.remote())
        trim_seconds, trim_gib = self._trim_teacher_before_training(state)
        metrics["streamopd/teacher_trim_seconds"] = trim_seconds
        metrics["streamopd/teacher_trim_freed_gib"] = trim_gib
        metrics["streamopd/teacher_sleep_seconds"] = self._maybe_sleep_teacher(state)
        try:
            return super()._update_actor(batch, metrics)
        finally:
            ray.get(self._scheduler.training_finished.remote(self._policy_version))

    def _wait_for_shared_rollout_idle(self) -> float:
        if not self.placement.shares_rollout:
            return 0.0
        stream_config = self.config.distillation.streamopd_kv
        poll_seconds = max(int(stream_config.scheduler_poll_interval_ms) / 1000.0, 0.01)
        timeout = float(stream_config.scheduler_timeout_seconds)
        started = time.perf_counter()
        while True:
            state = ray.get(self._scheduler.snapshot.remote())
            if int(state["terminal_trajectories"]) == int(state["expected_trajectories"]):
                return time.perf_counter() - started
            if time.perf_counter() - started > timeout:
                raise TimeoutError(f"timed out waiting for Trainer-shared Rollout pool to become idle: {state}")
            time.sleep(poll_seconds)

    def _acquire_training(self, trajectory_count: int) -> float:
        stream_config = self.config.distillation.streamopd_kv
        threshold = 0
        poll_seconds = int(stream_config.scheduler_poll_interval_ms) / 1000.0
        timeout = float(stream_config.scheduler_timeout_seconds)
        started = time.perf_counter()
        registered = False
        ray.get(
            self._scheduler.training_waiting.remote(
                self._policy_version,
                threshold,
                trajectory_count,
            )
        )
        registered = True
        try:
            while True:
                granted = ray.get(self._scheduler.try_training_started.remote(self._policy_version, threshold))
                if granted:
                    registered = False
                    return time.perf_counter() - started
                self._check_teacher_wake()
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
        self._check_teacher_wake(wait=True)
        if not self._teacher_sleeping:
            self._maybe_sleep_teacher({"teacher_drained": True})
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
        sync_metrics.update(self._policy_lifecycle_metrics)
        sync_metrics.update(teacher_memory_metrics)
        sync_metrics.update(rollout_memory_metrics)
        self._pending_sync_metrics = sync_metrics
        self._policy_version = None
        if self.global_steps >= self.total_training_steps:
            cleanup_started = time.perf_counter()
            released_bytes = cleanup_host_kv_pools(str(self.config.distillation.streamopd_kv.kv_handoff_dir))
            self._pending_sync_metrics["streamopd/host_slot_pool_released_gib"] = released_bytes / (1024**3)
            self._pending_sync_metrics["streamopd/host_slot_pool_cleanup_seconds"] = (
                time.perf_counter() - cleanup_started
            )

    def _get_n_gpus_for_throughput(self) -> int:
        trainer_gpus = self.resource_pool_manager.get_n_gpus()
        if self.placement.shares_rollout:
            return trainer_gpus
        rollout = self.config.actor_rollout_ref.rollout
        return trainer_gpus + rollout.n_gpus_per_node * rollout.nnodes
