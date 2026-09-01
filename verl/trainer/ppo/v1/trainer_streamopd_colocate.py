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
from pathlib import Path

import ray
from omegaconf import DictConfig, open_dict
from transfer_queue import KVBatchMeta

from verl.experimental.streamopd_kv.scheduler import StreamOPDTaskScheduler
from verl.single_controller.ray.base import split_resource_pool
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import LLMServerManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _checkpoint_weight_bytes(model_path: str) -> int:
    """Read the local safetensors payload size without loading model tensors."""

    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        import json

        payload = json.loads(index.read_text())
        total_size = int(payload.get("metadata", {}).get("total_size", 0))
        if total_size > 0:
            return total_size
    shards = list(root.glob("*.safetensors"))
    if shards:
        return sum(path.stat().st_size for path in shards)
    raise ValueError(f"cannot determine safetensors weight size for shared Rollout model: {model_path}")


def _rollout_kv_bytes_per_token(model_path: str, dtype: str) -> int:
    """Return one TP=1 causal KV token's byte footprint from HF config."""

    import json

    config = json.loads((Path(model_path) / "config.json").read_text())
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    hidden_size = int(config["hidden_size"])
    query_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", hidden_size // query_heads))
    dtype_bytes = 4 if str(dtype).lower() in {"float32", "fp32"} else 2
    return layers * kv_heads * head_dim * 2 * dtype_bytes


def _plan_shared_rollout_memory(
    *,
    total_memory_bytes: int,
    weight_bytes: int,
    kv_bytes_per_token: int,
    max_num_seqs: int,
    max_model_len: int,
    configured_utilization: float,
) -> dict[str, float]:
    """Reserve enough vLLM memory to avoid preemption on Trainer-shared GPUs."""

    if min(total_memory_bytes, weight_bytes, kv_bytes_per_token, max_num_seqs, max_model_len) < 1:
        raise ValueError("shared Rollout memory planning inputs must be positive")
    if not 0 < configured_utilization <= 1:
        raise ValueError("Rollout gpu_memory_utilization must be in (0, 1]")
    kv_bytes = kv_bytes_per_token * max_num_seqs * max_model_len
    runtime_reserve = max(2 * 1024**3, weight_bytes * 3 // 20)
    required_bytes = math.ceil((weight_bytes + kv_bytes + runtime_reserve) * 1.15)
    required_utilization = max(0.18, math.ceil(required_bytes / total_memory_bytes * 1000) / 1000)
    if required_utilization > configured_utilization:
        raise ValueError(
            "shared Rollout memory cap cannot hold model plus non-preemptible KV: "
            f"required={required_utilization:.3f}, configured={configured_utilization:.3f}"
        )
    return {
        "gpu_memory_utilization": required_utilization,
        "weight_gib": weight_bytes / (1024**3),
        "kv_gib": kv_bytes / (1024**3),
        "runtime_reserve_gib": runtime_reserve / (1024**3),
        "required_gib": required_bytes / (1024**3),
        "max_num_seqs": float(max_num_seqs),
        "max_model_len": float(max_model_len),
    }


def _minimum_device_total_bytes(value) -> int:
    totals: list[int] = []

    def visit(item) -> None:
        if isinstance(item, dict):
            if int(item.get("total_bytes", 0)) > 0:
                totals.append(int(item["total_bytes"]))
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list | tuple):
            for nested in item:
                visit(nested)

    visit(value)
    if not totals:
        raise RuntimeError("StreamOPD Trainer workers returned no device memory capacity")
    return min(totals)


def _streamopd_batch_sizes(
    train_batch_size: int,
    micro_batch_size: int,
    reverse_batch_size: int,
    *,
    planned_unit_size: int | None = None,
) -> list[int]:
    """Return accumulation units with an early reverse-capacity trigger."""
    if planned_unit_size is not None:
        if train_batch_size < 1 or planned_unit_size < 1:
            raise ValueError("planned StreamOPD training unit must be positive")
        return [
            min(planned_unit_size, train_batch_size - start) for start in range(0, train_batch_size, planned_unit_size)
        ]
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


def _planned_local_reverse_width(plan_result, fallback: int) -> int:
    """Extract the conservative rank-local preflight width from a Ray result."""

    widths: list[int] = []

    def visit(value) -> None:
        if isinstance(value, dict):
            width = value.get("slot_batch_size")
            if width is not None and int(width) > 0:
                widths.append(int(width))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list | tuple):
            for nested in value:
                visit(nested)

    visit(plan_result)
    return min(widths, default=fallback)


def _plan_teacher_admission(
    *,
    expected_trajectories: int,
    trajectory_tokens: int,
    vllm_capacity_tokens: int,
    page_size: int,
    max_batched_tokens: int,
    initial_chunk_tokens: int,
    train_launch_width: int,
    trajectory_cap: int = 0,
    token_cap: int = 0,
) -> dict[str, int]:
    """Choose one stable Teacher cohort before policy version zero."""

    values = (
        expected_trajectories,
        trajectory_tokens,
        vllm_capacity_tokens,
        page_size,
        max_batched_tokens,
        initial_chunk_tokens,
        train_launch_width,
    )
    if any(value < 1 for value in values) or trajectory_cap < 0 or token_cap < 0:
        raise ValueError("StreamOPD Teacher admission inputs must be positive and caps non-negative")
    trajectory_tokens = math.ceil(trajectory_tokens / page_size) * page_size
    if token_cap and token_cap < trajectory_tokens:
        raise ValueError("StreamOPD Teacher token cap cannot fit one trajectory reservation")
    # Leave room for vLLM scheduler metadata, output blocks, and page
    # fragmentation. This derives a static plan from vLLM's profiled blocks;
    # it is not an allocator query in the policy loop.
    safe_capacity = max(trajectory_tokens, (vllm_capacity_tokens * 3 // 4 // page_size) * page_size)
    if token_cap:
        safe_capacity = min(safe_capacity, token_cap)
    capacity_width = max(1, safe_capacity // trajectory_tokens)
    prefill_wave = max(1, max_batched_tokens // min(initial_chunk_tokens, trajectory_tokens))
    scheduler_width = 2 * prefill_wave
    width = min(expected_trajectories, capacity_width, scheduler_width)
    if trajectory_cap:
        width = min(width, trajectory_cap)
    if width >= train_launch_width:
        width = max(train_launch_width, width // train_launch_width * train_launch_width)
    return {
        "active_trajectories": width,
        "active_kv_tokens": width * trajectory_tokens,
        "vllm_capacity_tokens": vllm_capacity_tokens,
        "safe_capacity_tokens": safe_capacity,
        "trajectory_tokens": trajectory_tokens,
        "prefill_wave": prefill_wave,
    }


@register_trainer("streamopd_colocate")
class PPOTrainerStreamOPDColocate(PPOTrainer):
    """Strict placement-aware StreamOPD trainer.

    The actor worker is trainer-only. Teacher and Rollout remain independent
    model processes whose GPU resource sets may intersect Trainer. Raw
    gradients accumulate across preflight-sized units and weights are published
    only after the final policy-version barrier.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._scheduler = None
        self._policy_version: int | None = None
        self._training_unit_size = 1
        self._teacher_admission_plan: dict[str, int] = {}
        self._rollout_memory_plan: dict[str, float] = {}
        self._shared_reverse_batch_cap: int | None = None
        self._shared_reverse_reserve_gib = 0.0

    def _effective_scheduler_policy(self) -> str:
        configured = str(self.config.distillation.streamopd_kv.scheduler_policy)
        placement = str(self.config.distillation.streamopd_kv.trainer_placement)
        # Union Trainer work cannot begin before Rollout EOS and then shares
        # the remaining Teacher resource. No compute overlap is possible, so
        # interleaving only adds switches to an otherwise serial tail.
        if configured == "adaptive" and placement == "union":
            return "teacher_then_train"
        return configured

    def _setup(self):
        scheduler_name = f"verl-streamopd-scheduler-{uuid.uuid4().hex}"
        with open_dict(self.config.distillation.streamopd_kv):
            self.config.distillation.streamopd_kv.scheduler_actor_name = scheduler_name
        placement = str(self.config.distillation.streamopd_kv.trainer_placement)
        if placement == "teacher":
            teacher_resources = trainer_resources = ("teacher_trainer",)
        elif placement == "union":
            teacher_resources = ("teacher",)
            trainer_resources = ("teacher", "rollout")
        elif placement == "rollout":
            teacher_resources = ("teacher",)
            trainer_resources = ("rollout_trainer",)
        else:
            teacher_resources = ("teacher",)
            trainer_resources = ("trainer",)
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
        placement = str(self.config.distillation.streamopd_kv.trainer_placement)
        if placement not in {"rollout", "union"}:
            return super()._create_llm_server_manager(actor_rollout_resource_pool)

        rollout = self.config.actor_rollout_ref.rollout
        total_memory = _minimum_device_total_bytes(self.actor_rollout_wg.get_streamopd_device_memory_stats())
        model_path = str(self.actor_model_config.local_path or self.actor_model_config.path)
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
        max_model_len = int(
            rollout.max_model_len or (self.config.data.max_prompt_length + self.config.data.max_response_length + 1)
        )
        self._rollout_memory_plan = _plan_shared_rollout_memory(
            total_memory_bytes=total_memory,
            weight_bytes=_checkpoint_weight_bytes(model_path),
            kv_bytes_per_token=_rollout_kv_bytes_per_token(model_path, str(rollout.dtype)),
            max_num_seqs=effective_max_num_seqs,
            max_model_len=max_model_len,
            configured_utilization=float(rollout.gpu_memory_utilization),
        )
        checkpoint_config = rollout.checkpoint_engine
        original_bucket_mb = int(checkpoint_config.update_weights_bucket_megabytes)
        shared_bucket_mb = min(original_bucket_mb, 128)
        sync_reserve_gib = max(1.0, 2 * shared_bucket_mb / 1024)
        self._rollout_memory_plan.update(
            {
                "checkpoint_bucket_mb": float(shared_bucket_mb),
                "checkpoint_sync_reserve_gib": sync_reserve_gib,
                "configured_max_num_seqs": float(configured_max_num_seqs),
            }
        )
        with open_dict(rollout):
            rollout.gpu_memory_utilization = self._rollout_memory_plan["gpu_memory_utilization"]
            rollout.max_num_seqs = effective_max_num_seqs
            checkpoint_config.update_weights_bucket_megabytes = shared_bucket_mb
        stream_config = self.config.distillation.streamopd_kv
        slot_tokens = int(stream_config.reverse_slot_max_tokens)
        max_group_tokens = min(8192, int(stream_config.reverse_batch_max_tokens))
        max_rows = max(1, max_group_tokens // slot_tokens)
        self._shared_reverse_batch_cap = 1 << (max_rows.bit_length() - 1)
        self._shared_reverse_reserve_gib = sync_reserve_gib
        self._rollout_memory_plan["reverse_batch_cap"] = float(self._shared_reverse_batch_cap)

        rollout_offset = int(self.config.distillation.n_gpus_per_node) if placement == "union" else 0
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

    def on_init_end(self):
        # All shape and memory plans are frozen before the first policy version.
        if self.streamopd_kv_enabled:
            if self._shared_reverse_batch_cap is not None:
                self.actor_rollout_wg.configure_streamopd_reverse_preflight(
                    batch_cap=self._shared_reverse_batch_cap,
                    additional_reserve_gib=self._shared_reverse_reserve_gib,
                )
            plan_result = self.actor_rollout_wg.prepare_streamopd_reverse_plan()
            fallback = int(self.config.distillation.streamopd_kv.reverse_batch_size)
            local_width = _planned_local_reverse_width(plan_result, fallback)
            try:
                dp_mapping = self.actor_rollout_wg._query_dispatch_info("actor")
                dp_size = max(dp_mapping) + 1
            except (AttributeError, ValueError):
                dp_size = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
            self._training_unit_size = min(int(self.config.data.train_batch_size), local_width * dp_size)
            self.parameter_sync_step = len(
                _streamopd_batch_sizes(
                    int(self.config.data.train_batch_size),
                    1,
                    1,
                    planned_unit_size=self._training_unit_size,
                )
            )
            stream_config = self.config.distillation.streamopd_kv
            teacher_config = next(iter(self.config.distillation.teacher_models.values())).inference
            self._teacher_admission_plan = _plan_teacher_admission(
                expected_trajectories=int(self.config.data.train_batch_size),
                trajectory_tokens=int(stream_config.reverse_slot_max_tokens),
                vllm_capacity_tokens=int(self.teacher_model_manager.collect_kv_cache_capacity_tokens()),
                page_size=int(stream_config.teacher_prefill_kv_page_size),
                max_batched_tokens=int(teacher_config.max_num_batched_tokens or teacher_config.max_model_len),
                initial_chunk_tokens=int(stream_config.teacher_initial_chunk_size),
                train_launch_width=self._training_unit_size,
                trajectory_cap=int(stream_config.teacher_prefill_max_active_trajectories),
                token_cap=int(stream_config.teacher_prefill_max_active_kv_tokens),
            )
            with open_dict(stream_config):
                stream_config.teacher_prefill_max_active_trajectories = self._teacher_admission_plan[
                    "active_trajectories"
                ]
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
            )
        )
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
        metrics["streamopd/scheduler_topology_fallback"] = float(
            self._effective_scheduler_policy() != str(self.config.distillation.streamopd_kv.scheduler_policy)
        )
        return metrics

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        """Consume an early reverse cohort, then normal Teacher/Trainer microbatches."""
        train_batch_size = int(self.config.data.train_batch_size)
        stream_config = self.config.distillation.streamopd_kv
        batch_sizes = _streamopd_batch_sizes(
            train_batch_size,
            int(stream_config.micro_batch_size),
            int(stream_config.reverse_batch_size),
            planned_unit_size=self._training_unit_size,
        )
        prepare_metrics = self.prepare_step()
        metrics_aggregator = MetricsAggregator()
        if prepare_metrics:
            metrics_aggregator.add_step_metrics(prepare_metrics)
        if bool(stream_config.posthoc_ablation) or self._effective_scheduler_policy() == "teacher_then_train":
            with marked_timer("gen", timing_raw, color="red"):
                teacher_drain_wait = self._wait_for_teacher_drain()
            drain_metrics = {"streamopd/teacher_drain_wait_seconds": teacher_drain_wait}
            if bool(stream_config.posthoc_ablation):
                drain_metrics["streamopd/posthoc_ready_wait_seconds"] = teacher_drain_wait
            metrics_aggregator.add_step_metrics(drain_metrics)
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
                return time.perf_counter() - started
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
        try:
            return super()._update_actor(batch, metrics)
        finally:
            ray.get(self._scheduler.training_finished.remote(self._policy_version))

    def _wait_for_shared_rollout_idle(self) -> float:
        if str(self.config.distillation.streamopd_kv.trainer_placement) not in {"rollout", "union"}:
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
        if str(self.config.distillation.streamopd_kv.trainer_placement) in {"rollout", "union"}:
            return trainer_gpus
        rollout = self.config.actor_rollout_ref.rollout
        return trainer_gpus + rollout.n_gpus_per_node * rollout.nnodes
