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
import shutil
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import ray
from omegaconf import DictConfig, open_dict
from transfer_queue import KVBatchMeta

from verl.experimental.streamopd_kv.host_slot_pool import cleanup_host_kv_pools
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


def _rollout_kv_bytes_per_token(model_path: str, dtype: str, tensor_parallel_size: int = 1) -> int:
    """Return one rank's causal KV token footprint from HF config."""

    import json

    config = json.loads((Path(model_path) / "config.json").read_text())
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    local_kv_heads = max(1, math.ceil(kv_heads / tensor_parallel_size))
    hidden_size = int(config["hidden_size"])
    query_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", hidden_size // query_heads))
    dtype_bytes = 4 if str(dtype).lower() in {"float32", "fp32"} else 2
    return layers * local_kv_heads * head_dim * 2 * dtype_bytes


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

    return _plan_rollout_memory(
        total_memory_bytes=total_memory_bytes,
        weight_bytes=weight_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
        requested_max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        utilization_limit=configured_utilization,
        max_num_seqs_explicit=True,
        utilization_explicit=False,
    )


def _stable_rollout_concurrency(capacity: int) -> int:
    """Round an inferred vLLM concurrency down to a graph-friendly shape."""

    if capacity < 1:
        return 0
    if capacity < 8:
        return capacity
    return 1 << (capacity.bit_length() - 1)


def _plan_rollout_memory(
    *,
    total_memory_bytes: int,
    weight_bytes: int,
    kv_bytes_per_token: int,
    requested_max_num_seqs: int,
    max_model_len: int,
    utilization_limit: float,
    max_num_seqs_explicit: bool,
    utilization_explicit: bool,
) -> dict[str, float]:
    """Jointly solve vLLM memory and non-preemptible rollout concurrency."""

    if min(total_memory_bytes, weight_bytes, kv_bytes_per_token, requested_max_num_seqs, max_model_len) < 1:
        raise ValueError("Rollout memory planning inputs must be positive")
    if not 0 < utilization_limit <= 1:
        raise ValueError("Rollout gpu_memory_utilization must be in (0, 1]")
    runtime_reserve = max(2 * 1024**3, weight_bytes * 3 // 20)
    unscaled_kv_budget = math.floor(utilization_limit * total_memory_bytes / 1.15) - weight_bytes - runtime_reserve
    capacity = max(0, unscaled_kv_budget // (kv_bytes_per_token * max_model_len))
    if max_num_seqs_explicit and requested_max_num_seqs > capacity:
        raise ValueError(
            "Rollout memory cap cannot hold the explicitly requested non-preemptible KV: "
            f"max_num_seqs={requested_max_num_seqs}, capacity={capacity}, "
            f"gpu_memory_utilization={utilization_limit:.2f}"
        )
    max_num_seqs = min(requested_max_num_seqs, capacity)
    if not max_num_seqs_explicit:
        max_num_seqs = _stable_rollout_concurrency(max_num_seqs)
    if max_num_seqs < 1:
        raise ValueError("Rollout memory cap cannot fit one non-preemptible trajectory")

    kv_bytes = kv_bytes_per_token * max_num_seqs * max_model_len
    required_bytes = math.ceil((weight_bytes + kv_bytes + runtime_reserve) * 1.15)
    required_utilization = max(0.20, math.ceil(required_bytes / total_memory_bytes * 20) / 20)
    selected_utilization = utilization_limit if utilization_explicit else required_utilization
    if selected_utilization > utilization_limit:
        raise ValueError(
            "Rollout memory cap cannot hold model plus non-preemptible KV: "
            f"required={selected_utilization:.2f}, configured={utilization_limit:.2f}"
        )
    return {
        "gpu_memory_utilization": selected_utilization,
        "weight_gib": weight_bytes / (1024**3),
        "kv_gib": kv_bytes / (1024**3),
        "runtime_reserve_gib": runtime_reserve / (1024**3),
        "required_gib": required_bytes / (1024**3),
        "max_num_seqs": float(max_num_seqs),
        "capacity_at_limit": float(capacity),
        "max_model_len": float(max_model_len),
        "utilization_limit": utilization_limit,
        "max_num_seqs_explicit": float(max_num_seqs_explicit),
        "utilization_explicit": float(utilization_explicit),
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


def _shared_vllm_utilization_limit(
    value,
    *,
    rank_offset: int,
    world_size: int,
    required_free_bytes: int,
) -> dict[str, float]:
    """Cap a shared vLLM allocation while preserving frozen Trainer workspace."""

    rows: list[dict[str, int]] = []

    def visit(item) -> None:
        if isinstance(item, dict):
            if int(item.get("free_bytes", 0)) > 0 and int(item.get("total_bytes", 0)) > 0:
                rows.append(item)
                return
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list | tuple):
            for nested in item:
                visit(nested)

    visit(value)
    selected = rows[rank_offset : rank_offset + world_size]
    if len(selected) != world_size:
        raise RuntimeError(
            "StreamOPD Trainer workers returned incomplete shared Rollout memory stats: "
            f"offset={rank_offset}, world_size={world_size}, rows={len(rows)}"
        )
    free_bytes = min(int(row["free_bytes"]) for row in selected)
    total_bytes = min(int(row["total_bytes"]) for row in selected)
    usable_bytes = free_bytes - required_free_bytes
    utilization_limit = math.floor(usable_bytes / total_bytes * 20) / 20
    if utilization_limit < 0.20:
        raise ValueError(
            "Trainer-shared Rollout has insufficient memory after reverse preflight: "
            f"free={free_bytes / 1024**3:.2f} GiB, "
            f"reverse_reserve={required_free_bytes / 1024**3:.2f} GiB"
        )
    return {
        "utilization_limit": utilization_limit,
        "free_gib": free_bytes / (1024**3),
        "total_gib": total_bytes / (1024**3),
        "reverse_reserve_gib": required_free_bytes / (1024**3),
    }


def _plan_host_kv_handoff(
    *,
    handoff_dir: str,
    global_batch_size: int,
    max_model_len: int,
    kv_bytes_per_token: int,
) -> dict[str, float]:
    """Fail before rollout if the host backing cannot hold worst-case KV."""

    path = Path(handoff_dir)
    parent = next((candidate for candidate in (path, *path.parents) if candidate.exists()), None)
    if parent is None:
        raise ValueError(f"StreamOPD KV handoff path has no existing parent: {handoff_dir}")
    usage = shutil.disk_usage(parent)
    required_bytes = global_batch_size * max_model_len * kv_bytes_per_token
    reserve_bytes = max(4 * 1024**3, required_bytes // 10)
    if required_bytes + reserve_bytes > usage.free:
        raise ValueError(
            "StreamOPD host KV backing is too small for the configured batch and token limit: "
            f"required={(required_bytes + reserve_bytes) / 1024**3:.2f} GiB, "
            f"free={usage.free / 1024**3:.2f} GiB, path={handoff_dir}"
        )
    return {
        "host_kv_required_gib": required_bytes / (1024**3),
        "host_kv_reserve_gib": reserve_bytes / (1024**3),
        "host_kv_free_gib": usage.free / (1024**3),
    }


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


def _planned_reverse_required_free_gib(plan_result) -> float:
    required: list[float] = []

    def visit(value) -> None:
        if isinstance(value, dict):
            if "runtime_required_free_gib" in value:
                required.append(float(value["runtime_required_free_gib"]))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list | tuple):
            for nested in value:
                visit(nested)

    visit(plan_result)
    return max(required, default=0.0)


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
        self._teacher_memory_plan: dict[str, float] = {}
        self._rollout_memory_plan: dict[str, float] = {}
        self._shared_reverse_batch_cap: int | None = None
        self._shared_reverse_reserve_gib = 0.0
        self._early_reverse_plan_result = None
        self._reverse_runtime_required_free_gib = 0.0
        self._teacher_sleeping = False
        self._rollout_deep_sleeping = False
        self._teacher_wake_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamopd-teacher-wake")
        self._teacher_wake_future: Future | None = None
        self._policy_lifecycle_metrics: dict[str, float] = {}
        self._last_trimmed_teacher_chunks = -1

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
        shared_rollout = placement in {"rollout", "union"}
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
        rollout_offset = int(self.config.distillation.n_gpus_per_node) if placement == "union" else 0

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
            self._reverse_runtime_required_free_gib = _planned_reverse_required_free_gib(
                self._early_reverse_plan_result
            )

        device_memory = self.actor_rollout_wg.get_streamopd_device_memory_stats()
        total_memory = _minimum_device_total_bytes(device_memory)
        utilization_limit = configured_utilization
        shared_limit: dict[str, float] = {}
        if shared_rollout:
            shared_limit = _shared_vllm_utilization_limit(
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
            self._rollout_memory_plan = _plan_rollout_memory(
                total_memory_bytes=total_memory,
                weight_bytes=_checkpoint_weight_bytes(model_path),
                kv_bytes_per_token=_rollout_kv_bytes_per_token(model_path, str(rollout.dtype)),
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
            _plan_host_kv_handoff(
                handoff_dir=str(self.config.distillation.streamopd_kv.kv_handoff_dir),
                global_batch_size=int(self.config.data.train_batch_size),
                max_model_len=max_model_len,
                kv_bytes_per_token=_rollout_kv_bytes_per_token(model_path, str(rollout.dtype)),
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
        placement = str(self.config.distillation.streamopd_kv.trainer_placement)
        if placement in {"teacher", "union"} or str(self.config.distillation.streamopd_kv.runtime_profile) != "auto":
            return
        teacher_model = next(iter(self.config.distillation.teacher_models.values()))
        inference = teacher_model.inference
        total_memory = _minimum_device_total_bytes(self.actor_rollout_wg.get_streamopd_device_memory_stats())
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
        self._teacher_memory_plan = _plan_rollout_memory(
            total_memory_bytes=total_memory,
            weight_bytes=math.ceil(_checkpoint_weight_bytes(model_path) / tp_size),
            kv_bytes_per_token=_rollout_kv_bytes_per_token(model_path, str(inference.dtype), tp_size),
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
        if self.streamopd_kv_enabled:
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
            self._reverse_runtime_required_free_gib = _planned_reverse_required_free_gib(plan_result)
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
        stream_config = self.config.distillation.streamopd_kv
        placement = str(stream_config.trainer_placement)
        if (
            not bool(stream_config.enable_pool_sleep)
            or placement not in {"teacher", "union"}
            or self._teacher_sleeping
            or not bool(state["teacher_drained"])
        ):
            return 0.0
        self._check_teacher_wake(wait=True)
        started = time.perf_counter()
        self.teacher_model_manager.sleep()
        elapsed = time.perf_counter() - started
        self._teacher_sleeping = True
        self._policy_lifecycle_metrics["streamopd/teacher_sleep_seconds"] = elapsed
        return elapsed

    def _maybe_deep_sleep_rollout(self) -> float:
        stream_config = self.config.distillation.streamopd_kv
        if (
            not bool(stream_config.enable_pool_sleep)
            or not bool(stream_config.enable_rollout_sleep_level2)
            or str(stream_config.trainer_placement) not in {"rollout", "union"}
            or self._rollout_deep_sleeping
        ):
            return 0.0
        started = time.perf_counter()
        self.checkpoint_manager.sleep_replicas(level=2)
        elapsed = time.perf_counter() - started
        self._rollout_deep_sleeping = True
        self._policy_lifecycle_metrics["streamopd/rollout_sleep_seconds"] = elapsed
        return elapsed

    def _trim_teacher_before_training(self, state: dict) -> tuple[float, float]:
        stream_config = self.config.distillation.streamopd_kv
        if (
            not bool(stream_config.enable_pool_sleep)
            or str(stream_config.trainer_placement) not in {"teacher", "union"}
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
        metrics["streamopd/rollout_sleep_seconds"] = self._policy_lifecycle_metrics.get(
            "streamopd/rollout_sleep_seconds", 0.0
        )
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
        if str(self.config.distillation.streamopd_kv.trainer_placement) not in {"rollout", "union"}:
            return 0.0
        stream_config = self.config.distillation.streamopd_kv
        poll_seconds = max(int(stream_config.scheduler_poll_interval_ms) / 1000.0, 0.01)
        timeout = float(stream_config.scheduler_timeout_seconds)
        started = time.perf_counter()
        while True:
            state = ray.get(self._scheduler.snapshot.remote())
            if int(state["terminal_trajectories"]) == int(state["expected_trajectories"]):
                self._maybe_deep_sleep_rollout()
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
        self._rollout_deep_sleeping = False
        sync_metrics.update(scheduler_metrics)
        sync_metrics.update(self._policy_lifecycle_metrics)
        sync_metrics.update(teacher_memory_metrics)
        sync_metrics.update(rollout_memory_metrics)
        self._pending_sync_metrics = sync_metrics
        self._policy_version = None
        if self.global_steps >= self.total_training_steps and bool(
            self.config.distillation.streamopd_kv.cleanup_after_step
        ):
            cleanup_started = time.perf_counter()
            released_bytes = cleanup_host_kv_pools(str(self.config.distillation.streamopd_kv.kv_handoff_dir))
            self._pending_sync_metrics["streamopd/host_slot_pool_released_gib"] = released_bytes / (1024**3)
            self._pending_sync_metrics["streamopd/host_slot_pool_cleanup_seconds"] = (
                time.perf_counter() - cleanup_started
            )

    def _get_n_gpus_for_throughput(self) -> int:
        trainer_gpus = self.resource_pool_manager.get_n_gpus()
        if str(self.config.distillation.streamopd_kv.trainer_placement) in {"rollout", "union"}:
            return trainer_gpus
        rollout = self.config.actor_rollout_ref.rollout
        return trainer_gpus + rollout.n_gpus_per_node * rollout.nnodes
