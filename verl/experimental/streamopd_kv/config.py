# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

from omegaconf import DictConfig, OmegaConf, open_dict
from packaging.version import Version

from verl.base_config import BaseConfig


@dataclass
class StreamOPDKVConfig(BaseConfig):
    """Experimental placement-aware strict OPD execution settings.

    This flag is fail-closed. The initial backend supports text-only,
    single-teacher Qwen3 jobs with exact dense attention and vLLM KV export.
    """

    enabled: bool = False
    # ``auto`` resolves execution-only knobs from batch, sequence length, GPU
    # allocation, and placement. ``manual`` preserves every advanced override
    # for ablations and sensitivity studies.
    runtime_profile: str = "auto"
    # Populated from Hydra CLI overrides before automatic planning. Existing
    # verl options remain the public constraints; this field is diagnostic.
    planner_explicit_options: list[str] = field(default_factory=list)
    token_chunk_size: int = 256
    # Export complete trajectories after EOS. ``eos_host`` copies vLLM's
    # physical pages directly and performs the layout transform on CPU.
    rollout_kv_export_strategy: str = "eos_host"
    # Rollout export is independent of Teacher streaming granularity. Zero
    # resolves to a bounded 2048-token staging chunk before vLLM starts.
    rollout_kv_export_chunk_size: int = 0
    # This also bounds persistent pinned Host staging to one buffer per writer.
    rollout_kv_writer_threads: int = 4
    # Physical GPU allocation is user-controlled. ``teacher`` colocates the
    # fixed-topology trainer with Teacher vLLM; ``rollout`` colocates separate
    # rollout processes on Trainer GPUs; ``union`` spans disjoint Teacher and
    # Rollout subsets; ``dedicated`` gives each role its own pool.
    trainer_placement: str = "union"
    # Zero values are resolved into static caps before worker startup.
    reverse_chunk_size: int = 0
    reverse_chunk_min_size: int = 0
    reverse_page_size: int = 64
    reverse_batch_size: int = 0
    reverse_batch_max_tokens: int = 0
    # Persistent K/V/dK/dV rows planned and allocated before policy version 0.
    # A zero token limit is resolved from data.max_prompt/response_length.
    reverse_slot_max_tokens: int = 0
    reverse_slot_reserve_gib: float = 4.0
    scheduler_poll_interval_ms: int = 10
    scheduler_timeout_seconds: float = 600.0
    scheduler_actor_name: str = ""
    max_pending_teacher_chunks: int = 128
    # Optional preflight caps. Zero lets the scheduler derive stable values
    # from vLLM's profiled KV blocks and the reverse training-unit width.
    teacher_prefill_max_active_trajectories: int = 0
    # Optional global Teacher KV admission cap. Effective live-session count
    # is this budget divided by the configured trajectory length.
    teacher_prefill_max_active_kv_tokens: int = 0
    teacher_prefill_kv_page_size: int = 64
    # Host-side KV prefetch is overlapped with the current reverse unit.  The
    # GPU lease remains limited to one reverse microbatch; depth only controls
    # how many sealed host snapshots may be in flight.
    kv_prefetch_depth: int = 1
    kv_prefetch_workers: int = 4
    kv_handoff_dir: str = "/tmp/verl-streamopd-kv"
    require_same_tokenizer: bool = True
    validate_teacher_artifacts: bool = False
    validate_full_forward_loss: bool = False
    validation_atol: float = 1e-4

    def __post_init__(self) -> None:
        if self.runtime_profile not in {"auto", "manual"}:
            raise ValueError("streamopd_kv.runtime_profile must be 'auto' or 'manual'")
        if self.rollout_kv_export_strategy not in {"eos_host", "eos_triton", "incremental_triton"}:
            raise ValueError(
                "streamopd_kv.rollout_kv_export_strategy must be eos_host, eos_triton, or incremental_triton"
            )
        for name in (
            "token_chunk_size",
            "rollout_kv_writer_threads",
            "reverse_page_size",
            "scheduler_poll_interval_ms",
            "max_pending_teacher_chunks",
            "teacher_prefill_kv_page_size",
            "kv_prefetch_depth",
            "kv_prefetch_workers",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"streamopd_kv.{name} must be positive")
        if not self.kv_handoff_dir:
            raise ValueError("streamopd_kv.kv_handoff_dir must be non-empty")
        for name in (
            "rollout_kv_export_chunk_size",
            "reverse_chunk_size",
            "reverse_chunk_min_size",
            "reverse_batch_size",
            "reverse_batch_max_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"streamopd_kv.{name} must be non-negative")
        if self.reverse_chunk_size and self.reverse_chunk_min_size > self.reverse_chunk_size:
            raise ValueError("streamopd_kv.reverse_chunk_min_size must not exceed reverse_chunk_size")
        if self.reverse_page_size < 16 or self.reverse_page_size & (self.reverse_page_size - 1):
            raise ValueError("streamopd_kv.reverse_page_size must be a power of two and at least 16")
        if self.teacher_prefill_kv_page_size & (self.teacher_prefill_kv_page_size - 1):
            raise ValueError("streamopd_kv.teacher_prefill_kv_page_size must be a power of two")
        if (self.reverse_chunk_size and self.reverse_chunk_size % self.reverse_page_size) or (
            self.reverse_chunk_min_size and self.reverse_chunk_min_size % self.reverse_page_size
        ):
            raise ValueError("StreamOPD reverse chunk sizes must be divisible by reverse_page_size")
        if self.reverse_slot_max_tokens < 0:
            raise ValueError("streamopd_kv.reverse_slot_max_tokens must be non-negative")
        if self.reverse_slot_reserve_gib < 0:
            raise ValueError("streamopd_kv.reverse_slot_reserve_gib must be non-negative")
        if self.validation_atol < 0:
            raise ValueError("streamopd_kv.validation_atol must be non-negative")
        if self.teacher_prefill_max_active_trajectories < 0:
            raise ValueError("streamopd_kv.teacher_prefill_max_active_trajectories must be non-negative")
        if self.teacher_prefill_max_active_kv_tokens < 0:
            raise ValueError("streamopd_kv.teacher_prefill_max_active_kv_tokens must be non-negative")
        if self.scheduler_timeout_seconds <= 0:
            raise ValueError("streamopd_kv.scheduler_timeout_seconds must be positive")
        if self.trainer_placement not in {"teacher", "rollout", "union", "dedicated"}:
            raise NotImplementedError(
                "streamopd_kv.trainer_placement supports 'teacher', 'rollout', 'union', and 'dedicated'"
            )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def get_streamopd_teacher(distillation) -> tuple[str, DictConfig]:
    """Select the sole effective teacher using verl's default-entry convention."""
    teachers = dict(distillation.get("teacher_models", {}))
    if len(teachers) > 1:
        teachers.pop("teacher_model", None)
    if len(teachers) != 1:
        raise ValueError("StreamOPD requires exactly one teacher model")
    return next(iter(teachers.items()))


def prepare_streamopd_vllm_environment(transfer_config) -> None:
    """Select the vLLM runner that implements cross-layer Host KV export."""
    strategy = transfer_config.get("kv_connector_extra_config", {}).get("streamopd_kv_export_strategy", "eos_host")
    if strategy != "eos_host":
        return
    # vLLM 0.24 enables model runner V2 by default, but that runner only calls
    # register_kv_caches, bypassing the uniform cross-layer allocation contract.
    # Scope the selection to this Rollout server and its child processes.
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "0":
        raise ValueError("StreamOPD eos_host requires VLLM_USE_V2_MODEL_RUNNER=0; use eos_triton with runner V2")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"


_EXCLUSIVE_VLLM_MEMORY_KEY = "verl_exclusive_gpu_memory"
_STREAMING_TEACHER_MEMORY_KEY = "verl_streaming_teacher_logprobs"


def _hydra_task_override_paths() -> set[str]:
    """Return exact config paths supplied on the Hydra command line."""

    try:
        from hydra.core.hydra_config import HydraConfig

        if not HydraConfig.initialized():
            return set()
        overrides = HydraConfig.get().overrides.task
    except (AttributeError, ImportError, ValueError):
        return set()
    return {item.split("=", 1)[0].lstrip("+~") for item in overrides if "=" in item}


def _is_explicit(path: str, explicit_paths: set[str]) -> bool:
    return any(path == candidate or path.startswith(f"{candidate}.") for candidate in explicit_paths)


def _set_derived(target, key: str, value, path: str, explicit_paths: set[str]) -> None:
    if not _is_explicit(path, explicit_paths):
        target[key] = value


def _enable_exclusive_vllm_memory(
    inference, *, max_live_trajectories: int, reservation_tokens: int, streaming_teacher: bool = False
) -> None:
    """Tell the vLLM worker to size itself from its post-NCCL free memory."""

    engine_kwargs = dict(inference.get("engine_kwargs", {}) or {})
    vllm_kwargs = dict(engine_kwargs.get("vllm", {}) or {})
    additional_config = dict(vllm_kwargs.get("additional_config", {}) or {})
    additional_config[_EXCLUSIVE_VLLM_MEMORY_KEY] = True
    additional_config["verl_streamopd_max_live_trajectories"] = max_live_trajectories
    additional_config["verl_streamopd_reserved_trajectory_tokens"] = reservation_tokens
    if streaming_teacher:
        additional_config[_STREAMING_TEACHER_MEMORY_KEY] = True
    vllm_kwargs["additional_config"] = additional_config
    engine_kwargs["vllm"] = vllm_kwargs
    inference["engine_kwargs"] = engine_kwargs


def _auto_streamopd_runtime_profile(
    config: DictConfig,
    *,
    explicit_paths: set[str] | None = None,
) -> dict[str, float | int | str]:
    """Resolve execution-only knobs from the user-visible OPD configuration."""

    explicit_paths = explicit_paths or set()
    stream_config = config.distillation.streamopd_kv
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    teacher_name, teacher_model = get_streamopd_teacher(config.distillation)
    teacher = teacher_model.inference
    teacher_path = f"distillation.teacher_models.{teacher_name}.inference"
    placement = str(stream_config.get("trainer_placement", "union"))
    if placement not in {"teacher", "rollout", "union", "dedicated"}:
        raise ValueError(f"unsupported StreamOPD trainer placement: {placement}")

    # A shared Trainer is resident only during its training phase.  Let the
    # standard engine offload path clear model, gradient, and optimizer state
    # before either standalone vLLM process owns the same GPU pool.
    if placement != "dedicated":
        _set_derived(
            actor.fsdp_config,
            "param_offload",
            True,
            "actor_rollout_ref.actor.fsdp_config.param_offload",
            explicit_paths,
        )

        _set_derived(
            actor.fsdp_config,
            "optimizer_offload",
            True,
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
            explicit_paths,
        )

    global_batch = int(config.data.train_batch_size)
    prompt_tokens = int(config.data.max_prompt_length)
    response_tokens = int(config.data.max_response_length)
    trajectory_tokens = prompt_tokens + response_tokens
    if min(global_batch, prompt_tokens, response_tokens) < 1:
        raise ValueError("automatic StreamOPD runtime planning requires positive batch and sequence lengths")

    page_size = 64
    reservation_page = max(page_size, int(stream_config.get("teacher_prefill_kv_page_size", page_size)))
    reservation_tokens = _ceil_div(trajectory_tokens, reservation_page) * reservation_page
    # Keep at least four opportunities for Teacher overlap while bounding each
    # prefill fragment to 1024 tokens. Longer responses naturally create more
    # fragments without token-length-specific tiers.
    response_chunk = min(
        1024,
        max(256, _ceil_div(_ceil_div(response_tokens, 4), page_size) * page_size),
    )
    teacher_max_batched_tokens = min(8192, max(2048, _ceil_div(trajectory_tokens, 256) * 256))

    _set_derived(
        rollout,
        "max_model_len",
        trajectory_tokens + 1,
        "actor_rollout_ref.rollout.max_model_len",
        explicit_paths,
    )
    rollout_world_size = int(rollout.n_gpus_per_node) * int(rollout.nnodes)
    rollout_replica_size = (
        int(rollout.tensor_model_parallel_size)
        * int(rollout.data_parallel_size)
        * int(rollout.pipeline_model_parallel_size)
    )
    rollout_replicas = max(1, rollout_world_size // rollout_replica_size)
    _set_derived(
        rollout,
        "max_num_seqs",
        _ceil_div(global_batch, rollout_replicas),
        "actor_rollout_ref.rollout.max_num_seqs",
        explicit_paths,
    )
    _enable_exclusive_vllm_memory(rollout, max_live_trajectories=global_batch, reservation_tokens=reservation_tokens)
    _set_derived(
        rollout.checkpoint_engine,
        "backend",
        "host",
        "actor_rollout_ref.rollout.checkpoint_engine.backend",
        explicit_paths,
    )
    _set_derived(
        rollout.checkpoint_engine,
        "update_weights_bucket_megabytes",
        128 if placement in {"rollout", "union"} else 512,
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
        explicit_paths,
    )

    _set_derived(
        teacher,
        "max_model_len",
        trajectory_tokens + 1,
        f"{teacher_path}.max_model_len",
        explicit_paths,
    )
    _set_derived(
        teacher,
        "max_num_batched_tokens",
        teacher_max_batched_tokens,
        f"{teacher_path}.max_num_batched_tokens",
        explicit_paths,
    )
    teacher_world_size = int(config.distillation.n_gpus_per_node) * int(config.distillation.nnodes)
    teacher_replica_size = (
        int(teacher.get("tensor_model_parallel_size", 1))
        * int(teacher.get("data_parallel_size", 1))
        * int(teacher.get("pipeline_model_parallel_size", 1))
    )
    teacher_replicas = max(1, teacher_world_size // teacher_replica_size)
    teacher_prefill_wave = max(1, int(teacher.max_num_batched_tokens) // min(response_chunk, trajectory_tokens))
    # Keep enough resumable sessions resident to avoid head-of-line blocking
    # behind a cohort of long trajectories. Admission is tightened after vLLM
    # reports the KV blocks it actually allocated.
    teacher_target_concurrency = max(32, 1 << (2 * teacher_prefill_wave - 1).bit_length())
    _set_derived(
        teacher,
        "max_num_seqs",
        min(_ceil_div(global_batch, teacher_replicas), teacher_target_concurrency),
        f"{teacher_path}.max_num_seqs",
        explicit_paths,
    )
    _enable_exclusive_vllm_memory(
        teacher, max_live_trajectories=global_batch, reservation_tokens=reservation_tokens, streaming_teacher=True
    )

    derived_stream_values = {
        "token_chunk_size": response_chunk,
        "rollout_kv_export_strategy": "eos_host",
        "rollout_kv_export_chunk_size": min(trajectory_tokens, 2048),
        "rollout_kv_writer_threads": 4,
        "reverse_chunk_size": 0,
        "reverse_chunk_min_size": 0,
        "reverse_page_size": page_size,
        "reverse_batch_size": 0,
        "reverse_batch_max_tokens": 0,
        "reverse_slot_max_tokens": 0,
        "reverse_slot_reserve_gib": 4.0,
        "teacher_prefill_max_active_trajectories": 0,
        "teacher_prefill_max_active_kv_tokens": 0,
        "teacher_prefill_kv_page_size": page_size,
        "kv_prefetch_depth": 1,
        "kv_prefetch_workers": 4,
        "kv_handoff_dir": f"/dev/shm/verl-streamopd-kv-{os.getpid()}",
    }
    for key, value in derived_stream_values.items():
        _set_derived(
            stream_config,
            key,
            value,
            f"distillation.streamopd_kv.{key}",
            explicit_paths,
        )

    return {
        "profile": "auto",
        "trajectory_tokens": trajectory_tokens,
        "token_chunk_size": int(stream_config.token_chunk_size),
        "teacher_max_batched_tokens": int(teacher.max_num_batched_tokens),
        "vllm_memory_policy": "exclusive_free",
        "teacher_max_num_seqs": int(teacher.max_num_seqs),
        "rollout_max_num_seqs": int(rollout.max_num_seqs),
    }


def prepare_streamopd_kv_config(config: DictConfig) -> None:
    """Validate the supported envelope before installing StreamOPD runtime settings."""

    stream_config = config.distillation.get("streamopd_kv", {})
    trainer_mode = config.trainer.v1.trainer_mode if config.trainer.use_v1 else None
    if not stream_config.get("enabled", False):
        if trainer_mode == "streamopd":
            raise ValueError("trainer_mode=streamopd requires distillation.streamopd_kv.enabled=true")
        return
    if not config.trainer.use_v1 or trainer_mode != "streamopd":
        raise ValueError("strict StreamOPD requires trainer.v1.trainer_mode=streamopd")
    if not config.distillation.get("enabled", False):
        raise ValueError("streamopd_kv.enabled requires distillation.enabled=true")
    # Check raw settings before auto planning can mask invalid values or divide
    # by a zero page/replica size. The dataclass also rejects stale option names.
    StreamOPDKVConfig(**{key: value for key, value in stream_config.items() if key != "_target_"})
    _, teacher_model = get_streamopd_teacher(config.distillation)
    teacher_inference = teacher_model.inference
    rollout = config.actor_rollout_ref.rollout
    if teacher_inference.get("name", "vllm") != "vllm":
        raise NotImplementedError("StreamOPD requires a vLLM teacher for resumable StreamingInput")
    for role, inference, resources in (
        ("Rollout", rollout, rollout),
        ("Teacher", teacher_inference, config.distillation),
    ):
        pool_size = int(resources.n_gpus_per_node) * int(resources.nnodes)
        dimensions = [
            int(inference.get(key, 1))
            for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "data_parallel_size")
        ]
        if pool_size < 1 or min(dimensions) < 1:
            raise ValueError(f"StreamOPD {role} GPU count and parallel dimensions must be positive")
        replica_size = dimensions[0] * dimensions[1] * dimensions[2]
        if pool_size % replica_size:
            raise ValueError(f"StreamOPD {role} GPU allocation must contain whole inference replicas")
    if int(config.trainer.n_gpus_per_node) < 1 or int(config.data.train_batch_size) < 1:
        raise ValueError("StreamOPD Trainer GPU count and train_batch_size must be positive")
    if rollout.get("agent", {}).get("default_agent_loop", "single_turn_agent") != "single_turn_agent":
        raise NotImplementedError("StreamOPD supports the single_turn_agent loop only")
    try:
        installed_vllm = version("vllm")
    except PackageNotFoundError as error:
        raise RuntimeError("StreamOPD requires vLLM 0.15.1 or >= 0.18.0") from error
    if Version(installed_vllm) != Version("0.15.1") and Version(installed_vllm) < Version("0.18.0"):
        raise RuntimeError(f"StreamOPD requires vLLM 0.15.1 or >= 0.18.0, found {installed_vllm}")
    runtime_profile = str(stream_config.get("runtime_profile", "auto"))
    if runtime_profile not in {"auto", "manual"}:
        raise ValueError("streamopd_kv.runtime_profile must be 'auto' or 'manual'")
    if runtime_profile == "auto":
        explicit_paths = _hydra_task_override_paths()
        with open_dict(config):
            stream_config.planner_explicit_options = sorted(explicit_paths)
            _auto_streamopd_runtime_profile(config, explicit_paths=explicit_paths)
    rollout = config.actor_rollout_ref.rollout
    if rollout.name != "vllm":
        raise NotImplementedError("StreamOPD-KV MVP supports the vLLM rollout backend only")
    if rollout.tensor_model_parallel_size != 1 or rollout.pipeline_model_parallel_size != 1:
        raise NotImplementedError("StreamOPD-KV MVP requires rollout TP=1 and PP=1")
    if int(rollout.get("n", 1)) != 1:
        raise NotImplementedError("StreamOPD currently requires actor_rollout_ref.rollout.n=1")
    trainer_placement = str(stream_config.get("trainer_placement", "union"))
    actor = config.actor_rollout_ref.actor
    if actor.strategy not in ("fsdp", "fsdp2"):
        raise NotImplementedError("StreamOPD-KV MVP requires an FSDP actor")
    if trainer_placement != "dedicated" and (
        not bool(actor.fsdp_config.param_offload) or not bool(actor.fsdp_config.optimizer_offload)
    ):
        raise ValueError("a Trainer-shared StreamOPD pool requires actor FSDP parameter and optimizer offload")
    if actor.ppo_epochs != 1:
        raise ValueError("strict StreamOPD-KV requires actor.ppo_epochs=1")
    if actor.use_torch_compile:
        raise NotImplementedError("StreamOPD-KV MVP requires actor.use_torch_compile=false")
    if actor.loss_agg_mode != "token-mean":
        raise NotImplementedError("StreamOPD-KV MVP requires actor.loss_agg_mode=token-mean")
    loss = config.distillation.distillation_loss
    if loss.loss_mode != "forward_kl_topk" or loss.use_policy_gradient or loss.use_task_rewards:
        raise NotImplementedError(
            "StreamOPD-KV MVP requires direct distillation-only forward_kl_topk: "
            "loss_mode=forward_kl_topk, use_policy_gradient=false, use_task_rewards=false"
        )
    if config.trainer.nnodes != 1 or rollout.nnodes != 1 or config.distillation.nnodes != 1:
        raise NotImplementedError("StreamOPD currently supports one node only")
    if bool(config.distillation.get("colocate_teacher_with_student", False)):
        raise ValueError(
            "StreamOPD placement is controlled by streamopd_kv.trainer_placement; "
            "colocate_teacher_with_student is a sync-baseline option"
        )
    if trainer_placement == "teacher" and config.distillation.n_gpus_per_node > config.trainer.n_gpus_per_node:
        raise ValueError("a Teacher-shared StreamOPD trainer pool must cover every Teacher GPU")
    if trainer_placement == "rollout" and rollout.n_gpus_per_node > config.trainer.n_gpus_per_node:
        raise ValueError("a Rollout-shared StreamOPD trainer pool must cover every Rollout GPU")
    if trainer_placement == "union" and (
        config.distillation.n_gpus_per_node + rollout.n_gpus_per_node > config.trainer.n_gpus_per_node
    ):
        raise ValueError("a union StreamOPD trainer pool must cover disjoint Teacher and Rollout GPU subsets")
    if rollout.n_gpus_per_node < 1:
        raise ValueError("StreamOPD requires a positive Rollout GPU count")
    if trainer_placement in {"teacher", "union"} and not bool(teacher_inference.get("free_cache_engine", True)):
        raise ValueError("a Trainer-shared Teacher requires inference.free_cache_engine=true for phase switching")
    if trainer_placement in {"rollout", "union"} and not bool(rollout.get("free_cache_engine", True)):
        raise ValueError("a Trainer-shared Rollout requires rollout.free_cache_engine=true for phase switching")
    checkpoint_backend = str(rollout.checkpoint_engine.backend)
    if checkpoint_backend == "naive":
        raise ValueError("StreamOPD requires a cross-process checkpoint engine such as host or nccl")
    if trainer_placement in {"rollout", "union"} and checkpoint_backend != "host":
        raise ValueError("a Trainer-shared Rollout requires checkpoint_engine.backend=host for phase-exclusive sync")

    train_batch_size = int(config.data.train_batch_size)
    if int(stream_config.get("reverse_slot_max_tokens", 0)) == 0:
        prompt_limit = int(config.data.get("max_prompt_length", 0))
        response_limit = int(config.data.get("max_response_length", 0))
        configured_limit = prompt_limit + response_limit
        if configured_limit == 0:
            configured_limit = max(1, int(rollout.get("max_model_len", 4097)) - 1)
        with open_dict(stream_config):
            stream_config.reverse_slot_max_tokens = configured_limit
    page_size = int(stream_config.get("reverse_page_size", 64))
    slot_tokens = int(stream_config.reverse_slot_max_tokens)
    required_tokens = int(config.data.get("max_prompt_length", 0)) + int(config.data.get("max_response_length", 0))
    if slot_tokens < required_tokens:
        raise ValueError("reverse_slot_max_tokens must cover the configured prompt and response lengths")
    aligned_slot_tokens = ((slot_tokens + page_size - 1) // page_size) * page_size
    with open_dict(stream_config):
        if int(stream_config.get("rollout_kv_export_chunk_size", 0)) == 0:
            stream_config.rollout_kv_export_chunk_size = min(aligned_slot_tokens, 2048)
        # Zero-valued reverse knobs mean "derive the largest stable preflight
        # plan". Non-zero values remain optional user caps for experiments.
        if int(stream_config.get("reverse_batch_size", 0)) == 0:
            stream_config.reverse_batch_size = train_batch_size
        if int(stream_config.get("reverse_batch_max_tokens", 0)) == 0:
            stream_config.reverse_batch_max_tokens = train_batch_size * aligned_slot_tokens
        if int(stream_config.get("reverse_chunk_size", 0)) == 0:
            # The joint GPU preflight selects the largest useful token tile
            # that fits the actual Trainer pool.
            stream_config.reverse_chunk_size = aligned_slot_tokens
        if int(stream_config.get("reverse_chunk_min_size", 0)) == 0:
            stream_config.reverse_chunk_min_size = page_size
    with open_dict(config.trainer.v1.sampler):
        # ReplayBuffer measures the number of policy versions spanned by a
        # trajectory. A strict single-version trajectory therefore has span 1.
        config.trainer.v1.sampler.max_off_policy_threshold = 1
        # Dedicated Trainer pools may consume ready units before the complete
        # fixed policy batch finishes; the policy barrier rejects stale work.
        config.trainer.v1.sampler.max_off_policy_strategy = "drop"
    filter_groups = config.algorithm.get("filter_groups")
    if filter_groups is not None:
        with open_dict(filter_groups):
            # Direct distillation neither consumes rewards nor filters prompt
            # groups. More importantly, Sync DAPO filtering waits for every
            # in-flight prompt before sampling and would erase StreamOPD's
            # trajectory-level ready queue.
            filter_groups.enable = False

    engine_kwargs = rollout.get("engine_kwargs")
    if engine_kwargs is None:
        with open_dict(rollout):
            rollout.engine_kwargs = {}
        engine_kwargs = rollout.engine_kwargs
    configured_kv_dtype = str(engine_kwargs.get("vllm", {}).get("kv_cache_dtype", "auto")).lower()
    if configured_kv_dtype.startswith("fp8"):
        raise ValueError("StreamOPD-KV requires a non-quantized rollout KV cache")
    # Mark manual profiles too: ordinary OPD teachers must keep the native
    # vLLM prompt-logprob path, including when they run vLLM 0.15.1.
    with open_dict(teacher_inference):
        teacher_kwargs = dict(teacher_inference.get("engine_kwargs", {}) or {})
        vllm_kwargs = dict(teacher_kwargs.get("vllm", {}) or {})
        additional_config = dict(vllm_kwargs.get("additional_config", {}) or {})
        additional_config[_STREAMING_TEACHER_MEMORY_KEY] = True
        vllm_kwargs["additional_config"] = additional_config
        teacher_kwargs["vllm"] = vllm_kwargs
        teacher_inference.engine_kwargs = teacher_kwargs
    existing = engine_kwargs.get("vllm", {}).get("kv_transfer_config")
    if existing is not None:
        existing_container = (
            OmegaConf.to_container(existing, resolve=True) if isinstance(existing, DictConfig) else existing
        )
        if existing_container.get("kv_connector") != "StreamOPDKVConnector":
            raise ValueError("StreamOPD-KV cannot replace an existing vLLM kv_transfer_config")
    connector_config = {
        "kv_connector": "StreamOPDKVConnector",
        "kv_role": "kv_producer",
        "kv_connector_module_path": "verl.experimental.streamopd_kv.vllm_connector",
        "kv_connector_extra_config": {
            "streamopd_kv_handoff_dir": stream_config.kv_handoff_dir,
            "streamopd_kv_chunk_size": int(stream_config.rollout_kv_export_chunk_size),
            "streamopd_kv_export_strategy": str(stream_config.get("rollout_kv_export_strategy", "eos_host")),
            "streamopd_host_slot_count": int(config.data.train_batch_size),
            "streamopd_host_slot_tokens": int(stream_config.reverse_slot_max_tokens),
            "streamopd_writer_threads": int(stream_config.get("rollout_kv_writer_threads", 4)),
        },
    }
    with open_dict(engine_kwargs):
        if "vllm" not in engine_kwargs:
            engine_kwargs.vllm = {}
        engine_kwargs.vllm.kv_transfer_config = connector_config
