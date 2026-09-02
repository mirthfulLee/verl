# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import os

from omegaconf import DictConfig, OmegaConf, open_dict


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


_SHARED_TEACHER_MEMORY_FRACTION = 0.25
_MAX_DEDICATED_VLLM_MEMORY_FRACTION = 0.90


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


def _auto_streamopd_runtime_profile(
    config: DictConfig,
    *,
    explicit_paths: set[str] | None = None,
) -> dict[str, float | int | str]:
    """Resolve execution-only knobs from the user-visible OPD configuration."""

    explicit_paths = explicit_paths or set()
    stream_config = config.distillation.streamopd_kv
    rollout = config.actor_rollout_ref.rollout
    teacher = next(iter(config.distillation.teacher_models.values())).inference
    placement = str(stream_config.get("trainer_placement", "teacher"))
    if placement not in {"teacher", "rollout", "union", "dedicated"}:
        raise ValueError(f"unsupported StreamOPD trainer placement: {placement}")

    global_batch = int(config.data.train_batch_size)
    prompt_tokens = int(config.data.max_prompt_length)
    response_tokens = int(config.data.max_response_length)
    trajectory_tokens = prompt_tokens + response_tokens
    if min(global_batch, prompt_tokens, response_tokens) < 1:
        raise ValueError("automatic StreamOPD runtime planning requires positive batch and sequence lengths")

    page_size = 64
    response_chunk = min(1024, max(256, _ceil_div(_ceil_div(response_tokens, 8), page_size) * page_size))
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
    # This is only a ceiling. The controller derives the minimum sufficient
    # fraction and, if necessary, a smaller stable concurrency from physical
    # memory, model weights, and the non-preemptible KV footprint.
    _set_derived(
        rollout,
        "gpu_memory_utilization",
        _MAX_DEDICATED_VLLM_MEMORY_FRACTION,
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        explicit_paths,
    )
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
        "distillation.teacher_models.teacher_model.inference.max_model_len",
        explicit_paths,
    )
    _set_derived(
        teacher,
        "max_num_batched_tokens",
        teacher_max_batched_tokens,
        "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens",
        explicit_paths,
    )
    teacher_world_size = int(config.distillation.n_gpus_per_node) * int(config.distillation.nnodes)
    teacher_replica_size = (
        int(teacher.get("tensor_model_parallel_size", 1))
        * int(teacher.get("data_parallel_size", 1))
        * int(teacher.get("pipeline_model_parallel_size", 1))
    )
    teacher_replicas = max(1, teacher_world_size // teacher_replica_size)
    _set_derived(
        teacher,
        "max_num_seqs",
        _ceil_div(global_batch, teacher_replicas),
        "distillation.teacher_models.teacher_model.inference.max_num_seqs",
        explicit_paths,
    )
    # Shared Teacher gets one conventional quarter of each GPU. Reverse
    # preflight consumes the measured remainder; dedicated Teacher uses verl's
    # standard half-GPU vLLM envelope.
    _set_derived(
        teacher,
        "gpu_memory_utilization",
        (_SHARED_TEACHER_MEMORY_FRACTION if placement in {"teacher", "union"} else _MAX_DEDICATED_VLLM_MEMORY_FRACTION),
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization",
        explicit_paths,
    )

    derived_stream_values = {
        "token_chunk_size": response_chunk,
        "teacher_initial_chunk_size": response_chunk,
        "reverse_chunk_size": 0,
        "reverse_chunk_min_size": 0,
        "reverse_page_size": page_size,
        "reverse_batch_size": 0,
        "reverse_batch_max_tokens": 0,
        "reverse_fixed_slots": True,
        "reverse_slot_max_tokens": 0,
        "reverse_slot_reserve_gib": 4.0,
        "micro_batch_size": min(32, global_batch),
        "teacher_priority_threshold": 0,
        "teacher_prefill_max_active_trajectories": 0,
        "teacher_prefill_max_active_kv_tokens": 0,
        "teacher_prefill_kv_page_size": page_size,
        "kv_prefetch_depth": 1,
        "kv_prefetch_workers": 4,
        "kv_prefetch_pin_memory": True,
        "release_stage1_kv_after_copy": True,
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
        "token_chunk_size": response_chunk,
        "teacher_max_batched_tokens": teacher_max_batched_tokens,
        "teacher_gpu_memory_utilization": float(teacher.gpu_memory_utilization),
        "teacher_max_num_seqs": int(teacher.max_num_seqs),
        "rollout_gpu_memory_utilization_ceiling": float(rollout.gpu_memory_utilization),
        "rollout_max_num_seqs": int(rollout.max_num_seqs),
    }


def prepare_streamopd_kv_config(config: DictConfig) -> None:
    """Validate the MVP envelope and install the out-of-tree vLLM connector."""

    stream_config = config.distillation.get("streamopd_kv", {})
    trainer_mode = config.trainer.v1.trainer_mode if config.trainer.use_v1 else None
    if not stream_config.get("enabled", False):
        if trainer_mode == "streamopd_colocate":
            raise ValueError("trainer_mode=streamopd_colocate requires distillation.streamopd_kv.enabled=true")
        return
    if not config.trainer.use_v1 or trainer_mode != "streamopd_colocate":
        raise ValueError("strict StreamOPD requires trainer.v1.trainer_mode=streamopd_colocate")
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
    actor = config.actor_rollout_ref.actor
    if actor.strategy not in ("fsdp", "fsdp2"):
        raise NotImplementedError("StreamOPD-KV MVP requires an FSDP actor")
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
    trainer_placement = str(stream_config.get("trainer_placement", "teacher"))
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
    checkpoint_backend = str(rollout.checkpoint_engine.backend)
    if checkpoint_backend == "naive":
        raise ValueError("StreamOPD requires a cross-process checkpoint engine such as host or nccl")

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
    aligned_slot_tokens = ((slot_tokens + page_size - 1) // page_size) * page_size
    with open_dict(stream_config):
        # Zero-valued reverse knobs mean "derive the largest stable preflight
        # plan". Non-zero values remain optional user caps for experiments.
        if int(stream_config.get("reverse_batch_size", 0)) == 0:
            stream_config.reverse_batch_size = train_batch_size
        if int(stream_config.get("reverse_batch_max_tokens", 0)) == 0:
            stream_config.reverse_batch_max_tokens = train_batch_size * aligned_slot_tokens
        if int(stream_config.get("reverse_chunk_size", 0)) == 0:
            # OOMB kernels benefit from reasonably large token tiles, while
            # multi-thousand-token activation tiles reduce wavefront width and
            # have not improved throughput in profiling. The joint GPU
            # preflight still selects the largest divisor up to this cap.
            stream_config.reverse_chunk_size = min(aligned_slot_tokens, 1024)
        if int(stream_config.get("reverse_chunk_min_size", 0)) == 0:
            stream_config.reverse_chunk_min_size = page_size
    with open_dict(config.trainer.v1.streamopd_colocate):
        # Runtime preflight determines the actual training-unit width and
        # accumulation count. These values are placeholders for base-trainer
        # initialization only.
        config.trainer.v1.streamopd_colocate.micro_batch_size = train_batch_size
        config.trainer.v1.streamopd_colocate.parameter_sync_step = 1
    with open_dict(config.trainer.v1.sampler):
        # ReplayBuffer measures the number of policy versions spanned by a
        # trajectory. A strict single-version trajectory therefore has span 1.
        config.trainer.v1.sampler.max_off_policy_threshold = 1
        # Finished microbatches can train while the rest of the fixed policy
        # batch is still rolling out; the policy barrier rejects stale work.
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
    existing = engine_kwargs.get("vllm", {}).get("kv_transfer_config")
    if existing is not None:
        existing_container = (
            OmegaConf.to_container(existing, resolve=True) if isinstance(existing, DictConfig) else existing
        )
        if existing_container.get("kv_connector") != "StreamOPDKVConnector":
            raise ValueError("StreamOPD-KV cannot replace an existing vLLM kv_transfer_config")
        return
    connector_config = {
        "kv_connector": "StreamOPDKVConnector",
        "kv_role": "kv_producer",
        "kv_connector_module_path": "verl.experimental.streamopd_kv.vllm_connector",
        "kv_connector_extra_config": {
            "streamopd_kv_handoff_dir": stream_config.kv_handoff_dir,
            "streamopd_kv_chunk_size": int(stream_config.get("token_chunk_size", 256)),
            "streamopd_host_slot_count": int(config.data.train_batch_size),
            "streamopd_host_slot_tokens": int(stream_config.reverse_slot_max_tokens),
            "streamopd_writer_threads": 4,
        },
    }
    with open_dict(engine_kwargs):
        if "vllm" not in engine_kwargs:
            engine_kwargs.vllm = {}
        engine_kwargs.vllm.kv_transfer_config = connector_config
