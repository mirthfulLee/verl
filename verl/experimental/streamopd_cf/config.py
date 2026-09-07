# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from omegaconf import OmegaConf, open_dict

from verl.base_config import BaseConfig
from verl.experimental.streamopd_kv.config import get_streamopd_teacher


@dataclass
class OPDBatchingConfig(BaseConfig):
    """Memory budget shared by dedicated CF and full-trajectory OPD training."""

    memory_fraction: float = 0.8
    reserve_gib: float = 4.0

    def __post_init__(self):
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("distillation.batching.memory_fraction must be in (0, 1]")
        if self.reserve_gib < 0:
            raise ValueError("distillation.batching.reserve_gib must be non-negative")


@dataclass
class StreamOPDCFConfig(BaseConfig):
    """Independent inference/training pools with one backward per microbatch."""

    enabled: bool = False
    forward_chunk_size: int = 1024
    loss_chunk_size: int = 2048
    attention_backend: str = "flash_attention_2"
    token_chunk_size: int = 1024
    teacher_prefill_max_active_trajectories: int = 0
    teacher_prefill_max_active_kv_tokens: int = 0
    teacher_prefill_kv_page_size: int = 64
    max_pending_teacher_chunks: int = 128
    scheduler_actor_name: str = ""
    training_stream_actor_name: str = ""
    timeline_dir: str = ""
    scheduler_timeout_seconds: float = 600.0
    require_same_tokenizer: bool = True
    validate_teacher_artifacts: bool = False
    validation_atol: float = 1e-4

    def __post_init__(self):
        for key in (
            "forward_chunk_size",
            "loss_chunk_size",
            "token_chunk_size",
            "teacher_prefill_kv_page_size",
            "max_pending_teacher_chunks",
            "scheduler_timeout_seconds",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"streamopd_cf.{key} must be positive")
        for key in (
            "teacher_prefill_max_active_trajectories",
            "teacher_prefill_max_active_kv_tokens",
            "validation_atol",
        ):
            if getattr(self, key) < 0:
                raise ValueError(f"streamopd_cf.{key} must be non-negative")
        if self.attention_backend not in ("flash_attention_2", "sdpa"):
            raise ValueError("streamopd_cf.attention_backend must be flash_attention_2 or sdpa")


def prepare_streamopd_cf_config(config):
    """Validate only the opt-in strategy, before Ray or model initialization."""
    settings = config.distillation.get("streamopd_cf", {})
    trainer_mode = config.trainer.get("v1", {}).get("trainer_mode")
    selected = trainer_mode == "streamopd_cf"
    if trainer_mode in ("separate_sync", "union_sync"):
        if not config.trainer.use_v1 or not config.distillation.enabled:
            raise ValueError("separate_sync requires V1 and distillation.enabled=true")
        if settings.get("enabled", False) or config.distillation.get("streamopd_kv", {}).get("enabled", False):
            raise ValueError("separate_sync requires both streaming strategies disabled")
        validate_dedicated_opd_config(config)
        with open_dict(config):
            config.trainer.v1[trainer_mode] = {"parameter_sync_step": 1}
            config.trainer.v1.sampler.max_off_policy_threshold = 1
            if config.algorithm.get("filter_groups"):
                config.algorithm.filter_groups.enable = False
        return config
    if not settings.get("enabled", False) and not selected:
        return config
    if not selected or not config.trainer.use_v1 or not config.distillation.enabled:
        raise ValueError("StreamOPD-CF requires V1 trainer_mode=streamopd_cf and distillation.enabled=true")
    if config.distillation.get("streamopd_kv", {}).get("enabled", False):
        raise ValueError("streamopd-cf and StreamOPD reverse training are mutually exclusive")
    validate_dedicated_opd_config(config)
    with open_dict(config):
        config.distillation.streamopd_cf = OmegaConf.merge(OmegaConf.structured(StreamOPDCFConfig), settings)
        config.distillation.streamopd_cf.enabled = True
        config.trainer.v1[trainer_mode] = {"parameter_sync_step": 1}
        config.trainer.v1.sampler.max_off_policy_threshold = 1
        if config.algorithm.get("filter_groups"):
            config.algorithm.filter_groups.enable = False
        teacher = get_streamopd_teacher(config.distillation)[1].inference
        kwargs = teacher.engine_kwargs.get("vllm", {})
        additional = dict(kwargs.get("additional_config", {}) or {})
        additional["verl_streaming_teacher_logprobs"] = True
        teacher.engine_kwargs.vllm = OmegaConf.merge(kwargs, {"additional_config": additional})
    StreamOPDCFConfig(**OmegaConf.to_container(config.distillation.streamopd_cf, resolve=True))
    return config


def validate_dedicated_opd_config(config):
    """Shared objective and topology requirements for the matched strategies."""
    if config.distillation.get("colocate_teacher_with_student", False):
        raise ValueError("streamopd-cf requires independent Teacher and Trainer GPUs")
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    loss = config.distillation.distillation_loss
    if actor.strategy not in ("fsdp", "fsdp2") or actor.fsdp_config.ulysses_sequence_parallel_size != 1:
        raise NotImplementedError("streamopd-cf currently requires FSDP/FSDP2 without sequence parallelism")
    if actor.ppo_epochs != 1 or actor.loss_agg_mode != "token-mean":
        raise ValueError("streamopd-cf requires ppo_epochs=1 and token-mean loss normalization")
    if actor.ppo_mini_batch_size != config.data.train_batch_size:
        raise ValueError("streamopd-cf requires ppo_mini_batch_size=train_batch_size for one policy update")
    if loss.loss_mode != "forward_kl_topk" or loss.use_policy_gradient or loss.use_task_rewards:
        raise ValueError("streamopd-cf currently supports direct forward_kl_topk distillation only")
    if rollout.agent.default_agent_loop != "single_turn_agent" or rollout.n != 1:
        raise ValueError("streamopd-cf requires single_turn_agent and rollout.n=1")
    if rollout.name != "vllm" or get_streamopd_teacher(config.distillation)[1].inference.name != "vllm":
        raise ValueError("streamopd-cf requires vLLM Rollout and Teacher")
    if rollout.checkpoint_engine.backend != "host":
        raise ValueError("streamopd-cf currently requires the Host checkpoint backend")
    for role, nnodes, gpus in (
        ("Trainer", config.trainer.nnodes, config.trainer.n_gpus_per_node),
        ("Rollout", rollout.nnodes, rollout.n_gpus_per_node),
        ("Teacher", config.distillation.nnodes, config.distillation.n_gpus_per_node),
    ):
        if nnodes < 1 or gpus < 1:
            raise ValueError(f"streamopd-cf {role} requires a positive dedicated GPU allocation")
        if nnodes != 1:
            raise NotImplementedError("streamopd-cf Host weight transfer currently supports one node")
    if config.actor_rollout_ref.model.lora.get("rank", 0) > 0:
        raise NotImplementedError("streamopd-cf currently supports full-parameter training")
    dp_size = int(config.trainer.nnodes * config.trainer.n_gpus_per_node)
    if config.data.train_batch_size % dp_size:
        raise ValueError("dedicated OPD policy batch must be divisible by the Trainer DP size")
    if actor.use_dynamic_bsz:
        if actor.ppo_max_token_len_per_gpu is None or actor.ppo_max_token_len_per_gpu < 0:
            raise ValueError("ppo_max_token_len_per_gpu must be positive, or 0 for automatic planning")
        if config.trainer.v1.trainer_mode in ("separate_sync", "union_sync") and actor.ppo_max_token_len_per_gpu == 0:
            if not config.actor_rollout_ref.model.use_remove_padding or not loss.use_chunked_topk:
                raise ValueError("baseline auto planning requires use_remove_padding=True and use_chunked_topk=True")
    else:
        micro = actor.ppo_micro_batch_size_per_gpu
        if not micro or micro < 1 or (config.data.train_batch_size // dp_size) % micro:
            raise ValueError("fixed training microbatch must be positive and divide the per-rank policy batch")
    prepare_inference_batching(config)


def prepare_inference_batching(config):
    """Resolve each inference pool independently; vLLM profiles its own KV memory."""
    rollout = config.actor_rollout_ref.rollout
    teacher = get_streamopd_teacher(config.distillation)[1].inference
    trajectory_tokens = int(rollout.prompt_length + rollout.response_length)
    for inference, resources, is_teacher in (
        (rollout, rollout, False),
        (teacher, config.distillation, True),
    ):
        replicas = int(resources.nnodes * resources.n_gpus_per_node) // (
            int(inference.get("tensor_model_parallel_size", 1))
            * int(inference.get("pipeline_model_parallel_size", 1))
            * int(inference.get("data_parallel_size", 1))
        )
        if replicas < 1:
            raise ValueError("inference GPU allocation must fit at least one replica")
        per_replica = (int(config.data.train_batch_size) + replicas - 1) // replicas
        with open_dict(inference):
            if inference.max_num_batched_tokens == 0:
                inference.max_num_batched_tokens = min(8192, max(2048, ((trajectory_tokens + 255) // 256) * 256))
            if inference.max_num_seqs == 0:
                inference.max_num_seqs = min(per_replica, 32) if is_teacher else per_replica
        if inference.max_num_seqs < 1 or inference.max_num_batched_tokens < 1:
            raise ValueError("inference sequence/token budgets must be positive, or 0 for automatic selection")
