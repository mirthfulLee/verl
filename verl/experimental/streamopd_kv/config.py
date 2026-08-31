# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf, open_dict


def prepare_streamopd_kv_config(config: DictConfig) -> None:
    """Validate the MVP envelope and install the out-of-tree vLLM connector."""

    stream_config = config.distillation.get("streamopd_kv", {})
    trainer_mode = config.trainer.v1.trainer_mode if config.trainer.use_v1 else None
    dedicated_colocate = trainer_mode == "streamopd_colocate"
    if stream_config.get("colocate_teacher_with_student", False):
        if not config.trainer.use_v1 or trainer_mode not in ("sync", "streamopd_colocate"):
            raise ValueError("teacher/student colocation requires verl V1 sync or streamopd_colocate trainer")
        if config.trainer.nnodes != 1 or config.distillation.nnodes != 1:
            raise NotImplementedError("StreamOPD teacher/student colocation currently supports one node only")
        if not dedicated_colocate and config.distillation.n_gpus_per_node >= config.trainer.n_gpus_per_node:
            raise ValueError("colocated teacher GPU count must be smaller than the student pool GPU count")
    if not stream_config.get("enabled", False):
        if dedicated_colocate:
            raise ValueError("trainer_mode=streamopd_colocate requires distillation.streamopd_kv.enabled=true")
        return
    if not config.trainer.use_v1 or trainer_mode not in ("sync", "streamopd_colocate"):
        raise ValueError("strict StreamOPD-KV requires verl V1 sync or streamopd_colocate trainer")
    rollout = config.actor_rollout_ref.rollout
    if rollout.name != "vllm":
        raise NotImplementedError("StreamOPD-KV MVP supports the vLLM rollout backend only")
    if rollout.tensor_model_parallel_size != 1 or rollout.pipeline_model_parallel_size != 1:
        raise NotImplementedError("StreamOPD-KV MVP requires rollout TP=1 and PP=1")
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

    if dedicated_colocate:
        if not stream_config.get("colocate_teacher_with_student", False):
            raise ValueError("streamopd_colocate requires colocate_teacher_with_student=true")
        if config.trainer.nnodes != 1 or rollout.nnodes != 1 or config.distillation.nnodes != 1:
            raise NotImplementedError("streamopd_colocate currently supports one node only")
        if config.trainer.n_gpus_per_node != config.distillation.n_gpus_per_node:
            raise ValueError("streamopd_colocate requires teacher and trainer pools to use the same GPU count")
        if rollout.n_gpus_per_node < 1:
            raise ValueError("streamopd_colocate requires a positive standalone rollout GPU count")
        checkpoint_backend = str(rollout.checkpoint_engine.backend)
        if checkpoint_backend == "naive":
            raise ValueError("streamopd_colocate requires a cross-pool checkpoint engine such as host or nccl")

        train_batch_size = int(config.data.train_batch_size)
        micro_batch_size = int(stream_config.get("micro_batch_size", 0))
        if micro_batch_size < 1 or train_batch_size % micro_batch_size:
            raise ValueError(
                "StreamOPD-colocate requires data.train_batch_size to be divisible by streamopd_kv.micro_batch_size"
            )
        accumulation_steps = train_batch_size // micro_batch_size
        with open_dict(config.trainer.v1.streamopd_colocate):
            config.trainer.v1.streamopd_colocate.micro_batch_size = micro_batch_size
            config.trainer.v1.streamopd_colocate.parameter_sync_step = accumulation_steps
        with open_dict(config.trainer.v1.sampler):
            # ReplayBuffer measures the number of policy versions spanned by a
            # trajectory. A strict single-version trajectory therefore has span 1.
            config.trainer.v1.sampler.max_off_policy_threshold = 1
            # Only one exact-size policy cohort is ever submitted. ``drop`` allows
            # finished microbatches to train while the rest of that same cohort is
            # still rolling out; the policy barrier prevents a stale cohort from
            # ever being submitted.
            config.trainer.v1.sampler.max_off_policy_strategy = "drop"

    elif stream_config.get("overlap_rollout_training", False):
        train_batch_size = int(config.data.train_batch_size)
        rollout_micro_batch_size = int(stream_config.get("rollout_micro_batch_size", 0))
        if rollout_micro_batch_size < 1 or train_batch_size % rollout_micro_batch_size:
            raise ValueError(
                "StreamOPD overlap requires data.train_batch_size to be divisible by "
                "streamopd_kv.rollout_micro_batch_size"
            )
        accumulation_steps = train_batch_size // rollout_micro_batch_size
        if accumulation_steps < 2:
            raise ValueError("StreamOPD overlap requires at least two rollout microbatches")
        with open_dict(config.trainer.v1.sync):
            config.trainer.v1.sync.parameter_sync_step = accumulation_steps

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
            "streamopd_writer_threads": 4,
        },
    }
    with open_dict(engine_kwargs):
        if "vllm" not in engine_kwargs:
            engine_kwargs.vllm = {}
        engine_kwargs.vllm.kv_transfer_config = connector_config
