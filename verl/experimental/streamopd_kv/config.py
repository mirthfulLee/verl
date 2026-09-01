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
    if not stream_config.get("enabled", False):
        if trainer_mode == "streamopd_colocate":
            raise ValueError("trainer_mode=streamopd_colocate requires distillation.streamopd_kv.enabled=true")
        return
    if not config.trainer.use_v1 or trainer_mode != "streamopd_colocate":
        raise ValueError("strict StreamOPD requires trainer.v1.trainer_mode=streamopd_colocate")
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
    if rollout.n_gpus_per_node < 1:
        raise ValueError("StreamOPD requires a positive standalone rollout GPU count")
    checkpoint_backend = str(rollout.checkpoint_engine.backend)
    if checkpoint_backend == "naive":
        raise ValueError("StreamOPD requires a cross-pool checkpoint engine such as host or nccl")

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
            "streamopd_writer_threads": 4,
        },
    }
    with open_dict(engine_kwargs):
        if "vllm" not in engine_kwargs:
            engine_kwargs.vllm = {}
        engine_kwargs.vllm.kv_transfer_config = connector_config
