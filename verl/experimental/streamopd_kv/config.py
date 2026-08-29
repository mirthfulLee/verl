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
    if not stream_config.get("enabled", False):
        return
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("strict StreamOPD-KV requires verl V1 with trainer.v1.trainer_mode=sync")
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
            "streamopd_writer_threads": 4,
        },
    }
    with open_dict(engine_kwargs):
        if "vllm" not in engine_kwargs:
            engine_kwargs.vllm = {}
        engine_kwargs.vllm.kv_transfer_config = connector_config
