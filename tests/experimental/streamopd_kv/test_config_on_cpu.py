# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import copy

import pytest
from omegaconf import OmegaConf

from verl.experimental.streamopd_kv import (
    prepare_streamopd_kv_config,
)
from verl.experimental.streamopd_kv.config import _auto_streamopd_runtime_profile
from verl.workers.config.distillation import DistillationConfig, StreamOPDKVConfig


def test_prepare_config_installs_connector_only_for_streamopd() -> None:
    config = OmegaConf.create(
        {
            "trainer": {
                "use_v1": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "v1": {
                    "trainer_mode": "streamopd",
                    "streamopd": {},
                    "sampler": {"max_off_policy_threshold": 8, "max_off_policy_strategy": "drop"},
                },
            },
            "data": {"train_batch_size": 128},
            "algorithm": {"filter_groups": {"enable": True, "metric": "acc"}},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "nnodes": 1,
                    "n_gpus_per_node": 2,
                    "checkpoint_engine": {"backend": "nccl"},
                    "engine_kwargs": {},
                },
                "actor": {
                    "strategy": "fsdp",
                    "ppo_epochs": 1,
                    "loss_agg_mode": "token-mean",
                    "use_torch_compile": False,
                    "fsdp_config": {"param_offload": True, "optimizer_offload": True},
                },
            },
            "distillation": {
                "enabled": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "streamopd_kv": {
                    "enabled": True,
                    "runtime_profile": "manual",
                    "trainer_placement": "teacher",
                    "kv_handoff_dir": "/tmp/test-streamopd",
                },
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk",
                    "use_policy_gradient": False,
                    "use_task_rewards": False,
                },
            },
        }
    )
    config.distillation.teacher_models = {"teacher_model": {"inference": {"name": "vllm"}}}
    prepare_streamopd_kv_config(config)
    connector = config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config
    assert connector.kv_connector == "StreamOPDKVConnector"
    assert connector.kv_connector_extra_config.streamopd_kv_handoff_dir == "/tmp/test-streamopd"
    assert connector.kv_connector_extra_config.streamopd_host_slot_count == 128
    assert connector.kv_connector_extra_config.streamopd_host_slot_tokens == 4096
    assert OmegaConf.to_container(config.trainer.v1.streamopd) == {}
    assert config.trainer.v1.sampler.max_off_policy_threshold == 1
    assert config.trainer.v1.sampler.max_off_policy_strategy == "drop"
    assert config.algorithm.filter_groups.enable is False
    assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "nccl"
    assert config.distillation.streamopd_kv.reverse_slot_max_tokens == 4096
    assert config.distillation.streamopd_kv.reverse_batch_size == 128
    assert config.distillation.streamopd_kv.reverse_batch_max_tokens == 128 * 4096
    assert config.distillation.streamopd_kv.reverse_chunk_size == 4096
    assert config.distillation.streamopd_kv.reverse_chunk_min_size == 64

    stale_config = copy.deepcopy(config)
    stale_config.trainer.v1.trainer_mode = "sync"
    with pytest.raises(ValueError, match="trainer.v1.trainer_mode=streamopd"):
        prepare_streamopd_kv_config(stale_config)

    multi_sample_config = copy.deepcopy(config)
    multi_sample_config.actor_rollout_ref.rollout.n = 2
    with pytest.raises(NotImplementedError, match="rollout.n=1"):
        prepare_streamopd_kv_config(multi_sample_config)

    mismatched_shared = copy.deepcopy(config)
    mismatched_shared.distillation.n_gpus_per_node = 3
    with pytest.raises(ValueError, match="cover every Teacher GPU"):
        prepare_streamopd_kv_config(mismatched_shared)

    dedicated = copy.deepcopy(config)
    dedicated.distillation.streamopd_kv.trainer_placement = "dedicated"
    dedicated.distillation.n_gpus_per_node = 3
    prepare_streamopd_kv_config(dedicated)

    rollout_shared = copy.deepcopy(config)
    rollout_shared.distillation.streamopd_kv.trainer_placement = "rollout"
    rollout_shared.actor_rollout_ref.rollout.checkpoint_engine.backend = "host"
    prepare_streamopd_kv_config(rollout_shared)

    oversized_rollout = copy.deepcopy(rollout_shared)
    oversized_rollout.actor_rollout_ref.rollout.n_gpus_per_node = 3
    with pytest.raises(ValueError, match="cover every Rollout GPU"):
        prepare_streamopd_kv_config(oversized_rollout)

    union = copy.deepcopy(config)
    union.trainer.n_gpus_per_node = 4
    union.distillation.streamopd_kv.trainer_placement = "union"
    union.actor_rollout_ref.rollout.checkpoint_engine.backend = "host"
    prepare_streamopd_kv_config(union)

    undersized_union = copy.deepcopy(union)
    undersized_union.trainer.n_gpus_per_node = 3
    undersized_union.distillation.n_gpus_per_node = 2
    undersized_union.actor_rollout_ref.rollout.n_gpus_per_node = 2
    with pytest.raises(ValueError, match="disjoint Teacher and Rollout"):
        prepare_streamopd_kv_config(undersized_union)

    baseline_placement_flag = copy.deepcopy(config)
    baseline_placement_flag.distillation.colocate_teacher_with_student = True
    with pytest.raises(ValueError, match="sync-baseline option"):
        prepare_streamopd_kv_config(baseline_placement_flag)


def test_auto_runtime_profile_needs_only_resource_allocation() -> None:
    config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 128,
                "max_prompt_length": 1024,
                "max_response_length": 3072,
            },
            "actor_rollout_ref": {
                "actor": {
                    "fsdp_config": {
                        "param_offload": False,
                        "optimizer_offload": False,
                        "use_no_sync_for_gradient_accumulation": False,
                    }
                },
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 2,
                    "data_parallel_size": 1,
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n": 1,
                    "max_model_len": None,
                    "max_num_seqs": 1024,
                    "gpu_memory_utilization": 0.3,
                    "checkpoint_engine": {"backend": "naive"},
                },
            },
            "distillation": {
                "enabled": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "teacher_models": {
                    "teacher_model": {
                        "inference": {
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": None,
                            "max_num_seqs": 1024,
                            "max_num_batched_tokens": 8192,
                            "gpu_memory_utilization": 0.5,
                        }
                    }
                },
                "streamopd_kv": {"trainer_placement": "teacher"},
            },
        }
    )

    resource_config = (
        config.actor_rollout_ref.rollout.n_gpus_per_node,
        config.actor_rollout_ref.rollout.nnodes,
        config.actor_rollout_ref.rollout.tensor_model_parallel_size,
        config.actor_rollout_ref.rollout.pipeline_model_parallel_size,
        config.distillation.n_gpus_per_node,
        config.distillation.nnodes,
    )
    plan = _auto_streamopd_runtime_profile(config)
    assert plan == {
        "profile": "auto",
        "trajectory_tokens": 4096,
        "token_chunk_size": 768,
        "teacher_max_batched_tokens": 4096,
        "vllm_memory_policy": "exclusive_free",
        "teacher_max_num_seqs": 32,
        "rollout_max_num_seqs": 64,
    }
    assert config.actor_rollout_ref.rollout.nnodes == 1
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert config.actor_rollout_ref.rollout.pipeline_model_parallel_size == 1
    assert config.actor_rollout_ref.rollout.n == 1
    assert resource_config == (
        config.actor_rollout_ref.rollout.n_gpus_per_node,
        config.actor_rollout_ref.rollout.nnodes,
        config.actor_rollout_ref.rollout.tensor_model_parallel_size,
        config.actor_rollout_ref.rollout.pipeline_model_parallel_size,
        config.distillation.n_gpus_per_node,
        config.distillation.nnodes,
    )
    assert config.actor_rollout_ref.rollout.max_model_len == 4097
    assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "host"
    assert config.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes == 512
    assert config.actor_rollout_ref.actor.fsdp_config.param_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation is False
    assert config.distillation.streamopd_kv.reverse_batch_size == 0
    assert config.distillation.streamopd_kv.reverse_chunk_size == 0
    assert config.distillation.streamopd_kv.teacher_prefill_max_active_kv_tokens == 0
    assert config.distillation.streamopd_kv.kv_handoff_dir.startswith("/dev/shm/verl-streamopd-kv-")
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.3
    assert config.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization == 0.5
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.verl_exclusive_gpu_memory
    assert config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.get(
        "verl_exclusive_gpu_memory"
    )
    assert config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.get(
        "verl_streaming_teacher_logprobs"
    )

    config.distillation.streamopd_kv.trainer_placement = "rollout"
    shared_rollout_plan = _auto_streamopd_runtime_profile(config)
    assert shared_rollout_plan["vllm_memory_policy"] == "exclusive_free"
    assert config.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes == 128

    config.data.train_batch_size = 256
    config.data.max_response_length = 7168
    config.actor_rollout_ref.rollout.n_gpus_per_node = 4
    config.distillation.n_gpus_per_node = 4
    config.distillation.streamopd_kv.trainer_placement = "dedicated"
    long_context_plan = _auto_streamopd_runtime_profile(config)
    assert long_context_plan["trajectory_tokens"] == 8192
    assert long_context_plan["token_chunk_size"] == 1024
    assert config.distillation.streamopd_kv.rollout_kv_export_strategy == "eos_host"
    assert config.distillation.streamopd_kv.rollout_kv_export_chunk_size == 2048
    assert config.distillation.streamopd_kv.rollout_kv_writer_threads == 4
    assert long_context_plan["teacher_max_batched_tokens"] == 8192
    assert long_context_plan["teacher_max_num_seqs"] == 32
    assert long_context_plan["rollout_max_num_seqs"] == 64
    assert long_context_plan["vllm_memory_policy"] == "exclusive_free"


def test_auto_runtime_profile_preserves_explicit_existing_options() -> None:
    config = OmegaConf.create(
        {
            "data": {"train_batch_size": 128, "max_prompt_length": 1024, "max_response_length": 3072},
            "actor_rollout_ref": {
                "actor": {"fsdp_config": {"param_offload": False, "optimizer_offload": False}},
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 1,
                    "data_parallel_size": 1,
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n": 1,
                    "max_model_len": 6000,
                    "max_num_seqs": 48,
                    "gpu_memory_utilization": 0.7,
                    "checkpoint_engine": {"backend": "host", "update_weights_bucket_megabytes": 256},
                },
            },
            "distillation": {
                "enabled": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "teacher_models": {
                    "teacher_model": {
                        "inference": {
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": 6000,
                            "max_num_seqs": 24,
                            "max_num_batched_tokens": 3072,
                            "gpu_memory_utilization": 0.3,
                        }
                    }
                },
                "streamopd_kv": {"trainer_placement": "teacher", "token_chunk_size": 512},
            },
        }
    )
    explicit = {
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        "actor_rollout_ref.rollout.max_num_seqs",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
        "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens",
        "distillation.streamopd_kv.token_chunk_size",
    }

    plan = _auto_streamopd_runtime_profile(config, explicit_paths=explicit)

    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.7
    assert config.actor_rollout_ref.rollout.max_num_seqs == 48
    assert config.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes == 256
    assert config.distillation.teacher_models.teacher_model.inference.max_num_batched_tokens == 3072
    assert config.distillation.streamopd_kv.token_chunk_size == 512
    assert plan["teacher_max_batched_tokens"] == 3072
    assert plan["token_chunk_size"] == 512
    assert config.actor_rollout_ref.rollout.max_model_len == 4097
    assert config.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization == 0.3
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.verl_exclusive_gpu_memory
    assert config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.get(
        "verl_exclusive_gpu_memory"
    )


def test_prepare_config_applies_auto_runtime_profile_before_validation() -> None:
    config = OmegaConf.create(
        {
            "trainer": {
                "use_v1": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "v1": {
                    "trainer_mode": "streamopd",
                    "streamopd": {},
                    "sampler": {"max_off_policy_threshold": 8, "max_off_policy_strategy": "drop"},
                },
            },
            "data": {"train_batch_size": 128, "max_prompt_length": 1024, "max_response_length": 3072},
            "algorithm": {"filter_groups": None},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n": 1,
                    "nnodes": 1,
                    "n_gpus_per_node": 2,
                    "max_model_len": None,
                    "max_num_seqs": 1024,
                    "gpu_memory_utilization": 0.5,
                    "checkpoint_engine": {"backend": "naive", "update_weights_bucket_megabytes": 2048},
                    "engine_kwargs": {},
                },
                "actor": {
                    "strategy": "fsdp",
                    "ppo_epochs": 1,
                    "loss_agg_mode": "token-mean",
                    "use_torch_compile": False,
                    "fsdp_config": {
                        "param_offload": False,
                        "optimizer_offload": False,
                        "use_no_sync_for_gradient_accumulation": False,
                    },
                },
            },
            "distillation": {
                "enabled": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "teacher_models": {
                    "teacher_model": {
                        "inference": {
                            "tensor_model_parallel_size": 1,
                            "data_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": None,
                            "max_num_seqs": 1024,
                            "max_num_batched_tokens": 8192,
                            "gpu_memory_utilization": 0.5,
                        }
                    }
                },
                "streamopd_kv": {
                    "enabled": True,
                    "runtime_profile": "auto",
                    "trainer_placement": "teacher",
                    "kv_handoff_dir": "/tmp/test-streamopd-auto",
                },
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk",
                    "use_policy_gradient": False,
                    "use_task_rewards": False,
                },
            },
        }
    )

    prepare_streamopd_kv_config(config)

    assert config.distillation.streamopd_kv.runtime_profile == "auto"
    assert config.distillation.streamopd_kv.token_chunk_size == 768
    assert config.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization == 0.5
    assert config.actor_rollout_ref.rollout.max_num_seqs == 64
    assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "host"
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.5
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.verl_exclusive_gpu_memory
    assert config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.get(
        "verl_exclusive_gpu_memory"
    )
    assert config.actor_rollout_ref.actor.fsdp_config.param_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation is False
    assert config.distillation.teacher_models.teacher_model.inference.max_num_batched_tokens == 4096
    assert config.distillation.streamopd_kv.reverse_chunk_size == 4096
    assert config.distillation.streamopd_kv.reverse_batch_size == 128


def test_prepare_config_does_not_mutate_sync_baseline() -> None:
    config = OmegaConf.create(
        {
            "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync", "sync": {}}},
            "distillation": {
                "colocate_teacher_with_student": True,
                "streamopd_kv": {"enabled": False, "kv_handoff_dir": "/tmp/unused"},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "engine_kwargs": {"vllm": {"enforce_eager": True}},
                    "checkpoint_engine": {"backend": "naive"},
                }
            },
        }
    )
    before = OmegaConf.to_container(config, resolve=False)

    prepare_streamopd_kv_config(config)

    assert OmegaConf.to_container(config, resolve=False) == before
    assert "kv_transfer_config" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm


def test_distillation_config_coerces_streamopd_mapping_without_mutating_frozen_field() -> None:
    config = DistillationConfig(
        enabled=False,
        streamopd_kv={
            "enabled": False,
            "reverse_chunk_size": 32,
            "reverse_chunk_min_size": 16,
            "reverse_page_size": 16,
        },
    )
    assert isinstance(config.streamopd_kv, StreamOPDKVConfig)
    assert config.streamopd_kv.reverse_chunk_size == 32


def test_streamopd_runtime_profile_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="runtime_profile"):
        StreamOPDKVConfig(runtime_profile="guess")


def test_streamopd_default_trainer_placement_uses_teacher_and_rollout_pools() -> None:
    assert StreamOPDKVConfig().trainer_placement == "union"


@pytest.fixture
def streamopd_job(monkeypatch):
    """Compose the same public Hydra config used by the training entry point."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from verl.experimental.streamopd_kv import config as stream_config_module

    overrides = [
        "trainer.use_v1=true",
        "trainer.v1.trainer_mode=streamopd",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "data.train_batch_size=4",
        "data.max_prompt_length=64",
        "data.max_response_length=128",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.use_torch_compile=false",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n_gpus_per_node=1",
        "actor_rollout_ref.rollout.nnodes=1",
        "distillation.enabled=true",
        "distillation.n_gpus_per_node=1",
        "distillation.nnodes=1",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
        "distillation.distillation_loss.loss_mode=forward_kl_topk",
        "distillation.distillation_loss.use_policy_gradient=false",
        "distillation.distillation_loss.use_task_rewards=false",
        "distillation.streamopd_kv.enabled=true",
    ]
    directory = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    monkeypatch.setattr(stream_config_module, "version", lambda _: "0.24.0")
    monkeypatch.setattr(
        stream_config_module, "_hydra_task_override_paths", lambda: {x.split("=", 1)[0] for x in overrides}
    )
    return config


def test_public_config_preparation_is_idempotent(streamopd_job):
    prepare_streamopd_kv_config(streamopd_job)
    once = OmegaConf.to_container(streamopd_job, resolve=True)
    prepare_streamopd_kv_config(streamopd_job)
    assert OmegaConf.to_container(streamopd_job, resolve=True) == once


def test_repreparation_refreshes_connector_transport(streamopd_job, monkeypatch):
    from verl.experimental.streamopd_kv import config as stream_config_module

    prepare_streamopd_kv_config(streamopd_job)
    streamopd_job.distillation.streamopd_kv.kv_handoff_dir = "/tmp/new-streamopd-job"
    streamopd_job.data.train_batch_size = 8
    monkeypatch.setattr(
        stream_config_module, "_hydra_task_override_paths", lambda: {"distillation.streamopd_kv.kv_handoff_dir"}
    )
    prepare_streamopd_kv_config(streamopd_job)
    extra = streamopd_job.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config
    assert extra.streamopd_host_slot_count == 8
    assert extra.streamopd_kv_handoff_dir == "/tmp/new-streamopd-job"


def test_named_teacher_preserves_overrides_and_default_template(streamopd_job, monkeypatch):
    from omegaconf import open_dict

    from verl.experimental.streamopd_kv import config as stream_config_module

    teachers = streamopd_job.distillation.teacher_models
    template = OmegaConf.to_container(teachers.teacher_model, resolve=True)
    with open_dict(teachers):
        teachers.custom_teacher = copy.deepcopy(teachers.teacher_model)
        teachers.custom_teacher.inference.max_num_seqs = 2
        teachers.custom_teacher.inference.max_num_batched_tokens = 512
    monkeypatch.setattr(
        stream_config_module,
        "_hydra_task_override_paths",
        lambda: {
            "distillation.teacher_models.custom_teacher.inference.max_num_seqs",
            "distillation.teacher_models.custom_teacher.inference.max_num_batched_tokens",
        },
    )
    prepare_streamopd_kv_config(streamopd_job)
    assert teachers.custom_teacher.inference.max_num_seqs == 2
    assert teachers.custom_teacher.inference.max_num_batched_tokens == 512
    assert teachers.custom_teacher.inference.engine_kwargs.vllm.additional_config.verl_streaming_teacher_logprobs
    assert OmegaConf.to_container(teachers.teacher_model, resolve=True) == template


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("distillation.enabled", False, "distillation.enabled=true"),
        ("distillation.teacher_models", {}, "exactly one teacher"),
        ("distillation.teacher_models.teacher_model.inference.name", "sglang", "vLLM teacher"),
        ("distillation.streamopd_kv.reverse_page_size", 0, "reverse_page_size must be positive"),
        ("distillation.streamopd_kv.reverse_chunk_size", 65, "divisible by reverse_page_size"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", 0, "parallel dimensions must be positive"),
        ("actor_rollout_ref.rollout.agent.default_agent_loop", "tool_agent", "single_turn_agent"),
    ],
)
def test_invalid_public_config_fails_before_auto_mutation(streamopd_job, path, value, message):
    OmegaConf.update(streamopd_job, path, value, merge=False)
    before = OmegaConf.to_container(streamopd_job, resolve=True)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        prepare_streamopd_kv_config(streamopd_job)
    assert OmegaConf.to_container(streamopd_job, resolve=True) == before


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("actor_rollout_ref.rollout.checkpoint_engine.backend", "nccl", "checkpoint_engine.backend=host"),
        ("distillation.streamopd_kv.reverse_slot_max_tokens", 64, "must cover"),
    ],
)
def test_explicit_infeasible_runtime_constraints_are_rejected(streamopd_job, monkeypatch, path, value, message):
    from verl.experimental.streamopd_kv import config as stream_config_module

    OmegaConf.update(streamopd_job, path, value)
    monkeypatch.setattr(stream_config_module, "_hydra_task_override_paths", lambda: {path})
    with pytest.raises(ValueError, match=message):
        prepare_streamopd_kv_config(streamopd_job)


@pytest.mark.parametrize("strategy", ["eos_host", "eos_triton", "incremental_triton"])
def test_host_export_selects_only_its_required_model_runner(monkeypatch, strategy):
    import os

    from verl.experimental.streamopd_kv.config import prepare_streamopd_vllm_environment

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    transfer = {"kv_connector_extra_config": {"streamopd_kv_export_strategy": strategy}}
    prepare_streamopd_vllm_environment(transfer)
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == ("0" if strategy == "eos_host" else None)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    if strategy == "eos_host":
        with pytest.raises(ValueError, match="requires VLLM_USE_V2_MODEL_RUNNER=0"):
            prepare_streamopd_vllm_environment(transfer)
    else:
        prepare_streamopd_vllm_environment(transfer)
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


@pytest.mark.parametrize("installed", ["0.11.0", "0.15.0", "0.16.0", "0.17.0"])
def test_config_rejects_versions_outside_repository_and_legacy_support(streamopd_job, monkeypatch, installed):
    from verl.experimental.streamopd_kv import config as stream_config_module

    monkeypatch.setattr(stream_config_module, "version", lambda _: installed)
    with pytest.raises(RuntimeError, match="0.15.1 or >= 0.18.0"):
        prepare_streamopd_kv_config(streamopd_job)
