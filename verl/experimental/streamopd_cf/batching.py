# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.distributed as dist

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.workers.engine.utils import prepare_micro_batches
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from .memory import choose_micro_batch_size, estimate_training_workspace


def sequence_lengths(data):
    inputs = data["input_ids"]
    return inputs.offsets().diff() if inputs.is_nested else data["attention_mask"].sum(-1)


def prepare_padded_micro_batches(data, dp_group):
    """Reuse verl packing, also bounding CF's rank-aligned padded token count."""
    minimum = None
    while True:
        batches, indices = prepare_micro_batches(data, dp_group=dp_group, min_num_micro_batch=minimum)
        if not tu.get_non_tensor_data(data, "use_dynamic_bsz", True):
            return batches, indices
        sizes = torch.tensor(
            [(len(batch), int(sequence_lengths(batch).max())) for batch in batches],
            device=get_device_id(),
            dtype=torch.int64,
        )
        if dist.is_initialized() and dp_group is not None:
            dist.all_reduce(sizes, op=dist.ReduceOp.MAX, group=dp_group)
        if int(sizes.prod(-1).max()) <= tu.get_non_tensor_data(data, "max_token_len_per_gpu", None):
            return batches, indices
        # All ranks refine together, so layer and gradient collectives stay aligned.
        minimum = len(batches) + 1


class AutoBatchTrainingWorker(TrainingWorker):
    """Plan token capacity, then retain verl's native full-trajectory F+B."""

    chunked_forward = False

    def __init__(self, config):
        super().__init__(config)
        self.distillation = config.extra_context["distillation"]
        self.batching = self.distillation.batching

    def configure_training_batch(self, data):
        dynamic = self.engine_config.use_dynamic_bsz
        fixed = self.engine_config.micro_batch_size_per_gpu
        token_budget = self.engine_config.max_token_len_per_gpu
        metrics = {}
        if not dynamic:
            if not fixed or fixed < 1:
                raise ValueError("fixed OPD batching requires ppo_micro_batch_size_per_gpu > 0")
        elif token_budget is None or token_budget < 0:
            raise ValueError("ppo_max_token_len_per_gpu must be positive, or 0 for automatic planning")
        elif token_budget == 0:
            from verl.experimental.streamopd_kv.fsdp_worker import (
                _available_cuda_memory,
                _deferred_training_state_bytes,
                _unsharded_gradient_reserve_bytes,
            )

            model = self.engine.module
            config = getattr(model, "module", model).config
            length = self.config.extra_context["max_trajectory_length"]
            settings = self.distillation.streamopd_cf
            fixed_bytes, per_sample = estimate_training_workspace(
                config,
                sequence_length=length,
                chunk_size=settings.forward_chunk_size if self.chunked_forward else length,
                loss_chunk_size=(
                    settings.loss_chunk_size
                    if self.chunked_forward
                    else self.distillation.distillation_loss.chunked_topk_chunk_size
                ),
                dtype_bytes=torch.tensor([], dtype=self.engine._autocast_dtype).element_size(),
                checkpointing=self.model_config.enable_gradient_checkpointing,
            )
            reserve = _deferred_training_state_bytes(model, self.engine.optimizer)
            if self.engine_config.use_no_sync_for_gradient_accumulation:
                reserve += _unsharded_gradient_reserve_bytes(model, self.engine.get_data_parallel_size())
            budget = int(_available_cuda_memory(get_device_id()) * self.batching.memory_fraction)
            budget -= reserve + int(self.batching.reserve_gib * 1024**3)
            limits = torch.tensor([budget, len(data)], device=get_device_id(), dtype=torch.int64)
            dist.all_reduce(limits, op=dist.ReduceOp.MIN, group=self.engine.get_data_parallel_group())
            capacity = choose_micro_batch_size(
                available_bytes=int(limits[0]),
                fixed_bytes=fixed_bytes,
                per_sample_bytes=per_sample,
                batch_cap=int(limits[1]),
            )
            token_budget = capacity * length
            metrics.update(
                {
                    "batching/memory_budget_gib": int(limits[0]) / 1024**3,
                    "batching/estimated_workspace_gib": (fixed_bytes + capacity * per_sample) / 1024**3,
                    "batching/capacity_at_max_length": capacity,
                }
            )
        tu.assign_non_tensor(
            data,
            use_dynamic_bsz=dynamic,
            max_token_len_per_gpu=token_budget,
            micro_batch_size_per_gpu=fixed,
        )
        metrics.update(
            {
                "batching/dynamic": float(dynamic),
                "batching/max_tokens_per_gpu": token_budget if dynamic else 0,
                "batching/fixed_micro_batch_size": 0 if dynamic else fixed,
            }
        )
        return metrics

    def train_batch(self, data):
        # The standard mini-batch loop has already loaded parameters/optimizer.
        metrics = self.configure_training_batch(data)
        output = super().train_batch(data)
        if output is not None:
            tu.get(output, "metrics").update(metrics)
        return output


class AutoBatchActorWorker(ActorRolloutRefWorker):
    actor_worker_cls = AutoBatchTrainingWorker

    def _configure_actor_training_worker(self, training_config, distillation_config):
        training_config.extra_context.update(
            distillation=distillation_config,
            max_trajectory_length=self.config.rollout.prompt_length + self.config.rollout.response_length,
            max_response_length=self.config.rollout.response_length,
        )
