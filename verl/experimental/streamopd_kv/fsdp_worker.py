# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.engine_workers import TrainingWorker

from .attention import LayerKVTrace
from .protocol import TrajectoryKey
from .qwen3 import Qwen3ReverseTrainer
from .snapshot_io import cleanup_vllm_snapshot, load_vllm_snapshot


def _reverse_backward_calls(lengths: list[int], chunk_size: int) -> int:
    if not lengths:
        return 0
    return max(math.ceil(length / chunk_size) for length in lengths)


def _partition_reverse_microbatches(
    lengths: list[int], *, max_batch_size: int, max_batch_tokens: int
) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, length in enumerate(lengths):
        if current and (len(current) >= max_batch_size or current_tokens + length > max_batch_tokens):
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += length
    if current:
        groups.append(current)
    return groups


def _has_valid_response(sample: TensorDict) -> bool:
    response_mask = sample["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.values()
    return bool(response_mask.sum().item())


def _forward_kl_topk_sum(
    logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    temperature: float,
    use_chunked_topk: bool,
    log_prob_min_clamp: float | None,
    loss_max_clamp: float | None,
) -> torch.Tensor:
    """Match verl's native FSDP ``forward_kl_topk`` numerical path."""

    temperature_tensor = torch.as_tensor(max(temperature, 1e-8), dtype=logits.dtype, device=logits.device)
    scaled_logits = logits / temperature_tensor
    ids = teacher_ids.long().to(logits.device)
    teacher = teacher_logprobs.to(logits.device)
    if use_chunked_topk:
        scaled_logits_fp32 = scaled_logits.float()
        student = (scaled_logits_fp32.gather(-1, ids) - torch.logsumexp(scaled_logits_fp32, dim=-1, keepdim=True)).to(
            scaled_logits.dtype
        )
    else:
        student = torch.log_softmax(scaled_logits, dim=-1).gather(-1, ids)
    if log_prob_min_clamp is not None:
        student = student.clamp_min(log_prob_min_clamp)
        teacher = teacher.clamp_min(log_prob_min_clamp)
    teacher = teacher.float()
    student = student.float()
    token_loss = (teacher.exp() * (teacher - student)).sum(dim=-1).clamp_min(0.0)
    if loss_max_clamp is not None:
        token_loss = token_loss.clamp_max(loss_max_clamp)
    return token_loss.sum()


class StreamOPDKVTrainingWorker(TrainingWorker):
    """FSDP actor worker for exact direct forward-KL reverse training."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.streamopd_config = config.extra_context["streamopd_kv"]
        self.distillation_config = config.extra_context["distillation"]
        loss_config = self.distillation_config.distillation_loss
        if loss_config.loss_mode != "forward_kl_topk":
            raise NotImplementedError("StreamOPD-KV FSDP MVP requires loss_mode='forward_kl_topk'")
        if loss_config.use_policy_gradient or loss_config.use_task_rewards:
            raise NotImplementedError(
                "StreamOPD-KV FSDP MVP currently supports direct distillation-only updates "
                "(use_policy_gradient=false, use_task_rewards=false)"
            )
        if self.engine_config.ulysses_sequence_parallel_size != 1:
            raise NotImplementedError("StreamOPD-KV FSDP MVP does not support Ulysses sequence parallelism")
        self._accum_policy_version: int | None = None
        self._accum_next_step = 0
        self._accum_global_valid_tokens = 0.0

    def _reset_accumulation(self, *, zero_grad: bool) -> None:
        if zero_grad:
            self.engine.optimizer_zero_grad()
        self._accum_policy_version = None
        self._accum_next_step = 0
        self._accum_global_valid_tokens = 0.0

    @staticmethod
    def _sample_tensor(sample: TensorDict, name: str) -> torch.Tensor:
        value = sample[name]
        return value.values() if value.is_nested else value

    def _loss_fn(
        self,
        teacher_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        valid_positions: torch.Tensor,
        temperature: float,
    ):
        loss_config = self.distillation_config.distillation_loss

        def loss(logits: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, int]:
            local_mask = valid_positions[start:end]
            valid_count = int(local_mask.sum().item())
            if valid_count == 0:
                return logits.sum() * 0.0, 0
            return (
                _forward_kl_topk_sum(
                    logits[0, local_mask],
                    teacher_ids[start:end][local_mask],
                    teacher_logprobs[start:end][local_mask],
                    temperature=temperature,
                    use_chunked_topk=loss_config.use_chunked_topk,
                    log_prob_min_clamp=loss_config.log_prob_min_clamp,
                    loss_max_clamp=loss_config.loss_max_clamp,
                ),
                valid_count,
            )

        return loss

    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        disable_auto_offload = bool(tu.pop(data, key="disable_auto_offload", default=False))
        accumulation_step = int(tu.get_non_tensor_data(data, "streamopd_accumulation_step", 0))
        accumulation_steps = int(tu.get_non_tensor_data(data, "streamopd_accumulation_steps", 1))
        if not 0 <= accumulation_step < accumulation_steps:
            raise ValueError(f"invalid StreamOPD accumulation trigger {accumulation_step}/{accumulation_steps}")
        finalize = accumulation_step == accumulation_steps - 1
        with self.engine.train_mode(
            disable_auto_offload=disable_auto_offload,
            zero_grad_on_exit=finalize,
        ):
            return self._train_streamopd_mini_batch(
                data,
                accumulation_step=accumulation_step,
                accumulation_steps=accumulation_steps,
            )

    def _train_streamopd_mini_batch(
        self,
        data: TensorDict,
        *,
        accumulation_step: int,
        accumulation_steps: int,
    ) -> TensorDict:
        epochs = int(tu.get_non_tensor_data(data, "epochs", 1))
        if epochs != 1:
            raise ValueError("strict StreamOPD-KV requires ppo_epochs=1")

        all_samples = list(data.unbind(0))
        policy_versions = {int(sample["streamopd_policy_version"]) for sample in all_samples}
        if len(policy_versions) != 1:
            raise RuntimeError(f"StreamOPD cohort mixes policy versions: {sorted(policy_versions)}")
        local_policy_version = next(iter(policy_versions))
        minimum_version = torch.tensor(local_policy_version, device=self.device_name, dtype=torch.int64)
        maximum_version = minimum_version.clone()
        dist.all_reduce(minimum_version, op=dist.ReduceOp.MIN, group=self.engine.get_data_parallel_group())
        dist.all_reduce(maximum_version, op=dist.ReduceOp.MAX, group=self.engine.get_data_parallel_group())
        if minimum_version.item() != maximum_version.item():
            raise RuntimeError(
                "StreamOPD data-parallel ranks mix policy versions: "
                f"minimum={minimum_version.item()}, maximum={maximum_version.item()}"
            )
        # V1 pads a cohort to the data-parallel world size with a minimal
        # response whose mask is all zero. It has no serving KV snapshot and
        # must not enter reverse recomputation or snapshot cleanup.
        samples = [sample for sample in all_samples if _has_valid_response(sample)]
        trajectory_ids = [str(sample["streamopd_trajectory_id"]) for sample in samples]
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise RuntimeError("StreamOPD cohort contains duplicate trajectory identities")
        chunk_size = int(self.streamopd_config.reverse_chunk_size)
        trace_lengths = [self._sample_tensor(sample, "input_ids").numel() - 1 for sample in samples]
        local_chunks = sum(math.ceil(length / chunk_size) for length in trace_lengths)
        reverse_microbatches = _partition_reverse_microbatches(
            trace_lengths,
            max_batch_size=int(self.streamopd_config.reverse_batch_size),
            max_batch_tokens=int(self.streamopd_config.reverse_batch_max_tokens),
        )
        local_backward_calls = sum(
            _reverse_backward_calls([trace_lengths[idx] for idx in group], chunk_size) for group in reverse_microbatches
        )
        synchronized_calls_tensor = torch.tensor(local_backward_calls, device=self.device_name, dtype=torch.int64)
        dist.all_reduce(synchronized_calls_tensor, op=dist.ReduceOp.MAX, group=self.engine.get_data_parallel_group())
        synchronized_backward_calls = int(synchronized_calls_tensor.item())
        if synchronized_backward_calls < 1:
            raise RuntimeError("StreamOPD cohort contains no trainable token trace")

        local_valid_tokens = sum(int(self._sample_tensor(sample, "response_mask").sum().item()) for sample in samples)
        global_valid_tokens = torch.tensor(local_valid_tokens, device=self.device_name, dtype=torch.float32)
        dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM, group=self.engine.get_data_parallel_group())
        if global_valid_tokens.item() < 1:
            raise RuntimeError("StreamOPD cohort contains no valid response tokens")

        if accumulation_step == 0:
            if self._accum_policy_version is not None:
                raise RuntimeError("StreamOPD started a new cohort while gradient accumulation is active")
            self.engine.optimizer_zero_grad()
            self._accum_policy_version = local_policy_version
            self._accum_next_step = 0
            self._accum_global_valid_tokens = 0.0
        elif self._accum_policy_version != local_policy_version or self._accum_next_step != accumulation_step:
            raise RuntimeError(
                "StreamOPD accumulation order/version mismatch: "
                f"expected version={self._accum_policy_version}, step={self._accum_next_step}; "
                f"got version={local_policy_version}, step={accumulation_step}"
            )

        model = self.engine.module
        parameter_dtype = next(model.parameters()).dtype
        forward_dtype = getattr(self.engine, "_autocast_dtype", parameter_dtype)

        def mixed_precision_context():
            if forward_dtype == torch.float32:
                return nullcontext()
            return torch.autocast(device_type=get_device_name(), dtype=forward_dtype)

        was_training = model.training
        started = time.perf_counter()
        total_loss = torch.zeros((), device=self.device_name)
        full_forward_validation_loss = torch.zeros((), device=self.device_name)
        processed_backward_calls = 0
        handoff_seconds = 0.0
        paths_to_cleanup = [str(sample["streamopd_kv_path"]) for sample in samples]
        step_succeeded = False
        finalize = accumulation_step == accumulation_steps - 1

        def backward_context(chunk_idx: int, trajectory_chunks: int):
            del chunk_idx, trajectory_chunks
            nonlocal processed_backward_calls
            is_last = processed_backward_calls == synchronized_backward_calls - 1
            processed_backward_calls += 1
            return self.engine._gradient_sync_context(is_last_micro_batch=is_last)

        try:
            model.eval()
            for group in reverse_microbatches:
                snapshots = []
                try:
                    sequences = []
                    trajectory_layers = []
                    loss_fns = []
                    for sample_idx in group:
                        sample = samples[sample_idx]
                        sequence = self._sample_tensor(sample, "input_ids").long().to(self.device_name)
                        prompt = self._sample_tensor(sample, "prompts")
                        if prompt.numel() < 1:
                            raise RuntimeError("StreamOPD trajectory has an empty prompt")
                        response_mask = self._sample_tensor(sample, "response_mask").bool().to(self.device_name)
                        trace_ids = sequence[:-1]
                        trajectory_id = str(sample["streamopd_trajectory_id"])
                        policy_version = int(sample["streamopd_policy_version"])
                        base_path = str(sample["streamopd_kv_path"])
                        snapshot = load_vllm_snapshot(
                            base_path,
                            key=TrajectoryKey(policy_version, trajectory_id),
                            tp_rank=0,
                            expected_tp_size=1,
                            expected_token_ids=trace_ids.cpu().tolist(),
                            expected_prompt_length=prompt.numel(),
                            device=self.device_name,
                        ).acquire(policy_version)
                        snapshots.append(snapshot)
                        if snapshot.layers[0][0].dtype != forward_dtype:
                            raise RuntimeError(
                                "rollout KV dtype does not match the trainer forward dtype: "
                                f"KV={snapshot.layers[0][0].dtype}, trainer={forward_dtype}"
                            )
                        handoff_seconds += snapshot.handoff_seconds
                        teacher_ids = self._sample_tensor(sample, "teacher_ids")[: trace_ids.numel()].to(
                            self.device_name
                        )
                        teacher_logprobs = self._sample_tensor(sample, "teacher_logprobs")[: trace_ids.numel()].to(
                            self.device_name
                        )
                        valid_positions = torch.zeros(trace_ids.numel(), dtype=torch.bool, device=self.device_name)
                        first_target = prompt.numel() - 1
                        valid_positions[first_target : first_target + response_mask.numel()] = response_mask
                        temperature = float(tu.get_non_tensor_data(sample, "temperature", 1.0))
                        sequences.append(trace_ids.unsqueeze(0))
                        trajectory_layers.append([LayerKVTrace(key, value) for key, value in snapshot.layers])
                        loss_fns.append(self._loss_fn(teacher_ids, teacher_logprobs, valid_positions, temperature))

                    if self.streamopd_config.validate_full_forward_loss:
                        with torch.no_grad(), mixed_precision_context():
                            for sequence, loss_fn in zip(sequences, loss_fns, strict=True):
                                logits = model(input_ids=sequence, use_cache=False, return_dict=True).logits
                                sample_loss, _ = loss_fn(logits, 0, sequence.shape[1])
                                full_forward_validation_loss += sample_loss.float()

                    trainer = Qwen3ReverseTrainer(model, chunk_size=chunk_size)
                    with mixed_precision_context():
                        result = trainer.backward_batched(
                            sequences,
                            trajectory_layers,
                            loss_fns,
                            backward_context=backward_context,
                        )
                    total_loss += result.loss_sum
                finally:
                    for snapshot in snapshots:
                        if snapshot.refcount:
                            snapshot.release()

            dummy_token = torch.zeros((1, 1), dtype=torch.long, device=self.device_name)
            while processed_backward_calls < synchronized_backward_calls:
                with backward_context(0, 1), mixed_precision_context():
                    self.engine.module(input_ids=dummy_token, use_cache=False).logits.sum().mul(0.0).backward()

            self._accum_global_valid_tokens += global_valid_tokens.item()
            self._accum_next_step = accumulation_step + 1
            if finalize:
                scale = self.engine.get_data_parallel_size() / self._accum_global_valid_tokens
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
                grad_norm = self.engine.optimizer_step()
                lr = self.engine.lr_scheduler_step()
            else:
                grad_norm = 0.0
                lr = self.engine.optimizer.param_groups[0]["lr"]
            step_succeeded = True
        except Exception:
            self._reset_accumulation(zero_grad=True)
            raise
        finally:
            model.train(was_training)
            if self.streamopd_config.cleanup_after_step or not step_succeeded:
                for base_path in paths_to_cleanup:
                    cleanup_vllm_snapshot(base_path, tp_size=1)

        elapsed = time.perf_counter() - started
        reported_loss = total_loss.detach()
        dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM, group=self.engine.get_data_parallel_group())
        normalized_loss = reported_loss / global_valid_tokens
        if self.streamopd_config.validate_full_forward_loss:
            dist.all_reduce(
                full_forward_validation_loss,
                op=dist.ReduceOp.SUM,
                group=self.engine.get_data_parallel_group(),
            )
        metrics = {
            "loss": normalized_loss.item(),
            "grad_norm": grad_norm,
            "lr": lr,
            "mfu": 0.0,
            "streamopd/reverse_chunks": local_chunks,
            "streamopd/reverse_backward_calls": processed_backward_calls,
            "streamopd/reverse_microbatches": len(reverse_microbatches),
            "streamopd/handoff_seconds": handoff_seconds,
            "streamopd/training_seconds": elapsed,
            "streamopd/valid_tokens": local_valid_tokens,
            "streamopd/accumulation_step": accumulation_step,
            "streamopd/accumulation_steps": accumulation_steps,
            "streamopd/optimizer_finalized": float(finalize),
            "perf/max_memory_allocated_gb": get_torch_device().max_memory_allocated() / (1024**3),
            "perf/max_memory_reserved_gb": get_torch_device().max_memory_reserved() / (1024**3),
        }
        if self.streamopd_config.validate_full_forward_loss:
            metrics["streamopd/full_forward_validation_loss"] = (
                full_forward_validation_loss / global_valid_tokens
            ).item()
            metrics["streamopd/full_forward_loss_abs_error"] = abs(
                metrics["streamopd/full_forward_validation_loss"] - metrics["loss"]
            )
        if finalize:
            self._reset_accumulation(zero_grad=False)
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
