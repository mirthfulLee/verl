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
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.engine_workers import TrainingWorker

from .protocol import TrajectoryKey
from .qwen3 import Qwen3ReverseTrainer
from .snapshot_io import load_vllm_snapshot, release_vllm_snapshot


def _reverse_backward_calls(lengths: list[int], chunk_size: int) -> int:
    if not lengths:
        return 0
    return max(math.ceil(length / chunk_size) for length in lengths)


def _reverse_memory_estimate(
    model: torch.nn.Module,
    *,
    trajectory_count: int,
    trace_length: int,
    dtype: torch.dtype,
) -> tuple[int, int]:
    """Estimate fixed KV and per-current-token reverse working-set bytes."""

    config = getattr(model, "config", None)
    base_model = getattr(model, "model", None)
    if config is None and base_model is not None:
        config = getattr(base_model, "config", None)
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    layers = len(getattr(base_model, "layers", ()))
    kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    vocab_size = int(getattr(config, "vocab_size", 0) or 0)
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    if head_dim < 1 and hidden_size > 0:
        query_heads = int(getattr(config, "num_attention_heads", 1) or 1)
        head_dim = hidden_size // query_heads
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    # Account for the staged snapshot plus the contiguous OOMB page backing.
    kv_bytes = trajectory_count * trace_length * layers * kv_heads * head_dim * dtype_bytes * 2 * 2
    # Per-layer hidden, Q/K/V, gated-MLP intermediates, and their backward
    # workspaces are kept live by the reverse recomputation. The coefficient
    # is calibrated against the preflight peak probe rather than allocator
    # retries in the policy loop.
    activation_bytes_per_token = trajectory_count * layers * hidden_size * dtype_bytes * 32
    # The compact LM head still covers every valid response position. Account
    # for logits, log-prob workspace, and the logits gradient retained around
    # the loss/linear backward boundary.
    lm_head_bytes_per_token = trajectory_count * vocab_size * dtype_bytes * 3
    return kv_bytes, activation_bytes_per_token + lm_head_bytes_per_token


def _deferred_training_state_bytes(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> int:
    """Estimate gradient and optimizer tensors that must be loaded onto the GPU."""

    if optimizer is None:
        return 0
    optimizer_name = type(optimizer).__name__.lower()
    seen: set[int] = set()
    reserve = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameter_bytes = parameter.numel() * parameter.element_size()
            if parameter.grad is None or parameter.grad.device.type != get_device_name():
                reserve += parameter_bytes

            state = optimizer.state.get(parameter, {})
            if "adam" in optimizer_name:
                for name in ("exp_avg", "exp_avg_sq"):
                    value = state.get(name)
                    if not isinstance(value, torch.Tensor) or value.device.type != get_device_name():
                        reserve += parameter_bytes
            elif "sgd" in optimizer_name and float(group.get("momentum", 0.0)) > 0:
                value = state.get("momentum_buffer")
                if not isinstance(value, torch.Tensor) or value.device.type != get_device_name():
                    reserve += parameter_bytes
    return reserve


def _unsharded_gradient_reserve_bytes(model: torch.nn.Module, data_parallel_size: int) -> int:
    """Return the extra gradient storage retained by FSDP ``no_sync``."""

    if data_parallel_size < 1:
        raise ValueError("data_parallel_size must be positive")
    local_gradient_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters() if parameter.requires_grad
    )
    return local_gradient_bytes * (data_parallel_size - 1)


@dataclass(frozen=True)
class ReverseSlotPlan:
    batch_size: int
    token_capacity: int
    chunk_size: int
    slot_bytes: int
    estimated_workspace_bytes: int
    prefetch_kv: bool


def _fixed_reverse_slot_plan(
    model: torch.nn.Module,
    *,
    configured_batch_size: int,
    token_capacity: int,
    max_batch_tokens: int,
    max_chunk_size: int,
    min_chunk_size: int,
    page_size: int,
    dtype: torch.dtype,
    available_memory_bytes: int | None,
    reserve_bytes: int = 4 * 1024**3,
) -> ReverseSlotPlan:
    """Choose one stable row count and kernel shape before the first training phase."""

    if token_capacity < 1 or token_capacity % page_size:
        raise ValueError("fixed reverse token capacity must be positive and page aligned")
    max_rows = min(configured_batch_size, max_batch_tokens // token_capacity)
    if max_rows < 1:
        raise ValueError("reverse_batch_max_tokens cannot fit one fixed trajectory slot")
    candidate = 1 << (max_rows.bit_length() - 1)
    best_plan: ReverseSlotPlan | None = None
    best_score: tuple[int, bool, int, int] | None = None
    smallest_attempt: tuple[int, int, int, int] | None = None
    while candidate:
        base_slot_bytes, bytes_per_token = _reverse_memory_estimate(
            model,
            trajectory_count=candidate,
            trace_length=token_capacity,
            dtype=dtype,
        )
        if available_memory_bytes is None:
            chunk_limit = max_chunk_size
        else:
            workspace_budget = available_memory_bytes - reserve_bytes - base_slot_bytes
            chunk_limit = min(max_chunk_size, workspace_budget // max(1, bytes_per_token))
        smallest_attempt = (candidate, base_slot_bytes, bytes_per_token, chunk_limit)
        chunk_size = 0
        slot_bytes = base_slot_bytes
        prefetch_kv = False
        for proposed in range(chunk_limit // page_size * page_size, min_chunk_size - 1, -page_size):
            if token_capacity % proposed:
                continue
            proposed_prefetch_kv = proposed == token_capacity
            proposed_slot_bytes = base_slot_bytes + (base_slot_bytes // 2 if proposed_prefetch_kv else 0)
            if (
                available_memory_bytes is not None
                and reserve_bytes + proposed_slot_bytes + proposed * bytes_per_token > available_memory_bytes
            ):
                continue
            chunk_size = proposed
            slot_bytes = proposed_slot_bytes
            prefetch_kv = proposed_prefetch_kv
            break
        if chunk_size >= min_chunk_size:
            plan = ReverseSlotPlan(
                batch_size=candidate,
                token_capacity=token_capacity,
                chunk_size=chunk_size,
                slot_bytes=slot_bytes,
                estimated_workspace_bytes=chunk_size * bytes_per_token,
                prefetch_kv=prefetch_kv,
            )
            # Maximize useful model tokens per launch. For equal-sized tiles,
            # avoid a singleton batch and then prefer the longer chunk to
            # reduce wavefront depth and kernel launch count.
            score = (
                plan.batch_size * plan.chunk_size,
                plan.batch_size > 1,
                plan.chunk_size,
                plan.batch_size,
            )
            if best_score is None or score > best_score:
                best_plan = plan
                best_score = score
        candidate //= 2
    if best_plan is not None:
        return best_plan
    candidate, slot_bytes, bytes_per_token, chunk_limit = smallest_attempt or (0, 0, 0, 0)
    raise RuntimeError(
        "preflight could not fit one fixed reverse slot with the minimum chunk size: "
        f"available={available_memory_bytes}, reserve={reserve_bytes}, candidate_rows={candidate}, "
        f"slot_bytes={slot_bytes}, workspace_bytes_per_token={bytes_per_token}, "
        f"chunk_limit={chunk_limit}, minimum_chunk={min_chunk_size}"
    )


def _available_cuda_memory(device: str | torch.device) -> int:
    """Return driver-free memory plus PyTorch's immediately reusable cache."""

    device_module = get_torch_device()
    free_driver, _ = device_module.mem_get_info(device)
    reserved = device_module.memory_reserved(device)
    allocated = device_module.memory_allocated(device)
    return int(free_driver + max(0, reserved - allocated))


def _partition_reverse_microbatches(
    lengths: list[int], *, max_batch_size: int, max_batch_tokens: int
) -> list[list[int]]:
    if max_batch_size < 1 or max_batch_tokens < 1:
        raise ValueError("reverse batch size and token limit must be positive")
    groups: list[list[int]] = []
    current: list[int] = []
    current_max_length = 0
    for idx in sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True):
        length = lengths[idx]
        if length < 1:
            raise ValueError(f"reverse trace length must be positive, got {length}")
        next_max_length = max(current_max_length, length)
        if current and (len(current) >= max_batch_size or next_max_length * (len(current) + 1) > max_batch_tokens):
            groups.append(current)
            current = []
            current_max_length = 0
        current.append(idx)
        current_max_length = max(current_max_length, length)
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

    normalized_temperature = max(temperature, 1e-8)
    if normalized_temperature == 1.0:
        scaled_logits = logits
    else:
        temperature_tensor = torch.as_tensor(normalized_temperature, dtype=logits.dtype, device=logits.device)
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


class _StreamOPDTopKLoss:
    """Top-k loss that lets reverse recomputation skip unused LM-head rows."""

    def __init__(
        self,
        teacher_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        valid_positions: torch.Tensor,
        *,
        temperature: float,
        use_chunked_topk: bool,
        log_prob_min_clamp: float | None,
        loss_max_clamp: float | None,
    ) -> None:
        if valid_positions.device.type != "cpu" or valid_positions.dtype != torch.bool:
            raise ValueError("StreamOPD compact loss positions must be a CPU bool tensor")
        if valid_positions.ndim != 1:
            raise ValueError("StreamOPD compact loss positions must be one-dimensional")
        if teacher_ids.shape[0] != valid_positions.numel() or teacher_logprobs.shape[0] != valid_positions.numel():
            raise ValueError("teacher artifacts and compact loss positions must cover the same token range")
        self.teacher_ids = teacher_ids
        self.teacher_logprobs = teacher_logprobs
        self.valid_positions = valid_positions
        self.temperature = temperature
        self.use_chunked_topk = use_chunked_topk
        self.log_prob_min_clamp = log_prob_min_clamp
        self.loss_max_clamp = loss_max_clamp

    def compact(self, logits: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, int]:
        valid_count = positions.numel()
        if valid_count == 0:
            return logits.sum() * 0.0, 0
        return (
            _forward_kl_topk_sum(
                logits[0],
                self.teacher_ids.index_select(0, positions),
                self.teacher_logprobs.index_select(0, positions),
                temperature=self.temperature,
                use_chunked_topk=self.use_chunked_topk,
                log_prob_min_clamp=self.log_prob_min_clamp,
                loss_max_clamp=self.loss_max_clamp,
            ),
            valid_count,
        )

    def __call__(self, logits: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, int]:
        local_positions = self.valid_positions[start:end].nonzero(as_tuple=False).flatten()
        positions = (start + local_positions).to(logits.device)
        selected_logits = logits.index_select(1, local_positions.to(logits.device))
        return self.compact(selected_logits, positions)


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
        self._reverse_available_memory_bytes: int | None = None
        self._reverse_slot_plan: ReverseSlotPlan | None = None
        self._reverse_slot_pool = None
        self._reverse_pinned_layers = None
        self._gpu_kv_lease_active = False
        self._kv_prefetch_executor: ThreadPoolExecutor | None = None

    def _get_kv_prefetch_executor(self) -> ThreadPoolExecutor:
        if self._kv_prefetch_executor is None:
            self._kv_prefetch_executor = ThreadPoolExecutor(
                max_workers=int(self.streamopd_config.kv_prefetch_workers),
                thread_name_prefix="streamopd-kv-prefetch",
            )
        return self._kv_prefetch_executor

    def _reset_accumulation(self, *, zero_grad: bool) -> None:
        if zero_grad:
            self.engine.optimizer_zero_grad()
        self._accum_policy_version = None
        self._accum_next_step = 0
        self._accum_global_valid_tokens = 0.0

    def prepare_reverse_plan(self) -> dict[str, float]:
        """Plan reverse shapes from a clear pool without allocating GPU slots."""

        model = self.engine.module
        if self._reverse_available_memory_bytes is None:
            try:
                self._reverse_available_memory_bytes = _available_cuda_memory(self.device_name)
            except (AttributeError, RuntimeError):
                self._reverse_available_memory_bytes = 0
        metrics = {
            "available_memory_gib": (self._reverse_available_memory_bytes or 0) / (1024**3),
        }
        if self._reverse_slot_plan is None:
            deferred_training_state_bytes = _deferred_training_state_bytes(model, self.engine.optimizer)
            unsharded_gradient_reserve_bytes = 0
            if bool(self.engine_config.use_no_sync_for_gradient_accumulation):
                unsharded_gradient_reserve_bytes = _unsharded_gradient_reserve_bytes(
                    model,
                    self.engine.get_data_parallel_size(),
                )
            offloaded_parameter_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
                if parameter.device.type != get_device_name()
            )
            metrics["deferred_training_state_gib"] = deferred_training_state_bytes / (1024**3)
            metrics["unsharded_gradient_reserve_gib"] = unsharded_gradient_reserve_bytes / (1024**3)
            metrics["offloaded_parameter_gib"] = offloaded_parameter_bytes / (1024**3)
            parameter_dtype = next(model.parameters()).dtype
            forward_dtype = getattr(self.engine, "_autocast_dtype", parameter_dtype)
            page_size = int(self.streamopd_config.reverse_page_size)
            token_capacity = math.ceil(int(self.streamopd_config.reverse_slot_max_tokens) / page_size) * page_size
            configured_batch_size = int(self.streamopd_config.reverse_batch_size)
            self._reverse_slot_plan = _fixed_reverse_slot_plan(
                model,
                configured_batch_size=configured_batch_size,
                token_capacity=token_capacity,
                max_batch_tokens=int(self.streamopd_config.reverse_batch_max_tokens),
                max_chunk_size=int(self.streamopd_config.reverse_chunk_size),
                min_chunk_size=int(self.streamopd_config.reverse_chunk_min_size),
                page_size=page_size,
                dtype=forward_dtype,
                available_memory_bytes=self._reverse_available_memory_bytes or None,
                reserve_bytes=(
                    int(float(self.streamopd_config.reverse_slot_reserve_gib) * 1024**3)
                    + deferred_training_state_bytes
                    + unsharded_gradient_reserve_bytes
                    + offloaded_parameter_bytes
                ),
            )
            metrics.update(
                {
                    "slot_batch_size": float(self._reverse_slot_plan.batch_size),
                    "slot_token_capacity": float(self._reverse_slot_plan.token_capacity),
                    "slot_chunk_size": float(self._reverse_slot_plan.chunk_size),
                    "slot_prefetch_kv": float(self._reverse_slot_plan.prefetch_kv),
                    "slot_gib": self._reverse_slot_plan.slot_bytes / (1024**3),
                    "estimated_workspace_gib": self._reverse_slot_plan.estimated_workspace_bytes / (1024**3),
                    "runtime_required_free_gib": (
                        self._reverse_slot_plan.estimated_workspace_bytes
                        + self._reverse_slot_plan.slot_bytes
                        + deferred_training_state_bytes
                        + unsharded_gradient_reserve_bytes
                        + offloaded_parameter_bytes
                        + int(float(self.streamopd_config.reverse_slot_reserve_gib) * 1024**3)
                    )
                    / (1024**3),
                }
            )
        else:
            metrics.update(
                {
                    "slot_batch_size": float(self._reverse_slot_plan.batch_size),
                    "slot_token_capacity": float(self._reverse_slot_plan.token_capacity),
                    "slot_chunk_size": float(self._reverse_slot_plan.chunk_size),
                    "slot_prefetch_kv": float(self._reverse_slot_plan.prefetch_kv),
                    "slot_gib": self._reverse_slot_plan.slot_bytes / (1024**3),
                    "estimated_workspace_gib": self._reverse_slot_plan.estimated_workspace_bytes / (1024**3),
                }
            )
        return metrics

    def allocate_reverse_slots(self) -> None:
        """Allocate Trainer-only KV slots after shared inference enters sleep."""

        if self._reverse_slot_pool is not None:
            return
        if self._reverse_slot_plan is None:
            raise RuntimeError("reverse slots cannot be allocated before preflight planning")
        from .reverse_attention import ReverseKVSlotPool

        model = self.engine.module
        config = getattr(model, "config", None)
        base_model = getattr(model, "model", None)
        if config is None and base_model is not None:
            config = getattr(base_model, "config", None)
        hidden_size = int(getattr(config, "hidden_size", 0) or 0)
        query_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        head_dim = int(getattr(config, "head_dim", 0) or 0) or hidden_size // query_heads
        forward_dtype = getattr(self.engine, "_autocast_dtype", next(model.parameters()).dtype)
        self._reverse_slot_pool = ReverseKVSlotPool(
            batch_size=self._reverse_slot_plan.batch_size,
            token_capacity=self._reverse_slot_plan.token_capacity,
            num_layers=len(getattr(base_model, "layers", ())),
            num_kv_heads=int(getattr(config, "num_key_value_heads", 0) or 0),
            head_dim=head_dim,
            page_size=int(self.streamopd_config.reverse_page_size),
            dtype=forward_dtype,
            device=self.device_name,
            prefetch_kv=self._reverse_slot_plan.prefetch_kv,
            pinned_layers=self._reverse_pinned_layers,
        )
        self._reverse_pinned_layers = None

    def release_reverse_slots(self) -> None:
        """Release Trainer-only KV slots before a shared vLLM process wakes."""

        if self._gpu_kv_lease_active:
            raise RuntimeError("cannot release reverse slots while a training KV lease is active")
        if self._reverse_slot_pool is None:
            return
        self._reverse_slot_pool.copy_executor.shutdown(wait=True)
        self._reverse_pinned_layers = self._reverse_slot_pool.detach_pinned_layers()
        self._reverse_slot_pool = None

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
        return _StreamOPDTopKLoss(
            teacher_ids,
            teacher_logprobs,
            valid_positions,
            temperature=temperature,
            use_chunked_topk=loss_config.use_chunked_topk,
            log_prob_min_clamp=loss_config.log_prob_min_clamp,
            loss_max_clamp=loss_config.loss_max_clamp,
        )

    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        # Rollout may seal several host-resident microbatches ahead of the
        # trainer, but only the currently scheduled microbatch may acquire GPU
        # storage for its KV trace.
        if self._gpu_kv_lease_active:
            raise RuntimeError("StreamOPD trainer already holds a GPU KV lease for another microbatch")
        self._gpu_kv_lease_active = True
        try:
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
        finally:
            self._gpu_kv_lease_active = False

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
        streamed_tokens_before_eos = 0
        streamed_chunks_before_eos = 0
        model = self.engine.module
        parameter_dtype = next(model.parameters()).dtype
        forward_dtype = getattr(self.engine, "_autocast_dtype", parameter_dtype)
        page_size = int(self.streamopd_config.reverse_page_size)
        trace_lengths = [self._sample_tensor(sample, "input_ids").numel() - 1 for sample in samples]
        # The memory budget was captured once by prepare_reverse_plan before
        # the first training phase. Planning below is deterministic and does
        # not query the allocator or change kernel shapes after training starts.
        available_memory_bytes = self._reverse_available_memory_bytes or None
        if self._reverse_slot_plan is None or self._reverse_slot_pool is None:
            raise RuntimeError("reverse slots were not prepared before the first training phase")
        if max(trace_lengths) > self._reverse_slot_plan.token_capacity:
            raise RuntimeError(
                "trajectory exceeds the preflight reverse token capacity: "
                f"trace={max(trace_lengths)}, slot={self._reverse_slot_plan.token_capacity}"
            )
        planned_batch_size = self._reverse_slot_plan.batch_size
        reverse_microbatches = _partition_reverse_microbatches(
            trace_lengths,
            max_batch_size=planned_batch_size,
            max_batch_tokens=int(self.streamopd_config.reverse_batch_max_tokens),
        )
        reverse_chunk_sizes = [self._reverse_slot_plan.chunk_size] * len(reverse_microbatches)
        local_chunks = sum(
            sum(math.ceil(trace_lengths[idx] / chunk_size) for idx in group)
            for group, chunk_size in zip(reverse_microbatches, reverse_chunk_sizes, strict=True)
        )
        local_backward_calls = sum(
            _reverse_backward_calls([trace_lengths[idx] for idx in group], chunk_size)
            for group, chunk_size in zip(reverse_microbatches, reverse_chunk_sizes, strict=True)
        )
        synchronized_calls_tensor = torch.tensor(local_backward_calls, device=self.device_name, dtype=torch.int64)
        dist.all_reduce(synchronized_calls_tensor, op=dist.ReduceOp.MAX, group=self.engine.get_data_parallel_group())
        synchronized_backward_calls = int(synchronized_calls_tensor.item())
        if synchronized_backward_calls < 1:
            raise RuntimeError("StreamOPD cohort contains no trainable token trace")
        dummy_backward_calls = synchronized_backward_calls - local_backward_calls

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

        def mixed_precision_context():
            if forward_dtype == torch.float32:
                return nullcontext()
            return torch.autocast(device_type=get_device_name(), dtype=forward_dtype)

        was_training = model.training
        started = time.perf_counter()
        total_loss = torch.zeros((), device=self.device_name)
        full_forward_validation_loss = torch.zeros((), device=self.device_name)
        processed_backward_calls = 0
        max_parallel_trajectories = 0
        lm_head_tokens = 0
        dense_lm_head_tokens = 0
        padded_model_tokens = 0
        handoff_seconds = 0.0
        prefetch_host_seconds = 0.0
        prefetch_wait_seconds = 0.0
        prefetch_transfer_seconds = 0.0
        prefetched_snapshots = 0
        paths_to_cleanup = [str(sample["streamopd_kv_path"]) for sample in samples]
        finalize = accumulation_step == accumulation_steps - 1
        # Map sealed Host KV slots ahead of the reverse unit. This
        # queue is intentionally bounded by reverse units rather than
        # trajectories: the trainer still owns at most one GPU KV lease, while
        # the next unit's control lookup can run during the current reverse kernel.
        prefetch_depth = int(self.streamopd_config.kv_prefetch_depth)
        snapshot_specs: dict[int, tuple[str, TrajectoryKey, tuple[int, ...], int]] = {}
        for sample_idx, sample in enumerate(samples):
            input_ids_cpu = self._sample_tensor(sample, "input_ids").detach().cpu()
            if input_ids_cpu.dtype != torch.long:
                input_ids_cpu = input_ids_cpu.long()
            prompt = self._sample_tensor(sample, "prompts")
            snapshot_specs[sample_idx] = (
                str(sample["streamopd_kv_path"]),
                TrajectoryKey(int(sample["streamopd_policy_version"]), str(sample["streamopd_trajectory_id"])),
                tuple(int(token) for token in input_ids_cpu[:-1].tolist()),
                int(prompt.numel()),
            )
        prefetch_executor = self._get_kv_prefetch_executor()
        prefetch_futures: dict[int, list[tuple[int, Future]]] = {}

        def load_host_snapshot(sample_idx: int):
            base_path, key, token_ids, prompt_length = snapshot_specs[sample_idx]
            started = time.perf_counter()
            snapshot = load_vllm_snapshot(
                base_path,
                key=key,
                tp_rank=0,
                expected_tp_size=1,
                expected_token_ids=token_ids,
                expected_prompt_length=prompt_length,
            )
            return snapshot, time.perf_counter() - started

        def schedule_prefetch(group_idx: int) -> None:
            if group_idx >= len(reverse_microbatches) or group_idx in prefetch_futures:
                return
            futures = []
            for sample_idx in reverse_microbatches[group_idx]:
                future = prefetch_executor.submit(load_host_snapshot, sample_idx)
                futures.append((sample_idx, future))
            prefetch_futures[group_idx] = futures

        for group_idx in range(min(len(reverse_microbatches), prefetch_depth + 1)):
            schedule_prefetch(group_idx)

        resolved_host_groups: dict[int, list] = {}

        def host_group_ready(group_idx: int) -> bool:
            return group_idx in resolved_host_groups or all(
                future.done() for _, future in prefetch_futures.get(group_idx, ())
            )

        def resolve_host_group(group_idx: int) -> list:
            nonlocal handoff_seconds, prefetch_host_seconds, prefetch_wait_seconds
            nonlocal prefetched_snapshots, streamed_tokens_before_eos, streamed_chunks_before_eos
            cached = resolved_host_groups.get(group_idx)
            if cached is not None:
                return cached
            host_snapshots = []
            for _, future in prefetch_futures[group_idx]:
                wait_started = time.perf_counter() if not future.done() else None
                snapshot, load_seconds = future.result()
                prefetch_host_seconds += load_seconds
                if wait_started is not None:
                    prefetch_wait_seconds += time.perf_counter() - wait_started
                first_key = snapshot.layers[0].key
                if first_key.dtype != forward_dtype:
                    raise RuntimeError(
                        "rollout KV dtype does not match the trainer forward dtype: "
                        f"KV={first_key.dtype}, trainer={forward_dtype}"
                    )
                handoff_seconds += snapshot.handoff_seconds
                streamed_tokens_before_eos += snapshot.streamed_tokens_before_eos
                streamed_chunks_before_eos += snapshot.streamed_chunks_before_eos
                host_snapshots.append(snapshot)
            prefetch_futures.pop(group_idx)
            prefetched_snapshots += len(host_snapshots)
            schedule_prefetch(group_idx + prefetch_depth + 1)
            resolved_host_groups[group_idx] = host_snapshots
            return host_snapshots

        def backward_context(chunk_idx: int, trajectory_chunks: int):
            del chunk_idx, trajectory_chunks
            nonlocal processed_backward_calls
            is_last = processed_backward_calls == synchronized_backward_calls - 1
            processed_backward_calls += 1
            return self.engine._gradient_sync_context(is_last_micro_batch=finalize and is_last)

        try:
            model.eval()
            self._reverse_slot_pool.reset_metrics()
            for group_idx, (group, chunk_size) in enumerate(
                zip(reverse_microbatches, reverse_chunk_sizes, strict=True)
            ):
                host_snapshots = [] if group_idx > 0 else resolve_host_group(group_idx)
                sequences = []
                loss_fns = []
                for sample_idx in group:
                    sample = samples[sample_idx]
                    sequence = self._sample_tensor(sample, "input_ids").long().to(self.device_name)
                    prompt = self._sample_tensor(sample, "prompts")
                    if prompt.numel() < 1:
                        raise RuntimeError("StreamOPD trajectory has an empty prompt")
                    response_mask = self._sample_tensor(sample, "response_mask").bool().cpu()
                    trace_ids = sequence[:-1]
                    teacher_ids = self._sample_tensor(sample, "teacher_ids")[: trace_ids.numel()].to(self.device_name)
                    teacher_logprobs = self._sample_tensor(sample, "teacher_logprobs")[: trace_ids.numel()].to(
                        self.device_name
                    )
                    valid_positions = torch.zeros(trace_ids.numel(), dtype=torch.bool)
                    first_target = prompt.numel() - 1
                    valid_positions[first_target : first_target + response_mask.numel()] = response_mask
                    temperature = float(tu.get_non_tensor_data(sample, "temperature", 1.0))
                    sequences.append(trace_ids.unsqueeze(0))
                    loss_fns.append(self._loss_fn(teacher_ids, teacher_logprobs, valid_positions, temperature))

                if self.streamopd_config.validate_full_forward_loss:
                    with torch.no_grad(), mixed_precision_context():
                        for sequence, loss_fn in zip(sequences, loss_fns, strict=True):
                            logits = model(input_ids=sequence, use_cache=False, return_dict=True).logits
                            sample_loss, _ = loss_fn(logits, 0, sequence.shape[1])
                            full_forward_validation_loss += sample_loss.float()

                trainer = Qwen3ReverseTrainer(model, chunk_size=chunk_size, page_size=page_size)
                trajectory_layers = [list(snapshot.layers) for snapshot in host_snapshots]
                group_lengths = [sequence.shape[1] for sequence in sequences]
                padded_lengths = [math.ceil(length / chunk_size) * chunk_size for length in group_lengths]
                if group_idx == 0:
                    transfer_started = time.perf_counter()
                    self._reverse_slot_pool.prepare_next(trajectory_layers, group_lengths, padded_lengths)
                    self._reverse_slot_pool.activate_next()
                    prefetch_transfer_seconds += time.perf_counter() - transfer_started
                resolved_host_groups.pop(group_idx, None)
                next_idx = group_idx + 1
                next_prepared = False

                def prepare_next_group(*, block: bool, target_idx: int = next_idx) -> None:
                    nonlocal next_prepared
                    if next_prepared or target_idx >= len(reverse_microbatches):
                        return
                    if not block and not host_group_ready(target_idx):
                        return
                    next_hosts = resolve_host_group(target_idx)
                    next_group = reverse_microbatches[target_idx]
                    next_layers = [list(snapshot.layers) for snapshot in next_hosts]
                    next_lengths = [trace_lengths[idx] for idx in next_group]
                    next_chunk = reverse_chunk_sizes[target_idx]
                    next_padded = [math.ceil(length / next_chunk) * next_chunk for length in next_lengths]
                    self._reverse_slot_pool.prepare_next(next_layers, next_lengths, next_padded)
                    resolved_host_groups.pop(target_idx, None)
                    next_prepared = True

                prepare_next_group(block=False)

                def release_slot_depth(active: Sequence[int], start: int, end: int) -> None:
                    self._reverse_slot_pool.release_current_range(active, start, end)
                    prepare_next_group(block=False)

                with mixed_precision_context():
                    result = trainer.backward(
                        sequences,
                        loss_fns,
                        state=self._reverse_slot_pool.state(),
                        backward_context=backward_context,
                        on_depth_committed=release_slot_depth,
                    )
                self._reverse_slot_pool.finish_current()
                if next_idx < len(reverse_microbatches):
                    prepare_next_group(block=True)
                    transfer_started = time.perf_counter()
                    self._reverse_slot_pool.activate_next()
                    prefetch_transfer_seconds += time.perf_counter() - transfer_started
                max_parallel_trajectories = max(max_parallel_trajectories, result.max_parallel_trajectories)
                lm_head_tokens += result.lm_head_tokens
                dense_lm_head_tokens += result.dense_lm_head_tokens
                padded_model_tokens += result.padded_model_tokens
                total_loss += result.loss_sum

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
        except Exception:
            self._reverse_slot_pool.abort_groups()
            self._reset_accumulation(zero_grad=True)
            raise
        finally:
            for futures in prefetch_futures.values():
                for _, future in futures:
                    future.cancel()
                    try:
                        future.result()
                    except BaseException:
                        # The training exception, if any, is the actionable
                        # error. Futures are joined before slot release so no
                        # background reader can race row reuse.
                        pass
            model.train(was_training)
            for base_path in paths_to_cleanup:
                release_vllm_snapshot(base_path)

        elapsed = time.perf_counter() - started
        defer_gradient_sync = bool(self.engine_config.use_no_sync_for_gradient_accumulation)
        gradient_syncs = float(finalize) if defer_gradient_sync else float(processed_backward_calls)
        reported_loss = total_loss.detach()
        dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM, group=self.engine.get_data_parallel_group())
        normalized_loss = reported_loss / global_valid_tokens
        if self.streamopd_config.validate_full_forward_loss:
            dist.all_reduce(
                full_forward_validation_loss,
                op=dist.ReduceOp.SUM,
                group=self.engine.get_data_parallel_group(),
            )
        slot_copy_seconds = self._reverse_slot_pool.copy_cuda_seconds()
        slot_next_copy_seconds = self._reverse_slot_pool.copy_cuda_seconds(reused_only=True)
        slot_initial_wait_seconds = self._reverse_slot_pool.initial_wait_seconds
        slot_next_wait_seconds = self._reverse_slot_pool.next_wait_seconds
        slot_next_loaded_pages = self._reverse_slot_pool.next_loaded_pages
        slot_loaded_gib = self._reverse_slot_pool.loaded_bytes / (1024**3)
        slot_bytes_gib = self._reverse_slot_pool.slot_bytes / (1024**3)
        slot_copy_enqueue_seconds = self._reverse_slot_pool.copy_enqueue_seconds
        slot_next_copy_enqueue_seconds = self._reverse_slot_pool.next_copy_enqueue_seconds
        slot_pinned_staging_seconds = self._reverse_slot_pool.pinned_staging_seconds
        slot_pinned_staging_allocation_seconds = self._reverse_slot_pool.pinned_staging_allocation_seconds
        slot_pinned_staging_gib = self._reverse_slot_pool.pinned_staging_bytes / (1024**3)
        slot_pinned_staging_capacity_gib = self._reverse_slot_pool.pinned_staging_capacity_bytes / (1024**3)
        metrics = {
            "loss": normalized_loss.item(),
            "grad_norm": grad_norm,
            "lr": lr,
            "mfu": 0.0,
            "streamopd/reverse_chunks": local_chunks,
            "streamopd/reverse_backward_calls": processed_backward_calls,
            "streamopd/reverse_real_backward_calls": local_backward_calls,
            "streamopd/reverse_dummy_backward_calls": dummy_backward_calls,
            "streamopd/reverse_microbatches": len(reverse_microbatches),
            "streamopd/reverse_planned_batch_size": planned_batch_size,
            "streamopd/reverse_max_parallel_trajectories": max_parallel_trajectories,
            "streamopd/lm_head_tokens": lm_head_tokens,
            "streamopd/lm_head_token_fraction": lm_head_tokens / max(1, dense_lm_head_tokens),
            "streamopd/reverse_model_tokens": dense_lm_head_tokens,
            "streamopd/reverse_model_token_fraction": dense_lm_head_tokens / max(1, padded_model_tokens),
            "streamopd/reverse_padding_tokens_trimmed": padded_model_tokens - dense_lm_head_tokens,
            "streamopd/reverse_chunk_size_min": min(reverse_chunk_sizes, default=0),
            "streamopd/reverse_chunk_size_max": max(reverse_chunk_sizes, default=0),
            "streamopd/reverse_page_size": page_size,
            "streamopd/reverse_memory_budget_gib": (available_memory_bytes or 0) / (1024**3),
            "streamopd/reverse_slot_batch_size": planned_batch_size,
            "streamopd/reverse_slot_token_capacity": self._reverse_slot_plan.token_capacity,
            "streamopd/reverse_slot_prefetch_kv": float(self._reverse_slot_plan.prefetch_kv),
            "streamopd/reverse_slot_backing_gib": slot_bytes_gib,
            "streamopd/reverse_slot_loaded_gib": slot_loaded_gib,
            "streamopd/reverse_slot_copy_cuda_seconds": slot_copy_seconds,
            "streamopd/reverse_slot_next_copy_cuda_seconds": slot_next_copy_seconds,
            "streamopd/reverse_slot_copy_enqueue_seconds": slot_copy_enqueue_seconds,
            "streamopd/reverse_slot_next_copy_enqueue_seconds": slot_next_copy_enqueue_seconds,
            "streamopd/reverse_slot_initial_wait_seconds": slot_initial_wait_seconds,
            "streamopd/reverse_slot_next_wait_seconds": slot_next_wait_seconds,
            "streamopd/reverse_slot_overlap_seconds": max(0.0, slot_next_copy_seconds - slot_next_wait_seconds),
            "streamopd/reverse_slot_next_loaded_pages": slot_next_loaded_pages,
            "streamopd/reverse_slot_pinned_staging_seconds": slot_pinned_staging_seconds,
            "streamopd/reverse_slot_pinned_staging_allocation_seconds": slot_pinned_staging_allocation_seconds,
            "streamopd/reverse_slot_pinned_staging_gib": slot_pinned_staging_gib,
            "streamopd/reverse_slot_pinned_staging_capacity_gib": slot_pinned_staging_capacity_gib,
            "streamopd/reverse_slot_pinned_staging_groups": self._reverse_slot_pool.pinned_staging_groups,
            "streamopd/handoff_seconds": handoff_seconds,
            "streamopd/kv_prefetch_host_seconds": prefetch_host_seconds,
            "streamopd/kv_prefetch_wait_seconds": prefetch_wait_seconds,
            "streamopd/kv_prefetch_transfer_seconds": prefetch_transfer_seconds,
            "streamopd/kv_prefetched_snapshots": prefetched_snapshots,
            "streamopd/kv_streamed_tokens_before_eos": streamed_tokens_before_eos,
            "streamopd/kv_streamed_chunks_before_eos": streamed_chunks_before_eos,
            "streamopd/training_seconds": elapsed,
            "streamopd/valid_tokens": local_valid_tokens,
            "streamopd/accumulation_step": accumulation_step,
            "streamopd/accumulation_steps": accumulation_steps,
            "streamopd/optimizer_finalized": float(finalize),
            "streamopd/gradient_syncs": gradient_syncs,
            # MetricsAggregator sums names containing ``total`` across the
            # controller's training units.
            "streamopd/gradient_syncs_total": gradient_syncs,
            "streamopd/defer_gradient_sync": float(defer_gradient_sync),
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
            # Accumulation spans controller units, so the train context cannot
            # clear gradients on every exit. Once the optimizer has stepped,
            # release them explicitly: FSDP1 parameter offload does not move
            # gradient storage, which would otherwise occupy a shared vLLM pool.
            self._reset_accumulation(zero_grad=True)
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
