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


def _dynamic_reverse_chunk_size(
    max_chunk_size: int,
    min_chunk_size: int,
    microbatch_size: int,
    *,
    max_trace_length: int = 0,
    target_trace_length: int = 0,
    alignment: int = 1,
    available_memory_bytes: int | None = None,
    estimated_base_bytes: int = 0,
    estimated_bytes_per_token: int = 0,
) -> int:
    if (
        min_chunk_size < 1
        or max_chunk_size < min_chunk_size
        or microbatch_size < 1
        or alignment < 1
        or min_chunk_size % alignment
        or max_chunk_size % alignment
    ):
        raise ValueError("invalid dynamic reverse chunk configuration")
    # Start at the largest kernel-friendly chunk. The legacy token-pressure
    # heuristic is retained only for callers without a device memory estimate.
    if available_memory_bytes is None:
        pressure = max(1, math.ceil(microbatch_size / 16))
        if max_trace_length > 0 and target_trace_length > 0:
            pressure = max(pressure, math.ceil(max_trace_length / target_trace_length))
        chunk_size = max_chunk_size // pressure
    else:
        budget = max(0, int(available_memory_bytes * 0.75) - estimated_base_bytes)
        if estimated_bytes_per_token > 0:
            chunk_size = budget // estimated_bytes_per_token
        else:
            chunk_size = max_chunk_size
        chunk_size = min(max_chunk_size, chunk_size)
    chunk_size = max(min_chunk_size, chunk_size // alignment * alignment)
    return min(max_chunk_size, chunk_size)


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


def _memory_limited_reverse_batch_size(
    model: torch.nn.Module,
    *,
    configured_batch_size: int,
    trace_length: int,
    chunk_size: int,
    dtype: torch.dtype,
    available_memory_bytes: int | None,
    reserve_bytes: int = 4 * 1024**3,
) -> int:
    """Choose a stable wavefront width from the pre-policy memory budget."""

    if configured_batch_size < 1 or trace_length < 1 or chunk_size < 1:
        raise ValueError("invalid reverse batch planning configuration")
    if available_memory_bytes is None:
        return configured_batch_size

    candidate = 1 << (configured_batch_size.bit_length() - 1)
    budget = int(available_memory_bytes * 0.85)
    while candidate > 1:
        fixed_bytes, bytes_per_token = _reverse_memory_estimate(
            model,
            trajectory_count=candidate,
            trace_length=trace_length,
            dtype=dtype,
        )
        if reserve_bytes + fixed_bytes + chunk_size * bytes_per_token <= budget:
            break
        candidate //= 2
    return candidate


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
        self._gpu_kv_lease_active = False

    def _reset_accumulation(self, *, zero_grad: bool) -> None:
        if zero_grad:
            self.engine.optimizer_zero_grad()
        self._accum_policy_version = None
        self._accum_next_step = 0
        self._accum_global_valid_tokens = 0.0

    def prepare_reverse_plan(self) -> dict[str, float]:
        """Measure reverse headroom before the first policy version starts."""

        if self._reverse_available_memory_bytes is None:
            try:
                self._reverse_available_memory_bytes = _available_cuda_memory(self.device_name)
            except (AttributeError, RuntimeError):
                self._reverse_available_memory_bytes = 0
        return {
            "available_memory_gib": (self._reverse_available_memory_bytes or 0) / (1024**3),
        }

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
        max_chunk_size = int(self.streamopd_config.reverse_chunk_size)
        min_chunk_size = int(self.streamopd_config.reverse_chunk_min_size)
        page_size = int(self.streamopd_config.reverse_page_size)
        trace_lengths = [self._sample_tensor(sample, "input_ids").numel() - 1 for sample in samples]
        # The memory budget was captured once by prepare_reverse_plan before
        # policy version zero. Planning below is deterministic and does not
        # query the allocator or change kernel shapes after training starts.
        available_memory_bytes = self._reverse_available_memory_bytes or None
        planned_batch_size = _memory_limited_reverse_batch_size(
            model,
            configured_batch_size=int(self.streamopd_config.reverse_batch_size),
            trace_length=max(trace_lengths),
            chunk_size=max_chunk_size,
            dtype=forward_dtype,
            available_memory_bytes=available_memory_bytes,
        )
        reverse_microbatches = _partition_reverse_microbatches(
            trace_lengths,
            max_batch_size=planned_batch_size,
            max_batch_tokens=int(self.streamopd_config.reverse_batch_max_tokens),
        )
        target_trace_length = min(8192, int(self.streamopd_config.reverse_batch_max_tokens))
        reverse_chunk_sizes = []
        for group in reverse_microbatches:
            max_trace_length = max(trace_lengths[idx] for idx in group)
            estimated_base_bytes = 0
            estimated_bytes_per_token = 0
            if available_memory_bytes is not None:
                estimated_base_bytes, estimated_bytes_per_token = _reverse_memory_estimate(
                    model,
                    trajectory_count=len(group),
                    trace_length=max_trace_length,
                    dtype=forward_dtype,
                )
                estimated_base_bytes += 4 * 1024**3
            reverse_chunk_sizes.append(
                _dynamic_reverse_chunk_size(
                    max_chunk_size,
                    min_chunk_size,
                    int(self.streamopd_config.micro_batch_size),
                    max_trace_length=max_trace_length,
                    target_trace_length=target_trace_length,
                    alignment=page_size,
                    available_memory_bytes=available_memory_bytes,
                    estimated_base_bytes=estimated_base_bytes,
                    estimated_bytes_per_token=estimated_bytes_per_token,
                )
            )
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
            for group, chunk_size in zip(reverse_microbatches, reverse_chunk_sizes, strict=True):
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
                        response_mask = self._sample_tensor(sample, "response_mask").bool().cpu()
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
                        streamed_tokens_before_eos += snapshot.streamed_tokens_before_eos
                        streamed_chunks_before_eos += snapshot.streamed_chunks_before_eos
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
                        valid_positions = torch.zeros(trace_ids.numel(), dtype=torch.bool)
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

                    trainer = Qwen3ReverseTrainer(model, chunk_size=chunk_size, page_size=page_size)

                    def release_stage1_snapshots(stage1_snapshots=snapshots) -> None:
                        for snapshot in stage1_snapshots:
                            if snapshot.refcount:
                                snapshot.release()

                    with mixed_precision_context():
                        result = trainer.backward_batched(
                            sequences,
                            trajectory_layers,
                            loss_fns,
                            backward_context=backward_context,
                            stage1_release=release_stage1_snapshots,
                        )
                    max_parallel_trajectories = max(max_parallel_trajectories, result.max_parallel_trajectories)
                    lm_head_tokens += result.lm_head_tokens
                    dense_lm_head_tokens += result.dense_lm_head_tokens
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
            "streamopd/reverse_planned_batch_size": planned_batch_size,
            "streamopd/reverse_max_parallel_trajectories": max_parallel_trajectories,
            "streamopd/lm_head_tokens": lm_head_tokens,
            "streamopd/lm_head_token_fraction": lm_head_tokens / max(1, dense_lm_head_tokens),
            "streamopd/reverse_chunk_size_min": min(reverse_chunk_sizes, default=0),
            "streamopd/reverse_chunk_size_max": max(reverse_chunk_sizes, default=0),
            "streamopd/reverse_page_size": page_size,
            "streamopd/reverse_memory_budget_gib": (available_memory_bytes or 0) / (1024**3),
            "streamopd/handoff_seconds": handoff_seconds,
            "streamopd/kv_streamed_tokens_before_eos": streamed_tokens_before_eos,
            "streamopd/kv_streamed_chunks_before_eos": streamed_chunks_before_eos,
            "streamopd/training_seconds": elapsed,
            "streamopd/valid_tokens": local_valid_tokens,
            "streamopd/accumulation_step": accumulation_step,
            "streamopd/accumulation_steps": accumulation_steps,
            "streamopd/optimizer_finalized": float(finalize),
            "streamopd/gradient_syncs": 1.0,
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
