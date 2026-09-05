# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from verl.experimental.streamopd_kv.fsdp_worker import (
    StreamOPDKVTrainingWorker,
    _deferred_training_state_bytes,
    _forward_kl_topk_sum,
    _has_valid_response,
    _partition_reverse_microbatches,
    _reverse_backward_calls,
    _unsharded_gradient_reserve_bytes,
)
from verl.experimental.streamopd_kv.qwen3 import _build_reverse_wavefront, _wavefront_compute_end
from verl.experimental.streamopd_kv.reverse_attention import (
    FixedSlotPageState,
    ReverseKVSlotPool,
    _ContiguousKVBatchView,
)
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs


def test_contiguous_kv_view_keeps_native_gqa_heads_and_accumulates_gradients() -> None:
    layer = SimpleNamespace(
        key=torch.arange(3 * 5 * 2 * 3, dtype=torch.float32).reshape(3, 5, 2, 3),
        value=torch.ones(3, 5, 2, 3),
        key_grad=torch.zeros(3, 5, 2, 3),
        value_grad=torch.zeros(3, 5, 2, 3),
        num_kv_heads=2,
        head_dim=3,
    )
    view = _ContiguousKVBatchView(layer, active=[0, 2], start=2, end=5)

    key, value = view.key_value(query_heads=4)
    assert key.shape == value.shape == (2, 5, 2, 3)
    torch.testing.assert_close(key[1], layer.key[2])

    key_gradient = torch.ones_like(key)
    value_gradient = torch.full_like(value, 2)
    view.accumulate_gradients(key_gradient, value_gradient)
    torch.testing.assert_close(layer.key_grad[0], torch.ones_like(layer.key_grad[0]))
    torch.testing.assert_close(layer.key_grad[1], torch.zeros_like(layer.key_grad[1]))
    torch.testing.assert_close(layer.key_grad[2], torch.ones_like(layer.key_grad[2]))
    torch.testing.assert_close(layer.value_grad[2], torch.full_like(layer.value_grad[2], 2))
    current_key_grad, current_value_grad = view.grad
    assert current_key_grad.shape == current_value_grad.shape == (2, 3, 2, 3)

    with pytest.raises(ValueError, match="query heads"):
        view.key_value(query_heads=3)


def test_reverse_slot_abort_clears_failed_group_metadata() -> None:
    pool = ReverseKVSlotPool.__new__(ReverseKVSlotPool)
    pool.batch_size = 2
    pool.num_pages = 3
    pool._current_lengths = [192]
    pool._next_sources = [[object()]]
    pool._next_lengths = [64]
    pool._next_padded_lengths = [64]
    pool._pending_enqueues = []
    pool.page_states = [[FixedSlotPageState.CURRENT_ACTIVE] * 3 for _ in range(2)]
    pool.load_events = [[object()] * 3 for _ in range(2)]
    pool.free_events = [[object()] * 3 for _ in range(2)]

    pool.abort_groups()

    assert pool._current_lengths == []
    assert pool._next_sources is None
    assert pool._next_lengths == []
    assert pool._next_padded_lengths == []
    assert pool.page_states == [[FixedSlotPageState.FREE] * 3 for _ in range(2)]
    assert pool.load_events == [[None] * 3 for _ in range(2)]
    assert pool.free_events == [[None] * 3 for _ in range(2)]


def test_reverse_preflight_reserves_lazy_adam_state_and_gradients() -> None:
    model = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters())
    parameter_bytes = model.weight.numel() * model.weight.element_size()

    assert _deferred_training_state_bytes(model, optimizer) == 3 * parameter_bytes
    assert _unsharded_gradient_reserve_bytes(model, data_parallel_size=4) == 3 * parameter_bytes
    with pytest.raises(ValueError, match="data_parallel_size"):
        _unsharded_gradient_reserve_bytes(model, data_parallel_size=0)
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.state[model.weight]["exp_avg"] = torch.zeros_like(model.weight)
    assert _deferred_training_state_bytes(model, optimizer) == 3 * parameter_bytes


def test_trainer_rejects_a_second_gpu_kv_lease() -> None:
    worker = StreamOPDKVTrainingWorker.__new__(StreamOPDKVTrainingWorker)
    worker._gpu_kv_lease_active = True
    with pytest.raises(RuntimeError, match="already holds a GPU KV lease"):
        worker.train_mini_batch(TensorDict({}, batch_size=[]))


def test_reverse_microbatch_partition_and_call_count() -> None:
    lengths = [9, 8, 12, 4]
    assert _partition_reverse_microbatches(lengths, max_batch_size=3, max_batch_tokens=20) == [
        [2],
        [0, 1],
        [3],
    ]
    assert _reverse_backward_calls([8, 13], chunk_size=5) == 3
    assert _reverse_backward_calls([8, 12], chunk_size=5) == 3


def test_reverse_wavefront_only_batches_active_trajectories() -> None:
    assert _build_reverse_wavefront([4, 4, 6, 6], chunk_size=1) == [
        (6, [2, 3]),
        (5, [2, 3]),
        (4, [0, 1, 2, 3]),
        (3, [0, 1, 2, 3]),
        (2, [0, 1, 2, 3]),
        (1, [0, 1, 2, 3]),
    ]


def test_reverse_wavefront_trims_only_trailing_page_padding() -> None:
    lengths = [7010, 6577, 6100]

    assert _wavefront_compute_end(lengths, [0, 1], 6144, 8192, page_size=64) == 7040
    assert _wavefront_compute_end(lengths, [0, 1, 2], 4096, 6144, page_size=64) == 6144


def test_zero_loss_synthetic_padding_is_not_trainable() -> None:
    real = torch.tensor([1, 1, 0], dtype=torch.int64)
    padding = torch.zeros(1, dtype=torch.int64)

    assert _has_valid_response(TensorDict({"response_mask": real}, batch_size=[])) is True
    assert _has_valid_response(TensorDict({"response_mask": padding}, batch_size=[])) is False


def test_streamopd_resource_planners(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.experimental.streamopd_kv.planning import (
        kv_bytes_per_token,
        partition_training_units,
        plan_host_kv,
        plan_teacher_admission,
        plan_training_unit_size,
        planned_reverse_width,
    )

    assert partition_training_units(128, 16) == [16] * 8
    assert partition_training_units(130, 16) == [16] * 8 + [2]
    assert (
        plan_training_unit_size(
            train_batch_size=128,
            reverse_wave_size=16,
            resources_overlap=True,
            kv_prefetch_depth=1,
        )
        == 128
    )
    assert (
        plan_training_unit_size(
            train_batch_size=256,
            reverse_wave_size=16,
            resources_overlap=False,
            kv_prefetch_depth=1,
        )
        == 32
    )
    assert (
        plan_training_unit_size(
            train_batch_size=256,
            reverse_wave_size=16,
            resources_overlap=False,
            kv_prefetch_depth=3,
        )
        == 64
    )
    assert (
        planned_reverse_width(
            [{"slot_batch_size": 8.0}, [{"slot_batch_size": 4.0}]],
            fallback=16,
        )
        == 4
    )
    assert plan_teacher_admission(
        expected_trajectories=32,
        trajectory_tokens=1000,
        vllm_capacity_tokens=40000,
        page_size=64,
        max_batched_tokens=2048,
        initial_chunk_tokens=384,
    ) == {
        "active_trajectories": 32,
        "active_kv_tokens": 32768,
        "vllm_capacity_tokens": 40000,
        "safe_capacity_tokens": 40000,
        "trajectory_tokens": 1024,
        "teacher_replicas": 1,
        "prefill_wave_per_replica": 5,
        "prefill_wave": 5,
    }
    replicated = plan_teacher_admission(
        expected_trajectories=32,
        trajectory_tokens=1000,
        vllm_capacity_tokens=40000,
        page_size=64,
        max_batched_tokens=2048,
        initial_chunk_tokens=384,
        teacher_replicas=2,
    )
    assert replicated["active_trajectories"] == 32
    assert replicated["prefill_wave_per_replica"] == 5
    assert replicated["prefill_wave"] == 10
    capped = plan_teacher_admission(
        expected_trajectories=32,
        trajectory_tokens=4096,
        vllm_capacity_tokens=20000,
        page_size=64,
        max_batched_tokens=2048,
        initial_chunk_tokens=1024,
        trajectory_cap=16,
        token_cap=12000,
    )
    assert capped["active_trajectories"] == 2
    assert capped["active_kv_tokens"] == 8192
    with pytest.raises(ValueError, match="cannot fit one trajectory"):
        plan_teacher_admission(
            expected_trajectories=8,
            trajectory_tokens=4096,
            vllm_capacity_tokens=20000,
            page_size=64,
            max_batched_tokens=2048,
            initial_chunk_tokens=1024,
            token_cap=2048,
        )
    with pytest.raises(ValueError, match="vLLM capacity"):
        plan_teacher_admission(
            expected_trajectories=8,
            trajectory_tokens=4096,
            vllm_capacity_tokens=2048,
            page_size=64,
            max_batched_tokens=2048,
            initial_chunk_tokens=1024,
        )

    monkeypatch.setattr(
        "verl.experimental.streamopd_kv.planning.shutil.disk_usage",
        lambda path: SimpleNamespace(total=200 * 1024**3, used=0, free=200 * 1024**3),
    )
    host_plan = plan_host_kv(
        handoff_dir=str(tmp_path / "handoff"),
        global_batch_size=128,
        max_model_len=4096,
        kv_bytes_per_token=112 * 1024,
    )
    assert host_plan["host_kv_required_gib"] == 56.0
    with pytest.raises(ValueError, match="Host KV backing"):
        plan_host_kv(
            handoff_dir=str(tmp_path / "handoff"),
            global_batch_size=512,
            max_model_len=8192,
            kv_bytes_per_token=112 * 1024,
        )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"num_hidden_layers": 4, "num_key_value_heads": 2, '
        '"num_attention_heads": 8, "hidden_size": 512, "head_dim": 64, "vocab_size": 100}'
    )
    assert kv_bytes_per_token(str(model_dir), "bfloat16") == 4 * 2 * 64 * 2 * 2


@pytest.mark.parametrize("use_chunked_topk", [False, True])
def test_reverse_forward_kl_topk_matches_native_numerical_path(use_chunked_topk: bool) -> None:
    torch.manual_seed(31)
    logits = torch.randn(7, 29, dtype=torch.bfloat16, requires_grad=True)
    native_logits = logits.detach().clone().requires_grad_(True)
    teacher_ids = torch.randint(0, logits.shape[-1], (7, 5))
    teacher_logprobs = torch.log_softmax(torch.randn(7, 5), dim=-1)
    temperature = 0.73

    reverse_loss = _forward_kl_topk_sum(
        logits,
        teacher_ids,
        teacher_logprobs,
        temperature=temperature,
        use_chunked_topk=use_chunked_topk,
        log_prob_min_clamp=-10.0,
        loss_max_clamp=10.0,
    )
    scaled_native_logits = native_logits / torch.as_tensor(temperature, dtype=native_logits.dtype)
    if use_chunked_topk:
        native_student = _chunked_topk_log_probs(
            scaled_native_logits.unsqueeze(0), teacher_ids.unsqueeze(0), chunk_size=3
        ).squeeze(0)
    else:
        native_student = F.log_softmax(scaled_native_logits, dim=-1).gather(-1, teacher_ids)
    native_student = native_student.clamp_min(-10.0).float()
    native_teacher = teacher_logprobs.clamp_min(-10.0).float()
    native_loss = (native_teacher.exp() * (native_teacher - native_student)).sum(-1).clamp_min(0.0)
    native_loss = native_loss.clamp_max(10.0).sum()

    reverse_loss.backward()
    native_loss.backward()
    torch.testing.assert_close(reverse_loss, native_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(logits.grad, native_logits.grad, rtol=0.0, atol=0.0)


def test_reverse_forward_kl_topk_temperature_one_avoids_logits_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = torch.randn(3, 11, requires_grad=True)
    teacher_ids = torch.randint(0, logits.shape[-1], (3, 2))
    teacher_logprobs = torch.log_softmax(torch.randn(3, 2), dim=-1)
    original_log_softmax = torch.log_softmax

    def assert_input_aliases_logits(value: torch.Tensor, *args, **kwargs):
        assert value.data_ptr() == logits.data_ptr()
        return original_log_softmax(value, *args, **kwargs)

    monkeypatch.setattr(torch, "log_softmax", assert_input_aliases_logits)
    loss = _forward_kl_topk_sum(
        logits,
        teacher_ids,
        teacher_logprobs,
        temperature=1.0,
        use_chunked_topk=False,
        log_prob_min_clamp=-10.0,
        loss_max_clamp=10.0,
    )
    loss.backward()
    assert logits.grad is not None
