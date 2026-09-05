# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import copy
import os

import pytest
import torch
import torch.nn.functional as F

from verl.experimental.streamopd_kv import Qwen3ReverseTrainer
from verl.experimental.streamopd_kv.reverse_attention import ReverseKVSlotPool
from verl.experimental.streamopd_kv.snapshot_io import HostSlotLayerKV

MODEL_PATH = os.environ.get("STREAMOPD_TEST_MODEL_PATH")


def _capture_host_kv(model, input_ids: torch.Tensor) -> list[HostSlotLayerKV]:
    with torch.no_grad():
        cache = model(input_ids=input_ids, use_cache=True, return_dict=True).past_key_values
    layers = getattr(cache, "layers", None)
    if layers is None:
        values = cache.to_legacy_cache()
    else:
        values = [(layer.keys, layer.values) for layer in layers]
    return [
        HostSlotLayerKV(
            key.squeeze(0).transpose(0, 1).contiguous().cpu().pin_memory(),
            value.squeeze(0).transpose(0, 1).contiguous().cpu().pin_memory(),
        )
        for key, value in values
    ]


class _CrossEntropy:
    def __init__(self, targets: torch.Tensor, valid_from: int) -> None:
        self.targets = targets
        self.valid_positions = torch.arange(targets.shape[1], device="cpu") >= valid_from

    def compact(self, logits: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, int]:
        loss = F.cross_entropy(
            logits.float().flatten(0, 1),
            self.targets.index_select(1, positions).flatten(),
            reduction="sum",
        )
        return loss, positions.numel()

    def __call__(self, logits: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, int]:
        local = self.valid_positions[start:end].nonzero(as_tuple=False).flatten()
        positions = (start + local).to(logits.device)
        return self.compact(logits.index_select(1, local.to(logits.device)), positions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    bool(MODEL_PATH) and not os.path.isdir(MODEL_PATH), reason="requested local Qwen3 model is unavailable"
)
def test_qwen3_fixed_slot_wavefront_matches_full_sequence() -> None:
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(23)
    if MODEL_PATH:
        baseline = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        # Keep the chain-rule test runnable in GPU CI without a model download.
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=256,
        )
        config._attn_implementation = "sdpa"
        baseline = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
    baseline = baseline.cuda().eval()
    reverse = copy.deepcopy(baseline).cuda().eval()
    tokens = [
        torch.arange(1, 67, device="cuda").unsqueeze(0),
        torch.arange(101, 135, device="cuda").unsqueeze(0),
    ]
    sequences = [item[:, :-1] for item in tokens]
    losses = [_CrossEntropy(item[:, 1:], valid_from=8) for item in tokens]

    baseline_loss = torch.zeros((), device="cuda")
    for sequence, loss_fn in zip(sequences, losses, strict=True):
        logits = baseline(input_ids=sequence, use_cache=False, return_dict=True).logits
        sample_loss, _ = loss_fn(logits, 0, sequence.shape[1])
        baseline_loss += sample_loss
    baseline_loss.backward()

    sources = [_capture_host_kv(reverse, sequence) for sequence in sequences]
    first = sources[0][0]
    padded_lengths = [80, 48]
    slots = ReverseKVSlotPool(
        batch_size=2,
        token_capacity=80,
        num_layers=len(sources[0]),
        num_kv_heads=first.key.shape[1],
        head_dim=first.key.shape[-1],
        page_size=8,
        dtype=torch.bfloat16,
        device="cuda",
    )
    slots.prepare_next(sources, [sequence.shape[1] for sequence in sequences], padded_lengths)
    slots.activate_next()
    result = Qwen3ReverseTrainer(reverse, chunk_size=16, page_size=8).backward(
        sequences,
        losses,
        state=slots.state(),
        on_depth_committed=slots.release_current_range,
    )
    slots.finish_current()

    torch.testing.assert_close(result.loss_sum, baseline_loss.detach(), rtol=1e-2, atol=2e-1)
    assert result.chunks == 8
    assert result.backward_calls == 5
    assert result.lm_head_tokens < result.dense_lm_head_tokens
    assert result.dense_lm_head_tokens == 120
    assert result.padded_model_tokens == 128
    parameter_name = "model.layers.0.self_attn.q_proj.weight"
    expected = dict(baseline.named_parameters())[parameter_name].grad.float()
    actual = dict(reverse.named_parameters())[parameter_name].grad.float()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine.item() > 0.995
    relative_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-8)
    assert relative_error.item() < 0.09


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reverse_slot_reuses_released_pages_without_reallocation() -> None:
    torch.manual_seed(29)

    def sources(lengths: list[int]) -> list[list[HostSlotLayerKV]]:
        return [
            [
                HostSlotLayerKV(
                    torch.randn((length, 2, 16), dtype=torch.bfloat16, pin_memory=True),
                    torch.randn((length, 2, 16), dtype=torch.bfloat16, pin_memory=True),
                )
                for _ in range(2)
            ]
            for length in lengths
        ]

    current = sources([64, 32])
    following = sources([48, 64])
    pool = ReverseKVSlotPool(
        batch_size=2,
        token_capacity=64,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        page_size=16,
        dtype=torch.bfloat16,
        device="cuda",
    )
    pool.prepare_next(current, [64, 32], [64, 32])
    pool.activate_next()
    addresses = [
        (layer.key.data_ptr(), layer.value.data_ptr(), layer.key_grad.data_ptr(), layer.value_grad.data_ptr())
        for layer in pool.layers
    ]
    pool.prepare_next(following, [48, 64], [48, 64])
    for active, start, end in (([0], 48, 64), ([0], 32, 48), ([0, 1], 16, 32), ([0, 1], 0, 16)):
        pool.release_current_range(active, start, end)
    pool.finish_current()
    pool.activate_next()

    assert addresses == [
        (layer.key.data_ptr(), layer.value.data_ptr(), layer.key_grad.data_ptr(), layer.value_grad.data_ptr())
        for layer in pool.layers
    ]
    assert pool.next_loaded_pages == 7
    assert pool.copy_cuda_seconds(reused_only=True) > 0
    for row, trajectory in enumerate(following):
        for layer_idx, trace in enumerate(trajectory):
            torch.testing.assert_close(pool.layers[layer_idx].key[row, : trace.length], trace.key.cuda())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reverse_slot_prefetches_one_chunk_group_into_inactive_kv() -> None:
    torch.manual_seed(30)

    def sources(lengths: list[int]) -> list[list[HostSlotLayerKV]]:
        return [
            [
                HostSlotLayerKV(
                    torch.randn((length, 2, 16), dtype=torch.bfloat16, pin_memory=True),
                    torch.randn((length, 2, 16), dtype=torch.bfloat16, pin_memory=True),
                )
                for _ in range(2)
            ]
            for length in lengths
        ]

    current = sources([64, 32])
    following = sources([48, 64])
    pool = ReverseKVSlotPool(
        batch_size=2,
        token_capacity=64,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        page_size=16,
        dtype=torch.bfloat16,
        device="cuda",
        prefetch_kv=True,
    )
    pool.prepare_next(current, [64, 32], [64, 32])
    pool.activate_next()
    active_addresses = [
        (layer.key.data_ptr(), layer.value.data_ptr(), layer.key_grad.data_ptr(), layer.value_grad.data_ptr())
        for layer in pool.layers
    ]
    assert pool._inactive_layers is not None
    inactive_addresses = [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in pool._inactive_layers]

    pool.prepare_next(following, [48, 64], [48, 64])
    for active, start, end in (([0], 48, 64), ([0], 32, 48), ([0, 1], 16, 32), ([0, 1], 0, 16)):
        pool.release_current_range(active, start, end)
    pool.finish_current()
    pool.activate_next()

    assert inactive_addresses == [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in pool.layers]
    assert [(key_grad, value_grad) for _, _, key_grad, value_grad in active_addresses] == [
        (layer.key_grad.data_ptr(), layer.value_grad.data_ptr()) for layer in pool.layers
    ]
    assert pool.slot_bytes == pool.kv_pair_bytes * 3
    assert pool.next_loaded_pages == 7
    assert pool.copy_cuda_seconds(reused_only=True) > 0
    for row, trajectory in enumerate(following):
        for layer_idx, trace in enumerate(trajectory):
            torch.testing.assert_close(pool.layers[layer_idx].key[row, : trace.length], trace.key.cuda())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reverse_slot_reuses_one_pinned_group_for_pageable_sources() -> None:
    torch.manual_seed(31)

    def sources(lengths: list[int]) -> list[list[HostSlotLayerKV]]:
        return [
            [
                HostSlotLayerKV(
                    torch.randn((length, 2, 16), dtype=torch.bfloat16),
                    torch.randn((length, 2, 16), dtype=torch.bfloat16),
                )
                for _ in range(2)
            ]
            for length in lengths
        ]

    first = sources([64, 32])
    following = sources([48, 64])
    pool = ReverseKVSlotPool(
        batch_size=2,
        token_capacity=64,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        page_size=16,
        dtype=torch.bfloat16,
        device="cuda",
    )
    pool.prepare_next(first, [64, 32], [64, 32])
    pool.activate_next()
    pinned_addresses = [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in pool._pinned_layers or ()]
    assert pinned_addresses
    assert pool.pinned_staging_groups == 1

    pool.prepare_next(following, [48, 64], [48, 64])
    for active, start, end in (([0], 48, 64), ([0], 32, 48), ([0, 1], 16, 32), ([0, 1], 0, 16)):
        pool.release_current_range(active, start, end)
    pool.finish_current()
    pool.activate_next()

    assert pinned_addresses == [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in pool._pinned_layers or ()]
    assert pool.pinned_staging_groups == 2
    assert pool.pinned_staging_bytes == (64 + 32 + 48 + 64) * 2 * 2 * 16 * 2 * 2
    assert pool.pinned_staging_capacity_bytes == pool.slot_bytes // 2
    for row, trajectory in enumerate(following):
        for layer_idx, trace in enumerate(trajectory):
            torch.testing.assert_close(pool.layers[layer_idx].key[row, : trace.length], trace.key.cuda())

    pool.release_current_range([0], 0, 48)
    pool.release_current_range([1], 0, 64)
    pool.finish_current()
    pool.copy_executor.shutdown(wait=True)
    pinned_layers = pool.detach_pinned_layers()
    replacement = ReverseKVSlotPool(
        batch_size=2,
        token_capacity=64,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        page_size=16,
        dtype=torch.bfloat16,
        device="cuda",
        pinned_layers=pinned_layers,
    )
    replacement.prepare_next(first, [64, 32], [64, 32])
    replacement.activate_next()
    assert pinned_addresses == [
        (layer.key.data_ptr(), layer.value.data_ptr()) for layer in replacement._pinned_layers or ()
    ]
    assert replacement.pinned_staging_allocation_seconds == 0
