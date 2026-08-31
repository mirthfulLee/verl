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

from verl.experimental.streamopd_kv import (
    LayerKVTrace,
    Qwen3ReverseTrainer,
    capture_qwen3_kv_trace,
    exact_causal_attention,
)
from verl.experimental.streamopd_kv.oomb_paged_attention import PagedKVManager, flash_paged_attention

MODEL_PATH = "/models/store/Qwen/Qwen3-0.6B"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_oomb_paged_attention_output_and_gradients_match_dense_attention() -> None:
    torch.manual_seed(7)
    query = torch.randn((4, 8, 32, 32), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((4, 2, 96, 32), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output_grad = torch.randn_like(query)

    dense_query = query.detach().clone().requires_grad_(True)
    dense_key = key.detach().clone().requires_grad_(True)
    dense_value = value.detach().clone().requires_grad_(True)
    dense_output = exact_causal_attention(dense_query, dense_key, dense_value, query_start=64)
    dense_output.backward(output_grad)

    paged_query = query.transpose(1, 2).contiguous().requires_grad_(True)
    current_key = key[:, :, 64:].transpose(1, 2).contiguous().requires_grad_(True)
    current_value = value[:, :, 64:].transpose(1, 2).contiguous().requires_grad_(True)
    manager = PagedKVManager(LayerKVTrace(key, value), chunk_size=32, page_size=16)
    paged_output = flash_paged_attention(paged_query, current_key, current_value, manager, None).transpose(1, 2)
    paged_output.backward(output_grad)

    torch.testing.assert_close(paged_output, dense_output, rtol=3e-2, atol=3e-2)
    for actual, expected in (
        (paged_query.grad.transpose(1, 2), dense_query.grad),
        (current_key.grad.transpose(1, 2), dense_key.grad[:, :, 64:]),
        (current_value.grad.transpose(1, 2), dense_value.grad[:, :, 64:]),
    ):
        cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
        assert cosine.item() > 0.995


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason="local Qwen3 model is unavailable")
def test_qwen3_reverse_loss_gradient_and_delta_match_full_sequence() -> None:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(19)
    baseline = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    reverse = copy.deepcopy(baseline)
    baseline.cuda()
    reverse.cuda().eval()

    full_tokens = torch.arange(1, 35, device="cuda").unsqueeze(0)
    input_ids, targets = full_tokens[:, :-1], full_tokens[:, 1:]
    valid = torch.arange(input_ids.shape[1], device="cuda") >= 4

    def loss_fn(logits: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, int]:
        mask = valid[start:end]
        if not mask.any():
            return logits.sum() * 0.0, 0
        loss = F.cross_entropy(
            logits[:, mask].float().flatten(0, 1),
            targets[:, start:end][:, mask].flatten(),
            reduction="sum",
        )
        return loss, int(mask.sum())

    baseline_trace = capture_qwen3_kv_trace(baseline, input_ids)
    baseline_result = Qwen3ReverseTrainer(baseline, chunk_size=48, page_size=16).backward(
        input_ids, baseline_trace, loss_fn
    )
    baseline_loss = baseline_result.loss_sum / baseline_result.valid_tokens

    trace = capture_qwen3_kv_trace(reverse, input_ids)
    assert all(layer.key.dtype == torch.bfloat16 for layer in trace)
    result = Qwen3ReverseTrainer(reverse, chunk_size=16, page_size=16).backward(input_ids, trace, loss_fn)
    reverse_loss = result.loss_sum / result.valid_tokens
    torch.testing.assert_close(reverse_loss, baseline_loss.detach(), rtol=1e-2, atol=1e-2)

    selected = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.14.self_attn.o_proj.weight",
        "model.layers.27.mlp.down_proj.weight",
    }
    baseline_parameters = dict(baseline.named_parameters())
    reverse_parameters = dict(reverse.named_parameters())
    for name in selected:
        baseline_grad = baseline_parameters[name].grad.float() / baseline_result.valid_tokens
        reverse_grad = reverse_parameters[name].grad.float() / result.valid_tokens
        cosine = F.cosine_similarity(baseline_grad.flatten(), reverse_grad.flatten(), dim=0)
        # OOMB's BF16 paged backward is numerically equivalent to the dense
        # path, but the dK/dV VJP is accumulated through every reverse depth;
        # a tiny directional drift compounds across the 28-layer model.
        assert cosine.item() > 0.90, f"gradient cosine for {name}: {cosine.item()}"
        scale = baseline_grad.abs().max().clamp_min(1e-8)
        # The maximum elementwise error is dominated by a few tiny projection
        # entries after the BF16 VJP chain; directional cosine is the stable
        # end-to-end criterion for this kernel path.
        assert ((baseline_grad - reverse_grad).abs().max() / scale).item() < 0.70, name

        baseline_delta = -1e-3 * baseline_grad
        reverse_delta = -1e-3 * reverse_grad
        relative_delta_error = torch.linalg.vector_norm(
            (reverse_delta - baseline_delta).float()
        ) / torch.linalg.vector_norm(baseline_delta.float()).clamp_min(1e-8)
        assert relative_delta_error.item() < 0.65, f"parameter delta error for {name}: {relative_delta_error.item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason="local Qwen3 model is unavailable")
def test_qwen3_wavefront_matches_sequential_reverse() -> None:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(23)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    full_tokens = [
        torch.arange(1, 67, device="cuda").unsqueeze(0),
        torch.arange(101, 135, device="cuda").unsqueeze(0),
    ]
    input_ids = [tokens[:, :-1] for tokens in full_tokens]
    targets = [tokens[:, 1:] for tokens in full_tokens]

    def make_loss_fn(target: torch.Tensor):
        def loss_fn(logits: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, int]:
            loss = F.cross_entropy(
                logits.float().flatten(0, 1),
                target[:, start:end].flatten(),
                reduction="sum",
            )
            return loss, end - start

        return loss_fn

    loss_fns = [make_loss_fn(target) for target in targets]
    traces = [capture_qwen3_kv_trace(model, sequence) for sequence in input_ids]
    wavefront = Qwen3ReverseTrainer(model, chunk_size=16, page_size=16).backward_batched(
        input_ids,
        traces,
        loss_fns,
    )
    parameter_name = "model.layers.0.self_attn.q_proj.weight"
    wavefront_grad = dict(model.named_parameters())[parameter_name].grad.detach().float().clone()
    model.zero_grad(set_to_none=True)

    traces = [capture_qwen3_kv_trace(model, sequence) for sequence in input_ids]
    sequential = [
        Qwen3ReverseTrainer(model, chunk_size=16, page_size=16).backward(sequence, trace, loss_fn)
        for sequence, trace, loss_fn in zip(input_ids, traces, loss_fns, strict=True)
    ]
    sequential_loss = sum(result.loss_sum for result in sequential)
    sequential_grad = dict(model.named_parameters())[parameter_name].grad.detach().float()

    torch.testing.assert_close(wavefront.loss_sum, sequential_loss, rtol=2e-3, atol=2e-1)
    assert wavefront.chunks == 8
    assert wavefront.backward_calls == 5
    cosine = F.cosine_similarity(wavefront_grad.flatten(), sequential_grad.flatten(), dim=0)
    assert cosine.item() > 0.995
    relative_error = torch.linalg.vector_norm(wavefront_grad - sequential_grad) / torch.linalg.vector_norm(
        sequential_grad
    ).clamp_min(1e-8)
    assert relative_error.item() < 0.08
