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

from verl.experimental.streamopd_kv import Qwen3ReverseTrainer, capture_qwen3_kv_trace

MODEL_PATH = "/models/store/Qwen/Qwen3-0.6B"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason="local Qwen3 model is unavailable")
def test_qwen3_reverse_loss_gradient_and_delta_match_full_sequence() -> None:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(19)
    baseline = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    reverse = copy.deepcopy(baseline)
    baseline.cuda()
    reverse.cuda().eval()

    full_tokens = torch.tensor(
        [[1, 123, 456, 789, 42, 77, 91, 103, 119, 137, 151, 173, 197, 211, 229, 241, 257]],
        device="cuda",
    )
    input_ids, targets = full_tokens[:, :-1], full_tokens[:, 1:]
    valid = torch.arange(input_ids.shape[1], device="cuda") >= 4

    baseline_logits = baseline(input_ids=input_ids, use_cache=False).logits
    baseline_loss = F.cross_entropy(baseline_logits[:, valid].float().flatten(0, 1), targets[:, valid].flatten())
    baseline_loss.backward()

    trace = capture_qwen3_kv_trace(reverse, input_ids)
    assert all(layer.key.dtype == torch.float32 for layer in trace)

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

    result = Qwen3ReverseTrainer(reverse, chunk_size=5).backward(input_ids, trace, loss_fn)
    reverse_loss = result.loss_sum / result.valid_tokens
    torch.testing.assert_close(reverse_loss, baseline_loss.detach(), rtol=3e-3, atol=3e-3)

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
        baseline_grad = baseline_parameters[name].grad.float()
        reverse_grad = reverse_parameters[name].grad.float() / result.valid_tokens
        cosine = F.cosine_similarity(baseline_grad.flatten(), reverse_grad.flatten(), dim=0)
        assert cosine.item() > 0.995, f"gradient cosine for {name}: {cosine.item()}"
        scale = baseline_grad.abs().max().clamp_min(1e-8)
        assert ((baseline_grad - reverse_grad).abs().max() / scale).item() < 0.08, name

        baseline_delta = -1e-3 * baseline_grad
        reverse_delta = -1e-3 * reverse_grad
        relative_delta_error = torch.linalg.vector_norm(
            (reverse_delta - baseline_delta).float()
        ) / torch.linalg.vector_norm(baseline_delta.float()).clamp_min(1e-8)
        assert relative_delta_error.item() < 0.05, f"parameter delta error for {name}: {relative_delta_error.item()}"
