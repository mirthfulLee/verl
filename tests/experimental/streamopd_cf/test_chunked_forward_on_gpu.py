# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.experimental.streamopd_cf.loss import _kl_vjp, linear_topk_kl
from verl.experimental.streamopd_cf.qwen3 import use_qwen3_chunked_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_triton_kl_matches_reference_gradient(dtype, temperature):
    torch.manual_seed(5)
    logits = torch.randn(9, 17011, device="cuda", dtype=dtype, requires_grad=True)
    ids = torch.randint(0, 17011, (9, 32), device="cuda")
    ids[:, 1] = ids[:, 0]
    teacher = torch.randn(9, 32, device="cuda").log_softmax(-1)
    scaled = logits / torch.as_tensor(temperature, device="cuda", dtype=dtype)
    student = (scaled.float().gather(-1, ids) - torch.logsumexp(scaled.float(), -1, keepdim=True)).to(dtype)
    expected = (teacher.exp() * (teacher - student.float())).sum(-1).clamp_min(0).sum()
    grad = torch.autograd.grad(expected, logits)[0]
    actual, actual_grad = _kl_vjp(logits.detach(), ids, teacher, temperature, None, None)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_grad, grad, rtol=0.02 if dtype == torch.bfloat16 else 1e-4, atol=1e-5)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
def test_flash_chunked_forward_and_fused_loss_preserve_all_model_gradients(checkpoint, streamed):
    torch.manual_seed(12)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
    )
    config._attn_implementation = "flash_attention_2"
    full = Qwen3ForCausalLM(config).cuda().bfloat16().train()
    chunked = copy.deepcopy(full)
    if checkpoint:
        chunked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokens = torch.randint(0, 256, (2, 37), device="cuda")
    ids = torch.randint(0, 256, (2, 37, 8), device="cuda")
    teacher = torch.randn(2, 37, 8, device="cuda").log_softmax(-1)
    valid = torch.zeros_like(tokens, dtype=torch.bool)
    valid[:, -5:] = True
    logits = full(tokens, use_cache=False).logits[valid]
    selected = (logits.float().gather(-1, ids[valid]) - torch.logsumexp(logits.float(), -1, keepdim=True)).to(
        logits.dtype
    )
    expected = (teacher[valid].exp() * (teacher[valid] - selected.float())).sum(-1).clamp_min(0).mean()
    expected.backward()

    def loss(hidden, weight):
        hidden = hidden[:, : tokens.shape[1]]
        return linear_topk_kl(hidden[valid], weight, ids[valid], teacher[valid], chunk_size=4) / valid.sum()

    with use_qwen3_chunked_forward(chunked, 11, loss_function=loss):
        if streamed:
            with_eos = torch.cat((tokens, tokens[:, -1:]), dim=1)
            actual = chunked(chunk_iterator=iter(with_eos.split(11, dim=1)))
        else:
            actual = chunked(tokens)
        actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    for (name, p), (_, q) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        cosine = torch.nn.functional.cosine_similarity(p.grad.float().flatten(), q.grad.float().flatten(), dim=0)
        assert cosine > 0.998, (name, cosine.item())
        torch.testing.assert_close(q.grad, p.grad, rtol=0.1, atol=0.01, msg=name)
