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
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.experimental.streamopd_cf.loss import linear_topk_kl
from verl.experimental.streamopd_cf.qwen3 import use_qwen3_chunked_forward


def tiny_model():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config)


@pytest.mark.parametrize("chunk_size", [1, 4, 16])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_chunked_graph_matches_full_forward_and_all_parameter_gradients(chunk_size, checkpoint):
    torch.manual_seed(4)
    full = tiny_model().train()
    chunked = copy.deepcopy(full)
    if checkpoint:
        chunked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokens = torch.randint(0, 64, (2, 11))
    # Only the final tokens have a loss. Prefix embedding/projection gradients
    # must still arrive through every chunk boundary.
    target = torch.randint(0, 64, (2, 3))
    expected = full(tokens, use_cache=False).logits
    F.cross_entropy(expected[:, -3:].flatten(0, 1), target.flatten()).backward()
    with use_qwen3_chunked_forward(chunked, chunk_size, backend="sdpa"):
        actual = chunked(tokens)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        F.cross_entropy(actual[:, -3:].flatten(0, 1), target.flatten()).backward()
    for (name, param), (other_name, other) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other_name
        torch.testing.assert_close(other.grad, param.grad, rtol=2e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_committed_iterator_advances_forward_before_final_input_and_preserves_gradients(checkpoint):
    torch.manual_seed(12)
    full = tiny_model().train()
    streamed = copy.deepcopy(full)
    if checkpoint:
        streamed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokens = torch.randint(0, 64, (2, 11))
    expected = full(tokens, use_cache=False).logits
    expected[:, -2:].square().mean().backward()
    events = []
    hook = streamed.model.layers[-1].register_forward_hook(lambda *args: events.append("forward"))

    def chunks():
        yield tokens[:, :4]
        assert events == ["forward"]
        yield tokens[:, 4:8]
        assert events == ["forward", "forward"]
        events.append("teacher_complete")
        yield tokens[:, 8:]

    with use_qwen3_chunked_forward(streamed, 4, backend="sdpa"):
        actual = streamed(chunk_iterator=chunks())
        assert "teacher_complete" in events
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual[:, -2:].square().mean().backward()
    hook.remove()
    for expected, observed in zip(full.parameters(), streamed.parameters(), strict=True):
        torch.testing.assert_close(expected.grad, observed.grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_forwarding_unsupervised_eos_preserves_loss_and_all_gradients(checkpoint):
    torch.manual_seed(7)
    full = tiny_model().train()
    streamed = copy.deepcopy(full)
    if checkpoint:
        streamed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokens = torch.randint(0, 64, (2, 8))
    reference = full(tokens[:, :-1], use_cache=False).logits
    reference.square().mean().backward()
    with use_qwen3_chunked_forward(streamed, 4, backend="sdpa"):
        actual = streamed(chunk_iterator=iter(tokens.split(4, dim=1)))[:, :-1]
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        actual.square().mean().backward()
    for (name, expected), (_, observed) in zip(full.named_parameters(), streamed.named_parameters(), strict=True):
        torch.testing.assert_close(expected.grad, observed.grad, rtol=2e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.4])
@pytest.mark.parametrize("clamps", [(None, None), (-4.0, 0.2)])
def test_fused_linear_kl_matches_reference_loss_and_gradients(temperature, clamps):
    torch.manual_seed(8)
    hidden = torch.randn(7, 12, requires_grad=True)
    weight = torch.randn(31, 12, requires_grad=True)
    ids = torch.randint(0, 31, (7, 4))  # Includes duplicate IDs.
    target = torch.log_softmax(torch.randn(7, 31), -1).gather(-1, ids)
    student = torch.log_softmax((hidden @ weight.t()) / temperature, -1).gather(-1, ids)
    min_logp, max_loss = clamps
    if min_logp is not None:
        student, target = student.clamp_min(min_logp), target.clamp_min(min_logp)
    expected = (target.exp() * (target - student)).sum(-1).clamp_min(0.0)
    if max_loss is not None:
        expected = expected.clamp_max(max_loss)
    expected = expected.sum() / 7
    expected.backward()
    grad_h, grad_w = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    actual = (
        linear_topk_kl(
            hidden, weight, ids, target, chunk_size=3, temperature=temperature, min_logp=min_logp, max_loss=max_loss
        )
        / 7
    )
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden.grad, grad_h, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(weight.grad, grad_w, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("ids", [torch.tensor([[32]]), torch.tensor([[-1]]), torch.tensor([[1.5]]), torch.tensor([1])])
def test_linear_kl_rejects_invalid_targets_before_kernel_launch(ids):
    with pytest.raises(ValueError):
        linear_topk_kl(torch.zeros(1, 4), torch.zeros(32, 4), ids, torch.zeros_like(ids, dtype=torch.float32))
