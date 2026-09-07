# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Chunked linear top-k KL with fused normalization and analytical gradients.

Like fused linear cross-entropy kernels, this computes the linear-layer VJP
while each vocabulary tile is resident. One later autograd backward consumes
those gradients; the full-trajectory vocabulary tensor is never retained.
"""

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _normalization_parts(X, M, S, V: tl.constexpr, PARTS: tl.constexpr, TEMP: tl.constexpr, BLOCK: tl.constexpr):
        row, part = tl.program_id(0), tl.program_id(1)
        offsets = part * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + row * V + offsets, offsets < V, other=-float("inf"))
        x = (x.to(tl.float32) / TEMP).to(X.dtype.element_ty).to(tl.float32)
        maximum = tl.max(x, 0)
        total = tl.sum(tl.exp(x - maximum), 0)
        tl.store(M + row * PARTS + part, maximum)
        tl.store(S + row * PARTS + part, total)

    @triton.jit
    def _kl_and_dense_grad(
        X,
        IDS,
        TEACHER,
        M,
        S,
        GRAD,
        SELECTED_GRAD,
        LOSS,
        V: tl.constexpr,
        K: tl.constexpr,
        PARTS: tl.constexpr,
        TEMP: tl.constexpr,
        MIN_LOGP: tl.constexpr,
        MAX_LOSS: tl.constexpr,
        BLOCK: tl.constexpr,
        K_BLOCK: tl.constexpr,
        P_BLOCK: tl.constexpr,
    ):
        row, part = tl.program_id(0), tl.program_id(1)
        p = tl.arange(0, P_BLOCK)
        maxima = tl.load(M + row * PARTS + p, p < PARTS, other=-float("inf"))
        sums = tl.load(S + row * PARTS + p, p < PARTS, other=0.0)
        maximum = tl.max(maxima, 0)
        log_z = maximum + tl.log(tl.sum(sums * tl.exp(maxima - maximum), 0))
        k = tl.arange(0, K_BLOCK)
        ids = tl.load(IDS + row * K + k, k < K, other=0)
        selected = tl.load(X + row * V + ids)
        selected = (selected.to(tl.float32) / TEMP).to(X.dtype.element_ty).to(tl.float32)
        student = (selected - log_z).to(X.dtype.element_ty).to(tl.float32)
        teacher = tl.load(TEACHER + row * K + k, k < K, other=-float("inf")).to(tl.float32)
        clamped_teacher = tl.maximum(teacher, MIN_LOGP)
        probability = tl.exp(clamped_teacher)
        loss = tl.sum(tl.where(k < K, probability * (clamped_teacher - tl.maximum(student, MIN_LOGP)), 0.0), 0)
        active = (loss >= 0.0) & (loss <= MAX_LOSS)
        # Autograd through the BF16 selected-logprob cast rounds its incoming
        # derivative to BF16 before normalization backward.
        selected_grad = tl.where((k < K) & (student >= MIN_LOGP) & active, -probability, 0.0)
        selected_grad = selected_grad.to(X.dtype.element_ty).to(tl.float32)
        mass = -tl.sum(selected_grad, 0)
        offsets = part * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + row * V + offsets, offsets < V, other=-float("inf"))
        x = (x.to(tl.float32) / TEMP).to(X.dtype.element_ty).to(tl.float32)
        grad = tl.exp(x - log_z) * mass
        # Keep the unscaled dense gradient until the sparse subtraction, so
        # temperature scaling follows the same cast boundary as autograd.
        tl.store(GRAD + row * V + offsets, grad, offsets < V)
        if part == 0:
            tl.store(SELECTED_GRAD + row * K + k, selected_grad, k < K)
            tl.store(LOSS + row, tl.minimum(tl.maximum(loss, 0.0), MAX_LOSS))

    @triton.jit
    def _scatter_selected_grad(GRAD, IDS, SELECTED_GRAD, V: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        k = tl.arange(0, BLOCK)
        ids = tl.load(IDS + row * K + k, k < K, other=0)
        value = tl.load(SELECTED_GRAD + row * K + k, k < K, other=0.0)
        tl.atomic_add(GRAD + row * V + ids, value, k < K, sem="relaxed")


def _kl_vjp(logits, ids, teacher, temperature, min_logp, max_loss):
    """Return a summed loss and logits gradient, independent of global normalization."""
    temperature = float(torch.tensor(max(float(temperature), 1e-8), dtype=logits.dtype))
    if logits.is_cuda:
        if triton is None:
            raise RuntimeError("streamopd-cf fused KL requires Triton on CUDA")
        rows, vocab = logits.shape
        topk = ids.shape[1]
        block = 8192
        parts = triton.cdiv(vocab, block)
        maxima = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
        sums = torch.empty_like(maxima)
        # FP32 sparse accumulation supports duplicate target ids and avoids
        # unsupported 16-bit atomics on older GPUs.
        grad = torch.empty_like(logits, dtype=torch.float32)
        selected_grad = torch.empty_like(teacher, dtype=torch.float32)
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
        _normalization_parts[(rows, parts)](logits, maxima, sums, vocab, parts, temperature, block)
        _kl_and_dense_grad[(rows, parts)](
            logits,
            ids,
            teacher,
            maxima,
            sums,
            grad,
            selected_grad,
            losses,
            vocab,
            topk,
            parts,
            temperature,
            -float("inf") if min_logp is None else float(min_logp),
            float("inf") if max_loss is None else float(max_loss),
            block,
            triton.next_power_of_2(topk),
            triton.next_power_of_2(parts),
        )
        _scatter_selected_grad[(rows,)](grad, ids, selected_grad, vocab, topk, triton.next_power_of_2(topk))
        return losses.sum(), (grad.to(logits.dtype) / temperature)
    # CPU reference keeps the same cast boundaries as verl's chunked top-k loss.
    with torch.enable_grad():
        x = logits.detach().requires_grad_(True)
        scaled = x / torch.as_tensor(temperature, device=x.device, dtype=x.dtype)
        student = (scaled.float().gather(-1, ids) - torch.logsumexp(scaled.float(), -1, keepdim=True)).to(x.dtype)
        target = teacher
        if min_logp is not None:
            student, target = student.clamp_min(min_logp), target.clamp_min(min_logp)
        losses = (target.float().exp() * (target.float() - student.float())).sum(-1).clamp_min(0.0)
        if max_loss is not None:
            losses = losses.clamp_max(max_loss)
        loss = losses.sum()
        return loss.detach(), torch.autograd.grad(loss, x)[0]


class _LinearTopKKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, ids, teacher, chunk_size, temperature, min_logp, max_loss):
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_size):
            end = min(hidden.shape[0], start + chunk_size)
            h = hidden[start:end]
            logits = h @ weight.t()
            chunk_loss, grad_logits = _kl_vjp(
                logits, ids[start:end], teacher[start:end], temperature, min_logp, max_loss
            )
            loss += chunk_loss
            grad_hidden[start:end] = grad_logits @ weight
            grad_weight.add_((grad_logits.t() @ h).float())
        ctx.save_for_backward(grad_hidden, grad_weight.to(weight.dtype))
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        return grad_hidden * grad_output, grad_weight * grad_output, None, None, None, None, None, None


def linear_topk_kl(
    hidden, weight, teacher_ids, teacher_logprobs, *, chunk_size=512, temperature=1.0, min_logp=None, max_loss=None
):
    """Compute exact top-k forward KL without retaining full vocabulary logits.

    Returns the unnormalized token-loss sum. The caller divides by the valid
    token count over the complete policy batch, accounting for DP averaging.
    Teacher targets are constants; higher-order differentiation is unsupported.
    """
    if chunk_size < 1:
        raise ValueError("loss chunk_size must be positive")
    if (
        hidden.ndim != 2
        or weight.ndim != 2
        or teacher_ids.ndim != 2
        or hidden.shape[-1] != weight.shape[-1]
        or teacher_ids.shape != teacher_logprobs.shape
        or hidden.shape[0] != teacher_ids.shape[0]
    ):
        raise ValueError("hidden states and teacher targets must have matching token rows")
    if hidden.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("streamopd-cf fused KL supports BF16 and FP32 hidden states")
    if teacher_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("teacher token IDs must be integer tensors")
    if any(tensor.device != hidden.device for tensor in (weight, teacher_ids, teacher_logprobs)):
        raise ValueError("streamopd-cf loss tensors must be on the same device")
    if teacher_ids.shape[-1] < 1:
        raise ValueError("teacher top-k must be positive")
    if teacher_ids.numel() and (int(teacher_ids.min()) < 0 or int(teacher_ids.max()) >= weight.shape[0]):
        raise ValueError("teacher token IDs are outside the student vocabulary")
    return _LinearTopKKL.apply(
        hidden.contiguous(),
        weight,
        teacher_ids.long().contiguous(),
        teacher_logprobs.detach().contiguous(),
        chunk_size,
        temperature,
        min_logp,
        max_loss,
    )
