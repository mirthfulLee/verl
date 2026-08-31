# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dense paged attention used by StreamOPD reverse training.

The Triton kernels are adapted from wenhaoli-xmu/OOMB at commit
ba07f27e107fbf525b047ee263ad9a34f6850756. Sparse, distributed, and CPU
offload paths are intentionally omitted. StreamOPD uses the dense BF16 CUDA
path directly and fails closed when its kernel constraints are not met.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from .attention import LayerKVTrace


@triton.jit
def _bwd_preprocess_do_o_dot(
    out,
    dout,
    delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    EVEN_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    dout_ptrs = dout + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :]
    if EVEN_M:
        out_block = tl.load(out_ptrs).to(tl.float32)
        dout_block = tl.load(dout_ptrs).to(tl.float32)
    else:
        mask = offs_m[:, None] < seqlen_q
        out_block = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
        dout_block = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(delta + off_hb * seqlen_q_rounded + offs_m, tl.sum(out_block * dout_block, axis=1))


@triton.jit
def _fwd_kernel(
    query,
    page_table,
    out,
    lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_ob,
    stride_oh,
    stride_om,
    stride_kvb,
    stride_kvh,
    stride_kvn,
    nheads,
    seqlen_q,
    q_start_idx,
    headdim,
    seqlen_q_rounded,
    seqlen_k,
    num_kv_heads,
    num_kv_pages,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE
    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = query + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    kv_offs = off_kv_h * stride_kvh + offs_n[:, None] * stride_kvn + offs_d[None, :]
    page_table += off_b * num_kv_pages * 4
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q_idx = q_start_idx + offs_m
    start_m_block += 1
    for kv_block_idx in tl.range(0, tl.cdiv(q_start_idx, BLOCK_N) + start_m_block):
        k_idx = kv_block_idx * BLOCK_N + offs_n
        kv_mask = k_idx[:, None] < seqlen_k
        k_page_ptr = tl.cast(tl.load(page_table), tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(tl.load(page_table + 1), tl.pointer_type(tl.bfloat16))
        k = tl.load(k_page_ptr + kv_offs, mask=kv_mask, other=0.0)
        v = tl.load(v_page_ptr + kv_offs, mask=kv_mask, other=0.0)
        qk = tl.dot(q, k.T)
        qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk, float("-inf"))
        m_ij = tl.maximum(tl.max(qk, axis=1) * softmax_scale, m_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp(m_i - m_ij)[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
        page_table += 4
    acc_o = acc_o * tl.exp(m_i - lse_i)[:, None]
    tl.store(lse + off_hb * seqlen_q_rounded + offs_m, lse_i, mask=offs_m < seqlen_q)
    out_ptrs = out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _bwd_kernel(
    query,
    dout,
    dquery,
    page_table,
    lse,
    delta,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kvb,
    stride_kvh,
    stride_kvn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    nheads,
    seqlen_q,
    q_start_idx,
    headdim,
    seqlen_q_rounded,
    seqlen_k,
    num_kv_heads,
    num_kv_pages,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE
    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = query + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    kv_offs = off_kv_h * stride_kvh + offs_n[:, None] * stride_kvn + offs_d[None, :]
    page_table += off_b * num_kv_pages * 4
    dout_ptrs = dout + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :]
    dquery_ptrs = dquery + off_b * stride_dqb + off_h * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :]
    lse_ptrs = lse + off_hb * seqlen_q_rounded + offs_m
    delta_ptrs = delta + off_hb * seqlen_q_rounded + offs_m
    mask_m = offs_m < seqlen_q
    if EVEN_M:
        q = tl.load(q_ptrs)
        do = tl.load(dout_ptrs)
        lse_i = tl.load(lse_ptrs)
        delta_i = tl.load(delta_ptrs)
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(dout_ptrs, mask=mask_m[:, None], other=0.0)
        lse_i = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        delta_i = tl.load(delta_ptrs, mask=mask_m, other=0.0)
    dq_block = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    q_idx = q_start_idx + offs_m
    start_m_block += 1
    for kv_block_idx in tl.range(0, tl.cdiv(q_start_idx, BLOCK_N) + start_m_block):
        k_idx = kv_block_idx * BLOCK_N + offs_n
        kv_mask = k_idx[:, None] < seqlen_k
        k_page_ptr = tl.cast(tl.load(page_table), tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(tl.load(page_table + 1), tl.pointer_type(tl.bfloat16))
        dk_page_ptr = tl.cast(tl.load(page_table + 2), tl.pointer_type(tl.bfloat16))
        dv_page_ptr = tl.cast(tl.load(page_table + 3), tl.pointer_type(tl.bfloat16))
        k = tl.load(k_page_ptr + kv_offs, mask=kv_mask, other=0.0)
        v = tl.load(v_page_ptr + kv_offs, mask=kv_mask, other=0.0)
        qk = tl.dot(q, k.T)
        qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk, float("-inf"))
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        dv_block = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_page_ptr + kv_offs, dv_block, mask=kv_mask, sem="relaxed")
        dp = tl.dot(do, v.T)
        ds = (p * (dp - delta_i[:, None]) * softmax_scale).to(q.dtype)
        dk_block = tl.dot(ds.T, q)
        tl.atomic_add(dk_page_ptr + kv_offs, dk_block, mask=kv_mask, sem="relaxed")
        dq_block += tl.dot(ds, k)
        page_table += 4
    if EVEN_M:
        tl.store(dquery_ptrs, dq_block)
    else:
        tl.store(dquery_ptrs, dq_block, mask=mask_m[:, None])


class PagedKVManager:
    """GPU-resident BF16 KV/dKV pages for one layer and an equal-length batch."""

    def __init__(self, trace: LayerKVTrace, *, chunk_size: int, page_size: int) -> None:
        key = trace.key.transpose(1, 2).contiguous()
        value = trace.value.transpose(1, 2).contiguous()
        if not key.is_cuda or key.dtype != torch.bfloat16:
            raise TypeError("OOMB paged attention requires CUDA BF16 rollout KV")
        if key.shape[0] < 1:
            raise ValueError("OOMB paged attention requires a non-empty batch")
        if key.shape != value.shape:
            raise ValueError("rollout key/value shapes differ")
        if key.shape[-1] > 256:
            raise ValueError("OOMB paged attention requires head_dim <= 256")
        if page_size < 16 or page_size & (page_size - 1):
            raise ValueError("reverse_page_size must be a power of two and at least 16")
        if chunk_size % page_size:
            raise ValueError("reverse chunk size must be divisible by reverse_page_size")
        self.batch_size = key.shape[0]
        self.page_size = page_size
        self.num_kv_heads = key.shape[2]
        self.head_dim = key.shape[3]
        self.device = key.device
        self.num_kv = 0
        self.last_update_tokens: list[int] = []
        self.last_update_pages: list[int] = []
        self.key_pages: list[torch.Tensor] = []
        self.value_pages: list[torch.Tensor] = []
        self.key_grad_pages: list[torch.Tensor] = []
        self.value_grad_pages: list[torch.Tensor] = []
        self._page_table: torch.Tensor | None = None
        for start in range(0, key.shape[1], chunk_size):
            end = min(start + chunk_size, key.shape[1])
            self._append(key[:, start:end], value[:, start:end])

    def _append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        update_tokens = key.shape[1]
        if update_tokens % self.page_size:
            if self.num_kv % self.page_size:
                raise RuntimeError("only the final OOMB KV update may be page-padded")
            pad_tokens = self.page_size - update_tokens % self.page_size
            padding = (0, 0, 0, 0, 0, pad_tokens)
            key = torch.nn.functional.pad(key, padding)
            value = torch.nn.functional.pad(value, padding)
        # Aligned pages can share the immutable trace backing allocation. This
        # avoids a transient full-KV clone while the stage-1 handoff is sealed;
        # padded tail pages own their newly allocated padded backing instead.
        key_pages = list(key.split(self.page_size, dim=1))
        value_pages = list(value.split(self.page_size, dim=1))
        self.key_pages.extend(key_pages)
        self.value_pages.extend(value_pages)
        self.key_grad_pages.extend(torch.zeros_like(page) for page in key_pages)
        self.value_grad_pages.extend(torch.zeros_like(page) for page in value_pages)
        self.last_update_tokens.append(update_tokens)
        self.last_update_pages.append(len(key_pages))
        self.num_kv += update_tokens
        self._page_table = None

    @property
    def page_table(self) -> torch.Tensor:
        if self._page_table is None:
            pointers = [
                (
                    self.key_pages[page_idx][batch_idx].data_ptr(),
                    self.value_pages[page_idx][batch_idx].data_ptr(),
                    self.key_grad_pages[page_idx][batch_idx].data_ptr(),
                    self.value_grad_pages[page_idx][batch_idx].data_ptr(),
                )
                for batch_idx in range(self.batch_size)
                for page_idx in range(self.num_pages)
            ]
            self._page_table = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        return self._page_table

    @property
    def num_pages(self) -> int:
        return len(self.key_pages)

    @property
    def grad(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.last_update_pages:
            raise RuntimeError("OOMB KV manager has no active update")
        num_pages = self.last_update_pages[-1]
        update_tokens = self.last_update_tokens[-1]
        key_grad = torch.cat(self.key_grad_pages[-num_pages:], dim=1)[:, :update_tokens]
        value_grad = torch.cat(self.value_grad_pages[-num_pages:], dim=1)[:, :update_tokens]
        return key_grad, value_grad

    def remove_last_update(self) -> None:
        if not self.last_update_pages:
            raise RuntimeError("OOMB KV manager has no update to remove")
        update_tokens = self.last_update_tokens.pop()
        update_pages = self.last_update_pages.pop()
        self.num_kv -= update_tokens
        del self.key_pages[-update_pages:]
        del self.value_pages[-update_pages:]
        del self.key_grad_pages[-update_pages:]
        del self.value_grad_pages[-update_pages:]
        self._page_table = None


def _flash_paged_forward(query: torch.Tensor, manager: PagedKVManager, scale: float | None):
    batch, seqlen_q, nheads, head_dim = query.shape
    if query.dtype not in (torch.float16, torch.bfloat16) or not query.is_cuda:
        raise TypeError("OOMB paged attention requires a CUDA FP16/BF16 query")
    if nheads % manager.num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    block = manager.page_size
    seqlen_q_rounded = math.ceil(seqlen_q / block) * block
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=query.device, dtype=torch.float32)
    out = torch.empty_like(query)
    block_head_dim = max(triton.next_power_of_2(head_dim), 16)
    group_size = nheads // manager.num_kv_heads
    grid = (triton.cdiv(seqlen_q, block), batch * nheads)
    num_warps = 4 if head_dim <= 64 else 8
    softmax_scale = 1.0 / math.sqrt(head_dim) if scale is None else scale
    stride_kvb = manager.page_size * manager.num_kv_heads * manager.head_dim
    stride_kvn = manager.num_kv_heads * manager.head_dim
    stride_kvh = manager.head_dim
    _fwd_kernel[grid](
        query,
        manager.page_table,
        out,
        lse,
        softmax_scale,
        query.stride(0),
        query.stride(2),
        query.stride(1),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        stride_kvb,
        stride_kvh,
        stride_kvn,
        nheads,
        seqlen_q,
        manager.num_kv - seqlen_q,
        head_dim,
        seqlen_q_rounded,
        manager.num_kv,
        manager.num_kv_heads,
        manager.num_pages,
        BLOCK_M=block,
        BLOCK_N=block,
        BLOCK_HEADDIM=block_head_dim,
        EVEN_M=seqlen_q % block == 0,
        GROUP_SIZE=group_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return out, lse, softmax_scale


def _flash_paged_backward(
    out: torch.Tensor,
    dout: torch.Tensor,
    query: torch.Tensor,
    dquery: torch.Tensor,
    manager: PagedKVManager,
    lse: torch.Tensor,
    softmax_scale: float,
) -> None:
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen_q, nheads, head_dim = query.shape
    block = manager.page_size
    seqlen_q_rounded = math.ceil(seqlen_q / block) * block
    delta = torch.empty_like(lse)
    block_head_dim = max(triton.next_power_of_2(head_dim), 16)
    group_size = nheads // manager.num_kv_heads
    grid = (triton.cdiv(seqlen_q, block), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        out,
        dout,
        delta,
        out.stride(0),
        out.stride(2),
        out.stride(1),
        dout.stride(0),
        dout.stride(2),
        dout.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        EVEN_M=seqlen_q % block == 0,
        BLOCK_M=block,
        BLOCK_HEADDIM=block_head_dim,
    )
    stride_kvb = manager.page_size * manager.num_kv_heads * manager.head_dim
    stride_kvn = manager.num_kv_heads * manager.head_dim
    stride_kvh = manager.head_dim
    _bwd_kernel[grid](
        query,
        dout,
        dquery,
        manager.page_table,
        lse,
        delta,
        softmax_scale,
        query.stride(0),
        query.stride(2),
        query.stride(1),
        stride_kvb,
        stride_kvh,
        stride_kvn,
        dout.stride(0),
        dout.stride(2),
        dout.stride(1),
        dquery.stride(0),
        dquery.stride(2),
        dquery.stride(1),
        nheads,
        seqlen_q,
        manager.num_kv - seqlen_q,
        head_dim,
        seqlen_q_rounded,
        manager.num_kv,
        manager.num_kv_heads,
        manager.num_pages,
        BLOCK_M=block,
        BLOCK_N=block,
        BLOCK_HEADDIM=block_head_dim,
        EVEN_M=seqlen_q % block == 0,
        GROUP_SIZE=group_size,
        num_warps=4 if head_dim <= 64 else 8,
        num_stages=1,
    )


class _FlashPagedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        manager: PagedKVManager,
        scale: float | None,
    ) -> torch.Tensor:
        query = query if query.stride(-1) == 1 else query.contiguous()
        expected = (manager.batch_size, manager.last_update_tokens[-1], manager.num_kv_heads, manager.head_dim)
        if current_key.shape != expected or current_value.shape != expected:
            raise ValueError(f"current recomputed KV shape does not match the last OOMB update: expected {expected}")
        out, lse, softmax_scale = _flash_paged_forward(query, manager, scale)
        ctx.save_for_backward(query, out, lse)
        ctx.manager = manager
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        query, out, lse = ctx.saved_tensors
        dquery = torch.zeros_like(query)
        _flash_paged_backward(out, dout, query, dquery, ctx.manager, lse, ctx.softmax_scale)
        dkey, dvalue = ctx.manager.grad
        return dquery, dkey, dvalue, None, None


flash_paged_attention = _FlashPagedAttention.apply


class PagedKVBatchView:
    """Batch view over independent, equally advanced trajectory managers."""

    def __init__(self, managers: Sequence[PagedKVManager]) -> None:
        if not managers:
            raise ValueError("OOMB paged batch view requires at least one trajectory")
        if any(manager.batch_size != 1 for manager in managers):
            raise ValueError("wavefront managers must each own one trajectory")
        attributes = ("page_size", "num_kv_heads", "head_dim", "device", "num_kv", "num_pages")
        for attribute in attributes:
            values = {getattr(manager, attribute) for manager in managers}
            if len(values) != 1:
                raise ValueError(f"wavefront managers disagree on {attribute}: {values}")
        update_tokens = {manager.last_update_tokens[-1] for manager in managers}
        if len(update_tokens) != 1:
            raise ValueError(f"wavefront managers disagree on current chunk size: {update_tokens}")

        self.managers = list(managers)
        self.batch_size = len(managers)
        for attribute in attributes:
            setattr(self, attribute, getattr(managers[0], attribute))
        self.last_update_tokens = [next(iter(update_tokens))]
        self._page_table: torch.Tensor | None = None

    @property
    def page_table(self) -> torch.Tensor:
        if self._page_table is None:
            pointers = [
                (
                    manager.key_pages[page_idx][0].data_ptr(),
                    manager.value_pages[page_idx][0].data_ptr(),
                    manager.key_grad_pages[page_idx][0].data_ptr(),
                    manager.value_grad_pages[page_idx][0].data_ptr(),
                )
                for manager in self.managers
                for page_idx in range(self.num_pages)
            ]
            self._page_table = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        return self._page_table

    @property
    def grad(self) -> tuple[torch.Tensor, torch.Tensor]:
        gradients = [manager.grad for manager in self.managers]
        return (
            torch.cat([key_grad for key_grad, _ in gradients], dim=0),
            torch.cat([value_grad for _, value_grad in gradients], dim=0),
        )


class OOMBPagedWavefrontState:
    """Independent trajectory pages combined only for the active reverse depth."""

    def __init__(
        self,
        trajectories: Sequence[Sequence[LayerKVTrace]],
        *,
        chunk_size: int,
        page_size: int,
    ) -> None:
        if not trajectories or not trajectories[0]:
            raise ValueError("wavefront state requires trajectories with KV layers")
        layer_counts = {len(layers) for layers in trajectories}
        if len(layer_counts) != 1:
            raise ValueError("wavefront trajectories must have the same layer count")
        self.num_layers = next(iter(layer_counts))
        self.trajectory_managers: list[list[PagedKVManager]] = []
        self.sequence_lengths: list[int] = []
        for layers in trajectories:
            lengths = {layer.length for layer in layers}
            if len(lengths) != 1:
                raise ValueError("all layers in a wavefront trajectory must have the same length")
            sequence_length = next(iter(lengths))
            if sequence_length % chunk_size:
                raise ValueError("wavefront KV traces must be padded to a full reverse chunk")
            self.sequence_lengths.append(sequence_length)
            self.trajectory_managers.append(
                [PagedKVManager(layer, chunk_size=chunk_size, page_size=page_size) for layer in layers]
            )
            for layer in layers:
                layer.key = layer.key[:, :, :0]
                layer.value = layer.value[:, :, :0]

        self.active: list[int] = []
        self.start = 0
        self.end = 0
        self._visited: set[int] = set()
        self._current: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def begin(self, active: Sequence[int], start: int, end: int) -> None:
        if self._visited:
            raise RuntimeError("the prior OOMB wavefront depth was not committed")
        if not active or start < 0 or start >= end:
            raise ValueError(f"invalid OOMB wavefront depth: active={list(active)}, [{start}, {end})")
        for trajectory_idx in active:
            if not 0 <= trajectory_idx < len(self.trajectory_managers):
                raise ValueError(f"invalid wavefront trajectory index: {trajectory_idx}")
            for manager in self.trajectory_managers[trajectory_idx]:
                if manager.num_kv != end or manager.last_update_tokens[-1] != end - start:
                    raise RuntimeError(
                        "wavefront trajectories must be aligned at the same prefix and reverse chunk size"
                    )
        self.active = list(active)
        self.start, self.end = start, end
        self._current.clear()

    def attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        if layer_idx in self._visited:
            raise RuntimeError(f"layer {layer_idx} was visited twice in one reverse depth")
        if query.shape[0] != len(self.active):
            raise RuntimeError("wavefront query batch does not match the active trajectories")
        self._visited.add(layer_idx)
        manager = PagedKVBatchView(
            [self.trajectory_managers[trajectory_idx][layer_idx] for trajectory_idx in self.active]
        )
        output = flash_paged_attention(
            query.transpose(1, 2).contiguous(),
            current_key.transpose(1, 2).contiguous(),
            current_value.transpose(1, 2).contiguous(),
            manager,
            scale,
        )
        self._current[layer_idx] = (current_key, current_value)
        return output.transpose(1, 2)

    def gradient_injection(self) -> torch.Tensor:
        if len(self._visited) != self.num_layers:
            raise RuntimeError(f"expected {self.num_layers} visited layers, got {len(self._visited)}")
        injection = None
        for layer_idx, (current_key, current_value) in self._current.items():
            for row, trajectory_idx in enumerate(self.active):
                key_grad, value_grad = self.trajectory_managers[trajectory_idx][layer_idx].grad
                key_grad = key_grad.transpose(1, 2)
                value_grad = value_grad.transpose(1, 2)
                term = (current_key[row : row + 1] * key_grad.to(current_key.dtype)).sum()
                term = term + (current_value[row : row + 1] * value_grad.to(current_value.dtype)).sum()
                injection = term if injection is None else injection + term
        if injection is None:
            raise RuntimeError("wavefront reverse state has no current KV tensors")
        return injection

    def commit_prefix_gradients(self, *, release_processed_suffix: bool = True) -> None:
        if len(self._visited) != self.num_layers:
            raise RuntimeError(f"expected {self.num_layers} visited layers, got {len(self._visited)}")
        for trajectory_idx in self.active:
            for manager in self.trajectory_managers[trajectory_idx]:
                manager.remove_last_update()
        del release_processed_suffix
        self.active = []
        self._visited.clear()
        self._current.clear()


class OOMBPagedReverseState:
    """Qwen reverse-attention state backed exclusively by OOMB paged kernels."""

    def __init__(self, layers: Sequence[LayerKVTrace], *, chunk_size: int, page_size: int) -> None:
        if not layers:
            raise ValueError("reverse state requires at least one KV layer")
        lengths = {layer.length for layer in layers}
        if len(lengths) != 1:
            raise ValueError("all KV layers must cover the same token range")
        self.layers = list(layers)
        self._sequence_length = next(iter(lengths))
        self.managers = [PagedKVManager(layer, chunk_size=chunk_size, page_size=page_size) for layer in layers]
        # Managers now own independent pages; drop the assembled handoff tensors
        # so peak memory is bounded by pages plus the active recomputation chunk.
        for layer in self.layers:
            layer.key = layer.key[:, :, :0]
            layer.value = layer.value[:, :, :0]
        self.start = 0
        self.end = 0
        self._visited: set[int] = set()
        self._current: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    def begin(self, start: int, end: int) -> None:
        if self._visited:
            raise RuntimeError("the prior OOMB reverse chunk was not committed")
        if not 0 <= start < end <= self.sequence_length:
            raise ValueError(f"invalid reverse chunk [{start}, {end})")
        for manager in self.managers:
            if manager.num_kv != end or manager.last_update_tokens[-1] != end - start:
                raise RuntimeError("reverse chunks must consume prefix-aligned OOMB updates from suffix to prefix")
        self.start, self.end = start, end

    def attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        if layer_idx in self._visited:
            raise RuntimeError(f"layer {layer_idx} was visited twice in one reverse chunk")
        self._visited.add(layer_idx)
        query_t = query.transpose(1, 2).contiguous()
        key_t = current_key.transpose(1, 2).contiguous()
        value_t = current_value.transpose(1, 2).contiguous()
        out = flash_paged_attention(query_t, key_t, value_t, self.managers[layer_idx], scale)
        self._current[layer_idx] = (current_key, current_value)
        return out.transpose(1, 2)

    def gradient_injection(self) -> torch.Tensor:
        if len(self._visited) != len(self.layers):
            raise RuntimeError(f"expected {len(self.layers)} visited layers, got {len(self._visited)}")
        injection = None
        for layer_idx, (current_key, current_value) in self._current.items():
            key_grad, value_grad = self.managers[layer_idx].grad
            key_grad = key_grad.transpose(1, 2)
            value_grad = value_grad.transpose(1, 2)
            term = (current_key * key_grad.to(current_key.dtype)).sum()
            term = term + (current_value * value_grad.to(current_value.dtype)).sum()
            injection = term if injection is None else injection + term
        if injection is None:
            raise RuntimeError("reverse state has no current KV tensors")
        return injection

    def commit_prefix_gradients(self, *, release_processed_suffix: bool = True) -> None:
        if len(self._visited) != len(self.layers):
            raise RuntimeError(f"expected {len(self.layers)} visited layers, got {len(self._visited)}")
        for manager in self.managers:
            manager.remove_last_update()
        del release_processed_suffix
        self._visited.clear()
        self._current.clear()
