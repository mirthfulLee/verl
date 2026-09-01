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

import ctypes
import ctypes.util
import math
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any

import torch
import triton
import triton.language as tl

from .attention import LayerKVTrace


class _CudaRuntime:
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_DEVICE = 3

    def __init__(self) -> None:
        library = ctypes.util.find_library("cudart") or "libcudart.so"
        self.runtime = ctypes.CDLL(library)
        self.runtime.cudaMemcpyAsync.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        self.runtime.cudaMemcpyAsync.restype = ctypes.c_int
        self.runtime.cudaMemsetAsync.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self.runtime.cudaMemsetAsync.restype = ctypes.c_int
        self.runtime.cudaMemset2DAsync.argtypes = (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self.runtime.cudaMemset2DAsync.restype = ctypes.c_int

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result:
            raise RuntimeError(f"{operation} failed with CUDA runtime error {result}")

    def memcpy_async(self, destination: torch.Tensor, source: torch.Tensor, stream: torch.cuda.Stream) -> None:
        if not destination.is_contiguous() or not source.is_contiguous() or destination.nbytes != source.nbytes:
            raise ValueError("fixed-slot raw CUDA copy requires equal contiguous source and destination views")
        kind = self.MEMCPY_HOST_TO_DEVICE if source.device.type == "cpu" else self.MEMCPY_DEVICE_TO_DEVICE
        self._check(
            self.runtime.cudaMemcpyAsync(
                destination.data_ptr(),
                source.data_ptr(),
                source.nbytes,
                kind,
                stream.cuda_stream,
            ),
            "cudaMemcpyAsync",
        )

    def memset_async(self, destination: torch.Tensor, stream: torch.cuda.Stream) -> None:
        if not destination.is_contiguous():
            raise ValueError("fixed-slot raw CUDA memset requires a contiguous view")
        self._check(
            self.runtime.cudaMemsetAsync(destination.data_ptr(), 0, destination.nbytes, stream.cuda_stream),
            "cudaMemsetAsync",
        )

    def memset_rows_async(
        self,
        tensor: torch.Tensor,
        *,
        rows: int,
        start: int,
        end: int,
        stream: torch.cuda.Stream,
    ) -> None:
        element_bytes = tensor.element_size()
        row_pitch = tensor.shape[1] * tensor.shape[2] * tensor.shape[3] * element_bytes
        width = (end - start) * tensor.shape[2] * tensor.shape[3] * element_bytes
        pointer = tensor[0, start].data_ptr()
        self._check(
            self.runtime.cudaMemset2DAsync(pointer, row_pitch, 0, width, rows, stream.cuda_stream),
            "cudaMemset2DAsync",
        )


_CUDA_RUNTIME: _CudaRuntime | None = None


def _cuda_runtime() -> _CudaRuntime:
    global _CUDA_RUNTIME
    if _CUDA_RUNTIME is None:
        _CUDA_RUNTIME = _CudaRuntime()
    return _CUDA_RUNTIME


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
    page_table_stride_pages,
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
    page_table += off_b * page_table_stride_pages * 4
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
    page_table_stride_pages,
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
    page_table += off_b * page_table_stride_pages * 4
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
        self._page_table_stride_pages = 0
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
        self._page_table_stride_pages = 0

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
            self._page_table_stride_pages = self.num_pages
        return self._page_table

    @property
    def page_table_stride_pages(self) -> int:
        if self._page_table is None:
            _ = self.page_table
        return self._page_table_stride_pages

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
        manager.page_table_stride_pages,
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
        manager.page_table_stride_pages,
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


class _ContiguousKVLayer:
    """Batch-major KV/dKV storage for a wavefront layer."""

    def __init__(self, traces: Sequence[LayerKVTrace]) -> None:
        if not traces:
            raise ValueError("contiguous OOMB storage requires at least one trajectory")
        max_length = max(trace.length for trace in traces)
        keys = []
        values = []
        for trace in traces:
            key = trace.key.transpose(1, 2)
            value = trace.value.transpose(1, 2)
            if not key.is_cuda or key.dtype != torch.bfloat16:
                raise TypeError("contiguous OOMB attention requires CUDA BF16 rollout KV")
            if key.shape != value.shape or key.shape[0] != 1:
                raise ValueError("contiguous OOMB wavefront traces must contain one aligned trajectory")
            pad_tokens = max_length - key.shape[1]
            if pad_tokens:
                padding = (0, 0, 0, 0, 0, pad_tokens)
                key = torch.nn.functional.pad(key, padding)
                value = torch.nn.functional.pad(value, padding)
            keys.append(key)
            values.append(value)
        self.key = torch.cat(keys, dim=0).contiguous()
        self.value = torch.cat(values, dim=0).contiguous()
        self.key_grad = torch.zeros_like(self.key)
        self.value_grad = torch.zeros_like(self.value)
        self.num_kv_heads = self.key.shape[2]
        self.head_dim = self.key.shape[3]

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        token_capacity: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _ContiguousKVLayer:
        instance = cls.__new__(cls)
        shape = (batch_size, token_capacity, num_kv_heads, head_dim)
        instance.key = torch.empty(shape, dtype=dtype, device=device)
        instance.value = torch.empty_like(instance.key)
        instance.key_grad = torch.empty_like(instance.key)
        instance.value_grad = torch.empty_like(instance.key)
        instance.num_kv_heads = num_kv_heads
        instance.head_dim = head_dim
        return instance


class FixedSlotPageState(str, Enum):
    FREE = "free"
    LOADING_NEXT = "loading_next"
    NEXT_READY = "next_ready"
    CURRENT_ACTIVE = "current_active"
    BACKWARD_DONE = "backward_done"


class OOMBFixedSlotPool:
    """Persistent row/page KV backing shared by consecutive reverse groups.

    K/V and dK/dV addresses remain stable for the lifetime of the worker. The
    current group owns prefix pages until their reverse depth commits. A
    dedicated copy stream then overwrites those released pages with the next
    group's host-resident rollout KV.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        token_capacity: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if min(batch_size, token_capacity, num_layers, num_kv_heads, head_dim, page_size) < 1:
            raise ValueError("fixed reverse slot dimensions must be positive")
        if token_capacity % page_size:
            raise ValueError("fixed reverse token capacity must be page aligned")
        self.device = torch.device(device)
        if self.device.type != "cuda" or dtype != torch.bfloat16:
            raise TypeError("fixed reverse slots require CUDA BF16 storage")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.batch_size = batch_size
        self.token_capacity = token_capacity
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.num_pages = token_capacity // page_size
        self.layers = [
            _ContiguousKVLayer.allocate(
                batch_size,
                token_capacity,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=self.device,
            )
            for _ in range(num_layers)
        ]
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamopd-slot-h2d")
        self._pending_enqueues: list[Future] = []
        self.page_states = [[FixedSlotPageState.FREE for _ in range(self.num_pages)] for _ in range(batch_size)]
        self.free_events: list[list[torch.cuda.Event | None]] = [
            [None for _ in range(self.num_pages)] for _ in range(batch_size)
        ]
        self.load_events: list[list[torch.cuda.Event | None]] = [
            [None for _ in range(self.num_pages)] for _ in range(batch_size)
        ]
        self._current_lengths: list[int] = []
        self._next_sources: Sequence[Sequence[Any]] | None = None
        self._next_lengths: list[int] = []
        self._next_padded_lengths: list[int] = []
        self._copy_records: list[tuple[torch.cuda.Event, torch.cuda.Event, bool]] = []
        self._activation_count = 0
        self.initial_wait_seconds = 0.0
        self.next_wait_seconds = 0.0
        self.next_loaded_pages = 0
        self.loaded_bytes = 0
        self.copy_enqueue_seconds = 0.0
        self.next_copy_enqueue_seconds = 0.0

    @property
    def slot_bytes(self) -> int:
        return (
            self.batch_size
            * self.token_capacity
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
            * 4
        )

    @staticmethod
    def _validate_sources(sources: Sequence[Sequence[Any]], lengths: Sequence[int]) -> None:
        if len(sources) != len(lengths):
            raise ValueError("fixed reverse slot sources and lengths must be aligned")

    def reset_metrics(self) -> None:
        if self._current_lengths or self._next_sources is not None or self._pending_enqueues:
            raise RuntimeError("cannot reset fixed reverse slot metrics while a group is active")
        self._copy_records.clear()
        self._activation_count = 0
        self.initial_wait_seconds = 0.0
        self.next_wait_seconds = 0.0
        self.next_loaded_pages = 0
        self.loaded_bytes = 0
        self.copy_enqueue_seconds = 0.0
        self.next_copy_enqueue_seconds = 0.0

    def prepare_next(
        self,
        sources: Sequence[Sequence[Any]],
        lengths: Sequence[int],
        padded_lengths: Sequence[int],
    ) -> None:
        if self._next_sources is not None:
            raise RuntimeError("fixed reverse slot already has a pending next group")
        self._validate_sources(sources, lengths)
        if len(lengths) != len(padded_lengths) or len(lengths) > self.batch_size:
            raise ValueError("fixed reverse next group exceeds its row capacity")
        if any(
            not 0 < length <= padded <= self.token_capacity
            for length, padded in zip(lengths, padded_lengths, strict=True)
        ):
            raise ValueError("fixed reverse group length exceeds the token slot")
        if any(padded % self.page_size for padded in padded_lengths):
            raise ValueError("fixed reverse padded lengths must be page aligned")
        for trajectory in sources:
            if len(trajectory) != self.num_layers:
                raise ValueError("fixed reverse source layer count mismatch")
            for layer in trajectory:
                if layer.length > self.token_capacity or layer.key.ndim not in (3, 4):
                    raise ValueError("fixed reverse source does not fit one slot row")
                if layer.key.shape != layer.value.shape:
                    raise ValueError("fixed reverse source K/V shapes differ")
                if layer.key.ndim == 4 and layer.key.shape[0] != 1:
                    raise ValueError("fixed reverse batch-major source must contain one trajectory")
        self._next_sources = sources
        self._next_lengths = list(lengths)
        self._next_padded_lengths = list(padded_lengths)
        for row, padded_length in enumerate(self._next_padded_lengths):
            self._schedule_free_ranges(row, padded_length)

    def _schedule_free_ranges(self, row: int, padded_length: int) -> None:
        end_page = padded_length // self.page_size
        page = 0
        while page < end_page:
            while page < end_page and self.page_states[row][page] != FixedSlotPageState.FREE:
                page += 1
            start_page = page
            while page < end_page and self.page_states[row][page] == FixedSlotPageState.FREE:
                page += 1
            if start_page < page:
                self._submit_copy_rows([row], start_page * self.page_size, page * self.page_size)

    def _submit_copy_rows(self, rows: Sequence[int], start: int, end: int) -> None:
        self._pending_enqueues.append(self.copy_executor.submit(self._schedule_copy_rows, list(rows), start, end))

    def _schedule_copy_rows(self, rows: Sequence[int], start: int, end: int) -> None:
        if self._next_sources is None or not rows or start >= end:
            return
        rows = [row for row in rows if row < len(self._next_sources) and start < self._next_padded_lengths[row]]
        if not rows:
            return
        torch.cuda.set_device(self.device)
        enqueue_started = time.perf_counter()
        start_page = start // self.page_size
        wait_events = {
            self.free_events[row][page]
            for row in rows
            for page in range(start_page, min(end, self._next_padded_lengths[row]) // self.page_size)
            if self.free_events[row][page] is not None
        }
        for row in rows:
            padded_end = min(end, self._next_padded_lengths[row])
            for page in range(start_page, padded_end // self.page_size):
                if self.page_states[row][page] not in (
                    FixedSlotPageState.FREE,
                    FixedSlotPageState.BACKWARD_DONE,
                ):
                    raise RuntimeError(f"fixed reverse page ({row}, {page}) is not free for next-group loading")
                self.page_states[row][page] = FixedSlotPageState.LOADING_NEXT
        started = torch.cuda.Event(enable_timing=True)
        completed = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.copy_stream):
            runtime = _cuda_runtime()
            for event in wait_events:
                self.copy_stream.wait_event(event)
            started.record(self.copy_stream)
            for layer_idx in range(self.num_layers):
                destination = self.layers[layer_idx]
                for row in rows:
                    padded_end = min(end, self._next_padded_lengths[row])
                    valid_end = min(padded_end, self._next_lengths[row])
                    source = self._next_sources[row][layer_idx]
                    if valid_end < padded_end:
                        runtime.memset_async(destination.key[row, valid_end:padded_end], self.copy_stream)
                        runtime.memset_async(destination.value[row, valid_end:padded_end], self.copy_stream)
                    if start < valid_end:
                        if source.key.ndim == 3:
                            source_key = source.key[start:valid_end]
                            source_value = source.value[start:valid_end]
                            runtime.memcpy_async(destination.key[row, start:valid_end], source_key, self.copy_stream)
                            runtime.memcpy_async(
                                destination.value[row, start:valid_end], source_value, self.copy_stream
                            )
                        else:
                            source_key = source.key[0, :, start:valid_end].transpose(0, 1)
                            source_value = source.value[0, :, start:valid_end].transpose(0, 1)
                            destination.key[row, start:valid_end].copy_(source_key, non_blocking=True)
                            destination.value[row, start:valid_end].copy_(source_value, non_blocking=True)
                if rows == list(range(len(rows))) and all(end <= self._next_padded_lengths[row] for row in rows):
                    runtime.memset_rows_async(
                        destination.key_grad,
                        rows=len(rows),
                        start=start,
                        end=end,
                        stream=self.copy_stream,
                    )
                    runtime.memset_rows_async(
                        destination.value_grad,
                        rows=len(rows),
                        start=start,
                        end=end,
                        stream=self.copy_stream,
                    )
                else:
                    for row in rows:
                        padded_end = min(end, self._next_padded_lengths[row])
                        runtime.memset_async(destination.key_grad[row, start:padded_end], self.copy_stream)
                        runtime.memset_async(destination.value_grad[row, start:padded_end], self.copy_stream)
            completed.record(self.copy_stream)
        is_reuse = self._activation_count > 0
        enqueue_seconds = time.perf_counter() - enqueue_started
        self.copy_enqueue_seconds += enqueue_seconds
        if is_reuse:
            self.next_copy_enqueue_seconds += enqueue_seconds
        self._copy_records.append((started, completed, is_reuse))
        copied_tokens = sum(max(0, min(end, self._next_lengths[row]) - start) for row in rows)
        self.loaded_bytes += (
            copied_tokens
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
            * 2
        )
        if is_reuse:
            self.next_loaded_pages += sum(
                min(end, self._next_padded_lengths[row]) // self.page_size - start_page for row in rows
            )
        for row in rows:
            padded_end = min(end, self._next_padded_lengths[row])
            for page in range(start_page, padded_end // self.page_size):
                self.load_events[row][page] = completed
                self.free_events[row][page] = None

    def activate_next(self) -> list[int]:
        if self._next_sources is None:
            raise RuntimeError("fixed reverse slot has no prepared next group")
        wall_started = time.perf_counter()
        for future in self._pending_enqueues:
            future.result()
        self._pending_enqueues.clear()
        events = {
            self.load_events[row][page]
            for row, padded_length in enumerate(self._next_padded_lengths)
            for page in range(padded_length // self.page_size)
        }
        if None in events:
            raise RuntimeError("fixed reverse next group has pages that were never scheduled")
        for event in events:
            assert event is not None
            event.synchronize()
        wait_seconds = time.perf_counter() - wall_started
        if self._activation_count:
            self.next_wait_seconds += wait_seconds
        else:
            self.initial_wait_seconds += wait_seconds
        for row in range(self.batch_size):
            active_pages = (
                self._next_padded_lengths[row] // self.page_size if row < len(self._next_padded_lengths) else 0
            )
            for page in range(self.num_pages):
                if page < active_pages:
                    self.page_states[row][page] = FixedSlotPageState.NEXT_READY
                    self.page_states[row][page] = FixedSlotPageState.CURRENT_ACTIVE
                else:
                    self.page_states[row][page] = FixedSlotPageState.FREE
                self.load_events[row][page] = None
                self.free_events[row][page] = None
        self._current_lengths = list(self._next_padded_lengths)
        self._next_sources = None
        self._next_lengths = []
        self._next_padded_lengths = []
        self._activation_count += 1
        return list(self._current_lengths)

    def release_current_range(self, active: Sequence[int], start: int, end: int) -> None:
        if start % self.page_size or end % self.page_size:
            raise ValueError("fixed reverse release ranges must be page aligned")
        free_event = torch.cuda.Event(enable_timing=False)
        free_event.record(torch.cuda.current_stream(self.device))
        for row in active:
            for page in range(start // self.page_size, end // self.page_size):
                if self.page_states[row][page] != FixedSlotPageState.CURRENT_ACTIVE:
                    raise RuntimeError(f"fixed reverse page ({row}, {page}) is not current-active")
                self.page_states[row][page] = FixedSlotPageState.BACKWARD_DONE
                self.free_events[row][page] = free_event
                self.page_states[row][page] = FixedSlotPageState.FREE
        if self._next_sources is not None:
            self._submit_copy_rows(active, start, end)

    def finish_current(self) -> None:
        for row, padded_length in enumerate(self._current_lengths):
            for page in range(padded_length // self.page_size):
                if self.page_states[row][page] == FixedSlotPageState.CURRENT_ACTIVE:
                    raise RuntimeError("fixed reverse group finished before every current page was released")
        self._current_lengths = []

    def state(self) -> OOMBFlashWavefrontState:
        if not self._current_lengths:
            raise RuntimeError("fixed reverse slot has no active group")
        return OOMBFlashWavefrontState.from_fixed_slot(self.layers, self._current_lengths)

    def copy_cuda_seconds(self, *, reused_only: bool = False) -> float:
        total = 0.0
        for started, completed, is_reuse in self._copy_records:
            if reused_only and not is_reuse:
                continue
            completed.synchronize()
            total += started.elapsed_time(completed) / 1000.0
        return total


class _ContiguousKVBatchView:
    """Active wavefront view with one batched gradient accumulation kernel."""

    def __init__(self, layer: _ContiguousKVLayer, active: Sequence[int], start: int, end: int) -> None:
        self.layer = layer
        self.active = tuple(active)
        self.start = start
        self.end = end
        self.batch_size = len(active)
        self.num_kv_heads = layer.num_kv_heads
        self.head_dim = layer.head_dim
        self.group_size = 0
        self._active_tensor: torch.Tensor | None = None
        self._prefix_active = self.active == tuple(range(self.batch_size))

    def _select(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._prefix_active:
            return tensor[: self.batch_size, : self.end]
        if self._active_tensor is None:
            self._active_tensor = torch.tensor(self.active, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, self._active_tensor)[:, : self.end]

    def expanded_key_value(self, query_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        if query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self.group_size = query_heads // self.num_kv_heads
        key = self._select(self.layer.key).transpose(1, 2)
        value = self._select(self.layer.value).transpose(1, 2)
        return key.repeat_interleave(self.group_size, dim=1), value.repeat_interleave(self.group_size, dim=1)

    def accumulate_expanded_gradients(self, key_grad: torch.Tensor, value_grad: torch.Tensor) -> None:
        batch, _, tokens, head_dim = key_grad.shape
        expected_heads = self.num_kv_heads * self.group_size
        if batch != self.batch_size or key_grad.shape[1] != expected_heads or head_dim != self.head_dim:
            raise RuntimeError("FlashAttention returned an invalid contiguous OOMB gradient shape")
        key_grad = key_grad.view(batch, self.num_kv_heads, self.group_size, tokens, head_dim).sum(2)
        value_grad = value_grad.view(batch, self.num_kv_heads, self.group_size, tokens, head_dim).sum(2)
        key_grad = key_grad.transpose(1, 2)
        value_grad = value_grad.transpose(1, 2)
        if self._prefix_active:
            self.layer.key_grad[:batch, :tokens].add_(key_grad)
            self.layer.value_grad[:batch, :tokens].add_(value_grad)
        else:
            assert self._active_tensor is not None
            self.layer.key_grad[:, :tokens].index_add_(0, self._active_tensor, key_grad)
            self.layer.value_grad[:, :tokens].index_add_(0, self._active_tensor, value_grad)

    @property
    def grad(self) -> tuple[torch.Tensor, torch.Tensor]:
        key_grad = self._select(self.layer.key_grad)[:, self.start : self.end]
        value_grad = self._select(self.layer.value_grad)[:, self.start : self.end]
        return key_grad, value_grad


class _FlashContiguousAttention(torch.autograd.Function):
    """Fail-closed CUDA FlashAttention VJP over OOMB's persistent KV/dKV."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        manager: _ContiguousKVBatchView,
        scale: float | None,
    ) -> torch.Tensor:
        if query.dtype != torch.bfloat16 or not query.is_cuda or query.shape[-1] > 256:
            raise TypeError("contiguous OOMB FlashAttention requires CUDA BF16 with head_dim <= 256")
        expected = (manager.batch_size, manager.end - manager.start, manager.num_kv_heads, manager.head_dim)
        if current_key.shape != expected or current_value.shape != expected:
            raise ValueError(f"current recomputed KV shape does not match the OOMB wavefront: expected {expected}")
        query = query.transpose(1, 2).contiguous()
        key, value = manager.expanded_key_value(query.shape[1])
        softmax_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
        try:
            result = torch.ops.aten._scaled_dot_product_flash_attention(
                query,
                key,
                value,
                0.0,
                True,
                False,
                scale=softmax_scale,
            )
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError("the selected CUDA device does not support exact contiguous FlashAttention") from exc
        output, lse, cum_q, cum_k, max_q, max_k, rng_state = result[:7]
        ctx.save_for_backward(query, output, lse, rng_state)
        ctx.manager = manager
        ctx.cum_q = cum_q
        ctx.cum_k = cum_k
        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.softmax_scale = softmax_scale
        return output.transpose(1, 2)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        query, output, lse, rng_state = ctx.saved_tensors
        key, value = ctx.manager.expanded_key_value(query.shape[1])
        query_grad, key_grad, value_grad = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            output_grad.transpose(1, 2).contiguous(),
            query,
            key,
            value,
            output,
            lse,
            ctx.cum_q,
            ctx.cum_k,
            ctx.max_q,
            ctx.max_k,
            0.0,
            True,
            rng_state[0],
            rng_state[1],
            scale=ctx.softmax_scale,
        )
        ctx.manager.accumulate_expanded_gradients(key_grad, value_grad)
        current_key_grad, current_value_grad = ctx.manager.grad
        return query_grad.transpose(1, 2), current_key_grad, current_value_grad, None, None


flash_contiguous_attention = _FlashContiguousAttention.apply


class OOMBFlashWavefrontState:
    """Wavefront reverse state backed by batched contiguous FlashAttention."""

    def __init__(self, trajectories: Sequence[Sequence[LayerKVTrace]]) -> None:
        if not trajectories or not trajectories[0]:
            raise ValueError("FlashAttention wavefront state requires trajectories with KV layers")
        layer_counts = {len(layers) for layers in trajectories}
        if len(layer_counts) != 1:
            raise ValueError("wavefront trajectories must have the same layer count")
        self.num_layers = next(iter(layer_counts))
        self.sequence_lengths = []
        for layers in trajectories:
            lengths = {layer.length for layer in layers}
            if len(lengths) != 1:
                raise ValueError("all layers in a wavefront trajectory must have the same length")
            self.sequence_lengths.append(next(iter(lengths)))
        self.layers = []
        for layer_idx in range(self.num_layers):
            self.layers.append(_ContiguousKVLayer([trajectory[layer_idx] for trajectory in trajectories]))
            for trajectory in trajectories:
                trajectory[layer_idx].key = trajectory[layer_idx].key[:, :, :0]
                trajectory[layer_idx].value = trajectory[layer_idx].value[:, :, :0]
        self.active: list[int] = []
        self.start = 0
        self.end = 0
        self._visited: set[int] = set()
        self._current: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def from_fixed_slot(
        cls,
        layers: Sequence[_ContiguousKVLayer],
        sequence_lengths: Sequence[int],
    ) -> OOMBFlashWavefrontState:
        if not layers or not sequence_lengths:
            raise ValueError("fixed-slot wavefront state requires layers and sequence lengths")
        if any(length < 1 or length > layers[0].key.shape[1] for length in sequence_lengths):
            raise ValueError("fixed-slot wavefront sequence length is outside the slot capacity")
        if len(sequence_lengths) > layers[0].key.shape[0]:
            raise ValueError("fixed-slot wavefront group exceeds the row capacity")
        state = cls.__new__(cls)
        state.num_layers = len(layers)
        state.sequence_lengths = list(sequence_lengths)
        state.layers = list(layers)
        state.active = []
        state.start = 0
        state.end = 0
        state._visited = set()
        state._current = {}
        return state

    def begin(self, active: Sequence[int], start: int, end: int) -> None:
        if self._visited:
            raise RuntimeError("the prior FlashAttention wavefront depth was not committed")
        if not active or start < 0 or start >= end:
            raise ValueError(f"invalid FlashAttention wavefront depth: active={list(active)}, [{start}, {end})")
        if any(end > self.sequence_lengths[idx] for idx in active):
            raise RuntimeError("active FlashAttention wavefront trajectory is shorter than the reverse depth")
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
        manager = _ContiguousKVBatchView(self.layers[layer_idx], self.active, self.start, self.end)
        output = flash_contiguous_attention(
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
        current_key, _ = next(iter(self._current.values()))
        return torch.zeros((), device=current_key.device, dtype=current_key.dtype)

    def commit_prefix_gradients(self, *, release_processed_suffix: bool = True) -> None:
        if len(self._visited) != self.num_layers:
            raise RuntimeError(f"expected {self.num_layers} visited layers, got {len(self._visited)}")
        del release_processed_suffix
        self.active = []
        self._visited.clear()
        self._current.clear()


class PagedKVBatchView:
    """Batch view over independent, equally advanced trajectory managers."""

    def __init__(
        self,
        managers: Sequence[PagedKVManager],
        *,
        page_table: torch.Tensor | None = None,
        page_table_stride_pages: int = 0,
    ) -> None:
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
        if page_table is not None:
            if page_table.device != self.device or page_table.dtype != torch.int64:
                raise ValueError("cached OOMB page table has the wrong device or dtype")
            if page_table_stride_pages < self.num_pages:
                raise ValueError("cached OOMB page table is shorter than the active prefix")
        self._page_table = page_table
        self._page_table_stride_pages = page_table_stride_pages

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
            self._page_table_stride_pages = self.num_pages
        return self._page_table

    @property
    def page_table_stride_pages(self) -> int:
        if self._page_table is None:
            _ = self.page_table
        return self._page_table_stride_pages

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
        self._page_tables: dict[tuple[tuple[int, ...], int], tuple[torch.Tensor, int]] = {}

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
        cache_key = (tuple(self.active), layer_idx)
        cached_table = self._page_tables.get(cache_key)
        manager = PagedKVBatchView(
            [self.trajectory_managers[trajectory_idx][layer_idx] for trajectory_idx in self.active],
            page_table=cached_table[0] if cached_table is not None else None,
            page_table_stride_pages=cached_table[1] if cached_table is not None else 0,
        )
        if cached_table is None:
            self._page_tables[cache_key] = (manager.page_table, manager.page_table_stride_pages)
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
        current_key, _ = next(iter(self._current.values()))
        return torch.zeros((), device=current_key.device, dtype=current_key.dtype)

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
        current_key, _ = next(iter(self._current.values()))
        return torch.zeros((), device=current_key.device, dtype=current_key.dtype)

    def commit_prefix_gradients(self, *, release_processed_suffix: bool = True) -> None:
        if len(self._visited) != len(self.layers):
            raise RuntimeError(f"expected {len(self.layers)} visited layers, got {len(self._visited)}")
        for manager in self.managers:
            manager.remove_last_update()
        del release_processed_suffix
        self._visited.clear()
        self._current.clear()
