# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fixed-slot FlashAttention used by StreamOPD reverse training.

The reverse dataflow follows OOMB, while storage is specialized to the single
production path: persistent dense BF16 CUDA slots with wavefront batching.
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

from verl.utils.device import get_device_id, get_device_name, get_torch_device, get_vendor


class _CudaRuntime:
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_DEVICE = 3

    def __init__(self) -> None:
        library = ctypes.util.find_library("cudart") or "libcudart.so"
        self.runtime = ctypes.CDLL(library)
        self._memcpy = getattr(self.runtime, "cu" + "daMemcpyAsync")
        self._memcpy.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        self._memcpy.restype = ctypes.c_int
        self._memset = getattr(self.runtime, "cu" + "daMemsetAsync")
        self._memset.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self._memset.restype = ctypes.c_int
        self._memset_2d = getattr(self.runtime, "cu" + "daMemset2DAsync")
        self._memset_2d.argtypes = (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self._memset_2d.restype = ctypes.c_int

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result:
            raise RuntimeError(f"{operation} failed with CUDA runtime error {result}")

    @staticmethod
    def _stream_ptr(stream: Any) -> int:
        return getattr(stream, f"{get_device_name()}_stream")

    def memcpy_async(self, destination: torch.Tensor, source: torch.Tensor, stream: Any) -> None:
        if not destination.is_contiguous() or not source.is_contiguous() or destination.nbytes != source.nbytes:
            raise ValueError("fixed-slot raw CUDA copy requires equal contiguous source and destination views")
        kind = self.MEMCPY_HOST_TO_DEVICE if source.device.type == "cpu" else self.MEMCPY_DEVICE_TO_DEVICE
        self._check(
            self._memcpy(
                destination.data_ptr(),
                source.data_ptr(),
                source.nbytes,
                kind,
                self._stream_ptr(stream),
            ),
            "cudaMemcpyAsync",
        )

    def memset_async(self, destination: torch.Tensor, stream: Any) -> None:
        if not destination.is_contiguous():
            raise ValueError("fixed-slot raw CUDA memset requires a contiguous view")
        self._check(
            self._memset(destination.data_ptr(), 0, destination.nbytes, self._stream_ptr(stream)),
            "cudaMemsetAsync",
        )

    def memset_rows_async(
        self,
        tensor: torch.Tensor,
        *,
        rows: int,
        start: int,
        end: int,
        stream: Any,
    ) -> None:
        element_bytes = tensor.element_size()
        row_pitch = tensor.shape[1] * tensor.shape[2] * tensor.shape[3] * element_bytes
        width = (end - start) * tensor.shape[2] * tensor.shape[3] * element_bytes
        pointer = tensor[0, start].data_ptr()
        self._check(
            self._memset_2d(pointer, row_pitch, 0, width, rows, self._stream_ptr(stream)),
            "cudaMemset2DAsync",
        )


_CUDA_RUNTIME: _CudaRuntime | None = None


def _cuda_runtime() -> _CudaRuntime:
    global _CUDA_RUNTIME
    if _CUDA_RUNTIME is None:
        _CUDA_RUNTIME = _CudaRuntime()
    return _CUDA_RUNTIME


class _ContiguousKVLayer:
    """Batch-major KV/dKV storage for a wavefront layer."""

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


class _ContiguousKVBuffer:
    """Optional inactive K/V backing for one-chunk reverse prefetch."""

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
    ) -> _ContiguousKVBuffer:
        instance = cls.__new__(cls)
        shape = (batch_size, token_capacity, num_kv_heads, head_dim)
        instance.key = torch.empty(shape, dtype=dtype, device=device)
        instance.value = torch.empty_like(instance.key)
        return instance


class _PinnedKVLayer:
    """One layer of persistent Host staging for a single reverse group."""

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        token_capacity: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype,
    ) -> _PinnedKVLayer:
        instance = cls.__new__(cls)
        shape = (batch_size, token_capacity, num_kv_heads, head_dim)
        instance.key = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        instance.value = torch.empty_like(instance.key, pin_memory=True)
        return instance


class _PinnedKVSource:
    """Token-major view into one row of persistent pinned staging."""

    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key = key
        self.value = value

    @property
    def length(self) -> int:
        return self.key.shape[0]


class FixedSlotPageState(str, Enum):
    FREE = "free"
    LOADING_NEXT = "loading_next"
    NEXT_READY = "next_ready"
    CURRENT_ACTIVE = "current_active"
    BACKWARD_DONE = "backward_done"


class ReverseKVSlotPool:
    """Preallocated reverse KV with page reuse or full one-chunk prefetch."""

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
        prefetch_kv: bool = False,
        pinned_layers: list[_PinnedKVLayer] | None = None,
    ) -> None:
        if min(batch_size, token_capacity, num_layers, num_kv_heads, head_dim, page_size) < 1:
            raise ValueError("fixed reverse slot dimensions must be positive")
        if token_capacity % page_size:
            raise ValueError("fixed reverse token capacity must be page aligned")
        self.device = torch.device(device)
        if get_vendor() != "nvidia" or self.device.type != get_device_name() or dtype != torch.bfloat16:
            raise TypeError("fixed reverse slots require CUDA BF16 storage")
        if self.device.index is None:
            self.device = torch.device(get_device_name(), get_device_id())
        self.batch_size = batch_size
        self.token_capacity = token_capacity
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.prefetch_kv = bool(prefetch_kv)
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
        self._inactive_layers = (
            [
                _ContiguousKVBuffer.allocate(
                    batch_size,
                    token_capacity,
                    num_kv_heads,
                    head_dim,
                    dtype=dtype,
                    device=self.device,
                )
                for _ in range(num_layers)
            ]
            if self.prefetch_kv
            else None
        )
        device_module = get_torch_device()
        self.copy_stream = device_module.Stream(device=self.device)
        self.copy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamopd-slot-h2d")
        self._pending_enqueues: list[Future] = []
        self.page_states = [[FixedSlotPageState.FREE for _ in range(self.num_pages)] for _ in range(batch_size)]
        self.free_events: list[list[Any | None]] = [[None for _ in range(self.num_pages)] for _ in range(batch_size)]
        self.load_events: list[list[Any | None]] = [[None for _ in range(self.num_pages)] for _ in range(batch_size)]
        self._current_lengths: list[int] = []
        self._next_sources: Sequence[Sequence[Any]] | None = None
        self._next_lengths: list[int] = []
        self._next_padded_lengths: list[int] = []
        self._pinned_layers = pinned_layers
        if pinned_layers is not None:
            expected_shape = (batch_size, token_capacity, num_kv_heads, head_dim)
            if len(pinned_layers) != num_layers or any(
                layer.key.shape != expected_shape
                or layer.value.shape != expected_shape
                or layer.key.dtype != dtype
                or layer.value.dtype != dtype
                or not layer.key.is_pinned()
                or not layer.value.is_pinned()
                for layer in pinned_layers
            ):
                raise ValueError("reused pinned reverse staging does not match the fixed slot plan")
        self._copy_records: list[tuple[Any, Any, bool]] = []
        self._activation_count = 0
        self.initial_wait_seconds = 0.0
        self.next_wait_seconds = 0.0
        self.next_loaded_pages = 0
        self.loaded_bytes = 0
        self.copy_enqueue_seconds = 0.0
        self.next_copy_enqueue_seconds = 0.0
        self.pinned_staging_seconds = 0.0
        self.pinned_staging_allocation_seconds = 0.0
        self.pinned_staging_bytes = 0
        self.pinned_staging_groups = 0

    @property
    def kv_pair_bytes(self) -> int:
        return (
            self.batch_size
            * self.token_capacity
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
            * 2
        )

    @property
    def slot_bytes(self) -> int:
        return self.kv_pair_bytes * (3 if self.prefetch_kv else 2)

    @property
    def pinned_staging_capacity_bytes(self) -> int:
        if self._pinned_layers is None:
            return 0
        return self.kv_pair_bytes

    def detach_pinned_layers(self) -> list[_PinnedKVLayer] | None:
        """Keep Host staging alive while phase-shared GPU slots are released."""

        if self._current_lengths or self._next_sources is not None or self._pending_enqueues:
            raise RuntimeError("cannot detach pinned reverse staging while a group is active")
        pinned_layers = self._pinned_layers
        self._pinned_layers = None
        return pinned_layers

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
        self.pinned_staging_seconds = 0.0
        self.pinned_staging_allocation_seconds = 0.0
        self.pinned_staging_bytes = 0
        self.pinned_staging_groups = 0

    def _requires_pinned_staging(self, sources: Sequence[Sequence[Any]]) -> bool:
        return any(
            not layer.key.is_pinned() or not layer.value.is_pinned() for trajectory in sources for layer in trajectory
        )

    def _ensure_pinned_layers(self) -> list[_PinnedKVLayer]:
        if self._pinned_layers is None:
            started = time.perf_counter()
            self._pinned_layers = [
                _PinnedKVLayer.allocate(
                    self.batch_size,
                    self.token_capacity,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ]
            self.pinned_staging_allocation_seconds += time.perf_counter() - started
        return self._pinned_layers

    def _stage_pageable_sources(
        self,
        sources: Sequence[Sequence[Any]],
        lengths: Sequence[int],
    ) -> None:
        """Materialize one group in reusable pinned storage before H2D."""

        pinned_layers = self._ensure_pinned_layers()
        started = time.perf_counter()
        staged_sources: list[list[_PinnedKVSource]] = [[] for _ in sources]
        for layer_idx, destination in enumerate(pinned_layers):
            for row, (trajectory, length) in enumerate(zip(sources, lengths, strict=True)):
                source = trajectory[layer_idx]
                if source.key.ndim == 3:
                    source_key = source.key[:length]
                    source_value = source.value[:length]
                else:
                    source_key = source.key[0, :, :length].transpose(0, 1)
                    source_value = source.value[0, :, :length].transpose(0, 1)
                destination.key[row, :length].copy_(source_key)
                destination.value[row, :length].copy_(source_value)
                staged_sources[row].append(
                    _PinnedKVSource(
                        destination.key[row, :length],
                        destination.value[row, :length],
                    )
                )
        self._next_sources = staged_sources
        self.pinned_staging_seconds += time.perf_counter() - started
        self.pinned_staging_bytes += (
            sum(lengths)
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
            * 2
        )
        self.pinned_staging_groups += 1

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
        for trajectory, length in zip(sources, lengths, strict=True):
            for layer in trajectory:
                if layer.length > self.token_capacity or layer.key.ndim not in (3, 4):
                    raise ValueError("fixed reverse source does not fit one slot row")
                if layer.length < length:
                    raise ValueError("fixed reverse source is shorter than its trajectory")
                if layer.key.shape != layer.value.shape:
                    raise ValueError("fixed reverse source K/V shapes differ")
                if layer.key.ndim == 4 and layer.key.shape[0] != 1:
                    raise ValueError("fixed reverse batch-major source must contain one trajectory")
                if layer.key.device.type != "cpu" or layer.value.device.type != "cpu":
                    raise ValueError("fixed reverse source K/V must reside in Host memory")
                if layer.key.dtype != self.dtype or layer.value.dtype != self.dtype:
                    raise ValueError("fixed reverse source K/V dtype does not match its slot")
                if layer.key.shape[-2:] != (self.num_kv_heads, self.head_dim) and not (
                    layer.key.ndim == 4
                    and layer.key.shape[1] == self.num_kv_heads
                    and layer.key.shape[-1] == self.head_dim
                ):
                    raise ValueError("fixed reverse source K/V head shape does not match its slot")
        self._next_sources = sources
        self._next_lengths = list(lengths)
        self._next_padded_lengths = list(padded_lengths)
        if self._requires_pinned_staging(sources):
            self._pending_enqueues.append(
                self.copy_executor.submit(self._stage_pageable_sources, sources, list(lengths))
            )
        if self.prefetch_kv:
            self._submit_copy_rows(
                list(range(len(self._next_padded_lengths))),
                0,
                max(self._next_padded_lengths),
            )
        else:
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
        device_module = get_torch_device()
        device_module.set_device(self.device)
        enqueue_started = time.perf_counter()
        start_page = start // self.page_size
        wait_events = set()
        if not self.prefetch_kv:
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
        started = device_module.Event(enable_timing=True)
        completed = device_module.Event(enable_timing=True)
        with device_module.stream(self.copy_stream):
            runtime = _cuda_runtime()
            for event in wait_events:
                self.copy_stream.wait_event(event)
            started.record(self.copy_stream)
            for layer_idx in range(self.num_layers):
                destination = (
                    self._inactive_layers[layer_idx] if self._inactive_layers is not None else self.layers[layer_idx]
                )
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
                if self.prefetch_kv:
                    continue
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
        if self._inactive_layers is not None:
            for layer, inactive in zip(self.layers, self._inactive_layers, strict=True):
                layer.key, inactive.key = inactive.key, layer.key
                layer.value, inactive.value = inactive.value, layer.value
            current_stream = get_torch_device().current_stream(self.device)
            runtime = _cuda_runtime()
            for layer in self.layers:
                runtime.memset_async(layer.key_grad, current_stream)
                runtime.memset_async(layer.value_grad, current_stream)
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
        free_event = None
        if not self.prefetch_kv:
            device_module = get_torch_device()
            free_event = device_module.Event(enable_timing=False)
            free_event.record(device_module.current_stream(self.device))
        for row in active:
            for page in range(start // self.page_size, end // self.page_size):
                if self.page_states[row][page] != FixedSlotPageState.CURRENT_ACTIVE:
                    raise RuntimeError(f"fixed reverse page ({row}, {page}) is not current-active")
                self.page_states[row][page] = FixedSlotPageState.BACKWARD_DONE
                self.free_events[row][page] = free_event
                self.page_states[row][page] = FixedSlotPageState.FREE
        if not self.prefetch_kv and self._next_sources is not None:
            self._submit_copy_rows(active, start, end)

    def finish_current(self) -> None:
        for row, padded_length in enumerate(self._current_lengths):
            for page in range(padded_length // self.page_size):
                if self.page_states[row][page] == FixedSlotPageState.CURRENT_ACTIVE:
                    raise RuntimeError("fixed reverse group finished before every current page was released")
        self._current_lengths = []

    def abort_groups(self) -> None:
        """Drop group metadata after a failed backward before pool teardown."""

        for future in self._pending_enqueues:
            future.cancel()
            try:
                future.result()
            except BaseException:
                pass
        self._pending_enqueues.clear()
        self._current_lengths = []
        self._next_sources = None
        self._next_lengths = []
        self._next_padded_lengths = []
        for row in range(self.batch_size):
            for page in range(self.num_pages):
                self.page_states[row][page] = FixedSlotPageState.FREE
                self.load_events[row][page] = None
                self.free_events[row][page] = None

    def state(self) -> ReverseWavefrontState:
        if not self._current_lengths:
            raise RuntimeError("fixed reverse slot has no active group")
        return ReverseWavefrontState.from_fixed_slot(self.layers, self._current_lengths)

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
        self._active_tensor: torch.Tensor | None = None
        self._prefix_active = self.active == tuple(range(self.batch_size))

    def _select(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._prefix_active:
            return tensor[: self.batch_size, : self.end]
        if self._active_tensor is None:
            self._active_tensor = torch.tensor(self.active, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, self._active_tensor)[:, : self.end]

    def key_value(self, query_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        if query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        # CUDA FlashAttention kernels natively support GQA. Keep the compact
        # KV-head dimension instead of materializing query-head copies for
        # every layer and reverse depth.
        return self._select(self.layer.key), self._select(self.layer.value)

    def accumulate_gradients(self, key_grad: torch.Tensor, value_grad: torch.Tensor) -> None:
        batch, tokens, heads, head_dim = key_grad.shape
        if batch != self.batch_size or heads != self.num_kv_heads or head_dim != self.head_dim:
            raise RuntimeError("FlashAttention returned an invalid contiguous OOMB gradient shape")
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
        key, value = manager.key_value(query.shape[2])
        softmax_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
        try:
            from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward

            output, lse, _, rng_state = _wrapped_flash_attn_forward(
                query,
                key,
                value,
                0.0,
                softmax_scale,
                causal=True,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
            ctx.backend = "flash_attn"
            ctx.save_for_backward(query, output, lse, rng_state)
        except ImportError:
            query_heads_first = query.transpose(1, 2).contiguous()
            key_heads_first = key.transpose(1, 2)
            value_heads_first = value.transpose(1, 2)
            result = torch.ops.aten._scaled_dot_product_flash_attention(
                query_heads_first,
                key_heads_first,
                value_heads_first,
                0.0,
                True,
                False,
                scale=softmax_scale,
            )
            output, lse, cum_q, cum_k, max_q, max_k, rng_state = result[:7]
            ctx.backend = "aten"
            ctx.cum_q = cum_q
            ctx.cum_k = cum_k
            ctx.max_q = max_q
            ctx.max_k = max_k
            ctx.save_for_backward(query_heads_first, output, lse, rng_state)
        ctx.manager = manager
        ctx.softmax_scale = softmax_scale
        return output if ctx.backend == "flash_attn" else output.transpose(1, 2)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        query, output, lse, rng_state = ctx.saved_tensors
        if ctx.backend == "flash_attn":
            from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward

            key, value = ctx.manager.key_value(query.shape[2])
            query_grad = torch.empty_like(query)
            key_grad = torch.empty_like(key)
            value_grad = torch.empty_like(value)
            _wrapped_flash_attn_backward(
                output_grad.contiguous(),
                query,
                key,
                value,
                output,
                lse,
                query_grad,
                key_grad,
                value_grad,
                0.0,
                ctx.softmax_scale,
                True,
                -1,
                -1,
                0.0,
                None,
                False,
                rng_state=rng_state,
            )
        else:
            key, value = ctx.manager.key_value(query.shape[1])
            query_grad, key_grad, value_grad = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                output_grad.transpose(1, 2).contiguous(),
                query,
                key.transpose(1, 2),
                value.transpose(1, 2),
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
            query_grad = query_grad.transpose(1, 2)
            key_grad = key_grad.transpose(1, 2)
            value_grad = value_grad.transpose(1, 2)
        ctx.manager.accumulate_gradients(key_grad, value_grad)
        current_key_grad, current_value_grad = ctx.manager.grad
        return query_grad, current_key_grad, current_value_grad, None, None


flash_contiguous_attention = _FlashContiguousAttention.apply


class ReverseWavefrontState:
    """Wavefront reverse state backed by batched contiguous FlashAttention."""

    @classmethod
    def from_fixed_slot(
        cls,
        layers: Sequence[_ContiguousKVLayer],
        sequence_lengths: Sequence[int],
    ) -> ReverseWavefrontState:
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
        return output.transpose(1, 2)

    def validate_complete(self) -> None:
        if len(self._visited) != self.num_layers:
            raise RuntimeError(f"expected {self.num_layers} visited layers, got {len(self._visited)}")

    def commit_prefix_gradients(self) -> None:
        if len(self._visited) != self.num_layers:
            raise RuntimeError(f"expected {self.num_layers} visited layers, got {len(self._visited)}")
        self.active = []
        self._visited.clear()
