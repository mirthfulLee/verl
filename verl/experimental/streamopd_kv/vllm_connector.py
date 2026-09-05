# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""vLLM store-only connector for rollout KV snapshots."""

from __future__ import annotations

import queue
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

try:
    from cuda.bindings import runtime as cuda_runtime
except ImportError:  # pragma: no cover - CUDA vLLM environments provide cuda-python.
    cuda_runtime = None

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

try:
    # vLLM >= 0.24
    from vllm.v1.attention.backend import AttentionMetadata
except ImportError:
    # vLLM 0.11-0.23
    from vllm.attention.backends.abstract import AttentionMetadata

from verl.utils.device import get_torch_device

from .host_slot_pool import HostKVSlotPool
from .protocol import TRANSFER_MAX_KEYS, TRANSFER_SUM_KEYS
from .snapshot_io import extract_vllm_nhd_token_range, gather_vllm_cross_layers_nhd_into

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_EXPORT_STRATEGIES = {"eos_host", "eos_triton", "incremental_triton"}


def _empty_transfer_stats() -> dict[str, float]:
    return {key: 0.0 for key in (*TRANSFER_SUM_KEYS, *TRANSFER_MAX_KEYS)}


def _layer_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return (int(match.group(1)), name) if match else (2**31 - 1, name)


@dataclass
class _PendingSave:
    req_id: str
    trajectory_id: str
    slot_path: str
    block_ids_by_group: tuple[list[int], ...]
    policy_version: int
    prompt_length: int
    start: int
    end: int
    chunk_index: int
    terminal: bool = False
    token_ids: torch.Tensor | None = None
    streamed_tokens_before_eos: int = 0
    streamed_chunks_before_eos: int = 0


@dataclass
class _SchedulerSaveState:
    req_id: str
    trajectory_id: str
    slot_path: str
    block_ids_by_group: list[list[int]]
    policy_version: int
    prompt_length: int
    published_tokens: int = 0
    next_chunk_index: int = 0
    chunks: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class StreamOPDKVConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[_PendingSave] = field(default_factory=list)
    has_model_work: bool = True
    preempted_req_ids: set[str] = field(default_factory=set)


class StreamOPDKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Seal post-RoPE K/V pages before vLLM returns them to its allocator."""

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "streamopd_kv_handoff_dir", "/tmp/verl-streamopd-kv"
        )
        self._chunk_size = int(self._kv_transfer_config.get_from_extra_config("streamopd_kv_chunk_size", 256))
        if self._chunk_size < 1:
            raise ValueError("streamopd_kv_chunk_size must be positive")
        self._export_strategy = str(
            self._kv_transfer_config.get_from_extra_config("streamopd_kv_export_strategy", "eos_host")
        )
        if self._export_strategy not in _EXPORT_STRATEGIES:
            raise ValueError(
                "streamopd_kv_export_strategy must be one of "
                f"{sorted(_EXPORT_STRATEGIES)}, got {self._export_strategy!r}"
            )
        self._host_slot_count = int(self._kv_transfer_config.get_from_extra_config("streamopd_host_slot_count", 0))
        self._host_slot_tokens = int(self._kv_transfer_config.get_from_extra_config("streamopd_host_slot_tokens", 0))
        if self._host_slot_count < 1 or self._host_slot_tokens < 1:
            raise ValueError("StreamOPD requires a positive shared Host KV slot count and token capacity")
        self._scheduler_paths: dict[str, str] = {}
        self._scheduler_states: dict[str, _SchedulerSaveState] = {}
        self._pending: list[_PendingSave] = []

        self._kv_caches: dict[str, torch.Tensor] = {}
        self._cross_layers_kv_cache: torch.Tensor | None = None
        self._cross_layer_order: tuple[int, ...] | None = None
        self._cross_layer_order_tensor: torch.Tensor | None = None
        self._cross_output_buffers: list[torch.Tensor] = []
        self._cross_block_id_buffers: list[torch.Tensor] = []
        self._cross_block_id_host_buffers: list[torch.Tensor] = []
        self._cross_block_id_host_views: list[Any] = []
        self._layer_names: list[str] = []
        self._layer_groups: dict[str, int] = {}
        self._tp_rank = 0
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._device = get_torch_device()
        self._copy_stream: Any = None
        self._writer_threads = int(self._kv_transfer_config.get_from_extra_config("streamopd_writer_threads", 4))
        self._executor = ThreadPoolExecutor(
            max_workers=self._writer_threads,
            thread_name_prefix="streamopd-kv-save",
        )
        # CUDA submission must never wait for a Host staging slot on vLLM's
        # model-runner thread. One ordered submitter is enough because all D2H
        # copies use the same stream; Host commits still fan out above.
        self._transfer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="streamopd-kv-submit",
        )
        self._host_pool: HostKVSlotPool | None = None
        self._staging_buffers: list[torch.Tensor] = []
        self._staging_token_capacity = self._chunk_size
        self._staging_available: queue.Queue[int] = queue.Queue()
        self._seal_futures: dict[str, Future] = {}
        self._futures: dict[str, Future] = {}
        self._request_futures: dict[str, list[Future]] = {}
        self._finished_requests: set[str] = set()
        self._claimed_requests: set[str] = set()
        self._transfer_stats_lock = threading.Lock()
        self._transfer_stats = _empty_transfer_stats()

    def _ensure_transfer_stats(self) -> None:
        # A few focused tests construct the connector with ``__new__``.
        if not hasattr(self, "_transfer_stats_lock"):
            self._transfer_stats_lock = threading.Lock()
            self._transfer_stats = _empty_transfer_stats()

    def _record_transfer_stats(self, *, maxima: dict[str, float] | None = None, **increments: float) -> None:
        self._ensure_transfer_stats()
        with self._transfer_stats_lock:
            for key, value in increments.items():
                self._transfer_stats[key] += float(value)
            for key, value in (maxima or {}).items():
                self._transfer_stats[key] = max(self._transfer_stats[key], float(value))

    def reset_transfer_stats(self) -> None:
        """Reset policy-scoped transfer counters after prior writes drain."""

        if any(not future.done() for future in getattr(self, "_futures", {}).values()):
            raise RuntimeError("cannot reset StreamOPD transfer metrics while writes are active")
        self._ensure_transfer_stats()
        with self._transfer_stats_lock:
            self._transfer_stats = _empty_transfer_stats()

    def get_transfer_stats(self) -> dict[str, float]:
        """Return one rollout worker's policy-scoped transfer counters."""

        self._ensure_transfer_stats()
        with self._transfer_stats_lock:
            return dict(self._transfer_stats)

    def wait_for_all_exports(self) -> float:
        """Drain EOS exports before vLLM releases its KV cache allocation."""

        started = time.perf_counter()
        for future in list(self._seal_futures.values()):
            future.result()
        return time.perf_counter() - started

    def _get_copy_stream(self) -> Any:
        if self._copy_stream is None:
            self._copy_stream = self._device.Stream()
        return self._copy_stream

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._export_strategy == "eos_host":
            raise RuntimeError(
                "StreamOPD eos_host requires vLLM's uniform cross-layer cache; "
                "select eos_triton explicitly so preflight reserves its GPU gather workspace"
            )
        self._tp_rank = get_tensor_model_parallel_rank()
        self._tp_size = get_tensor_model_parallel_world_size()
        self._kv_caches = dict(kv_caches)
        self._layer_names = sorted(self._kv_caches, key=_layer_sort_key)
        if not self._layer_names:
            raise RuntimeError("vLLM did not register any KV cache layers")
        if self._kv_cache_config is not None:
            for group_idx, group in enumerate(self._kv_cache_config.kv_cache_groups):
                for layer_name in group.layer_names:
                    self._layer_groups[layer_name] = group_idx
        missing = [name for name in self._layer_names if name not in self._layer_groups]
        if missing:
            raise RuntimeError(f"KV cache group mapping is missing layers: {missing[:3]}")
        sample = extract_vllm_nhd_token_range(
            self._kv_caches[self._layer_names[0]],
            [0],
            self._block_size,
            0,
            1,
        )
        _, kv_axis, num_kv_heads, head_dim = sample.shape
        if kv_axis != 2:
            raise RuntimeError(f"invalid vLLM shared Host KV sample shape: {tuple(sample.shape)}")
        self._initialize_host_storage(len(self._layer_names), num_kv_heads, head_dim, sample.dtype)

    def register_cross_layers_kv_cache(self, kv_cache: torch.Tensor, attn_backend: type) -> None:
        """Register vLLM's block-major uniform NHD cache."""

        self._tp_rank = get_tensor_model_parallel_rank()
        self._tp_size = get_tensor_model_parallel_world_size()
        if kv_cache.ndim != 6 or kv_cache.shape[2] != 2 or kv_cache.shape[3] != self._block_size:
            raise RuntimeError(f"invalid vLLM cross-layer NHD cache shape: {tuple(kv_cache.shape)}")
        if not kv_cache.is_contiguous():
            raise RuntimeError("StreamOPD requires vLLM's physical cross-layer KV buffer to be contiguous")
        if self._kv_cache_config is None:
            raise RuntimeError("vLLM did not provide cross-layer KV cache metadata")
        cache_layer_names = []
        for tensor in self._kv_cache_config.kv_cache_tensors:
            if len(tensor.shared_by) != 1:
                raise RuntimeError("StreamOPD cross-layer cache does not support shared KV layers")
            cache_layer_names.append(tensor.shared_by[0])
        if len(cache_layer_names) != kv_cache.shape[1]:
            raise RuntimeError("vLLM cross-layer cache dimension does not match its layer metadata")
        self._layer_names = sorted(cache_layer_names, key=_layer_sort_key)
        self._cross_layer_order = tuple(cache_layer_names.index(name) for name in self._layer_names)
        self._cross_layers_kv_cache = kv_cache
        max_blocks = (self._chunk_size + 2 * self._block_size - 2) // self._block_size
        if self._export_strategy == "eos_host":
            self._staging_token_capacity = max_blocks * self._block_size
        self._initialize_host_storage(
            len(self._layer_names),
            int(kv_cache.shape[4]),
            int(kv_cache.shape[5]),
            kv_cache.dtype,
        )
        if self._export_strategy != "eos_host":
            self._cross_layer_order_tensor = torch.tensor(
                self._cross_layer_order,
                dtype=torch.long,
                device=kv_cache.device,
            )
            for _ in range(self._writer_threads):
                self._cross_output_buffers.append(
                    torch.empty(
                        len(self._layer_names),
                        self._chunk_size,
                        2,
                        kv_cache.shape[4],
                        kv_cache.shape[5],
                        dtype=kv_cache.dtype,
                        device=kv_cache.device,
                    )
                )
                host_ids = torch.empty(max_blocks, dtype=torch.long, device="cpu", pin_memory=True)
                self._cross_block_id_host_buffers.append(host_ids)
                self._cross_block_id_host_views.append(host_ids.numpy())
                self._cross_block_id_buffers.append(torch.empty(max_blocks, dtype=torch.long, device=kv_cache.device))
            # Compile the gather kernel before the first timed policy step.
            self._cross_block_id_buffers[0][0] = 0
            gather_vllm_cross_layers_nhd_into(
                kv_cache,
                self._cross_block_id_buffers[0][:1],
                self._cross_layer_order_tensor,
                self._cross_output_buffers[0],
                token_offset=0,
                token_count=1,
            )
            self._device.synchronize(kv_cache.device)

    def _initialize_host_storage(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self._host_pool = HostKVSlotPool.create_or_open(
            self._storage_path,
            tp_rank=self._tp_rank,
            slot_count=self._host_slot_count,
            token_capacity=self._host_slot_tokens,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=self._block_size,
            dtype=dtype,
        )
        staging_elements = num_layers * self._staging_token_capacity * 2 * num_kv_heads * head_dim
        for index in range(self._writer_threads):
            self._staging_buffers.append(torch.empty(staging_elements, dtype=dtype, device="cpu", pin_memory=True))
            self._staging_available.put(index)

    def _get_host_pool(self) -> HostKVSlotPool:
        if self._host_pool is None:
            self._host_pool = HostKVSlotPool.open_existing(self._storage_path, tp_rank=self._tp_rank)
        return self._host_pool

    def _staging_layer(self, staging_index: int, layer_index: int) -> torch.Tensor:
        pool = self._get_host_pool()
        layer_elements = self._staging_token_capacity * 2 * pool.num_kv_heads * pool.head_dim
        offset = layer_index * layer_elements
        return self._staging_buffers[staging_index][offset : offset + layer_elements].view(
            self._staging_token_capacity,
            2,
            pool.num_kv_heads,
            pool.head_dim,
        )

    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def handle_preemptions(self, metadata: StreamOPDKVConnectorMetadata | set[str]) -> None:
        """Drain earlier incremental copies before vLLM reuses their pages."""

        # Newer vLLM passes the connector metadata, while 0.15.1 passes ids.
        preempted_req_ids = (
            metadata.preempted_req_ids if isinstance(metadata, StreamOPDKVConnectorMetadata) else metadata
        )
        for req_id in preempted_req_ids:
            for future in self._request_futures.get(req_id, ()):
                future.result()

    @staticmethod
    def _block_runs(block_ids: list[int]) -> list[tuple[int, int, int, int]]:
        """Return (logical offset, physical start, stride, count) runs."""

        if not block_ids:
            return []
        runs = []
        logical_start = 0
        while logical_start < len(block_ids):
            physical_start = block_ids[logical_start]
            end = logical_start + 1
            stride = 1
            if end < len(block_ids):
                candidate_stride = block_ids[end] - physical_start
                if candidate_stride > 0:
                    stride = candidate_stride
                    end += 1
                    while end < len(block_ids) and block_ids[end] - block_ids[end - 1] == stride:
                        end += 1
            runs.append((logical_start, physical_start, stride, end - logical_start))
            logical_start = end
        return runs

    def _copy_raw_blocks_d2h(
        self,
        destination: torch.Tensor,
        block_ids: list[int],
        copy_stream: Any,
    ) -> tuple[int, int]:
        """Copy arithmetic physical-page runs without a GPU gather kernel."""

        assert self._cross_layers_kv_cache is not None
        page_bytes = self._cross_layers_kv_cache[0].numel() * self._cross_layers_kv_cache.element_size()
        runs = self._block_runs(block_ids)
        copy_calls = 0
        for logical_start, physical_start, stride, count in runs:
            source_pitch = stride * page_bytes
            if cuda_runtime is not None and source_pitch <= 2**31 - 1:
                result = cuda_runtime.cudaMemcpy2DAsync(
                    destination.data_ptr() + logical_start * page_bytes,
                    page_bytes,
                    self._cross_layers_kv_cache.data_ptr() + physical_start * page_bytes,
                    source_pitch,
                    page_bytes,
                    count,
                    cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    copy_stream.cuda_stream,
                )
                error = result[0] if isinstance(result, tuple) else result
                if error == cuda_runtime.cudaError_t.cudaSuccess:
                    copy_calls += 1
                    continue
                raise RuntimeError(f"cudaMemcpy2DAsync failed during StreamOPD KV export: {error}")
            for offset in range(count):
                destination[logical_start + offset].copy_(
                    self._cross_layers_kv_cache[physical_start + offset * stride],
                    non_blocking=True,
                )
                copy_calls += 1
        return len(runs), copy_calls

    @staticmethod
    def _copy_raw_token_range(
        destination: torch.Tensor,
        source_blocks: torch.Tensor,
        *,
        token_offset: int,
        token_count: int,
    ) -> None:
        """Reorder block-major Host rows directly into a token-major slot."""

        block_size = source_blocks.shape[1]
        source_block = 0
        copied = 0
        if token_offset:
            count = min(token_count, block_size - token_offset)
            destination[:count].copy_(source_blocks[0, token_offset : token_offset + count])
            source_block = 1
            copied = count
        full_blocks = (token_count - copied) // block_size
        if full_blocks:
            full_tokens = full_blocks * block_size
            destination[copied : copied + full_tokens].view(full_blocks, block_size, *destination.shape[1:]).copy_(
                source_blocks[source_block : source_block + full_blocks]
            )
            source_block += full_blocks
            copied += full_tokens
        if copied < token_count:
            destination[copied:token_count].copy_(source_blocks[source_block, : token_count - copied])

    def _raw_staging_blocks(self, staging_index: int, block_count: int) -> torch.Tensor:
        pool = self._get_host_pool()
        capacity_blocks = self._staging_token_capacity // self._block_size
        if not 0 < block_count <= capacity_blocks:
            raise RuntimeError("raw StreamOPD staging block count exceeds capacity")
        return self._staging_buffers[staging_index].view(
            capacity_blocks,
            pool.num_layers,
            2,
            self._block_size,
            pool.num_kv_heads,
            pool.head_dim,
        )[:block_count]

    def _commit_host_chunk(
        self,
        pool: HostKVSlotPool,
        slot: int,
        start: int,
        end: int,
        staging_index: int,
        started_event: Any,
        gathered_event: Any,
        event: Any,
        *,
        raw_block_count: int = 0,
        raw_token_offset: int = 0,
    ) -> None:
        wait_started = time.perf_counter()
        try:
            event.synchronize()
            d2h_wait_seconds = time.perf_counter() - wait_started
            gpu_copy_seconds = started_event.elapsed_time(event) / 1000.0
            if raw_block_count:
                gpu_gather_seconds = 0.0
                gpu_d2h_seconds = gpu_copy_seconds
            else:
                gpu_gather_seconds = started_event.elapsed_time(gathered_event) / 1000.0
                gpu_d2h_seconds = gathered_event.elapsed_time(event) / 1000.0
            commit_started = time.perf_counter()
            chunk_tokens = end - start
            if raw_block_count:
                assert self._cross_layer_order is not None
                raw = self._raw_staging_blocks(staging_index, raw_block_count)
                for layer_index, source_layer in enumerate(self._cross_layer_order):
                    key, value = pool.layer(slot, layer_index)
                    self._copy_raw_token_range(
                        key[start:end],
                        raw[:, source_layer, 0],
                        token_offset=raw_token_offset,
                        token_count=chunk_tokens,
                    )
                    self._copy_raw_token_range(
                        value[start:end],
                        raw[:, source_layer, 1],
                        token_offset=raw_token_offset,
                        token_count=chunk_tokens,
                    )
            else:
                for layer_index in range(pool.num_layers):
                    staging = self._staging_layer(staging_index, layer_index)[:chunk_tokens]
                    key, value = pool.layer(slot, layer_index)
                    key[start:end].copy_(staging[:, 0])
                    value[start:end].copy_(staging[:, 1])
            self._record_transfer_stats(
                copy_chunks=1,
                copy_bytes=chunk_tokens * pool.num_layers * 2 * pool.num_kv_heads * pool.head_dim * pool.element_size,
                gpu_gather_seconds=gpu_gather_seconds,
                gpu_d2h_seconds=gpu_d2h_seconds,
                gpu_copy_seconds=gpu_copy_seconds,
                d2h_wait_seconds=d2h_wait_seconds,
                host_commit_seconds=time.perf_counter() - commit_started,
            )
        finally:
            self._staging_available.put(staging_index)

    @staticmethod
    def _seal_host_slot(
        prior_futures: list[Future],
        pool: HostKVSlotPool,
        pending: _PendingSave,
    ) -> None:
        for future in prior_futures:
            future.result()
        pool.seal(
            pending.slot_path,
            request_id=pending.req_id,
            trajectory_id=pending.trajectory_id,
            policy_version=pending.policy_version,
            prompt_length=pending.prompt_length,
            token_ids=pending.token_ids,
            token_count=pending.end,
            streamed_tokens_before_eos=pending.streamed_tokens_before_eos,
            streamed_chunks_before_eos=pending.streamed_chunks_before_eos,
        )

    def _write_done(self, key: str, future: Future) -> None:
        self._futures.pop(key, None)
        if key.endswith(":seal"):
            req_id = key.removesuffix(":seal")
            self._request_futures.pop(req_id, None)
        if exception := future.exception():
            logger.error("StreamOPD KV write failed for %s: %r", key, exception)

    def _submit_save(self, pending: _PendingSave) -> None:
        if pending.terminal:
            self._claimed_requests.add(pending.req_id)
        self._submit_host_slot_save(pending)

    def _launch_host_slot_chunk(
        self,
        pending: _PendingSave,
        pool: HostKVSlotPool,
        slot: int,
        request_futures: list[Future],
        *,
        ready_event: Any | None = None,
    ) -> None:
        if not 0 <= pending.start <= pending.end <= pool.token_capacity:
            raise RuntimeError("StreamOPD KV chunk is outside its shared Host slot")
        chunk_tokens = pending.end - pending.start
        if chunk_tokens > self._chunk_size:
            raise RuntimeError("StreamOPD KV chunk exceeds the fixed Host staging capacity")
        if chunk_tokens:
            staging_wait_started = time.perf_counter()
            staging_index = self._staging_available.get()
            staging_wait_seconds = time.perf_counter() - staging_wait_started
            self._record_transfer_stats(
                staging_wait_seconds=staging_wait_seconds,
                maxima={"max_staging_wait_seconds": staging_wait_seconds},
            )
            copy_stream = self._get_copy_stream()
            if ready_event is None:
                ready_event = self._device.Event()
                ready_event.record()
            copy_stream.wait_event(ready_event)
            enqueue_started = time.perf_counter()
            copy_calls = 1
            block_runs = 0
            raw_block_count = 0
            raw_token_offset = 0
            try:
                with self._device.stream(copy_stream):
                    copy_started = self._device.Event(enable_timing=True)
                    copy_started.record(copy_stream)
                    if self._cross_layers_kv_cache is not None and self._export_strategy == "eos_host":
                        if len(pending.block_ids_by_group) != 1:
                            raise RuntimeError("cross-layer KV export requires one uniform cache group")
                        first_block = pending.start // self._block_size
                        last_block = (pending.end + self._block_size - 1) // self._block_size
                        selected_blocks = pending.block_ids_by_group[0][first_block:last_block]
                        raw_block_count = len(selected_blocks)
                        raw_token_offset = pending.start - first_block * self._block_size
                        raw_staging = self._raw_staging_blocks(staging_index, raw_block_count)
                        assert self._cross_layers_kv_cache is not None
                        gathered = self._device.Event(enable_timing=True)
                        gathered.record(copy_stream)
                        block_runs, copy_calls = self._copy_raw_blocks_d2h(
                            raw_staging,
                            selected_blocks,
                            copy_stream,
                        )
                    elif self._cross_layers_kv_cache is not None:
                        if len(pending.block_ids_by_group) != 1:
                            raise RuntimeError("cross-layer KV export requires one uniform cache group")
                        first_block = pending.start // self._block_size
                        last_block = (pending.end + self._block_size - 1) // self._block_size
                        selected_blocks = pending.block_ids_by_group[0][first_block:last_block]
                        block_count = len(selected_blocks)
                        self._cross_block_id_host_views[staging_index][:block_count] = selected_blocks
                        device_block_ids = self._cross_block_id_buffers[staging_index][:block_count]
                        device_block_ids.copy_(
                            self._cross_block_id_host_buffers[staging_index][:block_count],
                            non_blocking=True,
                        )
                        assert self._cross_layer_order_tensor is not None
                        gather_vllm_cross_layers_nhd_into(
                            self._cross_layers_kv_cache,
                            device_block_ids,
                            self._cross_layer_order_tensor,
                            self._cross_output_buffers[staging_index],
                            token_offset=pending.start - first_block * self._block_size,
                            token_count=chunk_tokens,
                        )
                        gathered = self._device.Event(enable_timing=True)
                        gathered.record(copy_stream)
                        staging = self._staging_buffers[staging_index].view(
                            pool.num_layers,
                            self._chunk_size,
                            2,
                            pool.num_kv_heads,
                            pool.head_dim,
                        )
                        # A partial slice is strided across layers. PyTorch's
                        # CUDA-to-CPU copy materializes a temporary contiguous
                        # GPU tensor for that shape, defeating the fixed-memory
                        # contract at terminal chunks. Copy the preallocated
                        # contiguous buffer; the Host commit below consumes only
                        # ``chunk_tokens`` from each layer.
                        staging.copy_(self._cross_output_buffers[staging_index], non_blocking=True)
                    else:
                        gathered = self._device.Event(enable_timing=True)
                        gathered.record(copy_stream)
                        copy_calls = len(self._layer_names)
                        for layer_index, layer_name in enumerate(self._layer_names):
                            group_index = self._layer_groups[layer_name]
                            extracted = extract_vllm_nhd_token_range(
                                self._kv_caches[layer_name],
                                pending.block_ids_by_group[group_index],
                                self._block_size,
                                pending.start,
                                pending.end,
                            )
                            staging = self._staging_layer(staging_index, layer_index)
                            staging[:chunk_tokens].copy_(extracted, non_blocking=True)
                copied = self._device.Event(enable_timing=True)
                copied.record(copy_stream)
            except BaseException:
                self._staging_available.put(staging_index)
                raise
            self._record_transfer_stats(
                copy_calls=copy_calls,
                block_runs=block_runs,
                copy_enqueue_seconds=time.perf_counter() - enqueue_started,
            )
            chunk_future = self._executor.submit(
                self._commit_host_chunk,
                pool,
                slot,
                pending.start,
                pending.end,
                staging_index,
                copy_started,
                gathered,
                copied,
                raw_block_count=raw_block_count,
                raw_token_offset=raw_token_offset,
            )
            future_key = f"{pending.req_id}:{pending.chunk_index}"
            self._futures[future_key] = chunk_future
            request_futures.append(chunk_future)
            self._record_transfer_stats(
                maxima={
                    "max_outstanding_writes": sum(not future.done() for future in self._futures.values()),
                }
            )
            chunk_future.add_done_callback(partial(self._write_done, future_key))

    def _submit_host_slot_save(self, pending: _PendingSave) -> None:
        pool = self._get_host_pool()
        slot = pool.validate_writer(
            pending.slot_path,
            request_id=pending.req_id,
            trajectory_id=pending.trajectory_id,
            policy_version=pending.policy_version,
        )
        request_futures = self._request_futures.setdefault(pending.req_id, [])
        self._launch_host_slot_chunk(pending, pool, slot, request_futures)

        if not pending.terminal:
            return
        if pending.token_ids is None or pending.token_ids.numel() != pending.end:
            raise RuntimeError("terminal StreamOPD KV chunk must carry the complete token identity")
        seal_future = self._executor.submit(self._seal_host_slot, list(request_futures), pool, pending)
        seal_key = f"{pending.req_id}:seal"
        self._futures[seal_key] = seal_future
        self._seal_futures[pending.req_id] = seal_future
        request_futures.append(seal_future)
        seal_future.add_done_callback(partial(self._write_done, seal_key))

    def _export_terminal_saves(self, pending_saves: list[_PendingSave], ready_event: Any) -> Future:
        if not pending_saves or not pending_saves[-1].terminal:
            raise RuntimeError("EOS-only StreamOPD export requires a terminal final chunk")
        req_id = pending_saves[-1].req_id
        malformed = any(
            pending.req_id != req_id or pending.terminal != (index == len(pending_saves) - 1)
            for index, pending in enumerate(pending_saves)
        )
        if malformed:
            raise RuntimeError("EOS-only StreamOPD export received a malformed request batch")
        terminal = pending_saves[-1]
        if terminal.token_ids is None or terminal.token_ids.numel() != terminal.end:
            raise RuntimeError("terminal StreamOPD KV chunk must carry the complete token identity")
        pool = self._get_host_pool()
        slot = pool.validate_writer(
            terminal.slot_path,
            request_id=terminal.req_id,
            trajectory_id=terminal.trajectory_id,
            policy_version=terminal.policy_version,
        )
        chunk_futures: list[Future] = []
        for pending in pending_saves:
            self._launch_host_slot_chunk(
                pending,
                pool,
                slot,
                chunk_futures,
                ready_event=ready_event,
            )
        # Do not occupy the ordered CUDA submitter while Host writers finish.
        # Returning the seal future lets the next trajectory enqueue D2H as
        # soon as a staging buffer becomes available.
        return self._executor.submit(self._seal_host_slot, chunk_futures, pool, terminal)

    @staticmethod
    def _propagate_future(source: Future, destination: Future) -> None:
        if destination.done():
            return
        try:
            destination.set_result(source.result())
        except BaseException as error:
            destination.set_exception(error)

    @classmethod
    def _terminal_export_submitted(cls, completion: Future, submission: Future) -> None:
        try:
            seal_future = submission.result()
        except BaseException as error:
            completion.set_exception(error)
            return
        seal_future.add_done_callback(partial(cls._propagate_future, destination=completion))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        has_model_work = True
        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if isinstance(metadata, StreamOPDKVConnectorMetadata):
                has_model_work = metadata.has_model_work
                if self._export_strategy == "incremental_triton":
                    for pending in metadata.pending_saves:
                        self._submit_save(pending)
                elif metadata.pending_saves:
                    ready_event = self._device.Event()
                    ready_event.record()
                    by_request: dict[str, list[_PendingSave]] = {}
                    for pending in metadata.pending_saves:
                        by_request.setdefault(pending.req_id, []).append(pending)
                    for req_id, pending_saves in by_request.items():
                        self._claimed_requests.add(req_id)
                        completion = Future()
                        submission = self._transfer_executor.submit(
                            self._export_terminal_saves,
                            pending_saves,
                            ready_event,
                        )
                        seal_key = f"{req_id}:seal"
                        self._futures[seal_key] = completion
                        self._seal_futures[req_id] = completion
                        submission.add_done_callback(partial(self._terminal_export_submitted, completion))
                        completion.add_done_callback(partial(self._write_done, seal_key))
        newly_finished = finished_req_ids & self._claimed_requests
        self._finished_requests.update(newly_finished)
        done: set[str] = set()
        seal_futures = getattr(self, "_seal_futures", {})
        for req_id in list(self._finished_requests):
            # Keep vLLM pages owned until the terminal control record is sealed
            # after every shared-slot commit completes.
            seal_future = seal_futures.get(req_id)
            # The final cohort request has no later model step on which to poll
            # the transfer. Waiting on the seal future here is bounded by
            # the writer pool and keeps the scheduler-side ownership contract
            # intact while allowing the tensor copies themselves to remain
            # asynchronous.
            if seal_future is None:
                continue
            if not seal_future.done():
                if has_model_work:
                    continue
                # There may be no model step after the last request in a
                # cohort. Complete the terminal seal here so vLLM can reclaim
                # its pages immediately.
                wait_started = time.perf_counter()
                seal_future.result()
                self._record_transfer_stats(terminal_wait_seconds=time.perf_counter() - wait_started)
            seal_future.result()
            self._finished_requests.remove(req_id)
            self._claimed_requests.discard(req_id)
            done.add(req_id)
            seal_futures.pop(req_id, None)
        return done or None, None

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        if num_external_tokens != 0:
            raise RuntimeError("StreamOPDKVConnector is store-only")

    def update_connector_output(self, connector_output: Any) -> None:
        # Scheduler-side ownership ends only after the worker has made the
        # terminal slot seal durable and reports finished_sending.
        for req_id in connector_output.finished_sending or ():
            self._claimed_requests.discard(req_id)

    @staticmethod
    def _update_block_ids(
        state: _SchedulerSaveState,
        new_block_ids: tuple[list[int], ...] | None,
        *,
        replace: bool,
    ) -> None:
        if new_block_ids is None:
            return
        if len(new_block_ids) != len(state.block_ids_by_group):
            raise RuntimeError("StreamOPD KV cache group count changed during rollout")
        for group_idx, block_ids in enumerate(new_block_ids):
            if block_ids is None:
                continue
            if replace:
                state.block_ids_by_group[group_idx] = list(block_ids)
            else:
                state.block_ids_by_group[group_idx].extend(block_ids)

    def _queue_committed(
        self,
        state: _SchedulerSaveState,
        committed_end: int,
        *,
        terminal: bool = False,
        token_ids: torch.Tensor | None = None,
    ) -> None:
        if committed_end < state.published_tokens:
            raise RuntimeError("StreamOPD computed-token progress moved behind published KV")
        capacity = min(len(group) * self._block_size for group in state.block_ids_by_group)
        if committed_end > capacity:
            raise RuntimeError(f"StreamOPD KV progress {committed_end} exceeds allocated block capacity {capacity}")
        if self._export_strategy != "incremental_triton" and not terminal:
            return
        published_before_terminal = state.published_tokens
        chunks_before_terminal = state.next_chunk_index
        while state.published_tokens + self._chunk_size <= committed_end and not (
            terminal and state.published_tokens + self._chunk_size == committed_end
        ):
            start = state.published_tokens
            end = start + self._chunk_size
            self._pending.append(
                _PendingSave(
                    req_id=state.req_id,
                    trajectory_id=state.trajectory_id,
                    slot_path=state.slot_path,
                    block_ids_by_group=tuple(list(group) for group in state.block_ids_by_group),
                    policy_version=state.policy_version,
                    prompt_length=state.prompt_length,
                    start=start,
                    end=end,
                    chunk_index=state.next_chunk_index,
                    terminal=False,
                    token_ids=None,
                )
            )
            state.chunks.append((start, end))
            state.published_tokens = end
            state.next_chunk_index += 1
        if terminal and state.published_tokens < committed_end:
            start = state.published_tokens
            self._pending.append(
                _PendingSave(
                    req_id=state.req_id,
                    trajectory_id=state.trajectory_id,
                    slot_path=state.slot_path,
                    block_ids_by_group=tuple(list(group) for group in state.block_ids_by_group),
                    policy_version=state.policy_version,
                    prompt_length=state.prompt_length,
                    start=start,
                    end=committed_end,
                    chunk_index=state.next_chunk_index,
                    terminal=True,
                    token_ids=token_ids,
                    streamed_tokens_before_eos=published_before_terminal,
                    streamed_chunks_before_eos=chunks_before_terminal,
                )
            )
            state.chunks.append((start, committed_end))
            state.published_tokens = committed_end
            state.next_chunk_index += 1
        elif terminal and state.published_tokens == committed_end:
            self._pending.append(
                _PendingSave(
                    req_id=state.req_id,
                    trajectory_id=state.trajectory_id,
                    slot_path=state.slot_path,
                    block_ids_by_group=tuple(list(group) for group in state.block_ids_by_group),
                    policy_version=state.policy_version,
                    prompt_length=state.prompt_length,
                    start=committed_end,
                    end=committed_end,
                    chunk_index=state.next_chunk_index,
                    terminal=True,
                    token_ids=token_ids,
                    streamed_tokens_before_eos=published_before_terminal,
                    streamed_chunks_before_eos=chunks_before_terminal,
                )
            )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        for req_id in scheduler_output.preempted_req_ids or ():
            state = self._scheduler_states.get(req_id)
            if state is not None:
                # vLLM resumes a preempted request with replacement blocks and
                # recomputes its prefix. EOS-only export has not copied data;
                # incremental export safely rewrites the recomputed prefix.
                state.published_tokens = 0
        for request in scheduler_output.scheduled_new_reqs:
            extra_args = request.sampling_params.extra_args if request.sampling_params else None
            params = (extra_args or {}).get("kv_transfer_params") or {}
            if not params.get("streamopd_kv", False):
                continue
            pool = self._get_host_pool()
            slot_path = pool.acquire(
                request_id=request.req_id,
                trajectory_id=str(params["trajectory_id"]),
                policy_version=int(params["policy_version"]),
                prompt_length=int(params["prompt_length"]),
            )
            self._scheduler_paths[request.req_id] = slot_path
            self._scheduler_states[request.req_id] = _SchedulerSaveState(
                req_id=request.req_id,
                trajectory_id=str(params["trajectory_id"]),
                slot_path=slot_path,
                block_ids_by_group=[list(group) for group in request.block_ids],
                policy_version=int(params["policy_version"]),
                prompt_length=int(params["prompt_length"]),
            )
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_blocks, num_computed in zip(
            cached.req_ids, cached.new_block_ids, cached.num_computed_tokens, strict=True
        ):
            state = self._scheduler_states.get(req_id)
            if state is None:
                continue
            self._update_block_ids(state, new_blocks, replace=req_id in cached.resumed_req_ids)
            self._queue_committed(state, int(num_computed))
        metadata = StreamOPDKVConnectorMetadata(
            pending_saves=list(self._pending),
            has_model_work=bool(scheduler_output.num_scheduled_tokens),
            preempted_req_ids=set(scheduler_output.preempted_req_ids or ()),
        )
        self._pending.clear()
        return metadata

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params or {}
        if not params.get("streamopd_kv", False):
            self._scheduler_paths.pop(request.request_id, None)
            self._scheduler_states.pop(request.request_id, None)
            return False, None
        if str(request.status) in {"FINISHED_ABORTED", "FINISHED_ERROR", "FINISHED_IGNORED"}:
            self._scheduler_paths.pop(request.request_id, None)
            self._scheduler_states.pop(request.request_id, None)
            return False, None
        request_id = request.request_id
        slot_path = self._scheduler_paths.pop(request_id, None)
        state = self._scheduler_states.pop(request_id, None)
        if state is None:
            if request_id in self._claimed_requests:
                # A finish callback can be delivered twice by an
                # asynchronously scheduled engine. Keep ownership delayed
                # until update_connector_output observes finished_sending;
                # returning False here would free the request before the
                # worker completion reaches the scheduler.
                logger.warning("coalescing duplicate StreamOPD KV finish callback for %s", request_id)
                return True, None
            raise RuntimeError(f"StreamOPD KV finish callback has no scheduler state for {request_id}")
        if slot_path is None:
            slot_path = state.slot_path
        state.block_ids_by_group = [list(group) for group in block_ids]
        token_ids = torch.tensor(list(request.all_token_ids)[:-1], dtype=torch.long)
        streamed_tokens_before_eos = state.published_tokens
        streamed_chunks_before_eos = state.next_chunk_index
        self._queue_committed(state, token_ids.numel(), terminal=True, token_ids=token_ids)
        self._claimed_requests.add(request.request_id)
        return True, {
            "streamopd_kv_path": slot_path,
            "streamopd_kv_tp_size": self._tp_size,
            "streamopd_kv_policy_version": state.policy_version,
            "streamopd_kv_num_tokens": token_ids.numel(),
            "streamopd_kv_chunks": state.next_chunk_index,
            "streamopd_kv_streamed_tokens_before_eos": streamed_tokens_before_eos,
            "streamopd_kv_streamed_chunks_before_eos": streamed_chunks_before_eos,
        }

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "NHD"

    def shutdown(self) -> None:
        self._transfer_executor.shutdown(wait=True)
        for future in list(self._futures.values()):
            future.result()
        self._executor.shutdown(wait=True)
        if self._host_pool is not None:
            self._host_pool.close()
            self._host_pool = None
