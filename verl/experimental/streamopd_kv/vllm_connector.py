# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""vLLM store-only connector for incremental rollout KV snapshots."""

from __future__ import annotations

import fcntl
import json
import os
import queue
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file
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
from .snapshot_io import extract_vllm_nhd_token_range

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _layer_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return (int(match.group(1)), name) if match else (2**31 - 1, name)


@dataclass
class _PendingSave:
    req_id: str
    trajectory_id: str
    base_path: str
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
    base_path: str
    block_ids_by_group: list[list[int]]
    policy_version: int
    prompt_length: int
    published_tokens: int = 0
    next_chunk_index: int = 0
    chunks: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class StreamOPDKVConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[_PendingSave] = field(default_factory=list)
    new_request_paths: dict[str, str] = field(default_factory=dict)


class StreamOPDKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Seal post-RoPE K/V pages before vLLM returns them to its allocator."""

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

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
        self._host_slot_count = int(self._kv_transfer_config.get_from_extra_config("streamopd_host_slot_count", 0))
        self._host_slot_tokens = int(self._kv_transfer_config.get_from_extra_config("streamopd_host_slot_tokens", 0))
        if (self._host_slot_count < 1) != (self._host_slot_tokens < 1):
            raise ValueError("shared Host KV slot count and token capacity must be configured together")
        self._scheduler_paths: dict[str, str] = {}
        self._scheduler_states: dict[str, _SchedulerSaveState] = {}
        self._pending: list[_PendingSave] = []

        self._kv_caches: dict[str, torch.Tensor] = {}
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
        self._host_pool: HostKVSlotPool | None = None
        self._staging_buffers: list[torch.Tensor] = []
        self._staging_available: queue.Queue[int] = queue.Queue()
        self._lock_fds: dict[str, int] = {}
        self._copy_events: dict[str, Any] = {}
        self._manifest_futures: dict[str, Future] = {}
        self._futures: dict[str, Future] = {}
        self._request_futures: dict[str, list[Future]] = {}
        self._finished_requests: set[str] = set()
        self._claimed_requests: set[str] = set()

    @staticmethod
    def _rank_path(base_path: str, tp_rank: int) -> str:
        return f"{base_path}.tp{tp_rank}.safetensors"

    @staticmethod
    def _manifest_path(base_path: str, tp_rank: int) -> str:
        return f"{base_path}.tp{tp_rank}.manifest.safetensors"

    @staticmethod
    def _chunk_path(base_path: str, tp_rank: int, chunk_index: int) -> str:
        return f"{base_path}.tp{tp_rank}.chunk{chunk_index:05d}.safetensors"

    def _get_copy_stream(self) -> Any:
        if self._copy_stream is None:
            self._copy_stream = self._device.Stream()
        return self._copy_stream

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
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
        if self._host_slot_count:
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
            self._host_pool = HostKVSlotPool.create_or_open(
                self._storage_path,
                tp_rank=self._tp_rank,
                slot_count=self._host_slot_count,
                token_capacity=self._host_slot_tokens,
                num_layers=len(self._layer_names),
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=self._block_size,
                dtype=sample.dtype,
            )
            staging_elements = len(self._layer_names) * self._chunk_size * 2 * num_kv_heads * head_dim
            for index in range(self._writer_threads):
                self._staging_buffers.append(
                    torch.empty(staging_elements, dtype=sample.dtype, device="cpu", pin_memory=True)
                )
                self._staging_available.put(index)

    def _get_host_pool(self) -> HostKVSlotPool | None:
        if not self._host_slot_count:
            return None
        if self._host_pool is None:
            self._host_pool = HostKVSlotPool.open_existing(self._storage_path, tp_rank=self._tp_rank)
        return self._host_pool

    def _staging_layer(self, staging_index: int, layer_index: int) -> torch.Tensor:
        pool = self._get_host_pool()
        if pool is None:
            raise RuntimeError("shared Host KV staging requested without a slot pool")
        layer_elements = self._chunk_size * 2 * pool.num_kv_heads * pool.head_dim
        offset = layer_index * layer_elements
        return self._staging_buffers[staging_index][offset : offset + layer_elements].view(
            self._chunk_size,
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
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, StreamOPDKVConnectorMetadata):
            return
        for req_id, base_path in metadata.new_request_paths.items():
            if HostKVSlotPool.is_slot_path(base_path):
                continue
            if req_id in self._lock_fds:
                continue
            filename = self._manifest_path(base_path, self._tp_rank)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fd = os.open(filename + ".lock", os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._lock_fds[req_id] = fd

    @staticmethod
    def _write_chunk(
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, str],
        event: Any,
        filename: str,
    ) -> None:
        event.synchronize()
        save_file(tensors, filename, metadata=metadata)

    @staticmethod
    def _seal_manifest(
        prior_futures: list[Future],
        token_ids: torch.Tensor,
        metadata: dict[str, str],
        filename: str,
        lock_fd: int,
    ) -> None:
        try:
            for future in prior_futures:
                future.result()
            save_file({"token_ids": token_ids}, filename, metadata=metadata)
        finally:
            os.close(lock_fd)

    def _commit_host_chunk(
        self,
        pool: HostKVSlotPool,
        slot: int,
        start: int,
        end: int,
        staging_index: int,
        event: Any,
    ) -> None:
        try:
            event.synchronize()
            chunk_tokens = end - start
            for layer_index in range(pool.num_layers):
                staging = self._staging_layer(staging_index, layer_index)[:chunk_tokens]
                key, value = pool.layer(slot, layer_index)
                key[start:end].copy_(staging[:, 0])
                value[start:end].copy_(staging[:, 1])
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
            pending.base_path,
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
        if key.endswith(":manifest"):
            req_id = key.removesuffix(":manifest")
            self._request_futures.pop(req_id, None)
        if exception := future.exception():
            logger.error("StreamOPD KV write failed for %s: %r", key, exception)

    def _submit_save(self, pending: _PendingSave) -> None:
        if HostKVSlotPool.is_slot_path(pending.base_path):
            self._submit_host_slot_save(pending)
            return
        # Scheduler and worker connectors are separate objects. Ownership is
        # claimed again on the worker when the save metadata arrives.
        if pending.end < pending.start:
            raise RuntimeError("StreamOPD KV chunk has a negative extent")
        if pending.terminal:
            self._claimed_requests.add(pending.req_id)
        copy_stream = self._get_copy_stream()
        ready = self._device.Event()
        ready.record()
        copy_stream.wait_event(ready)
        tensors: dict[str, torch.Tensor] = {}
        with self._device.stream(copy_stream):
            if pending.end > pending.start:
                for layer_idx, layer_name in enumerate(self._layer_names):
                    group_idx = self._layer_groups[layer_name]
                    extracted = extract_vllm_nhd_token_range(
                        self._kv_caches[layer_name],
                        pending.block_ids_by_group[group_idx],
                        self._block_size,
                        pending.start,
                        pending.end,
                    )
                    if extracted.ndim != 4 or extracted.shape[1] != 2:
                        raise RuntimeError(
                            f"expected NHD KV [tokens, 2, heads, dim] for {layer_name}, got {tuple(extracted.shape)}"
                        )
                    host = torch.empty_like(extracted, device="cpu", pin_memory=True)
                    host.copy_(extracted, non_blocking=True)
                    tensors[f"layer_{layer_idx:05d}"] = host
        copied = self._device.Event()
        copied.record(copy_stream)
        request_futures = self._request_futures.setdefault(pending.req_id, [])
        has_chunk = pending.end > pending.start
        if has_chunk:
            chunk_filename = self._chunk_path(pending.base_path, self._tp_rank, pending.chunk_index)
            chunk_metadata = {
                "format": "verl-streamopd-kv-v2-chunk",
                "request_id": pending.req_id,
                "trajectory_id": pending.trajectory_id,
                "policy_version": str(pending.policy_version),
                "tp_rank": str(self._tp_rank),
                "tp_size": str(self._tp_size),
                "chunk_index": str(pending.chunk_index),
                "start": str(pending.start),
                "end": str(pending.end),
            }
            chunk_future = self._executor.submit(self._write_chunk, tensors, chunk_metadata, copied, chunk_filename)
            future_key = f"{pending.req_id}:{pending.chunk_index}"
            self._futures[future_key] = chunk_future
            request_futures.append(chunk_future)
            chunk_future.add_done_callback(partial(self._write_done, future_key))

        if not pending.terminal:
            return
        if pending.token_ids is None or pending.token_ids.numel() != pending.end:
            raise RuntimeError("terminal StreamOPD KV chunk must carry the complete token identity")
        lock_fd = self._lock_fds.pop(pending.req_id, None)
        if lock_fd is None:
            raise RuntimeError(f"StreamOPD KV request {pending.req_id} has no manifest lock")
        manifest_metadata = {
            "format": "verl-streamopd-kv-v2",
            "request_id": pending.req_id,
            "trajectory_id": pending.trajectory_id,
            "policy_version": str(pending.policy_version),
            "prompt_length": str(pending.prompt_length),
            "tp_rank": str(self._tp_rank),
            "tp_size": str(self._tp_size),
            "page_size": str(self._block_size),
            "axis_order": "token_kv_head_dim",
            "rope_convention": "post_rope_key",
            "layer_names": json.dumps(self._layer_names),
            "num_chunks": str(pending.chunk_index + int(has_chunk)),
            "streamed_tokens_before_eos": str(pending.streamed_tokens_before_eos),
            "streamed_chunks_before_eos": str(pending.streamed_chunks_before_eos),
        }
        manifest_filename = self._manifest_path(pending.base_path, self._tp_rank)
        manifest_future = self._executor.submit(
            self._seal_manifest,
            list(request_futures),
            pending.token_ids.clone(),
            manifest_metadata,
            manifest_filename,
            lock_fd,
        )
        manifest_key = f"{pending.req_id}:manifest"
        self._futures[manifest_key] = manifest_future
        self._manifest_futures[pending.req_id] = manifest_future
        request_futures.append(manifest_future)
        manifest_future.add_done_callback(partial(self._write_done, manifest_key))
        self._copy_events[pending.req_id] = copied

    def _submit_host_slot_save(self, pending: _PendingSave) -> None:
        pool = self._get_host_pool()
        if pool is None:
            raise RuntimeError("shared Host KV slot descriptor received without a configured pool")
        if not 0 <= pending.start <= pending.end <= pool.token_capacity:
            raise RuntimeError("StreamOPD KV chunk is outside its shared Host slot")
        chunk_tokens = pending.end - pending.start
        if chunk_tokens > self._chunk_size:
            raise RuntimeError("StreamOPD KV chunk exceeds the fixed Host staging capacity")
        slot = pool.validate_writer(
            pending.base_path,
            request_id=pending.req_id,
            trajectory_id=pending.trajectory_id,
            policy_version=pending.policy_version,
        )
        request_futures = self._request_futures.setdefault(pending.req_id, [])
        copied = None
        if chunk_tokens:
            staging_index = self._staging_available.get()
            copy_stream = self._get_copy_stream()
            ready = self._device.Event()
            ready.record()
            copy_stream.wait_event(ready)
            try:
                with self._device.stream(copy_stream):
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
                copied = self._device.Event()
                copied.record(copy_stream)
            except BaseException:
                self._staging_available.put(staging_index)
                raise
            chunk_future = self._executor.submit(
                self._commit_host_chunk,
                pool,
                slot,
                pending.start,
                pending.end,
                staging_index,
                copied,
            )
            future_key = f"{pending.req_id}:{pending.chunk_index}"
            self._futures[future_key] = chunk_future
            request_futures.append(chunk_future)
            chunk_future.add_done_callback(partial(self._write_done, future_key))

        if not pending.terminal:
            return
        if pending.token_ids is None or pending.token_ids.numel() != pending.end:
            raise RuntimeError("terminal StreamOPD KV chunk must carry the complete token identity")
        seal_future = self._executor.submit(self._seal_host_slot, list(request_futures), pool, pending)
        manifest_key = f"{pending.req_id}:manifest"
        self._futures[manifest_key] = seal_future
        self._manifest_futures[pending.req_id] = seal_future
        request_futures.append(seal_future)
        seal_future.add_done_callback(partial(self._write_done, manifest_key))
        if copied is not None:
            self._copy_events[pending.req_id] = copied

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if isinstance(metadata, StreamOPDKVConnectorMetadata):
                for pending in metadata.pending_saves:
                    self._submit_save(pending)
        newly_finished = finished_req_ids & self._claimed_requests
        self._finished_requests.update(newly_finished)
        done: set[str] = set()
        manifest_futures = getattr(self, "_manifest_futures", {})
        for req_id in list(self._finished_requests):
            # The D2H event only means that staging is ready. Keep vLLM pages
            # owned until the terminal control record is sealed after every
            # shared-slot commit (or the legacy manifest write) completes.
            manifest_future = manifest_futures.get(req_id)
            # The final cohort request has no later model step on which to poll
            # the transfer.  Waiting on the manifest future here is bounded by
            # the writer pool and keeps the scheduler-side ownership contract
            # intact while allowing the tensor copies themselves to remain
            # asynchronous.
            if manifest_future is None:
                # Keep compatibility with lightweight connector fixtures and
                # with snapshots produced by older workers that only expose a
                # CUDA completion event.
                event = self._copy_events.get(req_id)
                if event is None or not event.query():
                    continue
            elif not manifest_future.done():
                if req_id not in newly_finished:
                    continue
                # There may be no model step after the last request in a
                # cohort. Complete the terminal seal here so vLLM can reclaim
                # its pages immediately.
                manifest_future.result()
            if manifest_future is not None:
                manifest_future.result()
            self._copy_events.pop(req_id, None)
            self._finished_requests.remove(req_id)
            self._claimed_requests.discard(req_id)
            done.add(req_id)
            if manifest_future is not None:
                manifest_futures.pop(req_id, None)
            fd = self._lock_fds.pop(req_id, None)
            if fd is not None:
                os.close(fd)
        return done or None, None

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        if num_external_tokens != 0:
            raise RuntimeError("StreamOPDKVConnector is store-only")

    def update_connector_output(self, connector_output: Any) -> None:
        # Scheduler-side ownership ends only after the worker has made the
        # terminal manifest durable and reports finished_sending.
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
        while state.published_tokens + self._chunk_size <= committed_end and not (
            terminal and state.published_tokens + self._chunk_size == committed_end
        ):
            start = state.published_tokens
            end = start + self._chunk_size
            self._pending.append(
                _PendingSave(
                    req_id=state.req_id,
                    trajectory_id=state.trajectory_id,
                    base_path=state.base_path,
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
                    base_path=state.base_path,
                    block_ids_by_group=tuple(list(group) for group in state.block_ids_by_group),
                    policy_version=state.policy_version,
                    prompt_length=state.prompt_length,
                    start=start,
                    end=committed_end,
                    chunk_index=state.next_chunk_index,
                    terminal=True,
                    token_ids=token_ids,
                    streamed_tokens_before_eos=state.published_tokens,
                    streamed_chunks_before_eos=state.next_chunk_index,
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
                    base_path=state.base_path,
                    block_ids_by_group=tuple(list(group) for group in state.block_ids_by_group),
                    policy_version=state.policy_version,
                    prompt_length=state.prompt_length,
                    start=committed_end,
                    end=committed_end,
                    chunk_index=state.next_chunk_index,
                    terminal=True,
                    token_ids=token_ids,
                    streamed_tokens_before_eos=state.published_tokens,
                    streamed_chunks_before_eos=state.next_chunk_index,
                )
            )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        if scheduler_output.preempted_req_ids:
            streamed = scheduler_output.preempted_req_ids & self._scheduler_states.keys()
            if streamed:
                raise RuntimeError(f"StreamOPD KV streaming does not support preemption: {sorted(streamed)}")
        new_request_paths: dict[str, str] = {}
        for request in scheduler_output.scheduled_new_reqs:
            extra_args = request.sampling_params.extra_args if request.sampling_params else None
            params = (extra_args or {}).get("kv_transfer_params") or {}
            if not params.get("streamopd_kv", False):
                continue
            pool = self._get_host_pool()
            if pool is None:
                safe_req_id = request.req_id.replace(os.sep, "_")
                base_path = os.path.join(self._storage_path, safe_req_id)
            else:
                base_path = pool.acquire(
                    request_id=request.req_id,
                    trajectory_id=str(params["trajectory_id"]),
                    policy_version=int(params["policy_version"]),
                    prompt_length=int(params["prompt_length"]),
                )
            self._scheduler_paths[request.req_id] = base_path
            self._scheduler_states[request.req_id] = _SchedulerSaveState(
                req_id=request.req_id,
                trajectory_id=str(params["trajectory_id"]),
                base_path=base_path,
                block_ids_by_group=[list(group) for group in request.block_ids],
                policy_version=int(params["policy_version"]),
                prompt_length=int(params["prompt_length"]),
            )
            new_request_paths[request.req_id] = base_path
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_blocks, num_computed in zip(
            cached.req_ids, cached.new_block_ids, cached.num_computed_tokens, strict=True
        ):
            state = self._scheduler_states.get(req_id)
            if state is None:
                continue
            self._update_block_ids(state, new_blocks, replace=req_id in cached.resumed_req_ids)
            self._queue_committed(state, int(num_computed))
        metadata = StreamOPDKVConnectorMetadata(pending_saves=list(self._pending), new_request_paths=new_request_paths)
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
        base_path = self._scheduler_paths.pop(request_id, None)
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
        if base_path is None:
            base_path = state.base_path
        state.block_ids_by_group = [list(group) for group in block_ids]
        token_ids = torch.tensor(list(request.all_token_ids)[:-1], dtype=torch.long)
        streamed_tokens_before_eos = state.published_tokens
        streamed_chunks_before_eos = state.next_chunk_index
        self._queue_committed(state, token_ids.numel(), terminal=True, token_ids=token_ids)
        self._claimed_requests.add(request.request_id)
        return True, {
            "streamopd_kv_path": base_path,
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
        for future in list(self._futures.values()):
            future.result()
        self._executor.shutdown(wait=True)
        if self._host_pool is not None:
            self._host_pool.close()
            self._host_pool = None
