# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""vLLM store-only connector for completed rollout KV snapshots."""

from __future__ import annotations

import fcntl
import json
import os
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

from .snapshot_io import extract_vllm_nhd_tokens

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
    base_path: str
    token_ids: torch.Tensor
    block_ids_by_group: tuple[list[int], ...]
    policy_version: int
    prompt_length: int


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
        self._scheduler_paths: dict[str, str] = {}
        self._pending: dict[str, _PendingSave] = {}

        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._layer_groups: dict[str, int] = {}
        self._tp_rank = 0
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._device = get_torch_device()
        self._copy_stream: Any = None
        self._executor = ThreadPoolExecutor(
            max_workers=int(self._kv_transfer_config.get_from_extra_config("streamopd_writer_threads", 4)),
            thread_name_prefix="streamopd-kv-save",
        )
        self._lock_fds: dict[str, int] = {}
        self._copy_events: dict[str, Any] = {}
        self._futures: dict[str, Future] = {}
        self._finished_requests: set[str] = set()
        self._claimed_requests: set[str] = set()

    @staticmethod
    def _rank_path(base_path: str, tp_rank: int) -> str:
        return f"{base_path}.tp{tp_rank}.safetensors"

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
            if req_id in self._lock_fds:
                continue
            filename = self._rank_path(base_path, self._tp_rank)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fd = os.open(filename + ".lock", os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._lock_fds[req_id] = fd

    @staticmethod
    def _write_snapshot(
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, str],
        event: Any,
        filename: str,
        lock_fd: int | None,
    ) -> None:
        try:
            event.synchronize()
            save_file(tensors, filename, metadata=metadata)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)

    def _write_done(self, req_id: str, future: Future) -> None:
        self._futures.pop(req_id, None)
        if exception := future.exception():
            logger.error("StreamOPD KV write failed for %s: %r", req_id, exception)

    def _submit_save(self, pending: _PendingSave) -> None:
        # Scheduler and worker connectors are separate objects. Ownership is
        # claimed again on the worker when the save metadata arrives.
        self._claimed_requests.add(pending.req_id)
        copy_stream = self._get_copy_stream()
        ready = self._device.Event()
        ready.record()
        copy_stream.wait_event(ready)
        tensors: dict[str, torch.Tensor] = {"token_ids": pending.token_ids.clone()}
        num_tokens = pending.token_ids.shape[0]
        with self._device.stream(copy_stream):
            for layer_idx, layer_name in enumerate(self._layer_names):
                group_idx = self._layer_groups[layer_name]
                extracted = extract_vllm_nhd_tokens(
                    self._kv_caches[layer_name],
                    pending.block_ids_by_group[group_idx],
                    self._block_size,
                    num_tokens,
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
        filename = self._rank_path(pending.base_path, self._tp_rank)
        metadata = {
            "format": "verl-streamopd-kv-v1",
            "request_id": pending.req_id,
            "policy_version": str(pending.policy_version),
            "prompt_length": str(pending.prompt_length),
            "tp_rank": str(self._tp_rank),
            "tp_size": str(self._tp_size),
            "page_size": str(self._block_size),
            "axis_order": "token_kv_head_dim",
            "rope_convention": "post_rope_key",
            "layer_names": json.dumps(self._layer_names),
        }
        lock_fd = self._lock_fds.pop(pending.req_id, None)
        if lock_fd is None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            lock_fd = os.open(filename + ".lock", os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        future = self._executor.submit(self._write_snapshot, tensors, metadata, copied, filename, lock_fd)
        self._copy_events[pending.req_id] = copied
        self._futures[pending.req_id] = future
        future.add_done_callback(partial(self._write_done, pending.req_id))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if isinstance(metadata, StreamOPDKVConnectorMetadata):
                for pending in metadata.pending_saves:
                    self._submit_save(pending)
        newly_finished = finished_req_ids & self._claimed_requests
        self._finished_requests.update(newly_finished)
        done: set[str] = set()
        for req_id in list(self._finished_requests):
            event = self._copy_events.get(req_id)
            # The final cohort request has no later model step on which to poll
            # the event. Synchronize only that request's D2H copy in the empty
            # cleanup step; snapshot serialization remains asynchronous.
            if req_id in newly_finished and event is not None and not event.query():
                event.synchronize()
            if event is not None and event.query():
                self._copy_events.pop(req_id, None)
                self._finished_requests.remove(req_id)
                self._claimed_requests.discard(req_id)
                done.add(req_id)
                fd = self._lock_fds.pop(req_id, None)
                if fd is not None:
                    os.close(fd)
        return done or None, None

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        if num_external_tokens != 0:
            raise RuntimeError("StreamOPDKVConnector is store-only")

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        metadata = StreamOPDKVConnectorMetadata(pending_saves=list(self._pending.values()))
        self._pending.clear()
        for request in scheduler_output.scheduled_new_reqs:
            extra_args = request.sampling_params.extra_args if request.sampling_params else None
            params = (extra_args or {}).get("kv_transfer_params") or {}
            if not params.get("streamopd_kv", False):
                continue
            safe_req_id = request.req_id.replace(os.sep, "_")
            base_path = os.path.join(self._storage_path, safe_req_id)
            self._scheduler_paths[request.req_id] = base_path
            metadata.new_request_paths[request.req_id] = base_path
        return metadata

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params or {}
        if not params.get("streamopd_kv", False):
            self._scheduler_paths.pop(request.request_id, None)
            return False, None
        if str(request.status) in {"FINISHED_ABORTED", "FINISHED_ERROR", "FINISHED_IGNORED"}:
            self._scheduler_paths.pop(request.request_id, None)
            fd = self._lock_fds.pop(request.request_id, None)
            if fd is not None:
                os.close(fd)
            return False, None
        base_path = self._scheduler_paths.pop(request.request_id)
        token_ids = torch.tensor(list(request.all_token_ids)[:-1], dtype=torch.long)
        pending = _PendingSave(
            req_id=request.request_id,
            base_path=base_path,
            token_ids=token_ids,
            block_ids_by_group=tuple(list(group) for group in block_ids),
            policy_version=int(params["policy_version"]),
            prompt_length=int(params["prompt_length"]),
        )
        self._pending[request.request_id] = pending
        self._claimed_requests.add(request.request_id)
        return True, {
            "streamopd_kv_path": base_path,
            "streamopd_kv_tp_size": self._tp_size,
            "streamopd_kv_policy_version": pending.policy_version,
            "streamopd_kv_num_tokens": token_ids.numel(),
        }

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "NHD"

    def shutdown(self) -> None:
        for future in list(self._futures.values()):
            future.result()
        self._executor.shutdown(wait=True)
