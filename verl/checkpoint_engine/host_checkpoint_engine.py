# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.utils.device import get_device_id
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_BUCKET_FORMAT = "verl-host-checkpoint-mmap-v1"


@dataclass(frozen=True)
class HostCheckpointMetadata:
    session_dir: str | None


@CheckpointEngineRegistry.register("host")
class HostCheckpointEngine(CheckpointEngine):
    """Publish checkpoint buckets through a node-local shared-memory directory.

    Actor rank 0 materializes and writes full parameters. The remaining actor
    ranks consume the same FSDP generator and stop at every bucket boundary so
    rank 0 cannot advance to the next parameter gather before the bucket is
    durable. Rollout workers independently read the immutable bucket files.
    """

    _SESSION_PREFIX = "verl-host-checkpoint-"

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        directory: str = "/dev/shm/verl-checkpoint",
        poll_interval: float = 0.01,
        timeout: float = 600.0,
        rollout_dtype: str | torch.dtype | None = None,
    ) -> None:
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}")
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be positive, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        self.bucket_size = bucket_size
        self.is_master = is_master
        self.directory = Path(directory).expanduser().resolve()
        self.poll_interval = poll_interval
        self.timeout = timeout
        if isinstance(rollout_dtype, str):
            dtype_name = rollout_dtype.removeprefix("torch.")
            rollout_dtype = getattr(torch, dtype_name, None)
            if not isinstance(rollout_dtype, torch.dtype):
                raise ValueError(f"unsupported host checkpoint rollout_dtype: {dtype_name}")
        self.rollout_dtype = rollout_dtype
        self.role: str | None = None
        self.session_dir: Path | None = None
        self.actor_world_size = 0

    def prepare(self) -> HostCheckpointMetadata:
        if not self.is_master:
            return HostCheckpointMetadata(session_dir=None)

        self.directory.mkdir(parents=True, exist_ok=True)
        session_dir = self.directory / f"{self._SESSION_PREFIX}{uuid.uuid4().hex}"
        session_dir.mkdir()
        self.session_dir = session_dir
        return HostCheckpointMetadata(session_dir=str(session_dir))

    @classmethod
    def build_topology(
        cls,
        actor_wg_world_size: int,
        rollout_world_size: int,
        metadata: list[HostCheckpointMetadata],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        if len(metadata) != actor_wg_world_size + rollout_world_size:
            raise ValueError(
                "host checkpoint metadata count does not match actor and rollout world sizes: "
                f"{len(metadata)} != {actor_wg_world_size} + {rollout_world_size}"
            )
        session_dir = metadata[0].session_dir
        if session_dir is None:
            raise ValueError("actor rank 0 must create the host checkpoint session")

        actor_roles = ["sender", *(["participant"] * (actor_wg_world_size - 1))]
        actor_kwargs = {
            "role": actor_roles,
            "session_dir": [session_dir] * actor_wg_world_size,
            "actor_world_size": [actor_wg_world_size] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "role": ["receiver"] * rollout_world_size,
            "session_dir": [session_dir] * rollout_world_size,
            "actor_world_size": [actor_wg_world_size] * rollout_world_size,
        }
        return actor_kwargs, rollout_kwargs

    def init_process_group(self, role: str, session_dir: str, actor_world_size: int) -> None:
        if role not in {"sender", "participant", "receiver"}:
            raise ValueError(f"invalid host checkpoint role: {role}")
        resolved_session = Path(session_dir).resolve()
        if resolved_session.parent != self.directory or not resolved_session.name.startswith(self._SESSION_PREFIX):
            raise ValueError(f"host checkpoint session is outside the configured directory: {resolved_session}")

        self.role = role
        self.session_dir = resolved_session
        self.actor_world_size = actor_world_size

    def _actor_barrier(self) -> None:
        if self.actor_world_size > 1:
            if not torch.distributed.is_initialized():
                raise RuntimeError("actor process group is not initialized")
            torch.distributed.barrier()

    def _bucket_metadata_path(self, bucket_index: int) -> Path:
        if self.session_dir is None:
            raise RuntimeError("host checkpoint session is not initialized")
        return self.session_dir / f"bucket-{bucket_index:06d}.meta.pt"

    def _bucket_data_path(self, bucket_index: int) -> Path:
        if self.session_dir is None:
            raise RuntimeError("host checkpoint session is not initialized")
        return self.session_dir / f"bucket-{bucket_index:06d}.bin"

    def _write_bucket(
        self,
        bucket_index: int,
        bucket: torch.Tensor,
        length: int,
        bucket_meta: dict[str, TensorMeta],
        is_last: bool,
    ) -> None:
        data_path = self._bucket_data_path(bucket_index)
        metadata_path = self._bucket_metadata_path(bucket_index)
        temporary_metadata_path = metadata_path.with_suffix(".tmp")
        fd = os.open(data_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, length)
        finally:
            os.close(fd)
        if length:
            mapped = torch.from_file(str(data_path), shared=True, size=length, dtype=torch.uint8)
            mapped.copy_(bucket[:length])
        # The small atomic metadata file is the publication record. Receivers
        # never observe a bucket until its raw shared mapping is complete.
        torch.save(
            {
                "format": _BUCKET_FORMAT,
                "bucket_meta": bucket_meta,
                "is_last": is_last,
                "length": length,
            },
            temporary_metadata_path,
        )
        os.replace(temporary_metadata_path, metadata_path)

    async def _materialize_weights(self, weights):
        """Gather FSDP1 shards only for this backend's single-sender protocol.

        Participants drive the same gathers and bucket barriers, but need only
        tensor metadata. Other transports retain the engine's original output.
        """
        async for name, param in ensure_async_iterator(weights):
            if isinstance(param, ShardedTensor):
                full = (
                    torch.empty(param.size(), dtype=param.dtype, device=get_device_id())
                    if self.role == "sender"
                    else None
                )
                param.gather(dst=0, out=full)
                param = full if full is not None else torch.empty(param.size(), dtype=param.dtype, device="meta")
            yield name, param

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> dict[str, float]:
        del global_steps
        weights = self._materialize_weights(weights)
        if self.role == "participant":
            await self._participate(weights)
            return {}
        if self.role != "sender":
            raise RuntimeError(f"host checkpoint role {self.role!r} cannot send weights")

        start_time = time.perf_counter()
        bucket: torch.Tensor | None = None
        bucket_meta: dict[str, TensorMeta] = {}
        bucket_index = 0
        offset = 0
        contains_cuda_copy = False
        total_bytes = 0

        async for tensor_meta, chunk in split_weight_chunks(
            weights,
            self.bucket_size,
            floating_dtype=self.rollout_dtype,
        ):
            assert chunk is not None
            if bucket is None:
                try:
                    bucket = torch.empty(
                        self.bucket_size,
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=chunk.is_cuda,
                    )
                except RuntimeError:
                    logger.warning("Pinned checkpoint bucket allocation failed; using pageable host memory")
                    bucket = torch.empty(self.bucket_size, dtype=torch.uint8, device="cpu")
            if offset + tensor_meta.chunk_size > self.bucket_size:
                if contains_cuda_copy:
                    torch.cuda.synchronize()
                self._write_bucket(bucket_index, bucket, offset, bucket_meta, is_last=False)
                self._actor_barrier()
                total_bytes += offset
                bucket_index += 1
                bucket_meta = {}
                offset = 0
                contains_cuda_copy = False

            if tensor_meta.name in bucket_meta:
                raise RuntimeError(f"duplicate tensor chunk {tensor_meta.name!r} in checkpoint bucket")
            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            bucket[offset : offset + tensor_meta.chunk_size].copy_(chunk, non_blocking=chunk.is_cuda)
            contains_cuda_copy |= chunk.is_cuda
            offset += tensor_meta.chunk_size

        if bucket is None:
            bucket = torch.empty(0, dtype=torch.uint8)
        if contains_cuda_copy:
            torch.cuda.synchronize()
        self._write_bucket(bucket_index, bucket, offset, bucket_meta, is_last=True)
        self._actor_barrier()
        total_bytes += offset

        elapsed = time.perf_counter() - start_time
        gib_per_second = total_bytes / max(elapsed, 1e-9) / (1024**3)
        logger.info(
            "Host checkpoint published %d buckets (%.2f GiB) in %.2fs (%.2f GiB/s)",
            bucket_index + 1,
            total_bytes / (1024**3),
            elapsed,
            gib_per_second,
        )
        return {
            "timing/checkpoint_host_seconds": elapsed,
            "checkpoint/host_gib_per_second": gib_per_second,
        }

    async def _participate(self, weights: Generator[tuple[str, torch.Tensor], None, None]) -> None:
        offset = 0
        async for tensor_meta, _ in split_weight_chunks(
            weights,
            self.bucket_size,
            meta_only=True,
            floating_dtype=self.rollout_dtype,
        ):
            if offset + tensor_meta.chunk_size > self.bucket_size:
                self._actor_barrier()
                offset = 0
            offset += tensor_meta.chunk_size
        self._actor_barrier()

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        del global_steps
        if self.role != "receiver":
            raise RuntimeError(f"host checkpoint role {self.role!r} cannot receive weights")
        async for name, weight in merge_weight_chunks(self._receive_chunks(), self.bucket_size):
            yield name, weight

    async def _receive_chunks(self) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        deadline = time.monotonic() + self.timeout
        bucket_index = 0
        while True:
            metadata_path = self._bucket_metadata_path(bucket_index)
            while not metadata_path.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for host checkpoint bucket {bucket_index}: {metadata_path}")
                await asyncio.sleep(self.poll_interval)

            metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
            if metadata.get("format") != _BUCKET_FORMAT:
                raise RuntimeError(f"unsupported host checkpoint bucket metadata: {metadata_path}")
            length = int(metadata["length"])
            data_path = self._bucket_data_path(bucket_index)
            if length < 0 or not data_path.is_file() or data_path.stat().st_size != length:
                raise RuntimeError(f"invalid host checkpoint bucket data: {data_path}")
            buffer = (
                torch.from_file(str(data_path), shared=True, size=length, dtype=torch.uint8)
                if length
                else torch.empty(0, dtype=torch.uint8)
            )
            for tensor_meta in metadata["bucket_meta"].values():
                start = tensor_meta.offset
                yield tensor_meta, buffer[start : start + tensor_meta.chunk_size]
            if metadata["is_last"]:
                return
            bucket_index += 1

    def finalize(self) -> None:
        if self.role == "sender" and self.session_dir is not None and self.session_dir.exists():
            if self.session_dir.parent != self.directory or not self.session_dir.name.startswith(self._SESSION_PREFIX):
                raise RuntimeError(f"refusing to remove invalid host checkpoint session: {self.session_dir}")
            shutil.rmtree(self.session_dir)
        self.role = None
        self.session_dir = None
        self.actor_world_size = 0
