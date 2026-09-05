# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import fcntl
import hashlib
import json
import mmap
import os
import re
import struct
import time
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch

_FORMAT = "verl-streamopd-host-kv-v1"
_SLOT_PATH = re.compile(r"^(?P<root>.+\.tp\d+)\.slot(?P<slot>\d+)\.g(?P<generation>\d+)$")
_CONTROL_RECORD = struct.Struct("<QB7xq16s16s16sQQQQ")
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HostSlotState(IntEnum):
    FREE = 0
    WRITING = 1
    SEALED = 2


def _identity_digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def _token_digest(token_ids: Any) -> bytes:
    tensor = torch.as_tensor(token_ids, dtype=torch.int64, device="cpu").contiguous()
    return hashlib.blake2b(tensor.numpy().tobytes(), digest_size=16).digest()


class HostKVSlotPool:
    """Fixed POSIX mmap backing shared by rollout writers and Trainer readers."""

    def __init__(self, root: str, descriptor: dict[str, Any]) -> None:
        self.root = root
        self.slot_count = int(descriptor["slot_count"])
        self.token_capacity = int(descriptor["token_capacity"])
        self.num_layers = int(descriptor["num_layers"])
        self.num_kv_heads = int(descriptor["num_kv_heads"])
        self.head_dim = int(descriptor["head_dim"])
        self.page_size = int(descriptor["page_size"])
        self.dtype_name = str(descriptor["dtype"])
        try:
            self.dtype = _DTYPES[self.dtype_name]
        except KeyError as error:
            raise ValueError(f"unsupported shared Host KV dtype: {self.dtype_name}") from error
        self.element_size = torch.empty((), dtype=self.dtype).element_size()
        self.layer_elements = self.token_capacity * self.num_kv_heads * self.head_dim
        self.slot_elements = self.num_layers * 2 * self.layer_elements
        self.data_bytes = self.slot_count * self.slot_elements * self.element_size
        self.control_bytes = self.slot_count * _CONTROL_RECORD.size

        self._lock_fd = os.open(self.lock_path(root), os.O_CREAT | os.O_RDWR, 0o600)
        self._data_fd = os.open(self.data_path(root), os.O_RDWR)
        self._control_fd = os.open(self.control_path(root), os.O_RDWR)
        self._data = mmap.mmap(self._data_fd, self.data_bytes, access=mmap.ACCESS_WRITE)
        self._control = mmap.mmap(self._control_fd, self.control_bytes, access=mmap.ACCESS_WRITE)

    @classmethod
    def create_or_open(
        cls,
        storage_path: str,
        *,
        tp_rank: int,
        slot_count: int,
        token_capacity: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
    ) -> HostKVSlotPool:
        values = (slot_count, token_capacity, num_layers, num_kv_heads, head_dim, page_size)
        if any(value < 1 for value in values):
            raise ValueError("shared Host KV slot dimensions must be positive")
        dtype_name = str(dtype).removeprefix("torch.")
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported shared Host KV dtype: {dtype_name}")
        os.makedirs(storage_path, exist_ok=True)
        root = os.path.join(storage_path, f"host_kv_pool.tp{tp_rank}")
        descriptor = {
            "format": _FORMAT,
            "slot_count": int(slot_count),
            "token_capacity": int(token_capacity),
            "num_layers": int(num_layers),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
            "page_size": int(page_size),
            "dtype": dtype_name,
        }
        lock_fd = os.open(cls.lock_path(root), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            descriptor_path = cls.descriptor_path(root)
            if os.path.exists(descriptor_path):
                existing = json.loads(Path(descriptor_path).read_text())
                if existing != descriptor:
                    raise RuntimeError(
                        "shared Host KV pool geometry changed within one run: "
                        f"existing={existing}, requested={descriptor}"
                    )
            else:
                element_size = torch.empty((), dtype=dtype).element_size()
                data_bytes = slot_count * num_layers * 2 * token_capacity * num_kv_heads * head_dim * element_size
                control_bytes = slot_count * _CONTROL_RECORD.size
                cls._resize_file(cls.data_path(root), data_bytes)
                cls._resize_file(cls.control_path(root), control_bytes)
                Path(descriptor_path).write_text(json.dumps(descriptor, sort_keys=True))
        finally:
            os.close(lock_fd)
        return cls(root, descriptor)

    @classmethod
    def open_for_slot(cls, slot_path: str) -> HostKVSlotPool:
        root, _, _ = cls.parse_slot_path(slot_path)
        return cls._open_root(root)

    @classmethod
    def open_existing(cls, storage_path: str, *, tp_rank: int) -> HostKVSlotPool:
        return cls._open_root(os.path.join(storage_path, f"host_kv_pool.tp{tp_rank}"))

    @classmethod
    def _open_root(cls, root: str) -> HostKVSlotPool:
        descriptor = json.loads(Path(cls.descriptor_path(root)).read_text())
        if descriptor.get("format") != _FORMAT:
            raise RuntimeError(f"unsupported shared Host KV pool format in {cls.descriptor_path(root)}")
        return cls(root, descriptor)

    @staticmethod
    def _resize_file(path: str, size: int) -> None:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
        finally:
            os.close(fd)

    @staticmethod
    def descriptor_path(root: str) -> str:
        return f"{root}.json"

    @staticmethod
    def data_path(root: str) -> str:
        return f"{root}.data"

    @staticmethod
    def control_path(root: str) -> str:
        return f"{root}.control"

    @staticmethod
    def lock_path(root: str) -> str:
        return f"{root}.lock"

    @staticmethod
    def is_slot_path(path: str) -> bool:
        return _SLOT_PATH.fullmatch(path) is not None

    @staticmethod
    def parse_slot_path(path: str) -> tuple[str, int, int]:
        match = _SLOT_PATH.fullmatch(path)
        if match is None:
            raise ValueError(f"invalid shared Host KV slot path: {path}")
        return match.group("root"), int(match.group("slot")), int(match.group("generation"))

    def format_slot_path(self, slot: int, generation: int) -> str:
        return f"{self.root}.slot{slot:06d}.g{generation:016d}"

    @contextmanager
    def _locked(self):
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _record_offset(self, slot: int) -> int:
        if not 0 <= slot < self.slot_count:
            raise IndexError(f"shared Host KV slot {slot} is outside capacity {self.slot_count}")
        return slot * _CONTROL_RECORD.size

    def _read_record(self, slot: int) -> tuple[Any, ...]:
        offset = self._record_offset(slot)
        return _CONTROL_RECORD.unpack(self._control[offset : offset + _CONTROL_RECORD.size])

    def _write_record(self, slot: int, values: tuple[Any, ...]) -> None:
        offset = self._record_offset(slot)
        self._control[offset : offset + _CONTROL_RECORD.size] = _CONTROL_RECORD.pack(*values)

    def acquire(
        self,
        *,
        request_id: str,
        trajectory_id: str,
        policy_version: int,
        prompt_length: int,
    ) -> str:
        if prompt_length < 1:
            raise ValueError("shared Host KV slot requires a positive prompt length")
        request_hash = _identity_digest(request_id)
        trajectory_hash = _identity_digest(trajectory_id)
        with self._locked():
            for slot in range(self.slot_count):
                generation, state, *_ = self._read_record(slot)
                if state != HostSlotState.FREE:
                    continue
                generation = int(generation) + 1
                self._write_record(
                    slot,
                    (
                        generation,
                        HostSlotState.WRITING,
                        int(policy_version),
                        trajectory_hash,
                        request_hash,
                        bytes(16),
                        int(prompt_length),
                        0,
                        0,
                        0,
                    ),
                )
                return self.format_slot_path(slot, generation)
        raise RuntimeError(f"shared Host KV slot pool is full: capacity={self.slot_count}")

    def _validate_owner(
        self,
        slot_path: str,
        *,
        request_id: str,
        trajectory_id: str,
        policy_version: int,
    ) -> tuple[int, int, tuple[Any, ...]]:
        root, slot, generation = self.parse_slot_path(slot_path)
        if root != self.root:
            raise RuntimeError("shared Host KV slot belongs to another pool")
        record = self._read_record(slot)
        if int(record[0]) != generation:
            raise RuntimeError("stale shared Host KV slot generation")
        if int(record[2]) != int(policy_version):
            raise RuntimeError("shared Host KV policy version mismatch")
        if record[3] != _identity_digest(trajectory_id) or record[4] != _identity_digest(request_id):
            raise RuntimeError("shared Host KV slot identity mismatch")
        return slot, generation, record

    def validate_writer(
        self,
        slot_path: str,
        *,
        request_id: str,
        trajectory_id: str,
        policy_version: int,
    ) -> int:
        with self._locked():
            slot, _, record = self._validate_owner(
                slot_path,
                request_id=request_id,
                trajectory_id=trajectory_id,
                policy_version=policy_version,
            )
            if record[1] != HostSlotState.WRITING:
                raise RuntimeError("shared Host KV slot is not writable")
            return slot

    def seal(
        self,
        slot_path: str,
        *,
        request_id: str,
        trajectory_id: str,
        policy_version: int,
        prompt_length: int,
        token_ids: Any,
        token_count: int,
        streamed_tokens_before_eos: int,
        streamed_chunks_before_eos: int,
    ) -> None:
        if not 0 < token_count <= self.token_capacity:
            raise ValueError("sealed shared Host KV token count exceeds slot capacity")
        with self._locked():
            slot, generation, record = self._validate_owner(
                slot_path,
                request_id=request_id,
                trajectory_id=trajectory_id,
                policy_version=policy_version,
            )
            if record[1] != HostSlotState.WRITING or int(record[6]) != prompt_length:
                raise RuntimeError("shared Host KV slot cannot be sealed from its current state")
            self._write_record(
                slot,
                (
                    generation,
                    HostSlotState.SEALED,
                    int(policy_version),
                    record[3],
                    record[4],
                    _token_digest(token_ids),
                    int(prompt_length),
                    int(token_count),
                    int(streamed_tokens_before_eos),
                    int(streamed_chunks_before_eos),
                ),
            )

    def metadata(
        self,
        slot_path: str,
        *,
        trajectory_id: str,
        policy_version: int,
        prompt_length: int,
        token_ids: Any,
        wait_timeout_seconds: float = 0.0,
        poll_interval_seconds: float = 0.001,
    ) -> dict[str, int]:
        root, slot, generation = self.parse_slot_path(slot_path)
        if root != self.root:
            raise RuntimeError("shared Host KV slot belongs to another pool")
        deadline = time.monotonic() + wait_timeout_seconds
        while True:
            with self._locked():
                record = self._read_record(slot)
                if int(record[0]) != generation:
                    raise RuntimeError("shared Host KV slot generation changed before it was consumed")
                state = HostSlotState(record[1])
                if state == HostSlotState.SEALED:
                    if int(record[2]) != int(policy_version) or record[3] != _identity_digest(trajectory_id):
                        raise RuntimeError("shared Host KV slot identity does not match the training trajectory")
                    if record[5] != _token_digest(token_ids):
                        raise RuntimeError("shared Host KV token identity does not match the training trajectory")
                    if int(record[6]) != prompt_length:
                        raise RuntimeError("shared Host KV prompt boundary does not match the training trajectory")
                    return {
                        "slot": slot,
                        "generation": generation,
                        "prompt_length": int(record[6]),
                        "token_count": int(record[7]),
                        "streamed_tokens_before_eos": int(record[8]),
                        "streamed_chunks_before_eos": int(record[9]),
                    }
                if state != HostSlotState.WRITING:
                    raise RuntimeError("shared Host KV slot is not sealed for this generation")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for shared Host KV slot to seal: {slot_path}")
            time.sleep(poll_interval_seconds)

    def release(self, slot_path: str) -> None:
        root, slot, generation = self.parse_slot_path(slot_path)
        if root != self.root:
            raise RuntimeError("shared Host KV slot belongs to another pool")
        with self._locked():
            record = self._read_record(slot)
            if int(record[0]) != generation:
                raise RuntimeError("refusing to release a stale shared Host KV slot generation")
            if record[1] == HostSlotState.FREE:
                return
            self._write_record(
                slot,
                (generation, HostSlotState.FREE, 0, bytes(16), bytes(16), bytes(16), 0, 0, 0, 0),
            )

    def state_counts(self) -> dict[str, int]:
        counts = {state.name.lower(): 0 for state in HostSlotState}
        with self._locked():
            for slot in range(self.slot_count):
                state = HostSlotState(self._read_record(slot)[1])
                counts[state.name.lower()] += 1
        return counts

    def layer(self, slot: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"shared Host KV layer {layer} is outside capacity {self.num_layers}")
        tensors = []
        for kv_index in range(2):
            element_offset = (slot * self.slot_elements) + (layer * 2 + kv_index) * self.layer_elements
            tensor = torch.frombuffer(
                self._data,
                dtype=self.dtype,
                count=self.layer_elements,
                offset=element_offset * self.element_size,
            ).view(self.token_capacity, self.num_kv_heads, self.head_dim)
            tensors.append(tensor)
        return tensors[0], tensors[1]

    def close(self) -> None:
        self._data.close()
        self._control.close()
        os.close(self._data_fd)
        os.close(self._control_fd)
        os.close(self._lock_fd)


def cleanup_host_kv_pools(storage_path: str) -> int:
    """Unlink fully released fixed pools after the final policy barrier."""

    released_bytes = 0
    for descriptor_path in Path(storage_path).glob("host_kv_pool.tp*.json"):
        root = str(descriptor_path).removesuffix(".json")
        descriptor = json.loads(descriptor_path.read_text())
        pool = HostKVSlotPool(root, descriptor)
        counts = pool.state_counts()
        if counts["writing"] or counts["sealed"]:
            pool.close()
            raise RuntimeError(f"cannot clean active shared Host KV pool {root}: {counts}")
        released_bytes += pool.data_bytes + pool.control_bytes
        pool.close()
        for path in (
            HostKVSlotPool.data_path(root),
            HostKVSlotPool.control_path(root),
            HostKVSlotPool.descriptor_path(root),
            HostKVSlotPool.lock_path(root),
        ):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    return released_bytes
