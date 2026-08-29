# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True, order=True)
class TrajectoryKey:
    """Identity shared by rollout KV, teacher artifacts, and backward."""

    policy_version: int
    trajectory_id: str

    def __post_init__(self) -> None:
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")


@dataclass(frozen=True)
class CommittedTokenChunk:
    """An immutable, accepted completion-token interval."""

    key: TrajectoryKey
    start: int
    token_ids: tuple[int, ...]
    terminal: bool = False
    prompt_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("chunk start must be non-negative")
        if not self.token_ids and not self.terminal:
            raise ValueError("only a terminal chunk may be empty")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token ids must be non-negative")

    @property
    def end(self) -> int:
        return self.start + len(self.token_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.key.policy_version,
            "trajectory_id": self.key.trajectory_id,
            "start": self.start,
            "token_ids": list(self.token_ids),
            "terminal": self.terminal,
            "prompt_ids": list(self.prompt_ids),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CommittedTokenChunk:
        return cls(
            key=TrajectoryKey(int(value["policy_version"]), str(value["trajectory_id"])),
            start=int(value["start"]),
            token_ids=tuple(int(token_id) for token_id in value["token_ids"]),
            terminal=bool(value.get("terminal", False)),
            prompt_ids=tuple(int(token_id) for token_id in value.get("prompt_ids", ())),
        )


@dataclass(frozen=True)
class KVLayout:
    """The physical and positional contract of a rollout KV shard."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    page_size: int
    tp_size: int = 1
    tp_rank: int = 0
    axis_order: str = "token_kv_head_dim"
    rope_convention: str = "post_rope_key"
    position_offset: int = 0

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_kv_heads", "head_dim", "page_size", "tp_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if self.axis_order != "token_kv_head_dim":
            raise ValueError(f"unsupported KV axis order: {self.axis_order}")
        if self.rope_convention != "post_rope_key":
            raise ValueError(f"unsupported RoPE convention: {self.rope_convention}")


class SnapshotState(str, Enum):
    SEALED = "sealed"
    LEASED = "leased"
    RELEASED = "released"
    INVALIDATED = "invalidated"


@dataclass
class SealedKVSnapshot:
    """Read-only ownership boundary for one TP shard of a completed request.

    Layer tensors use ``[batch, kv_heads, tokens, head_dim]``.  The serving
    engine has already applied RoPE to keys.  Tensors are detached when sealed;
    callers must not mutate them.
    """

    key: TrajectoryKey
    token_ids: tuple[int, ...]
    prompt_length: int
    layout: KVLayout
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    source: str = "unknown"
    handoff_seconds: float = 0.0
    _state: SnapshotState = field(default=SnapshotState.SEALED, init=False, repr=False)
    _refcount: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.layers) != self.layout.num_layers:
            raise ValueError(f"expected {self.layout.num_layers} KV layers, got {len(self.layers)}")
        if not 0 <= self.prompt_length <= len(self.token_ids):
            raise ValueError("prompt_length is outside the sealed token range")
        expected = (1, self.layout.num_kv_heads, len(self.token_ids), self.layout.head_dim)
        detached_layers = []
        for layer_idx, (key, value) in enumerate(self.layers):
            if key.shape != value.shape or tuple(key.shape) != expected:
                raise ValueError(
                    f"layer {layer_idx} KV shape mismatch: expected {expected}, "
                    f"got {tuple(key.shape)}/{tuple(value.shape)}"
                )
            if key.requires_grad or value.requires_grad:
                raise ValueError("a sealed rollout KV snapshot must not carry an autograd graph")
            detached_layers.append((key.detach(), value.detach()))
        self.layers = tuple(detached_layers)

    @property
    def state(self) -> SnapshotState:
        return self._state

    @property
    def refcount(self) -> int:
        return self._refcount

    @property
    def token_digest(self) -> str:
        payload = ",".join(str(token_id) for token_id in self.token_ids).encode("ascii")
        return hashlib.blake2b(payload, digest_size=16, person=b"StreamOPD-KV").hexdigest()

    def acquire(self, policy_version: int) -> SealedKVSnapshot:
        with self._lock:
            if policy_version != self.key.policy_version:
                raise RuntimeError(
                    f"KV policy version mismatch: snapshot={self.key.policy_version}, requested={policy_version}"
                )
            if self._state in (SnapshotState.RELEASED, SnapshotState.INVALIDATED):
                raise RuntimeError(f"cannot acquire a {self._state.value} KV snapshot")
            self._refcount += 1
            self._state = SnapshotState.LEASED
        return self

    def release(self) -> None:
        with self._lock:
            if self._refcount < 1:
                raise RuntimeError("KV snapshot release without a matching acquire")
            self._refcount -= 1
            if self._refcount == 0:
                self._state = SnapshotState.RELEASED
                self.layers = ()

    def invalidate(self) -> None:
        with self._lock:
            if self._refcount:
                raise RuntimeError("cannot invalidate a leased KV snapshot")
            self._state = SnapshotState.INVALIDATED
            self.layers = ()


class KVSnapshotStore:
    """Version-aware snapshot ownership registry."""

    def __init__(self) -> None:
        self._snapshots: dict[TrajectoryKey, SealedKVSnapshot] = {}
        self._lock = threading.Lock()

    def seal(self, snapshot: SealedKVSnapshot) -> None:
        with self._lock:
            if snapshot.key in self._snapshots:
                raise RuntimeError(f"duplicate KV snapshot for {snapshot.key}")
            if snapshot.state != SnapshotState.SEALED:
                raise RuntimeError("only a newly sealed snapshot can enter the store")
            self._snapshots[snapshot.key] = snapshot

    def acquire(self, key: TrajectoryKey) -> SealedKVSnapshot:
        with self._lock:
            try:
                snapshot = self._snapshots[key]
            except KeyError as exc:
                raise KeyError(f"no sealed KV snapshot for {key}") from exc
        return snapshot.acquire(key.policy_version)

    def invalidate_version(self, policy_version: int) -> int:
        with self._lock:
            selected = [key for key in self._snapshots if key.policy_version == policy_version]
            for key in selected:
                snapshot = self._snapshots[key]
                snapshot.invalidate()
                del self._snapshots[key]
        return len(selected)


@dataclass
class _TeacherTrajectory:
    next_start: int = 0
    terminal_end: int | None = None
    ids: list[torch.Tensor] = field(default_factory=list)
    logprobs: list[torch.Tensor] = field(default_factory=list)


class TeacherArtifactBuffer:
    """Forward-arrival, reverse-consumption teacher artifact buffer."""

    def __init__(self) -> None:
        self._items: dict[TrajectoryKey, _TeacherTrajectory] = {}

    def append(
        self,
        key: TrajectoryKey,
        start: int,
        teacher_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        *,
        terminal: bool = False,
    ) -> None:
        if teacher_ids.ndim < 1 or teacher_logprobs.ndim < 1:
            raise ValueError("teacher artifacts must have a token dimension")
        if teacher_ids.shape[0] != teacher_logprobs.shape[0]:
            raise ValueError("teacher id/logprob token dimensions differ")
        item = self._items.setdefault(key, _TeacherTrajectory())
        if start != item.next_start:
            raise RuntimeError(f"non-contiguous teacher coverage for {key}: expected {item.next_start}, got {start}")
        item.ids.append(teacher_ids.detach().cpu())
        item.logprobs.append(teacher_logprobs.detach().cpu())
        item.next_start += teacher_ids.shape[0]
        if terminal:
            item.terminal_end = item.next_start

    def is_complete(self, key: TrajectoryKey, required_tokens: int) -> bool:
        item = self._items.get(key)
        return item is not None and item.terminal_end == required_tokens

    def mark_terminal(self, key: TrajectoryKey, required_tokens: int) -> None:
        item = self._items.get(key)
        if item is None or item.next_start != required_tokens:
            covered = None if item is None else item.next_start
            raise RuntimeError(
                f"cannot close teacher artifacts for {key}: covered={covered}, required={required_tokens}"
            )
        item.terminal_end = required_tokens

    def materialize(self, key: TrajectoryKey, required_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_complete(key, required_tokens):
            raise RuntimeError(f"teacher coverage is incomplete for {key}")
        item = self._items[key]
        return torch.cat(item.ids, dim=0), torch.cat(item.logprobs, dim=0)

    def consume_reverse(self, key: TrajectoryKey, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self._items.get(key)
        if item is None or item.terminal_end is None:
            raise RuntimeError(f"teacher artifacts are not complete for {key}")
        if not 0 <= start < end <= item.terminal_end:
            raise ValueError(f"invalid teacher range [{start}, {end})")
        ids = torch.cat(item.ids, dim=0)
        logprobs = torch.cat(item.logprobs, dim=0)
        return ids[start:end], logprobs[start:end]

    def invalidate_version(self, policy_version: int) -> int:
        keys = [key for key in self._items if key.policy_version == policy_version]
        for key in keys:
            del self._items[key]
        return len(keys)


class PolicyVersionBarrier:
    """Fail-closed cohort barrier around one atomic optimizer update."""

    def __init__(self, policy_version: int, trajectory_ids: Iterable[str], valid_token_count: int) -> None:
        self.policy_version = policy_version
        self.pending = {TrajectoryKey(policy_version, trajectory_id) for trajectory_id in trajectory_ids}
        if not self.pending:
            raise ValueError("a policy cohort must contain at least one trajectory")
        if valid_token_count < 1:
            raise ValueError("valid_token_count must be positive")
        self.valid_token_count = valid_token_count
        self._stepped = False

    def mark_backward_complete(self, key: TrajectoryKey) -> None:
        if key.policy_version != self.policy_version:
            raise RuntimeError("cannot mix policy versions in one optimizer barrier")
        if key not in self.pending:
            raise RuntimeError(f"trajectory {key} is not pending in this cohort")
        self.pending.remove(key)

    @property
    def ready(self) -> bool:
        return not self.pending and not self._stepped

    def step(
        self,
        parameters: Iterable[torch.nn.Parameter],
        optimizer_step: Callable[[], Any],
        *,
        clip_grad_norm: float | None = None,
    ) -> Any:
        if not self.ready:
            raise RuntimeError(f"policy version {self.policy_version} is not ready; pending={len(self.pending)}")
        parameters = list(parameters)
        scale = 1.0 / self.valid_token_count
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
        result = optimizer_step()
        self._stepped = True
        return result
