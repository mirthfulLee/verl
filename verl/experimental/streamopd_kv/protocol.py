# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class TrajectoryKey:
    """Identity shared by rollout KV, Teacher artifacts, and backward."""

    policy_version: int
    trajectory_id: str

    def __post_init__(self) -> None:
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")


@dataclass(frozen=True)
class CommittedTokenChunk:
    """An immutable accepted completion-token interval."""

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
