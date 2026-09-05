# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def kv_bytes_per_token(model_path: str, dtype: str, tensor_parallel_size: int = 1) -> int:
    """Return one rank's causal KV token footprint from Hugging Face config."""

    config = json.loads((Path(model_path) / "config.json").read_text())
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    local_kv_heads = max(1, math.ceil(kv_heads / tensor_parallel_size))
    hidden_size = int(config["hidden_size"])
    query_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", hidden_size // query_heads))
    dtype_bytes = 4 if str(dtype).lower() in {"float32", "fp32"} else 2
    return layers * local_kv_heads * head_dim * 2 * dtype_bytes


def _mappings(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _mappings(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            yield from _mappings(nested)


def plan_host_kv(
    *,
    handoff_dir: str,
    global_batch_size: int,
    max_model_len: int,
    kv_bytes_per_token: int,
) -> dict[str, float]:
    """Fail before rollout if the Host backing cannot hold worst-case KV."""

    path = Path(handoff_dir)
    parent = next((candidate for candidate in (path, *path.parents) if candidate.exists()), None)
    if parent is None:
        raise ValueError(f"StreamOPD KV handoff path has no existing parent: {handoff_dir}")
    free_bytes = shutil.disk_usage(parent).free
    required_bytes = global_batch_size * max_model_len * kv_bytes_per_token
    reserve_bytes = max(4 * 1024**3, required_bytes // 10)
    if required_bytes + reserve_bytes > free_bytes:
        raise ValueError(
            "StreamOPD Host KV backing is too small for the configured batch and token limit: "
            f"required={(required_bytes + reserve_bytes) / 1024**3:.2f} GiB, "
            f"free={free_bytes / 1024**3:.2f} GiB, path={handoff_dir}"
        )
    return {
        "host_kv_required_gib": required_bytes / (1024**3),
        "host_kv_reserve_gib": reserve_bytes / (1024**3),
        "host_kv_free_gib": free_bytes / (1024**3),
    }


def partition_training_units(train_batch_size: int, unit_size: int) -> list[int]:
    if train_batch_size < 1 or unit_size < 1:
        raise ValueError("StreamOPD training batch and unit sizes must be positive")
    return [min(unit_size, train_batch_size - start) for start in range(0, train_batch_size, unit_size)]


def plan_training_unit_size(
    *,
    train_batch_size: int,
    reverse_wave_size: int,
    resources_overlap: bool,
    kv_prefetch_depth: int,
) -> int:
    """Choose a controller unit that contains enough reverse waves to pipeline copies."""

    if min(train_batch_size, reverse_wave_size) < 1 or kv_prefetch_depth < 0:
        raise ValueError("StreamOPD training-unit inputs must be positive and prefetch depth non-negative")
    if resources_overlap:
        # Shared inference has already entered level-2 sleep. Splitting the
        # batch cannot expose additional overlap and only defeats slot reuse.
        return train_batch_size
    # A unit needs the current wave plus every configured prefetched wave.
    # Two waves are the minimum that can overlap next-group transfer with the
    # current backward pass; waiting for additional ready waves only delays
    # the first dedicated-pool update.
    waves_per_unit = max(2, kv_prefetch_depth + 1)
    return min(train_batch_size, reverse_wave_size * waves_per_unit)


def planned_reverse_width(plan_result: Any, fallback: int) -> int:
    widths = [
        int(item["slot_batch_size"]) for item in _mappings(plan_result) if int(item.get("slot_batch_size", 0)) > 0
    ]
    return min(widths, default=fallback)


def plan_teacher_admission(
    *,
    expected_trajectories: int,
    trajectory_tokens: int,
    vllm_capacity_tokens: int,
    page_size: int,
    max_batched_tokens: int,
    initial_chunk_tokens: int,
    teacher_replicas: int = 1,
    trajectory_cap: int = 0,
    token_cap: int = 0,
) -> dict[str, int]:
    """Choose one stable Teacher session cohort before policy version zero."""

    values = (
        expected_trajectories,
        trajectory_tokens,
        vllm_capacity_tokens,
        page_size,
        max_batched_tokens,
        initial_chunk_tokens,
        teacher_replicas,
    )
    if any(value < 1 for value in values) or trajectory_cap < 0 or token_cap < 0:
        raise ValueError("StreamOPD Teacher admission inputs must be positive and caps non-negative")
    trajectory_tokens = math.ceil(trajectory_tokens / page_size) * page_size
    if token_cap and token_cap < trajectory_tokens:
        raise ValueError("StreamOPD Teacher token cap cannot fit one trajectory reservation")
    safe_capacity = vllm_capacity_tokens // page_size * page_size
    if token_cap:
        safe_capacity = min(safe_capacity, token_cap // page_size * page_size)
    if safe_capacity < trajectory_tokens:
        raise ValueError("StreamOPD Teacher vLLM capacity cannot fit one trajectory reservation")
    capacity_width = max(1, safe_capacity // trajectory_tokens)
    prefill_wave_per_replica = max(1, max_batched_tokens // min(initial_chunk_tokens, trajectory_tokens))
    prefill_wave = teacher_replicas * prefill_wave_per_replica
    # vLLM already batches live requests according to max_batched_tokens.
    # Restricting live sessions to a small number of prefill waves creates
    # head-of-line blocking when the first admitted trajectories are long.
    width = min(expected_trajectories, capacity_width)
    if trajectory_cap:
        width = min(width, trajectory_cap)
    return {
        "active_trajectories": width,
        "active_kv_tokens": width * trajectory_tokens,
        "vllm_capacity_tokens": vllm_capacity_tokens,
        "safe_capacity_tokens": safe_capacity,
        "trajectory_tokens": trajectory_tokens,
        "teacher_replicas": teacher_replicas,
        "prefill_wave_per_replica": prefill_wave_per_replica,
        "prefill_wave": prefill_wave,
    }
