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


def checkpoint_weight_bytes(model_path: str) -> int:
    """Read local safetensors payload size without loading model tensors."""

    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        total_size = int(json.loads(index.read_text()).get("metadata", {}).get("total_size", 0))
        if total_size > 0:
            return total_size
    shards = list(root.glob("*.safetensors"))
    if shards:
        return sum(path.stat().st_size for path in shards)
    raise ValueError(f"cannot determine safetensors weight size for model: {model_path}")


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


def _stable_concurrency(capacity: int) -> int:
    if capacity < 8:
        return max(0, capacity)
    return 1 << (capacity.bit_length() - 1)


def plan_vllm_memory(
    *,
    total_memory_bytes: int,
    weight_bytes: int,
    kv_bytes_per_token: int,
    requested_max_num_seqs: int,
    max_model_len: int,
    utilization_limit: float,
    max_num_seqs_explicit: bool,
    utilization_explicit: bool,
) -> dict[str, float]:
    """Jointly solve vLLM memory and non-preemptible request concurrency."""

    if min(total_memory_bytes, weight_bytes, kv_bytes_per_token, requested_max_num_seqs, max_model_len) < 1:
        raise ValueError("vLLM memory planning inputs must be positive")
    if not 0 < utilization_limit <= 1:
        raise ValueError("vLLM gpu_memory_utilization must be in (0, 1]")
    runtime_reserve = max(2 * 1024**3, weight_bytes * 3 // 20)
    kv_budget = math.floor(utilization_limit * total_memory_bytes / 1.15) - weight_bytes - runtime_reserve
    capacity = max(0, kv_budget // (kv_bytes_per_token * max_model_len))
    if max_num_seqs_explicit and requested_max_num_seqs > capacity:
        raise ValueError(
            "vLLM memory cap cannot hold the explicitly requested non-preemptible KV: "
            f"max_num_seqs={requested_max_num_seqs}, capacity={capacity}, "
            f"gpu_memory_utilization={utilization_limit:.2f}"
        )
    max_num_seqs = min(requested_max_num_seqs, capacity)
    if not max_num_seqs_explicit:
        max_num_seqs = _stable_concurrency(max_num_seqs)
    if max_num_seqs < 1:
        raise ValueError("vLLM memory cap cannot fit one non-preemptible trajectory")

    kv_bytes = kv_bytes_per_token * max_num_seqs * max_model_len
    required_bytes = math.ceil((weight_bytes + kv_bytes + runtime_reserve) * 1.15)
    required_utilization = max(0.20, math.ceil(required_bytes / total_memory_bytes * 20) / 20)
    selected_utilization = utilization_limit if utilization_explicit else required_utilization
    if selected_utilization > utilization_limit:
        raise ValueError(
            "vLLM memory cap cannot hold model plus non-preemptible KV: "
            f"required={selected_utilization:.2f}, configured={utilization_limit:.2f}"
        )
    return {
        "gpu_memory_utilization": selected_utilization,
        "weight_gib": weight_bytes / (1024**3),
        "kv_gib": kv_bytes / (1024**3),
        "runtime_reserve_gib": runtime_reserve / (1024**3),
        "required_gib": required_bytes / (1024**3),
        "max_num_seqs": float(max_num_seqs),
        "capacity_at_limit": float(capacity),
        "max_model_len": float(max_model_len),
        "utilization_limit": utilization_limit,
        "max_num_seqs_explicit": float(max_num_seqs_explicit),
        "utilization_explicit": float(utilization_explicit),
    }


def _mappings(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _mappings(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            yield from _mappings(nested)


def minimum_device_total_bytes(value: Any) -> int:
    totals = [int(item["total_bytes"]) for item in _mappings(value) if int(item.get("total_bytes", 0)) > 0]
    if not totals:
        raise RuntimeError("Trainer workers returned no device memory capacity")
    return min(totals)


def shared_vllm_utilization_limit(
    value: Any,
    *,
    rank_offset: int,
    world_size: int,
    required_free_bytes: int,
) -> dict[str, float]:
    """Cap shared vLLM allocation while preserving frozen Trainer workspace."""

    rows = [
        item for item in _mappings(value) if int(item.get("free_bytes", 0)) > 0 and int(item.get("total_bytes", 0)) > 0
    ]
    selected = rows[rank_offset : rank_offset + world_size]
    if len(selected) != world_size:
        raise RuntimeError(
            "Trainer workers returned incomplete shared vLLM memory stats: "
            f"offset={rank_offset}, world_size={world_size}, rows={len(rows)}"
        )
    free_bytes = min(int(row["free_bytes"]) for row in selected)
    total_bytes = min(int(row["total_bytes"]) for row in selected)
    utilization_limit = math.floor((free_bytes - required_free_bytes) / total_bytes * 20) / 20
    if utilization_limit < 0.20:
        raise ValueError(
            "Trainer-shared vLLM has insufficient memory after reverse preflight: "
            f"free={free_bytes / 1024**3:.2f} GiB, "
            f"reverse_reserve={required_free_bytes / 1024**3:.2f} GiB"
        )
    return {
        "utilization_limit": utilization_limit,
        "free_gib": free_bytes / (1024**3),
        "total_gib": total_bytes / (1024**3),
        "reverse_reserve_gib": required_free_bytes / (1024**3),
    }


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


def planned_reverse_width(plan_result: Any, fallback: int) -> int:
    widths = [
        int(item["slot_batch_size"]) for item in _mappings(plan_result) if int(item.get("slot_batch_size", 0)) > 0
    ]
    return min(widths, default=fallback)


def planned_reverse_required_free_gib(plan_result: Any) -> float:
    required = [
        float(item["runtime_required_free_gib"])
        for item in _mappings(plan_result)
        if "runtime_required_free_gib" in item
    ]
    return max(required, default=0.0)


def plan_teacher_admission(
    *,
    expected_trajectories: int,
    trajectory_tokens: int,
    vllm_capacity_tokens: int,
    page_size: int,
    max_batched_tokens: int,
    initial_chunk_tokens: int,
    train_launch_width: int,
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
        train_launch_width,
    )
    if any(value < 1 for value in values) or trajectory_cap < 0 or token_cap < 0:
        raise ValueError("StreamOPD Teacher admission inputs must be positive and caps non-negative")
    trajectory_tokens = math.ceil(trajectory_tokens / page_size) * page_size
    if token_cap and token_cap < trajectory_tokens:
        raise ValueError("StreamOPD Teacher token cap cannot fit one trajectory reservation")
    safe_capacity = max(trajectory_tokens, (vllm_capacity_tokens * 3 // 4 // page_size) * page_size)
    if token_cap:
        safe_capacity = min(safe_capacity, token_cap)
    capacity_width = max(1, safe_capacity // trajectory_tokens)
    prefill_wave = max(1, max_batched_tokens // min(initial_chunk_tokens, trajectory_tokens))
    width = min(expected_trajectories, capacity_width, 2 * prefill_wave)
    if trajectory_cap:
        width = min(width, trajectory_cap)
    if width >= train_launch_width:
        width = max(train_launch_width, width // train_launch_width * train_launch_width)
    return {
        "active_trajectories": width,
        "active_kv_tokens": width * trajectory_tokens,
        "vllm_capacity_tokens": vllm_capacity_tokens,
        "safe_capacity_tokens": safe_capacity,
        "trajectory_tokens": trajectory_tokens,
        "prefill_wave": prefill_wave,
    }
