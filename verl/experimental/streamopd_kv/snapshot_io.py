# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import fcntl
import glob
import json
import os
import time
from collections.abc import Sequence

import torch
from safetensors import safe_open

from .protocol import KVLayout, SealedKVSnapshot, TrajectoryKey


def extract_vllm_nhd_tokens(
    kv_cache: torch.Tensor,
    block_ids: Sequence[int],
    block_size: int,
    num_tokens: int,
    *,
    kv_axis: int | None = None,
) -> torch.Tensor:
    """Gather logical tokens from either generation of vLLM's NHD cache."""

    if kv_cache.ndim != 5 or kv_cache.shape[2] != block_size:
        raise ValueError(f"expected a 5-D vLLM NHD KV cache with block size {block_size}, got {tuple(kv_cache.shape)}")
    if kv_axis is None:
        candidates = [axis for axis in (0, 1) if kv_cache.shape[axis] == 2]
        if len(candidates) != 1:
            raise ValueError(f"cannot infer the K/V axis for vLLM cache shape {tuple(kv_cache.shape)}")
        kv_axis = candidates[0]
    if kv_axis not in (0, 1) or kv_cache.shape[kv_axis] != 2:
        raise ValueError(f"invalid K/V axis {kv_axis} for vLLM cache shape {tuple(kv_cache.shape)}")
    if num_tokens < 0 or num_tokens > len(block_ids) * block_size:
        raise ValueError("requested token count is outside the supplied KV blocks")
    blocks = torch.as_tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    if kv_axis == 0:
        selected = kv_cache.index_select(1, blocks)
        logical = selected.permute(1, 2, 0, 3, 4)
    else:
        selected = kv_cache.index_select(0, blocks)
        logical = selected.permute(0, 2, 1, 3, 4)
    return logical.flatten(0, 1)[:num_tokens].contiguous()


def extract_vllm_nhd_token_range(
    kv_cache: torch.Tensor,
    block_ids: Sequence[int],
    block_size: int,
    start: int,
    end: int,
) -> torch.Tensor:
    """Gather only the logical KV blocks intersecting ``[start, end)``."""

    capacity = len(block_ids) * block_size
    if not 0 <= start < end <= capacity:
        raise ValueError(f"invalid KV token range [{start}, {end}) for capacity {capacity}")
    first_block = start // block_size
    last_block = (end + block_size - 1) // block_size
    selected_blocks = block_ids[first_block:last_block]
    logical = extract_vllm_nhd_tokens(
        kv_cache,
        selected_blocks,
        block_size,
        len(selected_blocks) * block_size,
    )
    offset = start - first_block * block_size
    return logical[offset : offset + end - start].contiguous()


def _load_streamed_vllm_snapshot(
    base_path: str,
    *,
    key: TrajectoryKey,
    tp_rank: int,
    expected_tp_size: int,
    expected_token_ids: Sequence[int] | None,
    expected_prompt_length: int | None,
    device: torch.device | str,
    started: float,
) -> SealedKVSnapshot:
    manifest = f"{base_path}.tp{tp_rank}.manifest.safetensors"
    with open(manifest + ".lock") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_SH)
        with safe_open(manifest, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if metadata.get("format") != "verl-streamopd-kv-v2":
                raise RuntimeError(f"unsupported streamed StreamOPD KV format in {manifest}")
            if int(metadata["policy_version"]) != key.policy_version:
                raise RuntimeError("rollout KV policy version does not match the training cohort")
            if metadata["trajectory_id"] != key.trajectory_id:
                raise RuntimeError("rollout KV request identity does not match the training trajectory")
            if int(metadata["tp_rank"]) != tp_rank or int(metadata["tp_size"]) != expected_tp_size:
                raise RuntimeError("rollout/trainer TP layout mismatch")
            prompt_length = int(metadata["prompt_length"])
            if expected_prompt_length is not None and prompt_length != expected_prompt_length:
                raise RuntimeError("rollout KV prompt boundary does not match the training trajectory")
            token_ids_tensor = handle.get_tensor("token_ids")
            token_ids = tuple(int(value) for value in token_ids_tensor.tolist())
            if expected_token_ids is not None and token_ids != tuple(expected_token_ids):
                raise RuntimeError("rollout KV token identity does not match the training trajectory")
            layer_names = json.loads(metadata["layer_names"])
            num_chunks = int(metadata["num_chunks"])
            page_size = int(metadata["page_size"])
            axis_order = metadata["axis_order"]
            rope_convention = metadata["rope_convention"]

        layers: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        expected_start = 0
        for chunk_index in range(num_chunks):
            chunk_path = f"{base_path}.tp{tp_rank}.chunk{chunk_index:05d}.safetensors"
            with safe_open(chunk_path, framework="pt", device=str(device)) as chunk:
                chunk_metadata = chunk.metadata()
                if chunk_metadata.get("format") != "verl-streamopd-kv-v2-chunk":
                    raise RuntimeError(f"invalid StreamOPD KV chunk format in {chunk_path}")
                if int(chunk_metadata["chunk_index"]) != chunk_index:
                    raise RuntimeError("StreamOPD KV chunk index is not contiguous")
                start = int(chunk_metadata["start"])
                end = int(chunk_metadata["end"])
                if start != expected_start or not start < end <= len(token_ids):
                    raise RuntimeError(f"invalid streamed KV extent [{start}, {end}) at chunk {chunk_index}")
                tensor_names = sorted(name for name in chunk.keys() if name.startswith("layer_"))
                if len(tensor_names) != len(layer_names):
                    raise RuntimeError("streamed KV layer count does not match the manifest")
                if layers is None:
                    layers = []
                    for name in tensor_names:
                        packed = chunk.get_tensor(name)
                        if packed.ndim != 4 or packed.shape[1] != 2:
                            raise RuntimeError(f"invalid packed KV shape for {name}: {tuple(packed.shape)}")
                        shape = (1, packed.shape[2], len(token_ids), packed.shape[3])
                        layers.append(
                            (
                                torch.empty(shape, dtype=packed.dtype, device=device),
                                torch.empty(shape, dtype=packed.dtype, device=device),
                            )
                        )
                assert layers is not None
                for layer_idx, name in enumerate(tensor_names):
                    packed = chunk.get_tensor(name)
                    if packed.shape[0] != end - start:
                        raise RuntimeError("streamed KV chunk tensor length does not match its extent")
                    layers[layer_idx][0][:, :, start:end].copy_(packed[:, 0].transpose(0, 1).unsqueeze(0))
                    layers[layer_idx][1][:, :, start:end].copy_(packed[:, 1].transpose(0, 1).unsqueeze(0))
                expected_start = end
        if layers is None or expected_start != len(token_ids):
            raise RuntimeError("streamed KV chunks do not cover the sealed trajectory")
        layout = KVLayout(
            num_layers=len(layers),
            num_kv_heads=layers[0][0].shape[1],
            head_dim=layers[0][0].shape[-1],
            dtype=str(layers[0][0].dtype).removeprefix("torch."),
            page_size=page_size,
            tp_size=expected_tp_size,
            tp_rank=tp_rank,
            axis_order=axis_order,
            rope_convention=rope_convention,
        )
        return SealedKVSnapshot(
            key=key,
            token_ids=token_ids,
            prompt_length=prompt_length,
            layout=layout,
            layers=tuple(layers),
            source="vllm-stream-v2",
            handoff_seconds=time.perf_counter() - started,
            streamed_tokens_before_eos=int(metadata.get("streamed_tokens_before_eos", 0)),
            streamed_chunks_before_eos=int(metadata.get("streamed_chunks_before_eos", 0)),
        )


def load_vllm_snapshot(
    base_path: str,
    *,
    key: TrajectoryKey,
    tp_rank: int,
    expected_tp_size: int,
    expected_token_ids: Sequence[int] | None = None,
    expected_prompt_length: int | None = None,
    device: torch.device | str = "cpu",
) -> SealedKVSnapshot:
    """Acquire one TP-aligned vLLM shard after its async handoff completes."""

    started = time.perf_counter()
    manifest_lock = f"{base_path}.tp{tp_rank}.manifest.safetensors.lock"
    if os.path.exists(manifest_lock):
        return _load_streamed_vllm_snapshot(
            base_path,
            key=key,
            tp_rank=tp_rank,
            expected_tp_size=expected_tp_size,
            expected_token_ids=expected_token_ids,
            expected_prompt_length=expected_prompt_length,
            device=device,
            started=started,
        )
    filename = f"{base_path}.tp{tp_rank}.safetensors"
    lock_path = filename + ".lock"
    with open(lock_path) as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_SH)
        with safe_open(filename, framework="pt", device=str(device)) as handle:
            metadata = handle.metadata()
            if metadata.get("format") != "verl-streamopd-kv-v1":
                raise RuntimeError(f"unsupported StreamOPD KV snapshot format in {filename}")
            if int(metadata["policy_version"]) != key.policy_version:
                raise RuntimeError("rollout KV policy version does not match the training cohort")
            trajectory_id = metadata.get("trajectory_id", metadata["request_id"])
            if trajectory_id != key.trajectory_id:
                raise RuntimeError("rollout KV request identity does not match the training trajectory")
            if int(metadata["tp_rank"]) != tp_rank or int(metadata["tp_size"]) != expected_tp_size:
                raise RuntimeError("rollout/trainer TP layout mismatch")
            prompt_length = int(metadata["prompt_length"])
            if expected_prompt_length is not None and prompt_length != expected_prompt_length:
                raise RuntimeError("rollout KV prompt boundary does not match the training trajectory")
            token_ids_tensor = handle.get_tensor("token_ids").cpu()
            token_ids = tuple(int(value) for value in token_ids_tensor.tolist())
            if expected_token_ids is not None and token_ids != tuple(expected_token_ids):
                raise RuntimeError("rollout KV token identity does not match the training trajectory")
            tensor_names = sorted(name for name in handle.keys() if name.startswith("layer_"))
            layers = []
            for name in tensor_names:
                packed = handle.get_tensor(name)
                if packed.ndim != 4 or packed.shape[1] != 2:
                    raise RuntimeError(f"invalid packed KV shape for {name}: {tuple(packed.shape)}")
                layers.append(
                    (
                        packed[:, 0].transpose(0, 1).unsqueeze(0).contiguous(),
                        packed[:, 1].transpose(0, 1).unsqueeze(0).contiguous(),
                    )
                )
            if len(json.loads(metadata["layer_names"])) != len(layers):
                raise RuntimeError("KV layer metadata does not match the stored tensors")
            if not layers:
                raise RuntimeError("KV snapshot contains no attention layers")
            layout = KVLayout(
                num_layers=len(layers),
                num_kv_heads=layers[0][0].shape[1],
                head_dim=layers[0][0].shape[-1],
                dtype=str(layers[0][0].dtype).removeprefix("torch."),
                page_size=int(metadata["page_size"]),
                tp_size=expected_tp_size,
                tp_rank=tp_rank,
                axis_order=metadata["axis_order"],
                rope_convention=metadata["rope_convention"],
            )
            snapshot = SealedKVSnapshot(
                key=key,
                token_ids=token_ids,
                prompt_length=prompt_length,
                layout=layout,
                layers=tuple(layers),
                source="vllm-v0.24",
                handoff_seconds=time.perf_counter() - started,
            )
    return snapshot


def cleanup_vllm_snapshot(base_path: str, tp_size: int) -> None:
    for rank in range(tp_size):
        filename = f"{base_path}.tp{rank}.safetensors"
        streamed = glob.glob(f"{base_path}.tp{rank}.chunk*.safetensors")
        manifest = f"{base_path}.tp{rank}.manifest.safetensors"
        for path in (filename, filename + ".lock", manifest, manifest + ".lock", *streamed):
            if os.path.exists(path):
                os.remove(path)
