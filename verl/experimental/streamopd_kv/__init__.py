# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Correctness-first primitives for two-pool StreamOPD-KV.

The package is experimental.  Importing it does not import vLLM or
Transformers; backend-specific modules are loaded only when selected.
"""

from .attention import BatchedReverseChunkState, LayerKVTrace, ReverseChunkState, exact_causal_attention
from .config import prepare_streamopd_kv_config
from .protocol import (
    CommittedTokenChunk,
    KVLayout,
    KVSnapshotStore,
    PolicyVersionBarrier,
    SealedKVSnapshot,
    SnapshotState,
    TeacherArtifactBuffer,
    TrajectoryKey,
)
from .publisher import CommittedChunkPublisher
from .qwen3 import Qwen3ReverseTrainer, ReverseTrainingResult, capture_qwen3_kv_trace, use_qwen3_reverse_attention
from .snapshot_io import (
    cleanup_vllm_snapshot,
    extract_vllm_nhd_token_range,
    extract_vllm_nhd_tokens,
    load_vllm_snapshot,
    move_vllm_snapshot,
)
from .streaming_teacher import StreamingTeacherCoordinator

__all__ = [
    "CommittedChunkPublisher",
    "CommittedTokenChunk",
    "BatchedReverseChunkState",
    "KVLayout",
    "KVSnapshotStore",
    "LayerKVTrace",
    "PolicyVersionBarrier",
    "Qwen3ReverseTrainer",
    "ReverseChunkState",
    "ReverseTrainingResult",
    "SealedKVSnapshot",
    "SnapshotState",
    "StreamingTeacherCoordinator",
    "TeacherArtifactBuffer",
    "TrajectoryKey",
    "capture_qwen3_kv_trace",
    "cleanup_vllm_snapshot",
    "exact_causal_attention",
    "extract_vllm_nhd_token_range",
    "extract_vllm_nhd_tokens",
    "load_vllm_snapshot",
    "move_vllm_snapshot",
    "prepare_streamopd_kv_config",
    "use_qwen3_reverse_attention",
]
