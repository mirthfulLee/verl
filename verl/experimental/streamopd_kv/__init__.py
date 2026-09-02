# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Experimental strict on-policy distillation with streamed rollout KV."""

from .config import prepare_streamopd_kv_config
from .protocol import CommittedTokenChunk, TrajectoryKey
from .publisher import CommittedChunkPublisher
from .qwen3 import Qwen3ReverseTrainer, ReverseTrainingResult
from .streaming_teacher import StreamingTeacherCoordinator

__all__ = [
    "CommittedChunkPublisher",
    "CommittedTokenChunk",
    "Qwen3ReverseTrainer",
    "ReverseTrainingResult",
    "StreamingTeacherCoordinator",
    "TrajectoryKey",
    "prepare_streamopd_kv_config",
]
