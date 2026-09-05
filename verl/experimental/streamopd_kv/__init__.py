# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Experimental strict on-policy distillation with streamed rollout KV."""

from importlib import import_module

_EXPORTS = {
    "CommittedChunkPublisher": ("publisher", "CommittedChunkPublisher"),
    "CommittedTokenChunk": ("protocol", "CommittedTokenChunk"),
    "Qwen3ReverseTrainer": ("qwen3", "Qwen3ReverseTrainer"),
    "ReverseTrainingResult": ("qwen3", "ReverseTrainingResult"),
    "StreamingTeacherCoordinator": ("streaming_teacher", "StreamingTeacherCoordinator"),
    "TrajectoryKey": ("protocol", "TrajectoryKey"),
    "prepare_streamopd_kv_config": ("config", "prepare_streamopd_kv_config"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
