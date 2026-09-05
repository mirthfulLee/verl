# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations


def test_dapo_adapter_wraps_plain_prompt_as_chat_messages(monkeypatch) -> None:
    from examples.on_policy_distillation_trainer.dapo_math_dataset import DAPOMathDataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    dataset = DAPOMathDataset.__new__(DAPOMathDataset)
    assert dataset._build_messages({"prompt": "2 + 2?"}, "prompt") == [{"role": "user", "content": "2 + 2?"}]
    from benchmarks.streamopd_kv.dapo_math_dataset import DAPOMathDataset as BenchmarkDataset

    monkeypatch.setattr(RLHFDataset, "__getitem__", lambda _self, item: {"item": item})
    monkeypatch.setenv("STREAMOPD_RAGGED_RESPONSE_LENGTHS", "256,768")
    assert "max_response_tokens" not in dataset[100]
    dataset = BenchmarkDataset.__new__(BenchmarkDataset)
    assert dataset[100]["max_response_tokens"] == 256
    assert dataset[7]["max_response_tokens"] == 768
