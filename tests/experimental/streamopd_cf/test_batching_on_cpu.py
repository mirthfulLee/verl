# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.experimental.streamopd_cf import batching
from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import prepare_micro_batches


def batch(lengths, budget):
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor(
                [torch.ones(n, dtype=torch.long) for n in lengths], layout=torch.jagged
            )
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=True, max_token_len_per_gpu=budget)
    return data


def test_cf_refines_native_packing_to_bound_padding(monkeypatch):
    monkeypatch.setattr(batching, "get_device_id", lambda: "cpu")
    data = batch([12, 11, 4, 3, 2, 1], 20)
    native, _ = prepare_micro_batches(data)
    padded, indices = batching.prepare_padded_micro_batches(data, None)
    assert any(len(part) * int(batching.sequence_lengths(part).max()) > 20 for part in native)
    assert all(len(part) * int(batching.sequence_lengths(part).max()) <= 20 for part in padded)
    assert sorted(index for group in indices for index in group) == list(range(6))


@pytest.mark.parametrize("dynamic,fixed,budget", [(True, None, 20), (False, 2, 0)])
def test_manual_training_limits_are_preserved(dynamic, fixed, budget):
    worker = batching.AutoBatchTrainingWorker.__new__(batching.AutoBatchTrainingWorker)
    worker.engine_config = SimpleNamespace(
        use_dynamic_bsz=dynamic, micro_batch_size_per_gpu=fixed, max_token_len_per_gpu=budget
    )
    data = batch([8, 8, 4, 4], 99)
    worker.configure_training_batch(data)
    groups, _ = prepare_micro_batches(data)
    assert tu.get_non_tensor_data(data, "max_token_len_per_gpu", None) == budget
    if dynamic:
        assert all(int(batching.sequence_lengths(part).sum()) <= budget for part in groups)
    else:
        assert [len(part) for part in groups] == [2, 2]
