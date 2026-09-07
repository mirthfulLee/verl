# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest

from verl.experimental.streamopd_cf.config import OPDBatchingConfig
from verl.experimental.streamopd_cf.memory import choose_micro_batch_size, estimate_training_workspace


def test_auto_microbatch_respects_memory_and_per_rank_batch():
    args = dict(available_bytes=1000, fixed_bytes=100, per_sample_bytes=100, batch_cap=64)
    assert choose_micro_batch_size(**args) == 8
    assert choose_micro_batch_size(**(args | {"batch_cap": 3})) == 2
    assert choose_micro_batch_size(**(args | {"available_bytes": 500})) == 4
    with pytest.raises(RuntimeError, match="cannot fit one trajectory"):
        choose_micro_batch_size(**(args | {"available_bytes": 199}))


def test_memory_estimate_accounts_for_length_and_checkpointed_prefixes():
    config = SimpleNamespace(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_key_value_heads=8,
        num_attention_heads=16,
        head_dim=128,
        vocab_size=151936,
    )
    args = dict(sequence_length=4096, chunk_size=1024, loss_chunk_size=512, dtype_bytes=2, checkpointing=True)
    fixed, per_sample = estimate_training_workspace(config, **args)
    long_fixed, long_sample = estimate_training_workspace(config, **(args | {"sequence_length": 8192}))
    _, no_checkpoint = estimate_training_workspace(config, **(args | {"checkpointing": False}))
    assert long_fixed == fixed
    assert long_sample > per_sample
    assert no_checkpoint > 2 * per_sample
    # Typical 80-GiB case after parameters, future optimizer/grads and reserves.
    assert (
        choose_micro_batch_size(
            available_bytes=44 * 1024**3, fixed_bytes=fixed, per_sample_bytes=per_sample, batch_cap=64
        )
        == 16
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory_fraction": 0},
        {"memory_fraction": 1.1},
        {"reserve_gib": -1},
    ],
)
def test_invalid_microbatch_planner_config(kwargs):
    with pytest.raises(ValueError):
        OPDBatchingConfig(**kwargs)
