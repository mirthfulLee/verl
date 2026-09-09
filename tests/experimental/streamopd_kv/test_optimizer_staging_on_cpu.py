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

from verl.experimental.streamopd_kv import fsdp_worker, ray_worker


def test_backward_budget_excludes_only_deferred_optimizer_state():
    model = torch.nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    parameter_bytes = model.weight.numel() * model.weight.element_size()
    for materialized in (False, True):
        if materialized:
            model(torch.ones(1, 4)).sum().backward()
            optimizer.step()
        assert fsdp_worker._deferred_training_state_bytes(model, optimizer) == 3 * parameter_bytes
        assert fsdp_worker._deferred_training_state_bytes(model, optimizer, include_optimizer=False) == parameter_bytes


def test_optimizer_peak_limits_slots_even_when_backward_fits(monkeypatch):
    monkeypatch.setattr(
        fsdp_worker,
        "_reverse_memory_estimate",
        lambda model, trajectory_count, **kwargs: (100 * trajectory_count, 2 * trajectory_count),
    )
    options = dict(
        configured_batch_size=4,
        token_capacity=64,
        max_batch_tokens=256,
        max_chunk_size=64,
        min_chunk_size=16,
        page_size=16,
        dtype=torch.bfloat16,
        available_memory_bytes=10000,
        reserve_bytes=1000,
    )
    ordinary = fsdp_worker._fixed_reverse_slot_plan(None, **options)
    staged = fsdp_worker._fixed_reverse_slot_plan(None, optimizer_reserve_bytes=9700, **options)
    assert (ordinary.batch_size, ordinary.chunk_size) == (4, 64)
    assert (staged.batch_size, staged.chunk_size) == (2, 64)
    assert staged.slot_bytes + 9700 <= 10000
    with pytest.raises(RuntimeError, match="optimizer_reserve"):
        fsdp_worker._fixed_reverse_slot_plan(None, optimizer_reserve_bytes=10000, **options)


def test_shared_pool_restores_parameters_without_optimizer(monkeypatch):
    events = []
    monkeypatch.setattr(ray_worker, "aggressive_empty_cache", lambda **kwargs: events.append("clear"))
    monkeypatch.setattr(ray_worker, "get_device_name", lambda: "cuda")
    worker = SimpleNamespace(
        actor=SimpleNamespace(
            engine=SimpleNamespace(to=lambda **kwargs: events.append(kwargs)),
            allocate_reverse_slots=lambda: events.append("slots"),
        )
    )
    ray_worker.StreamOPDActorWorker.load_streamopd_trainer_state(worker)
    assert events == ["clear", dict(device="cuda", model=True, optimizer=False, grad=True), "slots"]
