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

from verl.workers.rollout.vllm_rollout.static_teacher import StaticTeacherWeightCache


class Allocator:
    def __init__(self):
        self.pointer_to_data = {
            1: SimpleNamespace(tag="weights", cpu_backup_tensor=None, value=torch.tensor([3, 8, 11])),
            2: SimpleNamespace(tag="kv_cache", cpu_backup_tensor=None, value=torch.tensor([19])),
        }
        self.copies = 0

    def sleep(self, offload_tags):
        for data in self.pointer_to_data.values():
            if data.tag in offload_tags:
                data.cpu_backup_tensor = data.value.clone()
                self.copies += 1
            data.value.zero_()

    def wake_up(self):
        for data in self.pointer_to_data.values():
            if data.cpu_backup_tensor is not None:
                data.value.copy_(data.cpu_backup_tensor)
                data.cpu_backup_tensor = None


def test_repeated_sleep_restores_transformed_weights_from_one_cpu_copy():
    allocator = Allocator()
    weights = allocator.pointer_to_data[1]
    weights.value.mul_(7)  # Storage contains the finalized execution layout.
    expected = weights.value.clone()
    cache = StaticTeacherWeightCache(allocator)
    for _ in range(3):
        cache.sleep(offload_tags=("weights",))
        assert not weights.value.any()
        assert allocator.pointer_to_data[2].cpu_backup_tensor is None
        allocator.wake_up()
        torch.testing.assert_close(weights.value, expected)
        assert weights.cpu_backup_tensor is None
    assert allocator.copies == 1
    assert cache.stats() == dict(backup_bytes=expected.numel() * expected.element_size(), backup_count=1, sleep_count=3)


@pytest.mark.parametrize("change", ["replace", "add", "remove"])
def test_changed_weight_storage_fails_before_discarding_live_data(change):
    allocator = Allocator()
    cache = StaticTeacherWeightCache(allocator)
    cache.sleep(offload_tags=("weights",))
    allocator.wake_up()
    if change == "replace":
        allocator.pointer_to_data[1] = SimpleNamespace(**vars(allocator.pointer_to_data[1]))
    elif change == "add":
        allocator.pointer_to_data[3] = SimpleNamespace(tag="weights", cpu_backup_tensor=None)
    else:
        del allocator.pointer_to_data[1]
    with pytest.raises(RuntimeError):
        cache.sleep(offload_tags=("weights",))
    assert cache.sleep_count == 1


def test_static_teacher_rejects_discarding_level_two_sleep():
    allocator = Allocator()
    cache = StaticTeacherWeightCache(allocator)
    with pytest.raises(ValueError, match="level=1"):
        cache.sleep(offload_tags=())
    assert allocator.copies == 0


def test_model_load_enters_weights_pool_and_vllm_config(monkeypatch):
    from contextlib import contextmanager

    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout import sleep_worker

    active = set()

    @contextmanager
    def context(name):
        active.add(name)
        try:
            yield
        finally:
            active.remove(name)

    def load_model(**kwargs):
        assert active == {"weights", "config"}
        assert kwargs == {"eep_scale_up": False}

    monkeypatch.delenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", raising=False)
    monkeypatch.setattr(sleep_worker, "set_current_vllm_config", lambda config: context("config"))
    worker = SimpleNamespace(
        vllm_config=object(),
        _maybe_get_memory_pool_context=lambda tag: context(tag),
        model_runner=SimpleNamespace(load_model=load_model),
    )
    sleep_worker.SleepManagedWorker.load_model(worker)
    assert not active
