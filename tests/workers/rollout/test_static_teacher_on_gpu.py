# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import subprocess
import sys

import pytest
import torch

from verl.utils.device import get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.static_teacher import StaticTeacherWeightCache


def _check_sleep_cycles():
    pytest.importorskip("vllm")
    from vllm.device_allocator.cumem import CuMemAllocator

    device = get_torch_device()
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool(tag="weights"):
        weights = torch.full((64 * 1024 * 1024,), 1.25, dtype=torch.bfloat16, device=get_device_name())
        weights.mul_(3)
    with allocator.use_memory_pool(tag="kv_cache"):
        kv = torch.ones(32 * 1024 * 1024, dtype=torch.bfloat16, device=get_device_name())
    pointer = weights.data_ptr()
    cache = StaticTeacherWeightCache(allocator)
    device.synchronize()
    for _ in range(3):
        before = device.mem_get_info()[0]
        cache.sleep(offload_tags=("weights",))
        device.synchronize()
        assert device.mem_get_info()[0] - before >= weights.numel() * weights.element_size()
        allocator.wake_up(tags=["weights"])
        assert weights.data_ptr() == pointer
        assert torch.all(weights == 3.75).item()
        allocator.wake_up(tags=["kv_cache"])
        kv.fill_(1)
        device.synchronize()
    assert cache.backup_count == 1
    assert cache.sleep_count == 3


def test_real_sleep_releases_gpu_and_preserves_weights_across_three_wakes():
    pytest.importorskip("vllm")
    # Like vLLM's worker lifecycle, dispose CuMem pools with the CUDA process.
    # Destroying a standalone PyTorch pluggable pool during interpreter teardown
    # can crash before its Python free callbacks run (vLLM/PyTorch limitation).
    subprocess.run([sys.executable, __file__], check=True, timeout=120)


if __name__ == "__main__":
    _check_sleep_cycles()
    os._exit(0)
