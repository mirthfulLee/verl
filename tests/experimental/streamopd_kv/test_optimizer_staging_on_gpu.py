# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import copy

import pytest
import torch

from verl.utils.fsdp_utils import load_fsdp_optimizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_delayed_adam_restore_preserves_warm_optimizer_updates():
    torch.manual_seed(7)
    baseline = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 4)).cuda()
    staged = copy.deepcopy(baseline)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3) for model in (baseline, staged)]
    for _ in range(3):
        inputs, targets = torch.randn(4, 8, device="cuda"), torch.randn(4, 4, device="cuda")
        for model, optimizer in zip((baseline, staged), optimizers, strict=True):
            if model is baseline:
                load_fsdp_optimizer(optimizer, "cuda")
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(inputs), targets).backward()
            if model is staged:
                assert all(
                    value.device.type == "cpu"
                    for state in optimizer.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                )
                load_fsdp_optimizer(optimizer, "cuda")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        for expected, actual in zip(baseline.parameters(), staged.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for optimizer in optimizers:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cpu()
