# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fail quickly when the pilot host's eight-rank NCCL path is unavailable."""

import datetime
import os
import time

import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
value = torch.full((1024 * 1024,), dist.get_rank() + 1.0, device="cuda")
started = time.perf_counter()
for _ in range(4):
    value.fill_(dist.get_rank() + 1.0)
    dist.all_reduce(value)
torch.cuda.synchronize()
assert value[0].item() == sum(range(1, dist.get_world_size() + 1))
print(f"rank={dist.get_rank()} elapsed={time.perf_counter() - started:.3f}s", flush=True)
dist.destroy_process_group()
