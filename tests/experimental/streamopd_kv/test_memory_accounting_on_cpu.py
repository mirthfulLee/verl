# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from verl.experimental.streamopd_kv.fsdp_worker import (
    _deferred_training_state_bytes,
    _unsharded_gradient_reserve_bytes,
)


def _check_sharded_reserves(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = torch.nn.Module()
        parameter = torch.nn.Parameter(distribute_tensor(torch.ones(8), mesh, [Shard(0)]))
        model.register_parameter("weight", parameter)
        optimizer = torch.optim.AdamW(model.parameters())
        assert parameter.numel() == 8
        assert parameter.to_local().numel() == 4
        assert _deferred_training_state_bytes(model, optimizer) == 3 * 4 * 4
        assert _unsharded_gradient_reserve_bytes(model, 2) == 4 * 4
    finally:
        dist.destroy_process_group()


def test_dtensor_memory_reserves_follow_local_shards(tmp_path):
    torch.multiprocessing.spawn(_check_sharded_reserves, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
