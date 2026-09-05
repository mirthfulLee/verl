# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio

import pytest
import torch

from verl.checkpoint_engine.base import split_weight_chunks
from verl.checkpoint_engine.host_checkpoint_engine import HostCheckpointEngine, HostCheckpointMetadata


def test_host_topology_assigns_one_sender_and_independent_receivers(tmp_path) -> None:
    session = tmp_path / "verl-host-checkpoint-test"
    metadata = [
        HostCheckpointMetadata(str(session)),
        HostCheckpointMetadata(None),
        HostCheckpointMetadata(None),
        HostCheckpointMetadata(None),
    ]

    actor, rollout = HostCheckpointEngine.build_topology(2, 2, metadata)

    assert actor["role"] == ["sender", "participant"]
    assert rollout["role"] == ["receiver", "receiver"]
    assert actor["session_dir"] == [str(session), str(session)]
    assert rollout["actor_world_size"] == [2, 2]


def test_host_checkpoint_round_trip_to_multiple_receivers(tmp_path) -> None:
    async def run_round_trip() -> None:
        root = str(tmp_path)
        sender = HostCheckpointEngine(bucket_size=32, is_master=True, directory=root)
        receiver_a = HostCheckpointEngine(bucket_size=32, directory=root, poll_interval=0.001)
        receiver_b = HostCheckpointEngine(bucket_size=32, directory=root, poll_interval=0.001)

        sender_metadata = sender.prepare()
        actor_kwargs, rollout_kwargs = HostCheckpointEngine.build_topology(
            1,
            2,
            [sender_metadata, receiver_a.prepare(), receiver_b.prepare()],
        )
        sender.init_process_group(**{name: values[0] for name, values in actor_kwargs.items()})
        receiver_a.init_process_group(**{name: values[0] for name, values in rollout_kwargs.items()})
        receiver_b.init_process_group(**{name: values[1] for name, values in rollout_kwargs.items()})

        expected = {
            "small": torch.arange(5, dtype=torch.float32),
            "large": torch.arange(20, dtype=torch.float32).reshape(4, 5),
            "tail": torch.arange(3, dtype=torch.bfloat16),
        }

        async def collect(receiver: HostCheckpointEngine) -> dict[str, torch.Tensor]:
            return {name: tensor.clone() async for name, tensor in receiver.receive_weights()}

        received_a, received_b, metrics = await asyncio.gather(
            collect(receiver_a),
            collect(receiver_b),
            sender.send_weights(iter(expected.items())),
        )
        for received in (received_a, received_b):
            assert received.keys() == expected.keys()
            for name, tensor in expected.items():
                torch.testing.assert_close(received[name], tensor)
        assert metrics["timing/checkpoint_host_seconds"] > 0
        assert metrics["checkpoint/host_gib_per_second"] > 0

        session_dir = sender.session_dir
        assert session_dir is not None
        metadata_paths = sorted(session_dir.glob("bucket-*.meta.pt"))
        data_paths = sorted(session_dir.glob("bucket-*.bin"))
        assert len(metadata_paths) == len(data_paths) > 1
        for metadata_path, data_path in zip(metadata_paths, data_paths, strict=True):
            metadata = torch.load(metadata_path, weights_only=False)
            assert "buffer" not in metadata
            assert data_path.stat().st_size == metadata["length"]
        receiver_a.finalize()
        receiver_b.finalize()
        sender.finalize()
        assert not session_dir.exists()

    asyncio.run(run_round_trip())


def test_host_checkpoint_rejects_incomplete_raw_bucket(tmp_path) -> None:
    engine = HostCheckpointEngine(bucket_size=32, is_master=True, directory=str(tmp_path))
    metadata = engine.prepare()
    assert metadata.session_dir is not None
    engine.init_process_group(role="receiver", session_dir=metadata.session_dir, actor_world_size=1)
    engine._bucket_data_path(0).write_bytes(b"\0")
    torch.save(
        {
            "format": "verl-host-checkpoint-mmap-v1",
            "bucket_meta": {},
            "is_last": True,
            "length": 2,
        },
        engine._bucket_metadata_path(0),
    )

    async def consume() -> None:
        async for _ in engine.receive_weights():
            pass

    with pytest.raises(RuntimeError, match="invalid host checkpoint bucket data"):
        asyncio.run(consume())


def test_meta_only_weight_split_does_not_view_payload() -> None:
    class MetadataOnlyWeight:
        shape = torch.Size((8,))
        dtype = torch.float32
        nbytes = 32

        def view(self, *_args):
            raise AssertionError("meta-only splitting must not view the payload")

    async def collect():
        weights = iter((("weight", MetadataOnlyWeight()),))
        return [item async for item in split_weight_chunks(weights, 16, meta_only=True)]

    chunks = asyncio.run(collect())
    assert [meta.chunk_size for meta, payload in chunks] == [16, 16]
    assert all(payload is None for _, payload in chunks)


def test_weight_split_casts_only_floating_wire_tensors() -> None:
    async def collect():
        weights = iter(
            (
                ("floating", torch.arange(8, dtype=torch.float32)),
                ("integer", torch.arange(4, dtype=torch.int64)),
            )
        )
        return [item async for item in split_weight_chunks(weights, 64, floating_dtype=torch.bfloat16)]

    chunks = asyncio.run(collect())
    assert [meta.dtype for meta, _ in chunks] == [torch.bfloat16, torch.int64]
    assert [meta.chunk_size for meta, _ in chunks] == [16, 32]
    torch.testing.assert_close(chunks[0][1].view(torch.bfloat16), torch.arange(8, dtype=torch.bfloat16))


def _send_fsdp1_shards(rank, rendezvous, directory, session_dir):
    from datetime import timedelta

    from torch.distributed._shard.sharded_tensor import Shard, init_from_local_shards
    from torch.distributed._shard.sharding_spec import ShardMetadata

    from verl.checkpoint_engine import host_checkpoint_engine

    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        host_checkpoint_engine.get_device_id = lambda: torch.device("cpu")
        local = torch.arange(rank * 5, (rank + 1) * 5, dtype=torch.float32)
        shard = Shard(local, ShardMetadata([rank * 5], [5], f"rank:{rank}/cpu"))
        weight = init_from_local_shards([shard], 10)
        engine = HostCheckpointEngine(bucket_size=8, directory=directory, rollout_dtype="bfloat16")
        engine.init_process_group("sender" if rank == 0 else "participant", session_dir, actor_world_size=2)
        asyncio.run(engine.send_weights(iter([("sharded", weight), ("dense", torch.tensor([11.0]))])))
    finally:
        torch.distributed.destroy_process_group()


def test_host_checkpoint_gathers_fsdp1_shards_with_matching_participant_buckets(tmp_path):
    """Both ranks must cross each gather/barrier; only rank 0 publishes payloads."""
    sender = HostCheckpointEngine(bucket_size=8, directory=str(tmp_path), is_master=True)
    session_dir = sender.prepare().session_dir
    torch.multiprocessing.spawn(
        _send_fsdp1_shards,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), session_dir),
        nprocs=2,
    )
    receiver = HostCheckpointEngine(bucket_size=8, directory=str(tmp_path))
    receiver.init_process_group("receiver", session_dir, actor_world_size=2)

    async def receive():
        return {name: tensor.clone() async for name, tensor in receiver.receive_weights()}

    actual = asyncio.run(receive())
    torch.testing.assert_close(actual["sharded"], torch.arange(10, dtype=torch.bfloat16))
    torch.testing.assert_close(actual["dense"], torch.tensor([11.0], dtype=torch.bfloat16))
    sender.init_process_group("sender", session_dir, actor_world_size=2)
    sender.finalize()
