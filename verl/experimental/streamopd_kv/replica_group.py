# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""StreamOPD telemetry and transfer barriers shared by both inference pools."""

import asyncio

from verl.utils.ray_utils import auto_await

from .protocol import TRANSFER_MAX_KEYS, TRANSFER_SUM_KEYS


class VLLMReplicaGroup:
    """Aggregate one collective RPC per replica without double-counting TP capacity."""

    def __init__(self, server_handles):
        self.server_handles = list(server_handles)

    async def _rpc(self, method):
        return await asyncio.gather(*(server.collective_rpc.remote(method) for server in self.server_handles))

    @auto_await
    async def reset_device_memory_stats(self) -> None:
        await self._rpc("reset_device_memory_stats")

    @auto_await
    async def collect_device_memory_stats(self) -> dict[str, int]:
        responses = await self._rpc("get_device_memory_stats")
        worker_stats = [stats for response in responses for stats in (response or [])]
        if not worker_stats:
            raise RuntimeError("vLLM workers returned no device memory statistics")
        result = {
            key: max(int(stats[key]) for stats in worker_stats)
            for key in ("allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes")
        }
        result["free_bytes"] = min(int(stats["free_bytes"]) for stats in worker_stats)
        result["total_bytes"] = min(int(stats["total_bytes"]) for stats in worker_stats)
        return result

    @auto_await
    async def collect_kv_cache_capacity_tokens(self) -> int:
        """Return total logical KV capacity across replicas."""

        responses = await self._rpc("get_kv_cache_capacity")
        replica_capacities = []
        for response in responses:
            worker_capacities = [int(stats["capacity_tokens"]) for stats in (response or [])]
            if not worker_capacities:
                raise RuntimeError("vLLM workers returned no KV cache capacity")
            replica_capacities.append(min(worker_capacities))
        if not replica_capacities:
            raise RuntimeError("StreamOPD has no vLLM replicas")
        return sum(replica_capacities)

    @auto_await
    async def reset_streamopd_kv_transfer_stats(self) -> None:
        await self._rpc("reset_streamopd_kv_transfer_stats")

    @auto_await
    async def collect_streamopd_kv_transfer_stats(self) -> dict[str, float]:
        responses = await self._rpc("get_streamopd_kv_transfer_stats")
        worker_stats = [stats for response in responses for stats in (response or []) if stats]
        if not worker_stats:
            raise RuntimeError("vLLM workers returned no StreamOPD KV transfer statistics")
        return {
            **{key: sum(float(stats[key]) for stats in worker_stats) for key in TRANSFER_SUM_KEYS},
            **{key: max(float(stats[key]) for stats in worker_stats) for key in TRANSFER_MAX_KEYS},
        }

    @auto_await
    async def wait_for_streamopd_kv_transfers(self) -> float:
        """Wait until every Rollout worker has sealed its Host KV slots."""

        responses = await self._rpc("wait_for_streamopd_kv_transfers")
        waits = [float(value) for response in responses for value in (response or [])]
        return max(waits, default=0.0)
