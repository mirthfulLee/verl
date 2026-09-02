# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device
from verl.workers.engine_workers import ActorRolloutRefWorker

from .fsdp_worker import StreamOPDKVTrainingWorker


class StreamOPDActorWorker(ActorRolloutRefWorker):
    """Actor-only Ray worker exposing StreamOPD reverse-planning RPCs."""

    actor_worker_cls = StreamOPDKVTrainingWorker

    def __init__(self, config, role: str, distillation_config=None, **kwargs) -> None:
        if config.actor.strategy not in ("fsdp", "fsdp2"):
            raise NotImplementedError("StreamOPD training currently supports FSDP/FSDP2 only")
        super().__init__(config, role, distillation_config, **kwargs)

    def _configure_actor_training_worker(self, training_config, distillation_config) -> None:
        if distillation_config is None or not distillation_config.streamopd_kv.enabled:
            raise ValueError("StreamOPD actor worker requires an enabled StreamOPD configuration")
        training_config.extra_context.update(
            streamopd_kv=distillation_config.streamopd_kv,
            distillation=distillation_config,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_streamopd_reverse_plan(self):
        return self.actor.prepare_reverse_plan()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def configure_streamopd_reverse_preflight(self, batch_cap=None, additional_reserve_gib=0.0) -> None:
        self.actor.configure_reverse_preflight(
            batch_cap=batch_cap,
            additional_reserve_gib=additional_reserve_gib,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_streamopd_memory_stats(self) -> None:
        get_torch_device().reset_peak_memory_stats()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_streamopd_device_memory_stats(self) -> dict[str, int]:
        device = get_torch_device()
        free_bytes, total_bytes = device.mem_get_info()
        return {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "allocated_bytes": int(device.memory_allocated()),
            "reserved_bytes": int(device.memory_reserved()),
        }
