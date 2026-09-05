# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.memory_utils import aggressive_empty_cache
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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_streamopd_allocator_cache(self) -> None:
        """Return inactive Trainer allocations before the next Teacher wake."""

        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_streamopd_trainer_state(self) -> None:
        """Release the shared pool after one contiguous training phase."""

        device = get_torch_device()
        device.synchronize()
        self.actor.release_reverse_slots()
        optimizer = self.actor.engine.optimizer
        if optimizer is not None:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if hasattr(value, "device") and value.device.type != "cpu":
                        # The generic helper uses an asynchronous pageable-host
                        # copy, which is not supported by every CUDA runtime.
                        state[key] = value.to("cpu", non_blocking=False)
        self.actor.engine.to(device="cpu", model=True, optimizer=False, grad=True)
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_streamopd_trainer_state(self) -> None:
        """Claim a shared pool after its inference process enters level-2 sleep."""

        aggressive_empty_cache(force_sync=True)
        self.actor.engine.to(device=get_device_name(), model=True, optimizer=True, grad=True)
        self.actor.allocate_reverse_slots()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def allocate_streamopd_reverse_slots(self) -> None:
        """Allocate persistent reverse slots on a dedicated Trainer pool."""

        self.actor.allocate_reverse_slots()
