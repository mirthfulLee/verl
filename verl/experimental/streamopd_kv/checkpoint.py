# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Weight handoff when Trainer and Rollout share a GPU pool."""

import ray

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.utils.profiler.performance import simple_timer
from verl.utils.ray_utils import auto_await


@auto_await
async def update_streamopd_weights(
    manager: CheckpointEngineManager, global_steps: int, *, shares_rollout: bool
) -> dict:
    """Use the ordinary transport unless GPU ownership requires a serial handoff.

    Host publication is durable: publish with Rollout asleep, release Trainer
    memory, then wake Rollout to load weights. Generation stays paused until the
    StreamOPD scheduler admits the next policy version.
    """
    if not shares_rollout:
        return await manager.update_weights(global_steps)
    if manager.backend != "host":
        raise ValueError("phase-exclusive weight sync requires checkpoint_engine.backend=host")

    rollout = manager.create_rollout_worker_group()
    actor = manager.actor_wg
    timings = {}
    with simple_timer("rollout_sleep", timings):
        await manager.sleep_replicas(level=2)
    manager.build_process_group(rollout)
    with simple_timer("publish", timings):
        results = ray.get(actor.update_weights(global_steps=global_steps, mode=manager.backend))
    with simple_timer("trainer_release", timings):
        actor.release_streamopd_allocator_cache()
    with simple_timer("weights_wake", timings):
        await manager.release_kv_cache_replicas()
    with simple_timer("receive", timings):
        ray.get(rollout.update_weights(global_steps=global_steps))
    manager.finalize_workers(rollout)
    with simple_timer("kv_wake", timings):
        await manager.resume_kv_cache_replicas()

    metrics = {}
    for result in results:
        if isinstance(result, dict):
            metrics.update(result)
    metrics.update({f"checkpoint/phase_exclusive_{key}_seconds": value for key, value in timings.items()})
    return metrics
