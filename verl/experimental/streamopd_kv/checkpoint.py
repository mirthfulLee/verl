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
    """Overlap shared-pool publication and reception with inference KV asleep.

    The shared Trainer has already offloaded optimizer state and released its
    reverse slots and gradients. Wake only Rollout weights while exporting the
    new parameters, then release Trainer export allocations before restoring
    inference KV. Generation waits for the scheduler's next policy version.
    """
    if not shares_rollout:
        # Actor-only rollout replicas use COLOCATED mode even on dedicated
        # GPUs. Their release_kv_cache() skips awake replicas, so explicitly
        # sleep before the ordinary weights-only wake and transfer sequence.
        if manager.backend != "naive":
            await manager.sleep_replicas(level=2)
        return await manager.update_weights(global_steps)
    if manager.backend != "host":
        raise ValueError("phase-exclusive weight sync requires checkpoint_engine.backend=host")

    rollout = manager.create_rollout_worker_group()
    actor = manager.actor_wg
    timings = {}
    with simple_timer("rollout_sleep", timings):
        await manager.sleep_replicas(level=2)
    manager.build_process_group(rollout)
    with simple_timer("weights_wake", timings):
        await manager.release_kv_cache_replicas()
    with simple_timer("publish_receive", timings):
        results = ray.get(
            actor.update_weights(global_steps=global_steps, mode=manager.backend)
            + rollout.update_weights(global_steps=global_steps)
        )
    with simple_timer("trainer_release", timings):
        actor.release_streamopd_allocator_cache()
    manager.finalize_workers(rollout)
    with simple_timer("kv_wake", timings):
        await manager.resume_kv_cache_replicas()

    metrics = {}
    for result in results:
        if isinstance(result, dict):
            metrics.update(result)
    metrics.update({f"checkpoint/phase_exclusive_{key}_seconds": value for key, value in timings.items()})
    return metrics
