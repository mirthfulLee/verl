# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import ray

from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.workers.rollout.llm_server import LLMServerManager

from .batching import AutoBatchActorWorker


class DedicatedOPDPools:
    """Shared placement hooks for strict synchronous OPD strategies."""

    actor_worker_cls = AutoBatchActorWorker

    def _init_resource_pool_mgr(self):
        if need_reference_policy(self.config) or self.use_critic:
            raise ValueError("streamopd-cf supports direct distillation without reference policy or critic")
        super()._init_resource_pool_mgr()
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        self.role_worker_mapping.pop(actor_role)
        self.mapping.pop(actor_role)
        self.role_worker_mapping[Role.Actor] = ray.remote(self.actor_worker_cls)
        self.mapping[Role.Actor] = "global_pool"
        pools = dict(self.resource_pool_manager.resource_pool_spec)
        rollout = self.config.actor_rollout_ref.rollout
        pools["rollout_pool"] = [int(rollout.n_gpus_per_node)] * int(rollout.nnodes)
        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=pools, mapping=self.mapping)

    def _uses_external_checkpoint_engine(self):
        return True

    def _create_llm_server_manager(self, actor_rollout_resource_pool):
        return LLMServerManager.create(
            config=self.config,
            rollout_resource_pool=self.resource_pool_manager.resource_pool_dict["rollout_pool"],
            colocate_without_worker_group=True,
        )
