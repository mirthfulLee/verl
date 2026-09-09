# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""vLLM 0.15.1 worker compatibility fix for the weights allocator scope."""

import os

from vllm.config import set_current_vllm_config
from vllm.v1.worker.gpu_worker import Worker


class SleepManagedWorker(Worker):
    """Enter both load contexts; native 0.15.1 accidentally uses boolean AND."""

    def load_model(self) -> None:
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(tag="weights"), set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)
