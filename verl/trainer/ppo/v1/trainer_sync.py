# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
            if self.use_teacher_policy and self.distillation_config.streamopd_kv.colocate_teacher_with_student:
                self.teacher_model_manager.wake_up()

    def on_sample_end(self):
        if (
            self.streamopd_kv_enabled
            and self.distillation_config.streamopd_kv.overlap_rollout_training
            and self.local_trigger_step < self.parameter_sync_step - 1
        ):
            # Keep rollout and teacher work active while reverse backward
            # accumulates gradients for train-ready microbatches. Parameters
            # remain unchanged until the final trigger.
            return
        # sleep all replicas to discard weights and kv cache
        if self.use_teacher_policy and self.distillation_config.streamopd_kv.colocate_teacher_with_student:
            self.teacher_model_manager.sleep()
        self.checkpoint_manager.sleep_replicas()
