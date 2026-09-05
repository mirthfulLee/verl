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
import time

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _stage_timing_metrics_from_tags(
    tags: list[dict], *, policy_started_at: float, training_seconds: float
) -> dict[str, float]:
    tags = [tag for tag in tags if not tag.get("is_padding", False)]
    metrics = {"stage/training_seconds": float(training_seconds)}
    completed = {}
    for stage in ("rollout", "teacher"):
        start_key, end_key = f"_stage_{stage}_started_at", f"_stage_{stage}_completed_at"
        stage_tags = [tag for tag in tags if start_key in tag and end_key in tag]
        if not stage_tags:
            continue
        started = min(float(tag[start_key]) for tag in stage_tags)
        completed[stage] = max(float(tag[end_key]) for tag in stage_tags)
        durations = [float(tag[f"_stage_{stage}_request_seconds"]) for tag in stage_tags]
        metrics.update(
            {
                f"stage/{stage}_span_seconds": max(0.0, completed[stage] - started),
                f"stage/{stage}_makespan_seconds": max(0.0, completed[stage] - policy_started_at),
                f"stage/{stage}_request_seconds/mean": sum(durations) / len(durations),
                f"stage/{stage}_request_seconds/max": max(durations),
            }
        )
    if "teacher" in completed and "rollout" in completed:
        metrics["stage/teacher_tail_seconds"] = max(0.0, completed["teacher"] - completed["rollout"])
    return metrics


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def prepare_step(self) -> dict:
        self._stage_policy_started_at = time.perf_counter()
        return super().prepare_step()

    def step(self, metrics: dict, timing_raw: dict):
        batch = super().step(metrics, timing_raw)
        metrics.update(
            _stage_timing_metrics_from_tags(
                batch.tags,
                policy_started_at=self._stage_policy_started_at,
                training_seconds=float(timing_raw.get("update_actor", 0.0)),
            )
        )
        return batch

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
            if self.use_teacher_policy and self.distillation_config.colocate_teacher_with_student:
                self.teacher_model_manager.wake_up()

    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        if self.use_teacher_policy and self.distillation_config.colocate_teacher_with_student:
            self.teacher_model_manager.sleep()
        self.checkpoint_manager.sleep_replicas()
