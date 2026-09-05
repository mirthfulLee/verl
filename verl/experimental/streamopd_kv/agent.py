# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging
from typing import Any

import ray
import torch

from .protocol import CommittedTokenChunk, TrajectoryKey
from .streaming_teacher import StreamingTeacherCoordinator

logger = logging.getLogger(__name__)


class StreamOPDAgentSession:
    """StreamOPD integration owned by one generic AgentLoop worker."""

    def __init__(self, config, rollout_config, tokenizer, model_type, teacher_manager, callback) -> None:
        stream_config = config.distillation.streamopd_kv
        if rollout_config.agent.default_agent_loop != "single_turn_agent":
            raise NotImplementedError("StreamOPD supports the single_turn_agent loop only")
        if model_type != "qwen3":
            raise NotImplementedError(f"StreamOPD reverse training supports Qwen3 students only, got {model_type!r}")
        if stream_config.require_same_tokenizer:
            from transformers import AutoTokenizer

            teacher_model = next(iter(config.distillation.teacher_models.values()))
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_model.model_path,
                trust_remote_code=bool(config.data.get("trust_remote_code", False)),
            )
            if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
                raise ValueError("StreamOPD requires identical student and Teacher token-id vocabularies")

        self.config = stream_config
        self.teacher_manager = teacher_manager
        scheduler_name = str(stream_config.scheduler_actor_name)
        scheduler = ray.get_actor(scheduler_name) if scheduler_name else None
        self.coordinator = StreamingTeacherCoordinator(
            self._score,
            max_pending_chunks=int(stream_config.max_pending_teacher_chunks),
            scheduler=scheduler,
            max_active_trajectories=int(stream_config.teacher_prefill_max_active_trajectories),
            max_active_kv_tokens=int(stream_config.teacher_prefill_max_active_kv_tokens),
            kv_page_size=int(stream_config.teacher_prefill_kv_page_size),
            kv_reservation_tokens=int(rollout_config.prompt_length + rollout_config.response_length),
        )
        callback.streamopd_callback = ray.get_runtime_context().current_actor
        callback.streamopd_chunk_size = int(stream_config.token_chunk_size)
        callback.streamopd_page_size = int(stream_config.teacher_prefill_kv_page_size)

    async def _score(self, token_ids: list[int], request_id: str, terminal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return await self.teacher_manager.compute_teacher_logprobs_streaming(
            token_ids=token_ids,
            request_id=request_id,
            terminal=terminal,
        )

    async def submit(self, value: dict[str, Any]) -> None:
        await self.coordinator.submit(CommittedTokenChunk.from_dict(value))

    async def result(
        self,
        output,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        routing_key: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output.multi_modal_data:
            raise NotImplementedError("StreamOPD does not support multimodal Teacher streaming")
        trajectory_id = output.extra_fields.get("streamopd_trajectory_id")
        policy_version = output.extra_fields.get("streamopd_policy_version")
        if trajectory_id is None or policy_version is None:
            raise RuntimeError("rollout output is missing StreamOPD trajectory identity")
        teacher_ids, teacher_logprobs = await self.coordinator.result(
            TrajectoryKey(int(policy_version), str(trajectory_id)),
            required_completion_tokens=len(response_ids),
        )
        if self.config.validate_teacher_artifacts:
            expected_ids, expected_logprobs = await self.teacher_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                routing_key=routing_key,
            )
            if not torch.equal(teacher_ids, expected_ids):
                mismatched = int((teacher_ids != expected_ids).sum().item())
                raise RuntimeError(f"streamed Teacher token ids differ from reference scoring at {mismatched} entries")
            error = float((teacher_logprobs - expected_logprobs).abs().max().item())
            if error > self.config.validation_atol:
                raise RuntimeError(
                    "streamed Teacher logprobs differ from reference scoring: "
                    f"max_abs_error={error}, tolerance={self.config.validation_atol}"
                )
            logger.info("StreamOPD Teacher validation max_abs_error=%g", error)
        return teacher_ids, teacher_logprobs
