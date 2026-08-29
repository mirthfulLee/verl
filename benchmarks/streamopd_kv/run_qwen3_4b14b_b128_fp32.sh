#!/usr/bin/env bash
set -euo pipefail

export STUDENT_MODEL=${STUDENT_MODEL:-/models/store/Qwen/Qwen3-4B}
export TEACHER_MODEL=${TEACHER_MODEL:-/models/store/Qwen/Qwen3-14B}
export BATCH_SIZE=${BATCH_SIZE:-128}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
export TOKEN_CHUNK_SIZE=${TOKEN_CHUNK_SIZE:-32}
export REVERSE_CHUNK_SIZE=${REVERSE_CHUNK_SIZE:-64}
export REVERSE_BATCH_SIZE=${REVERSE_BATCH_SIZE:-16}
export REVERSE_BATCH_MAX_TOKENS=${REVERSE_BATCH_MAX_TOKENS:-8192}
export ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-2048}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-3}
export VERL_USE_UV=${VERL_USE_UV:-0}

bash benchmarks/streamopd_kv/run_verl_sync_comparison.sh \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.rollout.dtype=float32 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.do_sample=False \
  actor_rollout_ref.rollout.agent.num_workers=16 \
  distillation.teacher_models.teacher_model.inference.dtype=bfloat16 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.55 \
  distillation.streamopd_kv.kv_handoff_dir=/data1/huanli/tmp/streamopd-kv-4b14b-b128 \
  '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype: fp32, reduce_dtype: fp32, buffer_dtype: fp32}' \
  "$@"
