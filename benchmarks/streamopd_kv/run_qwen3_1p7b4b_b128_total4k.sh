#!/usr/bin/env bash
set -euo pipefail

# The trajectory cap includes the prompt: 1024 prompt tokens + 3072 response tokens = 4096.
export STUDENT_MODEL=${STUDENT_MODEL:-/models/store/Qwen/Qwen3-1.7B}
export TEACHER_MODEL=${TEACHER_MODEL:-/models/store/Qwen/Qwen3-4B}
export BATCH_SIZE=${BATCH_SIZE:-128}
export TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((TOTAL_TRAJECTORY_LENGTH - MAX_PROMPT_LENGTH))}
export STUDENT_GPUS=${STUDENT_GPUS:-4}
export TEACHER_GPUS=${TEACHER_GPUS:-2}
export TOKEN_CHUNK_SIZE=${TOKEN_CHUNK_SIZE:-$(((MAX_RESPONSE_LENGTH + 1) / 2))}
export REVERSE_CHUNK_SIZE=${REVERSE_CHUNK_SIZE:-2048}
export REVERSE_BATCH_SIZE=${REVERSE_BATCH_SIZE:-4}
export REVERSE_BATCH_MAX_TOKENS=${REVERSE_BATCH_MAX_TOKENS:-8192}
export ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-$TOTAL_TRAJECTORY_LENGTH}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-24}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.25}
export OVERLAP_ROLLOUT_TRAINING=${OVERLAP_ROLLOUT_TRAINING:-False}
export ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-32}
export COLOCATE_TEACHER_WITH_STUDENT=${COLOCATE_TEACHER_WITH_STUDENT:-True}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
export VERL_USE_UV=${VERL_USE_UV:-0}

bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.rollout.dtype=float32 \
  distillation.teacher_models.teacher_model.inference.dtype=bfloat16 \
  actor_rollout_ref.rollout.do_sample=False \
  actor_rollout_ref.rollout.agent.num_workers=16 \
  distillation.streamopd_kv.kv_handoff_dir=${KV_HANDOFF_DIR:-/data1/huanli/tmp/streamopd-kv-1p7b4b-b128-total4k} \
  '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype: fp32, reduce_dtype: fp32, buffer_dtype: fp32}' \
  "$@"
