#!/usr/bin/env bash
set -euo pipefail

export TRAINER_MODE=streamopd_cf STREAMOPD_KV_ENABLED=False TRAINER_PLACEMENT=dedicated
export STUDENT_GPUS=${STUDENT_GPUS:-2} TEACHER_GPUS=${TEACHER_GPUS:-1} ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
export ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-streamopd_cf_qwen3}

train_dynamic=True
train_micro=null
if [[ ${TRAIN_MICRO_BATCH_SIZE:-0} != 0 ]]; then
  train_dynamic=False
  train_micro=$TRAIN_MICRO_BATCH_SIZE
fi

bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh \
  distillation.streamopd_cf.enabled=True \
  distillation.streamopd_cf.forward_chunk_size="${FORWARD_CHUNK_SIZE:-1024}" \
  distillation.streamopd_cf.loss_chunk_size="${LOSS_CHUNK_SIZE:-2048}" \
  distillation.streamopd_cf.token_chunk_size="${TOKEN_CHUNK_SIZE:-1024}" \
  actor_rollout_ref.actor.use_dynamic_bsz="$train_dynamic" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$train_micro" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${TRAIN_MAX_TOKENS_PER_GPU:-0}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$train_dynamic" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-1}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-0}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS:-0}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}" \
  distillation.teacher_models.teacher_model.inference.max_num_seqs="${TEACHER_MAX_NUM_SEQS:-0}" \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${TEACHER_MAX_BATCHED_TOKENS:-0}" \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.checkpoint_engine.backend=host \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.host.rollout_dtype=bfloat16 \
  distillation.distillation_loss.use_chunked_topk=True \
  "$@"
