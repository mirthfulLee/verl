#!/usr/bin/env bash
set -xeuo pipefail

STUDENT_MODEL=${STUDENT_MODEL:-/models/store/Qwen/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-/models/store/Qwen/Qwen3-4B}
DATASET=${DATASET:-/data1/models/hf/datasets--open-r1--DAPO-Math-17k-Processed/snapshots/31dd309567e3da778038cc87d868b6097a3ccf68/en/train-00000-of-00001.parquet}
STUDENT_GPUS=${STUDENT_GPUS:-2}
TEACHER_GPUS=${TEACHER_GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((TOTAL_TRAJECTORY_LENGTH - MAX_PROMPT_LENGTH))}
TOKEN_CHUNK_SIZE=${TOKEN_CHUNK_SIZE:-1024}
REVERSE_CHUNK_SIZE=${REVERSE_CHUNK_SIZE:-2048}
REVERSE_BATCH_SIZE=${REVERSE_BATCH_SIZE:-4}
REVERSE_BATCH_MAX_TOKENS=${REVERSE_BATCH_MAX_TOKENS:-8192}
ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-$TOTAL_TRAJECTORY_LENGTH}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.55}
OVERLAP_ROLLOUT_TRAINING=${OVERLAP_ROLLOUT_TRAINING:-False}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-32}
COLOCATE_TEACHER_WITH_STUDENT=${COLOCATE_TEACHER_WITH_STUDENT:-False}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
STREAMOPD_KV_ENABLED=${STREAMOPD_KV_ENABLED:-True}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-streamopd_kv_qwen3}

if (( TOTAL_TRAJECTORY_LENGTH < 1 || MAX_PROMPT_LENGTH < 1 || MAX_RESPONSE_LENGTH < 1 )); then
  echo "TOTAL_TRAJECTORY_LENGTH, MAX_PROMPT_LENGTH, and MAX_RESPONSE_LENGTH must be positive" >&2
  exit 2
fi
if (( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > TOTAL_TRAJECTORY_LENGTH )); then
  echo "MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH must not exceed TOTAL_TRAJECTORY_LENGTH" >&2
  exit 2
fi

LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
  LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
  RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi

"${LAUNCH[@]}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATASET" \
  data.val_files="$DATASET" \
  data.train_batch_size="$BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=False \
  data.custom_cls.path=benchmarks/streamopd_kv/dapo_math_dataset.py \
  data.custom_cls.name=DAPOMathDataset \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKENS_PER_GPU" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1)) \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=sync \
  trainer.n_gpus_per_node="$STUDENT_GPUS" \
  trainer.nnodes=1 \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.project_name=streamopd_kv \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.logger=console \
  distillation.enabled=True \
  distillation.n_gpus_per_node="$TEACHER_GPUS" \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1)) \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_GPU_MEMORY_UTILIZATION" \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk=32 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.streamopd_kv.enabled="$STREAMOPD_KV_ENABLED" \
  distillation.streamopd_kv.token_chunk_size="$TOKEN_CHUNK_SIZE" \
  distillation.streamopd_kv.reverse_chunk_size="$REVERSE_CHUNK_SIZE" \
  distillation.streamopd_kv.reverse_batch_size="$REVERSE_BATCH_SIZE" \
  distillation.streamopd_kv.reverse_batch_max_tokens="$REVERSE_BATCH_MAX_TOKENS" \
  distillation.streamopd_kv.overlap_rollout_training="$OVERLAP_ROLLOUT_TRAINING" \
  distillation.streamopd_kv.rollout_micro_batch_size="$ROLLOUT_MICRO_BATCH_SIZE" \
  distillation.streamopd_kv.colocate_teacher_with_student="$COLOCATE_TEACHER_WITH_STUDENT" \
  "${RAY[@]}" \
  "$@"
