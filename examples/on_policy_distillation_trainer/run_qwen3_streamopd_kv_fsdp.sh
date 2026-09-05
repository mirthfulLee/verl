#!/usr/bin/env bash
set -xeuo pipefail

STUDENT_MODEL=${STUDENT_MODEL:-/nasdata/Model/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-/nasdata/Model/Qwen3-4B}
DATASET=${DATASET:-/nasdata/Model/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}
# Aliases for verl's existing resource-pool options.
TEACHER_GPUS=${TEACHER_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-False}
USE_NO_SYNC_FOR_GRADIENT_ACCUMULATION=${USE_NO_SYNC_FOR_GRADIENT_ACCUMULATION:-True}
TRAINER_PLACEMENT=${TRAINER_PLACEMENT:-union}
if [[ $TRAINER_PLACEMENT == union ]]; then
  STUDENT_GPUS=${STUDENT_GPUS:-$((TEACHER_GPUS + ROLLOUT_GPUS))}
else
  STUDENT_GPUS=${STUDENT_GPUS:-2}
fi

# OPD algorithm settings.
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}

# Compatibility hooks used by the matched baseline harness. Normal StreamOPD
# runs leave these defaults unchanged; advanced studies use Hydra overrides.
DISTILLATION_COLOCATE_TEACHER_WITH_STUDENT=${DISTILLATION_COLOCATE_TEACHER_WITH_STUDENT:-False}
TRAINER_MODE=${TRAINER_MODE:-streamopd}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
STREAMOPD_KV_ENABLED=${STREAMOPD_KV_ENABLED:-True}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-streamopd_kv_qwen3}

if (( MAX_PROMPT_LENGTH < 1 || MAX_RESPONSE_LENGTH < 1 )); then
  echo "MAX_PROMPT_LENGTH and MAX_RESPONSE_LENGTH must be positive" >&2
  exit 2
fi

LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
  if [[ -n ${VIRTUAL_ENV:-} ]]; then
    LAUNCH=(uv run --active --no-sync python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv run --active --no-sync")
  else
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
  fi
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
  actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation="$USE_NO_SYNC_FOR_GRADIENT_ACCUMULATION" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS" \
  actor_rollout_ref.rollout.nnodes=1 \
  trainer.use_v1=True \
  trainer.v1.trainer_mode="$TRAINER_MODE" \
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
  distillation.colocate_teacher_with_student="$DISTILLATION_COLOCATE_TEACHER_WITH_STUDENT" \
  distillation.n_gpus_per_node="$TEACHER_GPUS" \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$TEACHER_TP_SIZE" \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk=32 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.streamopd_kv.enabled="$STREAMOPD_KV_ENABLED" \
  distillation.streamopd_kv.trainer_placement="$TRAINER_PLACEMENT" \
  "${RAY[@]}" \
  "$@"
