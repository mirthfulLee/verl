#!/usr/bin/env bash
set -euo pipefail

CASE=${CASE:-streamopd-cf}
export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-4B}
export DATASET=${DATASET:?Set DATASET to the DAPO-Math parquet file}
export BATCH_SIZE=${BATCH_SIZE:-128} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-3}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
export TOTAL_TRAJECTORY_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}
export USE_LIGER=${USE_LIGER:-True}
export RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_cf/results}
export CHECKPOINT_HOST_DIR=${CHECKPOINT_HOST_DIR:-/dev/shm/verl-streamopd-cf-$$}
mkdir -p "$RESULT_DIR"

if [[ $CASE == verl-async-opd || $CASE == verl-sync-opd-separate ]]; then
  export NATIVE_MODE=separate_async
  if [[ $CASE == verl-sync-opd-separate ]]; then export NATIVE_MODE=separate_sync; fi
  export STUDENT_GPUS=${STUDENT_GPUS:-2} ROLLOUT_GPUS=${ROLLOUT_GPUS:-1} TEACHER_GPUS=${TEACHER_GPUS:-1}
  exec bash benchmarks/streamopd_cf/run_native_opd.sh "$@"
fi

# Fixed-microbatch ablations for the streaming methods and legacy controls.
# Native baselines above inherit their own packing defaults.
fixed_args=()
if [[ -n ${FIXED_MICRO_BATCH_SIZE:-} ]]; then
  if [[ ! $FIXED_MICRO_BATCH_SIZE =~ ^[1-9][0-9]*$ ]]; then
    echo "FIXED_MICRO_BATCH_SIZE must be a positive integer" >&2
    exit 2
  fi
  export TRAIN_MICRO_BATCH_SIZE=$FIXED_MICRO_BATCH_SIZE
  fixed_args=(
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$FIXED_MICRO_BATCH_SIZE"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
  )
fi

if [[ $CASE == verl-sync-opd || $CASE == streamopd-kv ]]; then
  export MODE=verl-sync-opd
  if [[ $CASE != verl-sync-opd ]]; then export MODE=streamopd-kv; fi
  bash benchmarks/streamopd_kv/run_colocate_case.sh "${fixed_args[@]}" "$@"
  exit
fi
if [[ $CASE != streamopd-cf && $CASE != verl-sync-opd-separate && $CASE != verl-async-opd ]]; then
  echo "CASE must be streamopd-cf, streamopd-kv, verl-sync-opd, verl-sync-opd-separate or verl-async-opd" >&2
  exit 2
fi
export TEACHER_GPUS=${TEACHER_GPUS:-1} ROLLOUT_GPUS=${ROLLOUT_GPUS:-1} STUDENT_GPUS=${STUDENT_GPUS:-2}
export TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
export EXPERIMENT_NAME=streamopd_cf_total${TOTAL_TRAJECTORY_LENGTH}_bs${BATCH_SIZE}

bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_cf_fsdp.sh \
  actor_rollout_ref.model.use_liger="$USE_LIGER" \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.rollout.do_sample=False \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.max_model_len=$((TOTAL_TRAJECTORY_LENGTH + 1)) \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=False \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.host.directory="$CHECKPOINT_HOST_DIR" \
  distillation.teacher_models.teacher_model.inference.dtype=bfloat16 \
  distillation.teacher_models.teacher_model.inference.enforce_eager=False \
  distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True \
  distillation.teacher_models.teacher_model.inference.max_model_len=$((TOTAL_TRAJECTORY_LENGTH + 1)) \
  data.custom_cls.path=benchmarks/streamopd_kv/dapo_math_dataset.py \
  data.dataloader_num_workers=0 \
  distillation.distillation_loss.chunked_topk_chunk_size=512 \
  "${fixed_args[@]}" "$@" 2>&1 | tee "$RESULT_DIR/${EXPERIMENT_NAME}.log"
