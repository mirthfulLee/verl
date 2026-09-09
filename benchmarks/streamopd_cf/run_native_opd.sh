#!/usr/bin/env bash
# Match the OPD workload while inheriting verl's native training and inference defaults.
set -euo pipefail

: "${STUDENT_MODEL:?Set STUDENT_MODEL}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL}"
: "${DATASET:?Set DATASET}"
: "${STUDENT_GPUS:?Set STUDENT_GPUS}"
: "${ROLLOUT_GPUS:?Set ROLLOUT_GPUS}"
: "${TEACHER_GPUS:?Set TEACHER_GPUS}"
: "${RESULT_DIR:?Set RESULT_DIR}"
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
NATIVE_MODE=${NATIVE_MODE:-separate_async}
placement_args=()
case "$NATIVE_MODE" in
  separate_async) placement_args=(trainer.v1.separate_async.parameter_sync_step=1) ;;
  separate_sync) ;;
  union_sync)
    placement_args=(actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True)
    ;;
  *) echo "Unsupported native OPD mode: $NATIVE_MODE" >&2; exit 2 ;;
esac
EXPERIMENT_NAME=${NATIVE_MODE}_total$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))_bs${BATCH_SIZE}
mkdir -p "$RESULT_DIR"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATASET" data.val_files="$DATASET" \
  data.custom_cls.path=examples/on_policy_distillation_trainer/dapo_math_dataset.py \
  data.custom_cls.name=DAPOMathDataset \
  data.train_batch_size="$BATCH_SIZE" data.seed=1 data.shuffle=True \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=False \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.do_sample=False \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1)) \
  actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS" actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=host \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${CHECKPOINT_BUCKET_MB:-128}" \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.host.directory="$RESULT_DIR/checkpoint_host" \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.host.rollout_dtype=bfloat16 \
  trainer.use_v1=True trainer.v1.trainer_mode="$NATIVE_MODE" \
  trainer.n_gpus_per_node="$STUDENT_GPUS" trainer.nnodes=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-4}" \
  trainer.project_name=streamopd_benchmark trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.resume_mode=disable trainer.default_local_dir="$RESULT_DIR/checkpoints" \
  trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 trainer.logger=console \
  distillation.enabled=True distillation.n_gpus_per_node="$TEACHER_GPUS" distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP_SIZE:-1}" \
  distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1)) \
  distillation.distillation_loss.loss_mode=forward_kl_topk distillation.distillation_loss.topk=32 \
  distillation.distillation_loss.use_task_rewards=False distillation.distillation_loss.use_policy_gradient=False \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "${placement_args[@]}" "$@" 2>&1 | tee "$RESULT_DIR/${EXPERIMENT_NAME}.log"
