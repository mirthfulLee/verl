#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-streamopd}
export STUDENT_MODEL=${STUDENT_MODEL:-/nasdata/Model/Qwen3-1.7B}
export TEACHER_MODEL=${TEACHER_MODEL:-/nasdata/Model/Qwen3-4B}
export DATASET=${DATASET:-/nasdata/Model/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}
export TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-128}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}
TOPK_CHUNK_SIZE=${TOPK_CHUNK_SIZE:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((TOTAL_TRAJECTORY_LENGTH - MAX_PROMPT_LENGTH))}
RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/colocate_matrix}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-8}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-0}
KV_HANDOFF_DIR=${KV_HANDOFF_DIR:-/data1/huanli/tmp/${MODE}_total${TOTAL_TRAJECTORY_LENGTH}_mb${MICRO_BATCH_SIZE}}
CHECKPOINT_HOST_DIR=${CHECKPOINT_HOST_DIR:-/dev/shm/verl-streamopd-checkpoint-${MODE}-$$}
export USE_LIGER=${USE_LIGER:-True}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
export TEACHER_ENFORCE_EAGER=${TEACHER_ENFORCE_EAGER:-False}

if [[ ${CLEANUP_HANDOFF_DIR:-1} == 1 ]]; then
  case "$KV_HANDOFF_DIR" in
    /|/tmp|/data1|/data1/huanli|/data1/huanli/tmp|"$PWD")
      echo "Refusing to clean broad KV_HANDOFF_DIR=$KV_HANDOFF_DIR" >&2
      exit 2
      ;;
  esac
  # Each benchmark case owns a deterministic handoff directory.  Removing
  # Stale slot files from an interrupted run must not enter a later pool.
  rm -rf -- "$KV_HANDOFF_DIR"
fi
mkdir -p "$KV_HANDOFF_DIR"

export TOTAL_TRAJECTORY_LENGTH MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH BATCH_SIZE
export STREAMOPD_RAGGED_RESPONSE_LENGTHS=${STREAMOPD_RAGGED_RESPONSE_LENGTHS:-}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
export VERL_USE_UV=${VERL_USE_UV:-0}
export STREAMOPD_RUNTIME_PROFILE=${STREAMOPD_RUNTIME_PROFILE:-auto}
export ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-$TOTAL_TRAJECTORY_LENGTH}
derived_teacher_batched_tokens=$(( ((TOTAL_TRAJECTORY_LENGTH + 255) / 256) * 256 ))
derived_teacher_batched_tokens=$(( derived_teacher_batched_tokens < 2048 ? 2048 : derived_teacher_batched_tokens ))
derived_teacher_batched_tokens=$(( derived_teacher_batched_tokens > 8192 ? 8192 : derived_teacher_batched_tokens ))
export TEACHER_MAX_BATCHED_TOKENS=${TEACHER_MAX_BATCHED_TOKENS:-$derived_teacher_batched_tokens}
export TEACHER_PREFILL_MAX_ACTIVE_TRAJECTORIES=${TEACHER_PREFILL_MAX_ACTIVE_TRAJECTORIES:-$MICRO_BATCH_SIZE}
export TEACHER_PREFILL_MAX_ACTIVE_KV_TOKENS=${TEACHER_PREFILL_MAX_ACTIVE_KV_TOKENS:-0}
export TEACHER_PREFILL_KV_PAGE_SIZE=${TEACHER_PREFILL_KV_PAGE_SIZE:-64}
export TRAINER_PLACEMENT=${TRAINER_PLACEMENT:-union}
export REVERSE_SLOT_MAX_TOKENS=${REVERSE_SLOT_MAX_TOKENS:-$TOTAL_TRAJECTORY_LENGTH}
export REVERSE_SLOT_RESERVE_GIB=${REVERSE_SLOT_RESERVE_GIB:-4.0}
export KV_PREFETCH_DEPTH=${KV_PREFETCH_DEPTH:-1}
export KV_PREFETCH_WORKERS=${KV_PREFETCH_WORKERS:-4}
export TOKEN_CHUNK_SIZE=${TOKEN_CHUNK_SIZE:-$(((MAX_RESPONSE_LENGTH + 1) / 2))}
export ROLLOUT_KV_EXPORT_STRATEGY=${ROLLOUT_KV_EXPORT_STRATEGY:-eos_host}
export ROLLOUT_KV_EXPORT_CHUNK_SIZE=${ROLLOUT_KV_EXPORT_CHUNK_SIZE:-0}
export ROLLOUT_KV_WRITER_THREADS=${ROLLOUT_KV_WRITER_THREADS:-4}
export REVERSE_CHUNK_SIZE=${REVERSE_CHUNK_SIZE:-0}
export REVERSE_CHUNK_MIN_SIZE=${REVERSE_CHUNK_MIN_SIZE:-0}
export REVERSE_PAGE_SIZE=${REVERSE_PAGE_SIZE:-64}
export REVERSE_BATCH_SIZE=${REVERSE_BATCH_SIZE:-$MICRO_BATCH_SIZE}
export REVERSE_BATCH_MAX_TOKENS=${REVERSE_BATCH_MAX_TOKENS:-$((MICRO_BATCH_SIZE * TOTAL_TRAJECTORY_LENGTH))}

case "$MODE" in
  verl-sync-opd)
    export TRAINER_MODE=sync STREAMOPD_KV_ENABLED=False
    export STUDENT_GPUS=${STUDENT_GPUS:-2} TEACHER_GPUS=${TEACHER_GPUS:-2} ROLLOUT_NNODES=0
    export DISTILLATION_COLOCATE_TEACHER_WITH_STUDENT=False CHECKPOINT_ENGINE_BACKEND=naive
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
    export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}
    export CHECKPOINT_HOST_ROLLOUT_DTYPE=${CHECKPOINT_HOST_ROLLOUT_DTYPE:-null}
    ;;
  streamopd|streamopd-teacher|streamopd-rollout|streamopd-union|streamopd-dedicated)
    export TRAINER_MODE=streamopd STREAMOPD_KV_ENABLED=True
    export TEACHER_GPUS=${TEACHER_GPUS:-2}
    export ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
    export ROLLOUT_NNODES=1
    export DISTILLATION_COLOCATE_TEACHER_WITH_STUDENT=False
    export CHECKPOINT_ENGINE_BACKEND=${CHECKPOINT_ENGINE_BACKEND:-host}
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
    export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.90}
    if [[ $MODE == streamopd ]]; then
      TRAINER_PLACEMENT=union
      STUDENT_GPUS=${STUDENT_GPUS:-$((TEACHER_GPUS + ROLLOUT_GPUS))}
    fi
    if [[ $MODE == streamopd-teacher ]]; then
      TRAINER_PLACEMENT=teacher
      STUDENT_GPUS=${STUDENT_GPUS:-$TEACHER_GPUS}
    fi
    if [[ $MODE == streamopd-rollout ]]; then
      TRAINER_PLACEMENT=rollout
      STUDENT_GPUS=${STUDENT_GPUS:-$ROLLOUT_GPUS}
    fi
    if [[ $MODE == streamopd-union ]]; then
      TRAINER_PLACEMENT=union
      STUDENT_GPUS=${STUDENT_GPUS:-$((TEACHER_GPUS + ROLLOUT_GPUS))}
    fi
    if [[ $MODE == streamopd-dedicated ]]; then
      TRAINER_PLACEMENT=dedicated
      STUDENT_GPUS=${STUDENT_GPUS:-2}
    fi
    CHECKPOINT_HOST_ROLLOUT_DTYPE=${CHECKPOINT_HOST_ROLLOUT_DTYPE:-bfloat16}
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

if (( TEACHER_TP_SIZE < 1 || TEACHER_GPUS < TEACHER_TP_SIZE || TEACHER_GPUS % TEACHER_TP_SIZE != 0 )); then
  echo "TEACHER_GPUS must contain a whole number of TEACHER_TP_SIZE replicas" >&2
  exit 2
fi
teacher_replicas=$((TEACHER_GPUS / TEACHER_TP_SIZE))
if [[ -z ${TEACHER_MAX_NUM_SEQS:-} ]]; then
  teacher_batch_per_replica=$(( (BATCH_SIZE + teacher_replicas - 1) / teacher_replicas ))
  TEACHER_MAX_NUM_SEQS=$(( teacher_batch_per_replica < 32 ? teacher_batch_per_replica : 32 ))
fi
export TEACHER_MAX_NUM_SEQS

if (( ROLLOUT_MAX_NUM_SEQS < 1 )); then
  if [[ $MODE == verl-sync-opd ]]; then
    rollout_replicas=$STUDENT_GPUS
  else
    rollout_replicas=$ROLLOUT_GPUS
  fi
  ROLLOUT_MAX_NUM_SEQS=$(( (BATCH_SIZE + rollout_replicas - 1) / rollout_replicas ))
fi

mkdir -p "$RESULT_DIR"
if [[ $MODE == streamopd* ]]; then
  CASE_NAME="${MODE}_total${TOTAL_TRAJECTORY_LENGTH}_bs${BATCH_SIZE}_mb${MICRO_BATCH_SIZE}"
else
  CASE_NAME="${MODE}_total${TOTAL_TRAJECTORY_LENGTH}_bs${BATCH_SIZE}"
fi
LOG_FILE="$RESULT_DIR/${CASE_NAME}.log"
export EXPERIMENT_NAME="$CASE_NAME"

matched_inference_overrides=(
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
  distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
  distillation.teacher_models.teacher_model.inference.max_num_seqs="$TEACHER_MAX_NUM_SEQS"
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="$TEACHER_MAX_BATCHED_TOKENS"
)

profile_overrides=(
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_ENGINE_BACKEND"
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.multi_sender=False
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=True
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_GPU_MEMORY_UTILIZATION"
  distillation.streamopd_kv.runtime_profile="$STREAMOPD_RUNTIME_PROFILE"
  distillation.streamopd_kv.token_chunk_size="$TOKEN_CHUNK_SIZE"
  distillation.streamopd_kv.rollout_kv_export_strategy="$ROLLOUT_KV_EXPORT_STRATEGY"
  distillation.streamopd_kv.rollout_kv_export_chunk_size="$ROLLOUT_KV_EXPORT_CHUNK_SIZE"
  distillation.streamopd_kv.rollout_kv_writer_threads="$ROLLOUT_KV_WRITER_THREADS"
  distillation.streamopd_kv.reverse_chunk_size="$REVERSE_CHUNK_SIZE"
  distillation.streamopd_kv.reverse_chunk_min_size="$REVERSE_CHUNK_MIN_SIZE"
  distillation.streamopd_kv.reverse_page_size="$REVERSE_PAGE_SIZE"
  distillation.streamopd_kv.reverse_batch_size="$REVERSE_BATCH_SIZE"
  distillation.streamopd_kv.reverse_batch_max_tokens="$REVERSE_BATCH_MAX_TOKENS"
  distillation.streamopd_kv.reverse_slot_max_tokens="$REVERSE_SLOT_MAX_TOKENS"
  distillation.streamopd_kv.reverse_slot_reserve_gib="$REVERSE_SLOT_RESERVE_GIB"
  distillation.streamopd_kv.teacher_prefill_max_active_trajectories="$TEACHER_PREFILL_MAX_ACTIVE_TRAJECTORIES"
  distillation.streamopd_kv.teacher_prefill_max_active_kv_tokens="$TEACHER_PREFILL_MAX_ACTIVE_KV_TOKENS"
  distillation.streamopd_kv.teacher_prefill_kv_page_size="$TEACHER_PREFILL_KV_PAGE_SIZE"
  distillation.streamopd_kv.kv_prefetch_depth="$KV_PREFETCH_DEPTH"
  distillation.streamopd_kv.kv_prefetch_workers="$KV_PREFETCH_WORKERS"
  distillation.streamopd_kv.kv_handoff_dir="$KV_HANDOFF_DIR"
)
if [[ $MODE == streamopd* && $STREAMOPD_RUNTIME_PROFILE == auto ]]; then
  profile_overrides=(
    distillation.streamopd_kv.runtime_profile=auto
    distillation.streamopd_kv.rollout_kv_export_strategy="$ROLLOUT_KV_EXPORT_STRATEGY"
  )
fi

bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh \
  trainer.v1.trainer_mode="$TRAINER_MODE" \
  distillation.streamopd_kv.enabled="$STREAMOPD_KV_ENABLED" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES" \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.host.directory="$CHECKPOINT_HOST_DIR" \
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
  actor_rollout_ref.model.use_liger="$USE_LIGER" \
  actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING" \
  +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPLEMENTATION" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.host.rollout_dtype="$CHECKPOINT_HOST_ROLLOUT_DTYPE" \
  data.custom_cls.path=benchmarks/streamopd_kv/dapo_math_dataset.py \
  data.dataloader_num_workers=0 \
  distillation.teacher_models.teacher_model.inference.dtype=bfloat16 \
  distillation.teacher_models.teacher_model.inference.enforce_eager="$TEACHER_ENFORCE_EAGER" \
  distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True \
  distillation.distillation_loss.use_chunked_topk=True \
  distillation.distillation_loss.chunked_topk_chunk_size="$TOPK_CHUNK_SIZE" \
  distillation.streamopd_kv.trainer_placement="$TRAINER_PLACEMENT" \
  actor_rollout_ref.rollout.do_sample=False \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS" \
  "${matched_inference_overrides[@]}" \
  "${profile_overrides[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
