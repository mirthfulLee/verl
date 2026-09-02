#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-streamopd}
TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((TOTAL_TRAJECTORY_LENGTH - MAX_PROMPT_LENGTH))}
RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/colocate_matrix}
BASELINE_AGENT_LOOP_WORKERS=${BASELINE_AGENT_LOOP_WORKERS:-8}
STREAMOPD_AGENT_LOOP_WORKERS=${STREAMOPD_AGENT_LOOP_WORKERS:-8}
BASELINE_ROLLOUT_MAX_NUM_SEQS=${BASELINE_ROLLOUT_MAX_NUM_SEQS:-32}
STREAMOPD_ROLLOUT_MAX_NUM_SEQS=${STREAMOPD_ROLLOUT_MAX_NUM_SEQS:-32}
KV_HANDOFF_DIR=${KV_HANDOFF_DIR:-/data1/huanli/tmp/${MODE}_total${TOTAL_TRAJECTORY_LENGTH}_mb${MICRO_BATCH_SIZE}}

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
export STREAMOPD_RUNTIME_PROFILE=${STREAMOPD_RUNTIME_PROFILE:-manual}
export ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-$TOTAL_TRAJECTORY_LENGTH}
export TEACHER_MAX_BATCHED_TOKENS=${TEACHER_MAX_BATCHED_TOKENS:-2048}
export TEACHER_PREFILL_MAX_ACTIVE_TRAJECTORIES=${TEACHER_PREFILL_MAX_ACTIVE_TRAJECTORIES:-$MICRO_BATCH_SIZE}
export TEACHER_PREFILL_MAX_ACTIVE_KV_TOKENS=${TEACHER_PREFILL_MAX_ACTIVE_KV_TOKENS:-0}
export TEACHER_PREFILL_KV_PAGE_SIZE=${TEACHER_PREFILL_KV_PAGE_SIZE:-64}
export TRAINER_PLACEMENT=${TRAINER_PLACEMENT:-teacher}
export STREAMOPD_SCHEDULER_POLICY=${STREAMOPD_SCHEDULER_POLICY:-adaptive}
export REVERSE_SLOT_MAX_TOKENS=${REVERSE_SLOT_MAX_TOKENS:-$TOTAL_TRAJECTORY_LENGTH}
export REVERSE_SLOT_RESERVE_GIB=${REVERSE_SLOT_RESERVE_GIB:-4.0}
export KV_PREFETCH_DEPTH=${KV_PREFETCH_DEPTH:-1}
export KV_PREFETCH_WORKERS=${KV_PREFETCH_WORKERS:-4}
export TOKEN_CHUNK_SIZE=${TOKEN_CHUNK_SIZE:-$(((MAX_RESPONSE_LENGTH + 1) / 2))}
export REVERSE_CHUNK_SIZE=${REVERSE_CHUNK_SIZE:-0}
export REVERSE_CHUNK_MIN_SIZE=${REVERSE_CHUNK_MIN_SIZE:-0}
export REVERSE_PAGE_SIZE=${REVERSE_PAGE_SIZE:-64}
export REVERSE_BATCH_SIZE=${REVERSE_BATCH_SIZE:-$MICRO_BATCH_SIZE}
export REVERSE_BATCH_MAX_TOKENS=${REVERSE_BATCH_MAX_TOKENS:-$((MICRO_BATCH_SIZE * TOTAL_TRAJECTORY_LENGTH))}

case "$MODE" in
  verl-sync-opd)
    export TRAINER_MODE=sync STREAMOPD_KV_ENABLED=False
    export STUDENT_GPUS=2 TEACHER_GPUS=2 ROLLOUT_NNODES=0
    export AGENT_LOOP_WORKERS="$BASELINE_AGENT_LOOP_WORKERS"
    export ROLLOUT_MAX_NUM_SEQS="$BASELINE_ROLLOUT_MAX_NUM_SEQS"
    export DISTILLATION_COLOCATE_TEACHER_WITH_STUDENT=False CHECKPOINT_ENGINE_BACKEND=naive
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
    export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.55}
    export USE_LIGER=${USE_LIGER:-False} ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
    export TEACHER_ENFORCE_EAGER=${TEACHER_ENFORCE_EAGER:-True}
    export CHECKPOINT_HOST_ROLLOUT_DTYPE=${CHECKPOINT_HOST_ROLLOUT_DTYPE:-null}
    ;;
  verl-colocate-opd)
    export TRAINER_MODE=sync STREAMOPD_KV_ENABLED=False
    export STUDENT_GPUS=4 TEACHER_GPUS=2 ROLLOUT_NNODES=0
    export AGENT_LOOP_WORKERS="$BASELINE_AGENT_LOOP_WORKERS"
    export ROLLOUT_MAX_NUM_SEQS="$BASELINE_ROLLOUT_MAX_NUM_SEQS"
    export DISTILLATION_COLOCATE_TEACHER_WITH_STUDENT=True CHECKPOINT_ENGINE_BACKEND=naive
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
    export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.25}
    export USE_LIGER=${USE_LIGER:-False} ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
    export TEACHER_ENFORCE_EAGER=${TEACHER_ENFORCE_EAGER:-True}
    export CHECKPOINT_HOST_ROLLOUT_DTYPE=${CHECKPOINT_HOST_ROLLOUT_DTYPE:-null}
    ;;
  streamopd|streamopd-adaptive|streamopd-teacher-then-train|streamopd-rollout|streamopd-rollout-baseline|streamopd-union|streamopd-union-baseline|streamopd-dedicated|streamopd-dedicated-baseline)
    export TRAINER_MODE=streamopd STREAMOPD_KV_ENABLED=True
    export STUDENT_GPUS=${STUDENT_GPUS:-2}
    export TEACHER_GPUS=${TEACHER_GPUS:-$STUDENT_GPUS}
    export ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
    export ROLLOUT_NNODES=1
    export AGENT_LOOP_WORKERS="$STREAMOPD_AGENT_LOOP_WORKERS"
    export ROLLOUT_MAX_NUM_SEQS="$STREAMOPD_ROLLOUT_MAX_NUM_SEQS"
    export DISTILLATION_COLOCATE_TEACHER_WITH_STUDENT=False
    export CHECKPOINT_ENGINE_BACKEND=${CHECKPOINT_ENGINE_BACKEND:-host}
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
    # Leave headroom for the colocated trainer's exact-attention recomputation.
    export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.19}
    USE_LIGER=${USE_LIGER:-True}
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
    TEACHER_ENFORCE_EAGER=${TEACHER_ENFORCE_EAGER:-False}
  # Score every committed rollout/KV chunk with one stateful Teacher forward.
    if [[ $MODE == streamopd-teacher-then-train || $MODE == streamopd-rollout-baseline || $MODE == streamopd-union-baseline || $MODE == streamopd-dedicated-baseline ]]; then
      STREAMOPD_SCHEDULER_POLICY=teacher_then_train
    fi
    if [[ $MODE == streamopd-rollout || $MODE == streamopd-rollout-baseline ]]; then
      TRAINER_PLACEMENT=rollout
    fi
    if [[ $MODE == streamopd-union || $MODE == streamopd-union-baseline ]]; then
      TRAINER_PLACEMENT=union
    fi
    if [[ $MODE == streamopd-dedicated || $MODE == streamopd-dedicated-baseline ]]; then
      TRAINER_PLACEMENT=dedicated
    fi
    CHECKPOINT_HOST_ROLLOUT_DTYPE=${CHECKPOINT_HOST_ROLLOUT_DTYPE:-bfloat16}
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$RESULT_DIR"
if [[ $MODE == streamopd* ]]; then
  CASE_NAME="${MODE}_total${TOTAL_TRAJECTORY_LENGTH}_mb${MICRO_BATCH_SIZE}"
else
  CASE_NAME="${MODE}_total${TOTAL_TRAJECTORY_LENGTH}"
fi
LOG_FILE="$RESULT_DIR/${CASE_NAME}.log"
export EXPERIMENT_NAME="$CASE_NAME"

bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1)) \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES" \
  actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_ENGINE_BACKEND" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.multi_sender=False \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=True \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.host.directory=/dev/shm/verl-streamopd-checkpoint \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_liger="$USE_LIGER" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.host.rollout_dtype="$CHECKPOINT_HOST_ROLLOUT_DTYPE" \
  data.dataloader_num_workers=0 \
  distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1)) \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_GPU_MEMORY_UTILIZATION" \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="$TEACHER_MAX_BATCHED_TOKENS" \
  distillation.teacher_models.teacher_model.inference.dtype=bfloat16 \
  distillation.teacher_models.teacher_model.inference.enforce_eager="$TEACHER_ENFORCE_EAGER" \
  distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True \
  distillation.streamopd_kv.runtime_profile="$STREAMOPD_RUNTIME_PROFILE" \
  distillation.streamopd_kv.token_chunk_size="$TOKEN_CHUNK_SIZE" \
  distillation.streamopd_kv.trainer_placement="$TRAINER_PLACEMENT" \
  distillation.streamopd_kv.scheduler_policy="$STREAMOPD_SCHEDULER_POLICY" \
  distillation.streamopd_kv.reverse_chunk_size="$REVERSE_CHUNK_SIZE" \
  distillation.streamopd_kv.reverse_chunk_min_size="$REVERSE_CHUNK_MIN_SIZE" \
  distillation.streamopd_kv.reverse_page_size="$REVERSE_PAGE_SIZE" \
  distillation.streamopd_kv.reverse_batch_size="$REVERSE_BATCH_SIZE" \
  distillation.streamopd_kv.reverse_batch_max_tokens="$REVERSE_BATCH_MAX_TOKENS" \
  distillation.streamopd_kv.reverse_slot_max_tokens="$REVERSE_SLOT_MAX_TOKENS" \
  distillation.streamopd_kv.reverse_slot_reserve_gib="$REVERSE_SLOT_RESERVE_GIB" \
  distillation.streamopd_kv.teacher_prefill_max_active_trajectories="$TEACHER_PREFILL_MAX_ACTIVE_TRAJECTORIES" \
  distillation.streamopd_kv.teacher_prefill_max_active_kv_tokens="$TEACHER_PREFILL_MAX_ACTIVE_KV_TOKENS" \
  distillation.streamopd_kv.teacher_prefill_kv_page_size="$TEACHER_PREFILL_KV_PAGE_SIZE" \
  distillation.streamopd_kv.kv_prefetch_depth="$KV_PREFETCH_DEPTH" \
  distillation.streamopd_kv.kv_prefetch_workers="$KV_PREFETCH_WORKERS" \
  actor_rollout_ref.rollout.do_sample=False \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS" \
  distillation.streamopd_kv.kv_handoff_dir="$KV_HANDOFF_DIR" \
  "$@" 2>&1 | tee "$LOG_FILE"
