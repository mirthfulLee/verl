#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-streamopd-colocate}
TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((TOTAL_TRAJECTORY_LENGTH - MAX_PROMPT_LENGTH))}
RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/colocate_matrix}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-8}
KV_HANDOFF_DIR=${KV_HANDOFF_DIR:-/data1/huanli/tmp/${MODE}_total${TOTAL_TRAJECTORY_LENGTH}_mb${MICRO_BATCH_SIZE}}

if [[ ${CLEANUP_HANDOFF_DIR:-1} == 1 ]]; then
  case "$KV_HANDOFF_DIR" in
    /|/tmp|/data1|/data1/huanli|/data1/huanli/tmp|"$PWD")
      echo "Refusing to clean broad KV_HANDOFF_DIR=$KV_HANDOFF_DIR" >&2
      exit 2
      ;;
  esac
  # Each benchmark case owns a deterministic handoff directory.  Removing
  # stale chunks before a rerun prevents an interrupted case from being read
  # as part of a later manifest.
  rm -rf -- "$KV_HANDOFF_DIR"
fi
mkdir -p "$KV_HANDOFF_DIR"

export TOTAL_TRAJECTORY_LENGTH MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH BATCH_SIZE
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
export VERL_USE_UV=${VERL_USE_UV:-0}
export ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-$TOTAL_TRAJECTORY_LENGTH}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-24}
export TEACHER_MAX_BATCHED_TOKENS=${TEACHER_MAX_BATCHED_TOKENS:-2048}
export TOKEN_CHUNK_SIZE=${TOKEN_CHUNK_SIZE:-$(((MAX_RESPONSE_LENGTH + 1) / 2))}
export REVERSE_CHUNK_SIZE=${REVERSE_CHUNK_SIZE:-1024}
export REVERSE_CHUNK_MIN_SIZE=${REVERSE_CHUNK_MIN_SIZE:-256}
export REVERSE_PAGE_SIZE=${REVERSE_PAGE_SIZE:-64}
export REVERSE_BATCH_SIZE=${REVERSE_BATCH_SIZE:-16}
export REVERSE_BATCH_MAX_TOKENS=${REVERSE_BATCH_MAX_TOKENS:-32768}
export ROLLOUT_MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE

case "$MODE" in
  verl-sync-opd)
    export TRAINER_MODE=sync STREAMOPD_KV_ENABLED=False
    export STUDENT_GPUS=2 TEACHER_GPUS=2 ROLLOUT_NNODES=0
    export COLOCATE_TEACHER_WITH_STUDENT=False CHECKPOINT_ENGINE_BACKEND=naive
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
    export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.55}
    ;;
  verl-colocate-opd)
    export TRAINER_MODE=sync STREAMOPD_KV_ENABLED=False
    export STUDENT_GPUS=4 TEACHER_GPUS=2 ROLLOUT_NNODES=0
    export COLOCATE_TEACHER_WITH_STUDENT=True CHECKPOINT_ENGINE_BACKEND=naive
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
    export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.25}
    ;;
  streamopd-colocate)
    export TRAINER_MODE=streamopd_colocate STREAMOPD_KV_ENABLED=True
    export STUDENT_GPUS=2 TEACHER_GPUS=2 ROLLOUT_GPUS=2 ROLLOUT_NNODES=1
    export COLOCATE_TEACHER_WITH_STUDENT=True CHECKPOINT_ENGINE_BACKEND=host
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
    # Leave headroom for the colocated trainer's exact-attention recomputation.
    export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.19}
    export TEACHER_PRIORITY_THRESHOLD=${TEACHER_PRIORITY_THRESHOLD:-4}
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$RESULT_DIR"
LOG_FILE="$RESULT_DIR/${MODE}_total${TOTAL_TRAJECTORY_LENGTH}_mb${MICRO_BATCH_SIZE}.log"
export EXPERIMENT_NAME="${MODE}_total${TOTAL_TRAJECTORY_LENGTH}_mb${MICRO_BATCH_SIZE}"

bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \
  data.dataloader_num_workers=0 \
  distillation.teacher_models.teacher_model.inference.dtype=bfloat16 \
  distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True \
  actor_rollout_ref.rollout.do_sample=False \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS" \
  distillation.streamopd_kv.kv_handoff_dir="$KV_HANDOFF_DIR" \
  "$@" 2>&1 | tee "$LOG_FILE"
