#!/usr/bin/env bash
set -euo pipefail

: "${RESULT_DIR:?Set a unique RESULT_DIR}"
: "${METHOD:?Set METHOD}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL}"
: "${DATASET:?Set DATASET}"
: "${STUDENT_GPUS:?Set STUDENT_GPUS}"
: "${ROLLOUT_GPUS:?Set ROLLOUT_GPUS}"
: "${TEACHER_GPUS:?Set TEACHER_GPUS}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export BATCH_SIZE=${BATCH_SIZE:-128} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
export MAX_RESPONSE_LENGTH=$((TOTAL_TRAJECTORY_LENGTH - MAX_PROMPT_LENGTH))
export TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
export ROLLOUT_MAX_NUM_SEQS=$(((BATCH_SIZE + ROLLOUT_GPUS - 1) / ROLLOUT_GPUS))
export TEACHER_MAX_NUM_SEQS=32
export ROLLOUT_MAX_BATCHED_TOKENS=$TOTAL_TRAJECTORY_LENGTH
export TEACHER_MAX_BATCHED_TOKENS=$TOTAL_TRAJECTORY_LENGTH
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 VERL_USE_UV=0
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OPD_BENCH_TIMELINE_DIR="$RESULT_DIR/timelines"
export CHECKPOINT_HOST_DIR="$RESULT_DIR/checkpoint_host"
export KV_HANDOFF_DIR="$RESULT_DIR/kv_handoff"
export RAY_TMPDIR="/tmp/opd8-ray-$$"
export TRAIN_MAX_TOKENS_PER_GPU=${TRAIN_MAX_TOKENS_PER_GPU:-0}
export ENABLE_GRADIENT_CHECKPOINTING=True USE_LIGER=True
mkdir -p "$RESULT_DIR"
python3 -c 'import json, os; from pathlib import Path; (Path(os.environ["RESULT_DIR"]) / "runtime.json").write_text(json.dumps({"ray_tmpdir": os.environ["RAY_TMPDIR"]}) + "\n")'

IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
device_count=${#visible_devices[@]}
if [[ $METHOD == *union ]]; then
  ((STUDENT_GPUS == device_count && ROLLOUT_GPUS + TEACHER_GPUS == device_count))
else
  ((STUDENT_GPUS + ROLLOUT_GPUS + TEACHER_GPUS == device_count))
fi
((BATCH_SIZE % STUDENT_GPUS == 0 && TEACHER_GPUS % TEACHER_TP_SIZE == 0))

# Native baselines receive workload controls, not CF/KV runtime tuning.
case "$METHOD" in
  verl-async-opd) export NATIVE_MODE=separate_async ;;
  verl-sync-opd-separate) export NATIVE_MODE=separate_sync ;;
  verl-sync-opd-union) export NATIVE_MODE=union_sync ;;
esac
if [[ $METHOD == verl-* ]]; then
  exec bash benchmarks/streamopd_cf/run_native_opd.sh "$@"
fi

common=(
  data.seed=1 data.shuffle=True trainer.resume_mode=disable
  trainer.default_local_dir="$RESULT_DIR/checkpoints"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${CHECKPOINT_BUCKET_MB:-128}"
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_BATCHED_TOKENS"
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
  distillation.teacher_models.teacher_model.inference.max_num_seqs="$TEACHER_MAX_NUM_SEQS"
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="$TEACHER_MAX_BATCHED_TOKENS"
)
if [[ $METHOD == streamopd-cf ]]; then
  common+=(
    distillation.batching.memory_fraction=0.95
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.8
  )
fi
case "$METHOD" in
  streamopd-kv-union|streamopd-kv-dedicated)
    export MODE=$METHOD
    # Keep the KV inference budget fixed while retaining automatic reverse sizing.
    export STREAMOPD_RUNTIME_PROFILE=manual
    export ROLLOUT_GPU_MEMORY_UTILIZATION=0.85 TEACHER_GPU_MEMORY_UTILIZATION=0.85
    export TOKEN_CHUNK_SIZE=$(( (( (MAX_RESPONSE_LENGTH + 3) / 4 + 63) / 64) * 64 ))
    export TOKEN_CHUNK_SIZE=$(( TOKEN_CHUNK_SIZE < 256 ? 256 : TOKEN_CHUNK_SIZE ))
    export TOKEN_CHUNK_SIZE=$(( TOKEN_CHUNK_SIZE > 1024 ? 1024 : TOKEN_CHUNK_SIZE ))
    export ROLLOUT_KV_EXPORT_CHUNK_SIZE=2048
    export REVERSE_BATCH_SIZE=0 REVERSE_BATCH_MAX_TOKENS=0 REVERSE_CHUNK_SIZE=0
    export TEACHER_PREFILL_MAX_ACTIVE_TRAJECTORIES=0 TEACHER_PREFILL_MAX_ACTIVE_KV_TOKENS=0
    export KV_HANDOFF_DIR="/dev/shm/opd8-kv-$$"
    shared_args=()
    if [[ $METHOD == streamopd-kv-union ]]; then
      export TOKEN_CHUNK_SIZE=1024
      shared_args=(
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
      )
    fi
    bash benchmarks/streamopd_kv/run_colocate_case.sh "${shared_args[@]}" "${common[@]}" "$@"
    ;;
  streamopd-cf)
    export CASE=$METHOD
    bash benchmarks/streamopd_cf/run_case.sh \
      distillation.streamopd_cf.timeline_dir="$OPD_BENCH_TIMELINE_DIR" \
      "${common[@]}" "$@"
    ;;
  *) echo "Unknown METHOD=$METHOD" >&2; exit 2 ;;
esac
