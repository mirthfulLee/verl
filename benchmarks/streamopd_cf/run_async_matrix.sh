#!/usr/bin/env bash
set -euo pipefail

# One-factor coverage around the default, deliberately not a Cartesian product.
MODEL_ROOT=${MODEL_ROOT:?Set MODEL_ROOT to the directory containing Qwen3 models}
MATRIX_RESULT_DIR=${MATRIX_RESULT_DIR:-benchmarks/streamopd_cf/results/async_matrix}
MATRIX_SETTINGS=${MATRIX_SETTINGS:-"default model tokens batch"}
MATRIX_METHODS=${MATRIX_METHODS:-"streamopd-cf verl-sync-opd-separate verl-async-opd"}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
export MAX_PROMPT_LENGTH=1024
mkdir -p "$MATRIX_RESULT_DIR"

for setting in $MATRIX_SETTINGS; do
  export STUDENT_MODEL="$MODEL_ROOT/Qwen3-1.7B" TEACHER_MODEL="$MODEL_ROOT/Qwen3-4B"
  export MAX_RESPONSE_LENGTH=3072 BATCH_SIZE=128
  export ASYNC_TRAIN_MAX_TOKENS_PER_GPU=16384
  case "$setting" in
    default) ;;
    model)
      export STUDENT_MODEL="$MODEL_ROOT/Qwen3-4B" TEACHER_MODEL="$MODEL_ROOT/Qwen3-14B"
      export ASYNC_TRAIN_MAX_TOKENS_PER_GPU=8192
      ;;
    tokens) export MAX_RESPONSE_LENGTH=7168 ;;
    batch) export BATCH_SIZE=256 ;;
    *) echo "Unknown matrix setting: $setting" >&2; exit 2 ;;
  esac
  for method in $MATRIX_METHODS; do
    export CASE=$method RESULT_DIR="$MATRIX_RESULT_DIR/$setting/$method"
    export CHECKPOINT_HOST_DIR="/dev/shm/opd-matrix-$setting-$method-$$"
    mkdir -p "$RESULT_DIR"
    uv run --active --no-sync python -c '
import json, os
from pathlib import Path
keys = ("CASE", "STUDENT_MODEL", "TEACHER_MODEL", "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH",
        "BATCH_SIZE", "TOTAL_TRAINING_STEPS", "ASYNC_TRAIN_MAX_TOKENS_PER_GPU", "CUDA_VISIBLE_DEVICES",
        "FORWARD_CHUNK_SIZE", "TOKEN_CHUNK_SIZE", "LOSS_CHUNK_SIZE", "TRAIN_MICRO_BATCH_SIZE",
        "TRAIN_MAX_TOKENS_PER_GPU", "ROLLOUT_MAX_NUM_SEQS", "TEACHER_MAX_NUM_SEQS",
        "ROLLOUT_MAX_BATCHED_TOKENS", "TEACHER_MAX_BATCHED_TOKENS")
Path(os.environ["RESULT_DIR"], "settings.json").write_text(json.dumps({k: os.environ.get(k) for k in keys}, indent=2))
'
    uv run --active --no-sync python -m benchmarks.streamopd_cf.profile_run \
      --output "$RESULT_DIR/gpu.csv" --devices "${CUDA_VISIBLE_DEVICES:?Set the four benchmark GPUs}" --require-idle -- \
      bash benchmarks/streamopd_cf/run_case.sh \
      trainer.default_local_dir="$RESULT_DIR/checkpoints" trainer.resume_mode=disable \
      distillation.streamopd_cf.timeline_dir="$RESULT_DIR/timelines" "$@"
  done
done
