#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/posthoc_fixed_slot_ablation}
TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}

for mode in streamopd-posthoc-legacy streamopd-posthoc-fixed streamopd-posthoc-fixed-wide; do
  case "$mode" in
    streamopd-posthoc-fixed-wide)
      reverse_batch_max_tokens=65536
      ;;
    *)
      reverse_batch_max_tokens=32768
      ;;
  esac
  MODE="$mode" \
    RESULT_DIR="$RESULT_DIR" \
    TOTAL_TRAJECTORY_LENGTH="$TOTAL_TRAJECTORY_LENGTH" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
    REVERSE_BATCH_MAX_TOKENS="$reverse_batch_max_tokens" \
    KV_HANDOFF_DIR="/data1/huanli/tmp/${mode}_total${TOTAL_TRAJECTORY_LENGTH}_mb${MICRO_BATCH_SIZE}" \
    bash benchmarks/streamopd_kv/run_colocate_case.sh
done

python3 benchmarks/streamopd_kv/summarize_colocate_matrix.py "$RESULT_DIR"
