#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/posthoc_ablation}
TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}

for mode in streamopd streamopd-posthoc; do
  MODE="$mode" \
    RESULT_DIR="$RESULT_DIR" \
    TOTAL_TRAJECTORY_LENGTH="$TOTAL_TRAJECTORY_LENGTH" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
    KV_HANDOFF_DIR="/data1/huanli/tmp/${mode}_posthoc_ablation_total${TOTAL_TRAJECTORY_LENGTH}_mb${MICRO_BATCH_SIZE}" \
    bash benchmarks/streamopd_kv/run_colocate_case.sh
done

python3 benchmarks/streamopd_kv/summarize_colocate_matrix.py "$RESULT_DIR"
