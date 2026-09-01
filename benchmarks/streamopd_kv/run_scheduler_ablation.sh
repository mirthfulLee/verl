#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/scheduler_ablation}
TRAINER_PLACEMENT=${TRAINER_PLACEMENT:-teacher}
TOTAL_TRAJECTORY_LENGTH=${TOTAL_TRAJECTORY_LENGTH:-4096}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}

case "$TRAINER_PLACEMENT" in
  teacher)
    modes=(streamopd-teacher-then-train streamopd-adaptive)
    ;;
  dedicated)
    modes=(streamopd-dedicated-baseline streamopd-dedicated)
    ;;
  rollout)
    modes=(streamopd-rollout-baseline streamopd-rollout)
    ;;
  union)
    modes=(streamopd-union-baseline streamopd-union)
    ;;
  *)
    echo "Unsupported TRAINER_PLACEMENT=$TRAINER_PLACEMENT" >&2
    exit 2
    ;;
esac

for mode in "${modes[@]}"; do
  MODE="$mode" \
    RESULT_DIR="$RESULT_DIR" \
    TOTAL_TRAJECTORY_LENGTH="$TOTAL_TRAJECTORY_LENGTH" \
    TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
    KV_HANDOFF_DIR="/data1/huanli/tmp/${mode}_scheduler_total${TOTAL_TRAJECTORY_LENGTH}" \
    bash benchmarks/streamopd_kv/run_colocate_case.sh
done

python3 benchmarks/streamopd_kv/summarize_colocate_matrix.py "$RESULT_DIR"
