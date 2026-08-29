#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/verl_sync_comparison}
mkdir -p "$RESULT_DIR"

# The matched denominator is this repository's V1 Native OPD sync path.
STREAMOPD_KV_ENABLED=False \
EXPERIMENT_NAME=verl_sync_opd \
bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh "$@" \
  2>&1 | tee "$RESULT_DIR/verl_sync_opd.log"

STREAMOPD_KV_ENABLED=True \
EXPERIMENT_NAME=streamopd_kv \
bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh "$@" \
  2>&1 | tee "$RESULT_DIR/streamopd_kv.log"
