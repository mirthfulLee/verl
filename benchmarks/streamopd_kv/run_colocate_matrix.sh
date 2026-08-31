#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/colocate_matrix}
BASELINE_DIR=${BASELINE_DIR:-}
MICROBATCH_SIZES=${MICROBATCH_SIZES:-"16 32"}
TOKEN_LENGTHS=${TOKEN_LENGTHS:-"4096 8192"}
COLOCATE_NCCL_P2P_LEVEL=${COLOCATE_NCCL_P2P_LEVEL:-}
failed_cases=()

run_case() {
  local mode=$1
  local total_tokens=$2
  local microbatch=$3

  local -a command=(bash benchmarks/streamopd_kv/run_colocate_case.sh)
  if [[ $mode == verl-colocate-opd && -n $COLOCATE_NCCL_P2P_LEVEL ]]; then
    command=(env NCCL_P2P_LEVEL="$COLOCATE_NCCL_P2P_LEVEL" "${command[@]}")
  fi
  if ! MODE=$mode TOTAL_TRAJECTORY_LENGTH=$total_tokens MICRO_BATCH_SIZE=$microbatch RESULT_DIR=$RESULT_DIR \
    "${command[@]}"; then
    failed_cases+=("${mode}/total${total_tokens}/mb${microbatch}")
  fi
}

for total_tokens in $TOKEN_LENGTHS; do
  for baseline in verl-sync-opd verl-colocate-opd; do
    run_case "$baseline" "$total_tokens" 32
  done
  for microbatch in $MICROBATCH_SIZES; do
    run_case streamopd-colocate "$total_tokens" "$microbatch"
  done
done

summary_args=("$RESULT_DIR")
if [[ -n $BASELINE_DIR ]]; then
  summary_args+=(--baseline-dir "$BASELINE_DIR")
fi
python3 benchmarks/streamopd_kv/summarize_colocate_matrix.py "${summary_args[@]}"

if ((${#failed_cases[@]})); then
  printf 'Failed cases:\n' >&2
  printf '  %s\n' "${failed_cases[@]}" >&2
  exit 1
fi
