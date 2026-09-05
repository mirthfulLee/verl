#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=${RESULT_DIR:-benchmarks/streamopd_kv/results/pool_matrix}
BATCH_SIZES=${BATCH_SIZES:-"128 256"}
TOKEN_LENGTHS=${TOKEN_LENGTHS:-"4096 8192"}
MODEL_PAIRS=${MODEL_PAIRS:-"1.7B:4B:1 4B:14B:2"}
POOL_SCHEMES=${POOL_SCHEMES:-"union dedicated"}
COVERING_CASES=${COVERING_CASES:-"1.7B:4B:1:4096:128:union 1.7B:4B:1:8192:256:dedicated 4B:14B:2:8192:128:union 4B:14B:2:4096:256:dedicated"}
TOPK_CHUNK_SIZE=${TOPK_CHUNK_SIZE:-512}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}
MATCHED_TEACHER_MAX_NUM_SEQS=${MATCHED_TEACHER_MAX_NUM_SEQS:-32}
failed_cases=()

run_case() {
  local topology=$1
  local implementation=$2
  local total_tokens=$3
  local batch_size=$4
  local student_name=$5
  local teacher_name=$6
  local teacher_tp=$7
  local mode
  local trainer_gpus
  local teacher_gpus=2
  local rollout_gpus=2

  case "$topology:$implementation" in
    union:baseline)
      mode=verl-sync-opd
      trainer_gpus=2
      ;;
    union:streamopd)
      mode=streamopd-union
      trainer_gpus=4
      ;;
    dedicated:baseline)
      mode=verl-sync-opd
      trainer_gpus=4
      ;;
    dedicated:streamopd)
      mode=streamopd-dedicated
      trainer_gpus=2
      ;;
    *)
      echo "Unsupported topology/implementation: $topology/$implementation" >&2
      return 2
      ;;
  esac

  local case_dir="$RESULT_DIR/qwen3-${student_name}_qwen3-${teacher_name}/$topology"
  local log_file
  local -a model_overrides=(
    distillation.distillation_loss.use_chunked_topk=True
    distillation.distillation_loss.chunked_topk_chunk_size="$TOPK_CHUNK_SIZE"
    distillation.teacher_models.teacher_model.inference.max_num_seqs="$MATCHED_TEACHER_MAX_NUM_SEQS"
    actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING"
  )
  if [[ $implementation == streamopd ]]; then
    log_file="$case_dir/${mode}_total${total_tokens}_bs${batch_size}_mb${MICRO_BATCH_SIZE:-16}.log"
  else
    log_file="$case_dir/${mode}_total${total_tokens}_bs${batch_size}.log"
  fi
  if [[ ${SKIP_COMPLETED:-0} == 1 && -f $log_file ]]; then
    local completed_steps
    completed_steps=$(grep -Ec 'step:[0-9]+ -' "$log_file" || true)
    if ((completed_steps >= ${TOTAL_TRAINING_STEPS:-2})); then
      echo "Skipping completed case: $log_file"
      return
    fi
  fi
  local handoff="/dev/shm/verl-streamopd-${topology}-${student_name}-${teacher_name}-${total_tokens}-${batch_size}"
  if ! MODE=$mode \
    STUDENT_MODEL="/nasdata/Model/Qwen3-${student_name}" \
    TEACHER_MODEL="/nasdata/Model/Qwen3-${teacher_name}" \
    TEACHER_TP_SIZE=$teacher_tp \
    STUDENT_GPUS=$trainer_gpus \
    TEACHER_GPUS=$teacher_gpus \
    ROLLOUT_GPUS=$rollout_gpus \
    TOTAL_TRAJECTORY_LENGTH=$total_tokens \
    BATCH_SIZE=$batch_size \
    RESULT_DIR="$case_dir" \
    KV_HANDOFF_DIR="$handoff" \
    bash benchmarks/streamopd_kv/run_colocate_case.sh "${model_overrides[@]}"; then
    failed_cases+=("${student_name}/${teacher_name}/${topology}/${implementation}/total${total_tokens}/bs${batch_size}")
  fi
}

if [[ ${FULL_MATRIX:-0} == 1 ]]; then
  for model_pair in $MODEL_PAIRS; do
    IFS=: read -r student_name teacher_name teacher_tp <<< "$model_pair"
    for total_tokens in $TOKEN_LENGTHS; do
      for batch_size in $BATCH_SIZES; do
        for topology in $POOL_SCHEMES; do
          run_case "$topology" baseline "$total_tokens" "$batch_size" "$student_name" "$teacher_name" "$teacher_tp"
          run_case "$topology" streamopd "$total_tokens" "$batch_size" "$student_name" "$teacher_name" "$teacher_tp"
        done
      done
    done
  done
else
  for case_spec in $COVERING_CASES; do
    IFS=: read -r student_name teacher_name teacher_tp total_tokens batch_size topology <<< "$case_spec"
    run_case "$topology" baseline "$total_tokens" "$batch_size" "$student_name" "$teacher_name" "$teacher_tp"
    run_case "$topology" streamopd "$total_tokens" "$batch_size" "$student_name" "$teacher_name" "$teacher_tp"
  done
fi

find "$RESULT_DIR" -mindepth 2 -maxdepth 2 -type d -print0 | while IFS= read -r -d '' case_dir; do
  .venv-cu128/bin/python benchmarks/streamopd_kv/summarize_colocate_matrix.py "$case_dir"
done

if ((${#failed_cases[@]})); then
  printf 'Failed cases:\n' >&2
  printf '  %s\n' "${failed_cases[@]}" >&2
  exit 1
fi
