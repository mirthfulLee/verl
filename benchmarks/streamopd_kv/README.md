# StreamOPD benchmark

The end-to-end comparison uses this repository's V1 Native OPD as the sync baseline. StreamOPD supports four
physical Trainer placements: shared with Teacher GPUs, shared with Rollout GPUs, spanning disjoint Teacher and
Rollout subsets (`union`), or dedicated GPUs. Teacher and Rollout remain separate resident model processes in every
placement.

StreamOPD is under active development. Benchmark outputs under `benchmarks/streamopd_kv/results/` remain local
because scheduler, memory-planning, and transport changes can quickly invalidate them. Once the method stabilizes,
publication-quality comparisons and their complete experimental settings should be recorded in dedicated
documentation.

## Environment

The development benchmark environment uses Python 3.12, PyTorch with CUDA 12.8, vLLM 0.15.1, and A100-80GB GPUs.
Activate the CUDA environment for the driver and Ray workers:

```bash
export PATH="$PWD/.venv-cu128/bin:$PATH"
export VIRTUAL_ENV="$PWD/.venv-cu128"
export VERL_USE_UV=0
```

The benchmark harness intentionally sets `distillation.streamopd_kv.runtime_profile=manual` so controlled ablations
can pin the compared settings. This is not the normal user interface. The production example defaults to `auto` and
reuses verl's existing Student/Trainer, Teacher, and Rollout resource options. StreamOPD adds no separate GPU-count
configuration; its only optional default-path topology setting is `trainer_placement`.

The normal auto profile jointly selects Rollout `gpu_memory_utilization` and `max_num_seqs` from model weights, KV
bytes per token, maximum trajectory length, physical GPU memory, pool placement, and any explicit user constraint.
Reverse batch width and chunk size are selected by startup preflight and remain fixed inside the policy loop.

## End-to-end matrix

Run the 4096/8192-token matrix with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=128 TOTAL_TRAINING_STEPS=1 \
  bash benchmarks/streamopd_kv/run_colocate_matrix.sh
```

The benchmark wrapper calls the current path `streamopd`; the internal trainer mode remains `streamopd_colocate`.
Rollout uses an independent `max_num_seqs`, and baseline modes do not consume StreamOPD compatibility fields. The
matrix exposes separate `BASELINE_ROLLOUT_MAX_NUM_SEQS`/`STREAMOPD_ROLLOUT_MAX_NUM_SEQS` and
`BASELINE_AGENT_LOOP_WORKERS`/`STREAMOPD_AGENT_LOOP_WORKERS` overrides so StreamOPD tuning cannot silently change
the baseline denominator.

Generate a local machine-readable summary with:

```bash
python benchmarks/streamopd_kv/summarize_colocate_matrix.py <result-dir>
```

## Scheduler ablation

`run_scheduler_ablation.sh` compares the same streaming Rollout/Teacher implementation with only the scheduling
policy changed. `teacher_then_train` waits for complete Teacher drain; `adaptive` admits reverse units from the ready
queue. Both use the same GPU allocation, model processes, strict policy barrier, and startup preflight plan.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash benchmarks/streamopd_kv/run_scheduler_ablation.sh
```

The scheduler derives whether Teacher, Trainer, and Rollout work can overlap from their GPU resource sets. A Trainer
that spans both Teacher and Rollout subsets uses drain-first scheduling because neither pool can compute concurrently
with it.

## Teacher StreamingInput validation

Validate resumable Teacher sessions independently with:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 .venv-cu128/bin/python \
  benchmarks/streamopd_kv/validate_vllm_streaming_input.py \
  --model /models/store/Qwen/Qwen3-4B --num-sessions 16 \
  --max-model-len 4097 --max-num-batched-tokens 2048 \
  --sequence-length 3200 --prompt-length 100 --response-chunk 1066 \
  --skip-tokenizer-init
```

## Kernel stage benchmark

Run a two-GPU numerical and stage-timing check with:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. .venv-cu128/bin/python \
  benchmarks/streamopd_kv/benchmark_qwen3.py \
  --student /models/store/Qwen/Qwen3-4B \
  --teacher /models/store/Qwen/Qwen3-14B \
  --student-dtype float32 --teacher-dtype float32 \
  --dataset-index 601 --prompt-tokens 512 --response-tokens 64 \
  --token-chunk-size 32 --reverse-chunk-size 256
```

This benchmark excludes model loading and warmup. End-to-end runs are required to include Host KV transport,
scheduler gaps, checkpoint publication, and pipeline fill/drain.

## Post-hoc Teacher/Trainer ablation

Run the streaming-versus-post-hoc comparison with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash benchmarks/streamopd_kv/run_posthoc_ablation.sh
```

Post-hoc submits one complete Teacher request when each trajectory reaches EOS, while Trainer waits for the global
barrier. The comparison isolates Teacher request granularity and early reverse overlap. Rollout KV remains streamed
before EOS in both modes.

## Fixed-slot reverse ablation

Run the legacy/fixed-slot and reverse batch/chunk comparison with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash benchmarks/streamopd_kv/run_fixed_slot_ablation.sh
```

The comparison covers fixed backing allocation, reverse wavefront width, reverse chunk size, Host prefetch wait,
next-group DMA overlap, and Trainer peak memory. Use repeated post-warmup policy steps and fresh process launches
before treating any output as a stable result.
