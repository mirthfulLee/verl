# StreamOPD benchmark

The end-to-end comparison uses this repository's V1 Native OPD as the sync baseline. StreamOPD uses a dedicated
2-GPU rollout pool and a 2-GPU Teacher/Trainer Pool on GPU0-3. Teacher forward and student reverse training are
serialized inside the shared Teacher/Trainer Pool.

## Environment

The measured environment uses Python 3.12.13, PyTorch 2.9.1+cu128, vLLM 0.15.1, transformers 4.57.6, and four
A100-80GB GPUs. Activate the CUDA 12.8 environment for the driver and Ray workers:

```bash
export PATH="$PWD/.venv-cu128/bin:$PATH"
export VIRTUAL_ENV="$PWD/.venv-cu128"
export VERL_USE_UV=0
```

Run the 4096/8192 matrix with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=128 TOTAL_TRAINING_STEPS=1 \
  bash benchmarks/streamopd_kv/run_colocate_matrix.sh
```

The benchmark wrapper calls the current path `streamopd`; the internal trainer mode remains `streamopd_colocate`.
StreamOPD microbatch values 16/32 configure only Teacher/Trainer scheduling. Rollout uses an independent
`max_num_seqs`, and baseline modes do not consume the StreamOPD microbatch setting. The matrix exposes separate
`BASELINE_ROLLOUT_MAX_NUM_SEQS`/`STREAMOPD_ROLLOUT_MAX_NUM_SEQS` and
`BASELINE_AGENT_LOOP_WORKERS`/`STREAMOPD_AGENT_LOOP_WORKERS` overrides so StreamOPD tuning cannot silently change the
baseline denominator.

`verl-sync-opd` uses a 2-GPU student pool plus a separate 2-GPU teacher pool. `verl-colocate-opd` is a sync-baseline
placement ablation configured by `distillation.colocate_teacher_with_student`; it does not enable any StreamOPD code.

## Current 4K result

The following one-step GPU0-3 runs use Qwen3-1.7B/Qwen3-4B, BF16, batch 128, deterministic rollout, the same DAPO
data, and a total trajectory cap of 4096. They were sampled from the same checkout after removing fixed Teacher
cohorts.

| Path | Microbatch | Step | Rollout | Actor update | Throughput | Reverse parallel/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `verl-sync-opd` | - | 184.17 s | 104.78 s | 27.38 s | 547.98 tok/s | - |
| StreamOPD | 16 | 123.37 s | 52.32 s | 43.11 s | 818.51 tok/s | 8 |
| StreamOPD | 32 | 112.63 s | 38.23 s | 45.20 s | 894.74 tok/s | 8 |

The MB32 run is 1.635x faster than the matched sync baseline, a 38.85% wall-time reduction. Both paths report maximum
policy staleness zero. These are one-step engineering measurements; publication-quality numbers should use multiple
post-warmup steps and repeated process launches.

The earlier fixed-cohort scheduler reduced each 4096-token Teacher cohort to eight global trajectories because of
the 32768-token reservation. That limited reverse width to four trajectories per DP rank and increased the step to
152.46 seconds. The fixed-cohort code and its vLLM sleep/wake ablation have been removed.

Machine-readable historical two-pool results remain under `benchmarks/streamopd_kv/results/`. Generate a summary for
new logs with:

```bash
python benchmarks/streamopd_kv/summarize_colocate_matrix.py <result-dir>
```

## Teacher StreamingInput validation

Validate resumable teacher sessions independently with:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 .venv-cu128/bin/python \
  benchmarks/streamopd_kv/validate_vllm_streaming_input.py \
  --model /models/store/Qwen/Qwen3-4B --num-sessions 16 \
  --max-model-len 4097 --max-num-batched-tokens 2048 \
  --sequence-length 3200 --prompt-length 100 --response-chunk 1066 \
  --skip-tokenizer-init
```

## Kernel stage benchmark

For a two-GPU numerical and stage-timing check:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. .venv-cu128/bin/python \
  benchmarks/streamopd_kv/benchmark_qwen3.py \
  --student /models/store/Qwen/Qwen3-4B \
  --teacher /models/store/Qwen/Qwen3-14B \
  --student-dtype float32 --teacher-dtype float32 \
  --dataset-index 601 --prompt-tokens 512 --response-tokens 64 \
  --token-chunk-size 32 --reverse-chunk-size 256
```

This stage benchmark excludes model loading and warmup. End-to-end runs are required to include host KV transport,
scheduler gaps, checkpoint publication, and pipeline fill/drain.
