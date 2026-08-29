# StreamOPD-KV benchmark

The matched end-to-end denominator is this repository's V1 Native OPD with `trainer_mode=sync`, referred to as
`verl-sync-opd`. No external OPD implementation is used.

## Environment

This workspace was configured with `uv` in `.venv-cu128`. The measured environment uses Python 3.12.13,
PyTorch 2.9.0+cu128, vLLM 0.11.2, and transformers 4.57.1; `uv pip check` passes for all 225 installed packages.
The CUDA 12.8 environment is used because this host's 560.35.03 driver cannot load the repository's CUDA 13 lock.

Activate it for Ray workers as well as the driver:

```bash
export PATH="$PWD/.venv-cu128/bin:$PATH"
export VIRTUAL_ENV="$PWD/.venv-cu128"
export VERL_USE_UV=0
```

Run the matched 4K-total-trajectory experiment (prompt is capped at 1024, so response is capped at 3072):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash benchmarks/streamopd_kv/run_qwen3_1p7b4b_b128_total4k.sh
```

The wrapper defaults to one measured step, batch 128, Qwen3-1.7B student, Qwen3-4B teacher, `MAX_PROMPT_LENGTH=1024`,
and `TOTAL_TRAJECTORY_LENGTH=4096`. Set `TOTAL_TRAJECTORY_LENGTH=8192` for the 8K variant; the response limit is
derived as `total - prompt`. `ACTOR_MAX_TOKENS_PER_GPU` is only the dynamic training microbatch budget and does not
change the per-trajectory cap.

For a quick two-GPU stage benchmark using DAPO data and independent student/teacher CUDA processes:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. .venv-cu128/bin/python \
  benchmarks/streamopd_kv/benchmark_qwen3.py \
  --student /models/store/Qwen/Qwen3-4B \
  --teacher /models/store/Qwen/Qwen3-14B \
  --student-dtype float32 --teacher-dtype float32 \
  --dataset-index 601 --prompt-tokens 512 --response-tokens 64 \
  --token-chunk-size 32 --reverse-chunk-size 256
```

The stage benchmark excludes model loading and warmup. It includes rollout, teacher scoring, and student backward,
but the rollout KV remains in the student process, so serialization handoff cost is zero. The end-to-end comparison
is required to measure the current pinned-CPU/safetensors transport.

## Measured results

### End-to-end V1 sync comparison (total trajectory length 4096)

Four A100-80GB GPUs are split into a two-GPU Student Hybrid Pool and a two-GPU Teacher Pool. Student execution is
FP32 for the strict numerical audit; the frozen teacher is BF16. The reported values are from one complete training
step after model and service initialization.

| Path | Step | Response mean / max | Actor peak allocated | Actor peak reserved |
| --- | ---: | ---: | ---: | ---: |
| `verl-sync-opd` | 448.61 s | 3063 / 3072 | 55.16 GiB | 60.76 GiB |
| StreamOPD-KV (1024/2048 chunks) | 393.64 s | 3058 / 3072 | 64.66 GiB | 71.03 GiB |

StreamOPD-KV is **1.140x faster** and reduces step time by **12.26%** (54.97 seconds) on four A100-80GB GPUs.
Both runs use this checkout's `verl-sync-opd` (`trainer.use_v1=True`, `trainer.v1.trainer_mode=sync`) as the strict
no-staleness denominator, with the same policy version, models, data, precision, batch, and total trajectory cap.
The Stream run overlaps teacher chunk scoring with rollout, reuses the sealed rollout KV, and performs reverse OPD;
the sync run post-processes teacher scores and recomputes the student sequence. Exact stage metrics are in
`results/qwen3_1p7b_4b_b128_total4k.json`.

The selected long-trajectory granularity is a 1024-token teacher chunk and 2048-token reverse chunk. It cuts reverse
backward calls from 128 to 64 while remaining below the 80-GiB card limit. The correctness-first safetensors handoff
is linear in trajectory length, so use a large data filesystem for 4K/8K runs.

### Correctness

A multi-chunk four-GPU validation run compares streamed teacher artifacts against post-hoc full-prefix scoring and
also recomputes the conventional student full forward on the same artifacts. Teacher validation passed at `1e-4`;
reverse loss was `0.4903037250`, conventional loss was `0.4900929034`, for absolute error `2.108e-4` (0.043%). A
separate deterministic FP32 run matched loss within `1.81e-5` and gradient norm within 0.01%. Unit tests additionally
compare every tiny-Qwen3 parameter gradient for ragged batched reverse traversal.

### Stage benchmark and short smoke results

All rows use `open-r1/DAPO-Math-17k-Processed`, top-k 32, one A100 for the student and one A100 for the teacher.

| Student / teacher | Precision | Trace / reverse chunk | verl-sync-opd stage | StreamOPD-KV | Speedup | Loss error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 4B / 14B | FP32 / FP32 | 380 / 256 | 3.470 s | 2.999 s | 1.157x | 1.10e-4 |
| Qwen3 4B / 14B | BF16 / FP32 | 380 / 512 | 2.838 s | 2.192 s | 1.295x | 0.00 |
| Qwen3 1.7B / 4B | BF16 / FP32 | 206 / 512 | 1.729 s | 1.648 s | 1.049x | 8.58e-6 |

These short rows are smoke tests for the transport and numerical path, not the primary long-trajectory benchmark.
The 1.7B/4B batch-one case is rollout dominated; the 4B/14B rows exercise teacher overlap and dK/dV propagation.
See the JSON files under `results/` for stage breakdowns and peak memory.
