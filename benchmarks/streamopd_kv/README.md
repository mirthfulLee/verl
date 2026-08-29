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
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 \
bash benchmarks/streamopd_kv/run_qwen3_1p7b4b_b128_total4k.sh
```

The wrapper defaults to two measured steps, batch 128, Qwen3-1.7B student, Qwen3-4B teacher,
`MAX_PROMPT_LENGTH=1024`, and `TOTAL_TRAJECTORY_LENGTH=4096`. It uses four student execution ranks and two teacher
replicas on the same four physical GPUs. The teachers are active during rollout and sleep before all four GPUs enter
reverse training. Step 2 is the stable measurement because V1 creates some workers lazily during step 1.

Set `TOTAL_TRAJECTORY_LENGTH=8192` for the 8K variant; the response limit is derived as `total - prompt`.
`ACTOR_MAX_TOKENS_PER_GPU` is only the dynamic training microbatch budget and does not change the per-trajectory cap.
`NCCL_P2P_DISABLE=1` is required by the GPU 0-3 topology on the measured host, but is not a general StreamOPD-KV
requirement.

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

The primary comparison uses the same phase-shared placement on four A100-80GB GPUs: four student execution ranks,
two teacher replicas on GPU ranks 0-1 during rollout, then four student trainer ranks after the teachers sleep.
Student execution is FP32 for the strict numerical audit; the frozen teacher is BF16. Values below are stable step 2.

| Path | Step | Generation / teacher | Student scoring + update | Actor peak allocated |
| --- | ---: | ---: | ---: | ---: |
| `verl-sync-opd` | 312.77 s | 129.82 s | 46.03 + 130.31 s | 51.54 GiB |
| StreamOPD-KV (1536/2048 chunks) | 291.76 s | 141.98 s | 143.15 s | 64.36 GiB |

StreamOPD-KV is **1.072x faster** and reduces both step time and four-GPU-seconds by **6.72%** (21.01 seconds).
Both paths use this checkout's V1 Native OPD with `trainer_mode=sync`, the same placement, models, data, precision,
batch, and 4096-token total trajectory cap. Both report maximum trajectory staleness zero. StreamOPD-KV removes the
46.03-second conventional student scoring pass; reverse training costs 12.84 seconds more than the conventional
actor update, leaving the measured net gain.

The selected long-trajectory granularity is a 1536-token teacher chunk and a 2048-token reverse chunk. A full
3072-token response therefore submits teacher work once before EOS and once at completion. Terminal-only scoring is
an 8.45% throughput upper bound on this workload, but is kept as an ablation because it removes pre-EOS streaming.
The correctness-first safetensors handoff is linear in trajectory length, so use a large data filesystem for 4K/8K
runs. Exact metrics and ablations are in
`results/qwen3_1p7b_4b_b128_total4k_phase_shared.json`.

### Scheduling and topology ablations

| StreamOPD-KV placement / schedule | Step | Generation / teacher | Reverse actor | Result |
| --- | ---: | ---: | ---: | --- |
| Dedicated 2 student + 2 teacher, rollout concurrency 32 | 381.57 s | 163.00 s | 189.91 s | Better batching, but only two trainer GPUs |
| Dedicated 3 student + 1 teacher | 367.00 s | 155.02 s | 184.39 s | Teacher GPU is idle during reverse training |
| Same-card rollout/training overlap, cohorts of 32 | 395.55 s | 95.31 s wait | 267.48 s | Strict, but kernel and memory contention dominate |
| Same-card rollout/training overlap, cohorts of 64 | 392.13 s | 105.12 s wait | 259.10 s | Strict, still slower than phase switching |
| Phase-shared 4 student + 2 teacher, terminal-only | 286.36 s | 137.91 s | 141.83 s | Throughput upper bound |
| Phase-shared 4 student + 2 teacher, two chunks | 291.76 s | 141.98 s | 143.15 s | Selected full streaming |

The overlap mode accumulates unnormalized gradients across rollout cohorts and performs exactly one normalization,
clip, optimizer step, and weight publication at the final version barrier. It is therefore strict on-policy, but is
disabled by default because concurrent vLLM and backward kernels were slower here; a 2048-token overlap run also
exceeded 80 GiB. The 3:1 direct split is valid with DP=3 padding, but phase sharing is 1.134x faster in matched
one-step runs because all four GPUs train after the teacher sleeps.

A 4096-token reverse chunk with two trajectories per microbatch OOMed near 78.29 GiB. Reducing it to one trajectory
completed at 315.90 seconds with a 141.26-second actor update, which did not improve over the selected 2048-token
chunk. The 2048-token setting remains the performance and memory default.

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
