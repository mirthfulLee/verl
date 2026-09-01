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

## Post-hoc Teacher/Trainer ablation

Run the matched two-step comparison with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash benchmarks/streamopd_kv/run_posthoc_ablation.sh
```

Both modes use a 2-GPU Rollout Pool and a 2-GPU Teacher/Trainer Pool, B128, total trajectory length 4096, and MB32.
The table reports stable step 2, after lazy agent workers and metrics RPCs are warm. Post-hoc submits one complete
Teacher request immediately when each individual trajectory reaches EOS; only Trainer waits for the global barrier.
This archived Teacher-schedule comparison predates fixed reverse slots and used `reverse_fixed_slots=false` in both
columns, so the later fixed-slot ablation remains orthogonal.

| Metric | Streaming StreamOPD | Post-hoc ablation |
| --- | ---: | ---: |
| Step | 98.98 s | 109.71 s |
| Throughput | 1023.43 tok/s | 922.53 tok/s |
| First Teacher start | 8.51 s | 28.16 s |
| All rollout EOS | 59.22 s | 56.67 s |
| All Teacher complete | 84.62 s | 63.66 s |
| Teacher drain after all EOS | 25.41 s | 7.00 s |
| Teacher busy before all EOS | 12.96 s | 9.58 s |
| Teacher busy total | 19.98 s | 16.57 s |
| Reverse busy | 38.43 s | 38.25 s |
| Teacher/Trainer Pool idle | 33.06 s | 47.42 s |
| Teacher/Trainer Pool utilization | 63.86% | 53.62% |
| Teacher forwards | 180 | 128 |
| Rollout peak allocated/reserved | not sampled | 51.29 / 51.50 GiB |
| Trainer peak allocated/reserved | 53.80 / 61.50 GiB | 53.80 / 61.06 GiB |
| Teacher peak allocated/reserved | 12.42 / 15.25 GiB | 12.42 / 15.25 GiB |

Post-hoc is 10.83% slower, or equivalently streaming is 1.108x faster. Full requests save 3.41 seconds of Teacher
busy time and reduce request count by 28.9%, but the global reverse barrier loses more overlap: the 38.25-second
reverse phase begins only after Teacher completion. The allocator peaks are effectively unchanged, so this ablation
does not create useful Trainer headroom under the current persistent-Teacher layout. A conservative sum of independent
reserved peaks is 76.75 GiB for streaming and 76.32 GiB for post-hoc; because Teacher and Trainer kernels are
serialized, this sum is an upper bound rather than a simultaneous allocation measurement.

The matched timing run predates the Rollout Pool allocator RPC, so its streaming column does not claim a measured
rollout peak. A separate two-step post-hoc memory run measured 51.29 GiB allocated and 51.50 GiB reserved per rollout
worker on stable step 2, close to the configured 65% vLLM target. The reset and collection RPCs took 28 ms and 14 ms,
respectively; stable step time was 110.81 seconds versus 109.71 seconds in the matched post-hoc run, so allocator
sampling adds no material hot-path cost. This separate run is used only to complete the memory accounting, not to
replace the matched timing comparison.

Rollout KV remains streamed before EOS in both modes (about 28 chunks and 43K KV tokens per rank in these runs), so
the result isolates Teacher request granularity and the reverse barrier rather than changing KV transport. Exact
step-level metrics are stored in `results/posthoc_ablation_b128_4k_2step/summary.json`; the Rollout Pool memory
re-sample is stored in `results/posthoc_ablation_b128_4k_2step_memory/summary.json`.

## Fixed-slot reverse ablation

Run the legacy/fixed/B16 comparison with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash benchmarks/streamopd_kv/run_fixed_slot_ablation.sh
```

The completed matched run used physical GPU0,1,3 because an unrelated persistent serving process occupied GPU2: a
one-GPU Rollout Pool and a two-GPU Teacher/Trainer Pool, B128, total length 4096, MB32, stable step 2. Post-hoc makes
reverse training serial after rollout/Teacher completion, so reverse time, H2D overlap, and Teacher/Trainer memory are
directly comparable even though total step includes the slower one-GPU rollout.

| Metric | Legacy B8/C960 | Fixed B8/C1024 | Fixed B16/C256 | Fixed B16/C512 |
| --- | ---: | ---: | ---: | ---: |
| Step | 158.36 s | 154.74 s | 160.24 s | 174.54 s |
| Rollout/Teacher window | 114.81 s | 115.67 s | 114.84 s | 115.11 s |
| Reverse/update | 36.74 s | 32.92 s | 38.55 s | 52.60 s |
| Scheduler training busy | 36.72 s | 32.91 s | 38.53 s | 52.57 s |
| Planned trajectories/rank | 8 | 8 | 16 | 16 |
| Backward calls/rank | 7.0 | 7.0 | 13.8 | 7.0 |
| Host prefetch exposed wait | 1.09 s | 0.51 s | 0.81 s | 0.95 s |
| Next-group DMA / exposed tail | - | 0.20 / 0.09 s | - | - |
| Trainer peak allocated | 53.81 GiB | 50.06 GiB | 42.47 GiB | 57.50 GiB |
| Trainer peak reserved | 61.21 GiB | 59.27 GiB | 48.16 GiB | 61.59 GiB |

The selected B8 plan reduces reverse time by 10.39% and end-to-end step by 2.29%. Allocated/reserved peaks fall by
3.75/1.94 GiB. The next-group DMA starts while the current suffix backward is active; 0.11 seconds of its 0.20-second
copy is hidden, and only 0.09 seconds remains at group activation. Token-major host loading and raw CUDA submission
reduce copy enqueue from the initial fixed-slot prototype's 1.23 seconds to 0.07 seconds.

B16 is memory-feasible and reduces peak memory further, but it forces chunk 256 under the same preflight reserve.
The doubled wavefront depth dominates the larger batch: reverse is 4.93% slower than legacy and 17.10% slower than
B8. Forcing B16/C512 with no preflight reserve is slower still: its larger activation and LM-head working set makes
reverse 59.79% slower than B8/C1024 and raises reserved memory to 61.59 GiB. The default 4096 plan therefore retains
B8/C1024. Stable metrics are in `results/posthoc_fixed_slot_dma_w4_b128_4k_3gpu/summary.json`; the forced C512 probe
is in `results/posthoc_fixed_slot_dma_w4_b16_chunk512_b128_4k_3gpu/summary.json`.
