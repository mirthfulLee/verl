# StreamOPD Experiments

## Methods And Baselines

Run comparisons on **4 or 8 GPUs** with the same Student, Teacher, data and
total GPU count within each pair.

| Method | Execution | Matched baseline |
| --- | --- | --- |
| StreamOPD-KV, shared | Reuses Rollout KV; Trainer runs on the union of Rollout/Teacher GPUs after inference | `verl-sync-opd-union` (`union_sync`), same GPU allocation |
| StreamOPD-KV, dedicated | Reuses Rollout KV with a dedicated Trainer pool | `verl-sync-opd-separate` (`separate_sync`), same GPU allocation |
| StreamOPD-CF | Streams ascending Trainer forwards on dedicated GPUs; final loss/backward waits for complete supervision | `verl-async-opd` (`separate_async`), same GPU allocation, with staleness |

Both sync baselines score complete trajectories at EOS and train after the
whole policy batch is supervised, with blocking weight synchronization and
zero staleness. Shared sync sleeps both inference pools before training and
offloads Trainer state before inference resumes. Async retains native prefetch,
hybrid replicas and old-logprob/advantage preparation; record observed staleness.
CF can additionally use dedicated sync as a strict on-policy control.

## Settings

- Students: Qwen3-4B and Qwen3-8B. Teachers: Qwen3-14B, Qwen3-32B and Qwen3-30B-A3B.
- Maximum total tokens: 4096, including a prompt cap of 1024.
- DAPO-Math, seed 1, greedy generation, global batch 128, forward KL top-k=32.
- BF16, FlashAttention 2, Liger and `no_sync` accumulation; one optimizer update
  per policy batch. Enable model gradient checkpointing, noting that KV reverse
  training runs in eval mode and does not activate Hugging Face checkpoints.
- Shared KV uses SHARD_GRAD_OP. Dedicated KV, CF and all native baselines retain
  FULL_SHARD. KV reverse shapes and CF microbatches use their automatic planners.
  Native 8B sync/async training uses an 8192-token budget per Trainer GPU.
- All methods use the same **128 MiB Host weight-transfer bucket**.
- Match inference TP and batching settings within each pair. Record explicit
  overrides and failed configurations; do not shorten sequences to hide an OOM.

| GPU count | Dedicated Trainer/Rollout/Teacher starting split | Shared Trainer; Rollout/Teacher starting split |
| --- | --- | --- |
| 4 | 2/1/1 | Trainer 4; 2/2 |
| 8 | 4/2/2 | Trainer 8; 4/4 |

These are starting allocations, not fit guarantees. Screen allocations first.
Prefer shared KV unless dedicated KV reduces step time and increases consumed
response-token throughput by at least 10% each; otherwise omit dedicated KV and
its sync baseline. Select Teacher TP to fit the model and available pool.

## Running

Use a compatible Python/CUDA/vLLM environment managed with `uv`, with the
repository on `PYTHONPATH`. Set `MODEL_ROOT`, `DATASET` and free GPU IDs locally.
Run cases sequentially with a fresh result directory for each configuration.

The existing single-case runner supports both GPU counts despite its filename.
A four-GPU case JSON can contain:

```json
{
  "method": "streamopd-cf", "trainer": 2, "rollout": 1, "teacher": 1, "tp": 1,
  "student": "Qwen3-4B", "teacher_model": "Qwen3-14B", "tokens": 4096
}
```

Keep case JSON files under a local result directory. For eight GPUs, adjust
the case allocation and `DEVICES` accordingly. `RESULT_DIR` must be under one
of the ignored result roots below.

```bash
python -m benchmarks.streamopd_kv.pilot_8gpu single \
  --case-json "$CASE_JSON" --root "$RESULT_DIR" --models "$MODEL_ROOT" \
  --dataset "$DATASET" --devices "$DEVICES" --batch 128 --warmup 1
python -m benchmarks.streamopd_kv.export_pilot_timelines "$RESULT_DIR"
```

The preliminary runner measures one step after warmup. Formal experiments
should measure multiple steps, exclude warmup, and report the mean and spread.

## Results

Store **all results locally**, including reports, tables, figures and logs, in
`benchmarks/streamopd_kv/results/` or `benchmarks/streamopd_cf/results/`.
Both roots are git-ignored. Commit experiment instructions and reusable scripts,
not results or optimization diaries.

Retain each run's request, source revision, dependencies, completion status,
actual token count, step time, memory and staleness. KV/CF timelines record stage
start/end times; service timers include waits and differ from CUDA intervals.
Do not sum overlapping stages or report speedups from failed/incomplete runs.

Implementation details: [KV](../verl/experimental/streamopd_kv/README.md) and
[CF](../verl/experimental/streamopd_cf/README.md).
