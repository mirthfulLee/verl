# StreamOPD Experiments

## Methods And Baselines

Run comparisons on **4 or 8 GPUs** with the same Student, Teacher, data and
total GPU count within each pair.

| Method | Execution | Matched baseline |
| --- | --- | --- |
| StreamOPD-KV, shared | Reuses Rollout KV; Trainer runs on the union of Rollout/Teacher GPUs after inference | `verl-sync-opd-union` (`union_sync`), same GPU allocation |
| StreamOPD-KV, dedicated | Reuses Rollout KV with a dedicated Trainer pool | `verl-sync-opd-separate` (`separate_sync`), same GPU allocation |
| StreamOPD-CF | Streams ascending Trainer forwards on dedicated GPUs; final loss/backward waits for complete supervision | `verl-async-opd` (`separate_async`), same GPU allocation, with staleness |

All baselines launch through `benchmarks/streamopd_cf/run_native_opd.sh` and
retain verl's native full-trajectory Teacher scoring, old-logprob/advantage
preparation, Trainer, inference defaults and weight-update manager. Sync uses
the native synchronous replay buffer with zero staleness; async keeps native
prefetch and hybrid replicas. Sync's thin placement adapters allocate independent
Rollout/Teacher pools or share their union with Trainer; shared sync sleeps
inference and enables native parameter/optimizer offload before switching phases.

Async `parameter_sync_step=1` matches one optimizer update per benchmark batch
(the native default of 4 requires a different minibatch size). Dynamic training
batching is enabled as in the native OPD example. All baselines retain the shared
Host weight-transfer backend and 128 MiB bucket as explicit transport controls.
CF can additionally use dedicated sync as a strict on-policy control.

## Settings

- Current full matrix: Student Qwen3-8B, Teacher Qwen3-32B; maximum total tokens
  4096/8192 × global batch 128/256. The prompt cap is 1024.
  Earlier model screening also covers Student Qwen3-4B and Teachers Qwen3-14B/Qwen3-30B-A3B.
- DAPO-Math, seed 1, greedy generation, forward KL top-k=32.
- KV/CF use BF16, FlashAttention 2, Liger and `no_sync` accumulation;
  baselines inherit native kernel and accumulation defaults. All methods make
  one optimizer update per policy batch. Enable model gradient checkpointing;
  KV reverse training runs in eval mode and does not activate HF checkpoints.
- KV and CF prefer SHARD_GRAD_OP; explicitly set
  `actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False` for dedicated
  KV and CF. Baselines inherit native FSDP defaults (currently FULL_SHARD).
  KV reverse shapes and CF microbatches use their automatic planners.
- All methods use the same **128 MiB Host weight-transfer bucket**.
- Match inference TP within each pair. Baselines inherit native inference
  batching and memory settings; no CF/KV microbatch, chunk, kernel or concurrency
  tuning is injected. Set inference `max_model_len` to the total token cap plus 1.
  If needed to resolve runtime errors, prefer native configuration/environment
  options and record every adjustment. The 8B native async run uses
  `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192` after the default 16384
  budget caused OOM; this changes microbatch packing, not the global batch.
- KV standalone inference uses GPU memory utilization 0.85. CF uses Rollout 0.85,
  Teacher 0.8 and `distillation.batching.memory_fraction=0.95`.
  KV uses the manual profile with automatic reverse planning.
- Preserve failed attempts and earlier runtime configurations separately.
  Historical sync/async measurements using benchmark-specific workers or kernel
  overrides do not measure the native baseline. If host memory limits prevent
  KV from completing, defer it until the limit is adjusted instead of changing
  backing storage for the comparison.

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
  "student": "Qwen3-4B", "teacher_model": "Qwen3-14B", "tokens": 4096,
  "overrides": ["actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False"]
}
```

Keep case JSON files under a local result directory. For eight GPUs, adjust
the case allocation and `DEVICES` accordingly. `RESULT_DIR` must be under one
of the ignored result roots below.

```bash
python -m benchmarks.streamopd_kv.pilot_8gpu single \
  --case-json "$CASE_JSON" --root "$RESULT_DIR" --models "$MODEL_ROOT" \
  --dataset "$DATASET" --devices "$DEVICES" --batch 128 --warmup 1 --measure 3
python -m benchmarks.streamopd_kv.export_pilot_timelines "$RESULT_DIR"
```

The runner defaults to one measured step for preliminary checks. Full experiments
use `--measure 3`: exclude the warmup, then report all three step times, their mean
and sample standard deviation. Use separate result roots for each batch size.

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
