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
CF performance is acceptable when it is not clearly slower than sync for the
same workload and total GPU count; being slower than async alone does not
justify targeted CF optimization.

## Ablations

Run student 4B/8B × teacher 14B/32B at 4096 max tokens and batch 128.
Use four-GPU shared pools and eight-GPU shared and
dedicated pools, with each workload's selected GPU allocation. Measure one
warmup and three timed steps. Record expected OOM failures without rescue tuning.

- Disable Teacher streaming prefill: wait for each trajectory's EOS, then
  submit its complete prompt and response for prefill; do not wait for the
  entire rollout batch.
- Disable reverse chunked backward with KV: use sync's native `TrainingWorker`
  on a complete policy batch, with no rollout KV export, reverse slots or reverse
  backward. Retain the other KV configuration controls for this single-component
  ablation, including SGO, kernels, token budget, loss and weight transport.

These are 24 ablation cases, paired with 12 compatible complete-KV references,
including eight-GPU dedicated 4B/14B, 4B/32B and 8B/14B references outside the
main study. Preserve original records and configuration provenance separately.

## Settings

Use the following study groups instead of the full Cartesian product. Model
sizes below refer to Qwen3; token limits include prompt and response.

| Group | Student | Teacher | Max tokens | Batch | GPUs | Methods | Cases |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| Main results | 4B / 8B | 14B / 32B | 4096 / 8192 | 128 | 4 / 8 | Shared KV, shared sync, CF, async | 64 |
| Dedicated-pool control | 8B | 32B | 4096 / 8192 | 128 | 8 | Dedicated KV, dedicated sync | 4 |
| Batch extension | 8B | 32B | 4096 / 8192 | 256 | 4 / 8 | Shared KV, shared sync | 8 |

The study has **76 cases: 38 baselines and 38 KV/CF cases** before reuse or
failures. Complete baselines first (18 four-GPU and 20 eight-GPU cases), then
run KV/CF. Start with four-GPU baselines and prioritize 4096 within each GPU
count. Two four-GPU cases may run concurrently when all eight GPUs are free.
Within the method phase, complete CF before KV; run four-GPU cases before
eight-GPU cases and prioritize 4096 within each phase. Preserve the matched
baseline's physical GPU placement; serialize shared pools when concurrent host
memory use would approach the user limit.
Qwen3-30B-A3B and other combinations are outside the current study, not removed
permanently; retain their configurations and existing results as supplementary
records. Do not launch four-GPU dedicated sync or batch-256 async in this study.

- The prompt cap is 1024.
- DAPO-Math, seed 1, greedy generation, forward KL top-k=32.
- KV/CF use BF16, FlashAttention 2, Liger and `no_sync` accumulation;
  baselines inherit native kernel and accumulation defaults. All methods make
  one optimizer update per policy batch. Enable model gradient checkpointing;
  KV reverse training runs in eval mode and does not activate HF checkpoints.
- KV and CF prefer SHARD_GRAD_OP; explicitly set
  `actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False` for dedicated
  KV and CF. Baselines inherit native FSDP defaults (currently FULL_SHARD).
  KV reverse shapes and CF microbatches use their automatic planners.
  KV automatic reverse chunks now have a 2048-token default cap; earlier runs
  allowed chunks up to the trajectory length. Record the effective cap and
  actual chunk size when comparing runs across this change.
- All methods use the same **128 MiB Host weight-transfer bucket**.
- Match inference TP within each pair. Baselines inherit native inference
  batching and memory settings (Rollout/Teacher `gpu_memory_utilization=0.5/0.5`
  in these runs); no CF/KV microbatch, chunk, kernel or concurrency tuning is
  injected. Set inference `max_model_len` to the total token cap plus 1.
  If needed to resolve runtime errors, prefer native configuration/environment
  options and record every adjustment. The 8B native sync/async cases use
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

| GPUs | Student | Max tokens | Dedicated Trainer/Rollout/Teacher | Shared Trainer; Rollout/Teacher |
| --- | --- | --- | --- | --- |
| 4 | 4B / 8B | 4096 / 8192 | 2/1/1 | Trainer 4; 2/2 |
| 8 | 4B | 4096 | 4/2/2 | Trainer 8; 4/4 |
| 8 | 4B | 8192 | 2/4/2 | Trainer 8; 4/4 |
| 8 | 8B | 4096 / 8192 | 4/2/2 | Trainer 8; 4/4 |

Teacher TP is 2 for four-GPU shared pools and 1 for four-GPU dedicated pools.
On eight GPUs, use Teacher TP1 for 14B and TP2 for 32B. These are the current
study references, not fit guarantees or claims of a global optimum; retain
selection status and evidence with the local case records. Select allocations by GPU
count, total token cap, Student/Teacher models and shared/dedicated topology.
Use KV performance to select each topology: shared sync inherits shared KV's
allocation; CF, dedicated sync and async inherit dedicated KV's allocation.
Match Teacher TP as well as pool sizes. Reuse the selection across batch 128/256.
For screening, use batch 128 with one warmup and one measured step. Inspect
Rollout/Teacher completion times and Trainer intervals to choose promising
alternatives instead of exhaustively searching; on eight GPUs, vary pool sizes
in units of two. Record OOM without changing training settings to rescue a split.
When step times are close, prefer balanced allocations, considering phase
completion times and memory headroom.
For practical four-GPU checks, reuse reasonable existing results and skip known
memory failures; only try another split when phase timings justify it. Check
physical GPU topology when placing TP ranks, keeping them within an NVLink group
where available.

The dedicated-pool control above is retained regardless of its speed relative
to shared KV. Select Teacher TP to fit the model and available pool; record
infeasible cases without silently substituting another allocation.

## Running

Use a compatible Python/CUDA/vLLM environment managed with `uv`, with the
repository on `PYTHONPATH`. Set `MODEL_ROOT`, `DATASET` and free GPU IDs locally.
Use a fresh result directory for each configuration. Each GPU group runs one
case at a time; concurrent groups need separate Ray sessions and service ports.

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
Reuse a completed result only when its workload, allocation, runtime settings
and measured-step count match; a historical one-step screen is not a three-step
baseline result.

## Results

Store **all results locally**, including reports, tables, figures and logs, in
`benchmarks/streamopd_kv/results/` or `benchmarks/streamopd_cf/results/`.
Both roots are git-ignored. Commit experiment instructions and reusable scripts,
not results or optimization diaries.

Retain each run's request, source revision, dependencies, completion status,
actual token count, step time, memory and staleness. KV/CF timelines record stage
start/end times; service timers include waits and differ from CUDA intervals.
Do not sum overlapping stages or report speedups from failed/incomplete runs.
For native baselines, retain per-step service timers and sampled per-GPU memory
peaks. External Ray request intervals can describe asynchronous overlap without
changing scheduling; distinguish request time, batch sampling wait and CUDA
execution. Report sampling resolution and any approximate step boundaries.
Completed-request unions can omit in-flight requests or missing final states;
record trace coverage and keep native service timers as the primary durations.
When running two four-GPU cases concurrently, record physical GPU placement and
overlapping runs; user-level host-memory samples include both cases. Use the
same physical GPU group for each method and its matched baseline.

Implementation details: [KV](../verl/experimental/streamopd_kv/README.md) and
[CF](../verl/experimental/streamopd_cf/README.md).
