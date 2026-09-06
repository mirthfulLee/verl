# StreamOPD-CF benchmarks

Compare the same OPD objective under independent or colocated GPU placement.
The four-GPU defaults are:

| Case | Trainer | Rollout | Teacher | Total physical GPUs |
| --- | ---: | ---: | ---: | ---: |
| `verl-sync-opd` | 2 shared with Rollout | same 2 | 2 | 4 |
| StreamOPD-CF (`streamopd-cf`) | 2 independent | 1 | 1 | 4 |
| `verl-sync-opd-separate` | 2 independent | 1 | 1 | 4 |
| StreamOPD-KV (`streamopd-kv`, union placement) | borrows all 4 after inference | 2 | 2 | 4 |

Use `CASE=streamopd-cf` (default) or `CASE=streamopd-kv`.

All cases default to Qwen3-1.7B / Qwen3-4B, DAPO-Math, batch 128, prompt limit
1024, response limit 3072, greedy generation and forward KL with Teacher top-k
32. Gradient checkpointing, Liger and inference CUDA graphs are enabled.
`verl-sync-opd` uses verl's native synchronous training path.
`verl-sync-opd-separate` is the matched-placement, strict on-policy control:
Teacher receives each complete trajectory at EOS; Trainer runs native
full-trajectory forward/backward after the full policy batch is supervised.
It uses the synchronous replay buffer and blocking weight synchronization,
rejects stale or mixed-version trajectories, and skips PPO-only preparation
just as StreamOPD-CF does. Full-trajectory Teacher requests can overlap other
unfinished rollouts; there is no token streaming within a trajectory.
`verl-async-opd` selects native `separate_async` as a separate throughput control
that permits policy staleness; it is not an equivalent strict on-policy method.

Set `DATASET` and optionally local model paths. Run cases sequentially on the
same visible devices and Python environment. Keep the policy batch fixed: GPU
count sets the per-rank batch capacity, while microbatch size controls memory
and gradient accumulation, not the number of samples in an optimizer update.

For the primary controlled comparison, compare `verl-sync-opd-separate` with
`StreamOPD-CF` and set the same `FIXED_MICRO_BATCH_SIZE` in both
cases. This disables native dynamic token packing; for the original baseline,
it also fixes the old-logprob microbatch. The separate baseline equalizes
placement and removes PPO-only preparation;
this remains a joint comparison of Teacher streaming and the training strategy.
Also report `CASE=verl-sync-opd` as the original-layout reference.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATASET=/path/to/train.parquet
export TOTAL_TRAINING_STEPS=3
FIXED_MICRO_BATCH_SIZE=4 CASE=verl-sync-opd-separate RESULT_DIR=/tmp/opd-baseline bash benchmarks/streamopd_cf/run_case.sh
FIXED_MICRO_BATCH_SIZE=4 CASE=streamopd-cf RESULT_DIR=/tmp/opd-streamopd-cf bash benchmarks/streamopd_cf/run_case.sh
uv run --active --no-sync python -m benchmarks.streamopd_cf.summarize \
  /tmp/opd-baseline/separate_sync_total4096_bs128.log \
  /tmp/opd-streamopd-cf/streamopd_cf_total4096_bs128.log \
  --baseline-label verl-sync-opd-separate \
  --output benchmarks/streamopd_cf/results/comparison
```

Separately measure practical settings: omit `FIXED_MICRO_BATCH_SIZE` and use
`TRAIN_MICRO_BATCH_SIZE=0` and `TRAIN_MAX_TOKENS_PER_GPU=0` (defaults) for shared
memory-based planning in both StreamOPD-CF and the matched-placement baseline.
A positive training token budget limits native dynamic packing in the baseline
and the maximum-length streaming window in CF. Rollout and Teacher have
independent sequence/token limits:

```bash
TRAIN_MAX_TOKENS_PER_GPU=0 ROLLOUT_MAX_NUM_SEQS=128 TEACHER_MAX_NUM_SEQS=32 \
  CASE=verl-sync-opd-separate bash benchmarks/streamopd_cf/run_case.sh
```

`ROLLOUT_MAX_BATCHED_TOKENS`, `TEACHER_MAX_BATCHED_TOKENS`, and each role's
`*_GPU_MEMORY_UTILIZATION` are also independent. Zero inference sequence/token
limits derive defaults; the vLLM schedulers profile and manage KV memory in their
own pools. Hold the inference settings fixed when isolating training differences.
The original-layout `verl-sync-opd` keeps its existing harness configuration.
Record all settings and actual memory/step times. Auto planning is a conservative
heuristic, not a search for the fastest batch size. Report
controlled and individually tuned results separately; a speedup does not by
itself isolate the effect of the new kernels.

The summarizer requires complete runs, excludes warmup, averages all remaining
steps and reports sample standard deviation and actual response lengths.
Inspect `streamopd_cf/stream/forward_chunks_before_eos` and
`forward_tokens_before_eos` for confirmed GPU forward completion while the
corresponding rollouts are still active. Compare `first_forward_seconds` with
`all_rollouts_terminal_seconds`; cold initialization can eliminate overlap in
short warmup steps. `actor/streamopd_cf/{forward_chunks,backward_calls}` records
ascending chunk forwards and one backward per window. The summarizer checks
that all trajectories completed rollout, Teacher and training, with one policy
update and zero staleness. Scheduler training launch does not itself prove GPU
compute overlap, and Teacher supervision need not be complete at that point.
`timing_s/train_stream` includes exposed input waits; use
`actor/streamopd_cf/{forward,loss,backward}_gpu_seconds` for compute and
`actor/streamopd_cf/stream_wait_seconds` for stream RPC/input waits. These
overlapping stage times must not be added to rollout time as serial work.

The initial validated dependency environment uses PyTorch 2.9.1, vLLM 0.15.1,
Transformers 4.57.6, FlashAttention 2.8.3 and Liger 0.8.2. Use `uv` to manage/select
that environment; no environment directory or machine-specific model path is
part of the source contribution. Results are local ignored development artifacts.

## Matched Sync and Async Coverage

`run_async_matrix.sh` changes one setting at a time around Qwen3-1.7B/4B,
4096 maximum total tokens (1024 prompt + 3072 response), and policy batch 128:

| Setting | Student / Teacher | Max total tokens | Policy batch |
| --- | --- | ---: | ---: |
| default | Qwen3-1.7B / Qwen3-4B | 4096 | 128 |
| model | Qwen3-4B / Qwen3-14B | 4096 | 128 |
| tokens | Qwen3-1.7B / Qwen3-4B | 8192 | 128 |
| batch | Qwen3-1.7B / Qwen3-4B | 4096 | 256 |

All three methods use four physical GPUs: Trainer 2, standalone Rollout 1,
Teacher 1. The matrix runs CF, `verl-sync-opd-separate` and `verl-async-opd` for
each setting. CF and matched sync independently plan training budgets from
their memory requirements; inference budgets remain independent of training.
Native async additionally initializes hybrid rollout replicas on the Trainer
GPUs, then sleeps them before steady training (`hybrid_rollout.enable_switch=false`).
Its standalone rollout uses 85% GPU memory; hybrid replicas use 35%. The
`standalone_gpu_memory_utilization` setting is now applied by V1 separate async.
The original async replay buffer, partial-rollout handling, full-trajectory
Teacher requests and native Trainer F+B remain intact. Async uses one optimizer
update and weight synchronization per policy batch, one prefetched warmup batch,
and `max_off_policy_threshold=8`. Its native old-logprob/advantage preparation
is retained and reported separately. Its token budget is explicitly recorded
(16384 for 1.7B, 8192 for 4B); CF uses the automatic memory planner.

```bash
MODEL_ROOT=/path/to/models MATRIX_RESULT_DIR=/tmp/opd-matrix \
  bash benchmarks/streamopd_cf/run_async_matrix.sh
uv run --active --no-sync python -m benchmarks.streamopd_cf.summarize_async_matrix \
  /tmp/opd-matrix --steps 5 --warmup 2 --output /tmp/opd-matrix/report
```

Each case runs five steps and discards two warmup steps. Compare consumed
response tokens/second as well as step duration, because generation length and
staleness can differ. Prefetched trajectories left at shutdown are not counted
as trained throughput. The report includes stage times and maximum behavior
version lag. A faster stale-policy run does not establish equal training quality.

Every run records `gpu.csv` with sampled physical GPU utilization/memory. CF
also records CUDA phase intervals in `timelines/step-N.json`, with a common
single-host monotonic clock. Render a measured step with:

```bash
uv run --active --no-sync --with matplotlib python -m benchmarks.streamopd_cf.plot_timeline \
  /tmp/opd-matrix/default/streamopd-cf/timelines/step-3.json \
  --gpu-csv /tmp/opd-matrix/default/streamopd-cf/gpu.csv \
  --output /tmp/opd-matrix/default/streamopd-cf/timeline-step-3.png
```

CUDA intervals exclude waits for input packets. Rollout lifetime and Teacher
service intervals include host/queue time; sampled GPU utilization provides a
separate check of physical activity and is not a kernel trace. The overlap
metrics report actual Trainer compute during those service intervals, not the
entire lifetime of the dispatched Trainer task. Use fresh result directories
after changing source or settings; historical runs must not be mixed into an
optimization ablation.

`profile_loss.py` isolates CF's actual fused linear/KL loss at the model's hidden
and vocabulary dimensions. Larger loss tiles reduce repeated LM-head gradient
GEMMs and memory traffic; they consume more workspace, which the memory planner
accounts for. Use `LOSS_CHUNK_SIZE=512` to reproduce the original tile setting.
