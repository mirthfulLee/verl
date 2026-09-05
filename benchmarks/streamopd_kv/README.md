# StreamOPD benchmark

The end-to-end comparison uses this repository's V1 Native OPD as the sync baseline. StreamOPD supports four
physical Trainer placements: phase-shared with Teacher GPUs, phase-shared with Rollout GPUs, spanning disjoint
Teacher and Rollout subsets (`union`), or dedicated GPUs. A shared pool has one owner at a time: its vLLM process
enters level-2 sleep before Trainer state is loaded, and Trainer FSDP state is offloaded before vLLM wakes again.
The reverse planner measures the pool after the first sleep and accounts for CUDA contexts, graph pools, and any
other process allocations that vLLM sleep retains; sleep is not treated as an empty-GPU guarantee.
The default is `union`: with a 2 + 2 Teacher/Rollout topology, Trainer uses all four GPUs during training.

StreamOPD is under active development. Benchmark outputs under `benchmarks/streamopd_kv/results/` remain local
because scheduler, memory-planning, and transport changes can quickly invalidate them. Once the method stabilizes,
publication-quality comparisons and their complete experimental settings should be recorded in dedicated
documentation.

## Environment

The development benchmark environment uses Python 3.12, PyTorch with CUDA 12.8, vLLM 0.15.1, FlashAttention 2,
Liger, and A800-80GB GPUs. The environment is created and maintained with `uv`.
Activate the CUDA environment for the driver and Ray workers:

```bash
export PATH="$PWD/.venv-cu128/bin:$PATH"
export VIRTUAL_ENV="$PWD/.venv-cu128"
export VERL_USE_UV=0
```

vLLM 0.15.1 currently constrains Transformers to 4.x while the repository metadata requires Transformers 5.x.
The validated benchmark environment keeps Transformers 4.57.6 for vLLM runtime compatibility; `uv pip check` will
report this single metadata conflict. Do not let project-level `uv run --all-packages` replace `.venv-cu128` with an
unvalidated CUDA/Transformers solve when reproducing these results.

The benchmark and production example default to `distillation.streamopd_kv.runtime_profile=auto` and reuse verl's
existing Student/Trainer, Teacher, and Rollout resource options. StreamOPD adds no separate GPU-count configuration;
its topology setting is `trainer_placement`. Baseline modes ignore `MICRO_BATCH_SIZE`.

The auto profile gives each Rollout and Teacher vLLM instance exclusive ownership of its assigned pool during its
active phase. Each vLLM worker sizes its byte budget from the free memory measured after CUDA and NCCL initialization,
reserves one measured activation peak for runtime and each configured CUDA graph mode plus deterministic sampler and
StreamOPD workspaces, and preserves vLLM's native 150 MiB profiling-error allowance before reporting the KV blocks it
actually allocated. No model-size heuristic or
`gpu_memory_utilization` tier is used. A shared Rollout uses a durable two-phase Host checkpoint so Trainer state is
offloaded before Rollout weights are woken for publication. A dedicated Trainer selects reverse shape at startup; a
shared Trainer selects it once after its inference pools first enter level-2 sleep, so retained process allocations
are included in measured headroom. The selected batch width and chunk size then remain fixed. The default EOS Host
exporter allocates no GPU gather output, so exclusive vLLM sizing does not reserve one; GPU workspace is reserved only
for export strategies that actually launch the gather kernel.

Auto does not select model parallelism. Teacher TP/replica layout remains a user resource choice because it changes
both Teacher throughput and, for a shared Trainer, the memory retained after sleep. The matrix records the requested
TP explicitly rather than choosing it from a model name or a weight-size heuristic.

## End-to-end matrix

Run the factor-covering Qwen3-1.7B/4B and Qwen3-4B/14B benchmark at 4096/8192 tokens and batch sizes
128/256 with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 TOTAL_TRAINING_STEPS=2 \
  bash benchmarks/streamopd_kv/run_colocate_matrix.sh
```

Set `SKIP_COMPLETED=1` to resume a partially completed matrix without replacing logs that already contain every
configured training step.

The default four cases cover both model pairs, both token lengths, both batch sizes, and both pool schemes without
running their full Cartesian product. Set `FULL_MATRIX=1` for the exhaustive matrix, or override `COVERING_CASES`
with space-separated `student:teacher:teacher_tp:tokens:batch:topology` records.

Both implementations use FlashAttention 2, Liger, vLLM CUDA graphs, and FSDP `no_sync` gradient accumulation by
default. These are common execution settings, not StreamOPD optimizations. StreamOPD reverse attention uses
FlashAttention's native GQA path. Auto preflight considers reverse chunks up to the trajectory length and maximizes
the total token tile; for equal tiles it avoids a singleton batch before preferring a longer chunk. A one-chunk plan
has no page-reuse window, so preflight also reserves an inactive K/V buffer and enables whole-group prefetch only when
that extra allocation fits the measured memory budget. Multi-chunk plans retain the smaller page-reuse slot. vLLM
batch-token and sequence limits are derived from the trajectory length, global batch, and replica allocation so the
baseline is not measured with artificially lower inference concurrency.

Every matrix case enables memory-bounded top-k normalization. The sync baseline's small custom autograd function
recomputes each 512-token FP32 chunk during backward, so vocabulary intermediates do not accumulate across the token
dimension. StreamOPD instead bounds the same normalization to the valid rows of its active reverse tile. Neither path
selects behavior from a StreamOPD topology or model name; set `TOPK_CHUNK_SIZE` to change the baseline workspace cap.

Gradient checkpointing is likewise applied uniformly to every matrix case and both implementations. Set
`ENABLE_GRADIENT_CHECKPOINTING=False` for a matrix-wide ablation; the runner never selects it from a model name.
Teacher `max_num_seqs` is also matched at 32 by default and can be changed for both sides with
`MATCHED_TEACHER_MAX_NUM_SEQS`. Rollout and Teacher model-length, batch-token, and sequence limits are passed through
one common override list, so StreamOPD auto planning cannot silently drift from the baseline controls. A paired
batch-128 check found less than 0.3% full-step difference between Teacher limits 32 and 64, so matching this
low-impact scheduler bound removes an unnecessary configuration difference without tuning either method to the
tested model.

### Attribution boundary

StreamOPD performance work is limited to its KV export and handoff, streaming Teacher scheduling, reverse Trainer,
and the phase transitions required by shared pools. Changes to kernels, losses, batching, data loading, or other
paths also used by `verl-sync-opd` are environment controls: the matrix must apply them identically to both methods,
and their effect must be reported separately rather than counted as StreamOPD speedup. A common-path optimization is
included in this harness only when it is required to make a covered workload feasible or to keep the comparison
symmetric; it is not used to tune one side of an ablation.

| Category | Matched or intentionally different settings |
| --- | --- |
| Matched execution controls | Dataset and seed, greedy sampling, batch and token limits, FlashAttention 2, Liger, gradient checkpointing, FSDP `no_sync`, forward-KL top-k objective and `topk=32`, CUDA graphs, worker count, and Teacher `max_num_seqs` |
| StreamOPD method | Stateful streaming Teacher requests and scheduling; EOS KV export followed by chunked reverse backward over the Rollout KV cache |
| Required pool mechanics | Exclusive-phase vLLM sizing, level-2 sleep/wake, Trainer state offload for shared placements, and the Host weight checkpoint |
| Resource-allocation difference | Only the GPU split implied by `union` or `dedicated`; comparisons keep the total physical GPU count equal |

Required pool mechanics are timed and reported, but are not presented as algorithmic innovations. In particular,
the auto profile does not modify common Trainer kernel choices; the benchmark runner applies such choices to both
implementations.

Within the streaming-Teacher path, prompt-logprob LM-head projection and FP32 normalization use a bounded 1024-row
tile. This matches the auto profile's maximum fragment size and has a runtime free-memory fallback for larger manual
fragments; it does not select a tile from a model name, GPU type, batch size, or token limit. Cross-request LM-head
batching is intentionally absent: paired measurements found no gain, while it added a second request aggregation
layer on top of vLLM's scheduler.

The sync baseline retains its native bounded Rollout allocation. Giving that Rollout the StreamOPD
`exclusive_free` policy is not valid: the decode phase can consume the measured free memory, but native weight sync
tries to restore the KV allocation while actor/training state is still resident. The batch-128 Qwen3-1.7B/4B check
failed at that wake with a CUDA OOM. StreamOPD can use `exclusive_free` because its phase-exclusive Host checkpoint
releases Trainer state before waking Rollout. This is a measured architectural constraint and is reported as pool
mechanics, not attributed to streaming Teacher scoring or reverse backward.

The default matrix compares two topologies. `union` uses four physical GPUs: Teacher 2 + Rollout 2, followed by
Trainer on all 4; its baseline uses Trainer/Rollout 2 + Teacher 2. `dedicated` uses six physical GPUs: Teacher 2 +
Rollout 2 + Trainer 2; its matched-total-GPU baseline uses Trainer/Rollout 4 + Teacher 2. Results are stored in a
separate directory per topology so the two baseline allocations cannot be mixed.

Every run records the three stages separately. `Rollout EOS` and `Teacher done` are batch completion times measured
from policy dispatch, while `rollout_span` and `teacher_span` cover the first start through the last completion of
each overlapping stage. `teacher_tail` is the interval from the final Rollout EOS to the final Teacher score. The
training value is the actor update duration for the sync baseline and scheduler-accounted Trainer busy time for
StreamOPD. Per-trajectory Rollout and Teacher request mean/max values are retained in `summary.json`; overlapping
stage times must not be added to estimate total step time.

The auto scheduler counts Teacher prefill capacity across all inference replicas. Shared Trainer placements submit
the complete batch only after their inference pools sleep. A dedicated Trainer submits several reverse waves per
unit so the next wave's Host-to-Device KV copy can overlap the current backward pass; the derived unit size, wave
size, and waves per unit are included in `summary.json`.

Teacher sessions that exceed the measured live-KV capacity wait once on a central FIFO admission queue. Session
release and asynchronous Teacher wake notify that queue directly; no model-dependent retry interval or client-side
admission polling is used.

For an isolated same-workload Rollout KV export A/B, use:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  .venv-cu128/bin/python benchmarks/streamopd_kv/benchmark_vllm_kv_export.py \
  --model /nasdata/Model/Qwen3-4B --mode eos_host \
  --batch-size 128 --max-num-seqs 64 --max-total-tokens 4096 \
  --chunk-size 2048 --writer-threads 4 --warmup-batches 1
```

The available modes are `no_export`, `eos_host`, `eos_triton`, and `incremental_triton`. The tool validates every
sealed slot, reports a token digest for cross-mode identity, and separates full-batch warmup from the measured batch.
The EOS Host submitter pipelines D2H for later trajectories while prior Host commits and seals finish. Four writers
are the default bounded double-buffering pool; increasing it also increases persistent pinned Host memory linearly.

Generate a local machine-readable summary with:

```bash
python benchmarks/streamopd_kv/summarize_colocate_matrix.py <result-dir>
```

## Teacher StreamingInput validation

Validate resumable Teacher sessions independently with:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 .venv-cu128/bin/python \
  benchmarks/streamopd_kv/validate_vllm_streaming_input.py \
  --model /nasdata/Model/Qwen3-4B --num-sessions 16 \
  --max-model-len 4097 --max-num-batched-tokens 2048 \
  --sequence-length 3200 --prompt-length 100 --response-chunk 1066 \
  --skip-tokenizer-init --attention-backend FLASH_ATTN
```

For a bitwise full-request oracle, add `VLLM_BATCH_INVARIANT=1` to the environment. Normal high-throughput
FlashAttention is sensitive to prefill partition shape, so its full-vs-streamed top-k comparison is a numerical
stability measurement rather than an exact session-alignment test. The validator reports per-fragment and boundary
metrics plus `-1/0/+1` row-offset matches to distinguish kernel drift from an indexing error.
Add `--exclusive-gpu-memory --report-sleep-memory` to report free memory before and after level-2 sleep when auditing
a shared-pool topology. This is a diagnostic only; production reverse planning obtains its headroom from the actual
shared Trainer ranks after all overlapping inference pools have slept.

Reverse Trainer metrics include the capacity, allocation time, staged bytes, Host staging time, CUDA time, and exposed
activation wait for its persistent one-group pinned buffer. Use `reverse_slot_next_wait_seconds`, not the sum of
overlapped transfer counters, when attributing training wall time.

## Kernel stage benchmark

Run a two-GPU numerical and stage-timing check with:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. .venv-cu128/bin/python \
  benchmarks/streamopd_kv/benchmark_qwen3.py \
  --student /nasdata/Model/Qwen3-4B \
  --teacher /nasdata/Model/Qwen3-14B \
  --student-dtype float32 --teacher-dtype float32 \
  --dataset-index 601 --prompt-tokens 512 --response-tokens 64 \
  --token-chunk-size 32 --reverse-chunk-size 256
```

This benchmark excludes model loading and warmup. End-to-end runs are required to include Host KV transport,
scheduler gaps, checkpoint publication, and pipeline fill/drain.
