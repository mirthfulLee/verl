# StreamOPD

StreamOPD is an experimental strict on-policy distillation path for verl V1. Teacher and Rollout remain separate
model processes; their physical GPU placement relative to Trainer is user-selected:

```text
teacher    Trainer spans Teacher GPUs; Rollout is separate
rollout    Trainer spans Rollout GPUs; Teacher is separate
union      Trainer spans disjoint Teacher + Rollout subsets
dedicated  Trainer, Teacher, and Rollout use disjoint GPUs
```

The Rollout server process remains resident and uses continuous batching. Shared roles are separate Ray processes in
the same placement group; model files are not reloaded per training unit, but vLLM-managed device mappings may be
discarded and restored at phase boundaries. The scheduler serializes kernels on intersecting resource sets and allows
work on disjoint sets to overlap. The trainer is registered as `streamopd`.

## Minimal configuration

The default `runtime_profile=auto` treats scheduling and memory layout as
implementation details. GPU pools remain entirely user-controlled through the
existing verl resource options; StreamOPD does not introduce parallel GPU-count
settings:

```bash
DATASET=/path/to/train.parquet \
TRAINER_PLACEMENT=union TEACHER_GPUS=2 ROLLOUT_GPUS=2 \
  bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh
```

The aliases above map to `trainer.n_gpus_per_node`,
`distillation.n_gpus_per_node`, and
`actor_rollout_ref.rollout.n_gpus_per_node`; they do not count as StreamOPD
options. Node counts and model parallelism likewise retain their existing verl
configuration. In the command above, the example derives `STUDENT_GPUS=4` for the default `union` placement, letting
Trainer use the combined Teacher and Rollout pools after inference;
`trainer_placement` is the only optional StreamOPD topology choice. Before any model server starts, the auto profile
derives context limits, Rollout and Teacher concurrency, Teacher fragment size, Teacher batched-token capacity,
checkpoint transport, and zero-valued reverse-plan sentinels from the global batch, maximum trajectory length, GPU
allocation, and placement. Existing verl options explicitly supplied as Hydra overrides are hard constraints rather
than values for auto mode to replace. Auto mode never derives, rewrites, or passes `gpu_memory_utilization`. Each
phase-exclusive vLLM worker sizes KV from the actual free bytes after CUDA/NCCL setup, activation/CUDA-graph profiling,
and deterministic workspace reservations. KV allocation is capped at the page-aligned worst case for the entire
global policy batch on each replica, without assuming even routing or reducing feasible session concurrency.
This avoids allocating unusable cache capacity for small jobs. It reserves no simultaneously active Trainer workspace because vLLM enters
level-2 sleep before Trainer state and reverse slots are loaded. A worst-case Host KV backing check also runs before
Rollout starts; auto mode uses
`/dev/shm`, while an explicitly selected path fails closed if it lacks capacity. Teacher session admission is refined
after vLLM reports its actual paged-KV capacity, so group/session widths are not fixed to the reference hardware.
Common Trainer execution choices such as gradient checkpointing, Liger, and FSDP `no_sync` remain ordinary verl
options and are never changed by the StreamOPD auto planner.

Auto deliberately does not select Teacher TP or replica count. Model parallelism remains a normal user resource
choice: more replicas can improve Teacher prefill throughput, while a shared Trainer may receive less reverse
headroom from the resulting sleeping processes. Shared reverse planning uses the post-sleep measurement, not a
model-name rule, so an infeasible chunk or batch width is rejected before the first training phase.

Set `distillation.streamopd_kv.runtime_profile=manual` only for ablations or
parameter-sensitivity studies. Manual mode preserves every low-level setting,
including engine utilization, chunk/page/group sizes, fixed-slot reserve,
Teacher session caps, and prefetch depth. Zero-valued planning sentinels are still resolved; manual mode does not
disable validation or the streaming Teacher protocol. The resolved auto
values are logged under `streamopd/runtime_profile_*`.

## Supported envelope

The implementation fails closed outside the following configuration:

- `trainer.use_v1=true` and `trainer.v1.trainer_mode=streamopd`;
- `trainer_placement=teacher`, `rollout`, `union`, or `dedicated` on one node;
- text-only, single-turn Qwen3 with one frozen teacher;
- vLLM rollout with TP=1, PP=1, `n=1`, and a non-quantized KV cache;
- FSDP/FSDP2 student, one PPO epoch, and global `token-mean` aggregation;
- direct `forward_kl_topk` distillation without task rewards or a policy-gradient term;
- exact dense attention using native-GQA CUDA FlashAttention for batched wavefront reverse, with no SDPA fallback;
- BF16 CUDA KV/query tensors, `head_dim <= 256`, right-padded batches, and page-aligned reverse chunks.

`distillation.streamopd_kv.enabled=true` is rejected unless `trainer.v1.trainer_mode=streamopd` is selected.
The default `union` placement runs Teacher and Rollout on disjoint pools, then lends both pools to Trainer after
their inference phases finish. `dedicated` keeps a separate Trainer pool for overlap experiments.

## Runtime compatibility

Use the repository's Python 3.12 / vLLM / FSDP environment (`uv sync --extra vllm --extra fsdp`).
vLLM 0.24.0 supports the Teacher StreamingInput artifacts natively. The experimental `vllm_patch.py` also retains
a narrowly scoped compatibility patch for vLLM 0.15.1; it is installed only on StreamOPD Teacher workers.
The older CUDA 12.8 development environment is optional and is not the repository's default dependency stack.

The default `eos_host` exporter requires vLLM's uniform cross-layer cache. Its Rollout server selects
`VLLM_USE_V2_MODEL_RUNNER=0` before creating the engine; the selection is local to that server and its children.
An explicit incompatible runner selection fails before startup. The `eos_triton` and `incremental_triton` exporters
remain available for controlled ablations and other supported cache layouts. Select them explicitly so the memory
planner reserves their GPU gather workspace; `eos_host` never silently switches to a GPU gather implementation.
Only `eos_host` is used by the default EOS-only pipeline described below.

For `trainer_placement=rollout` or `union`, `checkpoint_engine.backend=host` is required because Trainer publication
must finish and offload before Rollout wakes. Other placements can use a cross-process checkpoint backend.
The configured reverse slot must cover the complete prompt/response upper bounds. Invalid page sizes, missing or
multiple effective teachers, non-vLLM teachers, and incomplete GPU replica allocations are rejected during config
preparation, before starting model workers. Named teachers follow verl's existing convention: adding a custom teacher
entry replaces the `teacher_model` template, and auto mode preserves overrides using the custom entry's actual path.

## Code ownership

| Component | Responsibility |
| --- | --- |
| `config.py`, `planning.py`, `placement.py` | Configuration schema, startup validation, memory plans, and GPU placement |
| `publisher.py`, `streaming_teacher.py`, `scheduler.py` | Committed token coverage, Teacher sessions, and policy barriers |
| `vllm_connector.py`, `host_slot_pool.py`, `snapshot_io.py` | KV export, generation-checked Host slots, and tensor views |
| `qwen3.py`, `reverse_attention.py`, `fsdp_worker.py`, `ray_worker.py` | Reverse chain rule, fixed GPU slots, and actor worker |
| `teacher_client.py`, `vllm_teacher.py` | Sticky Teacher leases and resumable vLLM input/output sessions |
| `replica_group.py` | Shared Teacher/Rollout telemetry, capacity accounting, and KV export barriers |
| `checkpoint.py` | Serialized weight handoff for shared Trainer/Rollout pools |
| `vllm_patch.py` | Opt-in, version-specific vLLM integration |
| `verl/trainer/ppo/v1/trainer_streamopd.py` | V1 trainer registration and orchestration through base trainer hooks |

The public `verl.workers.config.StreamOPDKVConfig` import remains an alias for the experimental schema. General V1
trainer hooks retain their baseline defaults. The ordinary LLM client has no streaming-session state; the Teacher
RPC lazily constructs its experimental service. The standard checkpoint manager retains its concurrent update
sequence, while `checkpoint.py` reuses its worker-group and finalization methods for shared-pool handoff. FSDP1
single-sender shard gathering belongs to the Host checkpoint backend, leaving the FSDP engine's parameter export
unchanged from upstream. The example's DAPO adapter lives beside the example; benchmark-only
ragged response controls extend that adapter in `benchmarks/streamopd_kv`.

## Streaming path

1. `CommittedChunkPublisher` observes cumulative vLLM output and publishes committed token ranges to Teacher during
   generation. The first teacher input budget includes the complete prompt plus the initial response prefix; later
   resumable chunks contain only newly committed response tokens. Rollout KV export is deliberately independent of
   this path and starts only after the complete trajectory reaches EOS.
2. `StreamingTeacherCoordinator` submits chunks to the central scheduler. vLLM resumable `StreamingInput`
   sessions retain causal teacher KV across fragments. Contiguous queued fragments may be coalesced without changing
   token coverage. A session reservation is held until EOS; releasing it immediately permits another trajectory to
   use the bounded teacher KV capacity.
3. The scheduler owns Teacher session admission, Reverse Training readiness, and explicit resource sets. Fragment
   scoring needs no second scheduler admission because each resumable session already holds its worst-case KV
   reservation. Each trajectory reports its fragment count and score intervals once at completion; the scheduler merges
   those intervals for busy/concurrency metrics without putting actor RPCs between vLLM fragments. Capacity-limited
   sessions wait once on the scheduler's FIFO admission queue; session release and Teacher wake grant reservations and
   notify waiters directly, without client polling. A dedicated Trainer
   may start a unit as soon as every trajectory in that unit has complete Rollout KV and Teacher scores. A Trainer that shares
   Teacher GPUs waits for complete Teacher drain; one that shares Rollout GPUs waits for all Rollout EOS. A union
   Trainer waits for both conditions. There is no Teacher/Trainer alternation inside a policy.
4. A shared pool changes active owner once per policy. The outgoing vLLM process enters level-2 sleep to discard its
   sleep-managed weight/KV mappings, then the Trainer loads parameters and optimizer state once for its full training
   phase. CUDA contexts, graph pools, and allocations outside vLLM's sleep allocator may remain; the first post-sleep
   reverse preflight measures that real headroom. Trainer FSDP state is offloaded before the vLLM process wakes again.
5. `StreamOPDKVConnector` claims a finished request's vLLM pages through the HMA async-save contract and exports its
   complete trajectory after EOS. The default path copies arithmetic runs of physical cross-layer pages directly to
   pinned Host staging with `cudaMemcpy2DAsync`, then performs the block-to-layer layout transform in Host writer
   threads. It runs no GPU gather kernel, never blocks the model-runner thread on a staging slot, and reports completion
   only after the shared slot is sealed. Because this path allocates no GPU gather output, the exclusive vLLM memory
   profile does not reserve one. The ordered CUDA submitter may advance to a later trajectory while an earlier Host
   commit and seal finish; four reusable writer buffers bound both this pipeline and pinned Host memory. Each TP shard
   owns one stable data mmap and one fixed control array. Multiple
   generating or sealed
   training units may be host-resident, while each trainer worker may hold exactly one GPU KV lease. Trainer prefetch
   maps the next slot as a zero-copy tensor view while the current reverse unit executes. Within the leased microbatch, the
   next reverse group may reuse suffix pages released by the current group; the next microbatch still cannot acquire
   GPU KV until the current lease is released.
6. Once `EOS && student KV complete && teacher supervision complete`, the trainer uses rollout KV as OOMB Stage-1 and
   traverses chunks from suffix to prefix. Wavefront batches contain all trajectories active at the same reverse
   depth. The highest wavefront is computed only through the group's page-aligned longest trajectory rather than the
   full fixed chunk; the unused slot suffix is released after backward without entering the model. Completed suffix
   chunks leave the wavefront, and dK/dV continues into earlier chunks.
7. Before policy version zero, each phase-exclusive vLLM worker uses the free memory measured after CUDA/NCCL setup
   and reserves one measured activation peak for runtime and each configured CUDA graph mode plus deterministic sampler
   and StreamOPD connector/logit workspaces, then reports its profiled KV block capacity. Teacher admission and a
   dedicated Trainer's reverse plan are derived at startup. A shared Trainer freezes its reverse plan after the first
   inference sleep and before its first training phase, using the retained process footprint measured on every rank.
   These plans select a stable session budget, fixed `B_slot`, page-aligned `T_slot`, chunk size, and accumulation count
   without a model-size or utilization-fraction heuristic.
   Reverse candidates maximize the useful `batch * chunk` token tile. Equal tiles avoid a singleton batch and then
   prefer the longer chunk. The hot path performs no allocator query, OOM retry, tensor growth, or kernel-shape
   adaptation.
8. Raw gradients accumulate across the complete global batch. After the final optimizer step their storage is released
   before inference wakes; intermediate units retain gradients. Parameters stay at `theta_k` until all rollout,
   teacher coverage, and reverse backward work completes. Normalization, clipping, one optimizer step, and publication
   of `theta_(k+1)` occur only at the strict policy-version barrier.
9. When Trainer shares the Rollout pool, Host checkpoint publication is phase-exclusive: Rollout enters level-2 sleep,
   Trainer publishes and offloads the durable checkpoint, Rollout wakes only its weights to receive it, and KV wakes
   last. Active Trainer state is gone before Rollout mappings are restored; sleep-retained process allocations remain
   part of the measured shared-pool budget.

Rollout throughput is configured independently through vLLM continuous-batching limits such as `max_num_seqs`.
`rollout_kv_export_chunk_size` is also independent of `token_chunk_size`: auto uses a bounded 2048-token export chunk
and four Host writers, while the latter remains the Teacher streaming granularity. Set reverse batch/chunk and
Teacher admission caps to zero (the default) for automatic preflight planning; non-zero values are experimental caps.

## Fixed reverse slots

Each layer owns stable `[B_slot, T_slot, H_kv, D]` K, V, dK, and dV tensors.
`reverse_slot_max_tokens=0` resolves `T_slot` from the configured
prompt/response upper bounds and page-aligns it. Preflight evaluates the fixed backing, active reverse workspace,
LM-head workspace, model/optimizer reserve, and H2D reserve before selecting `B_slot` and a chunk size that divides
`T_slot`. The optimizer reserve is derived per rank from gradients and optimizer tensors that have not yet been
materialized; it is not a hardware-specific fixed allowance. The resulting addresses and kernel shapes are frozen
before the first training phase. Dedicated pools plan at startup; shared pools
plan once after their inference process first enters level-2 sleep so retained
CUDA state is included in the measured headroom.

Host slots expose token-major contiguous `[T, H_kv, D]` K/V views directly from the mmap; there is no read-time
deserialization. Because mmap views are pageable, a dedicated enqueue thread first copies exactly one reverse group
into persistent pinned Host storage, then uses raw async CUDA memcpy/memset operations on a copy stream. The pinned
group is reused by every reverse wave and remains alive across policy steps even when phase-shared GPU slots are
released. It is bounded to half of one fixed GPU slot backing (K/V without dK/dV), rather than the complete training
batch. Host staging for the next group overlaps the current backward traversal. Every GPU slot page tracks `FREE ->
LOADING_NEXT -> NEXT_READY -> CURRENT_ACTIVE -> BACKWARD_DONE -> FREE`. After a wavefront depth commits, the compute
stream records a free event; the copy stream waits on it before overwriting that suffix range with next-group KV.
Group activation waits for all required load events. Short-trajectory tail pages and unused rows can therefore load
before the current group reaches their reverse depth. When the selected chunk spans the complete slot, there is no
release depth before backward finishes. If measured free memory can also hold a second K/V pair, preflight enables an
inactive K/V buffer so the next complete group transfers during the current backward; dK/dV remains shared. If that
extra pair does not fit, preflight selects a smaller multi-chunk plan and retains page reuse.

The exposed transfer metrics separate pinned allocation, mmap-to-pinned staging, CUDA enqueue/copy, and activation
wait. In particular, `reverse_slot_next_wait_seconds` is the transfer time visible on the training critical path;
aggregate staging and CUDA durations may overlap backward and must not be added to step time.

Preflight enumerates power-of-two reverse widths and chunks up to the trajectory length under the fixed-slot,
activation, LM-head, optimizer, and transfer reserve. It first maximizes the useful `batch * chunk` token tile. Equal
tiles prefer at least two trajectories when feasible, then the longer chunk to reduce wavefront depth. The memory
test includes six KV tensors for a one-chunk prefetch plan and four for a page-reuse plan. Non-zero token/batch/chunk
caps remain available for controlled ablations.

## Baseline isolation

When `distillation.streamopd_kv.enabled=false`, StreamOPD config preparation returns without modifying the rollout
engine, KV connector, checkpoint backend, replay settings, or optimizer synchronization. The V1 `sync` baseline uses
its native rollout/trainer path. Its optional teacher placement is configured separately with
`distillation.colocate_teacher_with_student`; StreamOPD does not read that option.

## Transport and numerical behavior

The single-node transport preallocates `global_batch * T_slot` Host KV rows per TP shard. Fixed control records track
`FREE -> WRITING -> SEALED -> FREE`, with a monotonically increasing generation plus policy, request, trajectory, and
token-sequence digests. A stale descriptor therefore cannot release or consume a reused row. The default auto profile
places the mmap under `/dev/shm`. A future multi-node backend should preserve the
multiple-host-units/single-GPU-lease contract while replacing the shared mmap with RDMA or an equivalent streaming
transport.

The reverse chain rule, fixed-slot attention, and parameter gradients are checked against ordinary full-sequence
training. Reverse kernels are BF16-only, so tests use numerical tolerances rather than bitwise equality. Unsupported
device, dtype, head dimension, page size, or reverse shape fails before training starts.

vLLM FlashAttention can produce numerically different BF16 top-k values when the same sequence is evaluated with
different prefill partitioning. `VLLM_BATCH_INVARIANT=1` with an explicitly selected `FLASH_ATTN` backend is the
bitwise correctness oracle for StreamingInput tests; it is not a production default because its deterministic GEMM
and attention paths reduce Teacher throughput. In batch-invariant mode, multi-fragment Teacher artifacts match the
full request exactly, including across fragment boundaries.

## Validation

Tests are organized by component under `tests/experimental/streamopd_kv`. Run CPU protocol/config/transport tests with:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q tests/experimental/streamopd_kv
```

GPU tests compare reverse loss and gradients against a full-sequence Qwen3 forward and check fixed-slot page reuse
and prefetch. They build a small Qwen3 model locally by default, so no checkpoint download is required:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync pytest -q tests/experimental/streamopd_kv/test_qwen3_reverse_on_gpu.py
```

Set `STREAMOPD_TEST_MODEL_PATH=/path/to/Qwen3` to repeat the numerical comparison with a local pretrained model.
Runtime Teacher artifact validation compares every supervised row; the final native-API dummy row has no target
and is excluded. For Teacher fragment-boundary validation and end-to-end benchmark commands, see
[the benchmark guide](../../../benchmarks/streamopd_kv/README.md). These numerical and smoke checks do not establish
throughput gains; performance comparisons require matched resource allocations and execution settings.
