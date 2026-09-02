# StreamOPD

StreamOPD is an experimental strict on-policy distillation path for verl V1. Teacher and Rollout remain separate
model processes; their physical GPU placement relative to Trainer is user-selected:

```text
teacher    Trainer spans Teacher GPUs; Rollout is separate
rollout    Trainer spans Rollout GPUs; Teacher is separate
union      Trainer spans disjoint Teacher + Rollout subsets
dedicated  Trainer, Teacher, and Rollout use disjoint GPUs
```

The rollout engine remains resident and uses continuous batching. Shared roles are separate Ray processes in the same
placement group; model loading is not repeated per unit. The scheduler serializes kernels on intersecting resource
sets and allows work on disjoint sets to overlap. The trainer is registered as `streamopd`.

## Minimal configuration

The default `runtime_profile=auto` treats scheduling and memory layout as
implementation details. GPU pools remain entirely user-controlled through the
existing verl resource options; StreamOPD does not introduce parallel GPU-count
settings:

```bash
STUDENT_GPUS=2 TEACHER_GPUS=2 ROLLOUT_GPUS=2 \
  bash examples/on_policy_distillation_trainer/run_qwen3_streamopd_kv_fsdp.sh
```

The aliases above map to `trainer.n_gpus_per_node`,
`distillation.n_gpus_per_node`, and
`actor_rollout_ref.rollout.n_gpus_per_node`; they do not count as StreamOPD
options. Node counts and model parallelism likewise retain their existing verl
configuration. The default placement shares Teacher and Trainer GPUs;
`trainer_placement` is the only optional StreamOPD topology choice. Before any model server starts, the auto profile
derives context limits, Rollout and Teacher concurrency, committed-token chunk size, Teacher batched-token capacity,
checkpoint transport, and zero-valued reverse-plan sentinels from the global batch, maximum trajectory length, GPU
allocation, and placement. Existing verl options explicitly supplied as Hydra overrides are hard constraints rather
than values for auto mode to replace. If only Rollout memory is constrained, the planner lowers continuous-batching
concurrency; if only concurrency is constrained, it raises memory to the smallest sufficient `0.05` utilization step.
If neither is constrained, it targets the whole per-replica rollout batch and reduces concurrency only when the
physical GPU cannot hold its worst-case KV.

Both dedicated and Trainer-shared Rollout memory are calculated from checkpoint weight bytes, model KV geometry,
maximum trajectory length, physical GPU capacity, and runtime/checkpoint reserves. For `rollout` and `union`, fixed
reverse slots are planned before Rollout vLLM starts; the target Rollout ranks' remaining memory minus the frozen
reverse workspace determines a `0.05`-aligned vLLM ceiling and stable continuous-batching width. A worst-case host KV
backing check also runs before Rollout starts; auto mode uses `/dev/shm`, while an explicitly selected path fails closed
if it lacks capacity. Teacher/Trainer sharing starts from a `0.25` Teacher envelope and is refined after vLLM reports
its actual paged-KV capacity and Trainer preflight measures the remaining memory. Group/session widths therefore are
not fixed to the reference hardware.

Set `distillation.streamopd_kv.runtime_profile=manual` only for ablations or
parameter-sensitivity studies. Manual mode preserves every low-level setting,
including engine utilization, chunk/page/group sizes, fixed-slot reserve,
Teacher session caps, prefetch depth, and scheduler policy. The resolved auto
values are logged under `streamopd/runtime_profile_*`.

## Supported envelope

The implementation fails closed outside the following configuration:

- `trainer.use_v1=true` and `trainer.v1.trainer_mode=streamopd`;
- `trainer_placement=teacher`, `rollout`, `union`, or `dedicated` on one node;
- text-only, single-turn Qwen3 with one frozen teacher;
- vLLM rollout with TP=1, PP=1, `n=1`, and a non-quantized KV cache;
- FSDP/FSDP2 student, one PPO epoch, and global `token-mean` aggregation;
- direct `forward_kl_topk` distillation without task rewards or a policy-gradient term;
- exact dense attention using CUDA FlashAttention for batched wavefront reverse, with no SDPA fallback;
- BF16 CUDA KV/query tensors, `head_dim <= 256`, right-padded batches, and page-aligned reverse chunks.

`distillation.streamopd_kv.enabled=true` is rejected unless the dedicated StreamOPD trainer mode is selected.

## Streaming path

1. `CommittedChunkPublisher` observes cumulative vLLM output and publishes every committed token/KV range during
   generation. The first teacher input budget includes the complete prompt plus the initial response prefix; later
   resumable chunks contain only newly committed response tokens. EOS seals the already-streamed Host KV slot.
2. `StreamingTeacherCoordinator` submits chunks to the central scheduler. vLLM 0.15.1 resumable `StreamingInput`
   sessions retain causal teacher KV across fragments. Contiguous queued fragments may be coalesced without changing
   token coverage. A session reservation is held until EOS; releasing it immediately permits another trajectory to
   use the bounded teacher KV capacity.
3. The scheduler owns separate Teacher Chunk and Reverse Training queues plus explicit resource sets. Shared resources
   are mutually exclusive; dedicated resources can overlap. `teacher_then_train` is the drain-first baseline.
   `adaptive` launches as soon as one reverse cohort is genuinely ready. On shared GPUs, Teacher-complete trajectories
   that have not entered reverse training form a backlog: Trainer retains the pool while another full cohort is
   available, then yields to Teacher when it catches up. A short bounded handoff window lets the controller register
   that next cohort without leaving the GPU idle. This makes the reverse quantum follow the producer/consumer rates
   instead of a fixed model- or hardware-specific count. A Rollout-shared Trainer waits for all Rollout EOS because the
   current KV connector is non-preemptible, then overlaps reverse work with the dedicated Teacher tail. A union Trainer
   has no remaining disjoint critical-path work after Rollout EOS, so adaptive safely collapses to drain-first. Live
   Teacher sessions refill as earlier trajectories reach EOS; there is no per-unit model sleep/wake boundary.
4. At a shared Teacher/Trainer role switch, inactive Teacher allocator cache is trimmed while weights, live paged KV,
   and request state remain resident. Once all Teacher sessions drain, vLLM enters level-1 sleep; the next policy
   version wakes it behind a scheduler admission gate while Rollout is already starting.
5. `StreamOPDKVConnector` copies completed rollout KV ranges from live vLLM pages into a fixed shared Host KV slot pool
   before EOS. Each TP shard owns one stable data mmap and one fixed control array; a bounded pinned staging ring moves
   D2H chunks into their assigned rows without serializing per-chunk tensor files. Multiple generating or sealed
   training units may be host-resident, while each trainer worker may hold exactly one GPU KV lease. Trainer prefetch
   maps the next slot as a zero-copy tensor view while the current reverse unit executes. Within the leased microbatch, the
   next reverse group may reuse suffix pages released by the current group; the next microbatch still cannot acquire
   GPU KV until the current lease is released.
6. Once `EOS && student KV complete && teacher supervision complete`, the trainer uses rollout KV as OOMB Stage-1 and
   traverses chunks from suffix to prefix. Wavefront batches contain all trajectories active at the same reverse
   depth. Completed suffix chunks leave the wavefront, and dK/dV continues into earlier chunks.
7. Before policy version zero, vLLM reports its profiled KV block capacity and the trainer measures reusable memory
   once. Preflight derives a stable Teacher session budget, fixed `B_slot`, page-aligned `T_slot`, chunk size, and
   accumulation count. For Rollout-shared GPUs it also budgets complete non-preemptible Rollout KV, model weights,
   runtime workspace, and checkpoint sender/receiver buffers; it caps reverse group tokens before allocating slots.
   Reverse candidates prefer the largest feasible chunk and then the largest batch at that chunk. The hot path performs
   no allocator query, OOM retry, tensor growth, or kernel-shape adaptation.
8. Raw gradients accumulate across the complete global batch. Parameters stay at `theta_k` until all rollout,
   teacher coverage, and reverse backward work completes. Normalization, clipping, one optimizer step, and publication
   of `theta_(k+1)` occur only at the strict policy-version barrier.

Rollout throughput is configured independently through vLLM continuous-batching limits such as `max_num_seqs`. Set
reverse batch/chunk and Teacher admission caps to zero (the default) for automatic preflight planning; non-zero values
are experimental caps.

## Fixed reverse slots

Each layer owns stable `[B_slot, T_slot, H_kv, D]` K, V, dK, and dV tensors.
`reverse_slot_max_tokens=0` resolves `T_slot` from the configured
prompt/response upper bounds and page-aligns it. Preflight evaluates the fixed backing, active reverse workspace,
LM-head workspace, model/optimizer reserve, and H2D reserve before selecting `B_slot` and a chunk size that divides
`T_slot`. The optimizer reserve is derived per rank from gradients and optimizer tensors that have not yet been
materialized; it is not a hardware-specific fixed allowance. The resulting addresses and kernel shapes are frozen
before policy version zero.

Host slots expose token-major contiguous `[T, H_kv, D]` K/V views directly from the mmap; there is no read-time tensor
allocation or deserialization. A dedicated enqueue thread uses raw async CUDA memcpy/memset operations on a copy
stream, avoiding transpose dispatch and per-group GPU staging tensors. Every GPU slot page tracks `FREE ->
LOADING_NEXT -> NEXT_READY -> CURRENT_ACTIVE -> BACKWARD_DONE -> FREE`. After a wavefront depth commits, the compute
stream records a free event; the copy stream waits on it before overwriting that suffix range with next-group KV.
Group activation waits for all required load events. Short-trajectory tail pages and unused rows can therefore load
before the current group reaches their reverse depth.

Preflight enumerates power-of-two reverse widths under the fixed-slot, activation, LM-head, optimizer, and transfer
reserve. It first maximizes the feasible chunk (up to the profiled 1024-token kernel tile), then batch width. A larger
batch that forces a smaller chunk is not selected merely because it fits in memory. Non-zero token/batch/chunk caps
remain available for controlled ablations.

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
