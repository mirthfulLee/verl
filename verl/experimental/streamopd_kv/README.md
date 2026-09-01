# StreamOPD

StreamOPD is an experimental strict on-policy distillation path for verl V1. It uses two independent GPU pools:

```text
Rollout Pool
  `- student rollout engine

Teacher/Trainer Pool
  |- frozen teacher inference engine
  `- sharded student trainer
```

The rollout engine remains resident and uses continuous batching. The Teacher/Trainer Pool keeps both models loaded
and serializes teacher forward and reverse training, so their kernels never overlap in memory. The internal trainer
registration is currently named `streamopd_colocate`; this is the only supported StreamOPD topology.

## Supported envelope

The implementation fails closed outside the following configuration:

- `trainer.use_v1=true` and `trainer.v1.trainer_mode=streamopd_colocate`;
- text-only, single-turn Qwen3 with one frozen teacher;
- vLLM rollout with TP=1, PP=1, `n=1`, and a non-quantized KV cache;
- FSDP/FSDP2 student, one PPO epoch, and global `token-mean` aggregation;
- direct `forward_kl_topk` distillation without task rewards or a policy-gradient term;
- exact dense attention using CUDA FlashAttention for batched wavefront reverse and the vendored OOMB Triton kernel
  for singleton validation, with no SDPA fallback;
- BF16 CUDA KV/query tensors, `head_dim <= 256`, right-padded batches, and page-aligned reverse chunks.

The removed legacy integration enabled StreamOPD inside the `sync` trainer, where rollout and trainer shared a GPU
pool. `distillation.streamopd_kv.enabled=true` is now rejected unless the dedicated two-pool trainer mode is selected.

## Streaming path

1. `CommittedChunkPublisher` observes cumulative vLLM output and publishes every committed token/KV range during
   generation. The first teacher input budget includes the complete prompt plus the initial response prefix; later
   resumable chunks contain only newly committed response tokens. EOS seals the already-streamed KV manifest.
2. `StreamingTeacherCoordinator` submits chunks to the central scheduler. vLLM 0.15.1 resumable `StreamingInput`
   sessions retain causal teacher KV across fragments. Contiguous queued fragments may be coalesced without changing
   token coverage. A session reservation is held until EOS; releasing it immediately permits another trajectory to
   use the bounded teacher KV capacity.
3. The scheduler owns separate Teacher Chunk and Reverse Training queues. Teacher forward waits while a training unit
   is active, and training starts only after active teacher kernels finish and pending teacher work is within the
   configured priority threshold. There is no fixed Teacher cohort or cohort-boundary model sleep/wake path.
4. `StreamOPDKVConnector` copies completed rollout KV ranges from live vLLM pages into the host cache before EOS.
   Multiple generating or sealed microbatches may be host-resident, while each trainer worker may hold exactly one GPU
   KV lease. A bounded prefetch queue reads the next host snapshot while the current reverse unit executes. Within the
   leased microbatch, the next reverse group may reuse suffix pages released by the current group; the next microbatch
   still cannot acquire GPU KV until the current lease is released.
5. Once `EOS && student KV complete && teacher supervision complete`, the trainer uses rollout KV as OOMB Stage-1 and
   traverses chunks from suffix to prefix. Wavefront batches contain all trajectories active at the same reverse
   depth. Completed suffix chunks leave the wavefront, and dK/dV continues into earlier chunks.
6. Before policy version zero, the trainer measures reusable memory once and chooses fixed `B_slot`, page-aligned
   `T_slot`, and chunk size. It allocates persistent per-layer K/V/dK/dV backing and reuses the plan for all steps; the
   hot path performs no allocator query, OOM retry, tensor growth, or kernel-shape adaptation.
7. Raw gradients accumulate across the complete global batch. Parameters stay at `theta_k` until all rollout,
   teacher coverage, and reverse backward work completes. Normalization, clipping, one optimizer step, and publication
   of `theta_(k+1)` occur only at the strict policy-version barrier.

`micro_batch_size` controls Teacher/Trainer work only. Rollout throughput is configured independently through vLLM
continuous-batching limits such as `max_num_seqs`.

## Fixed reverse slots

`reverse_fixed_slots=true` removes per-group GPU snapshot and contiguous-backing allocation. Each layer owns stable
`[B_slot, T_slot, H_kv, D]` K, V, dK, and dV tensors. `reverse_slot_max_tokens=0` resolves `T_slot` from the configured
prompt/response upper bounds and page-aligns it. Preflight evaluates the fixed backing, active reverse workspace,
LM-head workspace, model/optimizer reserve, and H2D reserve before selecting `B_slot` and a chunk size that divides
`T_slot`.

Host snapshots are loaded directly in token-major `[T, H_kv, D]` layout. A dedicated enqueue thread uses raw async
CUDA memcpy/memset operations on a copy stream, avoiding transpose dispatch and per-group GPU staging tensors. Every
slot page tracks `FREE -> LOADING_NEXT -> NEXT_READY -> CURRENT_ACTIVE -> BACKWARD_DONE -> FREE`. After a wavefront
depth commits, the compute stream records a free event; the copy stream waits on it before overwriting that suffix
range with next-group KV. Group activation waits for all required load events. Short-trajectory tail pages and unused
rows can therefore load before the current group reaches their reverse depth.

The configured token cap remains part of planning. For a 4096-token trajectory, the default 32768-token cap selects
B8. Raising it to 65536 can fit B16, but may force a smaller chunk and increase backward depth; fitting in memory is
not by itself a throughput win.

## Post-hoc ablation

Set `distillation.streamopd_kv.posthoc_ablation=true` to isolate the value of Teacher chunk streaming and early
reverse training. This mode preserves the same two GPU pools, rollout KV connector, host KV cache, reverse kernels,
microbatch sizes, and strict policy barrier. It changes only the Teacher/Trainer schedule:

1. each trajectory sends no Teacher request before EOS;
2. as soon as that individual trajectory reaches EOS, it immediately submits one complete `prompt + response`
   prefill to Teacher vLLM; it does not wait for the other trajectories;
3. rollout KV continues to stream into the host cache before EOS;
4. reverse training remains blocked until every trajectory in the global batch is terminal and has complete Teacher
   artifacts, then processes the existing reverse microbatches on the Teacher/Trainer Pool only.

The scheduler records the first Teacher start, all-rollout EOS, all-Teacher completion, Teacher busy time before and
after all-rollout EOS, reverse busy time, pool idle/utilization, and the Teacher drain tail. Per-step allocator peaks
are reset and collected independently for rollout vLLM, Teacher vLLM, and student trainer workers.

## Baseline isolation

When `distillation.streamopd_kv.enabled=false`, StreamOPD config preparation returns without modifying the rollout
engine, KV connector, checkpoint backend, replay settings, or optimizer synchronization. The V1 `sync` baseline uses
its native rollout/trainer path. Its optional teacher placement is configured separately with
`distillation.colocate_teacher_with_student`; StreamOPD does not read that option.

## Transport and numerical behavior

The current numbered safetensors handoff is a correctness-first incremental host transport. A production multi-node
backend should preserve the multiple-host-MB/single-GPU-lease contract while replacing filesystem serialization with
CUDA IPC, RDMA, or an equivalent streaming transport.

The reverse chain rule, dense paged attention, and parameter gradients are checked against ordinary full-sequence
training. OOMB kernels are BF16-only, so tests use numerical tolerances rather than bitwise equality. Unsupported
device, dtype, head dimension, page size, or reverse shape fails before training starts.
