# StreamOPD-KV

This package implements an experimental, two-pool strict on-policy distillation path for verl V1 training. The
`streamopd_colocate` mode keeps student rollout on a standalone GPU pool and keeps the frozen teacher plus the
sharded student trainer resident on a second pool. Committed rollout tokens and newly computed vLLM KV ranges are
published incrementally; EOS only seals the manifest for the already-streamed KV trace.

This is distinct from legacy `sync`, whose actor rollout and trainer share the global pool. In the new mode only the
teacher and trainer share a pool; rollout remains an independent producer.

The initial backend deliberately fails closed outside this envelope:

- `trainer.use_v1=true` and `trainer.v1.trainer_mode=streamopd_colocate` (the older `sync` integration remains
  available for compatibility);
- text-only, single-turn Qwen3 with one teacher;
- vLLM rollout with TP=1, PP=1, and non-quantized KV;
- FSDP/FSDP2 actor, one PPO epoch, and direct `forward_kl_topk` distillation;
- global `token-mean` loss aggregation;
- `actor.use_torch_compile=false`, because reverse traversal replaces Qwen3 attention dynamically;
- no task-reward or policy-gradient term;
- exact dense attention through CUDA FlashAttention for batched wavefront reverse and the vendored OOMB paged
  Triton kernel for singleton validation, with no SDPA fallback and no sparse or page-selection approximation;
- BF16 CUDA KV/query tensors, `head_dim <= 256`, right-padded reverse batches, and page-aligned reverse chunks.

## Execution path

1. `CommittedChunkPublisher` observes cumulative vLLM outputs and publishes only accepted token prefixes.
2. `StreamingTeacherCoordinator` appends teacher work to a central queue. Teacher forward waits while a reverse
   training unit is active; reverse training waits for the configured teacher backlog threshold. The scheduler uses
   atomic admission, so teacher and trainer kernels never overlap on the shared GPU pool.
3. The rollout vLLM engine uses continuous batching independently of the Teacher/Trainer microbatch.
   `StreamOPDKVConnector` observes computed-token progress on every scheduler step. Each complete token range is
   copied from the live NHD pages to pinned CPU memory and serialized into the Teacher/Trainer Pool-visible host
   cache before EOS. Several generating or sealed microbatches may coexist in this host cache. The finish callback
   writes only the tail and a manifest containing token identity, policy version, and contiguous extents.
4. A trainer worker may hold only one microbatch GPU KV lease. The controller waits for the current update to return
   before it admits the next ready microbatch, and the worker fails closed on a second lease. Readiness gates reverse
   backward (`EOS && KV manifest complete && teacher supervision complete`), not the earlier host-side KV transfer.
   The leased rollout KV is the no-grad Stage-1 trace. During suffix-to-prefix recomputation, exact dense attention
   propagates dK/dV into earlier chunks and returns current-chunk gradients to the trainable K/V projections.
5. `micro_batch_size` configures Teacher/Trainer work only; rollout continuous-batching limits remain independent.
   Reverse chunk size is bounded between `reverse_chunk_min_size` and `reverse_chunk_size` and aligned to
   `reverse_page_size`. The A100 benchmark defaults to page size 64, maximum chunk 1024, and minimum chunk 256.
   Equal-rank trajectories are packed into wavefront batches while
   `max_sequence_length * batch_size <= reverse_batch_max_tokens`.
   Finished trajectories leave the batch at higher reverse depths, so shorter traces do not execute zero-loss
   suffix chunks.
   The worker starts from the largest configured chunk. Before policy version zero it samples reusable memory once,
   then analytically accounts for persistent KV, activation/backward workspace, vocabulary logits/gradients, and
   reserve space. It first preserves the large chunk and reduces the power-of-two wavefront width as needed; only if
   that is insufficient does it reduce the chunk. The resulting plan is reused for every step, so there is no
   allocator query, OOM retry, or kernel-shape change on the training path.
6. All local raw token-loss sums are backwarded before global token normalization, gradient clipping, and the single
   optimizer step. Parameters remain at `theta_k` for the entire global cohort; `theta_(k+1)` is published to the
   standalone rollout pool only after the policy-version barrier.

The numbered safetensors handoff is a correctness-first incremental host transport. It proves that KV enters the
trainer-side cache during generation rather than being recovered after EOS. A production multi-node implementation
should preserve the same multi-MB host-cache/single-MB GPU-lease contract while replacing filesystem serialization
with CUDA IPC, RDMA, or an equivalent streaming transport.

Dedicated rollout pools publish each new policy version through the `host` checkpoint engine. Actor rank 0 writes
immutable buckets to a unique node-local `/dev/shm` session while the other FSDP ranks participate in parameter
gathers and bucket barriers. All rollout replicas read the same buckets concurrently, then use their existing
same-GPU CUDA IPC path to update vLLM. This avoids constructing a cross-pool NCCL communicator at the strict policy
barrier; session files are removed after every receiver finishes.

## Numerical behavior

The reverse chain rule, paged attention output, and parameter gradients are checked against ordinary full-sequence
training. OOMB's dense kernel is BF16-only, so comparisons use numerical tolerances rather than bitwise equality.
The runtime has no SDPA fallback: unsupported dtype, device, head dimension, page size, or reverse batch shape fails
closed before training proceeds.
