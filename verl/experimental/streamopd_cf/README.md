# StreamOPD-CF

`StreamOPD-CF` (chunked forward) is synchronous
on-policy distillation with independent Trainer,
Rollout and Teacher GPU pools. Committed rollout tokens feed both the existing
streaming Teacher service and Trainer. Trainer runs differentiable forward
chunks as those tokens arrive, concurrently with rollout and Teacher prefill.
The last input chunk waits for complete trajectories and Teacher supervision
within its microbatch; loss evaluation and one backward then finish that
microbatch. Microbatches accumulate into one optimizer update per policy batch.

This complements StreamOPD-KV's rollout-KV reverse training. The benchmark label
`streamopd-kv` selects the existing StreamOPD-KV union placement, where Trainer
borrows inference GPUs. StreamOPD-KV also supports dedicated reverse training.

## Training implementation

- `stream.py` holds committed tokens and final supervision in a CPU Ray actor.
  Trainer starts consuming before rollout is launched. A bounded window of
  trajectories holds differentiable graphs on the Trainer GPUs; all DP ranks
  use the same window membership and chunk boundaries. Once a window finishes
  backward, its targets are released and the next window starts. Consequently,
  later windows can start after rollout has already finished. The method does
  not retain graphs for the entire policy batch at once.
- Full committed chunks run immediately without waiting for a lookahead token.
  If the final token was already forwarded when EOS arrives, its extra hidden
  row is excluded from the loss. Short terminal trajectories are right-padded while other rows advance.
  The final packet waits for every trajectory's Teacher output in the window;
  when EOS lands at an already-forwarded boundary, it supplies targets without
  an additional forward. Failed producers abort waiting consumers.
- Chunk boundary decisions are shared across DP consumers, including an EOS
  arriving between their RPCs. Each consumer receives only its own token rows
  and Teacher targets, avoiding DP-wide duplication of supervision transfers.
- `qwen3.py` keeps ordinary autograd dependencies through all preceding chunks.
  Prefix KV is never detached or taken from rollout. Decoder layers are invoked
  through their existing FSDP/checkpoint wrappers; the outer model is called once
  per microbatch. FlashAttention handles grouped query attention and the
  bottom-right causal alignment of suffix queries against a complete prefix.
- Each checkpoint references immutable per-chunk KV tensors. Recomputing a
  decoder layer concatenates those references transiently, avoiding saved copies
  of every full prefix. Without activation checkpointing, attention backward
  can still retain concatenated prefixes; chunking alone is not a memory bound.
- All chunk forwards finish before the loss callback runs. It selects supervised
  hidden rows and evaluates the LM head/top-k forward KL in vocabulary-workspace
  bounded token tiles. Triton fuses normalization, clipping and logits gradients;
  matrix products use the normal tensor-core GEMMs. Like fused linear loss
  kernels, it precomputes the LM-head VJP while logits are resident. Transformer
  backward runs only after the loss is assembled, via one autograd call.
- Early backwards use a common provisional denominator based on the configured
  maximum response count. After all windows, Trainer reduces the actual valid
  token count and rescales accumulated gradients before clipping and stepping.
  This gives the global policy-batch token-mean gradient despite unknown final
  lengths during early forward/backward. The DP factor compensates for averaged
  gradient reduction. Unequal-length and zero-loss
  padding rows never contribute supervision; ranks use matching wrapped-layer
  call counts. The next-token shift includes the final prompt position and
  excludes the unsupervised final trajectory row.
- Existing Liger RMSNorm/MLP and non-reentrant activation checkpointing are reused.
  The implementation supports first-order BF16/FP32 training; higher derivatives
  and FP16 loss scaling are not supported.

The current implementation supports single-node, text-only Qwen3, one vLLM
Teacher, dense causal attention with zero dropout, full-parameter FSDP/FSDP2,
and direct top-k forward KL. Host checkpoint transfer synchronizes independent
Trainer and Rollout processes. LoRA, sequence parallelism, PPO objectives,
multiple rollout samples per prompt and inference/training GPU sharing are
rejected for this strategy.

## Configuration

Select `trainer.v1.trainer_mode=streamopd_cf` and
`distillation.streamopd_cf.enabled=true`; keep `distillation.streamopd_kv.enabled=false`.
Allocate all three pools explicitly:

| Role | Allocation |
| --- | --- |
| Trainer | `trainer.nnodes`, `trainer.n_gpus_per_node` |
| Rollout | `actor_rollout_ref.rollout.nnodes`, `actor_rollout_ref.rollout.n_gpus_per_node` |
| Teacher | `distillation.nnodes`, `distillation.n_gpus_per_node` |

The default example uses 2 Trainer + 1 Rollout + 1 Teacher GPUs. Its parameters
and optimizer stay on the Trainer GPUs, and model activation checkpointing is
on. Training does not export rollout KV or allocate StreamOPD-KV reverse slots.

| `distillation.streamopd_cf` option | Default | Meaning |
| --- | ---: | --- |
| `forward_chunk_size` | 1024 | Tokens in each ascending model forward chunk |
| `loss_chunk_size` | 2048 | Supervised rows per fused linear/KL workspace tile |
| `token_chunk_size` | 1024 | Committed input budget per Teacher stream fragment |
| `attention_backend` | `flash_attention_2` | Production FlashAttention; `sdpa` is a reference path |

Both CF and the full-trajectory baseline use verl's actor batching fields:

| Option | Meaning |
| --- | --- |
| `actor_rollout_ref.actor.use_dynamic_bsz` | Enable token-based dynamic microbatches |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | Positive manual token budget; 0 estimates it from memory |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | Fixed trajectories per GPU when dynamic batching is disabled |
| `distillation.batching.memory_fraction` | Fraction of available/reusable memory used by auto planning (0.8) |
| `distillation.batching.reserve_gib` | Additional reserve for auto planning (4 GiB) |

Automatic planning accounts for model dimensions, configured maximum trajectory
length, activation checkpointing, available Trainer memory, and future optimizer
and gradient storage. It resolves a common token budget from the most constrained
Trainer rank on each policy batch. The full-trajectory path budgets a full layer
workspace; CF budgets its chunk workspace and retained differentiable KV.
The full-trajectory baseline uses verl's `prepare_micro_batches` and
`rearrange_micro_batches` with actual sequence lengths. CF cannot know final
lengths before EOS: its streaming window size is the token budget divided by the
configured maximum trajectory length, capped at the per-rank policy batch.
Windows follow first-committed trajectory order with round-robin DP assignment.
The offline CF adapter also reuses native packing and bounds DP-aligned padding.
Global token-mean normalization and one optimizer update per policy batch are
unchanged. This is a conservative estimate, not an OOM guarantee or a
throughput autotuner. Explicit budgets/fixed sizes bypass the memory estimate.
For native baseline auto planning, keep `model.use_remove_padding=True` and
`distillation.distillation_loss.use_chunked_topk=True`, as in the example;
other native memory paths require an explicitly profiled budget or fixed size.

Rollout and Teacher each use vLLM continuous batching, independently of training.
Each pool has its own `max_num_seqs`, `max_num_batched_tokens` and
`gpu_memory_utilization`. Zero sequence/token limits in these two dedicated
strategies derive defaults from the policy batch, replica count and maximum
trajectory length; positive limits are preserved. vLLM profiles the available KV
memory on its own GPUs and schedules actual active requests within that capacity.
Teacher streaming admission also uses its profiled KV capacity. These inference
limits do not set the training microbatch or the number of optimizer updates.

Teacher admission limits are derived from the actual profiled vLLM capacity.
CF reserves each session's actual prompt length plus the configured maximum
response and one final vLLM token, rounded to KV pages. A global token budget
limits concurrent sessions, with any user trajectory cap also respected.
This avoids reserving the maximum prompt length for every short prompt.
Optional caps and tokenizer/artifact checks use the existing streaming service.
The controller launches Trainer before rollout, then waits for all window
backwards, Teacher publication and the single optimizer step before synchronizing
weights. GPU-completion metrics record forward chunks executed while their
trajectories are still generating; scheduler launch time alone is not evidence
of compute overlap. Blocking weight synchronization precedes the next policy
batch, preserving strict on-policy sampling.

`streamopd_cf/overlap/*` reports CUDA compute seconds overlapping rollout
lifetime and Teacher service intervals, averaged across Trainer ranks. Teacher
service intervals include queue/RPC time and are not GPU busy intervals.
Set `distillation.streamopd_cf.timeline_dir` to save per-step interval JSON;
`benchmarks/streamopd_cf/plot_timeline.py` renders it with optional GPU utilization
samples. Existing scheduler training-busy metrics include input waits and must
not be interpreted as Trainer GPU utilization.

Run `examples/on_policy_distillation_trainer/run_qwen3_streamopd_cf_fsdp.sh` with
`DATASET` set to a DAPO-Math parquet file and optional `STUDENT_MODEL` and
`TEACHER_MODEL` paths or HF IDs. Existing example GPU/environment variables are
reused. `TRAIN_MICRO_BATCH_SIZE=0` (default) uses dynamic training and
`TRAIN_MAX_TOKENS_PER_GPU=0` (default) estimates its budget. A positive training
microbatch selects fixed batching; a positive training token budget selects a
manual dynamic budget. Use `ROLLOUT_MAX_NUM_SEQS`, `ROLLOUT_MAX_BATCHED_TOKENS`,
`TEACHER_MAX_NUM_SEQS`, and `TEACHER_MAX_BATCHED_TOKENS` independently (0 derives
defaults). Each inference role also has its own `*_GPU_MEMORY_UTILIZATION`.
`FORWARD_CHUNK_SIZE`, `LOSS_CHUNK_SIZE` and `TOKEN_CHUNK_SIZE` control computation
tiles/stream fragments, not microbatch size. The rollout `log_prob_*` fields
refer to optional actor logprob recomputation, not generation batching.

## Matched full-trajectory baseline

`trainer.v1.trainer_mode=separate_sync` selects the strict synchronous control
with the same independent GPU pools. Both streaming configurations stay disabled.
Teacher receives whole trajectories through verl's ordinary Teacher client, and
Trainer uses the ordinary FSDP training worker and full-trajectory model forward.
The synchronous replay buffer admits one complete policy batch before each
update; blocking weight synchronization finishes before the next batch is
submitted. Both minimum and maximum behavior-policy versions must equal the
current update's source version. This mode omits PPO-only preparation for the
same direct distillation objective. It does not use `separate_async`.

Use `CASE=verl-sync-opd-separate` in the benchmark wrapper and match
`FIXED_MICRO_BATCH_SIZE` with StreamOPD-CF to control for GPU placement and
microbatch size. Loss evaluation may still use native vocabulary-workspace
bounded tiles; "full trajectory" refers to the Transformer forward/backward.

## Validation

CPU tests compare full-sequence versus chunked logits and all parameter gradients,
with and without activation checkpointing; they also cover loss normalization,
clipping, target alignment, topology validation, streamed forward ordering,
final Teacher gating and producer failures.
GPU tests compare FlashAttention and the Triton loss/VJP against ordinary
PyTorch computation, including non-unit temperature and duplicate top-k IDs.

```bash
CUDA_VISIBLE_DEVICES='' uv run --active --no-sync pytest -q tests/experimental/streamopd_cf
CUDA_VISIBLE_DEVICES=0 uv run --active --no-sync pytest -q tests/experimental/streamopd_cf/test_chunked_forward_on_gpu.py
```

Performance scripts are in `benchmarks/streamopd_cf`; see the
[experiment instructions](../../../benchmarks/STREAMOPD_EXPERIMENTS.md) for controls and commands.
