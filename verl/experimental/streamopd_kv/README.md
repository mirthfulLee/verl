# StreamOPD-KV

This package implements an experimental, two-pool strict on-policy distillation path for verl V1 sync training.
It streams committed rollout tokens to the teacher, seals completed vLLM KV pages, and uses those pages as the
no-grad trace for dense reverse-chunk Qwen3 training.

The initial backend deliberately fails closed outside this envelope:

- `trainer.use_v1=true` and `trainer.v1.trainer_mode=sync`;
- text-only, single-turn Qwen3 with one teacher;
- vLLM rollout with TP=1, PP=1, and non-quantized KV;
- FSDP/FSDP2 actor, one PPO epoch, and direct `forward_kl_topk` distillation;
- global `token-mean` loss aggregation;
- `actor.use_torch_compile=false`, because reverse traversal replaces Qwen3 attention dynamically;
- no task-reward or policy-gradient term;
- exact dense attention, with no sparse/page-selection approximation.

## Execution path

1. `CommittedChunkPublisher` observes cumulative vLLM outputs and publishes only accepted token prefixes.
2. `StreamingTeacherCoordinator` scores increasing full prefixes on a sticky teacher replica. The terminal full
   result is retained so prompt-logprob rows at chunk boundaries have the same next-token shift as post-hoc scoring.
3. `StreamOPDKVConnector` claims completed request blocks before vLLM frees them, copies the logical NHD token
   sequence to pinned CPU memory, and writes one locked safetensors shard per TP rank.
4. `StreamOPDKVTrainingWorker` validates policy version, trajectory tokens, TP layout, RoPE convention, and dtype.
   It then traverses suffix to prefix, accumulating attention dK/dV and injecting those VJPs into earlier chunk
   recomputation.
5. All local raw token-loss sums are backwarded before global token normalization, gradient clipping, and the single
   optimizer step. The V1 sync barrier publishes new rollout weights only after that step.

The safetensors handoff is a correctness-first boundary. It is not the final low-latency transport; same-process page
ownership or CUDA IPC should replace it before using large cohorts with 4K-8K trajectories.

## Numerical behavior

The reverse chain rule and parameter gradients are tested against ordinary full-sequence training. In FP32, a real
Qwen3 4B/14B two-chunk DAPO run has a loss-sum error of `1.1e-4`. In BF16, full-sequence and lower-right suffix SDPA
can choose different kernels; top-k forward KL can amplify their logits differences. Treat BF16 multi-chunk runs as
numerically equivalent within a measured tolerance, not bitwise identical. Use FP32 for the strictest correctness
audit and always report loss and parameter-gradient deltas.
