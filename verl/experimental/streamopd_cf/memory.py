# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Conservative microbatch planning; no distributed trial-and-OOM loop."""

import math


def estimate_training_workspace(config, *, sequence_length, chunk_size, loss_chunk_size, dtype_bytes, checkpointing):
    """Return fixed and per-trajectory bytes, excluding parameters/optimizer.

    Estimates cover saved activations, differentiable KV and its gradients,
    transient attention/MLP workspaces, and the tiled linear-loss VJP. They are
    a sizing heuristic, not an allocator guarantee or an autotuned optimum.
    """
    length, chunk = sequence_length, min(sequence_length, chunk_size)
    hidden, intermediate = config.hidden_size, config.intermediate_size
    layers, vocab = config.num_hidden_layers, config.vocab_size
    head_dim = getattr(config, "head_dim", None) or hidden // config.num_attention_heads
    kv = config.num_key_value_heads * head_dim
    # LM-head gradients (FP32 accumulation plus cast/copy), and loss tile buffers.
    fixed = vocab * hidden * (4 + 2 * dtype_bytes) + loss_chunk_size * vocab * (4 + 2 * dtype_bytes)
    saved = length * (layers * (hidden + 4 * kv) + 8 * hidden)
    transient = chunk * (12 * hidden + 6 * intermediate) + length * 4 * kv
    if not checkpointing:
        saved += length * layers * (10 * hidden + 6 * intermediate)
        prefix_tokens = sum(min(end, length) for end in range(chunk, length + chunk, chunk))
        saved += 2 * layers * kv * prefix_tokens
    return fixed, math.ceil((saved + transient) * dtype_bytes * 1.2)


def choose_micro_batch_size(*, available_bytes, fixed_bytes, per_sample_bytes, batch_cap):
    """Choose a power of two bounded by memory and the local policy batch."""
    if per_sample_bytes <= 0 or batch_cap < 1:
        raise ValueError("microbatch planning requires positive workspace and batch capacity")
    capacity = min(batch_cap, (available_bytes - fixed_bytes) // per_sample_bytes)
    if capacity < 1:
        raise RuntimeError(
            "streamopd-cf auto microbatch estimate cannot fit one trajectory: "
            f"budget={available_bytes}, fixed={fixed_bytes}, per_sample={per_sample_bytes}; "
            "reduce sequence/chunk lengths or use an explicitly profiled ppo_micro_batch_size_per_gpu"
        )
    return 1 << (int(capacity).bit_length() - 1)
