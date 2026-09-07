# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Functional Qwen3 chunk state: one differentiable forward graph, one backward."""

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F


def chunk_attention(query, key, value, scale, backend):
    """Attend a suffix query to its entire prefix, preserving GQA gradients."""
    if backend == "flash_attention_2":
        from flash_attn import flash_attn_func

        return flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=scale, causal=True)
    if backend != "sdpa":
        raise ValueError(f"unsupported streamopd-cf attention backend: {backend}")
    # SDPA's is_causal uses top-left alignment when Q and KV lengths differ.
    # The reference needs bottom-right alignment for cached chunk prefixes.
    q_len, kv_len = query.shape[1], key.shape[1]
    allowed = torch.arange(kv_len, device=query.device)[None, :] <= (
        kv_len - q_len + torch.arange(q_len, device=query.device)[:, None]
    )
    result = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=allowed,
        dropout_p=0.0,
        scale=scale,
        enable_gqa=True,
    )
    return result.transpose(1, 2)


def _chunk_layer_forward(layer, hidden_states, *, position_embeddings, prefix_key=None, prefix_value=None, backend):
    """Return new KV as graph outputs; checkpoint replay never mutates a cache."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    attention = layer.self_attn
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
    query = attention.q_norm(attention.q_proj(hidden_states).view(shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden_states).view(shape)).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    query, key, value = (tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
    # Checkpoint inputs reference immutable per-chunk KV, not copied complete
    # prefixes. Concatenation is transient under activation checkpointing;
    # saved KV storage remains linear in trajectory length.
    all_key = key if not prefix_key else torch.cat((*prefix_key, key), dim=1)
    all_value = value if not prefix_value else torch.cat((*prefix_value, value), dim=1)
    output = chunk_attention(query, all_key, all_value, attention.scaling, backend)
    hidden_states = residual + attention.o_proj(output.reshape(*hidden_states.shape[:-1], -1))
    hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
    return hidden_states, key, value


@contextmanager
def use_qwen3_chunked_forward(model, chunk_size, *, backend="flash_attention_2", loss_function=None):
    """Keep patches active through backward, including activation checkpoint replay.

    Invoking the outer FSDP module once preserves its normal parameter/gradient
    lifecycle. Wrapped decoder layers are still invoked through their wrappers.
    Only right-padded text batches with dense attention and zero dropout are
    supported. The optional loss callback runs after all chunk forwards finish.
    """
    if chunk_size < 1:
        raise ValueError("streamopd-cf chunk_size must be positive")
    unwrapped = getattr(model, "module", model)
    if getattr(unwrapped.config, "model_type", None) != "qwen3":
        raise NotImplementedError("streamopd-cf currently supports Qwen3")
    originals = [(unwrapped, unwrapped.forward)]
    layers = unwrapped.model.layers
    for layer in layers:
        decoder = getattr(layer, "_fsdp_wrapped_module", layer)
        if getattr(decoder.self_attn, "sliding_window", None) is not None:
            raise NotImplementedError("streamopd-cf requires dense causal attention")
        if decoder.self_attn.attention_dropout:
            raise NotImplementedError("streamopd-cf requires zero attention dropout")

    def forward(module, input_ids=None, position_ids=None, chunk_iterator=None, **loss_inputs):
        if chunk_iterator is None:
            if input_ids is None:
                raise ValueError("CF forward requires input_ids or a committed chunk iterator")
            chunk_iterator = (
                input_ids[:, start : start + chunk_size] for start in range(0, input_ids.shape[1], chunk_size)
            )
        prefixes = [((), ()) for _ in layers]
        outputs = []
        start = 0
        for token_chunk in chunk_iterator:
            if not 0 < token_chunk.shape[1] <= chunk_size:
                raise ValueError("CF input chunks must be nonempty and bounded by forward_chunk_size")
            end = start + token_chunk.shape[1]
            positions = (
                torch.arange(start, end, device=token_chunk.device)[None, :]
                if position_ids is None
                else position_ids[:, start:end]
            )
            hidden = module.model.embed_tokens(token_chunk)
            embeddings = module.model.rotary_emb(hidden, positions)
            for index, layer in enumerate(layers):
                hidden, key, value = layer(
                    hidden,
                    position_embeddings=embeddings,
                    prefix_key=prefixes[index][0],
                    prefix_value=prefixes[index][1],
                    backend=backend,
                )
                prefixes[index] = prefixes[index][0] + (key,), prefixes[index][1] + (value,)
            outputs.append(module.model.norm(hidden))
            start = end
        hidden = torch.cat(outputs, dim=1)
        if loss_function is not None:
            return loss_function(hidden, module.lm_head.weight, **loss_inputs)
        return module.lm_head(hidden)

    try:
        unwrapped.forward = MethodType(forward, unwrapped)
        for layer in layers:
            decoder = getattr(layer, "_fsdp_wrapped_module", layer)
            originals.append((decoder, decoder.forward))
            decoder.forward = MethodType(_chunk_layer_forward, decoder)
        yield
    finally:
        for module, original in reversed(originals):
            module.forward = original
