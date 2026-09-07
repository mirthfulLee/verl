# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Check reverse gradients and FSDP parameter gathers under torchrun."""

import copy
import json
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from tests.experimental.streamopd_kv.test_qwen3_reverse_on_gpu import _capture_host_kv, _CrossEntropy
from verl.experimental.streamopd_kv.qwen3 import Qwen3ReverseTrainer
from verl.experimental.streamopd_kv.reverse_attention import ReverseKVSlotPool


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(23)
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=256,
        )
        config._attn_implementation = "sdpa"
        initial = Qwen3ForCausalLM(config).cuda().eval()
        tokens = (torch.arange(1, 130, device="cuda") + rank * 13).remainder(256).unsqueeze(0)
        sequence = tokens[:, :-1]
        loss = _CrossEntropy(tokens[:, 1:], valid_from=8)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            baseline_loss, _ = loss(initial(input_ids=sequence, use_cache=False).logits, 0, sequence.shape[1])
        baseline_loss.backward()
        expected = {}
        for name, parameter in initial.named_parameters():
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(dist.get_world_size())
            expected[name] = parameter.grad.detach().clone()
        capture_model = copy.deepcopy(initial).to(torch.bfloat16)
        sources = [_capture_host_kv(capture_model, sequence)]
        del capture_model
        use_orig_params = os.environ.get("OPD_CHECK_USE_ORIG_PARAMS", "1") == "1"
        if not use_orig_params:
            reference = copy.deepcopy(initial)
            reference.zero_grad(set_to_none=True)
            reference = FSDP(
                reference,
                device_id=rank,
                use_orig_params=False,
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}),
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
                ),
            ).eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reference_loss, _ = loss(reference(input_ids=sequence, use_cache=False).logits, 0, sequence.shape[1])
            reference_loss.backward()
            expected = {name: parameter.grad.detach().clone() for name, parameter in reference.named_parameters()}
            del reference
        reports = []
        for strategy in (ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP):
            module = copy.deepcopy(initial)
            module.zero_grad(set_to_none=True)
            model = FSDP(
                module,
                device_id=rank,
                use_orig_params=use_orig_params,
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}),
                sharding_strategy=strategy,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
                ),
            )
            model.eval()
            pool = ReverseKVSlotPool(
                batch_size=1,
                token_capacity=128,
                num_layers=2,
                num_kv_heads=2,
                head_dim=32,
                page_size=16,
                dtype=torch.bfloat16,
                device=f"cuda:{rank}",
            )
            pool.prepare_next(sources, [128], [128])
            pool.activate_next()

            def accumulation(index, total, current_model=model):
                return current_model.no_sync() if index < total - 1 else nullcontext()

            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = Qwen3ReverseTrainer(model, chunk_size=32, page_size=16).backward(
                        [sequence],
                        [loss],
                        state=pool.state(),
                        backward_context=accumulation,
                        on_depth_committed=pool.release_current_range,
                    )
            pool.finish_current()
            with FSDP.summon_full_params(model, with_grads=True) if use_orig_params else nullcontext():
                squared_error = squared_norm = dot = actual_norm = 0.0
                parameters = module.named_parameters() if use_orig_params else model.named_parameters()
                for name, parameter in parameters:
                    actual = parameter.grad.float()
                    target = expected[name.replace("_fsdp_wrapped_module.", "") if use_orig_params else name]
                    squared_error += (actual - target).square().sum().item()
                    squared_norm += target.square().sum().item()
                    actual_norm += actual.square().sum().item()
                    dot += (actual * target).sum().item()
                values = torch.tensor(
                    [squared_error, squared_norm, dot, actual_norm], device="cuda", dtype=torch.float64
                )
                if not use_orig_params:
                    dist.all_reduce(values)
                squared_error, squared_norm, dot, actual_norm = values.tolist()
                relative_error = (squared_error / squared_norm) ** 0.5
                cosine = dot / (squared_norm * actual_norm) ** 0.5
            assert relative_error < 0.09 and cosine > 0.995, (strategy, relative_error, cosine)
            gathers = {event.key: event.count for event in profiler.key_averages() if "allgather" in event.key.lower()}
            reports.append(
                {
                    "strategy": strategy.name,
                    "use_orig_params": use_orig_params,
                    "gradient_relative_l2": relative_error,
                    "gradient_cosine": cosine,
                    "backward_calls": result.backward_calls,
                    "gathers": gathers,
                }
            )
            del model, module, pool
            torch.cuda.empty_cache()
        if rank == 0:
            print(json.dumps(reports, indent=2))
            if os.environ.get("OPD_CHECK_OUTPUT"):
                Path(os.environ["OPD_CHECK_OUTPUT"]).write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
