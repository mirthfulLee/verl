# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import copy
from contextlib import nullcontext
from functools import partial
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from verl.experimental.streamopd_cf.worker import StreamOPDCFTrainingWorker
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")


def _check_policy_update(rank, rendezvous, strategy):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(3)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
        )
        config._attn_implementation = "sdpa"
        reference = Qwen3ForCausalLM(config).cuda().train()
        model = copy.deepcopy(reference)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if strategy == "fsdp":
            model = FSDP(
                model,
                device_id=rank,
                use_orig_params=True,
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}),
            )
        else:
            from torch.distributed.fsdp import fully_shard

            for layer in model.model.layers:
                fully_shard(layer)
            fully_shard(model)
        tokens = torch.randint(0, 64, (4, 8), device="cuda")
        teacher, ids = torch.randn(4, 7, 64, device="cuda").log_softmax(-1).topk(4, dim=-1)
        mask = torch.zeros(4, 8, dtype=torch.bool, device="cuda")
        for index, count in enumerate((2, 3, 4, 5)):
            mask[index, 7 - count : 7] = True
        logits = reference(tokens[:, :-1], use_cache=False).logits
        selected = logits.log_softmax(-1).gather(-1, ids)
        loss = (teacher.exp() * (teacher - selected)).sum(-1).clamp_min(0)[mask[:, :-1]].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        torch.optim.SGD(reference.parameters(), lr=0.01).step()

        engine = SimpleNamespace(
            module=model,
            _autocast_dtype=torch.float32,
            _qat_enabled=False,
            engine_config=SimpleNamespace(use_no_sync_for_gradient_accumulation=True),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            optimizer_config=SimpleNamespace(clip_grad=1.0),
            get_data_parallel_group=lambda: dist.group.WORLD,
            get_data_parallel_size=lambda: 2,
            train_mode=lambda **kwargs: nullcontext(),
            lr_scheduler_step=lambda: 0.01,
        )
        for name in ("optimizer_step", "optimizer_zero_grad", "_gradient_sync_context"):
            setattr(engine, name, MethodType(getattr(FSDPEngine, name), engine))
        worker = StreamOPDCFTrainingWorker.__new__(StreamOPDCFTrainingWorker)
        worker.engine = engine
        worker.settings = SimpleNamespace(forward_chunk_size=4, loss_chunk_size=4, attention_backend="sdpa")
        worker.config = SimpleNamespace(extra_context={"max_response_length": 16})
        worker.distillation = SimpleNamespace(
            distillation_loss=SimpleNamespace(log_prob_min_clamp=None, loss_max_clamp=None)
        )
        completed = []

        def prepare():
            def groups():
                for index in (rank, rank + 2):
                    targets = {
                        "valid_mask": mask[index : index + 1],
                        "teacher_ids": torch.nn.functional.pad(ids[index : index + 1], (0, 0, 0, 1)),
                        "teacher_logprobs": torch.nn.functional.pad(teacher[index : index + 1], (0, 0, 0, 1)),
                    }
                    yield iter(tokens[index : index + 1].split(4, dim=1)), targets, lambda: completed.append(True)

            return groups(), 2, {}

        output = worker._run_policy(prepare, expected=4, temperature=1.0)
        metrics = tu.get_non_tensor_data(output, "metrics", {})
        assert metrics["streamopd_cf/optimizer_steps"] == 1 and len(completed) == 2
        assert metrics["loss"] == pytest.approx(loss.item(), rel=2e-5)
        timeline = tu.get_non_tensor_data(output, "gpu_timeline", {})
        assert all(end > start for intervals in timeline.values() for start, end in intervals)
        context = FSDP.summon_full_params(model) if strategy == "fsdp" else nullcontext()
        with context:
            for expected, actual in zip(reference.parameters(), model.parameters(), strict=True):
                actual = actual.full_tensor() if hasattr(actual, "full_tensor") else actual
                torch.testing.assert_close(expected, actual, rtol=2e-5, atol=2e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_streamed_policy_update_matches_global_token_mean_after_clipping(tmp_path, strategy):
    torch.multiprocessing.spawn(
        _check_policy_update, args=(str(tmp_path / "rendezvous"), strategy), nprocs=2, join=True
    )
