# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
import time
from contextlib import nullcontext

import ray
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name, get_torch_device

from .batching import AutoBatchActorWorker, AutoBatchTrainingWorker, prepare_padded_micro_batches
from .loss import linear_topk_kl
from .qwen3 import use_qwen3_chunked_forward


def sample_tensor(sample, key):
    value = sample[key]
    return value.values() if value.is_nested else value


def pack_training_group(samples, topk, device, min_length=1):
    """Right-pad inputs and align response supervision with next-token logits."""
    inputs, masks, ids, logprobs = [], [], [], []
    for sample in samples:
        tokens = sample_tensor(sample, "input_ids")
        prompt_length = sample_tensor(sample, "prompts").numel()
        response = sample_tensor(sample, "response_mask").bool()
        length = tokens.numel() - 1
        mask = torch.zeros(length, dtype=torch.bool)
        mask[prompt_length - 1 : prompt_length - 1 + response.numel()] = response.cpu()
        inputs.append(tokens[:-1].cpu())
        masks.append(mask)
        ids.append(sample_tensor(sample, "teacher_ids")[:length].cpu())
        logprobs.append(sample_tensor(sample, "teacher_logprobs")[:length].cpu())
    if not inputs:
        inputs, masks = [torch.zeros(min_length, dtype=torch.long)], [torch.zeros(min_length, dtype=torch.bool)]
        ids = [torch.zeros((min_length, topk), dtype=torch.long)]
        logprobs = [torch.zeros((min_length, topk))]
    packed = [pad_sequence(values, batch_first=True) for values in (inputs, masks, ids, logprobs)]
    if packed[0].shape[1] < min_length:
        extra = min_length - packed[0].shape[1]
        packed = [
            torch.cat((value, value.new_zeros((value.shape[0], extra, *value.shape[2:]))), dim=1) for value in packed
        ]
    return tuple(value.to(device, non_blocking=True) for value in packed)


def rescale_gradients(model, factor):
    """Apply the final policy-wide token denominator before clipping/stepping."""
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)


class StreamOPDCFTrainingWorker(AutoBatchTrainingWorker):
    """Bounded forward graphs, one backward per window, one policy update."""

    chunked_forward = True

    def __init__(self, config):
        super().__init__(config)
        self.settings = self.distillation.streamopd_cf

    def train_mini_batch(self, data):
        if int(tu.get_non_tensor_data(data, "epochs", 1)) != 1:
            raise ValueError("StreamOPD-CF requires one epoch per policy batch")
        dp_size = self.engine.get_data_parallel_size()
        group = self.engine.get_data_parallel_group()

        def prepare():
            metrics = self.configure_training_batch(data)
            batches, _ = prepare_padded_micro_batches(data, group)

            def groups():
                for batch in batches:
                    samples = list(batch.unbind(0))
                    length = max(sample_tensor(sample, "input_ids").numel() - 1 for sample in samples)
                    maximum = torch.tensor(length, device=get_device_id(), dtype=torch.long)
                    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
                    inputs, mask, ids, logprobs = pack_training_group(
                        samples, self.distillation.distillation_loss.topk, get_device_id(), int(maximum)
                    )
                    chunks = (
                        inputs[:, start : start + self.settings.forward_chunk_size]
                        for start in range(0, inputs.shape[1], self.settings.forward_chunk_size)
                    )
                    yield chunks, {"valid_mask": mask, "teacher_ids": ids, "teacher_logprobs": logprobs}, None

            return groups(), len(batches), metrics

        temperature = float(tu.get_non_tensor_data(data, "temperature", 1.0))
        return self._run_policy(prepare, len(data) * dp_size, temperature)

    def train_stream(self, version, expected, temperature, stream_name):
        stream = ray.get_actor(stream_name)
        dp_size, rank = self.engine.get_data_parallel_size(), self.engine.get_data_parallel_rank()

        def prepare():
            skeleton = TensorDict({}, batch_size=[expected // dp_size])
            metrics = self.configure_training_batch(skeleton)
            if self.engine_config.use_dynamic_bsz:
                width = int(tu.get_non_tensor_data(skeleton, "max_token_len_per_gpu", 0))
                width //= self.config.extra_context["max_trajectory_length"]
                if width < 1:
                    raise ValueError("streaming token budget must fit one maximum-length trajectory")
                width = min(width, expected // dp_size)
            else:
                width = self.engine_config.micro_batch_size_per_gpu
            metrics["batching/stream_window_size"] = width
            count = math.ceil(expected / (width * dp_size))

            def groups():
                for index in range(count):
                    before = time.perf_counter()
                    keys = ray.get(stream.group.remote(version, index, width * dp_size))
                    self._stream_wait_seconds += time.perf_counter() - before
                    local_keys = keys[rank::dp_size]
                    targets = {}

                    def chunks(keys=keys, local_keys=local_keys, targets=targets):
                        cursor = 0
                        while True:
                            before = time.perf_counter()
                            packet = ray.get(
                                stream.next_chunk.remote(
                                    version, keys, cursor, self.settings.forward_chunk_size, rank, dp_size
                                )
                            )
                            self._stream_wait_seconds += time.perf_counter() - before
                            if packet["final"]:
                                samples = packet["samples"]
                                inputs, mask, ids, logprobs = pack_training_group(
                                    samples,
                                    self.distillation.distillation_loss.topk,
                                    get_device_id(),
                                    packet["end"],
                                )
                                targets.update(valid_mask=mask, teacher_ids=ids, teacher_logprobs=logprobs)
                                token_chunk = inputs[:, cursor : packet["end"]]
                            else:
                                token_chunk = torch.tensor(packet["tokens"], dtype=torch.long, device=get_device_id())
                            if packet["end"] > cursor:
                                yield token_chunk
                                if cursor == 0 or packet["rollout_pending"]:
                                    # Confirm GPU completion before recording overlap with live rollout.
                                    get_torch_device().synchronize()
                                    ray.get(
                                        stream.forward_completed.remote(version, local_keys, packet["end"] - cursor)
                                    )
                            cursor = packet["end"]
                            if packet["final"]:
                                break

                    def complete(local_keys=local_keys):
                        ray.get(stream.backward_completed.remote(version, local_keys))

                    yield chunks(), targets, complete

            return groups(), count, metrics

        try:
            return self._run_policy(prepare, expected, temperature)
        except Exception as exc:
            ray.get(stream.abort.remote(version, str(exc)))
            raise

    def _run_policy(self, prepare, expected, temperature):
        model = self.engine.module
        group, dp_size = self.engine.get_data_parallel_group(), self.engine.get_data_parallel_size()
        dtype = getattr(self.engine, "_autocast_dtype", torch.bfloat16)
        if dtype == torch.float16:
            raise NotImplementedError("StreamOPD-CF supports BF16 or FP32 training")
        bound = expected * self.config.extra_context["max_response_length"]
        self._stream_wait_seconds = 0.0
        total_loss = torch.zeros((), device=get_device_id())
        local_tokens = 0
        current_targets = {}
        forward_events, loss_events, backward_events = [], [], []
        micro_sizes = []

        def stamp():
            event = get_torch_device().Event(enable_timing=True)
            event.record()
            return event

        def timed_chunks(chunks):
            for tokens in chunks:
                started = stamp()
                yield tokens
                forward_events.append((started, stamp()))

        def loss(hidden, weight):
            nonlocal local_tokens
            started = stamp()
            mask = current_targets["valid_mask"]
            local_tokens += int(mask.sum())
            if not mask.any():
                result = hidden.sum() * 0.0 + weight.sum() * 0.0
            else:
                config = self.distillation.distillation_loss
                result = linear_topk_kl(
                    hidden[mask],
                    weight,
                    current_targets["teacher_ids"][mask],
                    current_targets["teacher_logprobs"][mask],
                    chunk_size=self.settings.loss_chunk_size,
                    temperature=temperature,
                    min_logp=config.log_prob_min_clamp,
                    max_loss=config.loss_max_clamp,
                )
            loss_events.append((started, stamp()))
            return result

        started = time.perf_counter()
        clock_anchor = stamp()
        clock_anchor.synchronize()
        clock_time = time.perf_counter()
        with (
            self.engine.train_mode(disable_auto_offload=True),
            use_qwen3_chunked_forward(
                model, self.settings.forward_chunk_size, backend=self.settings.attention_backend, loss_function=loss
            ),
        ):
            groups, count, batching_metrics = prepare()
            self.engine.optimizer_zero_grad()
            for index, (chunks, current_targets, complete) in enumerate(groups):
                autocast = (
                    nullcontext()
                    if dtype == torch.float32
                    else torch.autocast(device_type=get_device_name(), dtype=dtype)
                )
                with self.engine._gradient_sync_context(is_last_micro_batch=index == count - 1), autocast:
                    loss_sum = model(chunk_iterator=timed_chunks(chunks))
                    normalized = loss_sum * (dp_size / bound)
                    backward_started = stamp()
                    normalized.backward()
                    backward_events.append((backward_started, stamp()))
                    total_loss += loss_sum.detach().float()
                micro_sizes.append(current_targets["valid_mask"].shape[0])
                if complete is not None:
                    complete()
                del loss_sum, normalized
            tokens = torch.tensor(local_tokens, dtype=torch.long, device=get_device_id())
            dist.all_reduce(tokens, group=group)
            valid_tokens = int(tokens)
            if valid_tokens <= 0:
                raise ValueError("CF policy has no supervised tokens")
            # Every backward uses one common provisional scale. FSDP has now
            # averaged all gradients; restore the exact global token mean.
            rescale_gradients(model, bound / valid_tokens)
            grad_norm = self.engine.optimizer_step()
            lr = self.engine.lr_scheduler_step()
        dist.all_reduce(total_loss, group=group)
        backward_events[-1][-1].synchronize()
        metrics = {
            **batching_metrics,
            "loss": total_loss.item() / valid_tokens,
            "grad_norm": grad_norm,
            "lr": lr,
            "mfu": 0.0,
            "streamopd_cf/training_seconds": time.perf_counter() - started,
            "streamopd_cf/stream_wait_seconds": self._stream_wait_seconds,
            "streamopd_cf/forward_chunks": len(forward_events),
            "streamopd_cf/backward_calls": len(backward_events),
            "streamopd_cf/optimizer_steps": 1,
            "streamopd_cf/valid_tokens": local_tokens,
            "streamopd_cf/micro_batch_size_min": min(micro_sizes),
            "streamopd_cf/micro_batch_size_max": max(micro_sizes),
            "perf/max_memory_allocated_gb": get_torch_device().max_memory_allocated() / 1024**3,
            "perf/max_memory_reserved_gb": get_torch_device().max_memory_reserved() / 1024**3,
        }
        timeline = {}
        for name, events in (("forward", forward_events), ("loss", loss_events), ("backward", backward_events)):
            metrics[f"streamopd_cf/{name}_gpu_seconds"] = sum(a.elapsed_time(b) for a, b in events) / 1000
            timeline[name] = [
                (clock_time + clock_anchor.elapsed_time(a) / 1000, clock_time + clock_anchor.elapsed_time(b) / 1000)
                for a, b in events
            ]
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics, "gpu_timeline": timeline}).cpu()


class StreamOPDCFActorWorker(AutoBatchActorWorker):
    actor_worker_cls = StreamOPDCFTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def train_stream(self, version, expected, temperature, stream_name):
        return self.actor.train_stream(version, expected, temperature, stream_name)
