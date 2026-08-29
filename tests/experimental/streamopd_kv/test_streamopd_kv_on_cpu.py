# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch import nn

from verl.experimental.streamopd_kv import (
    BatchedReverseChunkState,
    CommittedChunkPublisher,
    CommittedTokenChunk,
    KVLayout,
    KVSnapshotStore,
    LayerKVTrace,
    PolicyVersionBarrier,
    Qwen3ReverseTrainer,
    ReverseChunkState,
    SealedKVSnapshot,
    SnapshotState,
    StreamingTeacherCoordinator,
    TeacherArtifactBuffer,
    TrajectoryKey,
    capture_qwen3_kv_trace,
    exact_causal_attention,
    extract_vllm_nhd_tokens,
    load_vllm_snapshot,
    prepare_streamopd_kv_config,
)
from verl.experimental.streamopd_kv.fsdp_worker import (
    _forward_kl_topk_sum,
    _partition_reverse_microbatches,
    _reverse_backward_calls,
)
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.workers.config.distillation import DistillationConfig, StreamOPDKVConfig


class _ToyBlock(nn.Module):
    def __init__(self, hidden_size: int, query_heads: int, kv_heads: int) -> None:
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = hidden_size // query_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        layer_idx: int,
        reverse_state: ReverseChunkState | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, tokens, _ = hidden.shape
        normed = self.norm(hidden)
        query = self.q_proj(normed).view(batch, tokens, self.query_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(normed).view(batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(normed).view(batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)
        if reverse_state is None:
            attention = exact_causal_attention(query, key, value, query_start=0)
        else:
            attention = reverse_state.attention(layer_idx, query, key, value)
        attention = attention.transpose(1, 2).reshape(batch, tokens, -1)
        hidden = hidden + self.o_proj(attention)
        hidden = hidden + self.ff(self.ff_norm(hidden))
        return hidden, (key, value)


class _ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(31, 16)
        self.layers = nn.ModuleList([_ToyBlock(16, 4, 2), _ToyBlock(16, 4, 2)])
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 31, bias=False)

    def forward(
        self, input_ids: torch.Tensor, reverse_state: ReverseChunkState | None = None
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden = self.embedding(input_ids)
        cache = []
        for layer_idx, layer in enumerate(self.layers):
            hidden, kv = layer(hidden, layer_idx=layer_idx, reverse_state=reverse_state)
            cache.append(kv)
        return self.head(self.norm(hidden)), tuple(cache)


@pytest.mark.asyncio
async def test_committed_chunk_publisher_emits_only_accepted_contiguous_tokens() -> None:
    emitted = []

    async def submit(chunk) -> None:
        emitted.append(chunk)

    publisher = CommittedChunkPublisher(TrajectoryKey(4, "req"), [1, 2], chunk_size=2, submit=submit)
    await publisher.observe([10])
    await publisher.observe([10, 11, 12])
    await publisher.observe([10, 11, 12, 13], terminal=True)

    assert [(chunk.start, chunk.token_ids, chunk.terminal) for chunk in emitted] == [
        (0, (10, 11), False),
        (2, (12, 13), True),
    ]
    assert emitted[0].prompt_ids == (1, 2)
    assert emitted[1].prompt_ids == ()
    with pytest.raises(RuntimeError, match="retracted or replaced"):
        other = CommittedChunkPublisher(TrajectoryKey(4, "other"), [], 2, submit)
        await other.observe([8, 9])
        await other.observe([8, 7])


@pytest.mark.asyncio
async def test_teacher_streaming_scores_increasing_prefixes_and_closes_exact_chunk_boundary() -> None:
    calls = []

    async def score(sequence: list[int], request_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((list(sequence), request_id))
        ids = torch.tensor(sequence).unsqueeze(-1)
        return ids, -ids.float() - len(sequence)

    coordinator = StreamingTeacherCoordinator(score, max_pending_chunks=2)
    key = TrajectoryKey(9, "streamed")
    await coordinator.submit(
        CommittedTokenChunk(key=key, start=0, token_ids=(10, 11), prompt_ids=(1, 2), terminal=False)
    )
    await coordinator.submit(CommittedTokenChunk(key=key, start=2, token_ids=(12, 13), terminal=False))
    await coordinator.submit(CommittedTokenChunk(key=key, start=4, token_ids=(), terminal=True))
    ids, logprobs = await coordinator.result(key, required_completion_tokens=4)
    assert [call[0] for call in calls] == [[1, 2, 10, 11], [1, 2, 10, 11, 12, 13]]
    assert calls[0][1] == calls[1][1]
    torch.testing.assert_close(ids[:, 0], torch.tensor([1, 2, 10, 11, 12, 13]))
    torch.testing.assert_close(logprobs[:, 0], -ids[:, 0].float() - 6)


def test_snapshot_lifecycle_teacher_coverage_and_version_barrier() -> None:
    key = TrajectoryKey(3, "trajectory")
    layout = KVLayout(num_layers=1, num_kv_heads=2, head_dim=4, dtype="float32", page_size=16)
    snapshot = SealedKVSnapshot(
        key=key,
        token_ids=(1, 2, 3),
        prompt_length=2,
        layout=layout,
        layers=((torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),),
    )
    store = KVSnapshotStore()
    store.seal(snapshot)
    lease = store.acquire(key)
    assert lease.state is SnapshotState.LEASED
    with pytest.raises(RuntimeError, match="leased"):
        store.invalidate_version(3)
    lease.release()
    assert lease.state is SnapshotState.RELEASED
    assert store.invalidate_version(3) == 1

    artifacts = TeacherArtifactBuffer()
    artifacts.append(key, 0, torch.tensor([[1], [2]]), torch.tensor([[-1.0], [-2.0]]))
    artifacts.append(key, 2, torch.tensor([[3]]), torch.tensor([[-3.0]]), terminal=True)
    assert artifacts.is_complete(key, 3)
    ids, logprobs = artifacts.consume_reverse(key, 1, 3)
    torch.testing.assert_close(ids[:, 0], torch.tensor([2, 3]))
    torch.testing.assert_close(logprobs[:, 0], torch.tensor([-2.0, -3.0]))

    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([6.0])
    barrier = PolicyVersionBarrier(3, ["a", "b"], valid_token_count=3)
    barrier.mark_backward_complete(TrajectoryKey(3, "a"))
    with pytest.raises(RuntimeError, match="not ready"):
        barrier.step([parameter], lambda: None)
    barrier.mark_backward_complete(TrajectoryKey(3, "b"))
    barrier.step([parameter], lambda: parameter.data.add_(-parameter.grad))
    torch.testing.assert_close(parameter, torch.tensor([-1.0]))


def test_vllm_snapshot_loader_validates_identity_and_restores_layout(tmp_path) -> None:
    base_path = str(tmp_path / "sealed")
    filename = base_path + ".tp0.safetensors"
    packed = torch.arange(3 * 2 * 2 * 4, dtype=torch.float32).reshape(3, 2, 2, 4)
    save_file(
        {"token_ids": torch.tensor([4, 5, 6]), "layer_00000": packed},
        filename,
        metadata={
            "format": "verl-streamopd-kv-v1",
            "request_id": "request",
            "policy_version": "2",
            "prompt_length": "2",
            "tp_rank": "0",
            "tp_size": "1",
            "page_size": "16",
            "axis_order": "token_kv_head_dim",
            "rope_convention": "post_rope_key",
            "layer_names": '["model.layers.0.self_attn"]',
        },
    )
    (tmp_path / "sealed.tp0.safetensors.lock").touch()
    snapshot = load_vllm_snapshot(
        base_path,
        key=TrajectoryKey(2, "request"),
        tp_rank=0,
        expected_tp_size=1,
        expected_token_ids=[4, 5, 6],
        expected_prompt_length=2,
    )
    assert snapshot.layout.num_layers == 1
    assert snapshot.layout.num_kv_heads == 2
    assert snapshot.layers[0][0].shape == (1, 2, 3, 4)
    torch.testing.assert_close(snapshot.layers[0][0][0, :, 1], packed[1, 0])
    with pytest.raises(RuntimeError, match="token identity"):
        load_vllm_snapshot(
            base_path,
            key=TrajectoryKey(2, "request"),
            tp_rank=0,
            expected_tp_size=1,
            expected_token_ids=[4, 5, 7],
        )
    with pytest.raises(RuntimeError, match="request identity"):
        load_vllm_snapshot(
            base_path,
            key=TrajectoryKey(2, "other-request"),
            tp_rank=0,
            expected_tp_size=1,
        )


def test_vllm_nhd_page_extraction_preserves_logical_token_order() -> None:
    cache = torch.arange(3 * 2 * 4 * 2 * 3).reshape(3, 2, 4, 2, 3)
    extracted = extract_vllm_nhd_tokens(cache, block_ids=[2, 0], block_size=4, num_tokens=6)
    expected = torch.cat((cache[2].transpose(0, 1), cache[0].transpose(0, 1)), dim=0)[:6]
    torch.testing.assert_close(extracted, expected)

    legacy_cache = cache.permute(1, 0, 2, 3, 4).contiguous()
    legacy_extracted = extract_vllm_nhd_tokens(legacy_cache, block_ids=[2, 0], block_size=4, num_tokens=6)
    torch.testing.assert_close(legacy_extracted, expected)


def test_reverse_microbatch_partition_and_call_count() -> None:
    lengths = [9, 8, 12, 4]
    assert _partition_reverse_microbatches(lengths, max_batch_size=3, max_batch_tokens=20) == [
        [0, 1],
        [2, 3],
    ]
    assert _reverse_backward_calls([8, 13], chunk_size=5) == 3
    assert _reverse_backward_calls([8, 12], chunk_size=5) == 3


def test_prepare_config_installs_connector_and_rejects_stale_trainer() -> None:
    config = OmegaConf.create(
        {
            "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "engine_kwargs": {},
                },
                "actor": {
                    "strategy": "fsdp",
                    "ppo_epochs": 1,
                    "loss_agg_mode": "token-mean",
                    "use_torch_compile": False,
                },
            },
            "distillation": {
                "streamopd_kv": {"enabled": True, "kv_handoff_dir": "/tmp/test-streamopd"},
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk",
                    "use_policy_gradient": False,
                    "use_task_rewards": False,
                },
            },
        }
    )
    prepare_streamopd_kv_config(config)
    connector = config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config
    assert connector.kv_connector == "StreamOPDKVConnector"
    assert connector.kv_connector_extra_config.streamopd_kv_handoff_dir == "/tmp/test-streamopd"
    config.trainer.v1.trainer_mode = "separate_async"
    with pytest.raises(ValueError, match="trainer_mode=sync"):
        prepare_streamopd_kv_config(config)


@pytest.mark.parametrize("use_chunked_topk", [False, True])
def test_reverse_forward_kl_topk_matches_native_numerical_path(use_chunked_topk: bool) -> None:
    torch.manual_seed(31)
    logits = torch.randn(7, 29, dtype=torch.bfloat16, requires_grad=True)
    native_logits = logits.detach().clone().requires_grad_(True)
    teacher_ids = torch.randint(0, logits.shape[-1], (7, 5))
    teacher_logprobs = torch.log_softmax(torch.randn(7, 5), dim=-1)
    temperature = 0.73

    reverse_loss = _forward_kl_topk_sum(
        logits,
        teacher_ids,
        teacher_logprobs,
        temperature=temperature,
        use_chunked_topk=use_chunked_topk,
        log_prob_min_clamp=-10.0,
        loss_max_clamp=10.0,
    )
    scaled_native_logits = native_logits / torch.as_tensor(temperature, dtype=native_logits.dtype)
    if use_chunked_topk:
        native_student = _chunked_topk_log_probs(
            scaled_native_logits.unsqueeze(0), teacher_ids.unsqueeze(0), chunk_size=3
        ).squeeze(0)
    else:
        native_student = F.log_softmax(scaled_native_logits, dim=-1).gather(-1, teacher_ids)
    native_student = native_student.clamp_min(-10.0).float()
    native_teacher = teacher_logprobs.clamp_min(-10.0).float()
    native_loss = (native_teacher.exp() * (native_teacher - native_student)).sum(-1).clamp_min(0.0)
    native_loss = native_loss.clamp_max(10.0).sum()

    reverse_loss.backward()
    native_loss.backward()
    torch.testing.assert_close(reverse_loss, native_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(logits.grad, native_logits.grad, rtol=0.0, atol=0.0)


def test_distillation_config_coerces_streamopd_mapping_without_mutating_frozen_field() -> None:
    config = DistillationConfig(enabled=False, streamopd_kv={"enabled": False, "reverse_chunk_size": 32})
    assert isinstance(config.streamopd_kv, StreamOPDKVConfig)
    assert config.streamopd_kv.reverse_chunk_size == 32


def test_dapo_adapter_wraps_plain_prompt_as_chat_messages() -> None:
    from benchmarks.streamopd_kv.dapo_math_dataset import DAPOMathDataset

    dataset = DAPOMathDataset.__new__(DAPOMathDataset)
    assert dataset._build_messages({"prompt": "2 + 2?"}, "prompt") == [{"role": "user", "content": "2 + 2?"}]


def test_vllm_connector_waits_for_claimed_copy_event_and_ignores_unclaimed_request() -> None:
    from verl.experimental.streamopd_kv.vllm_connector import StreamOPDKVConnector

    class Event:
        complete = False

        def query(self) -> bool:
            return self.complete

    connector = StreamOPDKVConnector.__new__(StreamOPDKVConnector)
    connector._connector_metadata = None
    connector._finished_requests = set()
    connector._claimed_requests = {"claimed"}
    connector._copy_events = {}
    connector._lock_fds = {}
    assert connector.get_finished({"claimed", "ordinary"}) == (None, None)
    assert connector._finished_requests == {"claimed"}

    event = Event()
    connector._copy_events["claimed"] = event
    assert connector.get_finished(set()) == (None, None)
    event.complete = True
    assert connector.get_finished(set()) == ({"claimed"}, None)
    assert not connector._claimed_requests


@pytest.mark.parametrize(("sequence_length", "chunk_size"), [(5, 8), (7, 3), (9, 4)])
def test_reverse_chunk_gradients_match_full_sequence_opd(sequence_length: int, chunk_size: int) -> None:
    torch.manual_seed(7)
    baseline = _ToyCausalLM().eval()
    reverse = copy.deepcopy(baseline).eval()
    full_tokens = torch.tensor([[2, 5, 7, 11, 13, 17, 19, 23, 29, 3]])[:, : sequence_length + 1]
    input_ids, targets = full_tokens[:, :-1], full_tokens[:, 1:]
    loss_mask = torch.arange(sequence_length).unsqueeze(0) >= 2

    baseline_logits, _ = baseline(input_ids)
    baseline_loss = F.cross_entropy(baseline_logits[loss_mask], targets[loss_mask], reduction="sum")
    baseline_loss.backward()

    with torch.no_grad():
        _, rollout_cache = reverse(input_ids)
    state = ReverseChunkState([LayerKVTrace(key, value) for key, value in rollout_cache])
    reverse_loss = torch.zeros(())
    for end in range(input_ids.shape[1], 0, -chunk_size):
        start = max(0, end - chunk_size)
        state.begin(start, end)
        logits, _ = reverse(input_ids[:, start:end], reverse_state=state)
        local_mask = loss_mask[:, start:end]
        chunk_loss = (
            F.cross_entropy(logits[local_mask], targets[:, start:end][local_mask], reduction="sum")
            if local_mask.any()
            else logits.sum() * 0.0
        )
        (chunk_loss + state.gradient_injection()).backward()
        state.commit_prefix_gradients(release_processed_suffix=True)
        reverse_loss += chunk_loss.detach()

    torch.testing.assert_close(reverse_loss, baseline_loss.detach(), rtol=1e-6, atol=1e-6)
    for (baseline_name, baseline_parameter), (reverse_name, reverse_parameter) in zip(
        baseline.named_parameters(), reverse.named_parameters(), strict=True
    ):
        assert baseline_name == reverse_name
        assert baseline_parameter.grad is not None, baseline_name
        assert reverse_parameter.grad is not None, reverse_name
        torch.testing.assert_close(
            reverse_parameter.grad,
            baseline_parameter.grad,
            rtol=2e-4,
            atol=2e-5,
            msg=lambda message, name=baseline_name: f"{name}: {message}",
        )


def test_batched_ragged_reverse_gradients_match_full_sequence() -> None:
    torch.manual_seed(43)
    baseline = _ToyCausalLM().eval()
    reverse = copy.deepcopy(baseline).eval()
    sequences = [
        torch.tensor([[2, 5, 7, 11, 13, 17, 19]]),
        torch.tensor([[3, 7, 11, 17, 23, 29, 2, 5, 13]]),
    ]
    targets = [torch.roll(sequence, shifts=-1, dims=1) for sequence in sequences]
    valid_masks = [torch.arange(sequence.shape[1]).unsqueeze(0) >= 2 for sequence in sequences]

    baseline_loss = torch.zeros(())
    for sequence, target, valid in zip(sequences, targets, valid_masks, strict=True):
        logits, _ = baseline(sequence)
        baseline_loss = baseline_loss + F.cross_entropy(logits[valid], target[valid], reduction="sum")
    baseline_loss.backward()

    traces = []
    with torch.no_grad():
        for sequence in sequences:
            _, cache = reverse(sequence)
            traces.append([LayerKVTrace(key, value) for key, value in cache])
    state = BatchedReverseChunkState(traces)
    chunk_size = 3
    ends = [sequence.shape[1] for sequence in sequences]
    reverse_loss = torch.zeros(())
    while active := [idx for idx, end in enumerate(ends) if end >= chunk_size]:
        starts = [ends[idx] - chunk_size for idx in active]
        active_ends = [ends[idx] for idx in active]
        state.begin(active, starts, active_ends)
        chunk_ids = torch.cat(
            [sequences[idx][:, start:end] for idx, start, end in zip(active, starts, active_ends, strict=True)]
        )
        logits, _ = reverse(chunk_ids, reverse_state=state)
        chunk_loss = logits.sum() * 0.0
        for row, (idx, start, end) in enumerate(zip(active, starts, active_ends, strict=True)):
            mask = valid_masks[idx][:, start:end]
            if mask.any():
                chunk_loss = chunk_loss + F.cross_entropy(
                    logits[row : row + 1][mask], targets[idx][:, start:end][mask], reduction="sum"
                )
        (chunk_loss + state.gradient_injection()).backward()
        state.commit_prefix_gradients(release_processed_suffix=True)
        reverse_loss += chunk_loss.detach()
        for idx, start in zip(active, starts, strict=True):
            ends[idx] = start

    if active := [idx for idx, end in enumerate(ends) if end]:
        starts = [0] * len(active)
        active_ends = [ends[idx] for idx in active]
        state.begin(active, starts, active_ends)
        max_residual = max(active_ends)
        chunk_ids = torch.cat(
            [
                F.pad(sequences[idx][:, :end], (0, max_residual - end))
                for idx, end in zip(active, active_ends, strict=True)
            ]
        )
        logits, _ = reverse(chunk_ids, reverse_state=state)
        chunk_loss = logits.sum() * 0.0
        for row, (idx, end) in enumerate(zip(active, active_ends, strict=True)):
            mask = valid_masks[idx][:, :end]
            if mask.any():
                chunk_loss = chunk_loss + F.cross_entropy(
                    logits[row : row + 1, :end][mask], targets[idx][:, :end][mask], reduction="sum"
                )
        (chunk_loss + state.gradient_injection()).backward()
        state.commit_prefix_gradients(release_processed_suffix=True)
        reverse_loss += chunk_loss.detach()

    torch.testing.assert_close(reverse_loss, baseline_loss.detach(), rtol=1e-6, atol=1e-6)
    for (baseline_name, baseline_parameter), (reverse_name, reverse_parameter) in zip(
        baseline.named_parameters(), reverse.named_parameters(), strict=True
    ):
        assert baseline_name == reverse_name
        torch.testing.assert_close(
            reverse_parameter.grad,
            baseline_parameter.grad,
            rtol=3e-4,
            atol=3e-5,
            msg=lambda message, name=baseline_name: f"{name}: {message}",
        )


def test_qwen3_batched_reverse_trainer_matches_full_sequence() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(47)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    baseline = Qwen3ForCausalLM(config).eval()
    reverse = copy.deepcopy(baseline).eval()
    sequences = [torch.tensor([[1, 3, 5, 7, 9, 11, 13]]), torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16, 18]])]
    targets = [torch.roll(sequence, shifts=-1, dims=1) for sequence in sequences]
    valid_masks = [torch.arange(sequence.shape[1]) >= 2 for sequence in sequences]

    baseline_loss = torch.zeros(())
    for sequence, target, valid in zip(sequences, targets, valid_masks, strict=True):
        logits = baseline(input_ids=sequence, use_cache=False).logits
        baseline_loss = baseline_loss + F.cross_entropy(
            logits[:, valid].flatten(0, 1), target[:, valid].flatten(), reduction="sum"
        )
    baseline_loss.backward()

    traces = [capture_qwen3_kv_trace(reverse, sequence) for sequence in sequences]
    loss_fns = []
    for target, valid in zip(targets, valid_masks, strict=True):

        def loss_fn(logits: torch.Tensor, start: int, end: int, *, target=target, valid=valid):
            local_valid = valid[start:end]
            if not local_valid.any():
                return logits.sum() * 0.0, 0
            loss = F.cross_entropy(
                logits[:, local_valid].flatten(0, 1),
                target[:, start:end][:, local_valid].flatten(),
                reduction="sum",
            )
            return loss, int(local_valid.sum())

        loss_fns.append(loss_fn)

    result = Qwen3ReverseTrainer(reverse, chunk_size=3).backward_batched(sequences, traces, loss_fns)
    torch.testing.assert_close(result.loss_sum, baseline_loss.detach(), rtol=2e-5, atol=2e-5)
    assert result.chunks == 6
    assert result.backward_calls == 3
    for (baseline_name, baseline_parameter), (reverse_name, reverse_parameter) in zip(
        baseline.named_parameters(), reverse.named_parameters(), strict=True
    ):
        assert baseline_name == reverse_name
        torch.testing.assert_close(
            reverse_parameter.grad,
            baseline_parameter.grad,
            rtol=2e-4,
            atol=2e-5,
            msg=lambda message, name=baseline_name: f"{name}: {message}",
        )
