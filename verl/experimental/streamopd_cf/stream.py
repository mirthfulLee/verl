# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU rendezvous between committed rollout tokens and bounded Trainer windows."""

import asyncio
import time

import torch

from verl.experimental.streamopd_kv.protocol import CommittedTokenChunk


class CFTrainingStream:
    def __init__(self, max_trajectory_tokens, timeout=600):
        self.max_tokens = max_trajectory_tokens
        self.timeout = timeout
        self.version = None
        self.records = {}
        self.changed = asyncio.Event()
        self.error = None
        self.completed = 0
        self.expected = 0

    def _notify(self):
        changed, self.changed = self.changed, asyncio.Event()
        changed.set()

    def _check(self, version):
        if version != self.version:
            raise RuntimeError(f"CF stream policy mismatch: expected {self.version}, got {version}")
        if self.error:
            raise RuntimeError(self.error)

    async def _wait(self, version, predicate):
        deadline = time.monotonic() + self.timeout
        while True:
            self._check(version)
            if predicate():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CF stream timed out for policy {version}")
            await asyncio.wait_for(self.changed.wait(), remaining)

    def begin(self, version, expected):
        if self.version is not None and self.completed != self.expected:
            raise RuntimeError("previous CF policy still has unfinished training")
        self.version, self.expected = version, expected
        self.records, self.order = {}, []
        self.packets = {}
        self.error, self.completed = None, 0
        self.started = time.perf_counter()
        self.first_forward = self.all_terminal = 0.0
        self.early_chunks = self.early_tokens = self.backward_count = 0
        self._notify()

    def abort(self, version, message):
        if version == self.version:
            self.error = str(message)
            self._notify()

    def submit(self, value):
        chunk = CommittedTokenChunk.from_dict(value)
        self._check(chunk.key.policy_version)
        key = chunk.key.trajectory_id
        if key not in self.records:
            if chunk.start != 0 or not chunk.prompt_ids or len(self.records) >= self.expected:
                raise RuntimeError("invalid initial CF trajectory chunk")
            self.records[key] = {
                "prompt": list(chunk.prompt_ids),
                "tokens": list(chunk.prompt_ids),
                "response_end": 0,
                "terminal": False,
                "ready": False,
                "targets": None,
                "done": False,
            }
            self.order.append(key)
        record = self.records[key]
        if record["terminal"] or chunk.start != record["response_end"]:
            raise RuntimeError("CF stream requires contiguous, non-duplicate committed tokens")
        if record["response_end"] and chunk.prompt_ids:
            raise RuntimeError("CF prompt may only appear in the initial chunk")
        if len(record["tokens"]) + len(chunk.token_ids) > self.max_tokens:
            raise RuntimeError("CF trajectory exceeds its planned maximum length")
        record["tokens"].extend(chunk.token_ids)
        record["response_end"] = chunk.end
        record["terminal"] = chunk.terminal
        if len(self.records) == self.expected and all(row["terminal"] for row in self.records.values()):
            self.all_terminal = time.perf_counter()
        self._notify()

    def finish(self, version, key, prompt_ids, response_ids, response_mask, teacher_ids, teacher_logprobs):
        self._check(version)
        record = self.records[key]
        tokens = list(prompt_ids) + list(response_ids)
        if not record["terminal"] or record["ready"] or tokens != record["tokens"]:
            raise RuntimeError("final CF trajectory differs from its committed prefix")
        if len(response_mask) != len(response_ids) or teacher_ids.shape != teacher_logprobs.shape:
            raise RuntimeError("invalid CF supervision shape")
        if teacher_ids.shape[0] != len(tokens):
            raise RuntimeError("CF Teacher supervision must cover the complete trajectory")
        record["targets"] = {
            "input_ids": torch.tensor(tokens),
            "prompts": torch.tensor(prompt_ids),
            "response_mask": torch.as_tensor(response_mask),
            "teacher_ids": teacher_ids.detach().cpu(),
            "teacher_logprobs": teacher_logprobs.detach().cpu(),
        }
        record["ready"] = True
        self._notify()

    async def group(self, version, index, width):
        end = min((index + 1) * width, self.expected)
        await self._wait(version, lambda: len(self.order) >= end)
        return self.order[index * width : end]

    async def next_chunk(self, version, keys, start, chunk_size, rank=0, world_size=1):
        self._check(version)
        if not 0 <= rank < world_size:
            raise ValueError("invalid CF consumer rank")
        rows = [self.records[key] for key in keys]
        target = start + chunk_size

        packet_key = (tuple(keys), start)

        def ready():
            if packet_key in self.packets:
                return True
            if any(not row["terminal"] and len(row["tokens"]) < target for row in rows):
                return False
            final = all(row["terminal"] for row in rows) and max(len(row["tokens"]) - 1 for row in rows) <= target
            return not final or all(row["ready"] for row in rows)

        await self._wait(version, ready)
        if packet_key not in self.packets:
            final = all(row["terminal"] for row in rows) and max(len(row["tokens"]) - 1 for row in rows) <= target
            # EOS may follow an already forwarded boundary. Its extra hidden
            # row is masked in the final targets, avoiding a whole emission
            # interval of lookahead latency for a single unknown next token.
            end = max(start, min(target, max(len(row["tokens"]) - 1 for row in rows))) if final else target
            # All DP consumers must see identical boundaries even if EOS
            # arrives between their RPCs. Keep plans, not Teacher tensors.
            self.packets[packet_key] = end, final
        end, final = self.packets[packet_key]
        blocks = []
        local_rows = rows[rank::world_size]
        for row in local_rows:
            tokens = row["tokens"][start:end]
            blocks.append(tokens + [0] * (end - start - len(tokens)))
        return {
            "tokens": blocks,
            "end": end,
            "final": final,
            "rollout_pending": sum(not row["terminal"] for row in rows),
            "samples": [row["targets"] for row in local_rows] if final else None,
        }

    def forward_completed(self, version, keys, token_count):
        self._check(version)
        now = time.perf_counter()
        if not self.first_forward:
            self.first_forward = now
        active = sum(not self.records[key]["terminal"] for key in keys)
        if active:
            self.early_chunks += 1
            self.early_tokens += token_count * active

    def backward_completed(self, version, keys):
        self._check(version)
        for key in keys:
            row = self.records[key]
            if not row["ready"] or row["done"]:
                raise RuntimeError("CF backward requires fresh complete supervision")
            row["done"] = True
            row["targets"] = None
            self.completed += 1
        self.backward_count += 1

    def snapshot(self, version):
        self._check(version)
        return {
            "streamopd_cf/stream/forward_chunks_before_eos": self.early_chunks,
            "streamopd_cf/stream/forward_tokens_before_eos": self.early_tokens,
            "streamopd_cf/stream/first_forward_seconds": self.first_forward - self.started if self.first_forward else 0,
            "streamopd_cf/stream/all_rollouts_terminal_seconds": self.all_terminal - self.started
            if self.all_terminal
            else 0,
            "streamopd_cf/stream/trained_trajectories": self.completed,
            "streamopd_cf/stream/backward_calls": self.backward_count,
        }
