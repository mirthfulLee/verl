# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import asyncio

import pytest
import torch

from verl.experimental.streamopd_cf.stream import CFTrainingStream
from verl.experimental.streamopd_cf.worker import rescale_gradients


def chunk(start, tokens, terminal=False, prompt=()):
    return {
        "policy_version": 0,
        "trajectory_id": "a",
        "start": start,
        "token_ids": tokens,
        "terminal": terminal,
        "prompt_ids": prompt,
    }


def test_forward_prefix_is_available_before_eos_but_final_chunk_waits_for_teacher():
    async def run():
        stream = CFTrainingStream(16, timeout=1)
        stream.begin(0, 1)
        stream.submit(chunk(0, [3, 4, 5, 6], prompt=[1, 2]))
        keys = await stream.group(0, 0, 1)
        prefix = await stream.next_chunk(0, keys, 0, 4)
        assert prefix["tokens"] == [[1, 2, 3, 4]]
        assert not prefix["final"] and prefix["rollout_pending"] == 1
        stream.forward_completed(0, keys, 4)
        final = asyncio.create_task(stream.next_chunk(0, keys, 4, 4))
        await asyncio.sleep(0)
        assert not final.done()
        stream.submit(chunk(4, [7], terminal=True))
        await asyncio.sleep(0)
        assert not final.done()
        stream.finish(0, "a", [1, 2], [3, 4, 5, 6, 7], [1] * 5, torch.zeros(7, 2), torch.zeros(7, 2))
        packet = await final
        assert packet["tokens"] == [[5, 6]] and packet["final"]
        stream.backward_completed(0, keys)
        stats = stream.snapshot(0)
        assert stats["streamopd_cf/stream/forward_chunks_before_eos"] == 1
        assert stats["streamopd_cf/stream/trained_trajectories"] == 1
        assert (
            stats["streamopd_cf/stream/first_forward_seconds"]
            < stats["streamopd_cf/stream/all_rollouts_terminal_seconds"]
        )

    asyncio.run(run())


def test_stream_rejects_stale_noncontiguous_and_changed_prefixes():
    stream = CFTrainingStream(16)
    stream.begin(0, 1)
    with pytest.raises(RuntimeError, match="policy mismatch"):
        stream.submit(chunk(0, [3], prompt=[1, 2]) | {"policy_version": 1})
    stream.submit(chunk(0, [3], prompt=[1, 2]))
    with pytest.raises(RuntimeError, match="contiguous"):
        stream.submit(chunk(2, [4]))
    with pytest.raises(RuntimeError, match="complete supervision"):
        stream.backward_completed(0, ["a"])
    stream.submit(chunk(1, [4], terminal=True))
    with pytest.raises(RuntimeError, match="committed prefix"):
        stream.finish(0, "a", [1, 2], [3, 9], [1, 1], torch.zeros(4, 2), torch.zeros(4, 2))


def test_abort_wakes_waiting_consumers():
    async def run():
        stream = CFTrainingStream(16, timeout=1)
        stream.begin(0, 1)
        waiting = asyncio.create_task(stream.group(0, 0, 1))
        await asyncio.sleep(0)
        stream.abort(0, "Teacher failed")
        with pytest.raises(RuntimeError, match="Teacher failed"):
            await waiting

    asyncio.run(run())


def test_ragged_window_pads_short_eos_and_releases_targets_before_next_policy():
    async def run():
        stream = CFTrainingStream(16, timeout=1)
        stream.begin(0, 2)
        stream.submit(chunk(0, [3], terminal=True, prompt=[1, 2]))
        stream.submit(chunk(0, [3, 4, 5], prompt=[1, 2]) | {"trajectory_id": "b"})
        keys = await stream.group(0, 0, 2)
        prefix = await stream.next_chunk(0, keys, 0, 4)
        assert prefix["tokens"] == [[1, 2, 3, 0], [1, 2, 3, 4]]
        stream.submit(chunk(3, [], terminal=True) | {"trajectory_id": "b"})
        final = asyncio.create_task(stream.next_chunk(0, keys, 4, 4))
        stream.finish(0, "a", [1, 2], [3], [1], torch.zeros(3, 2), torch.zeros(3, 2))
        await asyncio.sleep(0)
        assert not final.done()
        stream.finish(0, "b", [1, 2], [3, 4, 5], [1, 0, 1], torch.zeros(5, 2), torch.zeros(5, 2))
        packet = await final
        assert packet["final"] and packet["end"] == 4
        assert packet["tokens"] == [[], []]
        assert packet["samples"][1]["response_mask"].tolist() == [1, 0, 1]
        stream.backward_completed(0, ["a"])
        assert stream.records["a"]["targets"] is None
        with pytest.raises(RuntimeError, match="unfinished training"):
            stream.begin(1, 2)
        with pytest.raises(RuntimeError, match="fresh complete supervision"):
            stream.backward_completed(0, ["a"])
        stream.backward_completed(0, ["b"])
        stream.begin(1, 2)
        assert not stream.records and stream.completed == 0

    asyncio.run(run())


def test_window_membership_is_shared_and_waits_for_registered_trajectories():
    async def run():
        stream = CFTrainingStream(16, timeout=1)
        stream.begin(0, 4)
        consumers = [asyncio.create_task(stream.group(0, 0, 2)) for _ in range(2)]
        stream.submit(chunk(0, [3], prompt=[1, 2]))
        await asyncio.sleep(0)
        assert all(not consumer.done() for consumer in consumers)
        stream.submit(chunk(0, [3], prompt=[1, 2]) | {"trajectory_id": "b"})
        assert await asyncio.gather(*consumers) == [["a", "b"], ["a", "b"]]
        next_window = asyncio.create_task(stream.group(0, 1, 2))
        for key in ("c", "d"):
            stream.submit(chunk(0, [3], prompt=[1, 2]) | {"trajectory_id": key})
        assert await next_window == ["c", "d"]

    asyncio.run(run())


def test_late_token_normalization_matches_full_batch_gradient():
    torch.manual_seed(4)
    reference = torch.nn.Linear(3, 2)
    actual = torch.nn.Linear(3, 2)
    actual.load_state_dict(reference.state_dict())
    inputs = torch.randn(7, 3)
    reference(inputs).square().sum().div(7).backward()
    for group in inputs.split([2, 1, 4]):
        actual(group).square().sum().div(100).backward()
    rescale_gradients(actual, 100 / 7)
    for expected, observed in zip(reference.parameters(), actual.parameters(), strict=True):
        torch.testing.assert_close(expected.grad, observed.grad)


def test_exact_boundary_needs_no_lookahead_and_is_stable_across_rank_eos_race():
    async def run():
        stream = CFTrainingStream(16, timeout=1)
        stream.begin(0, 1)
        stream.submit(chunk(0, [3, 4], prompt=[1, 2]))
        first_rank = await stream.next_chunk(0, ["a"], 0, 4)
        assert first_rank["tokens"] == [[1, 2, 3, 4]]
        stream.submit(chunk(2, [], terminal=True))
        stream.finish(0, "a", [1, 2], [3, 4], [1, 1], torch.zeros(4, 2), torch.zeros(4, 2))
        second_rank = await stream.next_chunk(0, ["a"], 0, 4)
        assert second_rank["end"] == first_rank["end"] == 4
        assert not first_rank["final"] and not second_rank["final"]
        final = await stream.next_chunk(0, ["a"], 4, 4)
        assert final["final"] and final["tokens"] == [[]]
        from verl.experimental.streamopd_cf.worker import pack_training_group

        _, mask, _, _ = pack_training_group(final["samples"], 2, "cpu", min_length=final["end"])
        assert mask.tolist() == [[False, True, True, False]]

    asyncio.run(run())


def test_rank_packets_share_boundaries_but_only_transfer_local_targets():
    async def run():
        stream = CFTrainingStream(16, timeout=1)
        stream.begin(0, 2)
        for key, response in (("a", [3, 4]), ("b", [5, 6, 7])):
            stream.submit(chunk(0, response, terminal=True, prompt=[1, 2]) | {"trajectory_id": key})
            length = 2 + len(response)
            stream.finish(0, key, [1, 2], response, [1] * len(response), torch.zeros(length, 2), torch.zeros(length, 2))
        left = await stream.next_chunk(0, ["a", "b"], 0, 4, 0, 2)
        right = await stream.next_chunk(0, ["a", "b"], 0, 4, 1, 2)
        assert left["end"] == right["end"] == 4
        assert left["final"] and right["final"]
        assert len(left["samples"]) == len(right["samples"]) == 1
        assert left["samples"][0]["input_ids"].tolist() == [1, 2, 3, 4]
        assert right["samples"][0]["input_ids"].tolist() == [1, 2, 5, 6, 7]

    asyncio.run(run())
