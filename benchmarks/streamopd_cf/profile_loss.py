# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Profile the actual CF linear-loss forward/VJP at Qwen3 dimensions."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from verl.experimental.streamopd_cf.loss import linear_topk_kl
from verl.utils.device import get_torch_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--tiles", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    device = get_torch_device()
    torch.manual_seed(1)
    hidden = torch.randn(args.rows, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(config.vocab_size, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.randint(config.vocab_size, (args.rows, 32), device="cuda")
    teacher = torch.randn(args.rows, 32, device="cuda").log_softmax(-1)
    results = []
    for tile in args.tiles:
        times = []
        for iteration in range(4):
            hidden.grad = weight.grad = None
            device.reset_peak_memory_stats()
            start, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
            start.record()
            loss = linear_topk_kl(hidden, weight, ids, teacher, chunk_size=tile) / args.rows
            loss.backward()
            end.record()
            end.synchronize()
            if iteration:
                times.append(start.elapsed_time(end))
        results.append({"tile": tile, "milliseconds": times, "peak_gib": device.max_memory_allocated() / 1024**3})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": args.model, "rows": args.rows, "results": results}, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
