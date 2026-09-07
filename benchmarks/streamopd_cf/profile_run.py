# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sample physical GPU utilization while running a benchmark command."""

import argparse
import csv
import io
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("a command and positive sampling interval are required")
    if args.require_idle:
        result = subprocess.run(
            ["nvidia-smi", f"--id={args.devices}", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        occupied = [row for row in csv.reader(io.StringIO(result.stdout)) if float(row[1]) > 1024]
        if occupied:
            parser.error(f"benchmark GPUs are occupied (index, used MiB): {occupied}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["monotonic_seconds", "gpu_index", "gpu_utilization_percent", "memory_used_mib"])
        process = subprocess.Popen(command, start_new_session=True)
        warned = False
        try:
            while process.poll() is None:
                started = time.perf_counter()
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--id={args.devices}",
                            "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    timestamp = (started + time.perf_counter()) / 2
                    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
                        index, utilization, memory = (float(value) for value in row)
                        writer.writerow([timestamp, int(index), utilization, memory])
                    output.flush()
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    if not warned:
                        print(f"GPU sampling unavailable: {exc}", file=sys.stderr)
                        warned = True
                try:
                    process.wait(timeout=max(0.001, args.interval - (time.perf_counter() - started)))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # A failed driver can exit while spawned vLLM children remain.
            # The process group is private to this benchmark invocation.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
