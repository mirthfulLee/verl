# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Reuse vLLM's post-load CPU weight backup for immutable Teachers."""


class StaticTeacherWeightCache:
    """Keep level-1 weight backups across wake-ups without repeating D2H copies.

    Cache allocator storage, including transformed/quantized weights, rather
    than reconstructing a Hugging Face state dict. Native wake-up restores the
    same virtual addresses, so captured graphs retain their parameter pointers.
    This is only installed in a Teacher process whose weights never change.
    """

    def __init__(self, allocator):
        self.allocator = allocator
        self._sleep = allocator.sleep
        self._backups = None
        self.backup_bytes = 0
        self.backup_count = 0
        self.sleep_count = 0

    def sleep(self, offload_tags=None):
        """Back up weights once, then unmap them and reuse the saved CPU bytes."""
        if offload_tags != ("weights",):
            raise ValueError("Static Teacher weights require sleep(level=1)")
        weights = {ptr: data for ptr, data in self.allocator.pointer_to_data.items() if data.tag == "weights"}
        if not weights:
            raise RuntimeError("Teacher weights are not registered with vLLM's sleep allocator")
        if self._backups is None:
            self._sleep(offload_tags=("weights",))
            backups = {ptr: (data, data.cpu_backup_tensor) for ptr, data in weights.items()}
            if any(backup is None for _, backup in backups.values()):
                raise RuntimeError("vLLM did not create the level-1 Teacher weight backup")
            self._backups = backups
            self.backup_bytes = sum(t.numel() * t.element_size() for _, t in backups.values())
            self.backup_count += 1
        else:
            if weights.keys() != self._backups.keys() or any(
                data is not self._backups[ptr][0] for ptr, data in weights.items()
            ):
                raise RuntimeError("Static Teacher weight allocations changed after CPU backup")
            # Native wake_up clears cpu_backup_tensor after its H2D copy. Keep
            # our reference, then reattach it for the next native wake_up.
            for ptr, data in weights.items():
                data.cpu_backup_tensor = self._backups[ptr][1]
            self._sleep(offload_tags=())
        self.sleep_count += 1

    def stats(self):
        """Return copy counts for verifying repeated sleep/wake cycles."""
        return dict(backup_bytes=self.backup_bytes, backup_count=self.backup_count, sleep_count=self.sleep_count)
