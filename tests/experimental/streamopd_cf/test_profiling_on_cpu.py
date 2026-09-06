# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

from verl.experimental.streamopd_cf.profiling import summarize_overlap


def test_cuda_overlap_excludes_input_waits_and_averages_ranks():
    service = {"started": 100, "rollout_end": 110, "teacher_end": 112, "teacher": [(104, 108), (106, 112)]}
    timelines = [
        {"forward": [(102, 103), (105, 106)], "loss": [(111, 112)], "backward": [(112, 116)]},
        {"forward": [(102, 104), (105, 106)], "loss": [(111, 112)], "backward": [(112, 115)]},
    ]
    metrics = summarize_overlap(timelines, service)
    prefix = "streamopd_cf/overlap/training"
    assert metrics[f"{prefix}_gpu_seconds"] == 7
    assert metrics[f"{prefix}_during_rollout_seconds"] == 2.5
    assert metrics[f"{prefix}_during_teacher_service_seconds"] == 2
    assert metrics[f"{prefix}_during_both_seconds"] == 1
    assert metrics[f"{prefix}_after_teacher_seconds"] == 3.5
    assert metrics[f"{prefix}_rollout_fraction"] == pytest.approx(2.5 / 7)
    assert metrics[f"{prefix}_triple_fraction"] == pytest.approx(1 / 7)
