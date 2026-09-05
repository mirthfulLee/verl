# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from benchmarks.streamopd_kv.summarize_colocate_matrix import parse_steps, summarize_step_metrics


def test_summary_excludes_warmup_and_reports_variation(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("step:1 - timing_s/step:100.0\nstep:2 - timing_s/step:10.0\nstep:3 - timing_s/step:14.0\n")
    summary = summarize_step_metrics(parse_steps(log))
    assert summary["stable_step"]["timing_s/step"] == 12.0
    assert summary["measured_steps"] == 2
    assert summary["step_time_stddev"] == pytest.approx(8**0.5)


def test_warmup_alone_is_not_a_performance_measurement():
    summary = summarize_step_metrics([{"step": 1, "timing_s/step": 100.0}])
    assert summary["stable_step"] == {}
    assert summary["measured_steps"] == 0
    assert summary["step_time_stddev"] is None
