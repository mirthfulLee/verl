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

import os

from verl.utils.dataset.rl_dataset import RLHFDataset


class DAPOMathDataset(RLHFDataset):
    """Adapt DAPO's plain-string prompt column to verl's chat-message contract."""

    def _build_messages(self, example: dict, key: str) -> list[dict]:
        prompt = example[key]
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return super()._build_messages(example, key)

    def __getitem__(self, item):
        row = super().__getitem__(item)
        ragged_caps = os.getenv("STREAMOPD_RAGGED_RESPONSE_LENGTHS", "")
        if ragged_caps:
            caps = [int(value) for value in ragged_caps.split(",") if value]
            if not caps or any(cap < 1 for cap in caps):
                raise ValueError("STREAMOPD_RAGGED_RESPONSE_LENGTHS must contain positive integers")
            # The training sampler shuffles dataset indices. Assigning caps by
            # item id therefore gives every policy batch a different cohort
            # shape and makes scheduler ablations noisy. data.gen_batch_size=1
            # and dataloader_num_workers=0 make this access order exactly the
            # rollout dispatch order used by the benchmark.
            sequence = getattr(self, "_streamopd_ragged_sequence", 0)
            row["max_response_tokens"] = caps[sequence % len(caps)]
            self._streamopd_ragged_sequence = sequence + 1
        return row
