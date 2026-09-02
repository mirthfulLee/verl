# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from enum import Enum


class TrainerPlacement(str, Enum):
    TEACHER = "teacher"
    ROLLOUT = "rollout"
    UNION = "union"
    DEDICATED = "dedicated"

    @property
    def shares_teacher(self) -> bool:
        return self in {self.TEACHER, self.UNION}

    @property
    def shares_rollout(self) -> bool:
        return self in {self.ROLLOUT, self.UNION}

    @property
    def resource_sets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self is self.TEACHER:
            return ("teacher_trainer",), ("teacher_trainer",)
        if self is self.ROLLOUT:
            return ("teacher",), ("rollout_trainer",)
        if self is self.UNION:
            return ("teacher",), ("teacher", "rollout")
        return ("teacher",), ("trainer",)
