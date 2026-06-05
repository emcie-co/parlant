# Copyright 2026 Emcie Co Ltd.
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

from collections.abc import Sequence
from dataclasses import dataclass

from parlant.core.engines.sigma.response_state import EngineContext
from parlant.core.guidelines import Guideline


@dataclass(frozen=True)
class DistilledGuideline:
    guideline: Guideline
    reasoning: str | None
    is_relevant: bool
    distilled_action: str
    score: float


@dataclass(frozen=True)
class GuidelineDistillationResult:
    distilled_guidelines: Sequence[DistilledGuideline]


class GuidelineDistiller:
    async def distill(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> GuidelineDistillationResult:
        return GuidelineDistillationResult([])
