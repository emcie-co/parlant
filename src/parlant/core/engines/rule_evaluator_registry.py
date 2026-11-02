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

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum

from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.rules import Rule, RuleId

RuleCodeEvaluator = Callable[[EngineContext, Rule], Awaitable[RuleEvaluation]]


class RuleEvaluatorPhase(Enum):
    NORMAL = "normal"
    EARLY = "early"


@dataclass(frozen=True)
class RuleEvaluatorRegistration:
    evaluator: RuleCodeEvaluator
    phase: RuleEvaluatorPhase


class RuleEvaluatorRegistry:
    """Engine-agnostic registry of per-rule code evaluators."""

    def __init__(self) -> None:
        self._registrations: dict[RuleId, RuleEvaluatorRegistration] = {}

    def register(
        self,
        rule_id: RuleId,
        evaluator: RuleCodeEvaluator,
        *,
        phase: RuleEvaluatorPhase = RuleEvaluatorPhase.NORMAL,
    ) -> None:
        self._registrations[rule_id] = RuleEvaluatorRegistration(evaluator, phase)

    def get(self, rule_id: RuleId) -> RuleCodeEvaluator | None:
        registration = self._registrations.get(rule_id)
        return registration.evaluator if registration else None

    def phase(self, rule_id: RuleId) -> RuleEvaluatorPhase | None:
        registration = self._registrations.get(rule_id)
        return registration.phase if registration else None

    @property
    def evaluators(self) -> Mapping[RuleId, RuleCodeEvaluator]:
        return {
            rule_id: registration.evaluator for rule_id, registration in self._registrations.items()
        }
