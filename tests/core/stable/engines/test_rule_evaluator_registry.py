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

from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.rule_evaluator_registry import RuleEvaluatorPhase, RuleEvaluatorRegistry
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.rules import Rule, RuleId


async def _irrelevant_evaluator(context: EngineContext, rule: Rule) -> RuleEvaluation:
    return RuleEvaluation(rule=rule, reasoning="not relevant", is_relevant=False)


def test_that_a_registered_evaluator_can_be_retrieved_by_rule_id() -> None:
    registry = RuleEvaluatorRegistry()
    rule_id = RuleId("g-1")

    assert registry.get(rule_id) is None

    registry.register(rule_id, _irrelevant_evaluator)

    assert registry.get(rule_id) is _irrelevant_evaluator
    assert dict(registry.evaluators) == {rule_id: _irrelevant_evaluator}


def test_that_a_rule_without_a_registered_evaluator_returns_none() -> None:
    registry = RuleEvaluatorRegistry()
    registry.register(RuleId("g-1"), _irrelevant_evaluator)

    assert registry.get(RuleId("g-2")) is None


def test_that_evaluator_registration_has_an_explicit_execution_phase() -> None:
    registry = RuleEvaluatorRegistry()
    ordinary = RuleId("ordinary")
    early = RuleId("early")

    registry.register(ordinary, _irrelevant_evaluator)
    registry.register(early, _irrelevant_evaluator, phase=RuleEvaluatorPhase.EARLY)

    assert registry.phase(ordinary) == RuleEvaluatorPhase.NORMAL
    assert registry.phase(early) == RuleEvaluatorPhase.EARLY
