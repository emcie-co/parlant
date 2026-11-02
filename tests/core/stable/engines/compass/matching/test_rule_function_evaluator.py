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

from pytest import raises

from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.rule_evaluator_registry import RuleEvaluatorPhase, RuleEvaluatorRegistry
from parlant.core.engines.compass.matching.rule_function_evaluator import (
    RuleFunctionEvaluator,
)
from parlant.core.engines.compass.context import ResponseState
from parlant.core.rules import Rule
from parlant.core.loggers import StdoutLogger
from parlant.core.pricing import PricingConfigurationError
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    create_engine_context,
    create_rule,
)


def _make_evaluator(registry: RuleEvaluatorRegistry) -> RuleFunctionEvaluator:
    tracer = LocalTracer()
    return RuleFunctionEvaluator(StdoutLogger(tracer), tracer, registry)


def _context() -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    return context


async def test_that_a_code_evaluation_is_returned() -> None:
    registry = RuleEvaluatorRegistry()
    rule = create_rule(condition="customer says hi", action="greet back")

    async def evaluator(context: EngineContext, g: Rule) -> RuleEvaluation:
        return RuleEvaluation(rule=g, reasoning="relevant by code", is_relevant=True)

    registry.register(rule.id, evaluator)

    evaluations = await _make_evaluator(registry).evaluate(_context(), [rule])

    assert [evaluation.rule.id for evaluation in evaluations] == [rule.id]


async def test_that_an_irrelevant_code_evaluation_is_preserved() -> None:
    registry = RuleEvaluatorRegistry()
    rule = create_rule(condition="customer says hi", action="greet back")

    async def evaluator(context: EngineContext, g: Rule) -> RuleEvaluation:
        return RuleEvaluation(rule=g, reasoning="not relevant", is_relevant=False)

    registry.register(rule.id, evaluator)

    evaluations = await _make_evaluator(registry).evaluate(_context(), [rule])

    assert len(evaluations) == 1
    assert not evaluations[0].is_relevant


async def test_that_an_error_in_a_code_evaluator_is_handled_and_yields_no_evaluation() -> None:
    registry = RuleEvaluatorRegistry()
    rule = create_rule(condition="customer says hi", action="greet back")

    async def evaluator(context: EngineContext, g: Rule) -> RuleEvaluation:
        raise RuntimeError("boom")

    registry.register(rule.id, evaluator)

    evaluations = await _make_evaluator(registry).evaluate(_context(), [rule])

    assert list(evaluations) == []


async def test_that_a_pricing_configuration_error_from_a_code_evaluator_is_not_swallowed() -> None:
    registry = RuleEvaluatorRegistry()
    rule = create_rule(condition="budget exceeded", action="handoff")

    async def evaluator(context: EngineContext, g: Rule) -> RuleEvaluation:
        raise PricingConfigurationError("missing model price")

    registry.register(rule.id, evaluator, phase=RuleEvaluatorPhase.EARLY)

    with raises(PricingConfigurationError, match="missing model price"):
        await _make_evaluator(registry).evaluate(_context(), [rule])


async def test_that_passing_a_rule_without_a_code_evaluator_raises() -> None:
    registry = RuleEvaluatorRegistry()
    rule = create_rule(condition="customer says hi", action="greet back")

    # Not registered — the caller (engine) is responsible for passing only
    # function-attached rules, so this is a programming error.
    with raises(ValueError):
        await _make_evaluator(registry).evaluate(_context(), [rule])
