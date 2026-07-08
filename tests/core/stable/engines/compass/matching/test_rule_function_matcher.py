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

from dataclasses import replace

from pytest import raises

from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.rule_matcher_registry import RuleMatcherRegistry
from parlant.core.engines.compass.matching.rule_function_matcher import (
    RuleFunctionMatcher,
)
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.rules import Rule
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    RecordedEvent,
    RecordedSpan,
    RecordingTracer,
    create_engine_context,
    create_rule,
)


def _make_matcher(
    registry: RuleMatcherRegistry,
    tracer: LocalTracer | None = None,
) -> RuleFunctionMatcher:
    tracer = tracer or LocalTracer()
    return RuleFunctionMatcher(StdoutLogger(tracer), tracer, registry)


def _context() -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    return context


async def test_that_a_rule_with_a_matching_code_matcher_is_returned_as_matched() -> None:
    registry = RuleMatcherRegistry()
    rule = create_rule(condition="customer says hi", action="greet back")

    async def matcher(context: EngineContext, g: Rule) -> RuleMatch | None:
        return RuleMatch(rule=g, rationale="matched by code")

    registry.register(rule.id, matcher)

    matches = await _make_matcher(registry).match(_context(), [rule])

    assert [m.rule.id for m in matches] == [rule.id]


async def test_that_a_rule_whose_code_matcher_returns_none_is_not_matched() -> None:
    registry = RuleMatcherRegistry()
    rule = create_rule(condition="customer says hi", action="greet back")

    async def matcher(context: EngineContext, g: Rule) -> RuleMatch | None:
        return None

    registry.register(rule.id, matcher)

    matches = await _make_matcher(registry).match(_context(), [rule])

    assert list(matches) == []


async def test_that_an_error_in_a_code_matcher_is_handled_and_yields_no_match() -> None:
    registry = RuleMatcherRegistry()
    rule = create_rule(condition="customer says hi", action="greet back")

    async def matcher(context: EngineContext, g: Rule) -> RuleMatch | None:
        raise RuntimeError("boom")

    registry.register(rule.id, matcher)

    matches = await _make_matcher(registry).match(_context(), [rule])

    assert list(matches) == []


async def test_that_passing_a_rule_without_a_code_matcher_raises() -> None:
    registry = RuleMatcherRegistry()
    rule = create_rule(condition="customer says hi", action="greet back")

    # Not registered — the caller (engine) is responsible for passing only
    # function-attached rules, so this is a programming error.
    with raises(ValueError):
        await _make_matcher(registry).match(_context(), [rule])


async def test_that_function_matching_records_span_and_per_rule_result_events() -> None:
    registry = RuleMatcherRegistry()
    span_tracer = RecordingTracer()
    context_tracer = RecordingTracer()
    yes_rule = create_rule(
        condition="customer says hi",
        action="greet back",
        title="Greeting",
    )
    no_rule = create_rule(
        condition="customer asks for billing",
        action="start billing flow",
        title="Billing",
    )
    error_rule = create_rule(
        condition="customer asks for shipping",
        action="start shipping flow",
        title="Shipping",
    )
    context = replace(_context(), tracer=context_tracer)

    async def yes_matcher(context: EngineContext, g: Rule) -> RuleMatch | None:
        return RuleMatch(rule=g, rationale="matched by code")

    async def no_matcher(context: EngineContext, g: Rule) -> RuleMatch | None:
        return None

    async def error_matcher(context: EngineContext, g: Rule) -> RuleMatch | None:
        raise RuntimeError("do not expose me")

    registry.register(yes_rule.id, yes_matcher)
    registry.register(no_rule.id, no_matcher)
    registry.register(error_rule.id, error_matcher)

    await _make_matcher(registry, span_tracer).match(context, [yes_rule, no_rule, error_rule])

    assert span_tracer.started_spans == [RecordedSpan(name="match.rule.function", attributes={})]
    assert context_tracer.events == [
        RecordedEvent(
            name="matched.function.yes",
            attributes={
                "rule_id": str(yes_rule.id),
                "title": "Greeting",
            },
            span_id="<main>",
        ),
        RecordedEvent(
            name="matched.function.no",
            attributes={
                "rule_id": str(no_rule.id),
                "title": "Billing",
            },
            span_id="<main>",
        ),
        RecordedEvent(
            name="matched.function.no",
            attributes={
                "rule_id": str(error_rule.id),
                "title": "Shipping",
                "error_type": "RuntimeError",
            },
            span_id="<main>",
        ),
    ]
