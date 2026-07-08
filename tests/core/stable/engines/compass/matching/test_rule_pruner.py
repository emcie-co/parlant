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

import asyncio
from typing import Any, Mapping, cast

from lagom import Container
from pytest import fixture

from parlant.core.engines.compass.matching.rule_ranker import RuleRankSchema
from parlant.core.engines.compass.matching.rule_pruner import (
    RulePruner,
)
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.rules import Rule
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.generation import SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    RecordedSpan,
    RecordingTracer,
    create_engine_context,
    create_rule,
)


@fixture
def pruner(container: Container) -> RulePruner:
    return container[RulePruner]


def _make_pruner(
    generator: Any = None,
    tracer: LocalTracer | None = None,
) -> RulePruner:
    # The prompt-building helpers only read the context, so tests that don't
    # generate can pass a None generator.
    tracer = tracer or LocalTracer()
    return RulePruner(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        schematic_generator=cast(Any, generator),
    )


def _context(conversation: list[tuple[EventSource, str]]) -> EngineContext:
    context = create_engine_context(conversation=conversation)
    context.state = ResponseState()
    return context


def test_that_a_rule_pruner_can_be_created(
    pruner: RulePruner,
) -> None:
    assert pruner is not None


# --- Prompt construction & caching -------------------------------------------
#
# Unlike the turn evaluators, the pruner runs at end of turn (finalize), when
# the turn is complete: the FULL interaction history belongs to the shared,
# cacheable prefix, and the only per-call tail is the rule itself.


def _cached_prefix(pruner: RulePruner, context: EngineContext, rule: Rule) -> str:
    prompt = pruner._build_prompt(context, rule, shots=[]).build()
    breakpoint_marker = pruner._cache_breakpoint(context)
    index = prompt.find(breakpoint_marker)
    assert index != -1, f"cache breakpoint {breakpoint_marker!r} not found in prompt"
    return prompt[:index]


def test_that_the_pruner_prompt_places_the_rule_after_the_cache_breakpoint() -> None:
    pruner = _make_pruner()
    rule = create_rule(
        condition="the customer asks about toppings ZZQ_RULE",
        action="list the available toppings",
    )
    context = _context([(EventSource.CUSTOMER, "what toppings do you have?")])

    prompt = pruner._build_prompt(context, rule, shots=[]).build()
    prefix = _cached_prefix(pruner, context, rule)

    assert "ZZQ_RULE" in prompt
    assert "ZZQ_RULE" not in prefix


def test_that_the_pruner_prompt_includes_the_full_interaction_history_in_the_cached_prefix() -> (
    None
):
    # At finalize time the turn is complete, so even the latest customer message
    # is part of the stable shared prefix (contrast with the ranker, which splits
    # it into a per-turn tail).
    pruner = _make_pruner()
    rule = create_rule(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    context = _context(
        [
            (EventSource.CUSTOMER, "hi"),
            (EventSource.AI_AGENT, "hello, how can I help?"),
            (EventSource.CUSTOMER, "ZZQ_LATEST what toppings do you have?"),
        ]
    )

    prefix = _cached_prefix(pruner, context, rule)

    assert "ZZQ_LATEST" in prefix


def test_that_the_pruner_cached_prefix_is_identical_across_the_per_rule_fan_out() -> None:
    pruner = _make_pruner()
    rule_a = create_rule(condition="condition a", action="action a")
    rule_b = create_rule(condition="condition b", action="action b")
    context = _context([(EventSource.CUSTOMER, "hello")])

    assert _cached_prefix(pruner, context, rule_a) == _cached_prefix(pruner, context, rule_b)


def test_that_the_pruner_cache_key_is_namespaced_per_session() -> None:
    pruner = _make_pruner()
    context = _context([(EventSource.CUSTOMER, "hello")])

    key = pruner._cache_key(context)

    assert str(context.session.id) in key
    assert "pruner" in key


# --- Generation behavior ------------------------------------------------------


def _generation_info() -> GenerationInfo:
    return GenerationInfo(
        schema_name="RuleRankSchema",
        model="fake-model",
        duration=0.01,
        usage=UsageInfo(input_tokens=1, output_tokens=1),
    )


class _FailingGenerator:
    async def generate(
        self,
        prompt: Any,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[RuleRankSchema]:
        raise RuntimeError("boom")


async def test_that_a_failed_pruning_call_defaults_to_keeping_the_rule() -> None:
    # Losing a session rule to a transient generation error is worse than
    # keeping a stale one for another round.
    pruner = _make_pruner(_FailingGenerator())
    rule = create_rule(condition="condition", action="action")
    context = _context([(EventSource.CUSTOMER, "hello")])

    result = await pruner.prune(context, [rule])

    assert len(result.pruned_rules) == 1
    pruned = result.pruned_rules[0]
    assert pruned.rule == rule
    assert pruned.is_still_relevant is True
    assert pruned.score == 1.0


class _RecordingGenerator:
    """Records the concurrency shape of generate calls: the first call must
    complete before the remaining calls start (warm-first-then-fan-out)."""

    def __init__(self) -> None:
        self.started: list[int] = []
        self.finished: list[int] = []
        self._count = 0

    async def generate(
        self,
        prompt: Any,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[RuleRankSchema]:
        self._count += 1
        call_index = self._count
        self.started.append(call_index)
        await asyncio.sleep(0)
        self.finished.append(call_index)
        return SchematicGenerationResult(
            content=RuleRankSchema(tldr="ok", s=4),
            info=_generation_info(),
        )


async def test_that_pruning_completes_the_first_call_before_fanning_out_the_rest() -> None:
    # The first call warms the provider's prefix cache so the fan-out sends only
    # live suffixes; it must finish before any other call starts.
    generator = _RecordingGenerator()
    pruner = _make_pruner(generator)
    rules = [create_rule(condition=f"condition {i}", action="action") for i in range(3)]
    context = _context([(EventSource.CUSTOMER, "hello")])

    result = await pruner.prune(context, rules)

    assert len(result.pruned_rules) == 3
    assert generator.finished[0] == 1, "the warm call must complete first"
    assert generator.started.index(2) > generator.finished.index(1), (
        "the fan-out must not start before the warm call finished"
    )


async def test_that_pruner_records_span() -> None:
    tracer = RecordingTracer()
    pruner = _make_pruner(_RecordingGenerator(), tracer=tracer)
    rules = [create_rule(condition=f"condition {i}", action="action") for i in range(2)]
    context = _context([(EventSource.CUSTOMER, "hello")])

    await pruner.prune(context, rules)

    assert tracer.started_spans == [RecordedSpan(name="match.rule.prune", attributes={})]


async def test_that_pruning_scores_map_to_still_relevant_above_the_threshold() -> None:
    class _ScoredGenerator:
        def __init__(self) -> None:
            self._scores = iter([1, 3, 5])

        async def generate(
            self,
            prompt: Any,
            hints: Mapping[str, Any] = {},
        ) -> SchematicGenerationResult[RuleRankSchema]:
            return SchematicGenerationResult(
                content=RuleRankSchema(tldr="scored", s=next(self._scores)),
                info=_generation_info(),
            )

    pruner = _make_pruner(_ScoredGenerator())
    rules = [create_rule(condition=f"condition {i}", action="action") for i in range(3)]
    context = _context([(EventSource.CUSTOMER, "hello")])

    result = await pruner.prune(context, rules)

    by_score = sorted(result.pruned_rules, key=lambda c: c.score)
    assert [c.is_still_relevant for c in by_score] == [False, True, True]
    assert [c.score for c in by_score] == [1 / 5, 3 / 5, 5 / 5]


# --- LLM-behavioral -----------------------------------------------------------


async def test_that_a_rule_for_a_concluded_topic_is_judged_no_longer_relevant(
    pruner: RulePruner,
) -> None:
    rule = create_rule(
        condition="the customer wants to reset their password",
        action="walk them through the password reset flow",
    )
    context = _context(
        [
            (EventSource.CUSTOMER, "Hi, I need to reset my password."),
            (EventSource.AI_AGENT, "Sure! I've sent a reset link to your email."),
            (EventSource.CUSTOMER, "Got it, that worked. Thanks!"),
            (EventSource.AI_AGENT, "Happy to help! Anything else?"),
            (EventSource.CUSTOMER, "Yes - what are your store's opening hours on weekends?"),
            (EventSource.AI_AGENT, "We're open 9am-6pm on Saturdays and 10am-4pm on Sundays."),
            (EventSource.CUSTOMER, "And do you have parking nearby?"),
            (EventSource.AI_AGENT, "Yes, there's a free lot right behind the store."),
        ]
    )

    result = await pruner.prune(context, [rule])

    assert len(result.pruned_rules) == 1
    assert result.pruned_rules[0].is_still_relevant is False, (
        f"expected the concluded password-reset rule to be judged stale, "
        f"but got: {result.pruned_rules[0].reasoning}"
    )


async def test_that_a_rule_governing_an_open_request_is_judged_still_relevant(
    pruner: RulePruner,
) -> None:
    rule = create_rule(
        condition="the customer wants to book a flight",
        action=(
            "Collect the source and destination airports, the travel dates, and the "
            "traveler names, then book the flight."
        ),
    )
    context = _context(
        [
            (EventSource.CUSTOMER, "Hi, I'd like to book a flight to Rome."),
            (EventSource.AI_AGENT, "Great! Which airport will you be departing from?"),
            (EventSource.CUSTOMER, "From JFK."),
            (EventSource.AI_AGENT, "And what dates would you like to travel?"),
            (EventSource.CUSTOMER, "Let me check with my wife and get back to you in a moment."),
            (EventSource.AI_AGENT, "Of course, take your time!"),
        ]
    )

    result = await pruner.prune(context, [rule])

    assert len(result.pruned_rules) == 1
    assert result.pruned_rules[0].is_still_relevant is True, (
        f"expected the in-progress booking rule to be judged still relevant, "
        f"but got: {result.pruned_rules[0].reasoning}"
    )


async def test_that_a_standing_policy_rule_is_judged_still_relevant(
    pruner: RulePruner,
) -> None:
    # A standing constraint isn't tied to any topic; it must survive pruning even
    # when nothing in the recent conversation touches it (the over-eviction guard).
    rule = create_rule(
        condition="",
        action=None,
        description=(
            "Never share customer account details with anyone other than the verified "
            "account holder, and never state another customer's information."
        ),
        title="Account Data Privacy",
    )
    context = _context(
        [
            (EventSource.CUSTOMER, "What are your opening hours?"),
            (EventSource.AI_AGENT, "We're open 9am-6pm on weekdays."),
            (EventSource.CUSTOMER, "Great, and where's your nearest branch?"),
            (EventSource.AI_AGENT, "Our nearest branch is on 5th Avenue."),
        ]
    )

    result = await pruner.prune(context, [rule])

    assert len(result.pruned_rules) == 1
    assert result.pruned_rules[0].is_still_relevant is True, (
        f"expected the standing privacy policy to be judged still relevant, "
        f"but got: {result.pruned_rules[0].reasoning}"
    )
