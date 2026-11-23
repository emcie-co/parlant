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
from dataclasses import replace
from typing import Any, Mapping, Sequence, cast

from parlant.core.container import Container
from pytest import fixture, mark

from parlant.core.agents import Effort
from parlant.core.common import Weight
from parlant.core.nlp.prompt_routing import PromptConfigurationService
from parlant.core.engines.compass.matching.rule_ranker import RuleRankSchema
from parlant.core.engines.compass.matching.rule_pruner import (
    RulePruner,
)
from parlant.core.engines.compass.context import CompassDependencies, EngineContext, ResponseState
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.rules import Rule
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.decision import (
    DecisionMaker,
    DecisionResult,
    Proposition,
    PropositionAnswer,
)
from parlant.core.nlp.generation import SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    create_engine_context,
    create_rule,
)


@fixture
def pruner(container: Container) -> RulePruner:
    return container[RulePruner]


class _FakeModelRouter:
    def __init__(
        self,
        generator: Any,
        reasoning_effort: str | None,
        decision_maker: DecisionMaker | None = None,
    ) -> None:
        self._generator = generator
        self._reasoning_effort = reasoning_effort
        self._decision_maker = decision_maker
        self.decision_lookups: list[Mapping[str, Any]] = []

    async def get_schematic_generator(self, **kwargs: Any) -> Any:
        return self._generator

    async def get_decision_maker(self, **kwargs: Any) -> DecisionMaker | None:
        self.decision_lookups.append(kwargs)
        return self._decision_maker

    async def resolve_schematic_reasoning_effort(self, **kwargs: Any) -> str | None:
        return self._reasoning_effort


def _make_pruner(
    generator: Any = None,
    reasoning_effort: str | None = None,
    decision_maker: DecisionMaker | None = None,
    router: _FakeModelRouter | None = None,
) -> RulePruner:
    # The prompt-building helpers only read the context, so tests that don't
    # generate can pass a None generator.
    tracer = LocalTracer()
    return RulePruner(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(
            Any, router or _FakeModelRouter(generator, reasoning_effort, decision_maker)
        ),
        prompt_config=PromptConfigurationService(),
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
    # Losing a discovered rule to a transient generation error is worse than
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
        self.hints: list[Mapping[str, Any]] = []
        self._count = 0

    async def generate(
        self,
        prompt: Any,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[RuleRankSchema]:
        self._count += 1
        call_index = self._count
        self.started.append(call_index)
        self.hints.append(dict(hints))
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


async def test_that_pruning_gets_its_hedge_timeout_from_the_matching_rank_policy() -> None:
    class _CustomOptimizationPolicy(CompassOptimizationPolicy):
        def get_hedge_timeout(
            self,
            route: str,
            *,
            qualifiers: Mapping[str, str] | None = None,
        ) -> float | None:
            assert route == "matching.rank"
            assert qualifiers == {"effort": Effort.MEDIUM.value}
            return 2.5

    generator = _RecordingGenerator()
    pruner = _make_pruner(generator)
    context = _context([(EventSource.CUSTOMER, "hello")])
    context.dependencies = CompassDependencies(_CustomOptimizationPolicy())

    await pruner.prune(
        context,
        [create_rule(condition="condition", action="action")],
    )

    assert generator.hints[0]["hedge_timeout"] == 2.5


async def test_that_pruning_uses_the_reasoning_effort_from_its_model_route() -> None:
    generator = _RecordingGenerator()
    pruner = _make_pruner(generator, reasoning_effort="high")
    context = _context([(EventSource.CUSTOMER, "hello")])

    await pruner.prune(
        context,
        [create_rule(condition="condition", action="action")],
    )

    assert generator.hints[0]["reasoning_effort"] == "high"


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


# --- Decision path ------------------------------------------------------------
#
# Where the ranker would ask its rules as one batched decision (no per-rule
# explanation wanted, and a decision model configured for matching.rank), the
# pruner does the same: one decision call for every candidate, instead of one
# schematic call per rule.


class _RecordingDecisionMaker(DecisionMaker):
    """Answers every proposition from a fixed table (default 0.9), recording the asks."""

    def __init__(
        self,
        probabilities: Sequence[float] = (),
        fail: bool = False,
    ) -> None:
        self._probabilities = list(probabilities)
        self._fail = fail
        self.calls: list[tuple[Any, Mapping[str, Any]]] = []

    async def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        hints: Mapping[str, Any] = {},
    ) -> DecisionResult:
        self.calls.append((state, questions))
        if self._fail:
            raise RuntimeError("boom")
        probabilities = self._probabilities or [0.9] * len(questions)
        return DecisionResult(
            answers={
                name: PropositionAnswer(probability=probability)
                for name, probability in zip(questions, probabilities)
            },
            info=GenerationInfo(
                schema_name="Decision",
                model="test-decision-model",
                duration=0.01,
                usage=UsageInfo(input_tokens=100, output_tokens=0, requests=1),
            ),
        )

    @property
    def id(self) -> str:
        return "recording-decision-maker"

    @property
    def max_tokens(self) -> int:
        return 32768


class _CountingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        prompt: Any,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[RuleRankSchema]:
        self.calls += 1
        return SchematicGenerationResult(
            content=RuleRankSchema(tldr="scored", s=4),
            info=_generation_info(),
        )


def _effort_context(effort: Effort) -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(agent_effort=effort)
    return context


def _rules(count: int = 3) -> list[Rule]:
    return [create_rule(condition=f"condition {i}", action=f"action {i}") for i in range(count)]


@mark.parametrize("effort", [Effort.MIN, Effort.LOW, Effort.MEDIUM])
async def test_that_the_pruner_asks_every_rule_in_one_decision_up_to_medium_effort(
    effort: Effort,
) -> None:
    decision_maker = _RecordingDecisionMaker()
    generator = _CountingGenerator()
    pruner = _make_pruner(generator, decision_maker=decision_maker)
    rules = _rules()

    result = await pruner.prune(_effort_context(effort), rules)

    assert len(decision_maker.calls) == 1
    assert generator.calls == 0  # the per-rule fan-out never ran
    (_state, questions) = decision_maker.calls[0]
    assert len(questions) == len(rules)
    assert all(isinstance(q, Proposition) for q in questions.values())
    assert {p.rule.id for p in result.pruned_rules} == {r.id for r in rules}


async def test_that_the_pruner_keeps_the_per_rule_fan_out_at_max_effort() -> None:
    decision_maker = _RecordingDecisionMaker()
    generator = _CountingGenerator()
    router = _FakeModelRouter(generator, None, decision_maker)
    pruner = _make_pruner(router=router)

    await pruner.prune(_effort_context(Effort.MAX), _rules())

    assert decision_maker.calls == []
    assert router.decision_lookups == []  # no lookup it could not act on
    assert generator.calls == 3


async def test_that_at_high_effort_only_high_weight_rules_get_the_per_rule_fan_out() -> None:
    decision_maker = _RecordingDecisionMaker()
    generator = _CountingGenerator()
    pruner = _make_pruner(generator, decision_maker=decision_maker)
    rules = [
        replace(create_rule(condition=f"condition {i}", action="action"), weight=weight)
        for i, weight in enumerate([Weight.LOW, Weight.MEDIUM, Weight.HIGH])
    ]

    result = await pruner.prune(_effort_context(Effort.HIGH), rules)

    assert len(decision_maker.calls) == 1
    (_state, questions) = decision_maker.calls[0]
    assert set(questions) == {str(r.id) for r in rules if r.weight != Weight.HIGH}
    assert generator.calls == 1
    assert [p.rule.id for p in result.pruned_rules] == [r.id for r in rules]
    assert result.generation_info is not None
    assert result.generation_info.usage.requests == 2


async def test_that_the_pruner_keeps_the_per_rule_fan_out_without_a_decision_model() -> None:
    generator = _CountingGenerator()
    pruner = _make_pruner(generator)

    await pruner.prune(_effort_context(Effort.LOW), _rules())

    assert generator.calls == 3


async def test_that_the_pruner_looks_up_its_decision_model_on_the_matching_rank_route() -> None:
    router = _FakeModelRouter(_CountingGenerator(), None, _RecordingDecisionMaker())
    pruner = _make_pruner(router=router)

    await pruner.prune(_effort_context(Effort.LOW), _rules())

    assert [lookup["route"] for lookup in router.decision_lookups] == ["matching.rank"]


@mark.parametrize(
    ("probability", "still_relevant"),
    [(0.0, False), (0.49, False), (0.5, True), (0.99, True)],
)
async def test_that_a_decision_probability_becomes_the_pruned_rules_score(
    probability: float,
    still_relevant: bool,
) -> None:
    pruner = _make_pruner(
        _CountingGenerator(),
        decision_maker=_RecordingDecisionMaker(probabilities=[probability]),
    )

    result = await pruner.prune(_effort_context(Effort.LOW), _rules(1))

    (pruned,) = result.pruned_rules
    assert pruned.score == probability
    assert pruned.is_still_relevant is still_relevant
    assert pruned.reasoning  # says why without a tldr


async def test_that_the_decision_path_reports_the_decisions_usage() -> None:
    pruner = _make_pruner(_CountingGenerator(), decision_maker=_RecordingDecisionMaker())

    result = await pruner.prune(_effort_context(Effort.LOW), _rules())

    assert result.generation_info is not None
    assert result.generation_info.usage.requests == 1
    assert result.generation_info.usage.input_tokens == 100


async def test_that_a_failed_pruning_decision_keeps_every_rule() -> None:
    pruner = _make_pruner(_CountingGenerator(), decision_maker=_RecordingDecisionMaker(fail=True))
    rules = _rules()

    result = await pruner.prune(_effort_context(Effort.LOW), rules)

    assert {p.rule.id for p in result.pruned_rules} == {r.id for r in rules}
    assert all(p.is_still_relevant and p.score == 1.0 for p in result.pruned_rules)
    assert result.generation_info is None


async def test_that_the_pruning_decision_state_asks_no_score_or_json_output() -> None:
    decision_maker = _RecordingDecisionMaker()
    pruner = _make_pruner(_CountingGenerator(), decision_maker=decision_maker)

    await pruner.prune(_effort_context(Effort.LOW), _rules())

    (state, _questions) = decision_maker.calls[0]
    assert "OUTPUT FORMAT" not in state
    assert "integer score" not in state


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
        content=(
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
