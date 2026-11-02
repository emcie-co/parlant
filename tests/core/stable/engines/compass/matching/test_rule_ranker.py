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

from typing import Any, Mapping, Sequence, cast
from dataclasses import replace

from parlant.core.container import Container
from pytest import fixture, mark

from parlant.core.nlp.prompt_routing import PromptConfigurationService
from parlant.core.engines.compass.matching.rule_ranker import (
    PromptSections as RankerPromptSections,
)
from parlant.core.engines.compass.matching.rule_ranker import (
    RuleRanker,
    RuleRankSchema,
    _format_rule,
)
from parlant.core.engines.compass.matching.common import MAX_FULL_POLICY_CONTEXT_RULES
from parlant.core.agents import Effort
from parlant.core.engines.compass.context import EngineContext, ResponseState
from parlant.core.nlp.decision import (
    DecisionMaker,
    DecisionResult,
    Proposition,
    PropositionAnswer,
)
from parlant.core.nlp.generation import SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.common import UsageInfo
from parlant.core.rules import Rule
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    base_test_that_rules_are_ranked_correctly,
    create_engine_context,
    create_rule,
    create_term,
)


@fixture
def ranker(container: Container) -> RuleRanker:
    return container[RuleRanker]


def _make_ranker() -> RuleRanker:
    # The prompt-building helpers (`_build_prompt`/`_build_shared_prompt`/
    # `_cache_breakpoint`) only read the context, so the schematic generator isn't
    # exercised — and this avoids the (separately broken) container fixture.
    tracer = LocalTracer()
    return RuleRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, None),
        prompt_config=PromptConfigurationService(),
    )


def _cached_prefix(ranker: RuleRanker, context: EngineContext, rule: Rule) -> str:
    # The portion Gemini would cache: everything before the breakpoint marker (the
    # marker itself and the rest are the live, per-turn suffix).
    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]
    breakpoint_marker = ranker._cache_breakpoint(context)
    index = prompt.find(breakpoint_marker)
    assert index != -1, f"cache breakpoint {breakpoint_marker!r} not found in prompt"
    return prompt[:index]


def test_that_ranker_formats_description_only_policy_rule() -> None:
    assert _format_rule(
        title="Refund Eligibility",
        condition="",
        description="Refunds are available before shipment.",
    ) == ("Title: Refund Eligibility\n\nPolicy: Refunds are available before shipment.")


def test_that_ranker_formats_condition_action_rule_with_title_and_details() -> None:
    assert _format_rule(
        title="Refund Flow",
        condition="the customer asks for a refund",
        description="start the refund flow",
    ) == (
        "Title: Refund Flow\n\n"
        "Condition: the customer asks for a refund\n\n"
        "Details: start the refund flow"
    )


def test_that_the_ranker_prompt_includes_the_agent_reasoning_but_keeps_it_out_of_the_cached_prefix(
    ranker: RuleRanker,
) -> None:
    rule = create_rule(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")]
    )
    context.state = ResponseState(
        reasoning_steps=[
            "The customer asked which toppings are available.",
            "I should list the available toppings from current stock.",
        ],
    )
    shots: Sequence[object] = []  # shots are irrelevant to the reasoning section

    prompt = ranker._build_prompt(context, rule, shots).build()  # type: ignore[arg-type]
    assert "list the available toppings from current stock" in prompt

    # Caching invariant: per-step reasoning must NOT enter the cached shared prefix.
    shared = ranker._build_shared_prompt(context, shots).build()  # type: ignore[arg-type]
    assert "list the available toppings from current stock" not in shared


def test_that_the_latest_customer_message_is_kept_out_of_the_cached_prefix() -> None:
    # The latest customer message changes every turn, so it must live in the live
    # suffix — not the cached prefix — or the prefix hash shifts each turn and the
    # warmed cache is never reused.
    ranker = _make_ranker()
    rule = create_rule(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    context = create_engine_context(
        conversation=[
            (EventSource.CUSTOMER, "hi"),
            (EventSource.AI_AGENT, "hello, how can I help?"),
            (EventSource.CUSTOMER, "ZZQ_LATEST what toppings do you have?"),
        ]
    )
    context.state = ResponseState()

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]
    prefix = _cached_prefix(ranker, context, rule)

    assert "ZZQ_LATEST" in prompt  # present in the full prompt (the live suffix)
    assert "ZZQ_LATEST" not in prefix  # but NOT in the cached prefix


def test_that_the_cached_prefix_is_identical_with_and_without_a_trailing_customer_message() -> None:
    # The crux: the prefix `prefill` warms at the end of a turn (history ending in
    # the agent's reply) must be byte-identical to the prefix the next turn's
    # matching builds (that same history plus the new customer message).
    ranker = _make_ranker()
    rule = create_rule(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    base = [
        (EventSource.CUSTOMER, "hi"),
        (EventSource.AI_AGENT, "hello, how can I help?"),
    ]

    prefill_context = create_engine_context(conversation=base)
    prefill_context.state = ResponseState()

    matching_context = create_engine_context(
        conversation=[*base, (EventSource.CUSTOMER, "what toppings do you have?")]
    )
    matching_context.state = ResponseState()

    assert _cached_prefix(ranker, prefill_context, rule) == _cached_prefix(
        ranker, matching_context, rule
    )


def test_that_all_trailing_customer_messages_are_excluded_from_the_cached_prefix() -> None:
    # Truncation happens at the last AI-agent message, so multiple customer messages
    # sent back-to-back before the agent replies all land in the suffix.
    ranker = _make_ranker()
    rule = create_rule(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    context = create_engine_context(
        conversation=[
            (EventSource.CUSTOMER, "hi"),
            (EventSource.AI_AGENT, "hello, how can I help?"),
            (EventSource.CUSTOMER, "ZZQ_FIRST a question"),
            (EventSource.CUSTOMER, "ZZQ_SECOND and another"),
        ]
    )
    context.state = ResponseState()

    prefix = _cached_prefix(ranker, context, rule)

    assert "ZZQ_FIRST" not in prefix
    assert "ZZQ_SECOND" not in prefix


RULES_DICT: dict[str, dict[str, str]] = {
    "ask_toppings": {
        "condition": "the customer asks about toppings",
        "action": "list the available toppings",
    },
}


def test_that_a_rule_ranker_can_be_created(ranker: RuleRanker) -> None:
    assert ranker is not None


async def test_that_a_relevant_rule_is_ranked_as_relevant(ranker: RuleRanker) -> None:
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")],
        conversation_rule_names=["ask_toppings"],
        relevant_rule_names=["ask_toppings"],
        irrelevant_rule_names=[],
    )


# --- Per-rule glossary terms in the prompt tail -----------------------------
#
# A rule may depend on glossary terms to be interpreted correctly. The
# matcher resolves each rule's terms into state.terms_by_rule; the
# ranker renders them WITH the rule in the live tail (never the cached
# prefix), skipping terms already present in the turn's glossary section.


def test_that_a_rules_terms_are_rendered_in_the_prompt_tail() -> None:
    ranker = _make_ranker()
    rule = create_rule(
        condition="the customer reports PRS",
        action="escalate to a specialist",
    )
    term = create_term("PRS", "ZZQ_TERM Pinewood Rash Syndrome - an allergy to pinewood.")
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "my PRS flared up")])
    context.state = ResponseState(terms_by_rule={rule.id: [term]})

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]
    prefix = _cached_prefix(ranker, context, rule)

    assert "ZZQ_TERM" in prompt  # rendered with the rule...
    assert "ZZQ_TERM" not in prefix  # ...but never in the cached prefix


def test_that_rule_terms_already_in_the_turns_glossary_are_not_duplicated() -> None:
    ranker = _make_ranker()
    rule = create_rule(
        condition="the customer reports PRS",
        action="escalate to a specialist",
    )
    term = create_term("PRS", "ZZQ_TERM Pinewood Rash Syndrome.")
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "my PRS flared up")])
    context.state = ResponseState(
        glossary_terms={term},  # already in the shared-prefix glossary section
        terms_by_rule={rule.id: [term]},
    )

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert prompt.count("ZZQ_TERM") == 1


def test_that_the_terms_block_is_omitted_for_rules_without_terms() -> None:
    ranker = _make_ranker()
    rule = create_rule(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState()

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "TERMS USED BY THIS RULE" not in prompt


# --- The RuleEvaluator port -------------------------------------------------------
#
# The ranker is a turn evaluator: per-rule LLM judgment behind the unified
# RuleEvaluator port. It scores rules and never attaches highlights.


def test_that_the_rule_ranker_implements_the_turn_evaluator_port() -> None:
    from parlant.core.engines.compass.matching.rule_evaluation import TurnEvaluator

    assert isinstance(_make_ranker(), TurnEvaluator)


async def test_that_ranker_evaluations_carry_scores_and_no_highlights() -> None:
    from typing import Any, Mapping

    from parlant.core.engines.compass.matching.rule_ranker import RuleRankSchema
    from parlant.core.loggers import StdoutLogger
    from parlant.core.nlp.generation import SchematicGenerationResult
    from parlant.core.nlp.generation_info import GenerationInfo
    from parlant.core.nlp.common import UsageInfo
    from parlant.core.tracer import LocalTracer
    from parlant.core.engines.compass.context import ResponseState
    from parlant.core.sessions import EventSource

    class _FixedGenerator:
        def __init__(self) -> None:
            self.hints: Mapping[str, Any] | None = None

        async def generate(
            self, prompt: Any, hints: Mapping[str, Any] = {}
        ) -> SchematicGenerationResult[RuleRankSchema]:
            self.hints = hints
            return SchematicGenerationResult(
                content=RuleRankSchema(tldr="clearly relevant", s=5),
                info=GenerationInfo(
                    schema_name="RuleRankSchema",
                    model="fake",
                    duration=0.01,
                    usage=UsageInfo(input_tokens=1, output_tokens=1),
                ),
            )

    class _FakeModelRouter:
        def __init__(self, generator: _FixedGenerator) -> None:
            self._generator = generator

        async def get_schematic_generator(self, **kwargs: Any) -> _FixedGenerator:
            return self._generator

        async def get_decision_maker(self, **kwargs: Any) -> None:
            return None

        async def resolve_schematic_reasoning_effort(self, **kwargs: Any) -> str:
            return "high"

    tracer = LocalTracer()
    generator = _FixedGenerator()
    ranker = RuleRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, _FakeModelRouter(generator)),
        prompt_config=PromptConfigurationService(),
    )

    async def no_shots() -> Sequence[Any]:
        return []

    ranker.shots = no_shots  # type: ignore[method-assign]
    rule = create_rule(condition="the customer asks about toppings", action="list them")
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "what toppings?")])
    context.state = ResponseState()

    result = await ranker.evaluate(context, [rule])

    assert len(result.evaluations) == 1
    evaluation = result.evaluations[0]
    assert evaluation.rule == rule
    assert evaluation.is_relevant is True
    assert evaluation.score == 1.0  # s=5 -> 5/5
    assert evaluation.highlights == ()
    assert "clearly relevant" in evaluation.reasoning
    assert result.generation_info is not None
    assert generator.hints is not None
    assert generator.hints["hedge_timeout"] == 5.0
    assert generator.hints["reasoning_effort"] == "high"

    assert await ranker.warm_up(context) is not None
    assert generator.hints["reasoning_effort"] == "high"
    assert "cache" in generator.hints


# --- Policy context for cross-referential rules ---------------------------------
#
# A rule flagged as referencing the agent's other policies is ranked with its
# policy context — the discovered working set plus this turn's pending
# admissions — rendered in the live prompt tail, so conditions like "not
# covered by any of our policies" can actually be judged.


def _cross_referential_rule(condition: str, action: str = "act accordingly") -> Rule:
    return replace(
        create_rule(condition=condition, action=action),
        metadata={"references_other_policies": True},
    )


def test_that_a_large_inventory_uses_the_discovered_set_as_policy_context() -> None:
    ranker = _make_ranker()
    rule = _cross_referential_rule("the request is not covered by any of our policies")
    discovered = create_rule(
        condition="ZZQ_DISCOVERED the customer wants a refund", action="refund"
    )
    pending = create_rule(condition="ZZQ_PENDING the customer asks about delivery", action="answer")

    filler = [
        create_rule(condition=f"filler condition {i}", action=f"filler action {i}")
        for i in range(MAX_FULL_POLICY_CONTEXT_RULES + 1)
    ]
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "can you fix my sink?")])
    context.state = ResponseState(
        usable_rules=[rule, discovered, pending, *filler],
        discovered_rules={discovered},
    )
    context.state.matcher.pending_discovered_rules.add(pending)

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]
    prefix = _cached_prefix(ranker, context, rule)

    assert "THE AGENT'S OTHER ACTIVE POLICIES" in prompt
    assert "ZZQ_DISCOVERED" in prompt
    assert "ZZQ_PENDING" in prompt
    # The policy context is turn-varying; it must stay out of the cached prefix.
    assert "ZZQ_DISCOVERED" not in prefix
    assert "ZZQ_PENDING" not in prefix


def test_that_the_policy_context_excludes_the_rule_under_evaluation() -> None:
    ranker = _make_ranker()
    rule = _cross_referential_rule("ZZQ_SELF the request is not covered by our policies")
    other = create_rule(condition="ZZQ_OTHER the customer wants a refund", action="refund")

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(usable_rules=[rule, other], discovered_rules={rule, other})

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "ZZQ_OTHER" in prompt
    # The rule renders once, in the rule-under-evaluation section — not again
    # as its own policy context.
    assert prompt.count("ZZQ_SELF") == 1


def test_that_non_cross_referential_rules_omit_the_policy_context_section() -> None:
    ranker = _make_ranker()
    rule = create_rule(condition="the customer wants a refund", action="refund")
    other = create_rule(condition="the customer asks about delivery", action="answer")

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(discovered_rules={other})

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "THE AGENT'S OTHER ACTIVE POLICIES" not in prompt


def test_that_a_small_inventory_renders_the_full_inventory_as_policy_context() -> None:
    # Below the cap, the flagged rule sees the complete policy landscape —
    # including rules never discovered — with no dependency on phasing.
    ranker = _make_ranker()
    rule = _cross_referential_rule("the request is not covered by any of our policies")
    discovered = create_rule(
        condition="ZZQ_DISCOVERED the customer wants a refund", action="refund"
    )
    undiscovered = create_rule(
        condition="ZZQ_UNDISCOVERED the customer asks about warranty", action="explain"
    )

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "can you fix my sink?")])
    context.state = ResponseState(
        usable_rules=[rule, discovered, undiscovered],
        discovered_rules={discovered},
    )

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "THE AGENT'S OTHER ACTIVE POLICIES" in prompt
    assert "ZZQ_DISCOVERED" in prompt
    assert "ZZQ_UNDISCOVERED" in prompt


def test_that_an_empty_policy_context_renders_no_section() -> None:
    # Large landscape with nothing discovered and nothing pending: there is no
    # context to resolve references against, so the section is omitted rather
    # than rendered empty.
    ranker = _make_ranker()
    rule = _cross_referential_rule("the request is not covered by any of our policies")
    filler = [
        create_rule(condition=f"filler condition {i}", action=f"filler action {i}")
        for i in range(MAX_FULL_POLICY_CONTEXT_RULES + 1)
    ]

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(usable_rules=[rule, *filler])

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "THE AGENT'S OTHER ACTIVE POLICIES" not in prompt


def test_that_same_turn_positive_verdicts_join_the_policy_context() -> None:
    # In a large landscape, recall's misses are exactly what the ranker can
    # catch — a rule judged relevant in an earlier round this turn must be
    # visible to a deferred cross-referential rule, even if recall never
    # surfaced it.
    ranker = _make_ranker()
    rule = _cross_referential_rule("the request is not covered by any of our policies")
    matched = create_rule(
        condition="ZZQ_MATCHED the customer wants an exchange", action="handle the exchange"
    )

    filler = [
        create_rule(condition=f"filler condition {i}", action=f"filler action {i}")
        for i in range(MAX_FULL_POLICY_CONTEXT_RULES + 1)
    ]
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "can you fix my sink?")])
    context.state = ResponseState(usable_rules=[rule, matched, *filler])
    context.state.matcher.pending_matched_rules.add(matched)

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "THE AGENT'S OTHER ACTIVE POLICIES" in prompt
    assert "ZZQ_MATCHED" in prompt


# --- Dependency context ----------------------------------------------------------
#
# A rule that depends on other rules only applies where they apply, so it is
# ranked with its dependency targets' content rendered in the live prompt tail
# — the verdict judges the rule in the context it operates within (e.g. a
# journey-scoped observation is judged against the journey's procedure).


def test_that_a_dependent_rules_prompt_renders_its_dependency_targets() -> None:
    ranker = _make_ranker()
    rule = create_rule(
        condition="the procedure has reached payment confirmation",
        action="handle with care",
    )
    target = create_rule(
        title="Flight booking procedure",
        condition="ZZQ_TARGET the customer wants to book a flight",
        action="follow the booking steps in order",
    )

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(
        dependency_targets_by_rule={rule.id: [target]},
        dependency_prompt_targets_by_rule={rule.id: [target]},
    )

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]
    prefix = _cached_prefix(ranker, context, rule)

    assert "CONTEXT THIS RULE OPERATES WITHIN" in prompt
    assert "ZZQ_TARGET" in prompt
    assert "Flight booking procedure" in prompt
    # Dependency context is per-rule; it must stay out of the cached shared prefix.
    assert "ZZQ_TARGET" not in prefix


def test_that_dependency_context_without_a_scheduler_snapshot_is_marked_uncertain() -> None:
    ranker = _make_ranker()
    rule = create_rule(condition="the customer is at the payment step", action="review")
    target = create_rule(condition="ZZQ_TARGET booking procedure", action="book")
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(dependency_targets_by_rule={rule.id: [target]})

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "CONTEXT THIS RULE MAY OPERATE WITHIN" in prompt
    assert "Do not assume that every target applies" in prompt
    assert "ZZQ_TARGET" in prompt


def test_that_dependency_context_lists_only_the_rules_own_targets() -> None:
    ranker = _make_ranker()
    rule = create_rule(condition="the customer is at the payment step", action="review")
    own_target = create_rule(condition="ZZQ_OWN the booking procedure", action="book")
    other_rule = create_rule(condition="something else entirely", action="other")
    other_target = create_rule(condition="ZZQ_FOREIGN the returns procedure", action="return")

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(
        dependency_targets_by_rule={
            rule.id: [own_target],
            other_rule.id: [other_target],
        },
    )

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "ZZQ_OWN" in prompt
    assert "ZZQ_FOREIGN" not in prompt


def test_that_dependency_context_uses_only_the_targets_prepared_by_the_scheduler() -> None:
    ranker = _make_ranker()
    rule = create_rule(condition="the customer is at the payment step", action="review")
    satisfied = create_rule(condition="ZZQ_SATISFIED booking procedure", action="book")
    unsatisfied = create_rule(condition="ZZQ_UNSATISFIED returns procedure", action="return")

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(
        dependency_targets_by_rule={rule.id: [satisfied, unsatisfied]},
        dependency_prompt_targets_by_rule={rule.id: [satisfied]},
    )

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "ZZQ_SATISFIED" in prompt
    assert "ZZQ_UNSATISFIED" not in prompt


def test_that_rules_without_dependencies_omit_the_dependency_context_section() -> None:
    ranker = _make_ranker()
    rule = create_rule(condition="the customer wants a refund", action="refund")

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "CONTEXT THIS RULE OPERATES WITHIN" not in prompt


async def test_that_a_configured_prompt_provider_supplies_the_rankers_builder() -> None:
    from parlant.core.nlp.model_routing import ModelScope
    from parlant.core.nlp.prompt_routing import PromptConfigurationService

    from parlant.core.engines.compass.prompt_builder import PromptBuilder

    class _MarkedBuilder(PromptBuilder):
        pass

    prompt_config = PromptConfigurationService()
    await prompt_config.configure(ModelScope.server(), {"matching.rank": _MarkedBuilder})

    tracer = LocalTracer()
    ranker = RuleRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, None),
        prompt_config=prompt_config,
    )

    rule = create_rule(condition="the customer wants a refund", action="refund them")
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(usable_rules=[rule])

    builder = ranker._build_prompt(context, rule, shots=[])  # type: ignore[arg-type]

    assert type(builder) is _MarkedBuilder


async def test_that_an_adapt_section_provider_rewords_a_real_ranker_section() -> None:
    # The full customization loop: a provider registered on the route returns
    # an adapt_section subclass, and the ranker's real prompt comes out
    # reworded — no component knows anything happened.
    from typing import Any as _Any

    from parlant.core.engines.compass.prompt_builder import (
        VETO,
        PromptBuilder,
        SectionAdaptation,
    )
    from parlant.core.nlp.model_routing import ModelScope
    from parlant.core.nlp.prompt_routing import PromptConfigurationService

    class _Reworded(PromptBuilder):
        def adapt_section(
            self, name: _Any, template: str, props: dict[str, _Any]
        ) -> str | SectionAdaptation | None:
            if name == RankerPromptSections.TASK_DESCRIPTION:
                return "ZZQ_CUSTOM_TASK_DESCRIPTION"
            if name == RankerPromptSections.EXAMPLES:
                return VETO
            return None

    prompt_config = PromptConfigurationService()
    await prompt_config.configure(ModelScope.server(), {"matching.rank": _Reworded})

    tracer = LocalTracer()
    ranker = RuleRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, None),
        prompt_config=prompt_config,
    )
    rule = create_rule(condition="the customer wants a refund", action="refund them")
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(usable_rules=[rule])

    prompt = ranker._build_prompt(context, rule, shots=[]).build()  # type: ignore[arg-type]

    assert "ZZQ_CUSTOM_TASK_DESCRIPTION" in prompt
    assert str(RankerPromptSections.EXAMPLES) not in prompt  # vetoed section gone


# --- The decision path: one batched judgment instead of a fan-out -----------------
#
# Given a DecisionMaker, the ranker stops sending one model call per rule at the
# cheap efforts and asks all of them in a single decision instead. Those are the
# efforts where `_should_include_tldr` is already False, so nothing a decision
# model cannot produce is lost. Above them the schematic generator keeps its job.


class _RecordingDecisionMaker(DecisionMaker):
    """Answers every proposition with a fixed probability, recording the asks."""

    def __init__(self, probability: float = 0.9, cost: str = "0.0000206") -> None:
        self.probability = probability
        self.cost = cost
        self.calls: list[tuple[Any, Mapping[str, Any]]] = []

    async def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        hints: Mapping[str, Any] = {},
    ) -> DecisionResult:
        self.calls.append((state, questions))
        return DecisionResult(
            answers={name: PropositionAnswer(probability=self.probability) for name in questions},
            info=GenerationInfo(
                schema_name="Decision",
                model="test-decision-model",
                duration=0.01,
                usage=UsageInfo(
                    input_tokens=100,
                    output_tokens=10,
                    extra={"reported_cost": self.cost},
                    requests=1,
                ),
            ),
        )

    @property
    def id(self) -> str:
        return "recording-decision-maker"

    @property
    def max_tokens(self) -> int:
        return 32768


class _CountingGenerator:
    """Stands in for the schematic generator, counting per-rule calls."""

    def __init__(self, score: int = 5) -> None:
        self.score = score
        self.calls = 0

    async def generate(
        self, prompt: Any, hints: Mapping[str, Any] = {}
    ) -> SchematicGenerationResult[RuleRankSchema]:
        self.calls += 1
        return SchematicGenerationResult(
            content=RuleRankSchema(tldr="because", s=self.score),
            info=GenerationInfo(
                schema_name="RuleRankSchema",
                model="fake",
                duration=0.01,
                usage=UsageInfo(input_tokens=1, output_tokens=1),
            ),
        )


class _StubRouter:
    def __init__(
        self,
        generator: _CountingGenerator,
        decision_maker: DecisionMaker | None = None,
    ) -> None:
        self._generator = generator
        self._decision_maker = decision_maker
        self.decision_lookups = 0

    async def get_schematic_generator(self, **kwargs: Any) -> _CountingGenerator:
        return self._generator

    async def get_decision_maker(self, **kwargs: Any) -> DecisionMaker | None:
        self.decision_lookups += 1
        return self._decision_maker

    async def resolve_schematic_reasoning_effort(self, **kwargs: Any) -> str:
        return "low"


def _decision_ranker(
    decision_maker: DecisionMaker | None,
    generator: _CountingGenerator,
    router: _StubRouter | None = None,
) -> RuleRanker:
    tracer = LocalTracer()
    ranker = RuleRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, router or _StubRouter(generator, decision_maker)),
        prompt_config=PromptConfigurationService(),
    )

    async def no_shots() -> Sequence[Any]:
        return []

    ranker.shots = no_shots  # type: ignore[method-assign]
    return ranker


def _low_effort_context(effort: Effort = Effort.LOW) -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "what toppings?")])
    context.state = ResponseState(agent_effort=effort)
    return context


def _three_rules() -> list[Rule]:
    return [create_rule(condition=f"condition {i}", action=f"action {i}") for i in range(3)]


async def test_that_the_ranker_uses_the_decision_maker_at_low_effort() -> None:
    decision_maker = _RecordingDecisionMaker()
    generator = _CountingGenerator()
    ranker = _decision_ranker(decision_maker, generator)

    result = await ranker.evaluate(_low_effort_context(Effort.LOW), _three_rules())

    assert len(decision_maker.calls) == 1
    assert generator.calls == 0  # the per-rule fan-out never ran
    assert len(result.evaluations) == 3


async def test_that_the_ranker_uses_the_decision_maker_at_min_effort() -> None:
    decision_maker = _RecordingDecisionMaker()
    generator = _CountingGenerator()
    ranker = _decision_ranker(decision_maker, generator)

    await ranker.evaluate(_low_effort_context(Effort.MIN), _three_rules())

    assert len(decision_maker.calls) == 1
    assert generator.calls == 0


async def test_that_the_ranker_asks_one_question_per_rule_in_a_single_decision() -> None:
    decision_maker = _RecordingDecisionMaker()
    ranker = _decision_ranker(decision_maker, _CountingGenerator())
    rules = _three_rules()

    await ranker.evaluate(_low_effort_context(), rules)

    (_state, questions) = decision_maker.calls[0]
    assert len(questions) == len(rules)
    # Each rule becomes one proposition, and the rule's own text is what the
    # model is asked about.
    for question in questions.values():
        assert isinstance(question, Proposition)
        assert isinstance(question.instructions, dict)
        assert "condition" in str(question.instructions["rule"])


async def test_that_every_rule_gets_its_own_evaluation_back_from_one_decision() -> None:
    decision_maker = _RecordingDecisionMaker()
    ranker = _decision_ranker(decision_maker, _CountingGenerator())
    rules = _three_rules()

    result = await ranker.evaluate(_low_effort_context(), rules)

    assert {e.rule.id for e in result.evaluations} == {r.id for r in rules}


@mark.parametrize(
    ("probability", "expected_relevance"),
    [
        (0.0, False),
        (0.49, False),
        (0.5, True),  # the midpoint carries forward: a filter's cheap mistake
        (0.99, True),
    ],
)
async def test_that_a_relevance_probability_maps_onto_the_threshold(
    probability: float,
    expected_relevance: bool,
) -> None:
    ranker = _decision_ranker(
        _RecordingDecisionMaker(probability=probability), _CountingGenerator()
    )

    result = await ranker.evaluate(_low_effort_context(), [create_rule(condition="c", action="a")])

    (evaluation,) = result.evaluations
    assert evaluation.is_relevant is expected_relevance


async def test_that_the_calibrated_probability_becomes_the_evaluations_score() -> None:
    # The probability is already calibrated, so it needs no rescaling on the way
    # to the score downstream stages read.
    ranker = _decision_ranker(_RecordingDecisionMaker(probability=0.73), _CountingGenerator())

    result = await ranker.evaluate(_low_effort_context(), [create_rule(condition="c", action="a")])

    (evaluation,) = result.evaluations
    assert evaluation.score == 0.73


async def test_that_the_decision_path_reports_a_single_request_of_usage() -> None:
    # The saving has to be visible in usage reports: one request for the batch,
    # not one per rule.
    ranker = _decision_ranker(_RecordingDecisionMaker(), _CountingGenerator())

    result = await ranker.evaluate(_low_effort_context(), _three_rules())

    assert result.generation_info is not None
    assert result.generation_info.usage.requests == 1


async def test_that_the_decision_paths_cost_reaches_the_ranker_result() -> None:
    # The decision maker reports the authoritative charge; the ranker must hand it
    # onward untouched, or a batched turn would look free in usage reports.
    ranker = _decision_ranker(_RecordingDecisionMaker(cost="0.0000206"), _CountingGenerator())

    result = await ranker.evaluate(_low_effort_context(), _three_rules())

    assert result.generation_info is not None
    assert result.generation_info.usage.extra["reported_cost"] == "0.0000206"
    assert result.generation_info.model == "test-decision-model"


async def test_that_the_decision_path_explains_itself_without_a_tldr() -> None:
    # A decision model returns no prose, so the reasoning falls back to the
    # sentence the ranker already uses when a tldr is absent.
    ranker = _decision_ranker(_RecordingDecisionMaker(), _CountingGenerator())

    result = await ranker.evaluate(_low_effort_context(), [create_rule(condition="c", action="a")])

    (evaluation,) = result.evaluations
    assert evaluation.reasoning
    assert "relevant" in evaluation.reasoning.lower()


@mark.parametrize("effort", [Effort.MIN, Effort.LOW, Effort.MEDIUM])
async def test_that_the_ranker_uses_the_decision_maker_up_to_medium_effort(
    effort: Effort,
) -> None:
    decision_maker = _RecordingDecisionMaker()
    generator = _CountingGenerator()
    ranker = _decision_ranker(decision_maker, generator)

    await ranker.evaluate(_low_effort_context(effort), _three_rules())

    assert len(decision_maker.calls) == 1
    assert generator.calls == 0


@mark.parametrize("effort", [Effort.HIGH, Effort.MAX])
async def test_that_the_ranker_keeps_the_schematic_generator_above_medium_effort(
    effort: Effort,
) -> None:
    # Above MEDIUM the ranker needs the tldr, which a decision model cannot produce.
    decision_maker = _RecordingDecisionMaker()
    generator = _CountingGenerator()
    ranker = _decision_ranker(decision_maker, generator)
    rules = _three_rules()

    await ranker.evaluate(_low_effort_context(effort), rules)

    assert decision_maker.calls == []
    assert generator.calls == len(rules)


async def test_that_the_ranker_keeps_the_schematic_generator_without_a_decision_maker() -> None:
    # The untouched default: no DecisionMaker configured, so low effort behaves
    # exactly as it does today.
    generator = _CountingGenerator()
    ranker = _decision_ranker(None, generator)
    rules = _three_rules()

    result = await ranker.evaluate(_low_effort_context(Effort.LOW), rules)

    assert generator.calls == len(rules)
    assert len(result.evaluations) == len(rules)


async def test_that_the_ranker_skips_the_decision_path_for_no_rules() -> None:
    decision_maker = _RecordingDecisionMaker()
    ranker = _decision_ranker(decision_maker, _CountingGenerator())

    result = await ranker.evaluate(_low_effort_context(), [])

    assert decision_maker.calls == []
    assert result.evaluations == []


async def test_that_no_decision_model_is_looked_up_above_medium_effort() -> None:
    # A high-effort turn cannot use a decision model, so it must not pay for the
    # lookup — configuration resolution is not free.
    router = _StubRouter(_CountingGenerator(), _RecordingDecisionMaker())
    ranker = _decision_ranker(None, _CountingGenerator(), router)

    await ranker.evaluate(_low_effort_context(Effort.HIGH), _three_rules())

    assert router.decision_lookups == 0


async def test_that_the_decision_model_is_looked_up_at_cheap_efforts() -> None:
    router = _StubRouter(_CountingGenerator(), _RecordingDecisionMaker())
    ranker = _decision_ranker(None, _CountingGenerator(), router)

    await ranker.evaluate(_low_effort_context(Effort.LOW), _three_rules())

    assert router.decision_lookups == 1


async def test_that_the_decision_state_omits_the_json_output_format() -> None:
    # A decision model emits no JSON, so the schematic path's output-format
    # section is not merely wasted tokens in the state — it instructs the model to
    # produce something it structurally cannot.
    decision_maker = _RecordingDecisionMaker()
    ranker = _decision_ranker(decision_maker, _CountingGenerator())

    await ranker.evaluate(_low_effort_context(), _three_rules())

    (state, _questions) = decision_maker.calls[0]

    assert "OUTPUT FORMAT" not in state
    assert "tldr" not in state


def test_that_the_schematic_prompt_keeps_the_json_output_format() -> None:
    # The fan-out still needs it: that path does fill in a JSON structure.
    ranker = _make_ranker()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "what toppings?")])
    context.state = ResponseState()

    prompt = ranker._build_prompt(context, create_rule(condition="c", action="a"), shots=[]).build()

    assert "OUTPUT FORMAT" in prompt


async def test_that_the_decision_state_omits_the_five_point_scale() -> None:
    # The scale is the schematic path's answer format. Asking a yes/no question
    # while instructing the model to emit 1-5 is two different questions at once.
    decision_maker = _RecordingDecisionMaker()
    ranker = _decision_ranker(decision_maker, _CountingGenerator())

    await ranker.evaluate(_low_effort_context(), _three_rules())

    (state, _questions) = decision_maker.calls[0]

    assert "integer score from 1 to 5" not in state
    assert "Scores of 1 and 2" not in state
    # The framing that still applies is kept: this is a permissive first pass.
    assert "first-pass filter" in state
    assert "lean towards yes" in state


def test_that_the_schematic_prompt_keeps_the_five_point_scale() -> None:
    ranker = _make_ranker()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "what toppings?")])
    context.state = ResponseState()

    prompt = ranker._build_prompt(context, create_rule(condition="c", action="a"), shots=[]).build()

    assert "integer score from 1 to 5" in prompt
    assert "Scores of 1 and 2" in prompt
