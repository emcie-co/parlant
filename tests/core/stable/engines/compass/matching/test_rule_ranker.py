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
from typing import Any, Mapping, Sequence, cast

from lagom import Container
from pytest import fixture

from parlant.core.nlp.common import UsageInfo
from parlant.core.engines.compass.matching.rule_ranker import (
    RuleRanker,
    RuleRankSchema,
    _format_rule,
)
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.nlp.generation import SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.rules import Rule
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    RecordedEvent,
    RecordedSpan,
    RecordingTracer,
    base_test_that_rules_are_ranked_correctly,
    create_engine_context,
    create_rule,
    create_term,
)


@fixture
def ranker(container: Container) -> RuleRanker:
    return container[RuleRanker]


def _make_ranker(
    generator: Any = None,
    tracer: LocalTracer | None = None,
) -> RuleRanker:
    # The prompt-building helpers (`_build_prompt`/`_build_shared_prompt`/
    # `_cache_breakpoint`) only read the context, so the schematic generator isn't
    # exercised — and this avoids the (separately broken) container fixture.
    tracer = tracer or LocalTracer()
    return RuleRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        schematic_generator=cast(Any, generator),
    )


def _generation_info() -> GenerationInfo:
    return GenerationInfo(
        schema_name="RuleRankSchema",
        model="fake-model",
        duration=0.01,
        usage=UsageInfo(input_tokens=1, output_tokens=1),
    )


class _ScoredRankGenerator:
    def __init__(self, results: Sequence[tuple[str, int]]) -> None:
        self._results = iter(results)

    async def generate(
        self,
        prompt: Any,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[RuleRankSchema]:
        tldr, score = next(self._results)
        return SchematicGenerationResult(
            content=RuleRankSchema(tldr=tldr, s=score),
            info=_generation_info(),
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
        action=None,
        description="Refunds are available before shipment.",
    ) == ("Title: Refund Eligibility\n\nPolicy: Refunds are available before shipment.")


def test_that_ranker_formats_condition_action_rule_with_title_and_details() -> None:
    assert _format_rule(
        title="Refund Flow",
        condition="the customer asks for a refund",
        action="start the refund flow",
        description="Use the refund system.",
    ) == (
        "Title: Refund Flow\n\n"
        "When: the customer asks for a refund\n"
        "Then: start the refund flow\n\n"
        "Details: Use the refund system."
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


async def test_that_ranker_records_span_and_per_rule_result_events() -> None:
    span_tracer = RecordingTracer()
    context_tracer = RecordingTracer()
    ranker = _make_ranker(
        _ScoredRankGenerator(
            [
                ("refund is relevant", 5),
                ("shipping is unrelated", 2),
            ]
        ),
        tracer=span_tracer,
    )
    relevant_rule = create_rule(
        condition="customer asks for a refund",
        action="start refund flow",
        title="Refund Flow",
    )
    irrelevant_rule = create_rule(
        condition="customer asks about shipping",
        action="start shipping flow",
        title="Shipping Flow",
    )
    context = replace(
        create_engine_context(conversation=[(EventSource.CUSTOMER, "I want a refund")]),
        tracer=context_tracer,
    )
    context.state = ResponseState()

    await ranker.evaluate(context, [relevant_rule, irrelevant_rule])

    assert span_tracer.started_spans == [RecordedSpan(name="match.rule.rank", attributes={})]
    assert context_tracer.events == [
        RecordedEvent(
            name="matched.rank.yes",
            attributes={
                "rule_id": str(relevant_rule.id),
                "title": "Refund Flow",
                "score": 1.0,
                "reasoning": "refund is relevant",
            },
            span_id="<main>",
        ),
        RecordedEvent(
            name="matched.rank.no",
            attributes={
                "rule_id": str(irrelevant_rule.id),
                "title": "Shipping Flow",
                "score": 0.4,
                "reasoning": "shipping is unrelated",
            },
            span_id="<main>",
        ),
    ]


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
    from typing import Any, Mapping, cast

    from parlant.core.engines.compass.matching.rule_ranker import RuleRankSchema
    from parlant.core.loggers import StdoutLogger
    from parlant.core.nlp.generation import SchematicGenerationResult
    from parlant.core.nlp.generation_info import GenerationInfo
    from parlant.core.nlp.common import UsageInfo
    from parlant.core.tracer import LocalTracer
    from parlant.core.engines.compass.response_state import ResponseState
    from parlant.core.sessions import EventSource

    class _FixedGenerator:
        async def generate(
            self, prompt: Any, hints: Mapping[str, Any] = {}
        ) -> SchematicGenerationResult[RuleRankSchema]:
            return SchematicGenerationResult(
                content=RuleRankSchema(tldr="clearly relevant", s=5),
                info=GenerationInfo(
                    schema_name="RuleRankSchema",
                    model="fake",
                    duration=0.01,
                    usage=UsageInfo(input_tokens=1, output_tokens=1),
                ),
            )

    tracer = LocalTracer()
    ranker = RuleRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        schematic_generator=cast(Any, _FixedGenerator()),
    )
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
