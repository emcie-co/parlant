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

from collections.abc import Mapping
from types import SimpleNamespace

from parlant.core.container import Container
from typing import Any, cast

import pytest

from parlant.core.nlp.prompt_routing import PromptConfigurationService
from parlant.core.agents import Effort
from parlant.core.canned_responses import (
    CannedResponse,
    CannedResponseRelevanceResult,
    CannedResponseStore,
)
from parlant.core.entity_cq import TurnCannedResponses
from parlant.core.common import JSONSerializable
from parlant.core.engines.compass.context import EngineContext, ResponseState
from parlant.core.engines.compass.turn_directives import TurnDirective
from parlant.core.engines.compass.reviewer import (
    BasicReviewer,
    ReviewSchema,
    Reviewer,
)
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.generation import SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.nlp.react import ToolCallPart
from parlant.core.emissions import EmittedEvent
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer, Tracer

from tests.core.stable.engines.compass.matching.utils import (
    create_engine_context,
    create_staged_tool_event,
)


class _ScriptedReviewGenerator:
    """Returns a fixed review verdict, recording each call's prompt and hints."""

    def __init__(self, content: ReviewSchema) -> None:
        self._content = content
        self.prompts: list[Any] = []
        self.hints: list[Mapping[str, Any]] = []

    async def generate(
        self,
        prompt: Any,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[ReviewSchema]:
        self.prompts.append(prompt)
        self.hints.append(hints)
        return SchematicGenerationResult(
            content=self._content,
            info=GenerationInfo(
                schema_name="ReviewSchema",
                model="fake",
                duration=0.01,
                usage=UsageInfo(input_tokens=1, output_tokens=1),
            ),
        )


class _FakeModelRouter:
    """Delivers the scripted generator, recording each resolution request."""

    def __init__(
        self,
        generator: _ScriptedReviewGenerator | None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._generator = generator
        self._reasoning_effort = reasoning_effort
        self.generator_requests: list[Mapping[str, Any]] = []

    async def get_schematic_generator(self, **kwargs: Any) -> _ScriptedReviewGenerator | None:
        self.generator_requests.append(kwargs)
        return self._generator

    async def resolve_schematic_reasoning_effort(self, **kwargs: Any) -> str | None:
        return self._reasoning_effort


class _StubEntityQueries:
    def __init__(self, canned_responses: TurnCannedResponses | None = None) -> None:
        self._canned_responses = canned_responses or TurnCannedResponses(
            rule_scoped=[], agent_wide=[]
        )

    async def find_canned_responses_for_turn(
        self, *args: Any, **kwargs: Any
    ) -> TurnCannedResponses:
        return self._canned_responses


class _StubCannedResponseStore:
    def __init__(self, relevance_results: list[CannedResponseRelevanceResult]) -> None:
        self._relevance_results = relevance_results

    async def filter_relevant_canned_responses(
        self, *args: Any, **kwargs: Any
    ) -> list[CannedResponseRelevanceResult]:
        return self._relevance_results


def _reviewer(
    basic_schematic_generator: _ScriptedReviewGenerator | None = None,
    entity_queries: Any = None,
    canned_response_store: Any = None,
    reasoning_effort: str | None = None,
    tracer: LocalTracer | None = None,
) -> BasicReviewer:
    tracer = tracer or LocalTracer()
    model_router = _FakeModelRouter(basic_schematic_generator, reasoning_effort)
    return BasicReviewer(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(ModelRouter, model_router),
        prompt_config=PromptConfigurationService(),
        entity_queries=cast(Any, entity_queries or _StubEntityQueries()),
        canned_response_store=cast(Any, canned_response_store),
    )


def _context(effort: Effort = Effort.LOW) -> EngineContext:
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "How much is eggplant today?")]
    )
    context.state = ResponseState(agent_effort=effort)
    return context


def _tool_calls() -> list[ToolCallPart]:
    return [ToolCallPart(id="call-1", name="get_fruit_price", args={"item": "eggplant"})]


def _compliant_basic_generator() -> _ScriptedReviewGenerator:
    return _ScriptedReviewGenerator(ReviewSchema(breaches_or_discrepancies=False))


def test_that_the_basic_reviewer_implements_the_reviewer_port() -> None:
    assert isinstance(_reviewer(), Reviewer)


def test_that_the_basic_reviewer_prompt_uses_boolean_discrepancy_examples() -> None:
    context = _context()

    prompt = (
        _reviewer()
        ._build_basic_prompt(
            context,
            reasoning="I should look up the price.",
            tool_calls=_tool_calls(),
        )
        .build()
    )

    assert "If the agent proposes get_fruit_price for eggplant" in prompt
    assert 'set "breaches_or_discrepancies" to true' in prompt
    assert "get_fruit_price for apples and get_vegetable_price for carrots" in prompt


@pytest.mark.asyncio
async def test_that_the_basic_reviewer_reviews_with_the_basic_schema_at_high_effort() -> None:
    generator = _compliant_basic_generator()
    reviewer = _reviewer(basic_schematic_generator=generator)
    context = _context(effort=Effort.HIGH)

    result = await reviewer.review_tool_calls(
        context,
        reasoning="I should look up the price.",
        tool_calls=_tool_calls(),
    )

    router = cast(_FakeModelRouter, reviewer._model_router)
    assert len(generator.hints) == 1
    assert router.generator_requests[0]["route"] == "review.tools"
    assert router.generator_requests[0]["qualifiers"] == {"effort": "high"}
    assert generator.hints[0]["hedge_timeout"] == 15.0
    assert result is not None
    assert result.adjusted_reasoning is None


@pytest.mark.asyncio
async def test_that_the_basic_reviewer_uses_the_reasoning_effort_from_its_model_route() -> None:
    generator = _compliant_basic_generator()
    reviewer = _reviewer(
        basic_schematic_generator=generator,
        reasoning_effort="high",
    )

    await reviewer.review_tool_calls(
        _context(effort=Effort.LOW),
        reasoning="I should look up the price.",
        tool_calls=_tool_calls(),
    )

    assert generator.hints[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_that_the_basic_reviewer_passes_a_compliant_message() -> None:
    generator = _compliant_basic_generator()
    reviewer = _reviewer(generator, reasoning_effort="medium")

    result = await reviewer.review_message(_context(), "Sure, I can help with that.")

    router = cast(_FakeModelRouter, reviewer._model_router)
    assert len(generator.hints) == 1
    assert router.generator_requests[0]["route"] == "review.message"
    assert router.generator_requests[0]["qualifiers"] == {"effort": "low"}
    assert generator.hints[0]["hedge_timeout"] == 10.0
    assert generator.hints[0]["reasoning_effort"] == "medium"
    assert result is not None
    assert result.adjusted_reasoning is None


@pytest.mark.asyncio
async def test_that_the_basic_reviewer_adjusts_a_breaching_message() -> None:
    generator = _ScriptedReviewGenerator(
        ReviewSchema(
            breaches_or_discrepancies=True,
            adjusted_reasoning="Decline politely per policy.",
        )
    )
    reviewer = _reviewer(generator)

    result = await reviewer.review_message(_context(), "Yes, here is your neighbor's balance.")

    assert result is not None
    assert result.adjusted_reasoning == "Decline politely per policy."


def _rule_match(rule_id: str = "rule-1") -> Any:
    from datetime import datetime, timezone

    from parlant.core.common import Weight
    from parlant.core.engines.rule_evaluation import RuleEvaluation
    from parlant.core.rules import Rule, RuleId

    now = datetime.now(timezone.utc)
    rule = Rule(
        id=RuleId(rule_id),
        creation_utc=now,
        modified_utc=now,
        condition="c",
        content="a",
        enabled=True,
        groups=[],
        metadata={},
        weight=Weight.MEDIUM,
    )
    return RuleEvaluation(rule=rule, reasoning="matched", metadata={})


def _canned_response(value: str, note: str = "") -> Any:
    cr = CannedResponse.create_transient(value=value)
    return cast(Any, SimpleNamespace(**{**cr.__dict__, "note": note}))


@pytest.mark.asyncio
async def test_that_message_review_adds_relevant_canned_responses_as_guidance() -> None:
    generator = _compliant_basic_generator()
    canrep = _canned_response("Would you like a banana?", note="Always offer a banana.")
    entity_queries = _StubEntityQueries(TurnCannedResponses(rule_scoped=[canrep], agent_wide=[]))
    canned_store = _StubCannedResponseStore(
        [CannedResponseRelevanceResult(canned_response=canrep, score=0.9)]
    )
    reviewer = _reviewer(
        generator,
        entity_queries=entity_queries,
        canned_response_store=canned_store,
    )
    context = _context()
    context.state.ordinary_rule_matches = [_rule_match()]

    result = await reviewer.review_message(context, "Do you want an apple?")

    assert len(generator.prompts) == 1
    assert result is not None

    prompt = generator.prompts[0].build()
    # The policy audit is always present...
    assert "breaches any governing instruction or policy" in prompt
    # ...and the rendered canned response + its Note reach the same prompt as
    # conformance guidance.
    assert "Would you like a banana?" in prompt
    assert "Always offer a banana." in prompt
    assert "approved canned responses" in prompt


@pytest.mark.asyncio
async def test_that_message_review_omits_canned_guidance_when_none_are_relevant() -> None:
    generator = _compliant_basic_generator()
    reviewer = _reviewer(generator)
    context = _context()
    context.state.ordinary_rule_matches = [_rule_match()]

    result = await reviewer.review_message(context, "Do you want an apple?")

    # No canned responses anywhere: the review must still run (never decline),
    # auditing the governing policies.
    assert len(generator.prompts) == 1
    assert result is not None

    prompt = generator.prompts[0].build()
    assert "breaches any governing instruction or policy" in prompt
    assert "approved canned responses" not in prompt


def test_that_the_message_review_prompt_contains_the_message_under_review() -> None:
    prompt = (
        _reviewer()
        ._build_message_review_prompt(
            _context(),
            "The specific message being reviewed.",
            task_description="TASK",
            output_format="{}",
            additional_guidance=None,
        )
        .build()
    )

    assert "The specific message being reviewed." in prompt
    assert "MESSAGE TO REVIEW" in prompt


def test_that_the_message_review_prompt_requires_turn_directive_adherence() -> None:
    context = _context()
    context.state.turn_directives = [
        TurnDirective(
            instruction="Tell the customer the booking is confirmed.",
            rationale="Follow up with the customer.",
        )
    ]

    prompt = (
        _reviewer()
        ._build_message_review_prompt(
            context,
            "Your booking is still pending.",
            task_description="TASK",
            output_format="{}",
            additional_guidance=None,
        )
        .build()
    )

    assert "REQUIRED DIRECTIVES FOR THIS RESPONSE" in prompt
    assert "booking is confirmed" in prompt
    assert "omitting or contradicting a directive" in prompt
    assert context.state.message_review_requested is False


# --- Unsupported action claims -------------------------------------------------
#
# A message that claims an action happened (cancelled, sent, updated, paid...)
# must be backed by an actual tool result — in this turn's STAGED EVENTS or in
# the interaction history. The reviewer's prompt must carry this rule
# explicitly, and must be shown the staged tool events so it can tell a backed
# claim from a fabricated one.


_CLAIM_MESSAGE = "Done! I've cancelled your subscription, effective immediately."


def _cancellation_context() -> EngineContext:
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "Please cancel my subscription.")]
    )
    context.state = ResponseState(agent_effort=Effort.LOW)
    return context


def _staged_cancellation_result() -> EmittedEvent:
    return create_staged_tool_event(
        cast(
            JSONSerializable,
            {
                "tool_calls": [
                    {
                        "tool_id": "local:cancel_subscription",
                        "arguments": {"customer_id": "199877"},
                        "result": {
                            "data": {"status": "cancelled"},
                            "metadata": {},
                            "control": {},
                        },
                    }
                ]
            },
        )
    )


def test_that_the_message_review_task_forbids_unsupported_action_claims() -> None:
    task = _reviewer()._message_review_task_description()

    assert "Unsupported action claims" in task
    assert "STAGED EVENTS" in task


def test_that_the_message_review_prompt_includes_staged_tool_events() -> None:
    context = _cancellation_context()
    context.state.tool_events = [_staged_cancellation_result()]

    prompt = (
        _reviewer()
        ._build_message_review_prompt(
            context,
            _CLAIM_MESSAGE,
            task_description=_reviewer()._message_review_task_description(),
            output_format="{}",
            additional_guidance=None,
        )
        .build()
    )

    # The rule references the STAGED EVENTS section by name; the section must
    # actually exist in the same prompt, carrying this turn's tool results.
    assert prompt.count("STAGED EVENTS") >= 2
    assert "cancel_subscription" in prompt


def _container_reviewer(container: Container) -> BasicReviewer:
    return BasicReviewer(
        logger=container[Logger],
        tracer=container[Tracer],
        model_router=container[ModelRouter],
        prompt_config=PromptConfigurationService(),
        entity_queries=container[EntityQueries],
        canned_response_store=container[CannedResponseStore],
    )


@pytest.mark.asyncio
async def test_that_a_message_claiming_an_action_without_a_tool_result_is_flagged(
    container: Container,
) -> None:
    reviewer = _container_reviewer(container)
    context = _cancellation_context()

    result = await reviewer.review_message(context, _CLAIM_MESSAGE)

    assert result is not None
    assert result.adjusted_reasoning is not None


@pytest.mark.asyncio
async def test_that_a_message_claiming_a_tool_backed_action_passes(
    container: Container,
) -> None:
    reviewer = _container_reviewer(container)
    context = _cancellation_context()
    context.state.tool_events = [_staged_cancellation_result()]

    result = await reviewer.review_message(context, _CLAIM_MESSAGE)

    assert result is not None
    assert result.adjusted_reasoning is None


def test_that_statically_included_rules_appear_in_the_review_prompt() -> None:
    # The reviewer audits messages against the same standing instructions the
    # responder saw — including bare cross-referential rules, which never
    # match but always render. See matching/common.py: "Static inclusion".
    from dataclasses import replace as _replace

    from tests.core.stable.engines.compass.matching.utils import create_rule

    reviewer = _reviewer()
    context = _context()
    bare = _replace(
        create_rule(
            condition="ZZQ_BARE the request is not covered by any of our policies",
            action="say you cannot help",
        ),
        metadata={"references_other_policies": True},
    )
    context.state.statically_included_rules = [bare]

    prompt = reviewer._build_message_review_prompt(
        context,
        "Some drafted message.",
        task_description="TASK",
        output_format="{}",
        additional_guidance=None,
    ).build()

    assert "ZZQ_BARE" in prompt


def test_that_a_low_weight_statically_included_rule_appears_in_the_review_prompt() -> None:
    from dataclasses import replace as _replace

    from parlant.core.common import Weight

    from tests.core.stable.engines.compass.matching.utils import create_rule

    reviewer = _reviewer()
    context = _context()
    bare = _replace(
        create_rule(
            condition="ZZQ_LOW_BARE the request is not covered by any of our policies",
            action="say you cannot help",
        ),
        weight=Weight.LOW,
        metadata={"references_other_policies": True},
    )
    context.state.statically_included_rules = [bare]

    prompt = reviewer._build_message_review_prompt(
        context,
        "Some drafted message.",
        task_description="TASK",
        output_format="{}",
        additional_guidance=None,
    ).build()

    assert "ZZQ_LOW_BARE" in prompt


class _EventRecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_events: list[tuple[str, dict[str, Any]]] = []

    def add_event(self, name: str, attributes: Any = {}) -> None:
        self.recorded_events.append((name, dict(attributes)))


async def test_that_a_passing_tool_review_emits_a_correlated_lifecycle() -> None:
    tracer = _EventRecordingTracer()
    reviewer = _reviewer(_compliant_basic_generator(), tracer=tracer)

    result = await reviewer.review_tool_calls(
        _context(effort=Effort.HIGH), "checking the price", _tool_calls()
    )
    assert result is not None

    started = [a for n, a in tracer.recorded_events if n == "review.started"]
    passed = [a for n, a in tracer.recorded_events if n == "review.passed"]
    reviewed = [a for n, a in tracer.recorded_events if n == "tool.reviewed"]

    assert len(started) == 1
    assert len(passed) == 1
    assert started[0]["review_ref"] == passed[0]["review_ref"]
    assert started[0]["tool_names"] == ["get_fruit_price"]

    assert len(reviewed) == 1
    assert reviewed[0]["tool_call_id"] == "call-1"
    assert reviewed[0]["approved"] is True
    assert not any(n == "review.rejected" for n, _ in tracer.recorded_events)


async def test_that_a_rejecting_tool_review_emits_a_rejected_lifecycle() -> None:
    tracer = _EventRecordingTracer()
    generator = _ScriptedReviewGenerator(
        ReviewSchema(
            breaches_or_discrepancies=True,
            adjusted_reasoning="Ask for confirmation before calling the tool.",
        )
    )
    reviewer = _reviewer(generator, tracer=tracer)

    result = await reviewer.review_tool_calls(
        _context(effort=Effort.HIGH), "calling immediately", _tool_calls()
    )
    assert result is not None

    started = [a for n, a in tracer.recorded_events if n == "review.started"]
    rejected = [a for n, a in tracer.recorded_events if n == "review.rejected"]
    reviewed = [a for n, a in tracer.recorded_events if n == "tool.reviewed"]

    assert len(started) == 1
    assert len(rejected) == 1
    assert started[0]["review_ref"] == rejected[0]["review_ref"]
    assert rejected[0]["has_adjusted_reasoning"] is True

    assert len(reviewed) == 1
    assert reviewed[0]["approved"] is False
    assert not any(n == "review.passed" for n, _ in tracer.recorded_events)
