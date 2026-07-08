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
from types import SimpleNamespace
from typing import Any, Mapping, cast
from unittest.mock import AsyncMock

import pytest

from parlant.core.agents import Effort
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.engines.compass.reviewer import (
    BasicReviewer,
    LowEffortReviewSchema,
    Reviewer,
)
from parlant.core.engines.compass.tracing import format_json_attr
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.generation import SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.react import ToolCallPart
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    RecordedEvent,
    RecordedSpan,
    RecordingTracer,
    create_engine_context,
)


def _reviewer(
    low_effort_schematic_generator: Any = None,
    *,
    tracer: LocalTracer | None = None,
) -> BasicReviewer:
    tracer = tracer or LocalTracer()
    return BasicReviewer(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        low_effort_schematic_generator=cast(Any, low_effort_schematic_generator),
    )


def _context(effort: Effort = Effort.LOW) -> EngineContext:
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "How much is eggplant today?")]
    )
    context.state = ResponseState(agent_effort=effort)
    return context


def _tool_calls() -> list[ToolCallPart]:
    return [ToolCallPart(id="call-1", name="get_fruit_price", args={"item": "eggplant"})]


def _compliant_low_effort_generator() -> AsyncMock:
    generator = AsyncMock()
    generator.generate = AsyncMock(
        return_value=SimpleNamespace(
            content=LowEffortReviewSchema(breaches_or_discrepancies=False),
            info=None,
        )
    )
    return generator


def _generation_info() -> GenerationInfo:
    return GenerationInfo(
        schema_name="LowEffortReviewSchema",
        model="fake-model",
        duration=0.01,
        usage=UsageInfo(input_tokens=1, output_tokens=1),
    )


class _LowEffortReviewGenerator:
    def __init__(
        self,
        *,
        breaches_or_discrepancies: bool | None,
        adjusted_reasoning: str | None,
    ) -> None:
        self._breaches_or_discrepancies = breaches_or_discrepancies
        self._adjusted_reasoning = adjusted_reasoning

    async def generate(
        self,
        prompt: Any,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[LowEffortReviewSchema]:
        return SchematicGenerationResult(
            content=LowEffortReviewSchema(
                breaches_or_discrepancies=self._breaches_or_discrepancies,
                adjusted_reasoning=self._adjusted_reasoning,
            ),
            info=_generation_info(),
        )


def test_that_the_basic_reviewer_implements_the_reviewer_port() -> None:
    assert isinstance(_reviewer(), Reviewer)


def test_that_low_effort_reviewer_prompt_uses_boolean_discrepancy_examples() -> None:
    context = _context()

    prompt = (
        _reviewer()
        ._build_low_effort_prompt(
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
async def test_that_the_basic_reviewer_reviews_with_the_low_effort_schema_at_high_effort() -> None:
    generator = _compliant_low_effort_generator()
    reviewer = _reviewer(low_effort_schematic_generator=generator)
    context = _context(effort=Effort.HIGH)

    result = await reviewer.review_tool_calls(
        context,
        reasoning="I should look up the price.",
        tool_calls=_tool_calls(),
    )

    generator.generate.assert_awaited_once()
    assert result is not None
    assert result.adjusted_reasoning is None


@pytest.mark.asyncio
async def test_that_tool_review_records_passed_events() -> None:
    tracer = RecordingTracer()
    reviewer = _reviewer(
        _LowEffortReviewGenerator(
            breaches_or_discrepancies=False,
            adjusted_reasoning=None,
        ),
        tracer=tracer,
    )
    context = replace(
        create_engine_context(conversation=[(EventSource.CUSTOMER, "Please look this up")]),
        tracer=tracer,
    )
    context.state = ResponseState()
    tool_call = ToolCallPart(id="call-1", name="lookup_account", args={"account_id": "A-1"})

    await reviewer.review_tool_calls(context, "I should look up the account.", [tool_call])

    assert tracer.started_spans == [RecordedSpan(name="tools.review", attributes={})]
    assert tracer.events == [
        RecordedEvent(
            name="review.passed",
            attributes={
                "tool_count": 1,
                "tool_names": ["lookup_account"],
            },
            span_id="tools.review",
        ),
        RecordedEvent(
            name="tool.reviewed",
            attributes={
                "tool_call_id": "call-1",
                "tool_name": "lookup_account",
                "arguments": format_json_attr({"account_id": "A-1"}),
                "review_status": "passed",
            },
            span_id="tools.review",
        ),
    ]


@pytest.mark.asyncio
async def test_that_tool_review_records_rejected_events() -> None:
    tracer = RecordingTracer()
    reviewer = _reviewer(
        _LowEffortReviewGenerator(
            breaches_or_discrepancies=True,
            adjusted_reasoning="Ask the customer for the account ID first.",
        ),
        tracer=tracer,
    )
    context = replace(
        create_engine_context(conversation=[(EventSource.CUSTOMER, "Please update my account")]),
        tracer=tracer,
    )
    context.state = ResponseState()
    tool_call = ToolCallPart(id="call-1", name="update_account", args={"account_id": "guessed"})

    await reviewer.review_tool_calls(context, "I should update the account.", [tool_call])

    assert tracer.events == [
        RecordedEvent(
            name="review.rejected",
            attributes={
                "tool_count": 1,
                "tool_names": ["update_account"],
                "todo": "",
                "adjusted_reasoning": "Ask the customer for the account ID first.",
            },
            span_id="tools.review",
        ),
        RecordedEvent(
            name="tool.reviewed",
            attributes={
                "tool_call_id": "call-1",
                "tool_name": "update_account",
                "arguments": format_json_attr({"account_id": "guessed"}),
                "review_status": "rejected",
                "todo": "",
                "adjusted_reasoning": "Ask the customer for the account ID first.",
            },
            span_id="tools.review",
        ),
    ]
