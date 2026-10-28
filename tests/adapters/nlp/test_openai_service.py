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

"""Tests for the OpenAI adapter (OpenAIReactGenerator) over the Responses API.

The provider-agnostic ReAct orchestration (TurnBuilder, step/stream_step/run,
usage, history editing, cancellation, schema building) is exercised in
test_gemini_service.py via a scripted fake provider — it tests react.py itself,
so it is not duplicated here. This file covers:

1. OpenAI request-building (`_encode`) and the pure `_map_finish_reason`
   transform, offline. No API response is fabricated.
2. Live integration tests against gpt-5.4-nano, hitting the real OpenAI
   Responses API, skipped unless OPENAI_API_KEY is set.

Together with the Gemini suite, these prove the same react.py interface backs
two very different providers with no interface changes and no abstraction leaks.
"""

import asyncio
import json
import os
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
import tiktoken
from openai import AsyncClient

from parlant.adapters.nlp.openai_service import (
    DEFAULT_DECISION_MODEL,
    OPENAI_ITEM_KEY,
    OpenAIDecisionMaker,
    OpenAIEstimatingTokenizer,
    OpenAIReactGenerator,
    OpenAITextEmbedding3Small,
    register_openai_models,
)
from parlant.core.nlp.decision import (
    Choice,
    ChoiceAnswer,
    DecisionError,
    Proposition,
    PropositionAnswer,
    PropositionCriteria,
    Scale,
    ScaleAnswer,
)
from parlant.core.container import Container
from parlant.core.nlp.common import ModelSize
from parlant.core.nlp.model_routing import (
    ModelCapability,
    ModelImplementationRegistry,
    ModelProvider,
    NullSecretLoader,
)
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import LocalMeter
from parlant.core.tracer import LocalTracer
from tests.test_utilities import salted_cacheable_prefix
from parlant.core.nlp.react import (
    CacheConfig,
    FinishReason,
    Message,
    ParameterSpec,
    ReasoningConfig,
    ReasoningPart,
    Role,
    StepCompleted,
    StepResult,
    TextDelta,
    TextPart,
    ReasoningDelta,
    ToolCallPart,
    ToolCallStarted,
    ToolResultPart,
    ToolSpec,
    Usage,
)


# ───────────────────────────── fixtures & helpers ──────────────────────────


@pytest.fixture
def logger() -> Logger:
    return StdoutLogger(LocalTracer())


WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters=[ParameterSpec(name="city", type="string", description="The city name")],
)


@pytest.fixture
def openai(logger: Logger) -> OpenAIReactGenerator:
    # _encode and _map_finish_reason are pure transforms that never call the
    # client, so a client built with a throwaway key keeps these tests offline
    # without mocking any API response.
    return OpenAIReactGenerator(
        model="gpt-5.4-nano",
        logger=logger,
        client=AsyncClient(api_key="offline-encode-tests"),
    )


# ════════════════════════ 1. OPENAI REQUEST BUILDING ════════════════════════


def test_that_encode_maps_roles_and_folds_system_into_instructions(
    openai: OpenAIReactGenerator,
) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="You are a test agent.")]),
        Message(role=Role.SYSTEM, parts=[TextPart(text="Extra system rule.")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(
            role=Role.ASSISTANT,
            parts=[
                TextPart(text="hello"),
                ToolCallPart(id="call_1", name="get_weather", args={"city": "Paris"}),
            ],
        ),
        Message(
            role=Role.TOOL,
            parts=[ToolResultPart(call_id="call_1", name="get_weather", content="sunny")],
        ),
    ]

    request = openai._encode(history, [], "auto", reasoning=ReasoningConfig())

    assert "You are a test agent." in request["instructions"]
    assert "Extra system rule." in request["instructions"]

    items = request["input"]
    # user message, assistant message, function_call, function_call_output
    assert items[0]["role"] == "user"
    assert items[0]["content"] == [{"type": "input_text", "text": "hi"}]
    assert items[1]["role"] == "assistant"
    assert items[1]["content"] == [{"type": "output_text", "text": "hello"}]
    assert items[2]["type"] == "function_call"
    assert items[3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "sunny",
    }


def test_that_encode_replays_raw_output_items_verbatim(openai: OpenAIReactGenerator) -> None:
    raw_reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}
    raw_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }
    # A reasoning part and a tool-call part, each carrying their raw OpenAI output
    # item (the analog of Gemini's thought_signature) under provider_data.
    history = [
        Message(
            role=Role.ASSISTANT,
            parts=[
                ReasoningPart(text="", provider_data={OPENAI_ITEM_KEY: raw_reasoning}),
                ToolCallPart(
                    id="call_1",
                    name="get_weather",
                    args={"city": "Paris"},
                    provider_data={OPENAI_ITEM_KEY: raw_call},
                ),
            ],
        )
    ]

    items = openai._encode(history, [WEATHER_TOOL], "auto", reasoning=ReasoningConfig())["input"]

    # Both items are replayed exactly as received.
    assert items[0] == raw_reasoning
    assert items[1] == raw_call


def test_that_encode_reconstructs_tool_calls_without_a_raw_item(
    openai: OpenAIReactGenerator,
) -> None:
    history = [
        Message(
            role=Role.ASSISTANT,
            parts=[ToolCallPart(id="call_9", name="get_weather", args={"city": "台北"})],
        )
    ]

    item = openai._encode(history, [WEATHER_TOOL], "auto", reasoning=ReasoningConfig())["input"][0]

    assert item == {
        "type": "function_call",
        "call_id": "call_9",
        "name": "get_weather",
        "arguments": '{"city": "台北"}',
    }


def test_that_encode_serializes_non_string_tool_results(openai: OpenAIReactGenerator) -> None:
    history = [
        Message(
            role=Role.ASSISTANT,
            parts=[
                ToolCallPart(id="c1", name="t", args={}),
                ToolCallPart(id="c2", name="t", args={}),
            ],
        ),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(call_id="c1", name="t", content="plain string"),
                ToolResultPart(call_id="c2", name="t", content={"status": "成功"}),
            ],
        ),
    ]

    items = openai._encode(history, [], "auto", reasoning=ReasoningConfig())["input"]

    assert items[2]["output"] == "plain string"
    assert items[3]["output"] == '{"status": "成功"}'  # JSON-encoded without ASCII escapes


def test_that_encode_sends_a_tool_result_without_a_matching_call_as_a_paired_call(
    openai: OpenAIReactGenerator,
) -> None:
    # The Responses API rejects a function_call_output without its function_call.
    # Staged retriever results reach history as results with no call (see
    # base_loop.build_result_only_tool_event_messages), each numbered from "1".
    history = [
        Message(role=Role.USER, parts=[TextPart(text="What are chongas?")]),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(
                    call_id="1",
                    name="built-in:rule-retriever-colour",
                    content="Chongas are bright purple",
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(
                    call_id="1",
                    name="built-in:rule-retriever-taste",
                    content="Chongas taste of mango",
                )
            ],
        ),
    ]

    items = openai._encode(history, [], "auto", reasoning=ReasoningConfig())["input"]

    calls = [item for item in items if item.get("type") == "function_call"]
    outputs = [item for item in items if item.get("type") == "function_call_output"]
    assert len(calls) == len(outputs) == 2
    assert len({call["call_id"] for call in calls}) == 2
    for call, output in zip(calls, outputs):
        assert items.index(call) < items.index(output)
        assert call["call_id"] == output["call_id"]
        assert re.fullmatch(r"[A-Za-z0-9_-]+", call["name"])
    assert "Chongas are bright purple" in outputs[0]["output"]
    assert "built-in:rule-retriever-colour" in outputs[0]["output"]
    assert "Chongas taste of mango" in outputs[1]["output"]
    assert not any(item.get("role") == "user" and "Chongas" in json.dumps(item) for item in items)


def test_that_encode_keeps_a_tool_result_that_answers_its_call_as_function_call_output(
    openai: OpenAIReactGenerator,
) -> None:
    history = [
        Message(role=Role.USER, parts=[TextPart(text="Weather in Paris?")]),
        Message(
            role=Role.ASSISTANT,
            parts=[ToolCallPart(id="call_1", name="get_weather", args={"city": "Paris"})],
        ),
        Message(
            role=Role.TOOL,
            parts=[ToolResultPart(call_id="call_1", name="get_weather", content="sunny")],
        ),
    ]

    items = openai._encode(history, [WEATHER_TOOL], "auto", reasoning=ReasoningConfig())["input"]

    assert items[-1] == {"type": "function_call_output", "call_id": "call_1", "output": "sunny"}


@pytest.mark.parametrize(
    "tool_choice, expected",
    [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
        ({"name": "get_weather"}, {"type": "function", "name": "get_weather"}),
    ],
)
def test_that_encode_maps_tool_choice(
    openai: OpenAIReactGenerator, tool_choice: Any, expected: Any
) -> None:
    request = openai._encode([], [WEATHER_TOOL], tool_choice, reasoning=ReasoningConfig())
    assert request["tool_choice"] == expected


def test_that_encode_tool_translates_nullable_to_json_schema_null_type(
    openai: OpenAIReactGenerator,
) -> None:
    # OpenAI ignores OpenAPI's `"nullable": true`; nullability must be a "null"
    # member of `type`. The adapter translates ToolSpec.json_schema() accordingly.
    tool = ToolSpec(
        name="set_value",
        description="d",
        parameters=[
            ParameterSpec(name="value", type="string", nullable=True, default="hi"),
            ParameterSpec(
                name="tags",
                type="array",
                items=ParameterSpec(name="tag", type="string", nullable=True),
            ),
        ],
    )

    properties = openai._encode_tool(tool)["parameters"]["properties"]

    assert properties["value"]["type"] == ["string", "null"]
    assert "nullable" not in properties["value"]
    assert properties["value"]["default"] == "hi"  # standard JSON Schema, kept as-is
    # Translation recurses into array items.
    assert properties["tags"]["items"]["type"] == ["string", "null"]
    assert "nullable" not in properties["tags"]["items"]


def test_that_encode_always_emits_reasoning_with_effort_mapped_per_model(
    openai: OpenAIReactGenerator,
    logger: Logger,
) -> None:
    # The adapter always emits the reasoning block — depth is controlled by
    # effort, and the encrypted_content include is needed for cross-step replay.
    # gpt-5.1+ (the fixture is gpt-5.4-nano) dropped "minimal" for "none".
    minimal = openai._encode([], [], "auto", reasoning=ReasoningConfig(effort="minimal"))
    assert minimal["reasoning"] == {"effort": "none", "summary": "auto"}
    assert minimal["include"] == ["reasoning.encrypted_content"]

    # The original gpt-5 family still exposes the native "minimal" tier.
    gpt5 = OpenAIReactGenerator(
        model="gpt-5",
        logger=logger,
        client=AsyncClient(api_key="offline-encode-tests"),
    )
    base = gpt5._encode([], [], "auto", reasoning=ReasoningConfig(effort="minimal"))
    assert base["reasoning"] == {"effort": "minimal", "summary": "auto"}

    high = openai._encode(
        [],
        [],
        "auto",
        reasoning=ReasoningConfig(effort="high", visibility="full"),
    )
    assert high["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert high["include"] == ["reasoning.encrypted_content"]


def test_that_encode_omits_summary_when_visibility_is_none(openai: OpenAIReactGenerator) -> None:
    request = openai._encode(
        [],
        [],
        "auto",
        reasoning=ReasoningConfig(effort="low", visibility="none"),
    )
    assert request["reasoning"] == {"effort": "low"}


def test_that_service_tier_maps_to_openai_values(openai: OpenAIReactGenerator) -> None:
    def tier_for(service_tier: str | None) -> str:
        hints = {"service_tier": service_tier} if service_tier else {}
        request = openai._encode([], [], "auto", reasoning=ReasoningConfig(), hints=hints)  # type: ignore[arg-type]
        encoded_tier: str = request["service_tier"]
        return encoded_tier

    assert tier_for(None) == "default"  # default is "standard" -> "default"
    assert tier_for("standard") == "default"
    assert tier_for("flex") == "flex"
    # OpenAI renamed the priority tier to "fast"; the seam keeps "priority".
    assert tier_for("priority") == "fast"


def test_that_a_route_level_service_tier_is_the_request_default_and_hints_override_it(
    logger: Logger,
) -> None:
    generator = OpenAIReactGenerator(
        model="gpt-5.4-nano",
        logger=logger,
        client=AsyncClient(api_key="offline-encode-tests"),
        service_tier="flex",
    )

    request = generator._encode([], [], "auto", reasoning=ReasoningConfig())
    assert request["service_tier"] == "flex"

    overridden = generator._encode(
        [], [], "auto", reasoning=ReasoningConfig(), hints={"service_tier": "priority"}
    )
    assert overridden["service_tier"] == "fast"


def test_that_service_tier_values_are_validated_with_helpful_errors() -> None:
    from parlant.adapters.nlp.common import validate_service_tier
    from parlant.core.nlp.model_routing import InvalidModelArgumentsError
    from pytest import raises

    assert validate_service_tier("openai", "flex") == "flex"
    assert validate_service_tier("openai", "priority") == "priority"

    with raises(InvalidModelArgumentsError) as exc_info:
        validate_service_tier("openai", "fast")
    assert "priority" in str(exc_info.value)  # points at the seam's name for it

    with raises(InvalidModelArgumentsError) as exc_info:
        validate_service_tier("openai", "turbo")
    assert "standard" in str(exc_info.value)
    assert "flex" in str(exc_info.value)
    assert "priority" in str(exc_info.value)


def test_that_prefill_request_appends_a_dummy_input_and_caps_output(
    openai: OpenAIReactGenerator,
) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="sys")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(role=Role.ASSISTANT, parts=[TextPart(text="hello")]),
    ]
    encoded = openai._encode(history, [], "auto", reasoning=ReasoningConfig())
    prefill = openai._build_prefill_request(encoded)

    assert prefill["max_output_tokens"] == 16
    assert "reasoning" not in prefill
    assert prefill["instructions"] == encoded["instructions"]
    # The original input prefix is preserved, with a dummy user item appended.
    assert prefill["input"][: len(encoded["input"])] == encoded["input"]
    assert prefill["input"][-1] == {"role": "user", "content": "."}


def test_that_min_cache_size_is_1024_for_known_models(openai: OpenAIReactGenerator) -> None:
    assert openai._min_cache_size("gpt-5.4-mini") == 1024
    assert openai._min_cache_size("o3") == 1024
    assert openai._min_cache_size("some-unknown-model") > 2048


async def test_that_a_short_prefix_is_not_prefilled(openai: OpenAIReactGenerator) -> None:
    # A tiny prompt is well below the 1024-token cache minimum: counting its
    # tokens (locally, via tiktoken) must decide against prefilling.
    history = [Message(role=Role.SYSTEM, parts=[TextPart(text="be concise")])]

    assert await openai._should_prefill(history, [], {}) is False


async def test_that_a_prefix_above_the_cache_minimum_is_prefilled(
    openai: OpenAIReactGenerator,
) -> None:
    # A prompt comfortably past 1024 tokens crosses the cache minimum.
    history = [Message(role=Role.SYSTEM, parts=[TextPart(text="word " * 4000)])]

    assert await openai._should_prefill(history, [], {}) is True


def test_that_a_leading_system_message_becomes_instructions(openai: OpenAIReactGenerator) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="main")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
    ]

    request = openai._encode(history, [], "auto", reasoning=ReasoningConfig())

    assert request["instructions"] == "main"
    assert all(item.get("role") != "developer" for item in request["input"])


def test_that_a_mid_conversation_system_message_becomes_an_inline_developer_turn(
    openai: OpenAIReactGenerator,
) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="main")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(role=Role.SYSTEM, parts=[TextPart(text="mid")]),
    ]

    request = openai._encode(history, [], "auto", reasoning=ReasoningConfig())

    # The leading system stays the top-level instructions...
    assert request["instructions"] == "main"
    # ...and the mid-conversation system is an inline developer turn at its spot.
    assert request["input"][-1] == {"role": "developer", "content": "mid"}


def test_that_cache_key_becomes_prompt_cache_key(openai: OpenAIReactGenerator) -> None:
    history = [
        Message(role=Role.USER, parts=[TextPart(text="ctx")], cache_key="agent-7"),
        Message(role=Role.USER, parts=[TextPart(text="q")]),
    ]
    request = openai._encode(history, [], "auto", reasoning=ReasoningConfig())
    assert request["prompt_cache_key"] == "agent-7"


def test_that_prompt_cache_key_uses_the_first_marked_message(openai: OpenAIReactGenerator) -> None:
    # OpenAI takes a single prompt_cache_key; the FIRST marker wins (the most
    # stable prefix boundary), and a marked system message counts.
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="big stable system")], cache_key="sys-v1"),
        Message(role=Role.USER, parts=[TextPart(text="ctx")], cache_key="later"),
        Message(role=Role.USER, parts=[TextPart(text="q")]),
    ]
    request = openai._encode(history, [], "auto", reasoning=ReasoningConfig())

    # The system message's text still folds into instructions...
    assert "big stable system" in request["instructions"]
    # ...and its key is used (first-wins), not the later message's.
    assert request["prompt_cache_key"] == "sys-v1"


def test_that_caching_disabled_omits_prompt_cache_key(logger: Logger) -> None:
    generator = OpenAIReactGenerator(
        logger=logger,
        cache=CacheConfig(enabled=False),
        client=AsyncClient(api_key="offline"),
    )
    history = [Message(role=Role.USER, parts=[TextPart(text="q")], cache_key="k")]
    request = generator._encode(history, [], "auto", reasoning=ReasoningConfig())
    assert "prompt_cache_key" not in request


def test_that_finish_reason_mapping_covers_response_statuses(
    openai: OpenAIReactGenerator,
) -> None:
    def response(status: str, reason: str | None = None) -> Any:
        details = SimpleNamespace(reason=reason) if reason else None
        return SimpleNamespace(status=status, incomplete_details=details)

    assert openai._map_finish_reason(response("completed")) == FinishReason.STOP
    assert openai._map_finish_reason(response("failed")) == FinishReason.ERROR
    assert (
        openai._map_finish_reason(response("incomplete", "max_output_tokens"))
        == FinishReason.MAX_TOKENS
    )
    assert (
        openai._map_finish_reason(response("incomplete", "content_filter"))
        == FinishReason.CONTENT_FILTER
    )
    # Unknown/other states degrade to STOP.
    assert openai._map_finish_reason(response("incomplete", "something_else")) == FinishReason.STOP


# ════════════════════════════ 2. LIVE INTEGRATION ══════════════════════════

LIVE = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live OpenAI integration tests",
)
LIVE_MODEL = "gpt-5.4-nano"


def _live_generator(logger: Logger, **kwargs: Any) -> OpenAIReactGenerator:
    return OpenAIReactGenerator(model=LIVE_MODEL, logger=logger, **kwargs)


@LIVE
async def test_that_live_openai_generates_text(logger: Logger) -> None:
    generator = _live_generator(logger)

    result = await generator.step(
        [
            Message(
                role=Role.SYSTEM, parts=[TextPart(text="Answer in exactly one short sentence.")]
            ),
            Message(role=Role.USER, parts=[TextPart(text="What is the capital of France?")]),
        ]
    )

    assert result.finish_reason == FinishReason.STOP
    assert "Paris" in result.message.text
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


@LIVE
async def test_that_live_openai_streams_text_deltas(logger: Logger) -> None:
    generator = _live_generator(logger)

    text_deltas: list[str] = []
    completed: StepResult | None = None
    async for event in generator.stream_step(
        [
            Message(role=Role.SYSTEM, parts=[TextPart(text="Be concise.")]),
            Message(role=Role.USER, parts=[TextPart(text="List three primary colors.")]),
        ]
    ):
        if isinstance(event, TextDelta):
            text_deltas.append(event.text)
        elif isinstance(event, StepCompleted):
            completed = event.result

    assert text_deltas
    assert completed is not None
    assert completed.message.text == "".join(text_deltas)


@LIVE
async def test_that_live_openai_calls_a_tool(logger: Logger) -> None:
    generator = _live_generator(logger)

    result = await generator.step(
        [
            Message(
                role=Role.USER, parts=[TextPart(text="Use the tool to get the weather in Paris.")]
            )
        ],
        [WEATHER_TOOL],
        tool_choice="auto",
    )

    assert result.needs_tools
    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert call.args.get("city", "").lower().startswith("paris")
    assert call.id.startswith("call_")
    # The raw function_call item is preserved for stateless replay.
    assert call.provider_data.get(OPENAI_ITEM_KEY)


@LIVE
async def test_that_live_openai_runs_a_full_react_loop(logger: Logger) -> None:
    generator = _live_generator(logger)

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        return ToolResultPart(
            call_id=call.id,
            name=call.name,
            content={"temperature_c": 18, "conditions": "sunny"},
        )

    step_usages: list[Usage] = []

    async def on_step(result: StepResult, history: list[Message]) -> None:
        step_usages.append(result.usage)

    history: list[Message] = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Use tools when needed, then answer.")]),
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris right now?")]),
    ]
    final_history = await generator.run(
        history,
        [WEATHER_TOOL],
        dispatch,
        on_step=on_step,
        max_steps=5,
    )

    assert any(m.role == Role.TOOL for m in final_history)
    assert final_history[-1].role == Role.ASSISTANT
    assert "18" in final_history[-1].text or "sunny" in final_history[-1].text.lower()
    assert sum((u.input_tokens for u in step_usages), 0) > 0


@LIVE
async def test_that_live_openai_makes_parallel_tool_calls(logger: Logger) -> None:
    generator = _live_generator(logger)

    result = await generator.step(
        [
            Message(
                role=Role.USER,
                parts=[TextPart(text="Get the weather for BOTH Paris and Tokyo using the tool.")],
            )
        ],
        [WEATHER_TOOL],
        tool_choice="required",
    )

    cities = sorted(call.args.get("city", "").lower() for call in result.tool_calls)
    assert len(result.tool_calls) >= 2
    assert any("paris" in c for c in cities)
    assert any("tokyo" in c for c in cities)


@LIVE
async def test_that_live_openai_reports_reasoning_when_enabled(logger: Logger) -> None:
    generator = _live_generator(logger)

    result = await generator.step(
        [
            Message(
                role=Role.USER,
                parts=[
                    TextPart(
                        text="Alice, Bob and Carol's ages sum to 90. Alice is twice Bob's age, "
                        "and Carol is 10 more than Bob. Find each age, step by step."
                    )
                ],
            )
        ],
        reasoning=ReasoningConfig(effort="high", visibility="summary"),
    )

    assert "20" in result.message.text  # Bob = 20 (Alice 40, Carol 30)
    # Thinking was enabled, so reasoning tokens must be reported and counted as
    # part of output tokens (OpenAI includes reasoning in output_tokens).
    assert result.usage.reasoning_tokens > 0
    assert result.usage.output_tokens >= result.usage.reasoning_tokens


@LIVE
async def test_that_live_openai_streams_a_visible_thought_summary(logger: Logger) -> None:
    generator = _live_generator(logger)

    streamed_reasoning: list[str] = []
    completed: StepResult | None = None
    async for event in generator.stream_step(
        [
            Message(
                role=Role.USER,
                parts=[
                    TextPart(
                        text="Solve step by step, showing your reasoning: a farmer has chickens "
                        "and rabbits with 35 heads and 94 legs. How many of each?"
                    )
                ],
            )
        ],
        reasoning=ReasoningConfig(effort="high", visibility="full"),
    ):
        if isinstance(event, ReasoningDelta):
            streamed_reasoning.append(event.text)
        elif isinstance(event, StepCompleted):
            completed = event.result

    assert completed is not None
    assert streamed_reasoning, "expected at least one ReasoningDelta with summary text"
    assert "".join(streamed_reasoning) == completed.message.reasoning
    assert completed.message.reasoning.strip()
    assert completed.usage.reasoning_tokens > 0
    assert "23" in completed.message.text and "12" in completed.message.text


@LIVE
async def test_that_live_reasoning_and_tool_items_round_trip_through_encode(
    logger: Logger,
) -> None:
    """A real reasoning + tool-call turn must re-encode with both items intact:
    the function_call (matched by call_id) and the reasoning item (with its
    encrypted_content) — the OpenAI analog of preserving Gemini signatures."""
    generator = _live_generator(logger)
    reasoning = ReasoningConfig(effort="high", visibility="summary")

    history: list[Message] = [
        Message(role=Role.USER, parts=[TextPart(text="Use the tool to get the weather in Paris.")])
    ]
    result = await generator.step(
        history, [WEATHER_TOOL], tool_choice="required", reasoning=reasoning
    )
    history.append(result.message)

    re_encoded = generator._encode(history, [WEATHER_TOOL], "auto", reasoning=reasoning)
    item_types = [item.get("type") for item in re_encoded["input"]]
    function_calls = [i for i in re_encoded["input"] if i.get("type") == "function_call"]

    assert "function_call" in item_types
    assert function_calls[0]["call_id"] == result.tool_calls[0].id

    # If the model emitted reasoning, it must round-trip with encrypted_content.
    if any(isinstance(part, ReasoningPart) for part in result.message.parts):
        reasoning_items = [i for i in re_encoded["input"] if i.get("type") == "reasoning"]
        assert reasoning_items and reasoning_items[0].get("encrypted_content")


@LIVE
async def test_that_system_prompt_can_change_between_steps_around_a_tool_call(
    logger: Logger,
) -> None:
    """Step N triggers a tool call; we run the tool; step N+1 uses a *modified*
    system prompt. The follow-up must succeed AND visibly respect the change."""
    generator = _live_generator(logger)
    base_system = "You are a weather assistant. Use the get_weather tool when asked about weather."

    history: list[Message] = [
        Message(role=Role.SYSTEM, parts=[TextPart(text=base_system)]),
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")]),
    ]

    first = await generator.step(history, [WEATHER_TOOL], tool_choice="required")
    assert first.needs_tools
    history.append(first.message)

    call = first.tool_calls[0]
    history.append(
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(
                    call_id=call.id,
                    name=call.name,
                    content={"temperature_c": 15, "conditions": "sunny"},
                )
            ],
        )
    )

    # Change the system prompt by editing the leading system message in history.
    modified_system = (
        base_system + ' When you give your final answer, end it with the exact word "PINEAPPLE".'
    )
    history[0] = Message(role=Role.SYSTEM, parts=[TextPart(text=modified_system)])
    second = await generator.step(history, [WEATHER_TOOL], tool_choice="auto")

    assert not second.needs_tools
    assert second.finish_reason == FinishReason.STOP
    assert "15" in second.message.text or "sunny" in second.message.text.lower()
    assert "PINEAPPLE" in second.message.text.upper()


@LIVE
async def test_that_live_history_can_be_edited_between_manual_steps(logger: Logger) -> None:
    generator = _live_generator(logger)

    history: list[Message] = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Use the tool, then answer the user.")]),
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")]),
    ]
    first = await generator.step(history, [WEATHER_TOOL], tool_choice="required")
    history.append(first.message)

    call = first.tool_calls[0]
    history.append(
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(
                    call_id=call.id, name=call.name, content={"temperature_c": 21, "sky": "clear"}
                )
            ],
        )
    )

    # History surgery: inject an extra instruction mid-conversation; the
    # assistant tool-call turn (with its raw item) is left untouched.
    history.insert(
        -1, Message(role=Role.USER, parts=[TextPart(text="(Reminder: answer in Celsius.)")])
    )

    second = await generator.step(history, [WEATHER_TOOL], tool_choice="auto")
    assert not second.needs_tools
    assert "21" in second.message.text


@LIVE
async def test_that_live_e2e_streams_thoughts_and_runs_tools_to_a_final_answer(
    logger: Logger,
) -> None:
    generator = _live_generator(logger)
    reasoning = ReasoningConfig(effort="high", visibility="full")

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        assert call.name == "get_weather"
        return ToolResultPart(
            call_id=call.id,
            name=call.name,
            content={"temperature_c": 15, "conditions": "rain likely"},
        )

    history: list[Message] = [
        Message(
            role=Role.SYSTEM,
            parts=[
                TextPart(
                    text="Use the tool to look things up, reason about the result, then answer."
                )
            ],
        ),
        Message(
            role=Role.USER,
            parts=[
                TextPart(
                    text="What's the weather in Paris, and should I take an umbrella? "
                    "Use the tool, then reason about the result."
                )
            ],
        ),
    ]

    tool_calls_seen: list[str] = []
    dispatched: list[str] = []
    total_reasoning_tokens = 0
    final_answer = ""  # text streamed by the terminal (no-tools) step
    reached_final_answer = False

    for _ in range(5):
        text_buf: list[str] = []
        completed: StepResult | None = None
        async for event in generator.stream_step(
            history, [WEATHER_TOOL], tool_choice="auto", reasoning=reasoning
        ):
            if isinstance(event, TextDelta):
                text_buf.append(event.text)
            elif isinstance(event, ToolCallStarted):
                tool_calls_seen.append(event.name)
            elif isinstance(event, StepCompleted):
                completed = event.result

        assert completed is not None
        assert "".join(text_buf) == completed.message.text
        total_reasoning_tokens += completed.usage.reasoning_tokens
        history.append(completed.message)

        if not completed.needs_tools:
            final_answer = "".join(text_buf)
            reached_final_answer = True
            break

        for call in completed.tool_calls:
            dispatched.append(call.args.get("city", ""))
            history.append(Message(role=Role.TOOL, parts=[await dispatch(call)]))

    # A tool ran in an earlier step...
    assert "get_weather" in tool_calls_seen
    assert any("paris" in city.lower() for city in dispatched)

    # ...and the loop went ALL THE WAY to a final assistant message generated
    # AFTER the tool result (no tool calls), streamed token-by-token.
    assert reached_final_answer, "loop never reached a tool-free final message"
    assert history[-1].role == Role.ASSISTANT
    assert not history[-1].tool_calls
    assert final_answer.strip(), "the final message was empty"
    assert final_answer == history[-1].text
    grounded = final_answer.lower()
    assert "15" in grounded or "umbrella" in grounded or "rain" in grounded
    assert total_reasoning_tokens > 0


@LIVE
async def test_that_live_openai_accepts_a_nullable_tool_parameter(logger: Logger) -> None:
    """The adapter translates ParameterSpec(nullable=True) to a JSON Schema
    null-union; the live API must accept that schema and invoke the tool, and a
    null value decodes to Python None. (Whether the model emits null or "" for a
    given turn is the model's choice, not the adapter's.)"""
    generator = _live_generator(logger)
    tool = ToolSpec(
        name="set_value",
        description="Set a value; pass JSON null when the value is unknown.",
        parameters=[ParameterSpec(name="value", type="string", nullable=True)],
    )

    result = await generator.step(
        [Message(role=Role.USER, parts=[TextPart(text="The value is unknown. Call set_value.")])],
        [tool],
        tool_choice="required",
    )

    call = result.tool_calls[0]
    assert call.name == "set_value"
    assert "value" in call.args
    assert call.args["value"] is None or isinstance(call.args["value"], str)


@LIVE
async def test_that_prompt_cache_key_is_accepted_live(logger: Logger) -> None:
    """A request carrying a cache_key (-> prompt_cache_key) must be accepted and
    answer normally; OpenAI's prefix caching is automatic so this is a hint."""
    generator = _live_generator(logger)

    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Be terse.")]),
        Message(role=Role.USER, parts=[TextPart(text="Some stable context.")], cache_key="ctx-1"),
        Message(role=Role.USER, parts=[TextPart(text="What is 2 + 2? Answer with a number.")]),
    ]
    result = await generator.step(history)

    assert result.finish_reason == FinishReason.STOP
    assert "4" in result.message.text


@LIVE
async def test_that_live_openai_step_can_be_cancelled(logger: Logger) -> None:
    generator = _live_generator(logger)

    task = asyncio.ensure_future(
        generator.step(
            [
                Message(role=Role.SYSTEM, parts=[TextPart(text="Write a very long essay.")]),
                Message(role=Role.USER, parts=[TextPart(text="Write 2000 words about the sea.")]),
            ]
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@LIVE
async def test_that_cancelling_mid_stream_closes_the_provider_stream(logger: Logger) -> None:
    """Cancel while consuming a live stream and confirm the underlying Responses
    stream (and its HTTP connection) is actually closed, not leaked."""
    generator = _live_generator(logger)
    stream_closed = asyncio.Event()
    real_create = generator._client.responses.create

    class _StreamProxy:
        def __init__(self, real: Any) -> None:
            self._real = real

        async def __aenter__(self) -> "_StreamProxy":
            await self._real.__aenter__()
            return self

        async def __aexit__(self, *args: Any) -> Any:
            stream_closed.set()
            return await self._real.__aexit__(*args)

        def __aiter__(self) -> Any:
            return self._real.__aiter__()

    async def spy_create(**kwargs: Any) -> Any:
        return _StreamProxy(await real_create(**kwargs))

    generator._client.responses.create = spy_create  # type: ignore[method-assign]

    first_event = asyncio.Event()

    async def consume() -> None:
        async for _ in generator.stream_step(
            [
                Message(role=Role.SYSTEM, parts=[TextPart(text="Write a long, detailed essay.")]),
                Message(
                    role=Role.USER, parts=[TextPart(text="Write a 1500-word essay about the sea.")]
                ),
            ]
        ):
            first_event.set()

    task = asyncio.ensure_future(consume())
    await asyncio.wait_for(first_event.wait(), timeout=30)  # we're mid-stream
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The provider stream was torn down (its async context exited).
    await asyncio.wait_for(stream_closed.wait(), timeout=5)


# ─────────────────────── cache cost accounting (live) ───────────────────────
#
# OpenAI caching is automatic (no cache_control): a >1024-token prefix caches
# on its own, and Message.cache_key rides along as the prompt_cache_key
# routing hint. Reads bill at a discount; writes bill only on GPT-5.6+ models,
# which also report them. On the pre-5.6 live model the write-meter test
# asserts the no-phantom-charges property: a model that is not billed for
# writes must never grow a write meter. When LIVE_MODEL moves to a 5.6-family
# id, flip the assertion to expect the meter on a salted (cold) prompt.


def _cacheable_conversation(prefix: str) -> list[Message]:
    return [
        Message(role=Role.SYSTEM, parts=[TextPart(text=prefix)], cache_key="cost-live"),
        Message(role=Role.USER, parts=[TextPart(text="Reply with the single word: ok")]),
    ]


@LIVE
async def test_that_an_immediate_repeat_request_reports_cache_reads(logger: Logger) -> None:
    generator = _live_generator(logger, cache=CacheConfig(enabled=True))
    conversation = _cacheable_conversation(salted_cacheable_prefix())

    result = await generator.step(conversation)

    # Automatic caching engages within moments for identical prompts; retry a
    # few times — this is the suite's only nondeterministic assertion.
    for _ in range(4):
        result = await generator.step(conversation)
        if result.usage.cached_input_tokens:
            break

    usage = result.usage
    assert usage.cached_input_tokens > 1_024
    # OpenAI semantics: cached is a subset of input_tokens.
    assert usage.input_tokens >= usage.cached_input_tokens


@LIVE
async def test_that_write_meters_are_not_fabricated_without_write_billing(logger: Logger) -> None:
    generator = _live_generator(logger, cache=CacheConfig(enabled=True))

    result = await generator.step(_cacheable_conversation(salted_cacheable_prefix()))

    usage = result.usage
    # Sanity: the salted prefix actually reached the model.
    assert usage.input_tokens > 6_000
    # Pre-5.6 models are not billed for cache writes and must not meter them.
    assert "cache_write_input_tokens" not in usage.extra


# --- OpenAIEstimatingTokenizer model→encoding mapping (offline, tiktoken-local) ---


@pytest.mark.parametrize("model_name", ["gpt-5", "gpt-5.1-mini", "gpt-5.4-nano", "gpt-5.6"])
def test_that_gpt5_variants_use_the_gpt5_encoding(model_name: str) -> None:
    tokenizer = OpenAIEstimatingTokenizer(model_name)
    assert tokenizer.model_name == model_name
    assert tokenizer.encoding.name == tiktoken.encoding_for_model("gpt-5").name


def test_that_non_gpt5_models_keep_their_own_encoding() -> None:
    tokenizer = OpenAIEstimatingTokenizer("gpt-4")
    assert tokenizer.encoding.name == tiktoken.encoding_for_model("gpt-4").name


@LIVE
async def test_that_live_openai_embedder_returns_embeddings_and_usage(logger: Logger) -> None:
    embedder = OpenAITextEmbedding3Small(
        logger=logger,
        tracer=LocalTracer(),
        meter=LocalMeter(logger),
    )

    result = await embedder.do_embed(["hello world", "goodbye world"])

    assert len(result.vectors) == 2
    assert all(len(vector) == 1536 for vector in result.vectors)
    assert result.usage.input_tokens > 0


def test_that_an_all_system_history_gets_a_faux_user_turn(openai: OpenAIReactGenerator) -> None:
    """The Responses API rejects an empty input array, so a history with no
    non-system turns (an agent-first greeting) gets a faux user turn."""
    from parlant.core.nlp.react import EMPTY_CONVERSATION_USER_TURN

    request = openai._encode(
        [Message(role=Role.SYSTEM, parts=[TextPart(text="Greet the customer.")])],
        [],
        "auto",
        reasoning=ReasoningConfig(),
    )

    assert len(request["input"]) == 1
    assert request["input"][0]["role"] == "user"
    assert EMPTY_CONVERSATION_USER_TURN in str(request["input"][0]["content"])

    # A history with any non-system turn is untouched.
    request = openai._encode(
        [
            Message(role=Role.SYSTEM, parts=[TextPart(text="Greet the customer.")]),
            Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello!")]),
        ],
        [],
        "auto",
        reasoning=ReasoningConfig(),
    )
    assert all(EMPTY_CONVERSATION_USER_TURN not in str(item) for item in request["input"])


# ═══════════════════════ REACT DEFAULT-MODEL RESOLUTION ═════════════════════


async def test_that_the_openai_react_default_resolves_by_effort(
    container: Container,
) -> None:
    registry = ModelImplementationRegistry(NullSecretLoader())
    register_openai_models(registry, container)

    def react_default(qualifiers: dict[str, str]) -> str | ModelProvider:
        return registry.default_config(
            "openai",
            ModelCapability.REACT,
            route="react",
            qualifiers=qualifiers,
        ).model

    assert react_default({"effort": "min"}) == "openai:gpt-6-luna"
    assert react_default({"effort": "low"}) == "openai:gpt-6-luna"
    assert react_default({"effort": "medium"}) == "openai:gpt-6-luna"
    assert react_default({"effort": "high"}) == "openai:gpt-6-luna"
    assert react_default({"effort": "max"}) == "openai:gpt-6.1-terra"
    assert react_default({}) == "openai:gpt-6-luna"


def test_that_the_large_model_size_resolves_to_gpt_6_1_terra(
    openai: OpenAIReactGenerator,
) -> None:
    request = openai._encode(
        [Message(role=Role.USER, parts=[TextPart(text="Hi")])],
        [],
        "auto",
        reasoning=ReasoningConfig(),
        hints={"model_size": ModelSize.LARGE},
    )

    assert request["model"] == "gpt-6.1-terra"


async def test_that_max_effort_replies_default_to_gpt_6_1_terra(
    container: Container,
) -> None:
    registry = ModelImplementationRegistry(NullSecretLoader())
    register_openai_models(registry, container)

    config = registry.default_config(
        "openai",
        ModelCapability.REACT,
        route="react",
        qualifiers={"effort": "max"},
    )

    assert config.model == "openai:gpt-6.1-terra"


# ═════════════════════════════ 6. DECISIONS ═════════════════════════════════
#
# OpenAI serves typed decisions on its own endpoint, through the SDK's
# `decisions` resource. The endpoint is stricter than our port: `instructions`
# must be a string, a predicate takes no criteria at all, and `input` is a string
# or a message array. The adapter is what absorbs that, so the DecisionMaker port
# stays structured for callers like the rule ranker.


def _decisions_client(
    handler: Callable[[httpx2.Request], httpx2.Response],
    requests: list[httpx2.Request] | None = None,
) -> AsyncClient:
    def record(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(request)
        return handler(request)

    return AsyncClient(
        api_key="offline-decision-tests",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(record)),
    )


def _decision_maker(
    handler: Callable[[httpx2.Request], httpx2.Response],
    requests: list[httpx2.Request] | None = None,
    model_name: str = DEFAULT_DECISION_MODEL,
) -> OpenAIDecisionMaker:
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    return OpenAIDecisionMaker(
        model_name=model_name,
        logger=logger,
        tracer=tracer,
        meter=LocalMeter(logger),
        client=_decisions_client(handler, requests),
    )


def _answer_response(
    answers: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            request=request,
            json={
                "model": "gpt-6-luna",
                "answers": answers,
                "usage": usage
                or {
                    "input_tokens": 492,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 492,
                },
            },
        )

    return handler


def _sent_body(request: httpx2.Request) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(request.content))


async def test_that_openai_decision_maker_posts_to_the_decisions_endpoint() -> None:
    requests: list[httpx2.Request] = []
    maker = _decision_maker(
        _answer_response([{"type": "predicate", "name": "q", "probability": 0.9}]),
        requests,
    )

    await maker.decide("the state", {"q": Proposition(instructions="Does it hold?")})

    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.openai.com/v1/decisions"
    body = _sent_body(requests[0])
    assert body["model"] == DEFAULT_DECISION_MODEL
    assert body["input"] == "the state"


async def test_that_openai_decision_maker_sends_questions_as_a_named_array() -> None:
    # OpenAI takes an array of questions each carrying its own `name`, where the
    # OpenRouter endpoint takes an object keyed by name. Names must round-trip,
    # because that is how an answer is matched back to its question.
    requests: list[httpx2.Request] = []
    maker = _decision_maker(
        _answer_response(
            [
                {"type": "predicate", "name": "first", "probability": 0.8},
                {"type": "predicate", "name": "second", "probability": 0.2},
            ]
        ),
        requests,
    )

    await maker.decide(
        "the state",
        {
            "first": Proposition(instructions="One?"),
            "second": Proposition(instructions="Two?"),
        },
    )

    questions = _sent_body(requests[0])["questions"]
    assert isinstance(questions, list)
    assert [q["name"] for q in questions] == ["first", "second"]


async def test_that_openai_decision_maker_maps_our_question_types_to_openai_primitives() -> None:
    requests: list[httpx2.Request] = []
    maker = _decision_maker(
        _answer_response(
            [
                {"type": "predicate", "name": "p", "probability": 0.5},
                {
                    "type": "choice",
                    "name": "c",
                    "choice": "a",
                    "probabilities": [
                        {"value": "a", "probability": 0.7},
                        {"value": "b", "probability": 0.3},
                    ],
                    "confidence": 0.6,
                },
                {
                    "type": "score",
                    "name": "s",
                    "score": 1.0,
                    "probabilities": [
                        {"value": 0, "label": "low", "probability": 0.5},
                        {"value": 1, "label": "high", "probability": 0.5},
                    ],
                    "confidence": 0.4,
                },
            ]
        ),
        requests,
    )

    await maker.decide(
        "the state",
        {
            "p": Proposition(instructions="Holds?"),
            "c": Choice(instructions="Which?", criteria={"a": "the first", "b": "the second"}),
            "s": Scale(instructions="How much?", criteria=["low", "high"]),
        },
    )

    by_name = {q["name"]: q for q in _sent_body(requests[0])["questions"]}

    assert by_name["p"]["type"] == "predicate"

    assert by_name["c"]["type"] == "choice"
    assert by_name["c"]["choices"] == [
        {"value": "a", "description": "the first"},
        {"value": "b", "description": "the second"},
    ]

    assert by_name["s"]["type"] == "score"
    assert by_name["s"]["levels"] == [
        {"label": "low", "description": "low"},
        {"label": "high", "description": "high"},
    ]


async def test_that_openai_decision_maker_describes_every_choice_option() -> None:
    # An option sent without a description comes back as a refusal rather than an
    # error, so a `None` criterion must still produce a description. The option's
    # own name is the honest fallback.
    requests: list[httpx2.Request] = []
    maker = _decision_maker(
        _answer_response(
            [
                {
                    "type": "choice",
                    "name": "c",
                    "choice": "refund",
                    "probabilities": [
                        {"value": "refund", "probability": 0.6},
                        {"value": "replace", "probability": 0.4},
                    ],
                    "confidence": 0.5,
                }
            ]
        ),
        requests,
    )

    await maker.decide(
        "the state",
        {"c": Choice(instructions="Which?", criteria={"refund": None, "replace": "a new unit"})},
    )

    choices = _sent_body(requests[0])["questions"][0]["choices"]
    assert all(choice.get("description") for choice in choices)
    assert choices == [
        {"value": "refund", "description": "refund"},
        {"value": "replace", "description": "a new unit"},
    ]


async def test_that_openai_decision_maker_serializes_structured_instructions_to_text() -> None:
    # The endpoint rejects a non-string `instructions` outright, and the rule
    # ranker sends a mapping of the rule plus its question. The adapter has to
    # render it, with every part still present in the text.
    requests: list[httpx2.Request] = []
    maker = _decision_maker(
        _answer_response([{"type": "predicate", "name": "q", "probability": 0.9}]),
        requests,
    )

    await maker.decide(
        "the state",
        {
            "q": Proposition(
                instructions={"rule": "the rule text", "question": "Does it apply?"},
            )
        },
    )

    instructions = _sent_body(requests[0])["questions"][0]["instructions"]
    assert isinstance(instructions, str)
    assert "the rule text" in instructions
    assert "Does it apply?" in instructions


async def test_that_openai_decision_maker_folds_a_propositions_criteria_into_its_instructions() -> (
    None
):
    # A predicate takes no criteria field — sending one is a 400. The port keeps
    # them structured, so the adapter folds them into the instructions text
    # rather than dropping a signal the ranker relies on.
    requests: list[httpx2.Request] = []
    maker = _decision_maker(
        _answer_response([{"type": "predicate", "name": "q", "probability": 0.9}]),
        requests,
    )

    await maker.decide(
        "the state",
        {
            "q": Proposition(
                instructions="Does the rule apply?",
                criteria=PropositionCriteria(
                    true="the rule governs the conversation",
                    false="the rule is beside the point",
                ),
            )
        },
    )

    question = _sent_body(requests[0])["questions"][0]
    assert "criteria" not in question
    instructions = question["instructions"]
    assert "Does the rule apply?" in instructions
    assert "the rule governs the conversation" in instructions
    assert "the rule is beside the point" in instructions


async def test_that_openai_decision_maker_serializes_a_non_string_state_to_text() -> None:
    # `input` accepts a string or a message array, never a bare JSON object,
    # while the port's state is any JSON value.
    requests: list[httpx2.Request] = []
    maker = _decision_maker(
        _answer_response([{"type": "predicate", "name": "q", "probability": 0.9}]),
        requests,
    )

    await maker.decide(
        {"conversation": "the customer is upset"},
        {"q": Proposition(instructions="Holds?")},
    )

    sent_input = _sent_body(requests[0])["input"]
    assert isinstance(sent_input, str)
    assert "the customer is upset" in sent_input


async def test_that_openai_decision_maker_sends_one_request_for_many_questions() -> None:
    requests: list[httpx2.Request] = []
    maker = _decision_maker(
        _answer_response(
            [{"type": "predicate", "name": f"q{i}", "probability": 0.5} for i in range(12)]
        ),
        requests,
    )

    result = await maker.decide(
        "the state",
        {f"q{i}": Proposition(instructions=f"Holds {i}?") for i in range(12)},
    )

    assert len(requests) == 1
    assert len(result.answers) == 12


async def test_that_openai_decision_maker_parses_every_answer_type() -> None:
    maker = _decision_maker(
        _answer_response(
            [
                {"type": "predicate", "name": "p", "probability": 0.84},
                {
                    "type": "choice",
                    "name": "c",
                    "choice": "refund",
                    "probabilities": [
                        {"value": "refund", "probability": 0.9},
                        {"value": "replace", "probability": 0.1},
                    ],
                    "confidence": 0.88,
                },
                {
                    "type": "score",
                    "name": "s",
                    "score": 1.25,
                    "probabilities": [
                        {"value": 0, "label": "low", "probability": 0.25},
                        {"value": 1, "label": "mid", "probability": 0.25},
                        {"value": 2, "label": "high", "probability": 0.5},
                    ],
                    "confidence": 0.5,
                },
            ]
        )
    )

    result = await maker.decide(
        "the state",
        {
            "p": Proposition(instructions="Holds?"),
            "c": Choice(instructions="Which?", criteria={"refund": None, "replace": None}),
            "s": Scale(instructions="How much?", criteria=["low", "mid", "high"]),
        },
    )

    proposition = result.answers["p"]
    assert isinstance(proposition, PropositionAnswer)
    assert proposition.probability == 0.84

    choice = result.answers["c"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.option == "refund"
    assert choice.probabilities == {"refund": 0.9, "replace": 0.1}
    assert choice.confidence == 0.88

    scale = result.answers["s"]
    assert isinstance(scale, ScaleAnswer)
    assert scale.score == 1.25
    assert scale.confidence == 0.5


async def test_that_openai_decision_maker_decodes_a_scales_zero_based_levels() -> None:
    # The endpoint indexes a scale's levels from zero, so a three-level scale
    # answers with 0, 1 or 2 and never reaches 3.
    maker = _decision_maker(
        _answer_response(
            [
                {
                    "type": "score",
                    "name": "s",
                    "score": 2.0,
                    "probabilities": [
                        {"value": 0, "label": "low", "probability": 0.0},
                        {"value": 1, "label": "mid", "probability": 0.0},
                        {"value": 2, "label": "high", "probability": 1.0},
                    ],
                    "confidence": 0.9,
                }
            ]
        )
    )

    result = await maker.decide(
        "the state",
        {"s": Scale(instructions="How much?", criteria=["low", "mid", "high"])},
    )

    scale = result.answers["s"]
    assert isinstance(scale, ScaleAnswer)
    assert scale.legend == {0: "low", 1: "mid", 2: "high"}
    assert scale.probabilities == {0: 0.0, 1: 0.0, 2: 1.0}


async def test_that_openai_decision_maker_raises_on_a_refusal() -> None:
    # A refusal is a real answer type, not an error status: the question came
    # back unanswered, and the port promises every question an answer.
    maker = _decision_maker(_answer_response([{"type": "refusal", "name": "q"}]))

    with pytest.raises(DecisionError):
        await maker.decide("the state", {"q": Proposition(instructions="Holds?")})


async def test_that_openai_decision_maker_raises_when_an_answer_is_missing() -> None:
    maker = _decision_maker(
        _answer_response([{"type": "predicate", "name": "first", "probability": 0.5}])
    )

    with pytest.raises(DecisionError):
        await maker.decide(
            "the state",
            {
                "first": Proposition(instructions="One?"),
                "second": Proposition(instructions="Two?"),
            },
        )


async def test_that_openai_decision_maker_surfaces_an_api_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400,
            request=request,
            json={"error": {"message": "Unknown parameter: 'questions[0].criteria'."}},
        )

    maker = _decision_maker(handler)

    with pytest.raises(DecisionError):
        await maker.decide("the state", {"q": Proposition(instructions="Holds?")})


async def test_that_openai_decision_maker_requires_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    with pytest.raises(Exception):
        OpenAIDecisionMaker(logger=logger, tracer=tracer, meter=LocalMeter(logger))


async def test_that_openai_decision_maker_reports_one_request_for_a_batch() -> None:
    # One request for the whole batch is the point of the port, and usage has to
    # say so or the saving is invisible to cost accounting.
    maker = _decision_maker(
        _answer_response(
            [{"type": "predicate", "name": f"q{i}", "probability": 0.5} for i in range(5)]
        )
    )

    result = await maker.decide(
        "the state",
        {f"q{i}": Proposition(instructions=f"Holds {i}?") for i in range(5)},
    )

    assert result.info.usage.requests == 1


async def test_that_openai_decision_maker_reports_cached_tokens_for_accounting() -> None:
    # The endpoint reports no cost, so tokens are all pricing has to work from.
    maker = _decision_maker(
        _answer_response(
            [{"type": "predicate", "name": "q", "probability": 0.5}],
            usage={
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 1000,
            },
        )
    )

    result = await maker.decide("the state", {"q": Proposition(instructions="Holds?")})

    assert result.info.usage.input_tokens == 1000
    assert result.info.usage.cached_input_tokens == 400
    assert result.info.usage.output_tokens == 0


async def test_that_openai_model_registry_serves_a_decision_maker(
    container: Container,
) -> None:
    registry = ModelImplementationRegistry(NullSecretLoader())
    register_openai_models(registry, container)

    assert (
        registry.default_config(
            "openai",
            ModelCapability.DECISION,
            route="matching.rank",
            qualifiers={},
        ).model
        == f"openai:{DEFAULT_DECISION_MODEL}"
    )


@LIVE
async def test_that_gpt_6_luna_answers_a_batch_of_questions_live() -> None:
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    maker = OpenAIDecisionMaker(logger=logger, tracer=tracer, meter=LocalMeter(logger))

    result = await maker.decide(
        "Customer: My coffee grinder arrived with a cracked burr assembly.\n"
        "Agent: I'm sorry about that.\n"
        "Customer: I'd like a replacement, not a refund.",
        {
            "damaged": Proposition(
                instructions="Does the customer report a damaged item?",
                criteria=PropositionCriteria(
                    true="the customer says an item arrived broken",
                    false="no damage is mentioned",
                ),
            ),
            "wants": Choice(
                instructions="What does the customer want?",
                criteria={
                    "refund": "the customer wants money back",
                    "replacement": "the customer wants a new unit",
                },
            ),
            "urgency": Scale(
                instructions="How urgent is this request?",
                criteria=["can wait", "should be handled today", "handle immediately"],
            ),
        },
    )

    damaged = result.answers["damaged"]
    assert isinstance(damaged, PropositionAnswer)
    assert damaged.probability > 0.5

    wants = result.answers["wants"]
    assert isinstance(wants, ChoiceAnswer)
    assert wants.option == "replacement"

    urgency = result.answers["urgency"]
    assert isinstance(urgency, ScaleAnswer)
    # Zero-based levels: a three-level scale tops out below 3.
    assert 0.0 <= urgency.score <= 2.0

    assert result.info.usage.requests == 1
    assert result.info.usage.input_tokens > 0
