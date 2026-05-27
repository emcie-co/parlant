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
import os
from types import SimpleNamespace
from typing import Any

import pytest
from openai import AsyncClient

from parlant.adapters.nlp.openai_service import OPENAI_ITEM_KEY, OpenAIReactGenerator
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.tracer import LocalTracer
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
        Message(role=Role.SYSTEM, parts=[TextPart(text="Extra system rule.")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(role=Role.ASSISTANT, parts=[TextPart(text="hello")]),
        Message(
            role=Role.TOOL,
            parts=[ToolResultPart(call_id="call_1", name="get_weather", content="sunny")],
        ),
    ]

    request = openai._encode(
        history, [], "auto", system="You are a test agent.", reasoning=ReasoningConfig()
    )

    assert "You are a test agent." in request["instructions"]
    assert "Extra system rule." in request["instructions"]

    items = request["input"]
    # user message, assistant message, function_call_output
    assert items[0]["role"] == "user"
    assert items[0]["content"] == [{"type": "input_text", "text": "hi"}]
    assert items[1]["role"] == "assistant"
    assert items[1]["content"] == [{"type": "output_text", "text": "hello"}]
    assert items[2] == {
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

    items = openai._encode(
        history, [WEATHER_TOOL], "auto", system=None, reasoning=ReasoningConfig()
    )["input"]

    # Both items are replayed exactly as received.
    assert items[0] == raw_reasoning
    assert items[1] == raw_call


def test_that_encode_reconstructs_tool_calls_without_a_raw_item(
    openai: OpenAIReactGenerator,
) -> None:
    history = [
        Message(
            role=Role.ASSISTANT,
            parts=[ToolCallPart(id="call_9", name="get_weather", args={"city": "Paris"})],
        )
    ]

    item = openai._encode(
        history, [WEATHER_TOOL], "auto", system=None, reasoning=ReasoningConfig()
    )["input"][0]

    assert item == {
        "type": "function_call",
        "call_id": "call_9",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }


def test_that_encode_serializes_non_string_tool_results(openai: OpenAIReactGenerator) -> None:
    history = [
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(call_id="c1", name="t", content="plain string"),
                ToolResultPart(call_id="c2", name="t", content={"temp": 18}),
            ],
        )
    ]

    items = openai._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())["input"]

    assert items[0]["output"] == "plain string"
    assert items[1]["output"] == '{"temp": 18}'  # JSON-encoded


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
    request = openai._encode(
        [], [WEATHER_TOOL], tool_choice, system=None, reasoning=ReasoningConfig()
    )
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


def test_that_encode_emits_reasoning_only_when_enabled(openai: OpenAIReactGenerator) -> None:
    disabled = openai._encode([], [], "auto", system=None, reasoning=ReasoningConfig())
    assert "reasoning" not in disabled
    assert "include" not in disabled

    enabled = openai._encode(
        [],
        [],
        "auto",
        system=None,
        reasoning=ReasoningConfig(enabled=True, effort="high", visibility="full"),
    )
    assert enabled["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert enabled["include"] == ["reasoning.encrypted_content"]


def test_that_encode_omits_summary_when_visibility_is_none(openai: OpenAIReactGenerator) -> None:
    request = openai._encode(
        [],
        [],
        "auto",
        system=None,
        reasoning=ReasoningConfig(enabled=True, effort="low", visibility="none"),
    )
    assert request["reasoning"] == {"effort": "low"}


def test_that_cache_key_becomes_prompt_cache_key(openai: OpenAIReactGenerator) -> None:
    history = [
        Message(role=Role.USER, parts=[TextPart(text="ctx")], cache_key="agent-7"),
        Message(role=Role.USER, parts=[TextPart(text="q")]),
    ]
    request = openai._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())
    assert request["prompt_cache_key"] == "agent-7"


def test_that_prompt_cache_key_uses_the_first_marked_message(openai: OpenAIReactGenerator) -> None:
    # OpenAI takes a single prompt_cache_key; the FIRST marker wins (the most
    # stable prefix boundary), and a marked system message counts.
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="big stable system")], cache_key="sys-v1"),
        Message(role=Role.USER, parts=[TextPart(text="ctx")], cache_key="later"),
        Message(role=Role.USER, parts=[TextPart(text="q")]),
    ]
    request = openai._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())

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
    request = generator._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())
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
        [Message(role=Role.USER, parts=[TextPart(text="What is the capital of France?")])],
        system="Answer in exactly one short sentence.",
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
        [Message(role=Role.USER, parts=[TextPart(text="List three primary colors.")])],
        system="Be concise.",
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
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris right now?")])
    ]
    final_history = await generator.run(
        history,
        [WEATHER_TOOL],
        dispatch,
        on_step=on_step,
        max_steps=5,
        system="Use tools when needed, then answer the user.",
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
        reasoning=ReasoningConfig(enabled=True, effort="high", visibility="summary"),
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
        reasoning=ReasoningConfig(enabled=True, effort="high", visibility="full"),
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
    reasoning = ReasoningConfig(enabled=True, effort="high", visibility="summary")

    history: list[Message] = [
        Message(role=Role.USER, parts=[TextPart(text="Use the tool to get the weather in Paris.")])
    ]
    result = await generator.step(
        history, [WEATHER_TOOL], tool_choice="required", reasoning=reasoning
    )
    history.append(result.message)

    re_encoded = generator._encode(
        history, [WEATHER_TOOL], "auto", system=None, reasoning=reasoning
    )
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
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")])
    ]

    first = await generator.step(
        history, [WEATHER_TOOL], tool_choice="required", system=base_system
    )
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

    modified_system = (
        base_system + ' When you give your final answer, end it with the exact word "PINEAPPLE".'
    )
    second = await generator.step(
        history, [WEATHER_TOOL], tool_choice="auto", system=modified_system
    )

    assert not second.needs_tools
    assert second.finish_reason == FinishReason.STOP
    assert "15" in second.message.text or "sunny" in second.message.text.lower()
    assert "PINEAPPLE" in second.message.text.upper()


@LIVE
async def test_that_live_history_can_be_edited_between_manual_steps(logger: Logger) -> None:
    generator = _live_generator(logger)
    system = "Use the tool, then answer the user."

    history: list[Message] = [
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")])
    ]
    first = await generator.step(history, [WEATHER_TOOL], tool_choice="required", system=system)
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

    second = await generator.step(history, [WEATHER_TOOL], tool_choice="auto", system=system)
    assert not second.needs_tools
    assert "21" in second.message.text


@LIVE
async def test_that_live_e2e_streams_thoughts_and_runs_tools_to_a_final_answer(
    logger: Logger,
) -> None:
    generator = _live_generator(logger)
    system = "Use the tool to look things up, reason about the result, then answer the user."
    reasoning = ReasoningConfig(enabled=True, effort="high", visibility="full")

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        assert call.name == "get_weather"
        return ToolResultPart(
            call_id=call.id,
            name=call.name,
            content={"temperature_c": 15, "conditions": "rain likely"},
        )

    history: list[Message] = [
        Message(
            role=Role.USER,
            parts=[
                TextPart(
                    text="What's the weather in Paris, and should I take an umbrella? "
                    "Use the tool, then reason about the result."
                )
            ],
        )
    ]

    tool_calls_seen: list[str] = []
    dispatched: list[str] = []
    total_reasoning_tokens = 0

    for _ in range(5):
        text_buf: list[str] = []
        completed: StepResult | None = None
        async for event in generator.stream_step(
            history, [WEATHER_TOOL], tool_choice="auto", system=system, reasoning=reasoning
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
            break

        for call in completed.tool_calls:
            dispatched.append(call.args.get("city", ""))
            history.append(Message(role=Role.TOOL, parts=[await dispatch(call)]))

    assert "get_weather" in tool_calls_seen
    assert any("paris" in city.lower() for city in dispatched)
    assert total_reasoning_tokens > 0
    final_text = history[-1].text.lower()
    assert "15" in final_text or "umbrella" in final_text or "rain" in final_text


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
        Message(role=Role.USER, parts=[TextPart(text="Some stable context.")], cache_key="ctx-1"),
        Message(role=Role.USER, parts=[TextPart(text="What is 2 + 2? Answer with a number.")]),
    ]
    result = await generator.step(history, system="Be terse.")

    assert result.finish_reason == FinishReason.STOP
    assert "4" in result.message.text


@LIVE
async def test_that_live_openai_step_can_be_cancelled(logger: Logger) -> None:
    generator = _live_generator(logger)

    task = asyncio.ensure_future(
        generator.step(
            [Message(role=Role.USER, parts=[TextPart(text="Write 2000 words about the sea.")])],
            system="Write a very long essay.",
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
                Message(
                    role=Role.USER, parts=[TextPart(text="Write a 1500-word essay about the sea.")]
                )
            ],
            system="Write a long, detailed essay.",
        ):
            first_event.set()

    task = asyncio.ensure_future(consume())
    await asyncio.wait_for(first_event.wait(), timeout=30)  # we're mid-stream
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The provider stream was torn down (its async context exited).
    await asyncio.wait_for(stream_closed.wait(), timeout=5)
