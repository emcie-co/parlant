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

"""Tests for the Anthropic adapter (AnthropicReactGenerator) over the Messages API.

The provider-agnostic ReAct core is tested in tests/core/nlp/test_react.py. This
file covers Anthropic specifics:

1. Request-building (`_encode`) and the pure `_map_finish_reason` transform,
   offline. No API response is fabricated.
2. Live integration tests against claude-haiku-4-5, hitting the real Messages
   API, skipped unless ANTHROPIC_API_KEY is set.

With the Gemini and OpenAI suites, these show the same react.py interface backs
three different providers with no interface changes and no abstraction leaks.
"""

import asyncio
import os
from typing import Any

import pytest
from anthropic import AsyncAnthropic

from parlant.adapters.nlp.anthropic_service import ANTHROPIC_BLOCK_KEY, AnthropicReactGenerator
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
def anthropic(logger: Logger) -> AnthropicReactGenerator:
    # _encode and _map_finish_reason are pure transforms that never call the
    # client, so a client built with a throwaway key keeps these tests offline.
    return AnthropicReactGenerator(
        logger=logger,
        client=AsyncAnthropic(api_key="offline-encode-tests"),
    )


# ═══════════════════════ 1. ANTHROPIC REQUEST BUILDING ══════════════════════


def test_that_encode_maps_roles_and_system(anthropic: AnthropicReactGenerator) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Extra system rule.")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(role=Role.ASSISTANT, parts=[TextPart(text="hello")]),
        Message(
            role=Role.TOOL,
            parts=[ToolResultPart(call_id="tool_1", name="get_weather", content="sunny")],
        ),
    ]

    request = anthropic._encode(
        history, [], "auto", system="You are a test agent.", reasoning=ReasoningConfig()
    )

    # System is a top-level parameter (not a message).
    assert "You are a test agent." in request["system"]
    assert "Extra system rule." in request["system"]

    messages = request["messages"]
    # user, assistant, and tool-result-as-user (Anthropic puts tool_result in user)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == [{"type": "text", "text": "hi"}]
    assert messages[1]["content"] == [{"type": "text", "text": "hello"}]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tool_1", "content": "sunny", "is_error": False}
    ]


def test_that_encode_replays_raw_blocks_verbatim(anthropic: AnthropicReactGenerator) -> None:
    raw_thinking = {"type": "thinking", "thinking": "hmm", "signature": "SIG"}
    raw_tool_use = {
        "type": "tool_use",
        "id": "tool_1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }
    history = [
        Message(
            role=Role.ASSISTANT,
            parts=[
                ReasoningPart(
                    text="hmm", signature="SIG", provider_data={ANTHROPIC_BLOCK_KEY: raw_thinking}
                ),
                ToolCallPart(
                    id="tool_1",
                    name="get_weather",
                    args={"city": "Paris"},
                    provider_data={ANTHROPIC_BLOCK_KEY: raw_tool_use},
                ),
            ],
        )
    ]

    blocks = anthropic._encode(
        history, [WEATHER_TOOL], "auto", system=None, reasoning=ReasoningConfig()
    )["messages"][0]["content"]

    # Both blocks replay exactly — preserving the thinking signature.
    assert blocks[0] == raw_thinking
    assert blocks[1] == raw_tool_use


def test_that_encode_reconstructs_tool_calls_without_a_raw_block(
    anthropic: AnthropicReactGenerator,
) -> None:
    history = [
        Message(
            role=Role.ASSISTANT,
            parts=[ToolCallPart(id="tool_9", name="get_weather", args={"city": "Paris"})],
        )
    ]

    block = anthropic._encode(
        history, [WEATHER_TOOL], "auto", system=None, reasoning=ReasoningConfig()
    )["messages"][0]["content"][0]

    assert block == {
        "type": "tool_use",
        "id": "tool_9",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }


def test_that_encode_serializes_tool_results_and_flags_errors(
    anthropic: AnthropicReactGenerator,
) -> None:
    history = [
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(call_id="c1", name="t", content="plain string"),
                ToolResultPart(call_id="c2", name="t", content={"temp": 18}),
                ToolResultPart(call_id="c3", name="t", content="boom", is_error=True),
            ],
        )
    ]

    blocks = anthropic._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())[
        "messages"
    ][0]["content"]

    assert blocks[0]["content"] == "plain string"
    assert blocks[1]["content"] == '{"temp": 18}'  # JSON-encoded
    assert blocks[2]["is_error"] is True


@pytest.mark.parametrize(
    "tool_choice, expected",
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        ({"name": "get_weather"}, {"type": "tool", "name": "get_weather"}),
    ],
)
def test_that_encode_maps_tool_choice(
    anthropic: AnthropicReactGenerator, tool_choice: Any, expected: Any
) -> None:
    request = anthropic._encode(
        [], [WEATHER_TOOL], tool_choice, system=None, reasoning=ReasoningConfig()
    )
    assert request["tool_choice"] == expected


def test_that_encode_emits_thinking_only_when_enabled(anthropic: AnthropicReactGenerator) -> None:
    disabled = anthropic._encode([], [], "auto", system=None, reasoning=ReasoningConfig())
    assert "thinking" not in disabled

    enabled = anthropic._encode(
        [],
        [],
        "auto",
        system=None,
        reasoning=ReasoningConfig(enabled=True, budget_tokens=4096),
    )
    # Default visibility ("summary") -> display "summarized".
    assert enabled["thinking"] == {
        "type": "enabled",
        "budget_tokens": 4096,
        "display": "summarized",
    }
    # max_tokens must exceed the thinking budget.
    assert enabled["max_tokens"] > 4096


def test_that_visibility_maps_to_the_display_knob(anthropic: AnthropicReactGenerator) -> None:
    # Claude 4 has no verbatim option: "none" omits the thinking block, while
    # "summary" and "full" both request the summary ("full" has no equivalent).
    def display_for(visibility: str) -> Any:
        request = anthropic._encode(
            [],
            [],
            "auto",
            system=None,
            reasoning=ReasoningConfig(enabled=True, visibility=visibility),  # type: ignore[arg-type]
        )
        return request["thinking"]["display"]

    assert display_for("none") == "omitted"
    assert display_for("summary") == "summarized"
    assert display_for("full") == "summarized"


def test_that_encode_uses_a_default_thinking_budget(anthropic: AnthropicReactGenerator) -> None:
    request = anthropic._encode(
        [], [], "auto", system=None, reasoning=ReasoningConfig(enabled=True)
    )
    assert request["thinking"]["budget_tokens"] == anthropic._DEFAULT_THINKING_BUDGET


def test_that_encode_tool_translates_nullable_to_json_schema_null_type(
    anthropic: AnthropicReactGenerator,
) -> None:
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

    properties = anthropic._encode_tool(tool)["input_schema"]["properties"]

    assert properties["value"]["type"] == ["string", "null"]
    assert "nullable" not in properties["value"]
    assert properties["value"]["default"] == "hi"  # standard JSON Schema kept as-is
    assert properties["tags"]["items"]["type"] == ["string", "null"]


def test_that_cache_key_marks_the_prefix_with_cache_control(
    anthropic: AnthropicReactGenerator,
) -> None:
    history = [
        Message(role=Role.USER, parts=[TextPart(text="big reference doc")], cache_key="doc-v1"),
        Message(role=Role.USER, parts=[TextPart(text="the live question")]),
    ]

    messages = anthropic._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())[
        "messages"
    ]

    # The breakpoint message's last block carries cache_control...
    assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # ...and the live suffix does not.
    assert "cache_control" not in messages[1]["content"][-1]


def test_that_cache_control_ttl_is_derived_from_config(logger: Logger) -> None:
    from datetime import timedelta

    generator = AnthropicReactGenerator(
        logger=logger,
        cache=CacheConfig(enabled=True, ttl=timedelta(hours=1)),
        client=AsyncAnthropic(api_key="offline"),
    )
    history = [Message(role=Role.USER, parts=[TextPart(text="doc")], cache_key="k")]
    block = generator._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())[
        "messages"
    ][0]["content"][-1]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_that_caching_disabled_omits_cache_control(logger: Logger) -> None:
    generator = AnthropicReactGenerator(
        logger=logger, cache=CacheConfig(enabled=False), client=AsyncAnthropic(api_key="offline")
    )
    history = [Message(role=Role.USER, parts=[TextPart(text="doc")], cache_key="k")]
    block = generator._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())[
        "messages"
    ][0]["content"][-1]
    assert "cache_control" not in block


def test_that_finish_reason_mapping_covers_stop_reasons(
    anthropic: AnthropicReactGenerator,
) -> None:
    assert anthropic._map_finish_reason("end_turn") == FinishReason.STOP
    assert anthropic._map_finish_reason("stop_sequence") == FinishReason.STOP
    assert (
        anthropic._map_finish_reason("tool_use") == FinishReason.STOP
    )  # builder derives TOOL_CALLS
    assert anthropic._map_finish_reason("max_tokens") == FinishReason.MAX_TOKENS
    assert anthropic._map_finish_reason("refusal") == FinishReason.CONTENT_FILTER
    assert anthropic._map_finish_reason("pause_turn") == FinishReason.PAUSE
    assert anthropic._map_finish_reason(None) == FinishReason.STOP


# ════════════════════════════ 2. LIVE INTEGRATION ══════════════════════════

LIVE = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live Anthropic integration tests",
)
LIVE_MODEL = "claude-haiku-4-5-20251001"


def _live_generator(logger: Logger, **kwargs: Any) -> AnthropicReactGenerator:
    return AnthropicReactGenerator(model=LIVE_MODEL, logger=logger, **kwargs)


@LIVE
async def test_that_live_anthropic_generates_text(logger: Logger) -> None:
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
async def test_that_live_anthropic_streams_text_deltas(logger: Logger) -> None:
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
async def test_that_live_anthropic_calls_a_tool(logger: Logger) -> None:
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
    assert call.id.startswith("toolu_")
    assert call.provider_data.get(ANTHROPIC_BLOCK_KEY)


@LIVE
async def test_that_live_anthropic_runs_a_full_react_loop(logger: Logger) -> None:
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
async def test_that_live_anthropic_makes_parallel_tool_calls(logger: Logger) -> None:
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
async def test_that_live_anthropic_reports_a_thinking_summary_when_enabled(logger: Logger) -> None:
    """Claude 4 returns a SUMMARY of its thinking (never verbatim), so the
    assistant message carries visible reasoning text. Anthropic does not report
    thinking tokens separately, so reasoning_tokens is 0."""
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
        reasoning=ReasoningConfig(enabled=True, budget_tokens=2048, visibility="summary"),
    )

    assert "20" in result.message.text  # Bob = 20
    assert result.message.reasoning.strip()  # summarized thinking text is present
    assert result.usage.reasoning_tokens == 0  # not separately reported by Anthropic


@LIVE
async def test_that_visibility_none_omits_the_thinking_summary_but_still_reasons(
    logger: Logger,
) -> None:
    """visibility="none" maps to display="omitted": Claude still reasons (and is
    billed for it) but returns no thinking text."""
    generator = _live_generator(logger)

    result = await generator.step(
        [
            Message(
                role=Role.USER,
                parts=[TextPart(text="Is 91 prime? Reason it out, then answer yes or no.")],
            )
        ],
        reasoning=ReasoningConfig(enabled=True, budget_tokens=2048, visibility="none"),
    )

    # No thinking summary surfaced...
    assert result.message.reasoning == ""
    # ...but the final answer is still correct (91 = 7 x 13, not prime).
    assert "no" in result.message.text.lower()


@LIVE
async def test_that_live_anthropic_streams_thinking_text(logger: Logger) -> None:
    generator = _live_generator(logger)

    streamed_reasoning: list[str] = []
    completed: StepResult | None = None
    async for event in generator.stream_step(
        [
            Message(
                role=Role.USER,
                parts=[TextPart(text="Briefly: is 91 prime? Reason it out.")],
            )
        ],
        reasoning=ReasoningConfig(enabled=True, budget_tokens=2048, visibility="full"),
    ):
        if isinstance(event, ReasoningDelta):
            streamed_reasoning.append(event.text)
        elif isinstance(event, StepCompleted):
            completed = event.result

    assert completed is not None
    assert streamed_reasoning, "expected streamed thinking deltas"
    assert completed.message.reasoning.strip()


@LIVE
async def test_that_live_thinking_and_tool_blocks_round_trip_through_a_second_call(
    logger: Logger,
) -> None:
    """The GOTCHA: a turn with a thinking block before a tool_use must replay
    with its signature intact, or Anthropic 400s. Run a real two-step loop with
    thinking enabled (tool_choice auto) and confirm the follow-up succeeds."""
    generator = _live_generator(logger)
    reasoning = ReasoningConfig(enabled=True, budget_tokens=1024, visibility="full")

    history: list[Message] = [
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris? Use the tool.")])
    ]
    first = await generator.step(history, [WEATHER_TOOL], tool_choice="auto", reasoning=reasoning)
    assert first.needs_tools
    history.append(first.message)

    # The assistant turn must carry a thinking block with a signature.
    reasoning_parts = [p for p in first.message.parts if isinstance(p, ReasoningPart)]
    assert reasoning_parts and reasoning_parts[0].signature

    # Re-encoding replays the thinking block verbatim, before the tool_use.
    re_encoded = generator._encode(
        history, [WEATHER_TOOL], "auto", system=None, reasoning=reasoning
    )
    assistant_blocks = re_encoded["messages"][1]["content"]
    assert assistant_blocks[0]["type"] == "thinking"
    assert assistant_blocks[0]["signature"]
    assert any(b["type"] == "tool_use" for b in assistant_blocks)

    call = first.tool_calls[0]
    history.append(
        Message(
            role=Role.TOOL,
            parts=[ToolResultPart(call_id=call.id, name=call.name, content={"temperature_c": 18})],
        )
    )
    # The follow-up replays the signed thinking block; it must not 400.
    second = await generator.step(history, [WEATHER_TOOL], tool_choice="auto", reasoning=reasoning)
    assert not second.needs_tools
    assert "18" in second.message.text


@LIVE
async def test_that_system_prompt_can_change_between_steps_around_a_tool_call(
    logger: Logger,
) -> None:
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

    # History surgery: rewrite the original question in place. (Anthropic
    # requires tool_result to immediately follow tool_use, so we edit existing
    # content rather than splicing a message between that pair.)
    history[0] = Message(
        role=Role.USER,
        parts=[TextPart(text="What's the weather in Paris? Reminder: answer in Celsius.")],
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
    reasoning = ReasoningConfig(enabled=True, budget_tokens=1024, visibility="full")

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
    saw_reasoning = False

    for _ in range(5):
        text_buf: list[str] = []
        completed: StepResult | None = None
        async for event in generator.stream_step(
            history, [WEATHER_TOOL], tool_choice="auto", system=system, reasoning=reasoning
        ):
            if isinstance(event, TextDelta):
                text_buf.append(event.text)
            elif isinstance(event, ReasoningDelta):
                saw_reasoning = True
            elif isinstance(event, ToolCallStarted):
                tool_calls_seen.append(event.name)
            elif isinstance(event, StepCompleted):
                completed = event.result

        assert completed is not None
        assert "".join(text_buf) == completed.message.text
        history.append(completed.message)

        if not completed.needs_tools:
            break

        for call in completed.tool_calls:
            dispatched.append(call.args.get("city", ""))
            history.append(Message(role=Role.TOOL, parts=[await dispatch(call)]))

    assert "get_weather" in tool_calls_seen
    assert any("paris" in city.lower() for city in dispatched)
    assert saw_reasoning  # full thinking was streamed
    final_text = history[-1].text.lower()
    assert "15" in final_text or "umbrella" in final_text or "rain" in final_text


@LIVE
async def test_that_live_anthropic_honors_a_nullable_tool_parameter(logger: Logger) -> None:
    generator = _live_generator(logger)
    tool = ToolSpec(
        name="set_value",
        description="Set a value, which may be null when unknown.",
        parameters=[ParameterSpec(name="value", type="string", nullable=True)],
    )

    result = await generator.step(
        [Message(role=Role.USER, parts=[TextPart(text="Call set_value with a null value.")])],
        [tool],
        tool_choice="required",
    )

    assert result.tool_calls[0].name == "set_value"
    assert result.tool_calls[0].args.get("value", "sentinel") is None


@LIVE
async def test_that_live_anthropic_caches_a_marked_prefix(logger: Logger) -> None:
    """A cache_key-marked prefix gets cache_control; a repeat call reads it from
    cache (observed via cached_input_tokens). Anthropic caching needs no explicit
    resource and silently no-ops below the minimum, so there is nothing to clean
    up and nothing to fail."""
    generator = _live_generator(logger, cache=CacheConfig(enabled=True))

    reference = "Reference material.\n" + ("Parlant is an agent framework. " * 700)
    history = [
        Message(role=Role.USER, parts=[TextPart(text=reference)], cache_key="parlant-doc-v1"),
        Message(role=Role.USER, parts=[TextPart(text="In one word, what is Parlant?")]),
    ]
    system = "Answer strictly from the provided reference material."

    await generator.step(history, system=system)  # creates the cache
    second = await generator.step(history, system=system)  # reads from cache

    assert second.finish_reason == FinishReason.STOP
    assert second.usage.cached_input_tokens > 0
    assert second.usage.input_tokens >= second.usage.cached_input_tokens


@LIVE
async def test_that_live_anthropic_step_can_be_cancelled(logger: Logger) -> None:
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
