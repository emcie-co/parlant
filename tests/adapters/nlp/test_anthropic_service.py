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
from unittest.mock import patch

import pytest
from anthropic import AsyncAnthropic
from lagom import Container

from parlant.adapters.nlp.anthropic_service import (
    ANTHROPIC_BLOCK_KEY,
    TURN_INSTRUCTIONS_OPEN,
    AnthropicReactGenerator,
    AnthropicService,
)
from parlant.core.engines.compass.guideline_matching.guideline_ranker import GuidelineRankSchema
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import Meter
from parlant.core.tracer import LocalTracer, Tracer
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
        Message(role=Role.SYSTEM, parts=[TextPart(text="You are a test agent.")]),
        Message(role=Role.SYSTEM, parts=[TextPart(text="Extra system rule.")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(role=Role.ASSISTANT, parts=[TextPart(text="hello")]),
        Message(
            role=Role.TOOL,
            parts=[ToolResultPart(call_id="tool_1", name="get_weather", content="sunny")],
        ),
    ]

    request = anthropic._encode(history, [], "auto", reasoning=ReasoningConfig())

    # System messages fold into the top-level system parameter.
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

    blocks = anthropic._encode(history, [WEATHER_TOOL], "auto", reasoning=ReasoningConfig())[
        "messages"
    ][0]["content"]

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

    block = anthropic._encode(history, [WEATHER_TOOL], "auto", reasoning=ReasoningConfig())[
        "messages"
    ][0]["content"][0]

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

    blocks = anthropic._encode(history, [], "auto", reasoning=ReasoningConfig())["messages"][0][
        "content"
    ]

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
    request = anthropic._encode([], [WEATHER_TOOL], tool_choice, reasoning=ReasoningConfig())
    assert request["tool_choice"] == expected


def test_that_minimal_effort_skips_the_thinking_block(
    anthropic: AnthropicReactGenerator,
) -> None:
    # On Sonnet/Haiku 4.5 manual mode, "minimal" effort means no thinking at
    # all — the request goes without a `thinking` block so the model runs in
    # standard mode at zero thinking tokens.
    minimal = anthropic._encode([], [], "auto", reasoning=ReasoningConfig(effort="minimal"))
    assert "thinking" not in minimal


def test_that_effort_maps_to_the_thinking_budget_ladder(
    anthropic: AnthropicReactGenerator,
) -> None:
    # Default effort ("medium") -> manual thinking with the medium-tier budget.
    medium = anthropic._encode([], [], "auto", reasoning=ReasoningConfig())
    assert medium["thinking"] == {
        "type": "enabled",
        "budget_tokens": anthropic._EFFORT_TO_BUDGET["medium"],
        "display": "summarized",
    }
    # max_tokens must exceed the thinking budget.
    assert medium["max_tokens"] > anthropic._EFFORT_TO_BUDGET["medium"]

    # Explicit "high" picks the higher tier.
    high = anthropic._encode([], [], "auto", reasoning=ReasoningConfig(effort="high"))
    assert high["thinking"]["budget_tokens"] == anthropic._EFFORT_TO_BUDGET["high"]


def test_that_service_tier_maps_to_anthropic_values(anthropic: AnthropicReactGenerator) -> None:
    # Anthropic only accepts "auto" / "standard_only" on requests, and has no
    # flex tier (it maps to standard).
    def tier_for(service_tier: str | None) -> str:
        hints = {"service_tier": service_tier} if service_tier else {}
        request = anthropic._encode([], [], "auto", reasoning=ReasoningConfig(), hints=hints)  # type: ignore[arg-type]
        encoded_tier: str = request["service_tier"]
        return encoded_tier

    assert tier_for(None) == "standard_only"
    assert tier_for("standard") == "standard_only"
    assert tier_for("flex") == "standard_only"
    assert tier_for("priority") == "auto"


def test_that_a_mid_conversation_system_message_is_inline_on_opus_4_8(logger: Logger) -> None:
    generator = AnthropicReactGenerator(
        model="claude-opus-4-8",
        logger=logger,
        client=AsyncAnthropic(api_key="offline-encode-tests"),
    )
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="main")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(role=Role.SYSTEM, parts=[TextPart(text="mid")]),
    ]

    request = generator._encode(history, [], "auto", reasoning=ReasoningConfig())

    # Leading system stays the top-level system field...
    assert request["system"] == "main"
    # ...and the mid-conversation system is an inline `system`-role turn.
    assert request["messages"][-1] == {
        "role": "system",
        "content": [{"type": "text", "text": "mid"}],
    }


def test_that_a_mid_conversation_system_message_rides_the_last_user_message_on_haiku(
    anthropic: AnthropicReactGenerator,
) -> None:
    # Haiku 4.5 has no inline system support, so a mid-conversation system message
    # is appended (wrapped) to the END of the last user message instead of folded
    # into the system block — keeping the system prompt stable and cacheable.
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="main")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(role=Role.SYSTEM, parts=[TextPart(text="mid")]),
    ]

    request = anthropic._encode(history, [], "auto", reasoning=ReasoningConfig())

    # No system-role messages; the leading system carries the protocol note.
    assert all(m["role"] != "system" for m in request["messages"])
    assert request["system"].startswith("main")
    assert "ADDITIONAL RESPONSE CONSIDERATIONS" in request["system"]
    # The mid-conversation instruction rides, wrapped, at the end of the user turn.
    last_user = request["messages"][-1]
    assert last_user["role"] == "user"
    assert last_user["content"][-1]["text"].startswith(TURN_INSTRUCTIONS_OPEN)
    assert "mid" in last_user["content"][-1]["text"]


def test_that_an_inline_mid_conversation_system_is_not_a_cache_breakpoint(logger: Logger) -> None:
    # On opus 4.8 the mid-conversation system stays inline, but it is dynamic, so
    # the cache breakpoint must stay on the last real message — not the inline
    # system — and the inline system must not carry cache_control.
    generator = AnthropicReactGenerator(
        model="claude-opus-4-8",
        logger=logger,
        cache=CacheConfig(enabled=True),
        client=AsyncAnthropic(api_key="offline-encode-tests"),
    )
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="main")], cache_key="s"),
        Message(role=Role.USER, parts=[TextPart(text="hi")], cache_key="s"),
        Message(role=Role.SYSTEM, parts=[TextPart(text="mid")], cache_key="s"),
    ]

    request = generator._encode(history, [], "auto", reasoning=ReasoningConfig())

    # Leading system is the cached block.
    assert request["system"] == [
        {"type": "text", "text": "main", "cache_control": {"type": "ephemeral"}}
    ]
    # The breakpoint is on the last real message (the user), not the inline system.
    assert "cache_control" in request["messages"][0]["content"][-1]
    assert request["messages"][-1] == {
        "role": "system",
        "content": [{"type": "text", "text": "mid"}],
    }


def test_that_a_mid_conversation_system_keeps_the_system_a_single_cached_block(
    anthropic: AnthropicReactGenerator,
) -> None:
    # With caching on, the mid-conversation instruction must NOT enter the system
    # block (which stays one cached block) — it rides at the tail of the last user
    # message, past the cache breakpoint, so the cached prefix is unaffected.
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="main")], cache_key="s"),
        Message(role=Role.USER, parts=[TextPart(text="hi")], cache_key="s"),
        Message(role=Role.SYSTEM, parts=[TextPart(text="mid")], cache_key="s"),
    ]

    request = anthropic._encode(history, [], "auto", reasoning=ReasoningConfig())

    # System is a single cached block: leading system + protocol note.
    assert len(request["system"]) == 1
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["system"][0]["text"].startswith("main")
    assert "ADDITIONAL RESPONSE CONSIDERATIONS" in request["system"][0]["text"]
    # The breakpoint stays on the user message's real content...
    last_user = request["messages"][-1]
    assert last_user["role"] == "user"
    assert "cache_control" in last_user["content"][0]
    # ...and the wrapped instruction is appended after it, uncached.
    assert last_user["content"][-1]["text"].startswith(TURN_INSTRUCTIONS_OPEN)
    assert "cache_control" not in last_user["content"][-1]
    assert "mid" in last_user["content"][-1]["text"]


def test_that_prefill_request_appends_an_uncached_dummy_and_caps_output(
    anthropic: AnthropicReactGenerator,
) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="sys")], cache_key="s"),
        Message(role=Role.USER, parts=[TextPart(text="hi")], cache_key="s"),
        Message(role=Role.ASSISTANT, parts=[TextPart(text="hello")], cache_key="s"),
    ]
    encoded = anthropic._encode(history, [], "auto", reasoning=ReasoningConfig())
    prefill = anthropic._build_prefill_request(encoded)

    assert prefill["max_tokens"] == 1
    assert "thinking" not in prefill
    # The cached prefix is unchanged (so the real call will hit this cache).
    assert prefill["system"] == encoded["system"]
    assert prefill["messages"][: len(encoded["messages"])] == encoded["messages"]
    # A dummy user turn is appended, and it is NOT cached.
    assert prefill["messages"][-1]["role"] == "user"
    assert "cache_control" not in prefill["messages"][-1]["content"][-1]


def test_that_prefill_skips_the_dummy_when_history_ends_with_a_user_turn(
    anthropic: AnthropicReactGenerator,
) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="sys")], cache_key="s"),
        Message(role=Role.USER, parts=[TextPart(text="hi")], cache_key="s"),
    ]
    encoded = anthropic._encode(history, [], "auto", reasoning=ReasoningConfig())
    prefill = anthropic._build_prefill_request(encoded)

    # No extra dummy turn — the existing user turn ends the array.
    assert len(prefill["messages"]) == len(encoded["messages"])
    assert prefill["messages"][-1]["role"] == "user"
    assert prefill["max_tokens"] == 1


def test_that_min_cache_size_is_known_per_model_family(
    anthropic: AnthropicReactGenerator,
) -> None:
    assert anthropic._min_cache_size("claude-haiku-4-5-20251001") == 2048
    assert anthropic._min_cache_size("claude-sonnet-4-6") == 1024
    assert anthropic._min_cache_size("claude-opus-4-8") == 1024
    assert anthropic._min_cache_size("some-unknown-model") > 2048


async def test_that_a_short_prefix_is_not_prefilled(anthropic: AnthropicReactGenerator) -> None:
    # Tokens estimated locally (gpt-5), so this needs no network. A tiny prompt
    # is well below the cache minimum.
    history = [Message(role=Role.SYSTEM, parts=[TextPart(text="be concise")])]

    assert await anthropic._should_prefill(history, [], {}) is False


async def test_that_a_prefix_above_the_cache_minimum_is_prefilled(
    anthropic: AnthropicReactGenerator,
) -> None:
    history = [Message(role=Role.SYSTEM, parts=[TextPart(text="word " * 4000)])]

    assert await anthropic._should_prefill(history, [], {}) is True


def test_that_visibility_maps_to_the_display_knob(anthropic: AnthropicReactGenerator) -> None:
    # Claude 4 has no verbatim option: "none" omits the thinking summary, while
    # "summary" and "full" both request the summary ("full" has no equivalent).
    def display_for(visibility: str) -> Any:
        request = anthropic._encode(
            [],
            [],
            "auto",
            reasoning=ReasoningConfig(visibility=visibility),  # type: ignore[arg-type]
        )
        return request["thinking"]["display"]

    assert display_for("none") == "omitted"
    assert display_for("summary") == "summarized"
    assert display_for("full") == "summarized"


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

    messages = anthropic._encode(history, [], "auto", reasoning=ReasoningConfig())["messages"]

    # The breakpoint message's last block carries cache_control...
    assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # ...and the live suffix does not.
    assert "cache_control" not in messages[1]["content"][-1]


def test_that_a_marked_system_message_caches_the_system(
    anthropic: AnthropicReactGenerator,
) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="big stable system")], cache_key="sys-v1"),
        Message(role=Role.USER, parts=[TextPart(text="q")]),
    ]

    request = anthropic._encode(history, [], "auto", reasoning=ReasoningConfig())

    # System is emitted as a single cache_control'd text block (not a plain string).
    assert len(request["system"]) == 1
    assert request["system"][0]["type"] == "text"
    assert request["system"][0]["text"].startswith("big stable system")
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    # The user message is not itself a cache breakpoint.
    assert "cache_control" not in request["messages"][0]["content"][-1]


def test_that_cache_control_ttl_is_derived_from_config(logger: Logger) -> None:
    from datetime import timedelta

    generator = AnthropicReactGenerator(
        logger=logger,
        cache=CacheConfig(enabled=True, ttl=timedelta(hours=1)),
        client=AsyncAnthropic(api_key="offline"),
    )
    history = [Message(role=Role.USER, parts=[TextPart(text="doc")], cache_key="k")]
    block = generator._encode(history, [], "auto", reasoning=ReasoningConfig())["messages"][0][
        "content"
    ][-1]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_that_caching_disabled_omits_cache_control(logger: Logger) -> None:
    generator = AnthropicReactGenerator(
        logger=logger, cache=CacheConfig(enabled=False), client=AsyncAnthropic(api_key="offline")
    )
    history = [Message(role=Role.USER, parts=[TextPart(text="doc")], cache_key="k")]
    block = generator._encode(history, [], "auto", reasoning=ReasoningConfig())["messages"][0][
        "content"
    ][-1]
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
async def test_that_live_anthropic_streams_text_deltas(logger: Logger) -> None:
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
        reasoning=ReasoningConfig(visibility="summary"),
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
        reasoning=ReasoningConfig(visibility="none"),
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
        reasoning=ReasoningConfig(visibility="full"),
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
    reasoning = ReasoningConfig(visibility="full")

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
    re_encoded = generator._encode(history, [WEATHER_TOOL], "auto", reasoning=reasoning)
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
        Message(role=Role.SYSTEM, parts=[TextPart(text=base_system)]),
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")]),
    ]
    # Anthropic rejects a forced tool_choice alongside a `thinking` block, so
    # the first (forced) call uses effort="minimal" to skip thinking entirely.
    first = await generator.step(
        history,
        [WEATHER_TOOL],
        tool_choice="required",
        reasoning=ReasoningConfig(effort="minimal"),
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
    # Forced tool_choice is incompatible with thinking on Anthropic — skip
    # thinking for this step.
    first = await generator.step(
        history,
        [WEATHER_TOOL],
        tool_choice="required",
        reasoning=ReasoningConfig(effort="minimal"),
    )
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

    second = await generator.step(history, [WEATHER_TOOL], tool_choice="auto")
    assert not second.needs_tools
    assert "21" in second.message.text


@LIVE
async def test_that_live_e2e_streams_thoughts_and_runs_tools_to_a_final_answer(
    logger: Logger,
) -> None:
    generator = _live_generator(logger)
    reasoning = ReasoningConfig(visibility="full")

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
    saw_reasoning = False
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
    assert saw_reasoning  # full thinking was streamed


@LIVE
async def test_that_live_anthropic_accepts_a_nullable_tool_parameter(logger: Logger) -> None:
    """The adapter translates ParameterSpec(nullable=True) to a JSON Schema
    null-union; the live API must accept that schema and invoke the tool, and a
    null value must decode to Python None. (Whether the model emits null or ""
    for a given turn is the model's choice, not the adapter's — so we assert the
    deterministic property: the nullable union is accepted and round-trips.)"""
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
    # The nullable union ["string", "null"] was accepted: value is present and is
    # a valid member of that type (None or a string), never missing or malformed.
    assert "value" in call.args
    assert call.args["value"] is None or isinstance(call.args["value"], str)


@LIVE
async def test_that_live_anthropic_caches_a_marked_prefix(logger: Logger) -> None:
    """A cache_key-marked prefix gets cache_control; a repeat call reads it from
    cache (observed via cached_input_tokens). Anthropic caching needs no explicit
    resource and silently no-ops below the minimum, so there is nothing to clean
    up and nothing to fail."""
    generator = _live_generator(logger, cache=CacheConfig(enabled=True))

    reference = "Reference material.\n" + ("Parlant is an agent framework. " * 700)
    history = [
        Message(
            role=Role.SYSTEM,
            parts=[TextPart(text="Answer strictly from the provided reference material.")],
        ),
        Message(role=Role.USER, parts=[TextPart(text=reference)], cache_key="parlant-doc-v1"),
        Message(role=Role.USER, parts=[TextPart(text="In one word, what is Parlant?")]),
    ]

    await generator.step(history)  # creates the cache
    second = await generator.step(history)  # reads from cache

    assert second.finish_reason == FinishReason.STOP
    assert second.usage.cached_input_tokens > 0
    assert second.usage.input_tokens >= second.usage.cached_input_tokens


@LIVE
async def test_that_live_anthropic_caches_a_marked_system_message(logger: Logger) -> None:
    """A cache_key on a system-role message caches the system prefix; a repeat
    call reads it from cache."""
    generator = _live_generator(logger, cache=CacheConfig(enabled=True))

    big_system = "Reference policy.\n" + ("Be precise and cite the policy. " * 700)
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text=big_system)], cache_key="policy-v1"),
        Message(role=Role.USER, parts=[TextPart(text="In one word, what should you cite?")]),
    ]

    await generator.step(history)  # creates the system cache
    second = await generator.step(history)  # reads it

    assert second.finish_reason == FinishReason.STOP
    assert second.usage.cached_input_tokens > 0


@LIVE
async def test_that_live_anthropic_step_can_be_cancelled(logger: Logger) -> None:
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
    """Cancel while consuming a live stream and confirm the underlying Anthropic
    message stream's async context is exited (closed), not leaked."""
    generator = _live_generator(logger)
    stream_closed = asyncio.Event()
    real_stream = generator._client.messages.stream

    class _ManagerProxy:
        def __init__(self, manager: Any) -> None:
            self._manager = manager

        async def __aenter__(self) -> Any:
            return await self._manager.__aenter__()

        async def __aexit__(self, *args: Any) -> Any:
            stream_closed.set()
            return await self._manager.__aexit__(*args)

    def spy_stream(**kwargs: Any) -> Any:
        return _ManagerProxy(real_stream(**kwargs))

    generator._client.messages.stream = spy_stream  # type: ignore[method-assign]

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
    await asyncio.wait_for(first_event.wait(), timeout=30)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(stream_closed.wait(), timeout=5)


async def test_that_the_guideline_ranker_is_served_by_haiku(container: Container) -> None:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"}):
        service = AnthropicService(
            container[Logger], container[Tracer], container[Meter], container[HealthReporter]
        )
        generator = await service.get_schematic_generator(GuidelineRankSchema)

    assert generator.model_name == "claude-haiku-4-5-20251001"
