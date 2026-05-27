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

"""Tests for the Gemini adapter (parlant.adapters.nlp.gemini_service.GeminiReactGenerator).

The provider-agnostic ReAct core (TurnBuilder, step/stream_step/run, usage,
history editing, cancellation, schema building) is tested in
tests/core/nlp/test_react.py. This file covers Gemini specifics:

1. Gemini request-building tests check ``_encode`` and the pure
   ``_map_finish_reason`` transform offline. No API response is fabricated.
2. Live integration tests run against gemini-3.1-flash-lite, hitting the real
   Gemini API, and are skipped unless GEMINI_API_KEY is set. All Gemini decode
   behavior (text, reasoning, tool calls, usage, signatures) is verified here.
"""

import asyncio
import os
from datetime import timedelta
from typing import Any

import google.genai
import google.genai.types as genai_types
import pytest

from parlant.adapters.nlp.gemini_service import (
    GEMINI_THOUGHT_SIGNATURE_KEY,
    GeminiReactGenerator,
)
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.tracer import LocalTracer
from parlant.core.nlp.react import (
    CacheConfig,
    FinishReason,
    Message,
    ParameterSpec,
    ReasoningConfig,
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


# ════════════════════════ 2. GEMINI ENCODE / DECODE ═════════════════════════


@pytest.fixture
def gemini(logger: Logger) -> GeminiReactGenerator:
    # _encode and _map_finish_reason are pure transforms that never call the
    # client, so a client built with a throwaway key keeps these tests offline
    # without mocking any API response.
    return GeminiReactGenerator(
        model="gemini-3.1-flash-lite",
        logger=logger,
        client=google.genai.Client(api_key="offline-encode-tests"),
    )


def test_that_encode_maps_roles_and_folds_system_messages(gemini: GeminiReactGenerator) -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Extra system rule.")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(role=Role.ASSISTANT, parts=[TextPart(text="hello")]),
        Message(
            role=Role.TOOL,
            parts=[ToolResultPart(call_id="c1", name="get_weather", content="sunny")],
        ),
    ]

    request = gemini._encode(
        history, [], "auto", system="You are a test agent.", reasoning=ReasoningConfig()
    )

    assert "You are a test agent." in request["system_instruction"]
    assert "Extra system rule." in request["system_instruction"]
    assert [c.role for c in request["all_contents"]] == ["user", "model", "tool"]
    function_response = request["all_contents"][2].parts[0].function_response
    assert function_response.name == "get_weather"
    assert function_response.response == {"result": "sunny"}


def test_that_encode_preserves_thought_signature_on_tool_calls(
    gemini: GeminiReactGenerator,
) -> None:
    history = [
        Message(
            role=Role.ASSISTANT,
            parts=[
                ToolCallPart(
                    id="c1",
                    name="get_weather",
                    args={"city": "Paris"},
                    provider_data={GEMINI_THOUGHT_SIGNATURE_KEY: b"signature-bytes"},
                )
            ],
        )
    ]

    request = gemini._encode(
        history, [WEATHER_TOOL], "auto", system=None, reasoning=ReasoningConfig()
    )
    encoded_part = request["all_contents"][0].parts[0]

    assert encoded_part.function_call.name == "get_weather"
    assert encoded_part.function_call.args == {"city": "Paris"}
    assert encoded_part.thought_signature == b"signature-bytes"


def test_that_encode_wraps_non_object_tool_results_and_passes_objects_through(
    gemini: GeminiReactGenerator,
) -> None:
    history = [
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(call_id="c1", name="t", content="plain string"),
                ToolResultPart(call_id="c2", name="t", content={"temp": 18}),
            ],
        )
    ]

    parts = gemini._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())[
        "all_contents"
    ][0].parts

    assert parts[0].function_response.response == {"result": "plain string"}
    assert parts[1].function_response.response == {"temp": 18}


@pytest.mark.parametrize(
    "tool_choice, expected_mode, expected_allowed",
    [
        ("auto", genai_types.FunctionCallingConfigMode.AUTO, None),
        ("none", genai_types.FunctionCallingConfigMode.NONE, None),
        ("required", genai_types.FunctionCallingConfigMode.ANY, None),
        ({"name": "get_weather"}, genai_types.FunctionCallingConfigMode.ANY, ["get_weather"]),
    ],
)
def test_that_encode_maps_tool_choice(
    gemini: GeminiReactGenerator,
    tool_choice: Any,
    expected_mode: genai_types.FunctionCallingConfigMode,
    expected_allowed: list[str] | None,
) -> None:
    request = gemini._encode(
        [], [WEATHER_TOOL], tool_choice, system=None, reasoning=ReasoningConfig()
    )
    config = request["tool_config"].function_calling_config

    assert config.mode == expected_mode
    assert config.allowed_function_names == expected_allowed


def test_that_encode_emits_thinking_config_only_when_reasoning_enabled(
    gemini: GeminiReactGenerator,
) -> None:
    # Reasoning is per-call: a disabled config emits no thinking_config...
    disabled = gemini._encode([], [], "auto", system=None, reasoning=ReasoningConfig())
    assert disabled["thinking_config"] is None

    # ...while an enabled one does.
    enabled = gemini._encode(
        [],
        [],
        "auto",
        system=None,
        reasoning=ReasoningConfig(enabled=True, budget_tokens=2048, visibility="summary"),
    )
    thinking = enabled["thinking_config"]
    assert thinking is not None
    assert thinking.include_thoughts is True
    assert thinking.thinking_budget == 2048


def _offline_generator(logger: Logger, cache: CacheConfig | None = None) -> GeminiReactGenerator:
    return GeminiReactGenerator(
        logger=logger, cache=cache, client=google.genai.Client(api_key="offline")
    )


def test_that_a_cache_marker_splits_history_into_prefix_and_suffix(logger: Logger) -> None:
    generator = _offline_generator(logger, CacheConfig(enabled=True))
    history = [
        Message(role=Role.USER, parts=[TextPart(text="big reference doc")], cache_key="doc-v1"),
        Message(role=Role.USER, parts=[TextPart(text="the live question")]),
    ]

    request = generator._encode(history, [], "auto", system="sys", reasoning=ReasoningConfig())

    assert [c.parts[0].text for c in request["prefix_contents"]] == ["big reference doc"]
    assert [c.parts[0].text for c in request["suffix_contents"]] == ["the live question"]
    assert request["cache_key"] == "doc-v1"


def test_that_caching_disabled_ignores_cache_markers(logger: Logger) -> None:
    generator = _offline_generator(logger, CacheConfig(enabled=False))
    history = [
        Message(role=Role.USER, parts=[TextPart(text="doc")], cache_key="doc-v1"),
        Message(role=Role.USER, parts=[TextPart(text="q")]),
    ]

    request = generator._encode(history, [], "auto", system=None, reasoning=ReasoningConfig())

    assert request["prefix_contents"] is None
    assert request["cache_key"] is None
    assert len(request["all_contents"]) == 2


def test_that_explicit_cache_name_is_read_from_provider_options(logger: Logger) -> None:
    generator = _offline_generator(
        logger, CacheConfig(provider_options={"gemini_cached_content": "cachedContents/abc"})
    )

    request = generator._encode(
        [Message(role=Role.USER, parts=[TextPart(text="q")])],
        [],
        "auto",
        system=None,
        reasoning=ReasoningConfig(),
    )

    assert request["explicit_cache_name"] == "cachedContents/abc"


def test_that_finish_reason_mapping_is_total_over_gemini_reasons(
    gemini: GeminiReactGenerator,
) -> None:
    # Pure enum->enum mapping; no API response is constructed or mocked.
    assert gemini._map_finish_reason(genai_types.FinishReason.STOP) == FinishReason.STOP
    assert gemini._map_finish_reason(genai_types.FinishReason.MAX_TOKENS) == FinishReason.MAX_TOKENS
    assert gemini._map_finish_reason(genai_types.FinishReason.SAFETY) == FinishReason.CONTENT_FILTER
    assert (
        gemini._map_finish_reason(genai_types.FinishReason.PROHIBITED_CONTENT)
        == FinishReason.CONTENT_FILTER
    )
    assert (
        gemini._map_finish_reason(genai_types.FinishReason.MALFORMED_FUNCTION_CALL)
        == FinishReason.ERROR
    )
    # Unknown reasons degrade to STOP rather than raising.
    assert gemini._map_finish_reason(genai_types.FinishReason.RECITATION) == FinishReason.STOP


# ════════════════════════════ 3. LIVE INTEGRATION ══════════════════════════

LIVE = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping live Gemini integration tests",
)
LIVE_MODEL = "gemini-3.1-flash-lite"


def _live_generator(logger: Logger, **kwargs: Any) -> GeminiReactGenerator:
    return GeminiReactGenerator(model=LIVE_MODEL, logger=logger, **kwargs)


@LIVE
async def test_that_live_gemini_generates_text(logger: Logger) -> None:
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
async def test_that_live_gemini_streams_text_deltas(logger: Logger) -> None:
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
async def test_that_live_gemini_calls_a_tool(logger: Logger) -> None:
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
    # Gemini attaches a thought_signature to the tool-call turn.
    assert call.provider_data.get(GEMINI_THOUGHT_SIGNATURE_KEY)


@LIVE
async def test_that_live_gemini_runs_a_full_react_loop(logger: Logger) -> None:
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
    # Usage was reported for every step.
    total = sum(step_usages, Usage())
    assert total.input_tokens > 0


@LIVE
async def test_that_live_e2e_streams_thoughts_and_runs_tools_to_a_final_answer(
    logger: Logger,
) -> None:
    """Full streaming e2e: a manual ReAct loop that streams each step's
    intermediate events (reasoning summaries, text, tool calls), runs the tool,
    feeds the result back, and reaches a grounded final answer.

    Note: gemini-3.1-flash-lite reliably *spends* reasoning tokens but only
    sometimes streams a visible thought summary, so the summary text is captured
    opportunistically while the reasoning spend is asserted.
    """
    generator = _live_generator(logger)
    system = "Use the tool to look things up, reason about the result, then answer the user."
    reasoning = ReasoningConfig(enabled=True, budget_tokens=4096, visibility="summary")

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

    # Per-step records of what the stream surfaced.
    reasoning_summaries: list[str] = []
    streamed_text_per_step: list[str] = []
    tool_calls_seen: list[str] = []
    step_usages: list[Usage] = []
    dispatched: list[str] = []

    for _ in range(5):
        reasoning_buf: list[str] = []
        text_buf: list[str] = []
        step_tool_calls: list[str] = []
        completed: StepResult | None = None

        async for event in generator.stream_step(
            history, [WEATHER_TOOL], tool_choice="auto", system=system, reasoning=reasoning
        ):
            if isinstance(event, ReasoningDelta):
                reasoning_buf.append(event.text)
            elif isinstance(event, TextDelta):
                text_buf.append(event.text)
            elif isinstance(event, ToolCallStarted):
                step_tool_calls.append(event.name)
            elif isinstance(event, StepCompleted):
                completed = event.result

        assert completed is not None, "every step must end with StepCompleted"

        if reasoning_buf:
            reasoning_summaries.append("".join(reasoning_buf))
        streamed_text_per_step.append("".join(text_buf))
        tool_calls_seen.extend(step_tool_calls)
        step_usages.append(completed.usage)

        # Streamed text deltas must reconstruct the assembled turn's text.
        assert "".join(text_buf) == completed.message.text

        history.append(completed.message)

        if not completed.needs_tools:
            break

        for call in completed.tool_calls:
            dispatched.append(call.args.get("city", ""))
            history.append(Message(role=Role.TOOL, parts=[await dispatch(call)]))

    # The tool was actually invoked, with Paris as the argument.
    assert "get_weather" in tool_calls_seen
    assert any("paris" in city.lower() for city in dispatched)

    # Thinking happened (reasoning tokens were spent across the loop), even if
    # the provider chose not to stream a visible summary on every step.
    total_usage = sum(step_usages, Usage())
    assert total_usage.reasoning_tokens > 0
    assert total_usage.input_tokens > 0
    assert total_usage.output_tokens >= total_usage.reasoning_tokens

    # Any thought summary that *was* streamed is non-empty.
    assert all(summary.strip() for summary in reasoning_summaries)

    # The loop ended on an assistant turn grounded in the tool result.
    assert history[-1].role == Role.ASSISTANT
    final_text = history[-1].text.lower()
    assert "15" in final_text or "umbrella" in final_text or "rain" in final_text


@LIVE
async def test_that_live_gemini_streams_a_visible_thought_summary(logger: Logger) -> None:
    """With a generous thinking budget and an explicit "explain your thinking"
    prompt, Gemini streams a visible thought summary. Assert it is actually
    present (not merely that reasoning tokens were spent), and that the streamed
    ReasoningDelta events reconstruct the assembled reasoning text."""
    generator = _live_generator(logger)

    prompt = (
        "Solve this step by step, showing your reasoning: A farmer has chickens and rabbits. "
        "There are 35 heads and 94 legs. How many of each? Explain your thinking."
    )

    streamed_reasoning: list[str] = []
    completed: StepResult | None = None
    async for event in generator.stream_step(
        [Message(role=Role.USER, parts=[TextPart(text=prompt)])],
        reasoning=ReasoningConfig(enabled=True, budget_tokens=8192, visibility="summary"),
    ):
        if isinstance(event, ReasoningDelta):
            streamed_reasoning.append(event.text)
        elif isinstance(event, StepCompleted):
            completed = event.result

    assert completed is not None

    # A visible thought summary was streamed...
    assert streamed_reasoning, "expected at least one ReasoningDelta with summary text"
    # ...and it reconstructs the assembled message's reasoning text.
    assert "".join(streamed_reasoning) == completed.message.reasoning
    assert completed.message.reasoning.strip()
    assert completed.usage.reasoning_tokens > 0

    # The reasoning led to the correct answer (23 chickens, 12 rabbits).
    assert "23" in completed.message.text and "12" in completed.message.text


@LIVE
async def test_that_live_history_can_be_edited_between_manual_steps(logger: Logger) -> None:
    """Drive the loop manually and rewrite history between real calls. The
    edited history (which keeps the tool-call turn and its preserved signature)
    must still be accepted by Gemini and produce a coherent answer."""
    generator = _live_generator(logger)
    system = "Use the tool, then answer the user."

    history: list[Message] = [
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")])
    ]

    first = await generator.step(history, [WEATHER_TOOL], tool_choice="required", system=system)
    assert first.needs_tools
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

    # History surgery: inject an extra instruction mid-conversation. The
    # assistant tool-call turn (with its thought_signature in provider_data) is
    # left untouched so the replay stays valid.
    history.insert(
        -1,
        Message(role=Role.USER, parts=[TextPart(text="(Reminder: answer in Celsius.)")]),
    )

    second = await generator.step(history, [WEATHER_TOOL], tool_choice="auto", system=system)

    assert not second.needs_tools
    assert "21" in second.message.text


@LIVE
async def test_that_system_prompt_can_change_between_steps_around_a_tool_call(
    logger: Logger,
) -> None:
    """Step N triggers a tool call; we run the tool; step N+1 uses a *modified*
    system prompt. The follow-up must succeed (no errors) AND visibly respect the
    changed instruction, proving system is honored per call on the same history."""
    generator = _live_generator(logger)
    base_system = "You are a weather assistant. Use the get_weather tool when asked about weather."

    history: list[Message] = [
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")])
    ]

    # Step N — with the base system prompt — triggers a tool call.
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

    # Step N+1 — same history, but a modified system prompt adding a distinctive,
    # unlikely-by-default instruction so we can detect that the change took hold.
    modified_system = (
        base_system + ' When you give your final answer, end it with the exact word "PINEAPPLE".'
    )
    second = await generator.step(
        history, [WEATHER_TOOL], tool_choice="auto", system=modified_system
    )

    # No errors, and the loop concluded with a normal answer...
    assert not second.needs_tools
    assert second.finish_reason == FinishReason.STOP
    # ...that incorporated the tool result...
    assert "15" in second.message.text or "sunny" in second.message.text.lower()
    # ...and respected the CHANGED system prompt.
    assert "PINEAPPLE" in second.message.text.upper()


@LIVE
async def test_that_live_gemini_makes_parallel_tool_calls(logger: Logger) -> None:
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
async def test_that_live_gemini_reports_reasoning_when_enabled(logger: Logger) -> None:
    generator = _live_generator(logger)

    result = await generator.step(
        [
            Message(
                role=Role.USER,
                parts=[
                    TextPart(text="A train covers 60 km in 45 minutes. What is its speed in km/h?")
                ],
            )
        ],
        reasoning=ReasoningConfig(enabled=True, budget_tokens=2048, visibility="summary"),
    )

    assert "80" in result.message.text
    # Thinking was enabled, so reasoning tokens must be reported...
    assert result.usage.reasoning_tokens > 0
    # ...and counted as a subset of output tokens.
    assert result.usage.output_tokens >= result.usage.reasoning_tokens


@LIVE
async def test_that_live_tool_call_signature_round_trips_through_encode(logger: Logger) -> None:
    """A real tool-call turn must re-encode with its thought_signature intact."""
    generator = _live_generator(logger)

    history: list[Message] = [
        Message(role=Role.USER, parts=[TextPart(text="Use the tool to get the weather in Paris.")])
    ]
    result = await generator.step(history, [WEATHER_TOOL], tool_choice="required")
    history.append(result.message)

    signature = result.tool_calls[0].provider_data.get(GEMINI_THOUGHT_SIGNATURE_KEY)
    assert signature, "expected Gemini to attach a thought_signature to the tool call"

    re_encoded = generator._encode(
        history, [WEATHER_TOOL], "auto", system=None, reasoning=ReasoningConfig()
    )
    assistant_content = re_encoded["all_contents"][1]
    assert assistant_content.role == "model"
    assert assistant_content.parts[0].function_call.name == "get_weather"
    assert assistant_content.parts[0].thought_signature == signature


@LIVE
async def test_that_live_streamed_text_signature_is_preserved(logger: Logger) -> None:
    """Gemini attaches a thought_signature to the trailing text part of a turn;
    it must survive into the assembled assistant message."""
    generator = _live_generator(logger)

    result = await generator.step(
        [Message(role=Role.USER, parts=[TextPart(text="Name one ocean.")])],
        system="Answer in one sentence.",
    )

    signatures = [
        part.provider_data.get(GEMINI_THOUGHT_SIGNATURE_KEY)
        for part in result.message.parts
        if part.provider_data.get(GEMINI_THOUGHT_SIGNATURE_KEY)
    ]
    assert signatures, "expected a preserved thought_signature on the text turn"


@LIVE
async def test_that_live_gemini_step_can_be_cancelled(logger: Logger) -> None:
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
    """Cancel while consuming a live stream and confirm the underlying google-genai
    stream is actually closed (aclose'd), not leaked."""
    generator = _live_generator(logger)
    stream_closed = asyncio.Event()
    real_create = generator._client.aio.models.generate_content_stream

    class _StreamProxy:
        def __init__(self, real: Any) -> None:
            self._real = real

        def __aiter__(self) -> Any:
            return self._real.__aiter__()

        async def aclose(self) -> None:
            stream_closed.set()
            aclose = getattr(self._real, "aclose", None)
            if aclose is not None:
                await aclose()

    async def spy_create(**kwargs: Any) -> Any:
        return _StreamProxy(await real_create(**kwargs))

    generator._client.aio.models.generate_content_stream = spy_create  # type: ignore[method-assign]

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
    await asyncio.wait_for(first_event.wait(), timeout=30)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(stream_closed.wait(), timeout=5)


@LIVE
async def test_that_live_managed_cache_is_created_reused_and_deleted(logger: Logger) -> None:
    """End-to-end explicit caching: a cache_key-marked prefix is turned into a
    real Gemini CachedContent, produces a cache hit, is reused across calls
    sharing the prefix, and is deleted by aclose()."""
    generator = _live_generator(logger, cache=CacheConfig(enabled=True, ttl=timedelta(seconds=120)))

    # The cached prefix must exceed Gemini's explicit-cache minimum token count.
    reference = "Reference material.\n" + ("Parlant is an agent framework. " * 600)
    history = [
        Message(role=Role.USER, parts=[TextPart(text=reference)], cache_key="parlant-doc-v1"),
        Message(role=Role.USER, parts=[TextPart(text="In one word, what is Parlant?")]),
    ]
    system = "Answer strictly from the provided reference material."

    try:
        first = await generator.step(history, system=system)
        assert first.finish_reason == FinishReason.STOP
        # The cached prefix was actually served from cache.
        assert first.usage.cached_input_tokens > 0
        assert len(generator._managed_caches) == 1

        # A second call with the same prefix reuses the same cache resource.
        second = await generator.step(history, system=system)
        assert second.usage.cached_input_tokens > 0
        assert len(generator._managed_caches) == 1
    finally:
        await generator.aclose()

    # aclose() deleted the managed cache.
    assert generator._managed_caches == {}


@LIVE
async def test_that_a_too_small_cache_prefix_degrades_gracefully(logger: Logger) -> None:
    """Asking to cache a prefix below Gemini's minimum token count must NOT fail
    the request — the generator falls back to an uncached inline call and still
    answers, and remembers the prefix is uncacheable to avoid retrying."""
    generator = _live_generator(logger, cache=CacheConfig(enabled=True))

    # Far below the ~1024-token explicit-cache minimum.
    history = [
        Message(role=Role.USER, parts=[TextPart(text="You are concise.")], cache_key="too-small"),
        Message(role=Role.USER, parts=[TextPart(text="What is the capital of France?")]),
    ]

    result = await generator.step(history)

    # The request succeeded despite caching being impossible...
    assert result.finish_reason == FinishReason.STOP
    assert "Paris" in result.message.text
    # ...no cache was created or used...
    assert result.usage.cached_input_tokens == 0
    assert generator._managed_caches == {}
    # ...and the prefix was remembered as uncacheable.
    assert generator._uncacheable_keys

    # A second call still works and doesn't re-attempt cache creation.
    second = await generator.step(history)
    assert "Paris" in second.message.text
    assert generator._managed_caches == {}
