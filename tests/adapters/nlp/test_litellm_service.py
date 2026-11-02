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

"""Tests for the LiteLLM NLP adapter.

Two layers:

1. Offline logic — request encoding, the tokenizer, and provider registration.
   No network, no mocked provider responses.

2. Live integration — real LiteLLM calls that prove cost is relayed correctly.
   Cost reporting can only be trusted against real responses (whether LiteLLM
   populates ``_hidden_params["response_cost"]``, whether it can price a model,
   what the usage block actually contains), so these deliberately hit the real
   API and assert a positive ``reported_cost`` — across OpenAI, Anthropic and
   Gemini, each through its own key. Each provider's test skips when its key is
   absent.
"""

import asyncio
import json
import os
import re

from typing import Any

import pytest

from parlant.core.container import Container

from parlant.adapters.nlp.litellm_service import (
    LiteLLMEmbedder,
    LiteLLMEstimatingTokenizer,
    LiteLLMReactGenerator,
    LiteLLMSchematicGenerator,
    register_litellm_models,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.model_routing import ModelImplementationRegistry, NullSecretLoader
from parlant.core.nlp.react import (
    FinishReason,
    Message,
    ParameterSpec,
    ReasoningConfig,
    Role,
    StepCompleted,
    StepResult,
    TextDelta,
    TextPart,
    ToolCallPart,
    ToolCallStarted,
    ToolResultPart,
    ToolSpec,
)
from parlant.core.tracer import Tracer


class SchemaData(DefaultBaseModel):
    test_field: str


# ════════════════════════════ 1. OFFLINE LOGIC ══════════════════════════════


def test_that_react_encode_maps_history_tools_and_tool_choice(container: Container) -> None:
    generator = LiteLLMReactGenerator(model="gpt-4o", logger=container[Logger])

    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="You are helpful")]),
        Message(role=Role.USER, parts=[TextPart(text="hi")]),
        Message(
            role=Role.ASSISTANT,
            parts=[ToolCallPart(id="c1", name="search", args={"q": "北京"})],
        ),
        Message(
            role=Role.TOOL,
            parts=[ToolResultPart(call_id="c1", name="search", content={"result": "已找到"})],
        ),
    ]
    tools = [
        ToolSpec(
            name="search",
            description="search the web",
            parameters=[ParameterSpec(name="q", description="query")],
        )
    ]

    request = generator._encode(history, tools, "auto", reasoning=ReasoningConfig())

    messages = request["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful"}
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "search"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"q": "北京"}'
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": '{"result": "已找到"}',
    }
    assert request["tools"][0]["function"]["name"] == "search"
    assert request["tool_choice"] == "auto"
    assert request["model"] == "gpt-4o"


def test_that_react_encode_sends_a_tool_result_without_a_matching_call_as_a_paired_call(
    container: Container,
) -> None:
    # Chat Completions providers reject a tool message that answers no tool call.
    generator = LiteLLMReactGenerator(model="gpt-4o", logger=container[Logger])
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

    messages = generator._encode(history, [], "auto", reasoning=ReasoningConfig())["messages"]

    issued: set[str] = set()
    results: list[dict[str, Any]] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            assert re.fullmatch(r"[A-Za-z0-9_-]+", call["function"]["name"])
            assert call["id"] not in issued
            issued.add(call["id"])
        if message["role"] == "tool":
            assert message["tool_call_id"] in issued
            results.append(message)
    assert len(results) == 2
    assert "Chongas are bright purple" in results[0]["content"]
    assert "built-in:rule-retriever-colour" in results[0]["content"]
    assert "Chongas taste of mango" in results[1]["content"]
    assert not any(m["role"] == "user" and "Chongas" in json.dumps(m) for m in messages)


async def test_that_the_estimating_tokenizer_counts_tokens() -> None:
    tokenizer = LiteLLMEstimatingTokenizer("gpt-4o")

    count = await tokenizer.estimate_token_count("Hello, world! This is a token count test.")

    assert count > 0


def test_that_registration_exposes_the_litellm_provider(container: Container) -> None:
    registry = ModelImplementationRegistry(NullSecretLoader())

    register_litellm_models(registry, container)

    assert "litellm" in registry.providers


# ════════════════════════ 2. LIVE INTEGRATION (COST) ════════════════════════

# (LiteLLM model id, env var holding that provider's key). Each case skips when
# its key is not configured.
_LIVE_MODELS = [
    pytest.param("openai/gpt-4o-mini", "OPENAI_API_KEY", id="openai"),
    pytest.param("anthropic/claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY", id="anthropic"),
    pytest.param("gemini/gemini-3.1-flash-lite", "GEMINI_API_KEY", id="gemini"),
]

_LIVE_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def _require_key(env_key: str) -> None:
    if not os.environ.get(env_key):
        pytest.skip(f"{env_key} not set; skipping live LiteLLM test")


@pytest.mark.parametrize("model, env_key", _LIVE_MODELS)
async def test_that_live_litellm_schematic_reports_usage_and_cost(
    container: Container,
    model: str,
    env_key: str,
) -> None:
    _require_key(env_key)

    generator = LiteLLMSchematicGenerator[SchemaData](
        model_name=model,
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    result = await generator.do_generate(
        'Return a JSON object with exactly one field named "test_field" whose value is '
        'the string "hello".'
    )

    assert result.content.test_field
    assert result.info.usage.input_tokens > 0
    assert result.info.usage.output_tokens > 0
    # The point of the whole exercise: LiteLLM priced the real response and the
    # adapter surfaced that cost.
    assert float(result.info.usage.extra["reported_cost"]) > 0


@pytest.mark.parametrize("model, env_key", _LIVE_MODELS)
async def test_that_live_litellm_react_reports_usage_and_cost(
    container: Container,
    model: str,
    env_key: str,
) -> None:
    _require_key(env_key)

    generator = LiteLLMReactGenerator(model=model, logger=container[Logger])

    result = await generator.step(
        [Message(role=Role.USER, parts=[TextPart(text="Reply with the single word: hello")])]
    )

    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    # The React path derives cost from tokens (streaming responses don't carry
    # _hidden_params), so this covers the other cost source end-to-end.
    assert float(result.usage.extra["reported_cost"]) > 0


async def test_that_live_litellm_embedding_reports_usage_and_cost(
    container: Container,
) -> None:
    _require_key("OPENAI_API_KEY")

    embedder = LiteLLMEmbedder(
        model_name=_LIVE_EMBEDDING_MODEL,
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    result = await embedder.do_embed(["hello world"])

    assert len(result.vectors) == 1
    assert len(result.vectors[0]) > 0
    assert result.usage.input_tokens > 0
    assert float(result.usage.extra["reported_cost"]) > 0


# ═══════════════════════ 3. LIVE INTEGRATION (REACT MATRIX) ═════════════════
#
# The same live matrix the first-party adapters run (see
# test_openai_service.py), through LiteLLM's provider-neutral chat-completions
# translation, on one cheap tool-capable primary model. Cost coverage across
# providers lives in section 2; reasoning tests are absent (the primary model
# has no reasoning support), and there is no mid-stream-closure spy because
# litellm.acompletion exposes no client object to proxy.

_LIVE_REACT_MODEL = "openai/gpt-4o-mini"

_LIVE_WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters=[ParameterSpec(name="city", type="string", description="City name")],
)


def _live_react_generator(container: Container) -> LiteLLMReactGenerator:
    _require_key("OPENAI_API_KEY")
    return LiteLLMReactGenerator(model=_LIVE_REACT_MODEL, logger=container[Logger])


async def test_that_live_litellm_generates_text(container: Container) -> None:
    generator = _live_react_generator(container)

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


async def test_that_live_litellm_streams_text_deltas(container: Container) -> None:
    generator = _live_react_generator(container)

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


async def test_that_live_litellm_calls_a_tool(container: Container) -> None:
    generator = _live_react_generator(container)

    result = await generator.step(
        [
            Message(
                role=Role.USER, parts=[TextPart(text="Use the tool to get the weather in Paris.")]
            )
        ],
        [_LIVE_WEATHER_TOOL],
        tool_choice="auto",
    )

    assert result.needs_tools
    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert call.args.get("city", "").lower().startswith("paris")
    assert call.id


async def test_that_live_litellm_runs_a_full_react_loop(container: Container) -> None:
    generator = _live_react_generator(container)

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        return ToolResultPart(
            call_id=call.id,
            name=call.name,
            content={"temperature_c": 18, "conditions": "sunny"},
        )

    history: list[Message] = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Use tools when needed, then answer.")]),
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris right now?")]),
    ]
    final_history = await generator.run(
        history,
        [_LIVE_WEATHER_TOOL],
        dispatch,
        max_steps=5,
    )

    assert any(m.role == Role.TOOL for m in final_history)
    assert final_history[-1].role == Role.ASSISTANT
    assert "18" in final_history[-1].text or "sunny" in final_history[-1].text.lower()


async def test_that_live_litellm_makes_parallel_tool_calls(container: Container) -> None:
    generator = _live_react_generator(container)

    result = await generator.step(
        [
            Message(
                role=Role.USER,
                parts=[TextPart(text="Get the weather for BOTH Paris and Tokyo using the tool.")],
            )
        ],
        [_LIVE_WEATHER_TOOL],
        tool_choice="required",
    )

    cities = sorted(call.args.get("city", "").lower() for call in result.tool_calls)
    assert len(result.tool_calls) >= 2
    assert any("paris" in c for c in cities)
    assert any("tokyo" in c for c in cities)


async def test_that_live_litellm_accepts_a_nullable_tool_parameter(container: Container) -> None:
    generator = _live_react_generator(container)
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
    assert call.args["value"] is None or isinstance(call.args.get("value"), str)


async def test_that_system_prompt_can_change_between_steps_around_a_tool_call(
    container: Container,
) -> None:
    generator = _live_react_generator(container)
    base_system = "You are a weather assistant. Use the get_weather tool when asked about weather."

    history: list[Message] = [
        Message(role=Role.SYSTEM, parts=[TextPart(text=base_system)]),
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")]),
    ]

    first = await generator.step(history, [_LIVE_WEATHER_TOOL], tool_choice="required")
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
    history[0] = Message(role=Role.SYSTEM, parts=[TextPart(text=modified_system)])
    second = await generator.step(history, [_LIVE_WEATHER_TOOL], tool_choice="auto")

    assert not second.needs_tools
    assert second.finish_reason == FinishReason.STOP
    assert "15" in second.message.text or "sunny" in second.message.text.lower()
    assert "PINEAPPLE" in second.message.text.upper()


async def test_that_live_history_can_be_edited_between_manual_steps(
    container: Container,
) -> None:
    generator = _live_react_generator(container)

    history: list[Message] = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Use the tool, then answer the user.")]),
        Message(role=Role.USER, parts=[TextPart(text="What's the weather in Paris?")]),
    ]
    first = await generator.step(history, [_LIVE_WEATHER_TOOL], tool_choice="required")
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

    # History surgery: inject an extra user instruction before the assistant's
    # tool-call turn. On the chat-completions wire an assistant tool-call turn
    # must be IMMEDIATELY followed by its tool results, so inserting between
    # the call and its result is not part of this adapter's editing contract.
    history.insert(
        2, Message(role=Role.USER, parts=[TextPart(text="(Reminder: answer in Celsius.)")])
    )

    second = await generator.step(history, [_LIVE_WEATHER_TOOL], tool_choice="auto")
    assert not second.needs_tools
    assert "21" in second.message.text


async def test_that_live_e2e_streams_and_runs_tools_to_a_final_answer(
    container: Container,
) -> None:
    generator = _live_react_generator(container)

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
            parts=[TextPart(text="Use the tool to look things up, then answer.")],
        ),
        Message(
            role=Role.USER,
            parts=[
                TextPart(
                    text="What's the weather in Paris, and should I take an umbrella? "
                    "Use the tool first."
                )
            ],
        ),
    ]

    tool_calls_seen: list[str] = []
    dispatched: list[str] = []
    final_answer = ""
    reached_final_answer = False

    for _ in range(5):
        text_buf: list[str] = []
        completed: StepResult | None = None
        async for event in generator.stream_step(history, [_LIVE_WEATHER_TOOL], tool_choice="auto"):
            if isinstance(event, TextDelta):
                text_buf.append(event.text)
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

    assert "get_weather" in tool_calls_seen
    assert any("paris" in city.lower() for city in dispatched)
    assert reached_final_answer, "loop never reached a tool-free final message"
    assert history[-1].role == Role.ASSISTANT
    assert not history[-1].tool_calls
    assert final_answer.strip(), "the final message was empty"
    grounded = final_answer.lower()
    assert "15" in grounded or "umbrella" in grounded or "rain" in grounded


async def test_that_live_litellm_step_can_be_cancelled(container: Container) -> None:
    generator = _live_react_generator(container)

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
