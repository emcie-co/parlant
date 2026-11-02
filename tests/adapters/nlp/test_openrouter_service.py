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

from types import SimpleNamespace
from typing import Any, Callable, Generator
import json
import os

import httpx
from parlant.core.container import Container
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, patch, Mock
import asyncio
from openai import AsyncClient
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from parlant.adapters.nlp.openrouter_service import (  # type: ignore[reportMissingImports]
    _OPENROUTER_SCHEMATIC_FALLBACK_WARNINGS,
    _OPENROUTER_SCHEMATIC_OUTPUT_MODE_CACHE,
    OpenRouterDecisionMaker,
    OpenRouterSchematicGenerator,
    OpenRouterEmbedder,
    OpenRouterEstimatingTokenizer,
    OpenRouterReactGenerator,
    register_openrouter_models,
)
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import LocalMeter
from parlant.core.tracer import LocalTracer
from parlant.core.common import DefaultBaseModel
from parlant.core.nlp.decision import (
    Choice as DecisionChoice,
    ChoiceAnswer,
    DecisionError,
    Proposition,
    PropositionAnswer,
    PropositionCriteria,
    Scale,
    ScaleAnswer,
)
from parlant.core.nlp.model_routing import MissingProviderCredentialsError
from parlant.core.nlp.common import UsageInfo
from parlant.core.usage_reporter import UsageReporter
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.pricing import ModelPricing, Pricing
from parlant.core.nlp.model_routing import (
    ModelCapability,
    ModelConfig,
    ModelImplementationRegistry,
    NullSecretLoader,
)
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


class SchemaData(DefaultBaseModel):
    """Test schema for type checking."""

    test_field: str = "test_value"


# Captured before the autouse fixture below overrides OPENROUTER_API_KEY with a fake
# value — the live embedding test passes this real key explicitly.
_REAL_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

_OPENROUTER_LIVE = pytest.mark.skipif(
    not _REAL_OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY not set; skipping live OpenRouter embedding test",
)


@pytest.fixture(autouse=True)
def set_api_keys() -> Generator[None, None, None]:
    """Set API keys for tests that use container fixture."""
    # Container fixture initializes ServiceRegistry which requires OPENAI_API_KEY
    # OpenRouter tests also need OPENROUTER_API_KEY
    _OPENROUTER_SCHEMATIC_OUTPUT_MODE_CACHE.clear()
    _OPENROUTER_SCHEMATIC_FALLBACK_WARNINGS.clear()
    try:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai-key",
                "OPENROUTER_API_KEY": "test-openrouter-key",
            },
            clear=False,
        ):
            yield
    finally:
        _OPENROUTER_SCHEMATIC_OUTPUT_MODE_CACHE.clear()
        _OPENROUTER_SCHEMATIC_FALLBACK_WARNINGS.clear()


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5.4-nano",
        "anthropic/claude-sonnet-4",
    ],
)
def test_that_openrouter_react_always_encodes_mid_conversation_system_as_tool_result(
    model: str,
    container: Container,
) -> None:
    generator = OpenRouterReactGenerator(
        model=model,
        logger=container[Logger],
        provider={"sort": "latency", "allow_fallbacks": True},
        client=AsyncClient(
            api_key="offline",
            base_url="https://openrouter.ai/api/v1",
        ),
    )
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="main")]),
        Message(role=Role.USER, parts=[TextPart(text="hello")]),
        Message(role=Role.SYSTEM, parts=[TextPart(text="dynamic")]),
    ]

    request = generator._encode(history, [], "auto", reasoning=ReasoningConfig())

    assert request["model"] == model
    assert request["instructions"].startswith("main")
    assert "OCCASIONAL SYSTEM UPDATES" in request["instructions"]
    assert "dynamic" not in request["instructions"]
    assert all(item.get("role") != "developer" for item in request["input"])
    assert request["input"][-2] == {
        "type": "function_call",
        "call_id": "call_system_update",
        "name": "system_update",
        "arguments": "{}",
    }
    assert request["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_system_update",
        "output": "dynamic",
    }
    assert request["extra_body"] == {"provider": {"sort": "latency", "allow_fallbacks": True}}


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5.4-nano",
        "anthropic/claude-sonnet-4",
    ],
)
def test_that_openrouter_react_never_sends_a_function_call_output_without_its_function_call(
    model: str,
    container: Container,
) -> None:
    generator = OpenRouterReactGenerator(
        model=model,
        logger=container[Logger],
        client=AsyncClient(api_key="offline", base_url="https://openrouter.ai/api/v1"),
    )
    # The shape of a real turn with two staged retrievers: each result has no
    # call, and the engine numbers both "1".
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="main")]),
        Message(role=Role.USER, parts=[TextPart(text="How does onboarding work?")]),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(
                    call_id="1",
                    name="built-in:rule-retriever-how-it-works",
                    content="Onboarding takes two weeks",
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(
                    call_id="1",
                    name="built-in:rule-retriever-getting-started",
                    content="Start with a one-month audit",
                )
            ],
        ),
        Message(role=Role.SYSTEM, parts=[TextPart(text="dynamic")]),
    ]

    items = generator._encode(history, [], "auto", reasoning=ReasoningConfig())["input"]

    called: set[str] = set()
    for item in items:
        if item.get("type") == "function_call":
            called.add(item["call_id"])
        elif item.get("type") == "function_call_output":
            assert item["call_id"] in called
            called.remove(item["call_id"])
    encoded = json.dumps(items)
    assert "Onboarding takes two weeks" in encoded
    assert "Start with a one-month audit" in encoded


def test_that_openrouter_react_preserves_reported_cost(container: Container) -> None:
    generator = OpenRouterReactGenerator(
        model="openai/gpt-5.4-nano",
        logger=container[Logger],
        client=AsyncClient(api_key="offline", base_url="https://openrouter.ai/api/v1"),
    )
    usage = SimpleNamespace(
        input_tokens=11,
        input_tokens_details=SimpleNamespace(cached_tokens=4, cache_write_tokens=2),
        output_tokens=7,
        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        cost=1e-7,
    )

    decoded = generator._decode_usage(usage, "openai/gpt-5.4-nano")

    assert decoded.input_tokens == 11
    assert decoded.output_tokens == 7
    assert decoded.cached_input_tokens == 4
    assert decoded.extra == {
        "cache_write_input_tokens": 2,
        "reported_cost": "0.0000001",
    }


def test_that_openrouter_react_treats_the_second_initial_system_message_as_dynamic(
    container: Container,
) -> None:
    generator = OpenRouterReactGenerator(
        model="google/gemini-3.1-flash-lite",
        logger=container[Logger],
        client=AsyncClient(api_key="offline", base_url="https://openrouter.ai/api/v1"),
    )
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="stable")]),
        Message(role=Role.SYSTEM, parts=[TextPart(text="first-turn update")]),
        Message(role=Role.USER, parts=[TextPart(text="hello")]),
    ]

    request = generator._encode(history, [], "auto", reasoning=ReasoningConfig())

    assert "first-turn update" not in request["instructions"]
    assert request["input"][-1]["output"] == "first-turn update"


def test_that_openrouter_estimating_tokenizer_works(container: Container) -> None:
    """Test OpenRouterEstimatingTokenizer token estimation."""
    tokenizer = OpenRouterEstimatingTokenizer(model_name="openai/gpt-4o")
    tokens = asyncio.run(tokenizer.estimate_token_count("Hello world"))
    assert tokens > 0


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
def test_that_openrouter_generator_sets_custom_headers(mock_client_class: Mock) -> None:
    """Test that OpenRouter generator sets custom headers from environment."""
    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_HTTP_REFERER": "https://example.com",
            "OPENROUTER_SITE_NAME": "My App",
        },
        clear=True,
    ):
        logger = StdoutLogger(LocalTracer())
        _ = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=logger,
            tracer=LocalTracer(),
            meter=LocalMeter(logger),
        )

        # Verify client was called with headers
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert "default_headers" in call_args[1]
        assert call_args[1]["default_headers"]["HTTP-Referer"] == "https://example.com"
        assert call_args[1]["default_headers"]["X-Title"] == "My App"


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
def test_that_openrouter_generator_without_custom_headers(mock_client_class: Mock) -> None:
    """With nothing configured, only Parlant's own identity is sent."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        logger = StdoutLogger(LocalTracer())
        _ = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=logger,
            tracer=LocalTracer(),
            meter=LocalMeter(logger),
        )

        mock_client_class.assert_called_once()
        headers = mock_client_class.call_args[1]["default_headers"]

        # No deployment-supplied headers, and Parlant identifying itself: the
        # client marker, plus OpenRouter's own app attribution.
        assert headers == {
            "X-Parlant-Client": "parlant",
            "X-Title": "Parlant",
            "HTTP-Referer": "https://parlant.io",
        }


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_generator_falls_back_when_structured_output_is_unsupported(
    mock_client_class: Mock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client

    from openai import BadRequestError

    schema_error = BadRequestError(
        "Model does not support json_schema response_format",
        body={"error": {"message": "Structured output is unsupported"}},
        response=Mock(),
    )
    json_object_error = BadRequestError(
        "Model does not support JSON mode",
        body={"error": {"message": "json_object is unsupported"}},
        response=Mock(),
    )
    response = Mock(
        choices=[Mock(message=Mock(content='{"test_field": "fallback"}'))],
        usage=CompletionUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
    )
    mock_client.chat.completions.create.side_effect = [
        schema_error,
        json_object_error,
        response,
    ]

    class _WarningCountingLogger(StdoutLogger):
        def __init__(self) -> None:
            super().__init__(LocalTracer())
            self.warnings = 0

        def warning(self, message: str) -> None:
            self.warnings += 1
            super().warning(message)

    logger = _WarningCountingLogger()

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="test-model",
            logger=logger,
            tracer=container[Tracer],
            meter=container[Meter],
        )

    result = await generator.do_generate("Test prompt")

    assert result.content.test_field == "fallback"
    assert mock_client.chat.completions.create.await_count == 3
    first, second, third = mock_client.chat.completions.create.await_args_list
    assert first.kwargs["response_format"]["type"] == "json_schema"
    assert second.kwargs["response_format"] == {"type": "json_object"}
    assert "response_format" not in third.kwargs
    assert third.kwargs["messages"][0]["role"] == "system"
    assert logger.warnings == 2

    mock_client.chat.completions.create.reset_mock()
    mock_client.chat.completions.create.side_effect = None
    mock_client.chat.completions.create.return_value = response
    second_generator = OpenRouterSchematicGenerator[SchemaData](
        model_name="test-model",
        logger=logger,
        tracer=container[Tracer],
        meter=container[Meter],
    )

    cached_result = await second_generator.do_generate("Another prompt")

    assert cached_result.content.test_field == "fallback"
    mock_client.chat.completions.create.assert_awaited_once()
    cached_request = mock_client.chat.completions.create.await_args.kwargs
    assert "response_format" not in cached_request
    assert cached_request["messages"][0]["role"] == "system"
    assert logger.warnings == 2


async def test_that_openrouter_generator_handles_successful_response(
    container: Container,
) -> None:
    """Test OpenRouter generator with successful JSON response."""
    mock_response = Mock(spec=ChatCompletion)
    mock_response.choices = [
        Choice(
            message=ChatCompletionMessage(role="assistant", content='{"test_field": "test_value"}'),
            finish_reason="stop",
            index=0,
        )
    ]
    mock_response.usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost=0.004,
    )

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            provider={"sort": "latency"},
        )

        result = await generator.do_generate(
            'Generate {"test_field": "test_value"}',
            hints={"reasoning_effort": "low"},
        )

        assert result.content.test_field == "test_value"
        assert result.info.usage.input_tokens == 10
        assert result.info.usage.output_tokens == 20
        assert result.info.usage.extra["reported_cost"] == "0.004"
        response_format = mock_client.chat.completions.create.await_args.kwargs["response_format"]
        assert response_format == {
            "type": "json_schema",
            "json_schema": {
                "name": "SchemaData",
                "strict": True,
                "schema": {
                    **SchemaData.model_json_schema(),
                    "additionalProperties": False,
                },
            },
        }
        assert mock_client.chat.completions.create.await_args.kwargs["extra_body"] == {
            "provider": {"sort": "latency"},
            "reasoning": {"effort": "low"},
        }


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_embedder_retries_empty_embedding_value_error(
    mock_client_class: Mock,
    mock_sleep: AsyncMock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=[
            ValueError("No embedding data received"),
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])]),
        ]
    )
    mock_client_class.return_value = mock_client

    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    result = await embedder.do_embed(["hello"])

    assert result.vectors == [[0.1, 0.2, 0.3]]
    assert mock_client.embeddings.create.await_count == 2
    mock_sleep.assert_awaited_once()


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_embedder_retries_empty_embedding_response_data(
    mock_client_class: Mock,
    mock_sleep: AsyncMock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=[
            Mock(data=[]),
            Mock(data=[Mock(embedding=[0.4, 0.5])]),
        ]
    )
    mock_client_class.return_value = mock_client

    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    result = await embedder.do_embed(["hello"])

    assert result.vectors == [[0.4, 0.5]]
    assert mock_client.embeddings.create.await_count == 2
    mock_sleep.assert_awaited_once()


@_OPENROUTER_LIVE
async def test_that_openrouter_embedder_returns_usage(
    container: Container,
) -> None:
    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        api_key=_REAL_OPENROUTER_API_KEY,
    )

    result = await embedder.do_embed(["hello world", "goodbye world"])

    assert len(result.vectors) == 2
    assert all(len(vector) > 0 for vector in result.vectors)
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens == 0
    # OpenRouter reports the exact USD cost of the call.
    assert float(result.usage.extra["reported_cost"]) >= 0


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_embedder_passes_provider_preferences(
    mock_client_class: Mock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        return_value=Mock(
            data=[Mock(embedding=[0.1, 0.2])],
            usage=SimpleNamespace(prompt_tokens=17, cost=0.0002),
        )
    )
    mock_client_class.return_value = mock_client

    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        provider={"preferred_max_latency": {"p90": 1.5}},
    )

    await embedder.do_embed(["hello"])

    assert mock_client.embeddings.create.await_args.kwargs["extra_body"] == {
        "provider": {"preferred_max_latency": {"p90": 1.5}}
    }


async def test_that_openrouter_model_registry_accepts_provider_preferences(
    container: Container,
) -> None:
    registry = ModelImplementationRegistry(NullSecretLoader())
    register_openrouter_models(registry, container)

    generator = await registry.construct(
        config=ModelConfig(
            "openrouter:openai/gpt-5.4-nano",
            args={"provider": {"only": ["openai"], "allow_fallbacks": False}},
        ),
        capability=ModelCapability.REACT,
        route="react",
        scope="server",
    )

    assert isinstance(generator, OpenRouterReactGenerator)
    request = generator._encode(
        [Message(role=Role.USER, parts=[TextPart(text="hello")])],
        [],
        "auto",
        reasoning=ReasoningConfig(),
    )
    assert request["extra_body"] == {"provider": {"only": ["openai"], "allow_fallbacks": False}}


async def test_that_high_effort_replies_default_to_openai_gpt_6_1_terra(
    container: Container,
) -> None:
    registry = ModelImplementationRegistry(NullSecretLoader())
    register_openrouter_models(registry, container)

    config = registry.default_config(
        "openrouter",
        ModelCapability.REACT,
        route="react",
        qualifiers={"effort": "high"},
    )

    assert config.model == "openrouter:openai/gpt-6.1-terra"


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_embedder_does_not_retry_unrelated_value_error(
    mock_client_class: Mock,
    mock_sleep: AsyncMock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=ValueError("Embedding payload is malformed")
    )
    mock_client_class.return_value = mock_client

    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    with pytest.raises(ValueError, match="Embedding payload is malformed"):
        await embedder.do_embed(["hello"])

    assert mock_client.embeddings.create.await_count == 1
    mock_sleep.assert_not_awaited()


def test_that_openrouter_generator_supports_correct_parameters(container: Container) -> None:
    """Test supported OpenRouter parameters."""
    generator = OpenRouterSchematicGenerator[SchemaData](
        model_name="openai/gpt-4o",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    expected_params = ["temperature", "max_tokens"]
    assert generator.supported_openrouter_params == expected_params


# ════════════════════════ 2. LIVE COST ACCOUNTING ═══════════════════════════
#
# OpenRouter is the one provider that reports the money it actually charged
# (``usage.cost``, with usage accounting enabled) — the oracle that validates
# the whole accounting chain (mapping → rate card → formula) against real
# spend rather than our reading of the docs.
#
# The autouse ``set_api_keys`` fixture patches OPENROUTER_API_KEY to a fake
# during every test body, so the real key is captured at import time and
# injected explicitly.

_LIVE_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

LIVE = pytest.mark.skipif(
    not _LIVE_OPENROUTER_KEY,
    reason="OPENROUTER_API_KEY not set; skipping live OpenRouter integration tests",
)

# Stable, JSON-mode-capable, cheap — with a long-stable public rate card.
_LIVE_COST_MODEL = "openai/gpt-4o-mini"
_LIVE_COST_PRICING = ModelPricing(
    input_tokens=Decimal("0.15"),
    cached_input_tokens=Decimal("0.075"),
    output_tokens=Decimal("0.60"),
)


def _live_cost_generator(container: Container) -> OpenRouterSchematicGenerator[SchemaData]:
    return OpenRouterSchematicGenerator[SchemaData](
        model_name=_LIVE_COST_MODEL,
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        api_key=_LIVE_OPENROUTER_KEY,
    )


@LIVE
async def test_that_usage_accounting_reports_the_authoritative_cost(
    container: Container,
) -> None:
    generator = _live_cost_generator(container)

    result = await generator.generate(
        'Return a JSON object of the form {"test_field": "<a color name>"}.'
    )

    usage = result.info.usage
    reported = Decimal(str(usage.extra.get("reported_cost", "0")))
    assert reported > 0
    assert usage.input_tokens > 0 and usage.output_tokens > 0


@LIVE
async def test_that_token_derived_cost_reconciles_with_the_reported_cost(
    container: Container,
) -> None:
    generator = _live_cost_generator(container)

    result = await generator.generate(
        'Return a JSON object of the form {"test_field": "<a color name>"}.'
    )

    usage = result.info.usage
    reported = Decimal(str(usage.extra.get("reported_cost", "0")))
    assert reported > 0

    # calculate() prices only configured extra keys, so reported_cost itself
    # never double-counts into the derivation.
    derived = _LIVE_COST_PRICING.calculate(usage)
    reported_decimal = reported

    # Generous band: absorbs routing variance and small provider repricings
    # while still catching wrong-field mappings (those are off by integer
    # factors, not percentages).
    assert Decimal("0.5") * reported_decimal <= derived <= Decimal("2") * reported_decimal, (
        f"derived={derived} reported={reported} usage={usage!r}"
    )


# ─────────────────────────── live react matrix ───────────────────────────
#
# The same live matrix the first-party adapters run (see
# test_openai_service.py), through OpenRouter's OpenAI-compatible Responses
# API. The routed model is cheap, stable and tool-capable; reasoning tests
# are deliberately absent (the routed model has no reasoning support).

_LIVE_REACT_MODEL = "openai/gpt-4o-mini"

_LIVE_WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters=[ParameterSpec(name="city", type="string", description="City name")],
)


def _live_react_generator(container: Container) -> OpenRouterReactGenerator:
    return OpenRouterReactGenerator(
        model=_LIVE_REACT_MODEL,
        logger=container[Logger],
        api_key=_LIVE_OPENROUTER_KEY,
    )


@LIVE
async def test_that_live_openrouter_generates_text(container: Container) -> None:
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


@LIVE
async def test_that_live_openrouter_streams_text_deltas(container: Container) -> None:
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


@LIVE
async def test_that_live_openrouter_calls_a_tool(container: Container) -> None:
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


@LIVE
async def test_that_live_openrouter_runs_a_full_react_loop(container: Container) -> None:
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


# Before the fix, Claude silently ignored a result without its call and the
# OpenAI route rejected it with HTTP 400 (reproduced 2026-10-23).
@LIVE
@pytest.mark.parametrize("model", ["anthropic/claude-haiku-5.5", "openai/gpt-5.6-terra"])
async def test_that_live_openrouter_answers_from_a_tool_result_without_a_matching_call(
    container: Container, model: str
) -> None:
    generator = OpenRouterReactGenerator(
        model=model,
        logger=container[Logger],
        api_key=_LIVE_OPENROUTER_KEY,
    )

    result = await generator.step(
        [
            Message(
                role=Role.SYSTEM,
                parts=[
                    TextPart(
                        text="Answer only from information you've been given. "
                        "If you weren't given the answer, reply UNKNOWN."
                    )
                ],
            ),
            Message(role=Role.USER, parts=[TextPart(text="What colour are chongas?")]),
            Message(
                role=Role.TOOL,
                parts=[
                    ToolResultPart(
                        call_id="1",
                        name="built-in:rule-retriever-chongas",
                        content="Chongas are bright purple.",
                    )
                ],
            ),
        ]
    )

    assert "purple" in result.message.text.lower()


@LIVE
async def test_that_live_openrouter_makes_parallel_tool_calls(container: Container) -> None:
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


@LIVE
async def test_that_live_openrouter_accepts_a_nullable_tool_parameter(
    container: Container,
) -> None:
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


@LIVE
async def test_that_system_prompt_can_change_between_steps_around_a_tool_call(
    container: Container,
) -> None:
    """OpenRouter's encode funnels mid-conversation system changes through a
    synthetic tool call/result pair (the upstream model is opaque); the routed
    model must still visibly respect the change."""
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


@LIVE
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
    # tool-call turn. Unlike native OpenAI Responses, OpenRouter re-translates
    # to chat completions upstream, where an assistant tool-call turn must be
    # IMMEDIATELY followed by its tool results - so inserting between the call
    # and its result is rejected by the provider and is not part of this
    # adapter's editing contract.
    history.insert(
        2, Message(role=Role.USER, parts=[TextPart(text="(Reminder: answer in Celsius.)")])
    )

    second = await generator.step(history, [_LIVE_WEATHER_TOOL], tool_choice="auto")
    assert not second.needs_tools
    assert "21" in second.message.text


@LIVE
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


@LIVE
async def test_that_live_openrouter_step_can_be_cancelled(container: Container) -> None:
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


@LIVE
async def test_that_cancelling_mid_stream_closes_the_provider_stream(
    container: Container,
) -> None:
    generator = _live_react_generator(container)
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
    await asyncio.wait_for(first_event.wait(), timeout=30)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(stream_closed.wait(), timeout=5)


@LIVE
async def test_that_live_openrouter_embedder_returns_embeddings_and_usage(
    container: Container,
) -> None:
    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        api_key=_LIVE_OPENROUTER_KEY,
    )

    result = await embedder.do_embed(["hello world", "goodbye world"])

    assert len(result.vectors) == 2
    assert all(len(vector) == 1536 for vector in result.vectors)
    assert result.usage.input_tokens > 0


# --- Jev decisions (typesafe) ----------------------------------------------------
#
# Jev is not a chat model: it lives on OpenRouter's /api/alpha/decisions endpoint,
# takes a state plus named typed questions, and returns typed answers with
# probabilities. These tests pin the wire contract we depend on, since it is an
# alpha endpoint that can move under us.


def _decisions_client(
    handler: Callable[[httpx.Request], httpx.Response],
    requests: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    def record(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(record))


def _decision_maker(
    handler: Callable[[httpx.Request], httpx.Response],
    requests: list[httpx.Request] | None = None,
    model_name: str = "~typesafe/jev-latest",
) -> OpenRouterDecisionMaker:
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    return OpenRouterDecisionMaker(
        model_name=model_name,
        logger=logger,
        tracer=tracer,
        meter=LocalMeter(logger),
        http=_decisions_client(handler, requests),
    )


def _answer_response(
    answers: dict[str, Any],
    usage: dict[str, Any] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "typesafe/jev-1.13-20260917",
            "answers": answers,
            "usage": usage or {"input_tokens": 492, "output_tokens": 38, "cost": 0.000020664},
            "id": "gen-dec-1790099229-ahAseXX5gNJzoLiCZICn",
            "provider": "TypeSafe",
        },
    )


async def test_that_openrouter_decision_maker_posts_to_the_alpha_decisions_endpoint() -> None:
    # The decisions endpoint is NOT under /api/v1, so it cannot ride the OpenAI
    # client's base URL. Guard the exact URL, the auth header, and the model.
    requests: list[httpx.Request] = []
    maker = _decision_maker(
        lambda _: _answer_response({"q": {"type": "noul", "noul": 0.9}}),
        requests,
    )

    await maker.decide({"text": "hello"}, {"q": Proposition(instructions="Is it a greeting?")})

    (request,) = requests
    assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer test-openrouter-key"

    body = json.loads(request.content)
    assert body["model"] == "~typesafe/jev-latest"
    assert body["state"] == {"text": "hello"}


async def test_that_openrouter_decision_maker_maps_our_question_types_to_jev_primitives() -> None:
    requests: list[httpx.Request] = []
    maker = _decision_maker(
        lambda _: _answer_response(
            {
                "p": {"type": "noul", "noul": 0.1},
                "c": {
                    "type": "choice",
                    "choice": "refund",
                    "probabilities": {"refund": 0.8, "exchange": 0.2},
                    "confidence": 0.7,
                },
                "s": {
                    "type": "score",
                    "score": 2,
                    "legend": {"0": "low", "1": "mid", "2": "high"},
                    "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
                    "confidence": 1.0,
                },
            }
        ),
        requests,
    )

    await maker.decide(
        {"message": "I want my money back"},
        {
            "p": Proposition(instructions="Is the customer upset?"),
            "c": DecisionChoice(
                instructions="What do they want?",
                criteria={"refund": "asks for money back", "exchange": None},
            ),
            "s": Scale(instructions="How urgent?", criteria=["low", "mid", "high"]),
        },
    )

    (request,) = requests
    questions = json.loads(request.content)["questions"]

    # A proposition's criteria are optional, and absent here.
    assert questions["p"] == {"type": "noul", "instructions": "Is the customer upset?"}
    assert questions["c"] == {
        "type": "choice",
        "instructions": "What do they want?",
        "criteria": {"refund": "asks for money back", "exchange": None},
    }
    assert questions["s"] == {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["low", "mid", "high"],
    }


async def test_that_openrouter_decision_maker_sends_one_request_for_many_questions() -> None:
    # The whole point of adopting Jev: N judgments cost one round trip, where the
    # schematic generator costs N requests.
    requests: list[httpx.Request] = []
    maker = _decision_maker(
        lambda _: _answer_response({f"q{i}": {"type": "noul", "noul": 0.5} for i in range(25)}),
        requests,
    )

    result = await maker.decide(
        {"text": "state"},
        {f"q{i}": Proposition(instructions=f"Question {i}?") for i in range(25)},
    )

    assert len(requests) == 1
    assert len(result.answers) == 25


async def test_that_openrouter_decision_maker_parses_every_answer_type() -> None:
    maker = _decision_maker(
        lambda _: _answer_response(
            {
                "p": {"type": "noul", "noul": 0.05},
                "c": {
                    "type": "choice",
                    "choice": "b",
                    "probabilities": {"a": 0.25, "b": 0.75},
                    "confidence": 0.5,
                },
                "s": {
                    "type": "score",
                    "score": 1.5,
                    "legend": {"0": "broken", "1": "worn", "2": "mint"},
                    "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
                    "confidence": 0.4,
                },
            }
        )
    )

    result = await maker.decide(
        {},
        {
            "p": Proposition(instructions="?"),
            "c": DecisionChoice(instructions="?", criteria={"a": None, "b": None}),
            "s": Scale(instructions="?", criteria=["broken", "worn", "mint"]),
        },
    )

    proposition = result.answers["p"]
    assert isinstance(proposition, PropositionAnswer)
    assert proposition.probability == 0.05

    choice = result.answers["c"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.option == "b"
    assert choice.probabilities == {"a": 0.25, "b": 0.75}
    assert choice.confidence == 0.5

    scale = result.answers["s"]
    assert isinstance(scale, ScaleAnswer)
    # Jev's score is a probability-weighted level, so it need not be an integer.
    assert scale.score == 1.5
    assert scale.legend == {0: "broken", 1: "worn", 2: "mint"}
    assert scale.probabilities == {0: 0.0, 1: 0.5, 2: 0.5}
    assert scale.confidence == 0.4


async def test_that_openrouter_decision_maker_preserves_the_reported_cost() -> None:
    # OpenRouter's usage.cost is the authoritative charge; it must survive as a
    # fixed-point string, never exponent notation (see format_reported_cost).
    maker = _decision_maker(
        lambda _: _answer_response(
            {"q": {"type": "noul", "noul": 0.5}},
            usage={"input_tokens": 492, "output_tokens": 38, "cost": 1e-7},
        )
    )

    result = await maker.decide({}, {"q": Proposition(instructions="?")})

    assert result.info.usage.input_tokens == 492
    assert result.info.usage.output_tokens == 38
    assert result.info.usage.extra["reported_cost"] == "0.0000001"


async def test_that_openrouter_decision_maker_reports_one_request_for_a_batch() -> None:
    # Usage reports aggregate `requests`; batching only shows up as a saving if a
    # batched decision counts as one request rather than one per question.
    maker = _decision_maker(
        lambda _: _answer_response({f"q{i}": {"type": "noul", "noul": 0.5} for i in range(10)})
    )

    result = await maker.decide({}, {f"q{i}": Proposition(instructions="?") for i in range(10)})

    assert result.info.usage.requests == 1
    assert result.info.model == "typesafe/jev-1.13-20260917"


async def test_that_openrouter_decision_maker_reports_zero_cost_without_one() -> None:
    maker = _decision_maker(
        lambda _: _answer_response(
            {"q": {"type": "noul", "noul": 0.5}},
            usage={"input_tokens": 10, "output_tokens": 2},
        )
    )

    result = await maker.decide({}, {"q": Proposition(instructions="?")})

    assert "reported_cost" not in result.info.usage.extra


async def test_that_openrouter_decision_maker_raises_on_an_unrecognized_answer_type() -> None:
    # An alpha endpoint may grow primitives. Fail loudly rather than silently
    # dropping a judgment the caller is waiting on.
    maker = _decision_maker(lambda _: _answer_response({"q": {"type": "tensor", "tensor": [1, 2]}}))

    with pytest.raises(DecisionError):
        await maker.decide({}, {"q": Proposition(instructions="?")})


async def test_that_openrouter_decision_maker_raises_when_an_answer_is_missing() -> None:
    maker = _decision_maker(lambda _: _answer_response({"a": {"type": "noul", "noul": 0.5}}))

    with pytest.raises(DecisionError):
        await maker.decide(
            {},
            {
                "a": Proposition(instructions="?"),
                "b": Proposition(instructions="?"),
            },
        )


async def test_that_openrouter_decision_maker_surfaces_an_api_error() -> None:
    maker = _decision_maker(lambda _: httpx.Response(429, json={"error": "slow down"}))

    with pytest.raises(DecisionError) as caught:
        await maker.decide({}, {"q": Proposition(instructions="?")})

    assert "429" in str(caught.value)


async def test_that_openrouter_decision_maker_requires_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    with pytest.raises(MissingProviderCredentialsError):
        OpenRouterDecisionMaker(
            model_name="~typesafe/jev-latest",
            logger=logger,
            tracer=tracer,
            meter=LocalMeter(logger),
        )


@_OPENROUTER_LIVE
async def test_that_jev_latest_answers_a_batch_of_questions_live() -> None:
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    maker = OpenRouterDecisionMaker(
        model_name="~typesafe/jev-latest",
        logger=logger,
        tracer=tracer,
        meter=LocalMeter(logger),
        api_key=_REAL_OPENROUTER_API_KEY,
    )

    result = await maker.decide(
        {
            "conversation": [
                {"role": "customer", "content": "My package arrived smashed to bits."},
            ]
        },
        {
            "damaged": Proposition(instructions="Is the customer reporting a damaged item?"),
            "wants_refund": Proposition(instructions="Is the customer asking for a refund?"),
            "severity": Scale(
                instructions="How severe is the problem the customer describes?",
                criteria=["trivial", "mildly annoying", "a real problem", "severe"],
            ),
        },
    )

    damaged = result.answers["damaged"]
    assert isinstance(damaged, PropositionAnswer)
    assert damaged.probability > 0.5  # the message is unambiguous

    severity = result.answers["severity"]
    assert isinstance(severity, ScaleAnswer)
    assert 0 <= severity.score <= 3

    # One round trip for three judgments, with a real cost attached.
    assert result.info.usage.requests == 1
    assert result.info.usage.input_tokens > 0
    assert Decimal(str(result.info.usage.extra["reported_cost"])) > 0


async def test_that_openrouter_decision_maker_reports_its_usage_for_accounting() -> None:
    # Usage is reported by the port implementation, the way BaseSchematicGenerator
    # and BaseEmbedder do it — nothing downstream reports an evaluator's info, so a
    # decision that stays quiet here is free as far as pricing, budgets and cost
    # control are concerned.
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    reported: list[tuple[str, UsageInfo]] = []
    usage_reporter = UsageReporter(tracer)
    usage_reporter.add_listener(lambda _trace, model, usage: reported.append((model, usage)))

    maker = OpenRouterDecisionMaker(
        model_name="~typesafe/jev-latest",
        logger=logger,
        tracer=tracer,
        meter=LocalMeter(logger),
        usage_reporter=usage_reporter,
        http=_decisions_client(
            lambda _: _answer_response(
                {"q": {"type": "noul", "noul": 0.5}},
                usage={"input_tokens": 492, "output_tokens": 38, "cost": 0.000020664},
            )
        ),
    )

    await maker.decide({}, {"q": Proposition(instructions="?")})

    (model, usage) = reported[-1]
    # The dated build that served us, not the floating alias we asked for,
    # qualified by the provider that served it.
    assert model == "openrouter:typesafe/jev-1.13-20260917"
    assert usage.input_tokens == 492
    assert usage.extra["reported_cost"] == "0.000020664"
    assert usage.requests == 1


async def test_that_a_reported_decision_cost_prices_without_a_pricing_table_entry() -> None:
    # The pricing table is keyed by model name, and a dated decision build will
    # never be in it. That is fine: a provider-reported cost is authoritative and
    # needs no entry — this pins that the two halves actually fit together.
    maker = _decision_maker(
        lambda _: _answer_response(
            {"q": {"type": "noul", "noul": 0.5}},
            usage={"input_tokens": 492, "output_tokens": 38, "cost": 0.000020664},
        )
    )

    result = await maker.decide({}, {"q": Proposition(instructions="?")})

    empty_table = Pricing(currency="USD", models={})

    assert empty_table.calculate_model(result.info.model, result.info.usage) == Decimal(
        "0.000020664"
    )


async def test_that_a_propositions_criteria_are_sent_when_given() -> None:
    # The true/false descriptions are where a subtle boundary gets drawn — and
    # where the ranker puts a rule's signals and anti-signals.
    requests: list[httpx.Request] = []
    maker = _decision_maker(
        lambda _: _answer_response({"q": {"type": "noul", "noul": 0.5}}),
        requests,
    )

    await maker.decide(
        {"text": "I asked three times already"},
        {
            "q": Proposition(
                instructions="Has the customer contacted support about this before?",
                criteria=PropositionCriteria(
                    true="Mentions a prior attempt, ticket, or having asked before",
                    false="No sign of any previous contact",
                ),
            )
        },
    )

    (request,) = requests
    assert json.loads(request.content)["questions"]["q"] == {
        "type": "noul",
        "instructions": "Has the customer contacted support about this before?",
        "criteria": {
            "true": "Mentions a prior attempt, ticket, or having asked before",
            "false": "No sign of any previous contact",
        },
    }


async def test_that_structured_criteria_survive_encoding() -> None:
    # Instructions and criteria accept objects and arrays, not only strings.
    requests: list[httpx.Request] = []
    maker = _decision_maker(
        lambda _: _answer_response({"q": {"type": "noul", "noul": 0.5}}),
        requests,
    )

    await maker.decide(
        {"resume": {"name": "John Smith"}},
        {
            "q": Proposition(
                instructions={
                    "candidate": {"name": "John Smith", "city": "Oakland"},
                    "question": "Is `resume` the same person as `candidate`?",
                },
                criteria=PropositionCriteria(
                    true=["same name", "same city"],
                    false="a different person",
                ),
            )
        },
    )

    (request,) = requests
    question = json.loads(request.content)["questions"]["q"]
    assert question["instructions"]["question"] == "Is `resume` the same person as `candidate`?"
    assert question["criteria"]["true"] == ["same name", "same city"]


async def test_that_a_decision_maker_built_with_a_custom_provider_name_reports_under_it() -> None:
    # The same gateway may be deployed twice (say, two accounts at different
    # rates). A hand-built adapter names its own provider, and its usage is
    # keyed under that name so it can be priced and filtered apart.
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    reported: list[str] = []
    usage_reporter = UsageReporter(tracer)
    usage_reporter.add_listener(lambda _trace, model, _usage: reported.append(model))

    maker = OpenRouterDecisionMaker(
        model_name="~typesafe/jev-latest",
        logger=logger,
        tracer=tracer,
        meter=LocalMeter(logger),
        usage_reporter=usage_reporter,
        provider_name="openrouter-eu",
        http=_decisions_client(lambda _: _answer_response({"q": {"type": "noul", "noul": 0.5}})),
    )

    assert maker.provider_name == "openrouter-eu"

    await maker.decide({}, {"q": Proposition(instructions="?")})

    assert reported == ["openrouter-eu:typesafe/jev-1.13-20260917"]
