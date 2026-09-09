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

import asyncio
import os
import re
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from lagom import Container
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from parlant.adapters.nlp.aimlapi_service import (
    AIMLAPI_BASE_URL,
    AIMLAPI_DEFAULT_EMBEDDER_MODEL,
    AIMLAPI_DEFAULT_MODEL,
    NULL_REJECTING_REQUEST_FIELDS,
    AIMLAPIEmbedder,
    AIMLAPIEstimatingTokenizer,
    AIMLAPISchematicGenerator,
    AIMLAPIService,
    AIMLAPIStreamingTextGenerator,
    AIMLAPI_GPT_4_1,
    build_attribution_headers,
    omit_unset_arguments,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer

PARTNER_ID_PATTERN = re.compile(r"^part_[A-Za-z0-9]{1,64}$")


class SchemaData(DefaultBaseModel):
    """Test schema for type checking."""

    test_field: str = "test_value"


@pytest.fixture(autouse=True)
def set_api_keys() -> Generator[None, None, None]:
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-openai-key", "AIMLAPI_API_KEY": "test-aimlapi-key"},
        clear=False,
    ):
        yield


def test_that_missing_aimlapi_api_key_returns_error_message() -> None:
    with patch.dict(os.environ, {}, clear=True):
        error = AIMLAPIService.verify_environment()
        assert error is not None
        assert "AIMLAPI_API_KEY is not set" in error


def test_that_present_api_key_returns_none() -> None:
    with patch.dict(os.environ, {"AIMLAPI_API_KEY": "test-key"}, clear=True):
        assert AIMLAPIService.verify_environment() is None


def test_that_aimlapi_service_initializes_with_default_models() -> None:
    with patch.dict(os.environ, {"AIMLAPI_API_KEY": "test-key"}, clear=True):
        service = AIMLAPIService(logger=Mock(), tracer=Mock(), meter=Mock(), health_reporter=Mock())
        assert service.model_name == AIMLAPI_DEFAULT_MODEL
        assert service.embedder_model_name == AIMLAPI_DEFAULT_EMBEDDER_MODEL


def test_that_aimlapi_service_uses_environment_model() -> None:
    with patch.dict(
        os.environ,
        {"AIMLAPI_API_KEY": "test-key", "AIMLAPI_MODEL": "anthropic/claude-sonnet-4.5"},
        clear=True,
    ):
        service = AIMLAPIService(logger=Mock(), tracer=Mock(), meter=Mock(), health_reporter=Mock())
        assert service.model_name == "anthropic/claude-sonnet-4.5"


def test_that_aimlapi_estimating_tokenizer_counts_tokens() -> None:
    tokenizer = AIMLAPIEstimatingTokenizer(model_name=AIMLAPI_DEFAULT_MODEL)
    assert asyncio.run(tokenizer.estimate_token_count("Hello world")) > 0


def test_that_gpt_4_1_generator_reports_its_full_context_window(container: Container) -> None:
    generator = AIMLAPI_GPT_4_1[SchemaData](
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )
    assert generator.model_name == "openai/gpt-4.1"
    assert generator.id == "aimlapi/openai/gpt-4.1"
    assert generator.max_tokens == 1_047_576


def test_that_the_partner_id_matches_the_attribution_contract() -> None:
    # A malformed partner id is silently ignored by the gateway rather than rejected,
    # so nothing at runtime would ever surface a typo here.
    headers = build_attribution_headers()
    assert PARTNER_ID_PATTERN.match(headers["X-AIMLAPI-Partner-ID"])
    assert headers["X-AIMLAPI-Source"] == "agent/parlant"


def test_that_attribution_headers_identify_parlant_and_not_the_provider() -> None:
    headers = build_attribution_headers()
    assert headers["HTTP-Referer"] == "https://github.com/emcie-co/parlant"
    assert headers["X-Title"] == "Parlant"


def test_that_attribution_headers_are_not_sent_to_another_host() -> None:
    assert build_attribution_headers("https://api.openai.com/v1") == {}
    assert build_attribution_headers("https://aimlapi-proxy.example.com/v1") == {}
    assert build_attribution_headers(AIMLAPI_BASE_URL) != {}


def test_that_attribution_headers_are_a_fresh_dict_per_call() -> None:
    first = build_attribution_headers()
    first["X-Title"] = "mutated"
    assert build_attribution_headers()["X-Title"] == "Parlant"


def test_that_referer_and_title_can_be_overridden_by_the_host_application() -> None:
    with patch.dict(
        os.environ,
        {
            "AIMLAPI_API_KEY": "test-key",
            "AIMLAPI_HTTP_REFERER": "https://myapp.example",
            "AIMLAPI_SITE_NAME": "My App",
        },
        clear=True,
    ):
        headers = build_attribution_headers()
        assert headers["HTTP-Referer"] == "https://myapp.example"
        assert headers["X-Title"] == "My App"
        # Attribution itself is not overridable by the host.
        assert headers["X-AIMLAPI-Partner-ID"] == "part_UOT3mCwOdpOUQKX2gIvYrCmv"


@patch("parlant.adapters.nlp.aimlapi_service.AsyncClient")
def test_that_the_client_is_created_against_aimlapi_with_attribution_headers(
    mock_client_class: Mock,
) -> None:
    _ = AIMLAPISchematicGenerator[SchemaData](
        model_name=AIMLAPI_DEFAULT_MODEL,
        logger=Mock(),
        tracer=Mock(),
        meter=Mock(),
        health_reporter=Mock(),
    )

    call_kwargs = mock_client_class.call_args[1]
    assert call_kwargs["base_url"] == AIMLAPI_BASE_URL
    assert call_kwargs["default_headers"]["X-AIMLAPI-Partner-ID"] == "part_UOT3mCwOdpOUQKX2gIvYrCmv"
    assert call_kwargs["default_headers"]["X-AIMLAPI-Source"] == "agent/parlant"


def test_that_unset_arguments_are_omitted_rather_than_passed_as_null() -> None:
    # aimlapi.com answers 400 for these fields when they arrive as an explicit null,
    # while OpenAI accepts null — so this must be checked here and not just in review.
    unset = {field: None for field in NULL_REJECTING_REQUEST_FIELDS}
    assert omit_unset_arguments({**unset, "model": "openai/gpt-4.1"}) == {"model": "openai/gpt-4.1"}


def _completion(content: str) -> Mock:
    response = Mock(spec=ChatCompletion)
    response.choices = [
        Choice(
            message=ChatCompletionMessage(role="assistant", content=content),
            finish_reason="stop",
            index=0,
        )
    ]
    response.usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return response


async def test_that_a_none_valued_hint_is_never_sent_to_the_api(container: Container) -> None:
    with patch("parlant.adapters.nlp.aimlapi_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_completion('{"test_field": "test_value"}')
        )
        mock_client_class.return_value = mock_client

        generator = AIMLAPISchematicGenerator[SchemaData](
            model_name=AIMLAPI_DEFAULT_MODEL,
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )

        await generator.do_generate("Generate something", hints={"temperature": None})

        request_kwargs: dict[str, Any] = mock_client.chat.completions.create.call_args[1]
        assert "temperature" not in request_kwargs
        assert [k for k, v in request_kwargs.items() if v is None] == []


async def test_that_a_set_hint_is_forwarded_to_the_api(container: Container) -> None:
    with patch("parlant.adapters.nlp.aimlapi_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_completion('{"test_field": "test_value"}')
        )
        mock_client_class.return_value = mock_client

        generator = AIMLAPISchematicGenerator[SchemaData](
            model_name=AIMLAPI_DEFAULT_MODEL,
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )

        await generator.do_generate("Generate something", hints={"temperature": 0.0})

        request_kwargs: dict[str, Any] = mock_client.chat.completions.create.call_args[1]
        assert request_kwargs["temperature"] == 0.0
        assert request_kwargs["response_format"] == {"type": "json_object"}


async def test_that_no_none_valued_argument_is_sent_when_streaming(container: Container) -> None:
    with patch("parlant.adapters.nlp.aimlapi_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=AsyncMock())
        mock_client_class.return_value = mock_client

        generator = AIMLAPIStreamingTextGenerator(
            model_name=AIMLAPI_DEFAULT_MODEL,
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )

        await generator.do_generate("Say hello", hints={"temperature": None, "max_tokens": 64})

        request_kwargs: dict[str, Any] = mock_client.chat.completions.create.call_args[1]
        assert "temperature" not in request_kwargs
        assert request_kwargs["max_tokens"] == 64
        assert [k for k, v in request_kwargs.items() if v is None] == []


async def test_that_the_generator_parses_a_successful_response(container: Container) -> None:
    with patch("parlant.adapters.nlp.aimlapi_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_completion('{"test_field": "test_value"}')
        )
        mock_client_class.return_value = mock_client

        generator = AIMLAPISchematicGenerator[SchemaData](
            model_name=AIMLAPI_DEFAULT_MODEL,
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )

        result = await generator.do_generate('Generate {"test_field": "test_value"}')

        assert result.content.test_field == "test_value"
        assert result.info.usage.input_tokens == 10
        assert result.info.usage.output_tokens == 20


def test_that_the_default_embedder_reports_its_known_dimensions(container: Container) -> None:
    embedder = AIMLAPIEmbedder(
        model_name=AIMLAPI_DEFAULT_EMBEDDER_MODEL,
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )
    assert embedder.dimensions == 3072
    assert embedder.id == "aimlapi/openai/text-embedding-3-large"


def test_that_the_service_returns_the_specialized_generator_for_the_default_model(
    container: Container,
) -> None:
    with patch.dict(os.environ, {"AIMLAPI_API_KEY": "test-key"}, clear=True):
        service = AIMLAPIService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert isinstance(generator, AIMLAPISchematicGenerator)
        assert generator.model_name == AIMLAPI_DEFAULT_MODEL
        assert generator.max_tokens == 1_047_576


def test_that_an_unknown_model_falls_back_to_a_dynamic_generator(container: Container) -> None:
    with patch.dict(
        os.environ,
        {
            "AIMLAPI_API_KEY": "test-key",
            "AIMLAPI_MODEL": "some-vendor/some-model",
            "AIMLAPI_MAX_TOKENS": "4096",
        },
        clear=True,
    ):
        service = AIMLAPIService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert generator.model_name == "some-vendor/some-model"
        assert generator.max_tokens == 4096
