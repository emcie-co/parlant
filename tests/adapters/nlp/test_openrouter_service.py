# Copyright 2025 Emcie Co Ltd.
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

import os
from lagom import Container
import pytest
from unittest.mock import AsyncMock, patch, Mock
import asyncio
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from parlant.adapters.nlp.openrouter_service import (  # type: ignore[reportMissingImports]
    OpenRouterService,
    OpenRouterSchematicGenerator,
    OpenRouterGPT4O,
    OpenRouterGPT4OMini,
    OpenRouterClaude35Sonnet,
    OpenRouterLlama33_70B,
    OpenRouterEstimatingTokenizer,
)
from parlant.core.loggers import Logger
from parlant.core.common import DefaultBaseModel
from parlant.core.meter import Meter


class TestSchemaData(DefaultBaseModel):
    """Test schema for type checking."""

    test_field: str = "test_value"


def test_that_missing_openrouter_api_key_returns_error_message() -> None:
    """Test that missing OPENROUTER_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = OpenRouterService.verify_environment()
        assert error is not None
        assert "OPENROUTER_API_KEY is not set" in error


def test_that_present_api_key_returns_none() -> None:
    """Test that present API key returns None (success)."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        error = OpenRouterService.verify_environment()
        assert error is None


def test_that_openrouter_service_initializes_with_default_model() -> None:
    """Test OpenRouterService initialization with default model."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        mock_logger = Mock()
        mock_meter = Mock()
        service = OpenRouterService(logger=mock_logger, meter=mock_meter)
        assert service.model_name == "openai/gpt-4o"


def test_that_openrouter_service_initializes_with_custom_model() -> None:
    """Test OpenRouterService initialization with custom model."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        mock_logger = Mock()
        mock_meter = Mock()
        service = OpenRouterService(
            logger=mock_logger,
            meter=mock_meter,
            model_name="anthropic/claude-3.5-sonnet",
        )
        assert service.model_name == "anthropic/claude-3.5-sonnet"


def test_that_openrouter_service_uses_environment_model() -> None:
    """Test OpenRouterService uses OPENROUTER_MODEL from environment."""
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "meta-llama/llama-3.3-70b-instruct"},
        clear=True,
    ):
        mock_logger = Mock()
        mock_meter = Mock()
        service = OpenRouterService(logger=mock_logger, meter=mock_meter)
        assert service.model_name == "meta-llama/llama-3.3-70b-instruct"


def test_that_openrouter_service_respects_custom_max_tokens() -> None:
    """Test OpenRouterService respects custom max_tokens parameter."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        mock_logger = Mock()
        mock_meter = Mock()
        service = OpenRouterService(
            logger=mock_logger, meter=mock_meter, max_tokens=4096
        )
        assert service._custom_max_tokens == 4096


def test_that_openrouter_estimating_tokenizer_works(container: Container) -> None:
    """Test OpenRouterEstimatingTokenizer token estimation."""
    tokenizer = OpenRouterEstimatingTokenizer(model_name="openai/gpt-4o")
    tokens = asyncio.run(tokenizer.estimate_token_count("Hello world"))
    assert tokens > 0


def test_that_openrouter_gpt4o_generator_initializes_correctly(container: Container) -> None:
    """Test OpenRouterGPT4O initialization."""
    generator = OpenRouterGPT4O[TestSchemaData](logger=container[Logger], meter=container[Meter])
    assert generator.model_name == "openai/gpt-4o"
    assert generator.id == "openrouter/openai/gpt-4o"
    assert generator.max_tokens == 128 * 1024


def test_that_openrouter_gpt4o_mini_generator_initializes_correctly(container: Container) -> None:
    """Test OpenRouterGPT4OMini initialization."""
    generator = OpenRouterGPT4OMini[TestSchemaData](logger=container[Logger], meter=container[Meter])
    assert generator.model_name == "openai/gpt-4o-mini"
    assert generator.max_tokens == 128 * 1024


def test_that_openrouter_claude_generator_initializes_correctly(container: Container) -> None:
    """Test OpenRouterClaude35Sonnet initialization."""
    generator = OpenRouterClaude35Sonnet[TestSchemaData](logger=container[Logger], meter=container[Meter])
    assert generator.model_name == "anthropic/claude-3.5-sonnet"
    assert generator.max_tokens == 8192


def test_that_openrouter_llama_generator_initializes_correctly(container: Container) -> None:
    """Test OpenRouterLlama33_70B initialization."""
    generator = OpenRouterLlama33_70B[TestSchemaData](logger=container[Logger], meter=container[Meter])
    assert generator.model_name == "meta-llama/llama-3.3-70b-instruct"
    assert generator.max_tokens == 8192


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
def test_that_openrouter_generator_sets_custom_headers(mock_client_class) -> None:
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
        mock_logger = Mock()
        mock_meter = Mock()
        generator = OpenRouterSchematicGenerator[TestSchemaData](
            model_name="openai/gpt-4o",
            logger=mock_logger,
            meter=mock_meter,
        )

        # Verify client was called with headers
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert "default_headers" in call_args[1]
        assert call_args[1]["default_headers"]["HTTP-Referer"] == "https://example.com"
        assert call_args[1]["default_headers"]["X-Title"] == "My App"


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
def test_that_openrouter_generator_without_custom_headers(mock_client_class) -> None:
    """Test OpenRouter generator without custom headers."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        mock_logger = Mock()
        mock_meter = Mock()
        generator = OpenRouterSchematicGenerator[TestSchemaData](
            model_name="openai/gpt-4o",
            logger=mock_logger,
            meter=mock_meter,
        )

        # Verify client was called without custom headers
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert call_args[1]["default_headers"] is None


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_generator_handles_json_mode_error(mock_client_class) -> None:
    """Test that OpenRouter generator handles JSON mode errors gracefully."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client

    # Mock BadRequestError for JSON mode
    from openai import BadRequestError

    mock_client.chat.completions.create.side_effect = BadRequestError(
        "Model does not support JSON mode", body={"error": {"message": "JSON mode error"}}, response=Mock()
    )

    mock_logger = Mock()
    mock_meter = Mock()

    generator = OpenRouterSchematicGenerator[TestSchemaData](
        model_name="test-model",
        logger=mock_logger,
        meter=mock_meter,
    )

    # Should fail since we're mocking the error
    with pytest.raises(BadRequestError):
        await generator.do_generate("Test prompt")


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
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[TestSchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            meter=container[Meter],
        )

        result = await generator.do_generate('Generate {"test_field": "test_value"}')

        assert result.content.test_field == "test_value"
        assert result.info.input_tokens == 10
        assert result.info.output_tokens == 20


def test_that_openrouter_service_returns_correct_generator(container: Container) -> None:
    """Test OpenRouterService.get_schematic_generator with default model."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(logger=container[Logger], meter=container[Meter])
        generator = asyncio.run(service.get_schematic_generator(TestSchemaData))
        assert isinstance(generator, OpenRouterSchematicGenerator)
        assert generator.model_name == "openai/gpt-4o"


def test_that_openrouter_service_returns_correct_generator_for_claude(
    container: Container,
) -> None:
    """Test OpenRouterService.get_schematic_generator with Claude model."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger],
            meter=container[Meter],
            model_name="anthropic/claude-3.5-sonnet",
        )
        generator = asyncio.run(service.get_schematic_generator(TestSchemaData))
        assert isinstance(generator, OpenRouterClaude35Sonnet)
        assert generator.model_name == "anthropic/claude-3.5-sonnet"


def test_that_openrouter_service_creates_dynamic_generator_for_unknown_model(
    container: Container,
) -> None:
    """Test OpenRouterService creates dynamic generator for unknown model."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger],
            meter=container[Meter],
            model_name="custom/model-name",
        )
        generator = asyncio.run(service.get_schematic_generator(TestSchemaData))
        assert isinstance(generator, OpenRouterSchematicGenerator)
        assert generator.model_name == "custom/model-name"


def test_that_openrouter_service_uses_custom_max_tokens(container: Container) -> None:
    """Test OpenRouterService uses custom max_tokens for unknown model."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger],
            meter=container[Meter],
            model_name="custom/model",
            max_tokens=2048,
        )
        generator = asyncio.run(service.get_schematic_generator(TestSchemaData))
        assert generator.max_tokens == 2048


def test_that_openrouter_service_uses_environment_max_tokens(container: Container) -> None:
    """Test OpenRouterService uses environment max_tokens."""
    with patch.dict(
        os.environ, {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MAX_TOKENS": "4096"}, clear=True
    ):
        service = OpenRouterService(logger=container[Logger], meter=container[Meter])
        generator = asyncio.run(service.get_schematic_generator(TestSchemaData))
        assert generator.max_tokens == 4096


def test_that_openrouter_service_sets_default_max_tokens_for_gpt4(container: Container) -> None:
    """Test OpenRouterService sets default max_tokens for GPT-4 models."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger],
            meter=container[Meter],
            model_name="openai/gpt-4-turbo",
        )
        generator = asyncio.run(service.get_schematic_generator(TestSchemaData))
        assert generator.max_tokens == 128 * 1024


def test_that_openrouter_service_sets_default_max_tokens_for_claude(container: Container) -> None:
    """Test OpenRouterService sets default max_tokens for Claude models."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger],
            meter=container[Meter],
            model_name="anthropic/claude-2",
        )
        generator = asyncio.run(service.get_schematic_generator(TestSchemaData))
        assert generator.max_tokens == 8192


def test_that_openrouter_service_sets_default_max_tokens_for_llama(container: Container) -> None:
    """Test OpenRouterService sets default max_tokens for Llama models."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger],
            meter=container[Meter],
            model_name="meta-llama/llama-2-70b",
        )
        generator = asyncio.run(service.get_schematic_generator(TestSchemaData))
        assert generator.max_tokens == 8192


def test_that_openrouter_service_returns_jina_embedder(container: Container) -> None:
    """Test OpenRouterService returns JinaAI embedder."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(logger=container[Logger], meter=container[Meter])
        embedder = asyncio.run(service.get_embedder())
        # JinaAIEmbedder is used as fallback since OpenRouter doesn't have embeddings
        assert embedder is not None


def test_that_openrouter_service_returns_no_moderation(container: Container) -> None:
    """Test OpenRouterService returns NoModeration."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(logger=container[Logger], meter=container[Meter])
        moderation = asyncio.run(service.get_moderation_service())
        from parlant.core.nlp.moderation import NoModeration
        assert isinstance(moderation, NoModeration)


def test_that_openrouter_generator_supports_correct_parameters(container: Container) -> None:
    """Test supported OpenRouter parameters."""
    generator = OpenRouterSchematicGenerator[TestSchemaData](
        model_name="openai/gpt-4o",
        logger=container[Logger],
        meter=container[Meter],
    )

    expected_params = ["temperature", "max_tokens"]
    assert generator.supported_openrouter_params == expected_params

