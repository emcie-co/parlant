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
from unittest.mock import patch, Mock
from typing import Any

from parlant.adapters.nlp.openai_service import (
    CustomOpenAISchematicGenerator,
)
from parlant.core.loggers import Logger
from parlant.core.common import DefaultBaseModel
from parlant.core.nlp.generation import SchematicGenerationResult


class SimpleSchema(DefaultBaseModel):
    """Test schema for type checking."""

    message: str


def test_that_custom_openai_schematic_generator_initializes_correctly(container: Container) -> None:
    """Test CustomOpenAISchematicGenerator initialization with custom base URL."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}, clear=True):
        generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
            model_name="test-model",
            logger=container[Logger],
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            max_tokens=4096,
        )

        assert generator.model_name == "test-model"
        assert generator.id == "custom-openai/test-model"
        assert generator.max_tokens == 4096


def test_that_custom_openai_schematic_generator_uses_default_api_key(container: Container) -> None:
    """Test CustomOpenAISchematicGenerator uses dummy key when not provided."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}, clear=True):
        generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
            model_name="test-model",
            logger=container[Logger],
            base_url="http://localhost:8000/v1",
        )

        assert generator.model_name == "test-model"


def test_that_custom_openai_schematic_generator_uses_default_max_tokens(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator with default max_tokens."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    )

    assert generator.max_tokens == 128 * 1024  # Default value


def test_that_custom_openai_schematic_generator_supports_correct_parameters(
    container: Container,
) -> None:
    """Test supported OpenAI parameters."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    )

    expected_params = ["temperature", "logit_bias", "max_tokens"]
    assert generator.supported_openai_params == expected_params

    expected_hints = expected_params + ["strict"]
    assert generator.supported_hints == expected_hints


@pytest.mark.asyncio
async def test_that_custom_openai_schematic_generator_generates_correctly(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator generation."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    )

    # Mock the AsyncClient
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"message": "Hello, World!"}'
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    with patch.object(generator._client.chat.completions, "create", return_value=mock_response) as mock_create:
        result = await generator.generate(prompt="Test prompt")

        # Verify the result
        assert isinstance(result, SchematicGenerationResult)
        assert result.content.message == "Hello, World!"
        assert result.info.schema_name == "SimpleSchema"
        assert result.info.model == "custom-openai/test-model"
        assert result.info.usage.input_tokens == 10
        assert result.info.usage.output_tokens == 20

        # Verify API was called correctly
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["response_format"] == {"type": "json_object"}
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][0]["content"] == "Test prompt"


@pytest.mark.asyncio
async def test_that_custom_openai_schematic_generator_handles_missing_usage_info(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator handles missing usage info."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    )

    # Mock the AsyncClient with no usage info
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"message": "Hello, World!"}'
    mock_response.usage = None

    with patch.object(generator._client.chat.completions, "create", return_value=mock_response):
        result = await generator.generate(prompt="Test prompt")

        # Verify the result handles missing usage
        assert isinstance(result, SchematicGenerationResult)
        assert result.content.message == "Hello, World!"
        assert result.info.usage.input_tokens == 0
        assert result.info.usage.output_tokens == 0


@pytest.mark.asyncio
async def test_that_custom_openai_schematic_generator_handles_invalid_json(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator handles invalid JSON."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    )

    # Mock the AsyncClient with invalid JSON but recoverable
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = 'Some text before {"message": "Hello, World!"} some text after'
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    with patch.object(generator._client.chat.completions, "create", return_value=mock_response):
        result = await generator.generate(prompt="Test prompt")

        # Should recover the JSON using jsonfinder
        assert isinstance(result, SchematicGenerationResult)
        assert result.content.message == "Hello, World!"


@pytest.mark.asyncio
async def test_that_custom_openai_schematic_generator_passes_hints_correctly(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator passes hints correctly."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    )

    # Mock the AsyncClient
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"message": "Hello, World!"}'
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    hints: dict[str, Any] = {"temperature": 0.7, "max_tokens": 100, "unsupported_param": "value"}

    with patch.object(generator._client.chat.completions, "create", return_value=mock_response) as mock_create:
        await generator.generate(prompt="Test prompt", hints=hints)

        # Verify only supported hints are passed
        call_args = mock_create.call_args
        assert "temperature" in call_args[1]
        assert call_args[1]["temperature"] == 0.7
        assert "max_tokens" in call_args[1]
        assert call_args[1]["max_tokens"] == 100
        assert "unsupported_param" not in call_args[1]


@pytest.mark.asyncio
async def test_that_custom_openai_schematic_generator_uses_system_role(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator uses system role for messages."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    )

    # Mock the AsyncClient
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"message": "Hello, World!"}'
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    with patch.object(generator._client.chat.completions, "create", return_value=mock_response) as mock_create:
        await generator.generate(prompt="Test prompt")

        # Verify system role is used
        call_args = mock_create.call_args
        assert call_args[1]["messages"][0]["role"] == "system"


def test_that_custom_openai_schematic_generator_uses_custom_tokenizer(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator uses custom tokenizer model."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        tokenizer_model_name="gpt-4o",
    )

    assert generator.tokenizer.model_name == "gpt-4o"


def test_that_custom_openai_schematic_generator_uses_default_tokenizer(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator uses default tokenizer when not specified."""
    generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
        model_name="test-model",
        logger=container[Logger],
        base_url="http://localhost:8000/v1",
        api_key="test-key",
    )

    assert generator.tokenizer.model_name == "gpt-4o-2024-11-20"


def test_that_custom_openai_schematic_generator_works_without_base_url(
    container: Container,
) -> None:
    """Test CustomOpenAISchematicGenerator works without custom base URL (uses default OpenAI)."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
        generator: CustomOpenAISchematicGenerator[SimpleSchema] = CustomOpenAISchematicGenerator(
            model_name="test-model",
            logger=container[Logger],
        )

        assert generator.model_name == "test-model"
        assert generator.id == "custom-openai/test-model"
