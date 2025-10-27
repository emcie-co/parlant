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
import pytest
from unittest.mock import AsyncMock, patch, Mock
from typing import cast

from parlant.adapters.nlp.openrouter_service import (
    OpenRouterService,
    OpenRouterSchematicGenerator,
    OpenRouterGPT4O,
    OpenRouterGPT4OMini,
    OpenRouterClaude35Sonnet,
    OpenRouterLlama33_70B,
    OpenRouterTextEmbedding,
    NoModeration,
)
from parlant.core.loggers import Logger
from parlant.core.common import DefaultBaseModel


class _TestSchema(DefaultBaseModel):
    """Test schema for type checking."""
    pass


def test_that_missing_openrouter_api_key_returns_error_message() -> None:
    """Test that missing OPENROUTER_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = OpenRouterService.verify_environment()
        assert error is not None
        assert "OPENROUTER_API_KEY is not set" in error
        assert "OpenRouter NLP service" in error


def test_that_valid_openrouter_api_key_passes_verification() -> None:
    """Test that valid OPENROUTER_API_KEY passes verification."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        error = OpenRouterService.verify_environment()
        assert error is None


def test_that_openrouter_service_initializes() -> None:
    """Test OpenRouterService initialization."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        service = OpenRouterService(logger=mock_logger)
        
        assert service._logger == mock_logger
        assert service.model_name == "openai/gpt-4o"
        mock_logger.info.assert_any_call("Initialized OpenRouterService")
        mock_logger.info.assert_any_call("OpenRouter model name: openai/gpt-4o")


def test_that_openrouter_service_initializes_with_custom_model() -> None:
    """Test OpenRouterService initialization with custom model."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key", "OPENROUTER_MODEL": "anthropic/claude-3.5-sonnet"}, clear=True):
        service = OpenRouterService(logger=mock_logger)
        
        assert service._logger == mock_logger
        assert service.model_name == "anthropic/claude-3.5-sonnet"
        mock_logger.info.assert_any_call("OpenRouter model name: anthropic/claude-3.5-sonnet")


def test_that_get_generator_class_for_model_returns_correct_classes() -> None:
    """Test _get_specialized_generator_class returns correct generator classes."""
    mock_logger = Mock(spec=Logger)
    service = OpenRouterService(logger=mock_logger)
    
    # Test known models
    assert service._get_specialized_generator_class("openai/gpt-4o", _TestSchema) is not None
    assert service._get_specialized_generator_class("openai/gpt-4o-mini", _TestSchema) is not None
    assert service._get_specialized_generator_class("anthropic/claude-3.5-sonnet", _TestSchema) is not None
    assert service._get_specialized_generator_class("meta-llama/llama-3.3-70b-instruct", _TestSchema) is not None


def test_that_get_generator_class_for_model_handles_unknown_models() -> None:
    """Test _get_specialized_generator_class handles unknown models."""
    mock_logger = Mock(spec=Logger)
    service = OpenRouterService(logger=mock_logger)
    
    # Test unknown model
    generator_class = service._get_specialized_generator_class("unknown/model", _TestSchema)
    
    # Should return a callable custom generator
    assert generator_class is not None
    assert callable(generator_class)
    
    # Verify that the model name is logged
    mock_logger.warning.assert_called_once()
    assert "Unrecognized OpenRouter model name" in mock_logger.warning.call_args[0][0]


@pytest.mark.asyncio
async def test_that_get_schematic_generator_returns_correct_generator() -> None:
    """Test get_schematic_generator returns correct generator."""
    mock_logger = Mock(spec=Logger)
    service = OpenRouterService(logger=mock_logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        generator = await service.get_schematic_generator(_TestSchema)
        
        # Should return an OpenRouter generator
        assert isinstance(generator, OpenRouterSchematicGenerator)
        assert generator.model_name == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_that_get_embedder_returns_correct_embedder() -> None:
    """Test get_embedder returns correct embedder."""
    mock_logger = Mock(spec=Logger)
    service = OpenRouterService(logger=mock_logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        embedder = await service.get_embedder()
        
        assert isinstance(embedder, OpenRouterTextEmbedding)
        mock_logger.info.assert_any_call("Using OpenRouter embedding model: text-embedding-ada-002")


@pytest.mark.asyncio
async def test_that_get_moderation_service_returns_no_moderation() -> None:
    """Test get_moderation_service returns NoModeration."""
    mock_logger = Mock(spec=Logger)
    service = OpenRouterService(logger=mock_logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        moderation_service = await service.get_moderation_service()
        assert isinstance(moderation_service, NoModeration)


def test_that_openrouter_gpt_4o_initializes_correctly() -> None:
    """Test OpenRouterGPT4O initialization."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
            with patch("parlant.adapters.nlp.openrouter_service.OpenRouterEstimatingTokenizer") as mock_tokenizer_class:
                generator = OpenRouterGPT4O(logger=mock_logger)
                
                assert generator.model_name == "openai/gpt-4o"
                assert generator._logger == mock_logger
                assert generator.id == "openrouter/openai/gpt-4o"
                assert generator.max_tokens == 128 * 1024
                
                # Verify client was created with correct base URL
                call_args = mock_client_class.call_args
                assert "base_url" in call_args.kwargs
                assert call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
                assert call_args.kwargs["api_key"] == "sk-or-test-key"
                
                mock_tokenizer_class.assert_called_once_with(model_name="openai/gpt-4o")


def test_that_openrouter_gpt_4o_mini_initializes_with_correct_model_name() -> None:
    """Test OpenRouterGPT4OMini initializes with correct model name."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openrouter_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openrouter_service.OpenRouterEstimatingTokenizer"):
                generator = OpenRouterGPT4OMini(logger=mock_logger)
                
                assert generator.model_name == "openai/gpt-4o-mini"
                assert generator.max_tokens == 128 * 1024


def test_that_openrouter_claude_35_sonnet_initializes_with_correct_model_name() -> None:
    """Test OpenRouterClaude35Sonnet initializes with correct model name."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openrouter_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openrouter_service.OpenRouterEstimatingTokenizer"):
                generator = OpenRouterClaude35Sonnet(logger=mock_logger)
                
                assert generator.model_name == "anthropic/claude-3.5-sonnet"
                assert generator.max_tokens == 8192


def test_that_openrouter_llama_33_70b_initializes_with_correct_model_name() -> None:
    """Test OpenRouterLlama33_70B initializes with correct model name."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openrouter_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openrouter_service.OpenRouterEstimatingTokenizer"):
                generator = OpenRouterLlama33_70B(logger=mock_logger)
                
                assert generator.model_name == "meta-llama/llama-3.3-70b-instruct"
                assert generator.max_tokens == 8192


def test_that_openrouter_schematic_generator_supports_correct_parameters() -> None:
    """Test OpenRouterSchematicGenerator supports correct parameters."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openrouter_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openrouter_service.OpenRouterEstimatingTokenizer"):
                generator = OpenRouterGPT4O(logger=mock_logger)
                
                expected_params = ["temperature", "max_tokens"]
                assert generator.supported_openrouter_params == expected_params


def test_that_openrouter_text_embedding_initializes_correctly() -> None:
    """Test OpenRouterTextEmbedding initialization."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openrouter_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openrouter_service.OpenRouterEstimatingTokenizer"):
                embedder = OpenRouterTextEmbedding(model_name="text-embedding-ada-002", logger=mock_logger)
                
                assert embedder.model_name == "text-embedding-ada-002"
                assert embedder.max_tokens == 8192
                assert embedder.id == "openrouter/text-embedding-ada-002"


def test_that_extra_headers_are_set_when_environment_variables_exist() -> None:
    """Test that extra headers are set when environment variables exist."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(
        os.environ, 
        {
            "OPENROUTER_API_KEY": "sk-or-test-key",
            "OPENROUTER_HTTP_REFERER": "https://example.com",
            "OPENROUTER_SITE_NAME": "My App",
        }, 
        clear=True
    ):
        with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
            with patch("parlant.adapters.nlp.openrouter_service.OpenRouterEstimatingTokenizer"):
                generator = OpenRouterGPT4O(_TestSchema, logger=mock_logger)
                
                # Verify client was created with extra headers
                call_args = mock_client_class.call_args
                assert call_args.kwargs["default_headers"] is not None
                assert call_args.kwargs["default_headers"]["HTTP-Referer"] == "https://example.com"
                assert call_args.kwargs["default_headers"]["X-Title"] == "My App"
