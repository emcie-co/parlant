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

from parlant.adapters.nlp.openai_service import (
    OpenAIService,
    OpenAISchematicGenerator,
    GPT_4o,
    GPT_4o_24_08_06,
    GPT_4_1,
    GPT_4o_Mini,
    OpenAITextEmbedding3Large,
    OmniModeration,
)
from parlant.core.loggers import Logger
from parlant.core.common import DefaultBaseModel
from parlant.core.nlp.generation import FallbackSchematicGenerator
from parlant.core.engines.alpha.tool_calling.single_tool_batch import SingleToolBatchSchema
from parlant.core.engines.alpha.guideline_matching.generic.journey_node_selection_batch import (
    JourneyNodeSelectionSchema,
)


class _TestSchema(DefaultBaseModel):
    """Test schema for type checking."""
    pass


def test_that_missing_openai_api_key_returns_error_message() -> None:
    """Test that missing OPENAI_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = OpenAIService.verify_environment()
        assert error is not None
        assert "OPENAI_API_KEY is not set" in error
        assert "OpenAI NLP service" in error


def test_that_valid_openai_api_key_passes_verification() -> None:
    """Test that valid OPENAI_API_KEY passes verification."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        error = OpenAIService.verify_environment()
        assert error is None


def test_that_openai_service_initializes_without_model_name() -> None:
    """Test OpenAIService initialization with default behavior."""
    mock_logger = Mock(spec=Logger)
    
    service = OpenAIService(logger=mock_logger)
    
    assert service._logger == mock_logger
    assert service._generative_model_name is None
    mock_logger.info.assert_called_once_with("Initialized OpenAIService")


def test_that_openai_service_initializes_with_single_model_name() -> None:
    """Test OpenAIService initialization with single model name."""
    mock_logger = Mock(spec=Logger)
    
    service = OpenAIService(logger=mock_logger, generative_model_name="gpt-4o-mini")
    
    assert service._logger == mock_logger
    assert service._generative_model_name == "gpt-4o-mini"
    mock_logger.info.assert_called_once_with("Initialized OpenAIService")


def test_that_openai_service_initializes_with_multiple_model_names() -> None:
    """Test OpenAIService initialization with multiple model names."""
    mock_logger = Mock(spec=Logger)
    models = ["gpt-4o-mini", "gpt-4o"]
    
    service = OpenAIService(logger=mock_logger, generative_model_name=models)
    
    assert service._logger == mock_logger
    assert service._generative_model_name == models
    mock_logger.info.assert_called_once_with("Initialized OpenAIService")


def test_that_get_generator_class_for_model_returns_correct_classes() -> None:
    """Test _get_generator_class_for_model returns correct generator classes."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger)
    
    # Test known models
    assert service._get_generator_class_for_model("gpt-4o") == GPT_4o
    assert service._get_generator_class_for_model("gpt-4o-2024-11-20") == GPT_4o
    assert service._get_generator_class_for_model("gpt-4o-2024-08-06") == GPT_4o_24_08_06
    assert service._get_generator_class_for_model("gpt-4.1") == GPT_4_1
    assert service._get_generator_class_for_model("gpt-4o-mini") == GPT_4o_Mini


def test_that_get_generator_class_for_model_handles_unknown_models() -> None:
    """Test _get_generator_class_for_model handles unknown models."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger)
    
    # Test unknown model
    generator_class = service._get_generator_class_for_model("gpt-3.5-turbo")
    
    # Should return a callable that creates OpenAISchematicGenerator
    assert callable(generator_class)
    mock_logger.warning.assert_called_once_with(
        "Unrecognized model name 'gpt-3.5-turbo'. Using dynamic OpenAISchematicGenerator."
    )


@pytest.mark.asyncio
async def test_that_get_schematic_generator_uses_default_behavior_without_model_name() -> None:
    """Test get_schematic_generator uses default behavior when no model name specified."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        # Test default schema mapping
        generator = await service.get_schematic_generator(SingleToolBatchSchema)
        assert isinstance(generator, GPT_4o)
        
        generator = await service.get_schematic_generator(JourneyNodeSelectionSchema)
        assert isinstance(generator, GPT_4_1)
        
        # Test default fallback for unknown schema
        generator = await service.get_schematic_generator(_TestSchema)
        assert isinstance(generator, GPT_4o_24_08_06)


@pytest.mark.asyncio
async def test_that_get_schematic_generator_uses_single_model() -> None:
    """Test get_schematic_generator uses single specified model."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger, generative_model_name="gpt-4o-mini")
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        generator = await service.get_schematic_generator(_TestSchema)
        assert isinstance(generator, GPT_4o_Mini)


@pytest.mark.asyncio
async def test_that_get_schematic_generator_uses_fallback_for_multiple_models() -> None:
    """Test get_schematic_generator uses FallbackSchematicGenerator for multiple models."""
    mock_logger = Mock(spec=Logger)
    models = ["gpt-4o-mini", "gpt-4o"]
    service = OpenAIService(logger=mock_logger, generative_model_name=models)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        generator = await service.get_schematic_generator(_TestSchema)
        assert isinstance(generator, FallbackSchematicGenerator)
        
        # Check that fallback contains the correct generators
        assert len(generator._generators) == 2
        assert isinstance(generator._generators[0], GPT_4o_Mini)
        assert isinstance(generator._generators[1], GPT_4o)


@pytest.mark.asyncio
async def test_that_get_schematic_generator_handles_custom_models() -> None:
    """Test get_schematic_generator handles custom model names."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger, generative_model_name="gpt-3.5-turbo")
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        generator = await service.get_schematic_generator(_TestSchema)
        assert isinstance(generator, OpenAISchematicGenerator)
        assert generator.model_name == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_that_get_embedder_returns_correct_embedder() -> None:
    """Test get_embedder returns correct embedder."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        embedder = await service.get_embedder()
        assert isinstance(embedder, OpenAITextEmbedding3Large)


@pytest.mark.asyncio
async def test_that_get_moderation_service_returns_correct_service() -> None:
    """Test get_moderation_service returns correct service."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        moderation_service = await service.get_moderation_service()
        assert isinstance(moderation_service, OmniModeration)


def test_that_openai_schematic_generator_initializes_correctly() -> None:
    """Test OpenAISchematicGenerator initialization using concrete implementation."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient") as mock_client_class:
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer") as mock_tokenizer_class:
                generator = GPT_4o(logger=mock_logger)
                
                assert generator.model_name == "gpt-4o-2024-11-20"
                assert generator._logger == mock_logger
                assert generator.id == "openai/gpt-4o-2024-11-20"
                mock_client_class.assert_called_once_with(api_key="sk-test-key")
                mock_tokenizer_class.assert_called_once_with(model_name="gpt-4o-2024-11-20")


def test_that_gpt_4o_initializes_with_correct_model_name() -> None:
    """Test GPT_4o initializes with correct model name."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                generator = GPT_4o(logger=mock_logger)
                
                assert generator.model_name == "gpt-4o-2024-11-20"
                assert generator.max_tokens == 128 * 1024


def test_that_gpt_4o_24_08_06_initializes_with_correct_model_name() -> None:
    """Test GPT_4o_24_08_06 initializes with correct model name."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                generator = GPT_4o_24_08_06(logger=mock_logger)
                
                assert generator.model_name == "gpt-4o-2024-08-06"
                assert generator.max_tokens == 128 * 1024


def test_that_gpt_4_1_initializes_with_correct_model_name() -> None:
    """Test GPT_4_1 initializes with correct model name."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                generator = GPT_4_1(logger=mock_logger)
                
                assert generator.model_name == "gpt-4.1"
                assert generator.max_tokens == 128 * 1024


def test_that_gpt_4o_mini_initializes_with_correct_model_name() -> None:
    """Test GPT_4o_Mini initializes with correct model name."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                generator = GPT_4o_Mini(logger=mock_logger)
                
                assert generator.model_name == "gpt-4o-mini"
                assert generator.max_tokens == 128 * 1024


def test_that_openai_schematic_generator_supports_correct_parameters() -> None:
    """Test OpenAISchematicGenerator supports correct parameters using concrete implementation."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                generator = GPT_4o(logger=mock_logger)
                
                expected_params = ["temperature", "logit_bias", "max_tokens"]
                assert generator.supported_openai_params == expected_params
                
                expected_hints = expected_params + ["strict"]
                assert generator.supported_hints == expected_hints


def test_that_openai_text_embedding_3_large_initializes_correctly() -> None:
    """Test OpenAITextEmbedding3Large initialization."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                embedder = OpenAITextEmbedding3Large(logger=mock_logger)
                
                assert embedder.model_name == "text-embedding-3-large"
                assert embedder.max_tokens == 8192
                assert embedder.dimensions == 3072
                assert embedder.id == "openai/text-embedding-3-large"


def test_that_omni_moderation_initializes_correctly() -> None:
    """Test OmniModeration initialization."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            moderation = OmniModeration(logger=mock_logger)
            
            assert moderation.model_name == "omni-moderation-latest"


def test_that_unsupported_params_by_model_filters_correctly() -> None:
    """Test that unsupported parameters are filtered correctly."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                # Create a generator with a model that has unsupported params
                class TestGPT5Generator(OpenAISchematicGenerator[_TestSchema]):
                    def __init__(self, logger: Logger):
                        super().__init__(model_name="gpt-5", logger=logger)
                    
                    @property
                    def max_tokens(self) -> int:
                        return 4096
                
                generator = TestGPT5Generator(logger=mock_logger)
                
                # Test filtering hints for unsupported model parameters
                hints = {"temperature": 0.7, "logit_bias": {}, "max_tokens": 100}
                filtered = generator._list_arguments(hints)
                
                # temperature should be filtered out for gpt-5
                assert "temperature" not in filtered
                assert "logit_bias" in filtered
                assert "max_tokens" in filtered


def test_that_fallback_generator_id_includes_all_generators() -> None:
    """Test that FallbackSchematicGenerator ID includes all generator IDs."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                gen1 = GPT_4o_Mini(logger=mock_logger)
                gen2 = GPT_4o(logger=mock_logger)
                
                fallback = FallbackSchematicGenerator(gen1, gen2, logger=mock_logger)
                
                assert "openai/gpt-4o-mini" in fallback.id
                assert "openai/gpt-4o-2024-11-20" in fallback.id
                assert "fallback(" in fallback.id


def test_that_fallback_generator_uses_first_generator_tokenizer() -> None:
    """Test that FallbackSchematicGenerator uses first generator's tokenizer."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                gen1 = GPT_4o_Mini(logger=mock_logger)
                gen2 = GPT_4o(logger=mock_logger)
                
                fallback = FallbackSchematicGenerator(gen1, gen2, logger=mock_logger)
                
                assert fallback.tokenizer == gen1.tokenizer


def test_that_fallback_generator_uses_minimum_max_tokens() -> None:
    """Test that FallbackSchematicGenerator uses minimum max_tokens."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            with patch("parlant.adapters.nlp.openai_service.OpenAIEstimatingTokenizer"):
                gen1 = GPT_4o_Mini(logger=mock_logger)
                gen2 = GPT_4o(logger=mock_logger)
                
                fallback = FallbackSchematicGenerator(gen1, gen2, logger=mock_logger)
                
                # Both have same max_tokens, so should be that value
                expected_max_tokens = min(gen1.max_tokens, gen2.max_tokens)
                assert fallback.max_tokens == expected_max_tokens
