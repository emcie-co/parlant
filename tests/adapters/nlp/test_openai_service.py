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
import asyncio
from typing import Any

from parlant.adapters.nlp.openai_service import (
    OpenAIService,
    OpenAISchematicGenerator,
    GPT_4o,
    GPT_4o_24_08_06,
    GPT_4_1,
    GPT_4o_Mini,
)
from parlant.core.loggers import Logger
from parlant.core.common import DefaultBaseModel
from parlant.core.nlp.generation import FallbackSchematicGenerator
from parlant.core.engines.alpha.tool_calling.single_tool_batch import SingleToolBatchSchema
from parlant.core.engines.alpha.guideline_matching.generic.journey_node_selection_batch import (
    JourneyNodeSelectionSchema,
)
from parlant.sdk import NLPServices


def test_that_missing_openai_api_key_returns_error_message() -> None:
    """Test that missing OPENAI_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = OpenAIService.verify_environment()
        assert error is not None
        assert "OPENAI_API_KEY is not set" in error
        assert "Please set OPENAI_API_KEY in your environment" in error


def test_that_openai_api_key_presence_passes_verification() -> None:
    """Test that presence of OPENAI_API_KEY passes verification."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}, clear=True):
        error = OpenAIService.verify_environment()
        assert error is None


def test_that_openai_service_initializes_without_model_name() -> None:
    """Test OpenAIService initialization without generative_model_name parameter."""
    mock_logger = Mock(spec=Logger)
    
    service = OpenAIService(logger=mock_logger)
    
    assert service._logger == mock_logger
    assert service._generative_model_name is None


def test_that_openai_service_initializes_with_single_model_name() -> None:
    """Test OpenAIService initialization with single generative_model_name."""
    mock_logger = Mock(spec=Logger)
    
    service = OpenAIService(logger=mock_logger, generative_model_name="gpt-4o-mini")
    
    assert service._logger == mock_logger
    assert service._generative_model_name == "gpt-4o-mini"


def test_that_openai_service_initializes_with_multiple_model_names() -> None:
    """Test OpenAIService initialization with list of generative_model_names."""
    mock_logger = Mock(spec=Logger)
    model_names = ["gpt-4o-mini", "gpt-4o"]
    
    service = OpenAIService(logger=mock_logger, generative_model_name=model_names)
    
    assert service._logger == mock_logger
    assert service._generative_model_name == model_names


def test_that_get_generator_class_for_model_returns_known_models() -> None:
    """Test _get_generator_class_for_model returns correct classes for known models."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger)
    
    assert service._get_generator_class_for_model("gpt-4o") == GPT_4o
    assert service._get_generator_class_for_model("gpt-4o-2024-11-20") == GPT_4o
    assert service._get_generator_class_for_model("gpt-4o-2024-08-06") == GPT_4o_24_08_06
    assert service._get_generator_class_for_model("gpt-4.1") == GPT_4_1
    assert service._get_generator_class_for_model("gpt-4o-mini") == GPT_4o_Mini


def test_that_get_generator_class_for_model_handles_unknown_models() -> None:
    """Test _get_generator_class_for_model creates dynamic generator for unknown models."""
    mock_logger = Mock(spec=Logger)
    service = OpenAIService(logger=mock_logger)
    
    generator_class = service._get_generator_class_for_model("gpt-3.5-turbo")
    
    # Should return a callable that creates a custom generator
    assert callable(generator_class)
    mock_logger.warning.assert_called_once_with(
        "Unrecognized model name 'gpt-3.5-turbo'. Using dynamic OpenAISchematicGenerator."
    )


@patch("parlant.adapters.nlp.openai_service.AsyncClient")
def test_that_default_behavior_still_works(mock_client_class: Mock) -> None:
    """Test that default schema-specific model selection still works."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client
    mock_logger = Mock(spec=Logger)
    
    service = OpenAIService(logger=mock_logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        # Test SingleToolBatchSchema gets GPT_4o
        generator = asyncio.run(service.get_schematic_generator(SingleToolBatchSchema))
        assert isinstance(generator, GPT_4o)
        
        # Test JourneyNodeSelectionSchema gets GPT_4_1
        generator = asyncio.run(service.get_schematic_generator(JourneyNodeSelectionSchema))
        assert isinstance(generator, GPT_4_1)
        
        # Test unknown schema gets GPT_4o_24_08_06
        generator = asyncio.run(service.get_schematic_generator(type('TestSchema', (DefaultBaseModel,), {})))
        assert isinstance(generator, GPT_4o_24_08_06)


@patch("parlant.adapters.nlp.openai_service.AsyncClient")
def test_that_single_model_selection_works(mock_client_class: Mock) -> None:
    """Test that single model selection returns correct generator."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client
    mock_logger = Mock(spec=Logger)
    
    service = OpenAIService(logger=mock_logger, generative_model_name="gpt-4o-mini")
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        generator = asyncio.run(service.get_schematic_generator(type('TestSchema', (DefaultBaseModel,), {})))
        assert isinstance(generator, GPT_4o_Mini)


@patch("parlant.adapters.nlp.openai_service.AsyncClient")
def test_that_multiple_model_selection_returns_fallback_generator(mock_client_class: Mock) -> None:
    """Test that multiple model selection returns FallbackSchematicGenerator."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client
    mock_logger = Mock(spec=Logger)
    
    model_names = ["gpt-4o-mini", "gpt-4o"]
    service = OpenAIService(logger=mock_logger, generative_model_name=model_names)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        generator = asyncio.run(service.get_schematic_generator(type('TestSchema', (DefaultBaseModel,), {})))
        assert isinstance(generator, FallbackSchematicGenerator)


@patch("parlant.adapters.nlp.openai_service.AsyncClient")
def test_that_custom_model_selection_works(mock_client_class: Mock) -> None:
    """Test that custom model selection creates dynamic generator."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client
    mock_logger = Mock(spec=Logger)
    
    service = OpenAIService(logger=mock_logger, generative_model_name="gpt-3.5-turbo")
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        generator = asyncio.run(service.get_schematic_generator(type('TestSchema', (DefaultBaseModel,), {})))
        assert isinstance(generator, OpenAISchematicGenerator)
        assert generator.model_name == "gpt-3.5-turbo"


@patch("parlant.adapters.nlp.openai_service.AsyncClient")
def test_that_mixed_known_and_custom_models_work_in_fallback(mock_client_class: Mock) -> None:
    """Test that fallback works with mix of known and custom models."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client
    mock_logger = Mock(spec=Logger)
    
    model_names = ["gpt-3.5-turbo", "gpt-4o-mini"]  # custom + known
    service = OpenAIService(logger=mock_logger, generative_model_name=model_names)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        generator = asyncio.run(service.get_schematic_generator(type('TestSchema', (DefaultBaseModel,), {})))
        assert isinstance(generator, FallbackSchematicGenerator)


def test_that_sdk_openai_method_accepts_model_name_parameter() -> None:
    """Test that SDK NLPServices.openai method accepts generative_model_name parameter."""
    from lagom import Container
    
    # Test with Container provided
    mock_container = Mock()
    mock_logger = Mock(spec=Logger)
    mock_container.__getitem__ = Mock(return_value=mock_logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        service = NLPServices.openai(container=mock_container, generative_model_name="gpt-4o-mini")
        assert isinstance(service, OpenAIService)
        assert service._generative_model_name == "gpt-4o-mini"


def test_that_sdk_openai_method_returns_factory_without_container() -> None:
    """Test that SDK NLPServices.openai method returns factory when no container provided."""
    factory = NLPServices.openai(generative_model_name="gpt-4o-mini")
    assert callable(factory)


def test_that_sdk_openai_factory_creates_service_with_model_name() -> None:
    """Test that SDK factory creates service with correct model name."""
    from lagom import Container
    
    factory = NLPServices.openai(generative_model_name="gpt-4o-mini")
    
    mock_container = Mock()
    mock_logger = Mock(spec=Logger)
    mock_container.__getitem__ = Mock(return_value=mock_logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        service = factory(mock_container)
        assert isinstance(service, OpenAIService)
        assert service._generative_model_name == "gpt-4o-mini"


def test_that_sdk_openai_method_works_with_multiple_models() -> None:
    """Test that SDK method works with list of model names."""
    from lagom import Container
    
    model_names = ["gpt-4o-mini", "gpt-4o"]
    mock_container = Mock()
    mock_logger = Mock(spec=Logger)
    mock_container.__getitem__ = Mock(return_value=mock_logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        service = NLPServices.openai(container=mock_container, generative_model_name=model_names)
        assert isinstance(service, OpenAIService)
        assert service._generative_model_name == model_names


def test_that_sdk_openai_method_handles_verification_error() -> None:
    """Test that SDK method raises SDKError when environment verification fails."""
    from parlant.sdk import SDKError
    from lagom import Container
    
    mock_container = Mock()
    
    with patch.dict(os.environ, {}, clear=True):  # No OPENAI_API_KEY
        with pytest.raises(SDKError) as exc_info:
            NLPServices.openai(container=mock_container)
        
        assert "OPENAI_API_KEY is not set" in str(exc_info.value)


def test_that_sdk_factory_handles_verification_error_on_call() -> None:
    """Test that SDK factory raises SDKError when called with invalid environment."""
    from parlant.sdk import SDKError
    from lagom import Container
    
    factory = NLPServices.openai(generative_model_name="gpt-4o")
    mock_container = Mock()
    
    with patch.dict(os.environ, {}, clear=True):  # No OPENAI_API_KEY
        with pytest.raises(SDKError) as exc_info:
            factory(mock_container)
        
        assert "OPENAI_API_KEY is not set" in str(exc_info.value)


@patch("parlant.adapters.nlp.openai_service.AsyncClient")
def test_that_openai_schematic_generator_initializes_correctly(mock_client_class: Mock) -> None:
    """Test OpenAISchematicGenerator initialization."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        generator = GPT_4o(logger=mock_logger)
        
        assert generator.model_name == "gpt-4o-2024-11-20"
        assert generator._logger == mock_logger
        assert generator.id == "openai/gpt-4o-2024-11-20"


def test_that_openai_schematic_generator_supports_correct_parameters() -> None:
    """Test OpenAISchematicGenerator supported parameters."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            generator = GPT_4o(logger=mock_logger)
            
            expected_params = ["temperature", "logit_bias", "max_tokens"]
            assert generator.supported_openai_params == expected_params
            
            expected_hints = expected_params + ["strict"]
            assert generator.supported_hints == expected_hints


def test_that_predefined_generators_have_correct_model_names() -> None:
    """Test that predefined generator classes have correct model names."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            gpt_4o = GPT_4o(logger=mock_logger)
            assert gpt_4o.model_name == "gpt-4o-2024-11-20"
            
            gpt_4o_mini = GPT_4o_Mini(logger=mock_logger)
            assert gpt_4o_mini.model_name == "gpt-4o-mini"
            
            gpt_4_1 = GPT_4_1(logger=mock_logger)
            assert gpt_4_1.model_name == "gpt-4.1"
            
            gpt_4o_24_08_06 = GPT_4o_24_08_06(logger=mock_logger)
            assert gpt_4o_24_08_06.model_name == "gpt-4o-2024-08-06"


def test_that_predefined_generators_have_correct_max_tokens() -> None:
    """Test that predefined generator classes have correct max_tokens."""
    mock_logger = Mock(spec=Logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        with patch("parlant.adapters.nlp.openai_service.AsyncClient"):
            gpt_4o = GPT_4o(logger=mock_logger)
            assert gpt_4o.max_tokens == 128 * 1024
            
            gpt_4o_mini = GPT_4o_Mini(logger=mock_logger)
            assert gpt_4o_mini.max_tokens == 128 * 1024
            
            gpt_4_1 = GPT_4_1(logger=mock_logger)
            assert gpt_4_1.max_tokens == 128 * 1024
            
            gpt_4o_24_08_06 = GPT_4o_24_08_06(logger=mock_logger)
            assert gpt_4o_24_08_06.max_tokens == 128 * 1024


def test_backward_compatibility_with_existing_code() -> None:
    """Test that existing code without generative_model_name still works."""
    from lagom import Container
    
    mock_container = Mock()
    mock_logger = Mock(spec=Logger)
    mock_container.__getitem__ = Mock(return_value=mock_logger)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        # This is how it was called before the changes
        service = NLPServices.openai(container=mock_container)
        assert isinstance(service, OpenAIService)
        assert service._generative_model_name is None


def test_that_empty_model_list_is_handled_gracefully() -> None:
    """Test that empty model list is handled gracefully."""
    mock_logger = Mock(spec=Logger)
    
    # Should not raise an error during initialization
    service = OpenAIService(logger=mock_logger, generative_model_name=[])
    assert service._generative_model_name == []
