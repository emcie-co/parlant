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

import os
from lagom import Container
from unittest.mock import patch, Mock
import asyncio

from parlant.adapters.nlp.zhipu_service import (
    ZhipuService,
    ZhipuSchematicGenerator,
    ZhipuEmbedder,
    ZhipuModerationService,
    GLM_4_Plus,
    GLM_4_Flash,
    GLM_4_Air,
    Embedding_3,
)
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.common import DefaultBaseModel


class TestSchema(DefaultBaseModel):
    """Test schema for type checking."""

    pass


def test_that_missing_api_key_returns_error_message() -> None:
    """Test that missing ZHIPUAI_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = ZhipuService.verify_environment()
        assert error is not None
        assert "ZHIPUAI_API_KEY is not set" in error
        assert "You're using the Zhipu AI NLP service" in error


def test_that_error_messages_include_helpful_instructions() -> None:
    """Test that error messages include helpful authentication instructions."""
    with patch.dict(os.environ, {}, clear=True):
        error = ZhipuService.verify_environment()
        assert error is not None

        # Verify error message contains Zhipu AI official website link
        assert "https://open.bigmodel.cn/" in error

        # Verify error message contains API key acquisition steps
        assert "To obtain an API key:" in error
        assert "Register or log in to your account" in error
        assert "Create an API key in the console" in error

        # Verify error message contains environment variable setting example
        assert "export ZHIPUAI_API_KEY=" in error


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_zhipu_schematic_generator_initializes_correctly(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test ZhipuSchematicGenerator initialization using GLM_4_Plus class."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        generator: GLM_4_Plus[TestSchema] = GLM_4_Plus(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        assert generator.model_name == "glm-4-plus"
        assert generator.id == "zhipu/glm-4-plus"
        mock_zhipuai_class.assert_called_once_with(api_key="test-api-key")


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_zhipu_schematic_generator_supports_correct_parameters(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test supported Zhipu parameters."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        generator: GLM_4_Plus[TestSchema] = GLM_4_Plus(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        expected_params = ["temperature", "max_tokens", "top_p"]
        assert generator.supported_zhipu_params == expected_params
        assert generator.supported_hints == expected_params


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_glm_4_plus_initializes_correctly(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test GLM_4_Plus initialization and max_tokens."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        generator: GLM_4_Plus[TestSchema] = GLM_4_Plus(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        assert generator.model_name == "glm-4-plus"
        assert generator.max_tokens == 128 * 1024
        mock_zhipuai_class.assert_called_once()


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_glm_4_flash_initializes_correctly(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test GLM_4_Flash initialization and max_tokens."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        generator: GLM_4_Flash[TestSchema] = GLM_4_Flash(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        assert generator.model_name == "glm-4-flash"
        assert generator.max_tokens == 128 * 1024
        mock_zhipuai_class.assert_called_once()


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_glm_4_air_initializes_correctly(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test GLM_4_Air initialization and max_tokens."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        generator: GLM_4_Air[TestSchema] = GLM_4_Air(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        assert generator.model_name == "glm-4-air"
        assert generator.max_tokens == 128 * 1024
        mock_zhipuai_class.assert_called_once()


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_zhipu_embedder_initializes_correctly(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test ZhipuEmbedder initialization using Embedding_3 class."""

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        embedder: Embedding_3 = Embedding_3(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        assert embedder.model_name == "embedding-3"
        assert embedder.id == "zhipu/embedding-3"
        assert embedder.max_tokens == 8192
        assert embedder.dimensions == 2048
        mock_zhipuai_class.assert_called_once_with(api_key="test-api-key")


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_zhipu_moderation_service_initializes_correctly(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test ZhipuModerationService initialization."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        moderation_service = ZhipuModerationService(
            model_name="moderation",
            logger=container[Logger],
            meter=container[Meter],
        )

        assert moderation_service.model_name == "moderation"
        mock_zhipuai_class.assert_called_once_with(api_key="test-api-key")


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_zhipu_service_returns_correct_schematic_generator(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test that ZhipuService returns correct schematic generator instance."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        service = ZhipuService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        # Test with TestSchema
        generator = asyncio.run(service.get_schematic_generator(TestSchema))

        # Verify it returns a ZhipuSchematicGenerator instance
        assert isinstance(generator, ZhipuSchematicGenerator)
        # Default should be GLM_4_Flash for unknown schemas
        assert isinstance(generator, GLM_4_Flash)
        assert generator.model_name == "glm-4-flash"


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_zhipu_service_returns_correct_embedder(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test that ZhipuService returns correct embedder instance."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        service = ZhipuService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        # Get embedder
        embedder = asyncio.run(service.get_embedder())

        # Verify it returns an Embedding_3 instance
        assert isinstance(embedder, Embedding_3)
        assert isinstance(embedder, ZhipuEmbedder)
        assert embedder.model_name == "embedding-3"


@patch("parlant.adapters.nlp.zhipu_service.ZhipuAI")
def test_that_zhipu_service_returns_correct_moderation_service(
    mock_zhipuai_class: Mock, container: Container
) -> None:
    """Test that ZhipuService returns correct moderation service instance."""
    from parlant.core.meter import Meter

    mock_client = Mock()
    mock_zhipuai_class.return_value = mock_client

    with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test-api-key"}, clear=True):
        service = ZhipuService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        # Get moderation service
        moderation_service = asyncio.run(service.get_moderation_service())

        # Verify it returns a ZhipuModerationService instance
        assert isinstance(moderation_service, ZhipuModerationService)
        assert moderation_service.model_name == "moderation"
