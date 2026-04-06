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
import pytest
from unittest.mock import AsyncMock, patch, Mock
import asyncio

from parlant.adapters.nlp.azure_service import (
    AzureService,
    create_azure_client,
    AzureSchematicGenerator,
    CustomAzureSchematicGenerator,
    CustomAzureEmbedder,
    AzureTextEmbedding3Large,
    AzureTextEmbedding3Small,
)
from parlant.core.loggers import Logger
from parlant.core.common import DefaultBaseModel
from parlant.core.tracer import Tracer
from parlant.core.meter import Meter


class TestSchema(DefaultBaseModel):
    """Test schema for type checking."""

    pass


def test_that_missing_azure_endpoint_returns_error_message() -> None:
    """Test that missing AZURE_ENDPOINT returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = AzureService.verify_environment()
        assert error is not None
        assert "AZURE_ENDPOINT is not set" in error
        assert "Required environment variables" in error


def test_that_api_key_authentication_is_detected_correctly() -> None:
    """Test that API key authentication is detected correctly."""
    with patch.dict(
        os.environ,
        {"AZURE_ENDPOINT": "https://test.openai.azure.com/", "AZURE_API_KEY": "test-api-key"},
        clear=True,
    ):
        error = AzureService.verify_environment()
        assert error is None


def test_that_azure_ad_authentication_path_is_attempted_when_no_api_key() -> None:
    """Test that Azure AD authentication path is attempted when no API key is present."""
    with patch.dict(os.environ, {"AZURE_ENDPOINT": "https://test.openai.azure.com/"}, clear=True):
        # Since we can't easily mock the complex async behavior,
        # we'll just test that the method doesn't crash and returns an error message
        # when Azure AD authentication is not available
        error = AzureService.verify_environment()
        assert error is not None
        assert "Azure authentication is not properly configured" in error
        assert "API Key Authentication" in error
        assert "Azure AD Authentication" in error


@patch("parlant.adapters.nlp.azure_service.DefaultAzureCredential")
def test_that_failed_azure_ad_authentication_returns_error_message(
    mock_credential_class: Mock,
) -> None:
    """Test that failed Azure AD authentication returns error message."""
    # Mock failed credential creation
    mock_credential_class.side_effect = Exception("Authentication failed")

    with patch.dict(os.environ, {"AZURE_ENDPOINT": "https://test.openai.azure.com/"}, clear=True):
        error = AzureService.verify_environment()
        assert error is not None
        assert "Azure authentication is not properly configured" in error
        assert "API Key Authentication" in error
        assert "Azure AD Authentication" in error


@patch("parlant.adapters.nlp.azure_service.DefaultAzureCredential")
def test_that_failed_token_retrieval_returns_error_message(mock_credential_class: Mock) -> None:
    """Test that failed token retrieval returns error message."""
    mock_credential = AsyncMock()
    mock_credential.get_token.side_effect = Exception("Token retrieval failed")
    mock_credential_class.return_value = mock_credential

    with patch.dict(os.environ, {"AZURE_ENDPOINT": "https://test.openai.azure.com/"}, clear=True):
        error = AzureService.verify_environment()
        assert error is not None
        assert "Azure authentication is not properly configured" in error


def test_that_error_messages_include_helpful_authentication_instructions() -> None:
    """Test that error messages include helpful authentication instructions."""
    with patch.dict(os.environ, {"AZURE_ENDPOINT": "https://test.openai.azure.com/"}, clear=True):
        with patch(
            "parlant.adapters.nlp.azure_service.DefaultAzureCredential"
        ) as mock_credential_class:
            mock_credential_class.side_effect = Exception("Auth failed")

            error = AzureService.verify_environment()
            assert error is not None

            # Check for specific authentication methods
            assert "az login" in error
            assert "AZURE_CLIENT_ID" in error
            assert "AZURE_CLIENT_SECRET" in error
            assert "AZURE_TENANT_ID" in error
            assert "Cognitive Services OpenAI User" in error


@patch("parlant.adapters.nlp.azure_service.AsyncAzureOpenAI")
def test_that_client_creation_with_api_key_works(mock_openai_class: Mock) -> None:
    """Test client creation with API key authentication."""
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    with patch.dict(
        os.environ,
        {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-api-key",
            "AZURE_API_VERSION": "2024-08-01-preview",
        },
        clear=True,
    ):
        client = create_azure_client()

        mock_openai_class.assert_called_once_with(
            api_key="test-api-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-08-01-preview",
        )
        assert client == mock_client


@patch("parlant.adapters.nlp.azure_service.DefaultAzureCredential")
@patch("parlant.adapters.nlp.azure_service.AsyncAzureOpenAI")
def test_that_client_creation_with_azure_ad_works(
    mock_openai_class: Mock, mock_credential_class: Mock
) -> None:
    """Test client creation with Azure AD authentication."""
    mock_client = Mock()
    mock_openai_class.return_value = mock_client
    mock_credential = Mock()
    mock_credential_class.return_value = mock_credential

    with patch.dict(
        os.environ,
        {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_VERSION": "2024-08-01-preview",
        },
        clear=True,
    ):
        create_azure_client()

        # Verify credential was created
        mock_credential_class.assert_called_once()

        # Verify client was created with token provider
        mock_openai_class.assert_called_once()
        call_args = mock_openai_class.call_args
        assert call_args[1]["azure_endpoint"] == "https://test.openai.azure.com/"
        assert call_args[1]["api_version"] == "2024-08-01-preview"
        assert "azure_ad_token_provider" in call_args[1]


@patch("parlant.adapters.nlp.azure_service.DefaultAzureCredential")
def test_that_client_creation_fails_with_azure_ad_authentication_error(
    mock_credential_class: Mock,
) -> None:
    """Test client creation failure with Azure AD authentication."""
    mock_credential_class.side_effect = Exception("Credential creation failed")

    with patch.dict(os.environ, {"AZURE_ENDPOINT": "https://test.openai.azure.com/"}, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            create_azure_client()

        assert "Failed to initialize Azure AD authentication" in str(exc_info.value)
        assert "az login" in str(exc_info.value)


def test_that_azure_schematic_generator_initializes_correctly(container: Container) -> None:
    """Test AzureSchematicGenerator initialization using GPT_4o class."""
    from parlant.adapters.nlp.azure_service import GPT_4o

    mock_client = AsyncMock()

    with patch.dict(
        os.environ,
        {"AZURE_ENDPOINT": "https://test.openai.azure.com/", "AZURE_API_KEY": "test-key"},
        clear=True,
    ):
        with patch("parlant.adapters.nlp.azure_service.create_azure_client") as mock_create_client:
            mock_create_client.return_value = mock_client
            generator: GPT_4o[TestSchema] = GPT_4o(
                logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
            )

            assert generator.model_name == "gpt-4o"
            assert generator.id == "azure/gpt-4o"


def test_that_azure_schematic_generator_supports_correct_parameters(container: Container) -> None:
    """Test supported Azure parameters."""
    # Use GPT_4o which is a concrete implementation
    from parlant.adapters.nlp.azure_service import GPT_4o

    mock_client = AsyncMock()

    with patch.dict(
        os.environ,
        {"AZURE_ENDPOINT": "https://test.openai.azure.com/", "AZURE_API_KEY": "test-key"},
        clear=True,
    ):
        with patch("parlant.adapters.nlp.azure_service.create_azure_client") as mock_create_client:
            mock_create_client.return_value = mock_client
            generator: GPT_4o[TestSchema] = GPT_4o(
                logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
            )

            expected_params = ["temperature", "logit_bias", "max_tokens"]
            assert generator.supported_azure_params == expected_params

            expected_hints = expected_params + ["strict"]
            assert generator.supported_hints == expected_hints


@patch("parlant.adapters.nlp.azure_service.create_azure_client")
def test_that_custom_azure_schematic_generator_initializes_correctly(
    container: Container,
    mock_create_client: Mock,
) -> None:
    """Test CustomAzureSchematicGenerator initialization."""
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    with patch.dict(
        os.environ,
        {"AZURE_GENERATIVE_MODEL_NAME": "gpt-4o", "AZURE_GENERATIVE_MODEL_WINDOW": "4096"},
        clear=True,
    ):
        generator: CustomAzureSchematicGenerator[TestSchema] = CustomAzureSchematicGenerator(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        assert generator.model_name == "gpt-4o"
        assert generator.max_tokens == 4096
        mock_create_client.assert_called_once()


def test_that_custom_azure_schematic_generator_uses_default_max_tokens(
    container: Container,
) -> None:
    """Test CustomAzureSchematicGenerator with default max_tokens."""
    with patch.dict(os.environ, {"AZURE_GENERATIVE_MODEL_NAME": "gpt-4o"}, clear=True):
        with patch("parlant.adapters.nlp.azure_service.create_azure_client"):
            generator: CustomAzureSchematicGenerator[TestSchema] = CustomAzureSchematicGenerator(
                logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
            )
            assert generator.max_tokens == 4096  # Default value


@patch("parlant.adapters.nlp.azure_service.create_azure_client")
def test_that_custom_azure_embedder_initializes_correctly(
    container: Container, mock_create_client: Mock
) -> None:
    """Test CustomAzureEmbedder initialization."""
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    with patch.dict(
        os.environ,
        {
            "AZURE_EMBEDDING_MODEL_NAME": "text-embedding-3-large",
            "AZURE_EMBEDDING_MODEL_WINDOW": "8192",
            "AZURE_EMBEDDING_MODEL_DIMS": "3072",
        },
        clear=True,
    ):
        embedder = CustomAzureEmbedder(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )

        assert embedder.model_name == "text-embedding-3-large"
        assert embedder.max_tokens == 8192
        assert embedder.dimensions == 3072
        mock_create_client.assert_called_once()


@patch("parlant.adapters.nlp.azure_service.create_azure_client")
def test_that_azure_text_embedding_3_large_initializes_correctly(
    container: Container, mock_create_client: Mock
) -> None:
    """Test AzureTextEmbedding3Large initialization."""
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    embedder = AzureTextEmbedding3Large(
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )

    assert embedder.model_name == "text-embedding-3-large"
    assert embedder.max_tokens == 8192
    assert embedder.dimensions == 3072
    mock_create_client.assert_called_once()


@patch("parlant.adapters.nlp.azure_service.create_azure_client")
def test_that_azure_text_embedding_3_small_initializes_correctly(
    container: Container, mock_create_client: Mock
) -> None:
    """Test AzureTextEmbedding3Small initialization."""
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    embedder = AzureTextEmbedding3Small(
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )

    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.max_tokens == 8192
    assert embedder.dimensions == 3072
    mock_create_client.assert_called_once()


@patch("parlant.adapters.nlp.azure_service.create_azure_client")
def test_that_azure_service_returns_custom_schematic_generator_when_configured(
    container: Container,
    mock_create_client: Mock,
) -> None:
    """Test AzureService.get_schematic_generator with custom model."""
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    service = AzureService(
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )

    with patch.dict(os.environ, {"AZURE_GENERATIVE_MODEL_NAME": "gpt-4o"}, clear=True):
        generator = asyncio.run(service.get_schematic_generator(TestSchema))
        assert isinstance(generator, CustomAzureSchematicGenerator)


@patch("parlant.adapters.nlp.azure_service.create_azure_client")
def test_that_azure_service_returns_default_schematic_generator_when_not_configured(
    container: Container,
    mock_create_client: Mock,
) -> None:
    """Test AzureService.get_schematic_generator with default model."""
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    service = AzureService(
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )

    with patch.dict(os.environ, {}, clear=True):
        generator = asyncio.run(service.get_schematic_generator(TestSchema))
        assert isinstance(generator, AzureSchematicGenerator)
        assert generator.model_name == "gpt-4o"


@patch("parlant.adapters.nlp.azure_service.create_azure_client")
def test_that_azure_service_returns_custom_embedder_when_configured(
    container: Container,
    mock_create_client: Mock,
) -> None:
    """Test AzureService.get_embedder with custom model."""
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    service = AzureService(
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )

    with patch.dict(
        os.environ, {"AZURE_EMBEDDING_MODEL_NAME": "text-embedding-3-large"}, clear=True
    ):
        embedder = asyncio.run(service.get_embedder())
        assert isinstance(embedder, CustomAzureEmbedder)


@patch("parlant.adapters.nlp.azure_service.create_azure_client")
def test_that_azure_service_returns_default_embedder_when_not_configured(
    container: Container,
    mock_create_client: Mock,
) -> None:
    """Test AzureService.get_embedder with default model."""
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    service = AzureService(
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )

    with patch.dict(os.environ, {}, clear=True):
        embedder = asyncio.run(service.get_embedder())
        assert isinstance(embedder, AzureTextEmbedding3Large)


@patch("parlant.adapters.nlp.azure_service.DefaultAzureCredential")
def test_that_create_azure_client_creates_client_with_token_provider(
    container: Container,
    mock_credential_class: Mock,
) -> None:
    """Test that create_azure_client creates client with token provider for Azure AD."""
    # Mock credential
    mock_credential = AsyncMock()
    mock_credential_class.return_value = mock_credential

    with patch.dict(os.environ, {"AZURE_ENDPOINT": "https://test.openai.azure.com/"}, clear=True):
        with patch("parlant.adapters.nlp.azure_service.AsyncAzureOpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            create_azure_client()

            # Verify credential was created
            mock_credential_class.assert_called_once()

            # Verify client was created with token provider
            mock_openai_class.assert_called_once()
            call_args = mock_openai_class.call_args
            assert "azure_ad_token_provider" in call_args[1]
            assert call_args[1]["azure_endpoint"] == "https://test.openai.azure.com/"


@patch("parlant.adapters.nlp.azure_service.DefaultAzureCredential")
def test_that_token_provider_errors_are_handled_properly(
    container: Container, mock_credential_class: Mock
) -> None:
    """Test that token provider errors are handled properly."""
    # Mock credential creation failure
    mock_credential_class.side_effect = Exception("Credential creation failed")

    with patch.dict(os.environ, {"AZURE_ENDPOINT": "https://test.openai.azure.com/"}, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            create_azure_client()

        assert "Failed to initialize Azure AD authentication" in str(exc_info.value)
        assert "az login" in str(exc_info.value)


def test_that_default_api_version_is_used_when_not_specified() -> None:
    """Test default API version handling."""
    with patch.dict(
        os.environ,
        {"AZURE_ENDPOINT": "https://test.openai.azure.com/", "AZURE_API_KEY": "test-key"},
        clear=True,
    ):
        with patch("parlant.adapters.nlp.azure_service.AsyncAzureOpenAI") as mock_openai_class:
            create_azure_client()

            call_args = mock_openai_class.call_args
            assert call_args[1]["api_version"] == "2024-08-01-preview"


def test_that_custom_api_version_is_used_when_specified() -> None:
    """Test custom API version handling."""
    with patch.dict(
        os.environ,
        {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-key",
            "AZURE_API_VERSION": "2023-12-01-preview",
        },
        clear=True,
    ):
        with patch("parlant.adapters.nlp.azure_service.AsyncAzureOpenAI") as mock_openai_class:
            create_azure_client()

            call_args = mock_openai_class.call_args
            assert call_args[1]["api_version"] == "2023-12-01-preview"


def test_that_azure_endpoint_is_required() -> None:
    """Test that AZURE_ENDPOINT is required."""
    with patch.dict(os.environ, {"AZURE_API_KEY": "test-key"}, clear=True):
        with pytest.raises(KeyError):
            create_azure_client()


def test_that_azure_ad_error_messages_contain_helpful_information() -> None:
    """Test that Azure AD error messages contain helpful information."""
    with patch.dict(os.environ, {"AZURE_ENDPOINT": "https://test.openai.azure.com/"}, clear=True):
        with patch(
            "parlant.adapters.nlp.azure_service.DefaultAzureCredential"
        ) as mock_credential_class:
            mock_credential_class.side_effect = Exception("Auth failed")

            error = AzureService.verify_environment()
            assert error is not None

            # Check for specific helpful content
            assert "Azure CLI" in error
            assert "Service Principal" in error
            assert "Managed Identity" in error
            assert "Environment Credential" in error
            assert "Workload Identity" in error
            assert "Cognitive Services OpenAI User" in error
            assert "https://docs.microsoft.com" in error


def test_that_api_key_authentication_takes_priority_over_azure_ad() -> None:
    """Test that API key authentication takes priority over Azure AD."""
    with patch.dict(
        os.environ,
        {"AZURE_ENDPOINT": "https://test.openai.azure.com/", "AZURE_API_KEY": "test-key"},
        clear=True,
    ):
        # Even if Azure AD would fail, API key should work
        with patch(
            "parlant.adapters.nlp.azure_service.DefaultAzureCredential"
        ) as mock_credential_class:
            mock_credential_class.side_effect = Exception("Azure AD failed")

            error = AzureService.verify_environment()
            assert error is None  # Should succeed because API key is present
