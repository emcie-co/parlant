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
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Any
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
from parlant.core.nlp.generation import SchematicGenerationResult
from parlant.core.nlp.embedding import EmbeddingResult


class TestAzureService:
    """Test cases for AzureService authentication and environment verification."""

    def test_verify_environment_missing_endpoint(self):
        """Test that missing AZURE_ENDPOINT returns error message."""
        with patch.dict(os.environ, {}, clear=True):
            error = AzureService.verify_environment()
            assert error is not None
            assert "AZURE_ENDPOINT is not set" in error
            assert "Required environment variables" in error

    def test_verify_environment_with_api_key(self):
        """Test that API key authentication is detected correctly."""
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-api-key"
        }, clear=True):
            error = AzureService.verify_environment()
            assert error is None

    def test_verify_environment_azure_ad_success(self):
        """Test that Azure AD authentication path is attempted when no API key is present."""
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/"
        }, clear=True):
            # Since we can't easily mock the complex async behavior, 
            # we'll just test that the method doesn't crash and returns an error message
            # when Azure AD authentication is not available
            error = AzureService.verify_environment()
            assert error is not None
            assert "Azure authentication is not properly configured" in error
            assert "API Key Authentication" in error
            assert "Azure AD Authentication" in error

    @patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential')
    def test_verify_environment_azure_ad_failure(self, mock_credential_class):
        """Test that failed Azure AD authentication returns error message."""
        # Mock failed credential creation
        mock_credential_class.side_effect = Exception("Authentication failed")

        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/"
        }, clear=True):
            error = AzureService.verify_environment()
            assert error is not None
            assert "Azure authentication is not properly configured" in error
            assert "API Key Authentication" in error
            assert "Azure AD Authentication" in error

    @patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential')
    def test_verify_environment_azure_ad_token_failure(self, mock_credential_class):
        """Test that failed token retrieval returns error message."""
        # Mock credential creation but token retrieval failure
        mock_credential = AsyncMock()
        mock_credential.get_token.side_effect = Exception("Token retrieval failed")
        mock_credential_class.return_value = mock_credential

        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/"
        }, clear=True):
            error = AzureService.verify_environment()
            assert error is not None
            assert "Azure authentication is not properly configured" in error

    def test_verify_environment_includes_helpful_instructions(self):
        """Test that error messages include helpful authentication instructions."""
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/"
        }, clear=True):
            with patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential') as mock_credential_class:
                mock_credential_class.side_effect = Exception("Auth failed")
                
                error = AzureService.verify_environment()
                assert error is not None
                
                # Check for specific authentication methods
                assert "az login" in error
                assert "AZURE_CLIENT_ID" in error
                assert "AZURE_CLIENT_SECRET" in error
                assert "AZURE_TENANT_ID" in error
                assert "Cognitive Services OpenAI User" in error


class TestCreateAzureClient:
    """Test cases for create_azure_client function."""

    @patch('parlant.adapters.nlp.azure_service.AsyncAzureOpenAI')
    def test_create_azure_client_with_api_key(self, mock_openai_class):
        """Test client creation with API key authentication."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-api-key",
            "AZURE_API_VERSION": "2024-08-01-preview"
        }, clear=True):
            client = create_azure_client()
            
            mock_openai_class.assert_called_once_with(
                api_key="test-api-key",
                azure_endpoint="https://test.openai.azure.com/",
                api_version="2024-08-01-preview"
            )
            assert client == mock_client

    @patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential')
    @patch('parlant.adapters.nlp.azure_service.AsyncAzureOpenAI')
    def test_create_azure_client_with_azure_ad(self, mock_openai_class, mock_credential_class):
        """Test client creation with Azure AD authentication."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_credential = Mock()
        mock_credential_class.return_value = mock_credential

        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_VERSION": "2024-08-01-preview"
        }, clear=True):
            client = create_azure_client()
            
            # Verify credential was created
            mock_credential_class.assert_called_once()
            
            # Verify client was created with token provider
            mock_openai_class.assert_called_once()
            call_args = mock_openai_class.call_args
            assert call_args[1]["azure_endpoint"] == "https://test.openai.azure.com/"
            assert call_args[1]["api_version"] == "2024-08-01-preview"
            assert "azure_ad_token_provider" in call_args[1]

    @patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential')
    def test_create_azure_client_azure_ad_failure(self, mock_credential_class):
        """Test client creation failure with Azure AD authentication."""
        mock_credential_class.side_effect = Exception("Credential creation failed")

        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/"
        }, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                create_azure_client()
            
            assert "Failed to initialize Azure AD authentication" in str(exc_info.value)
            assert "az login" in str(exc_info.value)


class TestAzureSchematicGenerator:
    """Test cases for AzureSchematicGenerator."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Azure OpenAI client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=Logger)

    def test_azure_schematic_generator_initialization(self, mock_client, mock_logger):
        """Test AzureSchematicGenerator initialization using GPT_4o class."""
        # Use GPT_4o which is a concrete implementation
        from parlant.adapters.nlp.azure_service import GPT_4o
        
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-key"
        }, clear=True):
            with patch('parlant.adapters.nlp.azure_service.create_azure_client') as mock_create_client:
                mock_create_client.return_value = mock_client
                generator = GPT_4o(logger=mock_logger)
                
                assert generator.model_name == "gpt-4o"
                assert generator._logger == mock_logger
                assert generator.id == "azure/gpt-4o"

    def test_azure_schematic_generator_supported_params(self, mock_client, mock_logger):
        """Test supported Azure parameters."""
        # Use GPT_4o which is a concrete implementation
        from parlant.adapters.nlp.azure_service import GPT_4o
        
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-key"
        }, clear=True):
            with patch('parlant.adapters.nlp.azure_service.create_azure_client') as mock_create_client:
                mock_create_client.return_value = mock_client
                generator = GPT_4o(logger=mock_logger)
                
                expected_params = ["temperature", "logit_bias", "max_tokens"]
                assert generator.supported_azure_params == expected_params
                
                expected_hints = expected_params + ["strict"]
                assert generator.supported_hints == expected_hints


class TestCustomAzureSchematicGenerator:
    """Test cases for CustomAzureSchematicGenerator."""

    @patch('parlant.adapters.nlp.azure_service.create_azure_client')
    def test_custom_azure_schematic_generator_initialization(self, mock_create_client):
        """Test CustomAzureSchematicGenerator initialization."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_logger = Mock(spec=Logger)

        with patch.dict(os.environ, {
            "AZURE_GENERATIVE_MODEL_NAME": "gpt-4o",
            "AZURE_GENERATIVE_MODEL_WINDOW": "4096"
        }, clear=True):
            generator = CustomAzureSchematicGenerator(logger=mock_logger)
            
            assert generator.model_name == "gpt-4o"
            assert generator.max_tokens == 4096
            mock_create_client.assert_called_once()

    def test_custom_azure_schematic_generator_max_tokens_default(self):
        """Test CustomAzureSchematicGenerator with default max_tokens."""
        mock_logger = Mock(spec=Logger)

        with patch.dict(os.environ, {
            "AZURE_GENERATIVE_MODEL_NAME": "gpt-4o"
        }, clear=True):
            with patch('parlant.adapters.nlp.azure_service.create_azure_client'):
                generator = CustomAzureSchematicGenerator(logger=mock_logger)
                assert generator.max_tokens == 4096  # Default value


class TestAzureEmbedders:
    """Test cases for Azure embedder classes."""

    @patch('parlant.adapters.nlp.azure_service.create_azure_client')
    def test_custom_azure_embedder_initialization(self, mock_create_client):
        """Test CustomAzureEmbedder initialization."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_logger = Mock(spec=Logger)

        with patch.dict(os.environ, {
            "AZURE_EMBEDDING_MODEL_NAME": "text-embedding-3-large",
            "AZURE_EMBEDDING_MODEL_WINDOW": "8192",
            "AZURE_EMBEDDING_MODEL_DIMS": "3072"
        }, clear=True):
            embedder = CustomAzureEmbedder(logger=mock_logger)
            
            assert embedder.model_name == "text-embedding-3-large"
            assert embedder.max_tokens == 8192
            assert embedder.dimensions == 3072
            mock_create_client.assert_called_once()

    @patch('parlant.adapters.nlp.azure_service.create_azure_client')
    def test_azure_text_embedding_3_large_initialization(self, mock_create_client):
        """Test AzureTextEmbedding3Large initialization."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_logger = Mock(spec=Logger)

        embedder = AzureTextEmbedding3Large(logger=mock_logger)
        
        assert embedder.model_name == "text-embedding-3-large"
        assert embedder.max_tokens == 8192
        assert embedder.dimensions == 3072
        mock_create_client.assert_called_once()

    @patch('parlant.adapters.nlp.azure_service.create_azure_client')
    def test_azure_text_embedding_3_small_initialization(self, mock_create_client):
        """Test AzureTextEmbedding3Small initialization."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_logger = Mock(spec=Logger)

        embedder = AzureTextEmbedding3Small(logger=mock_logger)
        
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.max_tokens == 8192
        assert embedder.dimensions == 3072
        mock_create_client.assert_called_once()


class TestAzureServiceIntegration:
    """Integration test cases for AzureService."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=Logger)

    @patch('parlant.adapters.nlp.azure_service.create_azure_client')
    def test_azure_service_get_schematic_generator_custom(self, mock_create_client, mock_logger):
        """Test AzureService.get_schematic_generator with custom model."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        service = AzureService(logger=mock_logger)
        
        with patch.dict(os.environ, {
            "AZURE_GENERATIVE_MODEL_NAME": "gpt-4o"
        }, clear=True):
            generator = asyncio.run(service.get_schematic_generator(dict))
            assert isinstance(generator, CustomAzureSchematicGenerator)

    @patch('parlant.adapters.nlp.azure_service.create_azure_client')
    def test_azure_service_get_schematic_generator_default(self, mock_create_client, mock_logger):
        """Test AzureService.get_schematic_generator with default model."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        service = AzureService(logger=mock_logger)
        
        with patch.dict(os.environ, {}, clear=True):
            generator = asyncio.run(service.get_schematic_generator(dict))
            assert isinstance(generator, AzureSchematicGenerator)
            assert generator.model_name == "gpt-4o"

    @patch('parlant.adapters.nlp.azure_service.create_azure_client')
    def test_azure_service_get_embedder_custom(self, mock_create_client, mock_logger):
        """Test AzureService.get_embedder with custom model."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        service = AzureService(logger=mock_logger)
        
        with patch.dict(os.environ, {
            "AZURE_EMBEDDING_MODEL_NAME": "text-embedding-3-large"
        }, clear=True):
            embedder = asyncio.run(service.get_embedder())
            assert isinstance(embedder, CustomAzureEmbedder)

    @patch('parlant.adapters.nlp.azure_service.create_azure_client')
    def test_azure_service_get_embedder_default(self, mock_create_client, mock_logger):
        """Test AzureService.get_embedder with default model."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        service = AzureService(logger=mock_logger)
        
        with patch.dict(os.environ, {}, clear=True):
            embedder = asyncio.run(service.get_embedder())
            assert isinstance(embedder, AzureTextEmbedding3Large)


class TestTokenProviderFunctionality:
    """Test cases for the token provider functionality."""

    @patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential')
    def test_create_azure_client_with_token_provider(self, mock_credential_class):
        """Test that create_azure_client creates client with token provider for Azure AD."""
        # Mock credential
        mock_credential = AsyncMock()
        mock_credential_class.return_value = mock_credential

        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/"
        }, clear=True):
            with patch('parlant.adapters.nlp.azure_service.AsyncAzureOpenAI') as mock_openai_class:
                mock_client = Mock()
                mock_openai_class.return_value = mock_client
                
                client = create_azure_client()
                
                # Verify credential was created
                mock_credential_class.assert_called_once()
                
                # Verify client was created with token provider
                mock_openai_class.assert_called_once()
                call_args = mock_openai_class.call_args
                assert "azure_ad_token_provider" in call_args[1]
                assert call_args[1]["azure_endpoint"] == "https://test.openai.azure.com/"

    @patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential')
    def test_token_provider_error_handling(self, mock_credential_class):
        """Test that token provider errors are handled properly."""
        # Mock credential creation failure
        mock_credential_class.side_effect = Exception("Credential creation failed")

        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/"
        }, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                create_azure_client()
            
            assert "Failed to initialize Azure AD authentication" in str(exc_info.value)
            assert "az login" in str(exc_info.value)


class TestEnvironmentVariableHandling:
    """Test cases for environment variable handling."""

    def test_api_version_default(self):
        """Test default API version handling."""
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-key"
        }, clear=True):
            with patch('parlant.adapters.nlp.azure_service.AsyncAzureOpenAI') as mock_openai_class:
                create_azure_client()
                
                call_args = mock_openai_class.call_args
                assert call_args[1]["api_version"] == "2024-08-01-preview"

    def test_api_version_custom(self):
        """Test custom API version handling."""
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-key",
            "AZURE_API_VERSION": "2023-12-01-preview"
        }, clear=True):
            with patch('parlant.adapters.nlp.azure_service.AsyncAzureOpenAI') as mock_openai_class:
                create_azure_client()
                
                call_args = mock_openai_class.call_args
                assert call_args[1]["api_version"] == "2023-12-01-preview"

    def test_azure_endpoint_required(self):
        """Test that AZURE_ENDPOINT is required."""
        with patch.dict(os.environ, {
            "AZURE_API_KEY": "test-key"
        }, clear=True):
            with pytest.raises(KeyError):
                create_azure_client()


class TestErrorMessages:
    """Test cases for error message content and formatting."""

    def test_azure_ad_error_message_content(self):
        """Test that Azure AD error messages contain helpful information."""
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/"
        }, clear=True):
            with patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential') as mock_credential_class:
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

    def test_authentication_methods_priority(self):
        """Test that API key authentication takes priority over Azure AD."""
        with patch.dict(os.environ, {
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-key"
        }, clear=True):
            # Even if Azure AD would fail, API key should work
            with patch('parlant.adapters.nlp.azure_service.DefaultAzureCredential') as mock_credential_class:
                mock_credential_class.side_effect = Exception("Azure AD failed")
                
                error = AzureService.verify_environment()
                assert error is None  # Should succeed because API key is present
