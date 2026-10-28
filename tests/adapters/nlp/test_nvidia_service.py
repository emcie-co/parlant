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

from collections.abc import Generator
import os
from lagom import Container
import pytest
from unittest.mock import AsyncMock, patch, Mock
import asyncio
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from parlant.adapters.nlp.nvidia_service import (  # type: ignore[reportMissingImports]
    NvidiaService,
    NvidiaSchematicGenerator,
    NvidiaEmbedder,
    NvidiaNemotron35SuperVL120B,
    NvidiaNemotron35Lightning30B,
    NvidiaNemotron35BoltE4B,
    NvidiaEstimatingTokenizer,
)
from parlant.core.loggers import Logger
from parlant.core.common import DefaultBaseModel
from parlant.core.health import HealthReporter
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer


class SchemaData(DefaultBaseModel):
    """Test schema for type checking."""

    test_field: str = "test_value"


@pytest.fixture(autouse=True)
def set_api_keys() -> Generator[None, None, None]:
    """Set API keys for tests that use container fixture."""
    # Container fixture initializes ServiceRegistry which requires OPENAI_API_KEY
    # NVIDIA tests also need NVIDIA_API_KEY
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "NVIDIA_API_KEY": "test-nvidia-key",
        },
        clear=False,
    ):
        yield


class _OfflineEncoding:
    """Stands in for a tiktoken encoding so tests never download encoding files."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())


@pytest.fixture(autouse=True)
def offline_tiktoken() -> Generator[None, None, None]:
    """Keep the tests offline: tiktoken otherwise downloads its encoding on first use."""
    with patch(
        "parlant.adapters.nlp.nvidia_service.tiktoken.encoding_for_model",
        return_value=_OfflineEncoding(),
    ):
        yield


def test_that_missing_nvidia_api_key_returns_error_message() -> None:
    """Test that missing NVIDIA_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = NvidiaService.verify_environment()
        assert error is not None
        assert "NVIDIA_API_KEY is not set" in error


def test_that_present_api_key_returns_none() -> None:
    """Test that present API key returns None (success)."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
        error = NvidiaService.verify_environment()
        assert error is None


def test_that_nvidia_service_initializes_with_default_model() -> None:
    """Test NvidiaService initialization with default model."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
        mock_logger = Mock()
        mock_meter = Mock()

        mock_health_reporter = Mock()
        mock_tracer = Mock()
        service = NvidiaService(
            logger=mock_logger,
            tracer=mock_tracer,
            meter=mock_meter,
            health_reporter=mock_health_reporter,
        )
        assert service.model_name == "nvidia/nemotron-3.5-super-vl-120b-a12b"


def test_that_nvidia_service_initializes_with_custom_model() -> None:
    """Test NvidiaService initialization with custom model from environment."""
    with patch.dict(
        os.environ,
        {
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_MODEL": "nvidia/nemotron-3.5-lightning-30b-a3b",
        },
        clear=True,
    ):
        mock_logger = Mock()
        mock_meter = Mock()

        mock_health_reporter = Mock()
        mock_tracer = Mock()
        service = NvidiaService(
            logger=mock_logger,
            tracer=mock_tracer,
            meter=mock_meter,
            health_reporter=mock_health_reporter,
        )
        assert service.model_name == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_that_nvidia_service_uses_environment_model() -> None:
    """Test NvidiaService uses NVIDIA_MODEL from environment."""
    with patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "test-key", "NVIDIA_MODEL": "nvidia/nemotron-3.5-bolt-e4b"},
        clear=True,
    ):
        mock_logger = Mock()
        mock_meter = Mock()

        mock_health_reporter = Mock()
        mock_tracer = Mock()
        service = NvidiaService(
            logger=mock_logger,
            tracer=mock_tracer,
            meter=mock_meter,
            health_reporter=mock_health_reporter,
        )
        assert service.model_name == "nvidia/nemotron-3.5-bolt-e4b"


def test_that_nvidia_service_respects_custom_max_tokens() -> None:
    """Test NvidiaService respects max_tokens from environment variable."""
    with patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "test-key", "NVIDIA_MAX_TOKENS": "4096"},
        clear=True,
    ):
        mock_logger = Mock()
        mock_meter = Mock()

        mock_health_reporter = Mock()
        mock_tracer = Mock()
        service = NvidiaService(
            logger=mock_logger,
            tracer=mock_tracer,
            meter=mock_meter,
            health_reporter=mock_health_reporter,
        )
        # max_tokens is used when creating generators, not stored in service
        assert service.model_name == "nvidia/nemotron-3.5-super-vl-120b-a12b"  # Default model


def test_that_nvidia_estimating_tokenizer_works(container: Container) -> None:
    """Test NvidiaEstimatingTokenizer token estimation."""
    tokenizer = NvidiaEstimatingTokenizer(model_name="nvidia/nemotron-3.5-super-vl-120b-a12b")
    tokens = asyncio.run(tokenizer.estimate_token_count("Hello world"))
    assert tokens > 0


def test_that_nvidia_nemotron_super_generator_initializes_correctly(
    container: Container,
) -> None:
    """Test NvidiaNemotron35SuperVL120B initialization."""
    generator = NvidiaNemotron35SuperVL120B[SchemaData](
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )
    assert generator.model_name == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert generator.id == "nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert generator.max_tokens == 8192


def test_that_nvidia_nemotron_lightning_generator_initializes_correctly(
    container: Container,
) -> None:
    """Test NvidiaNemotron35Lightning30B initialization."""
    generator = NvidiaNemotron35Lightning30B[SchemaData](
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )
    assert generator.model_name == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert generator.max_tokens == 8192


def test_that_nvidia_nemotron_bolt_generator_initializes_correctly(
    container: Container,
) -> None:
    """Test NvidiaNemotron35BoltE4B initialization."""
    generator = NvidiaNemotron35BoltE4B[SchemaData](
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )
    assert generator.model_name == "nvidia/nemotron-3.5-bolt-e4b"
    assert generator.max_tokens == 8192


@patch("parlant.adapters.nlp.nvidia_service.AsyncClient")
def test_that_nvidia_generator_uses_default_base_url_and_api_key(mock_client_class: Mock) -> None:
    """Test that NVIDIA generator uses the default endpoint and NVIDIA_API_KEY."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
        _ = NvidiaSchematicGenerator[SchemaData](
            model_name="nvidia/nemotron-3.5-super-vl-120b-a12b",
            logger=Mock(),
            tracer=Mock(),
            meter=Mock(),
            health_reporter=Mock(),
        )

        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert call_args[1]["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert call_args[1]["api_key"] == "test-key"


@patch("parlant.adapters.nlp.nvidia_service.AsyncClient")
def test_that_nvidia_generator_uses_base_url_from_environment(mock_client_class: Mock) -> None:
    """Test that NVIDIA generator honours NVIDIA_BASE_URL for self-hosted NIM."""
    with patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "test-key", "NVIDIA_BASE_URL": "http://localhost:8000/v1"},
        clear=True,
    ):
        _ = NvidiaSchematicGenerator[SchemaData](
            model_name="nvidia/nemotron-3.5-super-vl-120b-a12b",
            logger=Mock(),
            tracer=Mock(),
            meter=Mock(),
            health_reporter=Mock(),
        )

        call_args = mock_client_class.call_args
        assert call_args[1]["base_url"] == "http://localhost:8000/v1"


@patch("parlant.adapters.nlp.nvidia_service.AsyncClient")
def test_that_nvidia_embedder_uses_base_url_from_environment(mock_client_class: Mock) -> None:
    """Test that NVIDIA embedder honours NVIDIA_BASE_URL for self-hosted NIM."""
    with patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "test-key", "NVIDIA_BASE_URL": "http://localhost:8000/v1"},
        clear=True,
    ):
        _ = NvidiaEmbedder(
            model_name="nvidia/nemotron-3.5-embed-8b",
            logger=Mock(),
            tracer=Mock(),
            meter=Mock(),
            health_reporter=Mock(),
        )

        call_args = mock_client_class.call_args
        assert call_args[1]["base_url"] == "http://localhost:8000/v1"
        assert call_args[1]["api_key"] == "test-key"


@patch("parlant.adapters.nlp.nvidia_service.AsyncClient")
async def test_that_nvidia_generator_handles_json_mode_error(mock_client_class: Mock) -> None:
    """Test that NVIDIA generator handles JSON mode errors gracefully."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client

    # Mock BadRequestError for JSON mode
    from openai import BadRequestError

    mock_client.chat.completions.create.side_effect = BadRequestError(
        "Model does not support JSON mode",
        body={"error": {"message": "JSON mode error"}},
        response=Mock(),
    )

    mock_logger = Mock()
    mock_tracer = Mock()
    mock_meter = Mock()

    mock_health_reporter = Mock()

    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=False):
        generator = NvidiaSchematicGenerator[SchemaData](
            model_name="test-model",
            logger=mock_logger,
            tracer=mock_tracer,
            meter=mock_meter,
            health_reporter=mock_health_reporter,
        )

    # Should fail since we're mocking the error
    with pytest.raises(BadRequestError):
        await generator.do_generate("Test prompt")


async def test_that_nvidia_generator_handles_successful_response(
    container: Container,
) -> None:
    """Test NVIDIA generator with successful JSON response."""
    mock_response = Mock(spec=ChatCompletion)
    mock_response.choices = [
        Choice(
            message=ChatCompletionMessage(role="assistant", content='{"test_field": "test_value"}'),
            finish_reason="stop",
            index=0,
        )
    ]
    mock_response.usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    with patch("parlant.adapters.nlp.nvidia_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = NvidiaSchematicGenerator[SchemaData](
            model_name="nvidia/nemotron-3.5-super-vl-120b-a12b",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )

        result = await generator.do_generate('Generate {"test_field": "test_value"}')

        assert result.content.test_field == "test_value"
        assert result.info.usage.input_tokens == 10
        assert result.info.usage.output_tokens == 20

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_that_nvidia_service_returns_correct_generator(container: Container) -> None:
    """Test NvidiaService.get_schematic_generator with default model."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
        service = NvidiaService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert isinstance(generator, NvidiaNemotron35SuperVL120B)
        assert generator.model_name == "nvidia/nemotron-3.5-super-vl-120b-a12b"


def test_that_nvidia_service_returns_correct_generator_for_lightning(
    container: Container,
) -> None:
    """Test NvidiaService.get_schematic_generator with the Lightning model."""
    with patch.dict(
        os.environ,
        {
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_MODEL": "nvidia/nemotron-3.5-lightning-30b-a3b",
        },
        clear=True,
    ):
        service = NvidiaService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert isinstance(generator, NvidiaNemotron35Lightning30B)
        assert generator.model_name == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_that_nvidia_service_creates_dynamic_generator_for_unknown_model(
    container: Container,
) -> None:
    """Test NvidiaService creates dynamic generator for unknown model."""
    with patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "test-key", "NVIDIA_MODEL": "custom/model-name"},
        clear=True,
    ):
        service = NvidiaService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert isinstance(generator, NvidiaSchematicGenerator)
        assert generator.model_name == "custom/model-name"
        assert generator.max_tokens == 8192


def test_that_nvidia_service_uses_custom_max_tokens(container: Container) -> None:
    """Test NvidiaService uses max_tokens from environment for unknown model."""
    with patch.dict(
        os.environ,
        {
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_MODEL": "custom/model",
            "NVIDIA_MAX_TOKENS": "2048",
        },
        clear=True,
    ):
        service = NvidiaService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert generator.max_tokens == 2048


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.nvidia_service.AsyncClient")
async def test_that_nvidia_embedder_retries_empty_embedding_value_error(
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

    embedder = NvidiaEmbedder(
        model_name="nvidia/nemotron-3.5-embed-1b",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )

    result = await embedder.do_embed(["hello"])

    assert result.vectors == [[0.1, 0.2, 0.3]]
    assert mock_client.embeddings.create.await_count == 2
    mock_sleep.assert_awaited_once()


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.nvidia_service.AsyncClient")
async def test_that_nvidia_embedder_retries_empty_embedding_response_data(
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

    embedder = NvidiaEmbedder(
        model_name="nvidia/nemotron-3.5-embed-1b",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )

    result = await embedder.do_embed(["hello"])

    assert result.vectors == [[0.4, 0.5]]
    assert mock_client.embeddings.create.await_count == 2
    mock_sleep.assert_awaited_once()


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.nvidia_service.AsyncClient")
async def test_that_nvidia_embedder_does_not_retry_unrelated_value_error(
    mock_client_class: Mock,
    mock_sleep: AsyncMock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=ValueError("Embedding payload is malformed")
    )
    mock_client_class.return_value = mock_client

    embedder = NvidiaEmbedder(
        model_name="nvidia/nemotron-3.5-embed-1b",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )

    with pytest.raises(ValueError, match="Embedding payload is malformed"):
        await embedder.do_embed(["hello"])

    assert mock_client.embeddings.create.await_count == 1
    mock_sleep.assert_not_awaited()


@patch("parlant.adapters.nlp.nvidia_service.AsyncClient")
async def test_that_nvidia_embedder_sends_input_type(
    mock_client_class: Mock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=Mock(data=[Mock(embedding=[0.1])]))
    mock_client_class.return_value = mock_client

    embedder = NvidiaEmbedder(
        model_name="nvidia/nemotron-3.5-embed-1b",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )

    await embedder.do_embed(["hello"])

    call_kwargs = mock_client.embeddings.create.call_args[1]
    assert call_kwargs["extra_body"] == {"input_type": "passage"}


def test_that_nvidia_service_returns_nvidia_embedder(container: Container) -> None:
    """Test NvidiaService returns the default NVIDIA embedder."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
        service = NvidiaService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )
        embedder = asyncio.run(service.get_embedder())
        from parlant.adapters.nlp.nvidia_service import NvidiaNemotron35Embed8B

        assert isinstance(embedder, NvidiaNemotron35Embed8B)
        assert embedder.dimensions == 4096


def test_that_nvidia_service_returns_no_moderation(container: Container) -> None:
    """Test NvidiaService returns NoModeration."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
        service = NvidiaService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )
        moderation = asyncio.run(service.get_moderation_service())
        from parlant.core.nlp.moderation import NoModeration

        assert isinstance(moderation, NoModeration)


def test_that_nvidia_generator_supports_correct_parameters(container: Container) -> None:
    """Test supported NVIDIA parameters."""
    generator = NvidiaSchematicGenerator[SchemaData](
        model_name="nvidia/nemotron-3.5-super-vl-120b-a12b",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )

    expected_params = ["temperature", "max_tokens"]
    assert generator.supported_nvidia_params == expected_params


def test_that_sdk_nvidia_factory_returns_nvidia_service(container: Container) -> None:
    """Test that p.NLPServices.nvidia builds an NvidiaService."""
    from parlant.sdk import NLPServices

    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=False):
        factory = NLPServices.nvidia()
        assert callable(factory)
        service = factory(container)  # type: ignore[operator]
        assert isinstance(service, NvidiaService)
