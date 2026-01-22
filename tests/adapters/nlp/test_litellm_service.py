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
from unittest.mock import AsyncMock, patch, Mock

import pytest
from lagom import Container

from parlant.adapters.nlp.litellm_service import (
    LiteLLMEmbedder,
    LiteLLMService,
)
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer


@pytest.fixture
def container() -> Container:
    from parlant.core.loggers import StdoutLogger
    from parlant.core.tracer import LocalTracer
    from parlant.core.meter import LocalMeter

    container = Container()
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    container[Logger] = logger
    container[Tracer] = tracer
    container[Meter] = meter

    return container


def test_that_litellm_embedder_initializes_correctly(container: Container) -> None:
    """Test LiteLLMEmbedder initialization with correct properties."""
    embedder = LiteLLMEmbedder(
        model_name="text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        base_url=None,
    )

    assert embedder._model_name == "text-embedding-3-small"
    assert embedder.id == "litellm/text-embedding-3-small"
    assert embedder.max_tokens == 8192
    assert embedder.dimensions == 1536


def test_that_litellm_embedder_accepts_base_url(container: Container) -> None:
    """Test LiteLLMEmbedder initialization with custom base_url."""
    embedder = LiteLLMEmbedder(
        model_name="text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        base_url="http://localhost:8000",
    )

    assert embedder._base_url == "http://localhost:8000"


@patch("parlant.adapters.nlp.litellm_service.litellm")
def test_that_litellm_embedder_calls_aembedding_with_correct_params(
    mock_litellm: Mock, container: Container
) -> None:
    """Test LiteLLMEmbedder calls litellm.aembedding with correct parameters."""
    mock_response = Mock()
    mock_response.data = [
        {"embedding": [0.1, 0.2, 0.3]},
        {"embedding": [0.4, 0.5, 0.6]},
    ]
    mock_litellm.aembedding = AsyncMock(return_value=mock_response)

    embedder = LiteLLMEmbedder(
        model_name="text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        base_url="http://localhost:8000",
    )

    with patch.dict(os.environ, {"LITELLM_PROVIDER_API_KEY": "test-key"}, clear=False):
        result = asyncio.run(embedder.do_embed(["hello", "world"]))

    mock_litellm.aembedding.assert_called_once_with(
        model="text-embedding-3-small",
        input=["hello", "world"],
        api_key="test-key",
        api_base="http://localhost:8000",
    )
    assert result.vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@patch("parlant.adapters.nlp.litellm_service.litellm")
def test_that_litellm_embedder_passes_none_api_key_when_not_set(
    mock_litellm: Mock, container: Container
) -> None:
    """Test LiteLLMEmbedder passes None for api_key when env var not set."""
    mock_response = Mock()
    mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
    mock_litellm.aembedding = AsyncMock(return_value=mock_response)

    embedder = LiteLLMEmbedder(
        model_name="text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    # Ensure LITELLM_PROVIDER_API_KEY is not set
    env = {k: v for k, v in os.environ.items() if k != "LITELLM_PROVIDER_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        asyncio.run(embedder.do_embed(["test"]))

    call_kwargs = mock_litellm.aembedding.call_args[1]
    assert call_kwargs["api_key"] is None


def test_that_litellm_service_returns_litellm_embedder_when_embedding_model_configured(
    container: Container,
) -> None:
    """Test LiteLLMService.get_embedder returns LiteLLMEmbedder when LITELLM_EMBEDDING_MODEL_NAME is set."""
    with patch.dict(
        os.environ,
        {
            "LITELLM_PROVIDER_MODEL_NAME": "gpt-4",
            "LITELLM_EMBEDDING_MODEL_NAME": "text-embedding-3-small",
        },
        clear=False,
    ):
        service = LiteLLMService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        embedder = asyncio.run(service.get_embedder())

        assert isinstance(embedder, LiteLLMEmbedder)
        assert embedder._model_name == "text-embedding-3-small"


@patch("parlant.adapters.nlp.litellm_service.JinaAIEmbedder")
def test_that_litellm_service_returns_jina_embedder_when_embedding_model_not_configured(
    mock_jina_embedder: Mock, container: Container
) -> None:
    """Test LiteLLMService.get_embedder falls back to JinaAIEmbedder when LITELLM_EMBEDDING_MODEL_NAME is not set."""
    mock_jina_instance = Mock()
    mock_jina_embedder.return_value = mock_jina_instance

    env = {k: v for k, v in os.environ.items() if k != "LITELLM_EMBEDDING_MODEL_NAME"}
    env["LITELLM_PROVIDER_MODEL_NAME"] = "gpt-4"

    with patch.dict(os.environ, env, clear=True):
        service = LiteLLMService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        embedder = asyncio.run(service.get_embedder())

        assert embedder is mock_jina_instance
        mock_jina_embedder.assert_called_once()
