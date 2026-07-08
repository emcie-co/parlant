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
import pytest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from openai import AsyncClient
from lagom import Container

from parlant.adapters.nlp.qwen_service import (
    QwenReactGenerator,
    QwenSchematicGenerator,
    QwenService,
    QwenTextEmbedding_V4,
    get_qwen_base_url,
    QWEN_REGION_BASE_URLS,
)
from parlant.core.application_context import ApplicationContext
from parlant.core.common import DefaultBaseModel
from parlant.core.health import HealthReporter, NullHealthReporter
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import LocalMeter, Meter
from parlant.core.nlp.common import ModelSize
from parlant.core.nlp.react import (
    CacheConfig,
    Message,
    ParameterSpec,
    ReasoningConfig,
    ReasoningPart,
    Role,
    TextPart,
    ToolResultPart,
    ToolSpec,
)
from parlant.core.tracer import LocalTracer, Tracer


class _TestSchema(DefaultBaseModel):
    value: str


@pytest.fixture
def container() -> Container:
    container = Container()
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    container[Logger] = logger
    container[Tracer] = tracer
    container[Meter] = meter
    container[HealthReporter] = NullHealthReporter(
        application_context=ApplicationContext(instance_id="test")
    )

    return container


@pytest.fixture
def qwen_react(container: Container) -> QwenReactGenerator:
    return QwenReactGenerator(
        model="qwen3.7-plus",
        logger=container[Logger],
        cache=CacheConfig(enabled=False),
        client=AsyncClient(api_key="offline", base_url="https://example.test/v1"),
    )


def test_that_missing_api_key_returns_error_message() -> None:
    """Test that missing DASHSCOPE_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = QwenService.verify_environment()
        assert error is not None
        assert "DASHSCOPE_API_KEY is not set" in error


def test_that_verify_environment_returns_error_for_invalid_region() -> None:
    """Test that verify_environment returns error for invalid QWEN_REGION."""
    with patch.dict(
        os.environ,
        {"DASHSCOPE_API_KEY": "test-key", "QWEN_REGION": "invalid-region"},
        clear=True,
    ):
        error = QwenService.verify_environment()
        assert error is not None
        assert "Invalid QWEN_REGION 'invalid-region'" in error
        assert "Must be one of: international, domestic" in error


def test_that_get_qwen_base_url_returns_international_by_default() -> None:
    """Test that get_qwen_base_url returns international URL by default."""
    with patch.dict(os.environ, {}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]
        assert url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_that_get_qwen_base_url_returns_domestic_url_when_region_is_domestic() -> None:
    """Test that get_qwen_base_url returns domestic URL when QWEN_REGION is domestic."""
    with patch.dict(os.environ, {"QWEN_REGION": "domestic"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]
        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_that_get_qwen_base_url_returns_international_url_when_region_is_international() -> None:
    """Test that get_qwen_base_url returns international URL when QWEN_REGION is international."""
    with patch.dict(os.environ, {"QWEN_REGION": "international"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]


def test_that_get_qwen_base_url_is_case_insensitive() -> None:
    """Test that QWEN_REGION is case insensitive."""
    with patch.dict(os.environ, {"QWEN_REGION": "DOMESTIC"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]

    with patch.dict(os.environ, {"QWEN_REGION": "Domestic"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]

    with patch.dict(os.environ, {"QWEN_REGION": "INTERNATIONAL"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]


def test_that_get_qwen_base_url_raises_error_for_invalid_region() -> None:
    """Test that get_qwen_base_url raises ValueError for invalid region."""
    with patch.dict(os.environ, {"QWEN_REGION": "invalid_region"}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            get_qwen_base_url()
        assert "Invalid QWEN_REGION" in str(exc_info.value)
        assert "international" in str(exc_info.value)
        assert "domestic" in str(exc_info.value)


def test_that_qwen_base_url_env_var_takes_priority() -> None:
    """Test that QWEN_BASE_URL environment variable takes priority over QWEN_REGION."""
    custom_url = "https://custom.api.url/v1"
    with patch.dict(
        os.environ,
        {"QWEN_BASE_URL": custom_url, "QWEN_REGION": "domestic"},
        clear=True,
    ):
        url = get_qwen_base_url()
        assert url == custom_url


def test_that_qwen_base_url_env_var_works_alone() -> None:
    """Test that QWEN_BASE_URL works without QWEN_REGION set."""
    custom_url = "https://custom.api.url/v1"
    with patch.dict(os.environ, {"QWEN_BASE_URL": custom_url}, clear=True):
        url = get_qwen_base_url()
        assert url == custom_url


def test_that_react_encode_uses_qwen_responses_shape(qwen_react: QwenReactGenerator) -> None:
    weather_tool = ToolSpec(
        name="get_weather",
        description="Get weather for a city.",
        parameters=[ParameterSpec(name="city", type="string")],
    )
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Use tools when useful.")]),
        Message(role=Role.USER, parts=[TextPart(text="Weather in Paris?")]),
    ]

    request = qwen_react._encode(
        history,
        [weather_tool],
        "auto",
        reasoning=ReasoningConfig(effort="high", visibility="summary"),
    )

    assert request["model"] == "qwen3.7-plus"
    assert request["instructions"] == "Use tools when useful."
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["name"] == "get_weather"
    assert request["tool_choice"] == "auto"
    assert request["extra_body"] == {"enable_thinking": True}
    assert "reasoning" not in request
    assert "include" not in request
    assert "prompt_cache_key" not in request


def test_that_react_encode_disables_qwen_thinking_for_minimal_reasoning(
    qwen_react: QwenReactGenerator,
) -> None:
    request = qwen_react._encode(
        [Message(role=Role.USER, parts=[TextPart(text="Hi")])],
        [],
        "auto",
        reasoning=ReasoningConfig(effort="minimal", visibility="summary"),
    )

    assert "extra_body" not in request


def test_that_react_resolves_qwen_models_by_size(qwen_react: QwenReactGenerator) -> None:
    assert qwen_react.resolve_model({"model_size": ModelSize.SMALL}) == "qwen3.6-flash"
    assert qwen_react.resolve_model({"model_size": ModelSize.MEDIUM}) == "qwen3.7-plus"
    assert qwen_react.resolve_model({"model_size": ModelSize.LARGE}) == "qwen3.7-max"


def test_that_react_encode_preserves_tool_result_items(qwen_react: QwenReactGenerator) -> None:
    request = qwen_react._encode(
        [
            Message(
                role=Role.TOOL,
                parts=[
                    ToolResultPart(
                        call_id="call_123",
                        name="get_weather",
                        content={"temperature_c": 22},
                    )
                ],
            )
        ],
        [],
        "auto",
        reasoning=ReasoningConfig(effort="minimal"),
    )

    assert request["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": '{"temperature_c": 22}',
        }
    ]


def test_that_react_encode_drops_raw_reasoning_items(qwen_react: QwenReactGenerator) -> None:
    request = qwen_react._encode(
        [
            Message(
                role=Role.ASSISTANT,
                parts=[
                    ReasoningPart(
                        text="summary",
                        provider_data={
                            "openai_item": {
                                "type": "reasoning",
                                "id": "rs_123",
                                "summary": [{"text": "summary", "type": "summary_text"}],
                            }
                        },
                    ),
                    TextPart(text="done"),
                ],
            )
        ],
        [],
        "auto",
        reasoning=ReasoningConfig(effort="minimal"),
    )

    assert [item["type"] for item in request["input"]] == ["message"]


async def test_that_service_exposes_react_generator(container: Container) -> None:
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
        service = QwenService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )

        generator = await service.get_react_generator()

    assert service.supports_react is True
    assert isinstance(generator, QwenReactGenerator)
    assert generator.model == "qwen3.7-plus"


async def test_that_service_uses_configured_model_for_schematic_generator(
    container: Container,
) -> None:
    with patch.dict(
        os.environ,
        {"DASHSCOPE_API_KEY": "test-key", "QWEN_MODEL": "qwen3.7-plus"},
        clear=True,
    ):
        service = QwenService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            health_reporter=container[HealthReporter],
        )

        generator = await service.get_schematic_generator(_TestSchema)

    assert isinstance(generator, QwenSchematicGenerator)
    assert generator.model_name == "qwen3.7-plus"


async def test_that_embedder_batches_requests_to_qwen_limit(container: Container) -> None:
    embedder = QwenTextEmbedding_V4(
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )
    batch_sizes: list[int] = []

    async def create(**kwargs: Any) -> Any:
        inputs = kwargs["input"]
        batch_sizes.append(len(inputs))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(inputs))]) for _ in inputs],
            usage=SimpleNamespace(prompt_tokens=len(inputs)),
        )

    embedder._client.embeddings.create = create  # type: ignore[method-assign]

    result = await embedder.do_embed([f"text-{i}" for i in range(23)])

    assert batch_sizes == [10, 10, 3]
    assert len(result.vectors) == 23
    assert result.usage.input_tokens == 23
