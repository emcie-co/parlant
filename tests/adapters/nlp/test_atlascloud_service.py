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
from unittest.mock import patch

from lagom import Container

from parlant.adapters.nlp.atlascloud_service import (
    ATLASCLOUD_BASE_URL,
    ATLASCLOUD_DEFAULT_MODEL,
    AtlasCloudSchematicGenerator,
    AtlasCloudService,
    AtlasCloudStreamingTextGenerator,
)
from parlant.core.application_context import ApplicationContext
from parlant.core.health import HealthReporter, NullHealthReporter
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import LocalMeter, Meter
from parlant.core.tracer import LocalTracer, Tracer


def create_container() -> Container:
    container = Container()
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    container[Logger] = logger
    container[Tracer] = tracer
    container[Meter] = LocalMeter(logger)
    container[HealthReporter] = NullHealthReporter(ApplicationContext(instance_id="test"))
    return container


def create_service(container: Container) -> AtlasCloudService:
    return AtlasCloudService(
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
        health_reporter=container[HealthReporter],
    )


def test_that_missing_api_key_returns_error_message() -> None:
    with patch.dict(os.environ, {}, clear=True):
        error = AtlasCloudService.verify_environment()

    assert error is not None
    assert "ATLASCLOUD_API_KEY" in error


def test_that_api_key_satisfies_environment_check() -> None:
    with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=True):
        assert AtlasCloudService.verify_environment() is None


def test_that_service_uses_default_model() -> None:
    with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=True):
        service = create_service(create_container())

    assert service.model_name == ATLASCLOUD_DEFAULT_MODEL


def test_that_service_uses_configured_model() -> None:
    with patch.dict(
        os.environ,
        {"ATLASCLOUD_API_KEY": "test-key", "ATLASCLOUD_MODEL": "custom/model"},
        clear=True,
    ):
        service = create_service(create_container())

    assert service.model_name == "custom/model"


@patch("parlant.adapters.nlp.atlascloud_service.AsyncClient")
def test_that_generators_use_atlascloud_endpoint(mock_client: object) -> None:
    container = create_container()
    with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=True):
        service = create_service(container)
        schematic_generator = asyncio.run(service.get_schematic_generator(dict))
        streaming_generator = asyncio.run(service.get_streaming_text_generator())

    assert isinstance(schematic_generator, AtlasCloudSchematicGenerator)
    assert isinstance(streaming_generator, AtlasCloudStreamingTextGenerator)
    assert schematic_generator.id == f"atlascloud/{ATLASCLOUD_DEFAULT_MODEL}"
    assert streaming_generator.id == f"atlascloud-streaming/{ATLASCLOUD_DEFAULT_MODEL}"
    assert mock_client.call_count == 2  # type: ignore[attr-defined]
    for call in mock_client.call_args_list:  # type: ignore[attr-defined]
        assert call.kwargs == {"base_url": ATLASCLOUD_BASE_URL, "api_key": "test-key"}


def test_that_max_tokens_can_be_configured() -> None:
    container = create_container()
    with patch.dict(
        os.environ,
        {"ATLASCLOUD_API_KEY": "test-key", "ATLASCLOUD_MAX_TOKENS": "4096"},
        clear=True,
    ):
        service = create_service(container)
        generator = asyncio.run(service.get_schematic_generator(dict))
        assert generator.max_tokens == 4096
