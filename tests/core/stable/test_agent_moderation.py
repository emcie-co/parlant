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

from datetime import datetime, timezone
from typing import Any, cast

import pytest

from parlant.core.agents import Agent, AgentId, MessageOutputMode, Moderation
from parlant.core.app_modules.agents import (
    validate_moderation_configuration,
    validate_required_moderation,
)
from parlant.core.loggers import StdoutLogger
from parlant.core.tracer import LocalTracer
from parlant.core.nlp.model_routing import (
    ModelConfig,
    ModelConfigurationService,
    ModelImplementationRegistry,
    ModelRouter,
    ModelScope,
    RuntimeModelRouteSource,
    TransientModelRouteStore,
)
from parlant.core.nlp.moderation import (
    CustomerModerationContext,
    ModerationCheck,
    ModerationService,
    NoModeration,
)
from parlant.core.nlp.null_provider import load_null_models
from parlant.core.nlp.rate_limits import TransientRateLimiter


def _agent(moderation: Moderation) -> Agent:
    now = datetime.now(timezone.utc)
    return Agent(
        engine="compass",
        id=AgentId("agent-1"),
        name="test-agent",
        instructions=None,
        creation_utc=now,
        modified_utc=now,
        max_steps=15,
        groups=[],
        moderation=moderation,
    )


def _build_router() -> tuple[ModelRouter, ModelConfigurationService]:
    route_store = TransientModelRouteStore()
    runtime_routes = RuntimeModelRouteSource()
    registry = ModelImplementationRegistry(secret_loader=cast(Any, lambda name: None))
    load_null_models(registry)
    configuration = ModelConfigurationService(route_store, runtime_routes, registry)
    router = ModelRouter(route_store, runtime_routes, registry, TransientRateLimiter())
    return router, configuration


async def test_that_required_moderation_probes_both_routes() -> None:
    router, configuration = _build_router()
    await configuration.configure(
        ModelScope.server(),
        {"moderation.customer": ModelConfig("null"), "moderation.agent": ModelConfig("null")},
    )

    await validate_required_moderation(router, _agent(Moderation.ON))


async def test_that_required_moderation_fails_when_a_route_is_unconfigured() -> None:
    router, configuration = _build_router()
    await configuration.configure(
        ModelScope.server(),
        {"moderation.customer": ModelConfig("null")},
    )

    with pytest.raises(ValueError, match="moderation.agent"):
        await validate_required_moderation(router, _agent(Moderation.ON))


async def test_that_auto_and_off_moderation_require_nothing() -> None:
    router, _ = _build_router()  # nothing routed: any probe would fail

    await validate_required_moderation(router, _agent(Moderation.AUTO))
    await validate_required_moderation(router, _agent(Moderation.OFF))


async def test_that_the_null_provider_registers_a_never_flagging_moderation_service() -> None:
    from parlant.core.nlp.model_routing import ModelCapability, ModelImplementationRegistry
    from parlant.core.nlp.null_provider import load_null_models

    registry = ModelImplementationRegistry(secret_loader=cast(Any, lambda name: None))
    load_null_models(registry)

    factory = cast(Any, registry)._factories[("null", ModelCapability.MODERATION)]
    service = await factory("null", {})  # the model name is ignored

    assert isinstance(service, NoModeration)
    check = await service.moderate_customer(
        CustomerModerationContext(session=cast(Any, None), message="anything at all")
    )
    assert check.flagged is False


class _RealModeration(ModerationService):
    async def moderate_customer(self, context: CustomerModerationContext) -> ModerationCheck:
        return ModerationCheck(flagged=False, tags=[])


async def _configure_real_moderation(configuration: ModelConfigurationService) -> None:
    async def provide(context: Any) -> _RealModeration:
        return _RealModeration()

    await configuration.configure(
        ModelScope.server(),
        {
            "moderation.customer": ModelConfig(provide),
            "moderation.agent": ModelConfig(provide),
        },
    )


def _streaming_agent(moderation: Moderation) -> Agent:
    now = datetime.now(timezone.utc)
    return Agent(
        engine="compass",
        id=AgentId("agent-1"),
        name="test-agent",
        instructions=None,
        creation_utc=now,
        modified_utc=now,
        max_steps=15,
        groups=[],
        moderation=moderation,
        message_output_mode=MessageOutputMode.STREAM,
    )


class _RecordingLogger(StdoutLogger):
    def __init__(self) -> None:
        super().__init__(LocalTracer())
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)
        super().warning(message)


async def test_that_a_streaming_agent_with_agent_side_moderation_logs_a_warning() -> None:
    router, configuration = _build_router()
    await _configure_real_moderation(configuration)
    logger = _RecordingLogger()

    await validate_moderation_configuration(router, _streaming_agent(Moderation.ON), logger)

    assert any("moderation" in w and "block" in w for w in logger.warnings)


async def test_that_a_block_mode_agent_with_moderation_logs_no_warning() -> None:
    router, configuration = _build_router()
    await _configure_real_moderation(configuration)
    logger = _RecordingLogger()

    await validate_moderation_configuration(router, _agent(Moderation.ON), logger)

    assert logger.warnings == []


async def test_that_a_streaming_agent_with_auto_moderation_and_no_routes_logs_no_warning() -> None:
    router, _ = _build_router()  # nothing routed: agent-side moderation never runs anyway
    logger = _RecordingLogger()

    await validate_moderation_configuration(router, _streaming_agent(Moderation.AUTO), logger)

    assert logger.warnings == []


async def test_that_a_streaming_agent_with_the_agent_checkpoint_waived_logs_no_warning() -> None:
    router, configuration = _build_router()
    await configuration.configure(
        ModelScope.server(),
        {"moderation.customer": ModelConfig("null"), "moderation.agent": ModelConfig("null")},
    )
    logger = _RecordingLogger()

    await validate_moderation_configuration(router, _streaming_agent(Moderation.ON), logger)

    assert logger.warnings == []
