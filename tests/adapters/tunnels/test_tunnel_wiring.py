import os
from typing import Any, Coroutine, cast
from unittest.mock import AsyncMock, patch

from lagom import Container

import parlant.adapters.modules.parlant_cloud.lifecycle as lifecycle
from parlant.adapters.modules.parlant_cloud import WebSocketTunnelService, initialize_container
from parlant.adapters.modules.parlant_cloud.lifecycle import CloudProjectAuth
from parlant.adapters.modules.parlant_cloud.logger import ParlantCloudLogger
from parlant.adapters.modules.parlant_cloud.meter import ParlantCloudMeter
from parlant.adapters.modules.parlant_cloud.tracer import ParlantCloudTracer
from parlant.core.app_modules.agents import AgentModule
from parlant.core.app_modules.customers import CustomerModule
from parlant.core.app_modules.sessions import SessionModule
from parlant.core.app_modules.tags import TagModule
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import Logger
from parlant.core.tracer import LocalTracer
from parlant.sdk import _should_configure_parlant_cloud


class FakeBackgroundTaskService:
    def __init__(self) -> None:
        self.started = False
        self.tag: str | None = None

    async def start(self, f: Coroutine[Any, Any, None], /, *, tag: str) -> None:
        self.started = True
        self.tag = tag
        f.close()


class FakeLogger:
    def warning(self, message: str) -> None:
        pass


async def test_that_sdk_cloud_module_is_not_loaded_without_cloud_credentials() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PARLANT_CLOUD_API_KEY", None)
        os.environ.pop("PARLANT_CLOUD_PROJECT_TOKEN", None)

        assert _should_configure_parlant_cloud() is False


async def test_that_sdk_cloud_module_is_loaded_with_cloud_api_key() -> None:
    with patch.dict(os.environ, {"PARLANT_CLOUD_API_KEY": "test-api-key"}):
        os.environ.pop("PARLANT_CLOUD_PROJECT_TOKEN", None)

        assert _should_configure_parlant_cloud() is True


async def test_that_sdk_cloud_module_is_loaded_with_project_token() -> None:
    with patch.dict(os.environ, {"PARLANT_CLOUD_PROJECT_TOKEN": "test-token"}):
        os.environ.pop("PARLANT_CLOUD_API_KEY", None)

        assert _should_configure_parlant_cloud() is True


async def test_that_tunnel_is_not_created_without_project_token() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PARLANT_CLOUD_PROJECT_TOKEN", None)

        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
            agent_module=AsyncMock(),
            customer_module=AsyncMock(),
            tag_module=AsyncMock(),
            background_task_service=AsyncMock(),
        )

        assert result is None


async def test_that_tunnel_is_created_with_project_token() -> None:
    with patch.dict(os.environ, {"PARLANT_CLOUD_PROJECT_TOKEN": "test-token"}):
        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
            agent_module=AsyncMock(),
            customer_module=AsyncMock(),
            tag_module=AsyncMock(),
            background_task_service=AsyncMock(),
        )

        assert result is not None
        assert isinstance(result, WebSocketTunnelService)


async def test_that_tunnel_uses_cloud_base_url() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_CLOUD_PROJECT_TOKEN": "test-token",
            "PARLANT_CLOUD_BASE_URL": "https://api.emcie.xyz",
        },
    ):
        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
            agent_module=AsyncMock(),
            customer_module=AsyncMock(),
            tag_module=AsyncMock(),
            background_task_service=AsyncMock(),
        )

        assert result is not None
        assert isinstance(result, WebSocketTunnelService)
        assert result._url == "wss://api.emcie.xyz/cloud"


async def test_that_tunnel_uses_explicit_parlant_cloud_tunnel_url() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_CLOUD_PROJECT_TOKEN": "test-token",
            "PARLANT_CLOUD_BASE_URL": "https://api.emcie.xyz",
            "PARLANT_CLOUD_TUNNEL_URL": "http://localhost:2500",
        },
    ):
        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
            agent_module=AsyncMock(),
            customer_module=AsyncMock(),
            tag_module=AsyncMock(),
            background_task_service=AsyncMock(),
        )

        assert result is not None
        assert isinstance(result, WebSocketTunnelService)
        assert result._url == "ws://localhost:2500/cloud"


async def test_that_tunnel_preserves_explicit_websocket_cloud_path() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_CLOUD_PROJECT_TOKEN": "test-token",
            "PARLANT_CLOUD_TUNNEL_URL": "ws://localhost:2500/cloud",
        },
    ):
        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
            agent_module=AsyncMock(),
            customer_module=AsyncMock(),
            tag_module=AsyncMock(),
            background_task_service=AsyncMock(),
        )

        assert result is not None
        assert isinstance(result, WebSocketTunnelService)
        assert result._url == "ws://localhost:2500/cloud"


async def test_that_tunnel_ignores_cloud_api_url() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_CLOUD_PROJECT_TOKEN": "test-token",
            "PARLANT_CLOUD_API_URL": "https://api.emcie.xyz/inference",
        },
    ):
        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
            agent_module=AsyncMock(),
            customer_module=AsyncMock(),
            tag_module=AsyncMock(),
            background_task_service=AsyncMock(),
        )

        assert result is not None
        assert isinstance(result, WebSocketTunnelService)
        assert result._url == "wss://api.parlant.cloud/cloud"


async def test_that_cloud_otel_url_configures_logs_traces_and_metrics_collectors() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_CLOUD_CLOUD_OTEL_URL": "http://localhost:4318",
        },
    ):
        tracer = ParlantCloudTracer()
        logger = ParlantCloudLogger(tracer=LocalTracer())
        meter = ParlantCloudMeter()

        assert tracer._endpoint == "http://localhost:4318/v1/traces"
        assert logger._endpoint == "http://localhost:4318/v1/logs"
        assert meter._endpoint == "http://localhost:4318/v1/metrics"


async def test_that_cloud_initializer_starts_tunnel_after_session_module_exists() -> None:
    with patch.dict(os.environ, {"PARLANT_CLOUD_PROJECT_TOKEN": "test-token"}):
        lifecycle._cloud_project_auth = CloudProjectAuth(
            project_id="project-1",
            secure_connection_enabled=True,
            authenticated=True,
        )
        container = Container()
        background_task_service = FakeBackgroundTaskService()

        container[SessionModule] = cast(SessionModule, object())
        container[AgentModule] = cast(AgentModule, object())
        container[CustomerModule] = cast(CustomerModule, object())
        container[TagModule] = cast(TagModule, object())
        container[BackgroundTaskService] = cast(BackgroundTaskService, background_task_service)
        container[Logger] = cast(Logger, FakeLogger())

        await initialize_container(container)

        assert background_task_service.started
        assert background_task_service.tag == "parlant-cloud-tunnel"

        lifecycle._cloud_project_auth = None


async def test_that_cloud_initializer_does_not_start_tunnel_without_secure_connection() -> None:
    with patch.dict(os.environ, {"PARLANT_CLOUD_PROJECT_TOKEN": "test-token"}):
        lifecycle._cloud_project_auth = CloudProjectAuth(
            project_id="project-1",
            secure_connection_enabled=False,
            authenticated=True,
        )
        container = Container()
        background_task_service = FakeBackgroundTaskService()

        container[SessionModule] = cast(SessionModule, object())
        container[AgentModule] = cast(AgentModule, object())
        container[CustomerModule] = cast(CustomerModule, object())
        container[TagModule] = cast(TagModule, object())
        container[BackgroundTaskService] = cast(BackgroundTaskService, background_task_service)
        container[Logger] = cast(Logger, FakeLogger())

        await initialize_container(container)

        assert not background_task_service.started

        lifecycle._cloud_project_auth = None
