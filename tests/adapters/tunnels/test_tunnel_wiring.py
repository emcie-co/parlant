import os
from typing import Any, Coroutine, cast
from unittest.mock import AsyncMock, patch

from lagom import Container

from parlant.adapters.modules.parlant_cloud import WebSocketTunnelService, initialize_container
from parlant.core.app_modules.sessions import SessionModule
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import Logger
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
            background_task_service=AsyncMock(),
        )

        assert result is None


async def test_that_tunnel_is_created_with_project_token() -> None:
    with patch.dict(os.environ, {"PARLANT_CLOUD_PROJECT_TOKEN": "test-token"}):
        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
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
            background_task_service=AsyncMock(),
        )

        assert result is not None
        assert isinstance(result, WebSocketTunnelService)
        assert result._url == "wss://api.emcie.xyz/cloud"


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
            background_task_service=AsyncMock(),
        )

        assert result is not None
        assert isinstance(result, WebSocketTunnelService)
        assert result._url == "wss://api.parlant.cloud/cloud"


async def test_that_cloud_initializer_starts_tunnel_after_session_module_exists() -> None:
    with patch.dict(os.environ, {"PARLANT_CLOUD_PROJECT_TOKEN": "test-token"}):
        container = Container()
        background_task_service = FakeBackgroundTaskService()

        container[SessionModule] = cast(SessionModule, object())
        container[BackgroundTaskService] = cast(BackgroundTaskService, background_task_service)
        container[Logger] = cast(Logger, FakeLogger())

        await initialize_container(container)

        assert background_task_service.started
        assert background_task_service.tag == "parlant-cloud-tunnel"
