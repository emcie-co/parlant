import os
from unittest.mock import AsyncMock, patch

from parlant.adapters.modules.parlant_cloud import WebSocketTunnelService


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
