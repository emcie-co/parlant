import os
from unittest.mock import AsyncMock, patch

from parlant.adapters.modules.parlant_cloud import WebSocketTunnelService


async def test_that_tunnel_is_not_created_without_api_key() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PARLANT_CLOUD_API_KEY", None)

        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
            background_task_service=AsyncMock(),
        )

        assert result is None


async def test_that_tunnel_is_created_with_api_key() -> None:
    with patch.dict(os.environ, {"PARLANT_CLOUD_API_KEY": "test-token"}):
        from parlant.adapters.modules.parlant_cloud import _create_tunnel_service

        result = _create_tunnel_service(
            session_module=AsyncMock(),
            background_task_service=AsyncMock(),
        )

        assert result is not None
        assert isinstance(result, WebSocketTunnelService)


async def test_that_tunnel_uses_cloud_api_url_without_inference_suffix() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_CLOUD_API_KEY": "test-token",
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
        assert result._url == "wss://api.emcie.xyz/cloud"
