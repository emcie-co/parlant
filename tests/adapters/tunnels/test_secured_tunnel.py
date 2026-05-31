import pytest
from unittest.mock import AsyncMock

from parlant.adapters.tunnels.secured_tunnel import SecuredTunnelService
from parlant.core.tunnels import TunnelService


async def test_that_secured_tunnel_raises_if_token_is_empty() -> None:
    inner = AsyncMock(spec=TunnelService)
    tunnel = SecuredTunnelService(inner=inner, token="")
    with pytest.raises(ValueError, match="PARLANT_CLOUD_PROJECT_TOKEN"):
        await tunnel.start()
    inner.start.assert_not_awaited()


async def test_that_secured_tunnel_delegates_to_inner_when_token_is_valid() -> None:
    inner = AsyncMock(spec=TunnelService)
    tunnel = SecuredTunnelService(inner=inner, token="valid-token-123")
    await tunnel.start()
    inner.start.assert_awaited_once()


async def test_that_secured_tunnel_stop_delegates_to_inner() -> None:
    inner = AsyncMock(spec=TunnelService)
    tunnel = SecuredTunnelService(inner=inner, token="valid-token-123")
    await tunnel.stop()
    inner.stop.assert_awaited_once()
