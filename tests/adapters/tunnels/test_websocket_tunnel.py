from parlant.core.tunnels import TunnelService


async def test_that_tunnel_service_interface_exists() -> None:
    assert hasattr(TunnelService, "start")
    assert hasattr(TunnelService, "stop")
