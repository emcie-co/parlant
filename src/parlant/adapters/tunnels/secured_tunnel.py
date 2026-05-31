from typing_extensions import override

from parlant.core.tunnels import TunnelService


class SecuredTunnelService(TunnelService):
    """Decorator that wraps a TunnelService with token validation."""

    def __init__(self, inner: TunnelService, token: str) -> None:
        self._inner = inner
        self._token = token

    @override
    async def start(self) -> None:
        if not self._token:
            raise ValueError(
                "PARLANT_CLOUD_PROJECT_TOKEN is required to start the tunnel. "
                "Set it in your environment to connect to Parlant Cloud."
            )
        await self._inner.start()

    @override
    async def stop(self) -> None:
        await self._inner.stop()
