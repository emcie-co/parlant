import asyncio
import json
import logging
from typing import Any

import websockets

from parlant.adapters.tunnels.dispatcher import TunnelRequestDispatcher
from parlant.core.tunnels import TunnelRequest, TunnelResponse, TunnelService

_logger = logging.getLogger(__name__)

_MAX_RECONNECT_DELAY = 60.0


class WebSocketTunnelService(TunnelService):
    """Tunnel that connects to the platform via WebSocket."""

    def __init__(
        self,
        url: str,
        token: str,
        dispatcher: TunnelRequestDispatcher,
        initial_reconnect_delay: float = 1.0,
    ) -> None:
        self._url = url
        self._token = token
        self._dispatcher = dispatcher
        self._initial_reconnect_delay = initial_reconnect_delay
        self._running = False

    async def start(self) -> None:
        self._running = True
        reconnect_delay = self._initial_reconnect_delay

        while self._running:
            try:
                await self._connect_and_listen()
                reconnect_delay = self._initial_reconnect_delay
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._running:
                    return
                _logger.warning(
                    f"Tunnel connection failed: {e}. Reconnecting in {reconnect_delay:.1f}s..."
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_DELAY)

    async def stop(self) -> None:
        self._running = False

    async def _connect_and_listen(self) -> None:
        headers = {"Authorization": f"Bearer {self._token}"}

        async with websockets.connect(self._url, additional_headers=headers) as ws:
            _logger.info(f"Tunnel connected to {self._url}")

            async for raw_message in ws:
                if not self._running:
                    break

                try:
                    message: dict[str, Any] = json.loads(raw_message)
                    request = TunnelRequest(
                        request_id=message["request_id"],
                        method=message["method"],
                        params=message.get("params", {}),
                    )

                    response = await self._dispatcher.dispatch(request)
                    await ws.send(json.dumps(response.to_dict()))

                except Exception as e:
                    _logger.error(f"Error processing tunnel message: {e}")
                    request_id = (
                        message.get("request_id", "unknown")
                        if isinstance(message, dict)
                        else "unknown"
                    )
                    try:
                        error_resp = TunnelResponse(
                            request_id=request_id,
                            error=str(e),
                        )
                        await ws.send(json.dumps(error_resp.to_dict()))
                    except Exception:
                        pass
