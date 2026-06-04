# Copyright 2026 Parlant (Emcie Co Ltd.)
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

"""WebSocket tunnel between Parlant Cloud and a local Parlant Server."""

import asyncio
import json as _json_mod
import logging
import os
from typing import Any

import websockets

from parlant.core.app_modules.agents import AgentModule
from parlant.core.app_modules.customers import CustomerModule
from parlant.core.app_modules.sessions import SessionModule
from parlant.core.app_modules.tags import TagModule
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import Logger
from parlant.core.tunnels import (
    TunnelRequest,
    TunnelRequestDispatcher,
    TunnelResponse,
    TunnelService,
)

from .config import _get_cloud_base_url

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
        self._stop_event: asyncio.Event | None = None
        self._websocket: Any | None = None

    async def start(self) -> None:
        if not self._token:
            raise ValueError(
                "PARLANT_CLOUD_PROJECT_TOKEN is required to start the tunnel. "
                "Set it in your environment to connect to Parlant Cloud."
            )

        self._running = True
        self._stop_event = asyncio.Event()
        reconnect_delay = self._initial_reconnect_delay

        while self._running:
            try:
                await self._connect_and_listen()
                reconnect_delay = self._initial_reconnect_delay
            except asyncio.CancelledError:
                await self.stop()
                return
            except Exception as e:
                if not self._running:
                    return
                _logger.warning(
                    f"Tunnel connection failed: {e}. Reconnecting in {reconnect_delay:.1f}s..."
                )
                await self._wait_for_reconnect_or_stop(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_DELAY)

    async def stop(self) -> None:
        self._running = False
        if self._stop_event:
            self._stop_event.set()

        if self._websocket is not None:
            await self._websocket.close()

    async def _wait_for_reconnect_or_stop(self, delay: float) -> None:
        if not self._stop_event:
            await asyncio.sleep(delay)
            return

        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except TimeoutError:
            pass

    async def _connect_and_listen(self) -> None:
        headers = {"Authorization": f"Bearer {self._token}"}

        async with websockets.connect(self._url, additional_headers=headers) as ws:
            self._websocket = ws
            _logger.info(f"Tunnel connected to {self._url}")

            try:
                async for raw_message in ws:
                    if not self._running:
                        break

                    try:
                        message: dict[str, Any] = _json_mod.loads(raw_message)
                        request = TunnelRequest(
                            request_id=message["request_id"],
                            method=message["method"],
                            params=message.get("params", {}),
                        )

                        response = await self._dispatcher.dispatch(request)
                        await ws.send(_json_mod.dumps(response.to_dict()))

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
                            await ws.send(_json_mod.dumps(error_resp.to_dict()))
                        except Exception:
                            pass
            finally:
                self._websocket = None


def _create_tunnel_service(
    session_module: SessionModule,
    agent_module: AgentModule,
    customer_module: CustomerModule,
    tag_module: TagModule,
    background_task_service: BackgroundTaskService,
    logger: Logger | None = None,
) -> WebSocketTunnelService | None:
    """Create a tunnel service if PARLANT_CLOUD_PROJECT_TOKEN is set."""
    token = os.environ.get("PARLANT_CLOUD_PROJECT_TOKEN", "")
    if not token:
        return None

    base_url = _get_cloud_base_url()
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://") + "/cloud"

    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        agent_module=agent_module,
        customer_module=customer_module,
        tag_module=tag_module,
        logger=logger,
    )

    return WebSocketTunnelService(
        url=ws_url,
        token=token,
        dispatcher=dispatcher,
    )
