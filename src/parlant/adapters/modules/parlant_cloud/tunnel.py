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
from contextlib import suppress
import json as _json_mod
import logging
import os
from typing import Any

import websockets

from parlant.core.app_modules.agents import AgentModule
from parlant.core.app_modules.customers import CustomerModule
from parlant.core.app_modules.sessions import SessionModule
from parlant.core.app_modules.groups import GroupModule
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import Logger
from parlant.core.tunnels import (
    TunnelRequest,
    TunnelRequestDispatcher,
    TunnelResponse,
    TunnelService,
)

from parlant.adapters.modules.parlant_cloud.config import _get_cloud_tunnel_url

_logger = logging.getLogger(__name__)

_MAX_RECONNECT_DELAY = 60.0


class ParlantCloudTunnelService(TunnelService):
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
        self._send_lock = asyncio.Lock()
        self._stream_tasks: dict[str, asyncio.Task[None]] = {}

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

        await self._cancel_stream_tasks()

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
                        message_type = message.get("type")
                        if message_type == "stream_cancel":
                            await self._cancel_stream_task(message.get("request_id"))
                            continue

                        request = TunnelRequest(
                            request_id=message["request_id"],
                            method=message["method"],
                            params=message.get("params", {}),
                        )

                        if message_type == "stream":
                            self._start_stream_task(ws, request)
                            continue

                        response = await self._dispatcher.dispatch(request)
                        await self._send_json(ws, response.to_dict())

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
                            await self._send_json(ws, error_resp.to_dict())
                        except Exception:
                            pass
            finally:
                await self._cancel_stream_tasks()
                self._websocket = None

    def _start_stream_task(self, ws: Any, request: TunnelRequest) -> None:
        task = asyncio.create_task(self._handle_stream_request(ws, request))
        self._stream_tasks[request.request_id] = task
        task.add_done_callback(lambda _: self._stream_tasks.pop(request.request_id, None))

    async def _handle_stream_request(self, ws: Any, request: TunnelRequest) -> None:
        try:
            async for event in self._dispatcher.dispatch_stream(request):
                await self._send_json(
                    ws,
                    {
                        "request_id": request.request_id,
                        "type": "stream_data",
                        "event": "message",
                        "data": event,
                    },
                )

            await self._send_json(
                ws,
                {
                    "request_id": request.request_id,
                    "type": "stream_end",
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _logger.error(f"Error processing tunnel stream: {e}")
            with suppress(Exception):
                await self._send_json(
                    ws,
                    {
                        "request_id": request.request_id,
                        "type": "stream_error",
                        "error": str(e),
                    },
                )

    async def _cancel_stream_task(self, request_id: Any) -> None:
        if not isinstance(request_id, str):
            return

        task = self._stream_tasks.pop(request_id, None)
        if task is None:
            return

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _cancel_stream_tasks(self) -> None:
        tasks = list(self._stream_tasks.values())
        self._stream_tasks.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    async def _send_json(self, ws: Any, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await ws.send(_json_mod.dumps(payload))


def _create_tunnel_service(
    session_module: SessionModule,
    agent_module: AgentModule,
    customer_module: CustomerModule,
    group_module: GroupModule,
    background_task_service: BackgroundTaskService,
    logger: Logger | None = None,
) -> ParlantCloudTunnelService | None:
    """Create a tunnel service if PARLANT_CLOUD_PROJECT_TOKEN is set."""
    token = os.environ.get("PARLANT_CLOUD_PROJECT_TOKEN", "")
    if not token:
        return None

    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        agent_module=agent_module,
        customer_module=customer_module,
        group_module=group_module,
        logger=logger,
    )

    return ParlantCloudTunnelService(
        url=_get_cloud_tunnel_url(),
        token=token,
        dispatcher=dispatcher,
    )
