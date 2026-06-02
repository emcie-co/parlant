import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import websockets.asyncio.server

from parlant.adapters.modules.parlant_cloud import WebSocketTunnelService
from parlant.core.tunnels import TunnelRequestDispatcher
from parlant.core.tunnels import TunnelResponse, TunnelService


async def test_that_tunnel_service_interface_exists() -> None:
    assert hasattr(TunnelService, "start")
    assert hasattr(TunnelService, "stop")


async def test_that_tunnel_connects_and_dispatches_request() -> None:
    received_response: dict[str, Any] | None = None

    async def mock_platform_handler(websocket: websockets.asyncio.server.ServerConnection) -> None:
        nonlocal received_response

        await websocket.send(
            json.dumps(
                {
                    "request_id": "req-1",
                    "method": "sessions.list_events",
                    "params": {"session_id": "sess-1"},
                }
            )
        )

        raw = await websocket.recv()
        received_response = json.loads(raw)

    async with websockets.asyncio.server.serve(mock_platform_handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://localhost:{port}"

        dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
        dispatcher.dispatch = AsyncMock(
            return_value=TunnelResponse(
                request_id="req-1",
                result={"events": []},
            )
        )

        tunnel = WebSocketTunnelService(
            url=url,
            token="test-token",
            dispatcher=dispatcher,
        )

        task = asyncio.create_task(tunnel.start())

        for _ in range(50):
            if received_response is not None:
                break
            await asyncio.sleep(0.1)

        await tunnel.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert received_response is not None
    assert received_response["request_id"] == "req-1"
    assert received_response["result"] == {"events": []}
    assert dispatcher.dispatch.await_count >= 1


async def test_that_tunnel_reconnects_after_disconnect() -> None:
    connection_count = 0

    async def mock_handler(websocket: websockets.asyncio.server.ServerConnection) -> None:
        nonlocal connection_count
        connection_count += 1
        if connection_count == 1:
            await websocket.close()
            return
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass

    async with websockets.asyncio.server.serve(mock_handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://localhost:{port}"

        dispatcher = AsyncMock(spec=TunnelRequestDispatcher)

        tunnel = WebSocketTunnelService(
            url=url,
            token="test-token",
            dispatcher=dispatcher,
            initial_reconnect_delay=0.1,
        )

        task = asyncio.create_task(tunnel.start())

        for _ in range(50):
            if connection_count >= 2:
                break
            await asyncio.sleep(0.1)

        await tunnel.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert connection_count >= 2
