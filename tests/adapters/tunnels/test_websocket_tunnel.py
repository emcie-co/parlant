import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import websockets.asyncio.server

from parlant.adapters.modules.parlant_cloud import ParlantCloudTunnelService
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

        tunnel = ParlantCloudTunnelService(
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
        await asyncio.wait_for(task, timeout=1.0)

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

        tunnel = ParlantCloudTunnelService(
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
        await asyncio.wait_for(task, timeout=1.0)

    assert connection_count >= 2


async def test_that_stopping_tunnel_closes_active_websocket() -> None:
    connected = asyncio.Event()

    async def mock_handler(websocket: websockets.asyncio.server.ServerConnection) -> None:
        connected.set()
        await websocket.wait_closed()

    async with websockets.asyncio.server.serve(mock_handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://localhost:{port}"

        dispatcher = AsyncMock(spec=TunnelRequestDispatcher)

        tunnel = ParlantCloudTunnelService(
            url=url,
            token="test-token",
            dispatcher=dispatcher,
        )

        task = asyncio.create_task(tunnel.start())
        await asyncio.wait_for(connected.wait(), timeout=1.0)

        await tunnel.stop()
        await asyncio.wait_for(task, timeout=1.0)


async def test_that_stopping_tunnel_interrupts_reconnect_delay() -> None:
    connection_count = 0

    async def mock_handler(websocket: websockets.asyncio.server.ServerConnection) -> None:
        nonlocal connection_count
        connection_count += 1
        await websocket.close()

    async with websockets.asyncio.server.serve(mock_handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://localhost:{port}"

        dispatcher = AsyncMock(spec=TunnelRequestDispatcher)

        tunnel = ParlantCloudTunnelService(
            url=url,
            token="test-token",
            dispatcher=dispatcher,
            initial_reconnect_delay=10.0,
        )

        task = asyncio.create_task(tunnel.start())

        for _ in range(50):
            if connection_count >= 1:
                break
            await asyncio.sleep(0.1)

        await tunnel.stop()
        await asyncio.wait_for(task, timeout=1.0)
