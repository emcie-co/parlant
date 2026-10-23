import asyncio
from contextlib import suppress
from collections.abc import AsyncIterator
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
import websockets.asyncio.server

from parlant.adapters.modules.parlant_dashboard.tunnel import ParlantDashboardTunnelService
from parlant.core.loggers import StdoutLogger
from parlant.core.tracer import LocalTracer
from parlant.core.tunnels import TunnelRequest, TunnelRequestDispatcher, TunnelResponse


def create_service(port: int, dispatch: Any) -> ParlantDashboardTunnelService:
    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch.side_effect = dispatch
    service = ParlantDashboardTunnelService(
        url=f"ws://127.0.0.1:{port}",
        token="test-token",
        dispatcher=dispatcher,
        logger=StdoutLogger(LocalTracer()),
    )
    service._running = True
    return service


async def test_fifty_queued_unary_requests_leave_heartbeats_responsive_and_all_complete() -> None:
    release = asyncio.Event()
    started = asyncio.Event()
    peak = 0
    active = 0
    completed: set[str] = set()
    result = asyncio.get_running_loop().create_future()

    async def dispatch(request: TunnelRequest) -> TunnelResponse:
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        started.set()
        try:
            await release.wait()
            await asyncio.sleep(0)
            completed.add(request.request_id)
            return TunnelResponse(request_id=request.request_id, result={})
        finally:
            active -= 1

    async def platform(ws: websockets.asyncio.server.ServerConnection) -> None:
        try:
            for index in range(50):
                await ws.send(
                    json.dumps(
                        {"request_id": str(index), "method": "sessions.create", "params": {}}
                    )
                )
            await started.wait()
            # Heartbeats must be received even when the request queue is full.
            pong = await ws.ping()
            await asyncio.wait_for(pong, timeout=0.5)
            release.set()
            responses = [json.loads(await asyncio.wait_for(ws.recv(), 2)) for _ in range(50)]
            result.set_result(responses)
        except BaseException as error:
            if not result.done():
                result.set_exception(error)
        finally:
            release.set()

    async with websockets.asyncio.server.serve(
        platform, "127.0.0.1", 0, close_timeout=0.1
    ) as server:
        service = create_service(server.sockets[0].getsockname()[1], dispatch)
        task = asyncio.create_task(service._connect_and_listen())
        try:
            responses = await asyncio.wait_for(result, 5)
            assert {response["request_id"] for response in responses} == {
                str(index) for index in range(50)
            }
            assert completed == {str(index) for index in range(50)}
            assert 1 < peak <= 16
        finally:
            release.set()
            await service.stop()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def test_same_session_requests_remain_ordered_without_blocking_other_sessions() -> None:
    release = asyncio.Event()
    entered: list[str] = []
    result = asyncio.get_running_loop().create_future()

    async def dispatch(request: TunnelRequest) -> TunnelResponse:
        entered.append(request.request_id)
        if request.request_id == "a-first":
            await release.wait()
        return TunnelResponse(request_id=request.request_id, result={})

    async def platform(ws: websockets.asyncio.server.ServerConnection) -> None:
        try:
            for name, session in [("a-first", "a"), ("a-second", "a"), ("b-first", "b")]:
                await ws.send(
                    json.dumps(
                        {
                            "request_id": name,
                            "method": "sessions.create_event",
                            "params": {"session_id": session},
                        }
                    )
                )
            first = json.loads(await asyncio.wait_for(ws.recv(), 0.5))
            assert first["request_id"] == "b-first"
            assert "a-second" not in entered
            release.set()
            rest = [
                json.loads(await asyncio.wait_for(ws.recv(), 1))["request_id"] for _ in range(2)
            ]
            assert rest == ["a-first", "a-second"]
            result.set_result(None)
        except BaseException as error:
            if not result.done():
                result.set_exception(error)
        finally:
            release.set()

    async with websockets.asyncio.server.serve(
        platform, "127.0.0.1", 0, close_timeout=0.1
    ) as server:
        service = create_service(server.sockets[0].getsockname()[1], dispatch)
        task = asyncio.create_task(service._connect_and_listen())
        try:
            await asyncio.wait_for(result, 3)
        finally:
            release.set()
            await service.stop()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("teardown", ["disconnect", "stop"])
async def test_connection_teardown_cancels_active_and_queued_unary_requests(teardown: str) -> None:
    started: set[str] = set()
    cancelled: set[str] = set()
    all_sent = asyncio.Event()
    first_started = asyncio.Event()
    close = asyncio.Event()

    async def dispatch(request: TunnelRequest) -> TunnelResponse:
        started.add(request.request_id)
        first_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(request.request_id)
        raise AssertionError("blocked handler unexpectedly completed")

    async def platform(ws: websockets.asyncio.server.ServerConnection) -> None:
        for index in range(50):
            await ws.send(
                json.dumps({"request_id": str(index), "method": "sessions.create", "params": {}})
            )
        all_sent.set()
        await close.wait()
        await ws.close()

    async with websockets.asyncio.server.serve(
        platform, "127.0.0.1", 0, close_timeout=0.1
    ) as server:
        service = create_service(server.sockets[0].getsockname()[1], dispatch)
        task = asyncio.create_task(service._connect_and_listen())
        try:
            await asyncio.wait_for(all_sent.wait(), 1)
            await asyncio.wait_for(first_started.wait(), 1)
            if teardown == "stop":
                await asyncio.wait_for(service.stop(), 1)
                close.set()
            else:
                close.set()
            await asyncio.wait_for(task, 1)
            assert started
            assert cancelled == started
            assert not service._unary_tasks
            assert not service._session_requests
        finally:
            close.set()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await service.stop()


async def test_excess_pending_requests_close_connection_and_cancel_admitted_work() -> None:
    result = asyncio.get_running_loop().create_future()
    started = 0
    cancelled = 0

    async def dispatch(request: TunnelRequest) -> TunnelResponse:
        nonlocal started, cancelled
        started += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancelled += 1
        raise AssertionError("blocked handler unexpectedly completed")

    async def platform(ws: websockets.asyncio.server.ServerConnection) -> None:
        try:
            for index in range(129):
                await ws.send(
                    json.dumps(
                        {"request_id": str(index), "method": "sessions.create", "params": {}}
                    )
                )
            await asyncio.wait_for(ws.wait_closed(), 1)
            result.set_result(ws.close_code)
        except BaseException as error:
            if not result.done():
                result.set_exception(error)

    async with websockets.asyncio.server.serve(
        platform, "127.0.0.1", 0, close_timeout=0.1
    ) as server:
        service = create_service(server.sockets[0].getsockname()[1], dispatch)
        task = asyncio.create_task(service._connect_and_listen())
        try:
            assert await asyncio.wait_for(result, 3) == 1013
            await asyncio.wait_for(task, 1)
            assert started <= 16
            assert cancelled == started
            assert not service._unary_tasks
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await service.stop()


async def test_stream_and_stream_cancellation_continue_while_unary_request_waits() -> None:
    release = asyncio.Event()
    stream_cancelled = asyncio.Event()
    result = asyncio.get_running_loop().create_future()

    async def dispatch(request: TunnelRequest) -> TunnelResponse:
        await release.wait()
        return TunnelResponse(request_id=request.request_id, result={})

    async def stream(request: TunnelRequest) -> AsyncIterator[dict[str, str]]:
        try:
            yield {"message": "stream stays responsive"}
            await asyncio.Event().wait()
        finally:
            stream_cancelled.set()

    async def platform(ws: websockets.asyncio.server.ServerConnection) -> None:
        try:
            await ws.send(
                json.dumps(
                    {
                        "request_id": "unary",
                        "method": "sessions.read",
                        "params": {"session_id": "a"},
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "request_id": "stream",
                        "type": "stream",
                        "method": "sessions.stream_events",
                        "params": {"session_id": "a"},
                    }
                )
            )
            frame = json.loads(await asyncio.wait_for(ws.recv(), 1))
            assert frame["request_id"] == "stream"
            assert frame["type"] == "stream_data"
            await ws.send(json.dumps({"request_id": "stream", "type": "stream_cancel"}))
            await asyncio.wait_for(stream_cancelled.wait(), 1)
            release.set()
            response = json.loads(await asyncio.wait_for(ws.recv(), 1))
            assert response["request_id"] == "unary"
            result.set_result(None)
        except BaseException as error:
            if not result.done():
                result.set_exception(error)
        finally:
            release.set()

    async with websockets.asyncio.server.serve(
        platform, "127.0.0.1", 0, close_timeout=0.1
    ) as server:
        service = create_service(server.sockets[0].getsockname()[1], dispatch)
        setattr(service._dispatcher, "dispatch_stream", stream)
        task = asyncio.create_task(service._connect_and_listen())
        try:
            await asyncio.wait_for(result, 3)
        finally:
            release.set()
            await service.stop()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def test_concurrent_error_response_keeps_request_id_and_other_requests_complete() -> None:
    result = asyncio.get_running_loop().create_future()

    async def dispatch(request: TunnelRequest) -> TunnelResponse:
        await asyncio.sleep(0)
        if request.request_id == "failed":
            raise ValueError("request-specific failure")
        return TunnelResponse(request_id=request.request_id, result={"ok": True})

    async def platform(ws: websockets.asyncio.server.ServerConnection) -> None:
        try:
            for request_id in ["failed", "successful"]:
                await ws.send(
                    json.dumps({"request_id": request_id, "method": "agents.list", "params": {}})
                )
            responses = [json.loads(await asyncio.wait_for(ws.recv(), 1)) for _ in range(2)]
            by_id = {response["request_id"]: response for response in responses}
            assert by_id["failed"]["error"] == "request-specific failure"
            assert by_id["successful"]["result"] == {"ok": True}
            result.set_result(None)
        except BaseException as error:
            if not result.done():
                result.set_exception(error)

    async with websockets.asyncio.server.serve(
        platform, "127.0.0.1", 0, close_timeout=0.1
    ) as server:
        service = create_service(server.sockets[0].getsockname()[1], dispatch)
        task = asyncio.create_task(service._connect_and_listen())
        try:
            await asyncio.wait_for(result, 3)
        finally:
            await service.stop()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
