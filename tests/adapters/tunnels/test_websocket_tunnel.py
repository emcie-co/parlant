from collections.abc import Generator
import asyncio
import json
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from typing_extensions import override
import websockets.asyncio.server

from parlant.adapters.modules.parlant_dashboard import ParlantDashboardTunnelService
from parlant.core.loggers import StdoutLogger
from parlant.core.tracer import AttributeValue, CompositeTracer, LocalTracer
from parlant.core.tunnels import TunnelRequest, TunnelRequestDispatcher
from parlant.core.tunnels import TunnelResponse, TunnelService


async def test_that_tunnel_service_interface_exists() -> None:
    assert hasattr(TunnelService, "start")
    assert hasattr(TunnelService, "stop")


class _FailingSpanTracer(LocalTracer):
    def __init__(
        self,
        *,
        enter_error: BaseException | None = None,
        exit_error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self._enter_error = enter_error
        self._exit_error = exit_error
        self.enter_trace_ids: list[str] = []

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]:
        self.enter_trace_ids.append(self.trace_id)
        if self._enter_error is not None:
            raise self._enter_error

        with super().span(span_id, attributes):
            try:
                yield
            finally:
                if self._exit_error is not None:
                    raise self._exit_error


class _ExitObservingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.composite: CompositeTracer | None = None
        self.trace_context_on_exit: tuple[str, str] | None = None

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]:
        try:
            yield
        finally:
            assert self.composite is not None
            self.trace_context_on_exit = (
                self.composite._shared_trace_id.get(),
                self._trace_id.get(),
            )


class _SetOnlyTraceContext:
    def __init__(self) -> None:
        self.set_calls = 0

    def set(self, value: str) -> object:
        self.set_calls += 1
        return object()


class _NonContextVarTraceTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.private_trace_context = _SetOnlyTraceContext()
        self._trace_id = cast(Any, self.private_trace_context)
        self.entered = False

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]:
        self.entered = True
        yield


def _create_event_request(request_id: str = "request-1") -> TunnelRequest:
    return TunnelRequest(
        request_id=request_id,
        method="sessions.create_event",
        params={"session_id": "session-1"},
    )


async def test_that_tunnel_create_event_context_reaches_task_after_dispatch_returns() -> None:
    tracer = LocalTracer()
    task_created = asyncio.Event()
    continue_task = asyncio.Event()
    background_attributes: dict[str, Any] = {}
    background_task: asyncio.Task[None] | None = None

    async def process_in_background() -> None:
        task_created.set()
        await continue_task.wait()
        with tracer.span("engine.process"):
            background_attributes.update(
                {
                    "span_id": tracer.span_id,
                    "rpc.system": tracer.get_attribute("rpc.system"),
                    "rpc.method": tracer.get_attribute("rpc.method"),
                    "request_id": tracer.get_attribute("parlant.tunnel.request_id"),
                }
            )
            tracer.add_event("loop.reasoning.started")

    async def dispatch(request: TunnelRequest) -> TunnelResponse:
        nonlocal background_task
        assert tracer.get_attribute("rpc.system") == "parlant.tunnel"
        assert tracer.get_attribute("rpc.method") == "sessions.create_event"
        assert tracer.get_attribute("parlant.tunnel.request_id") == "request-1"
        background_task = asyncio.create_task(process_in_background())
        await task_created.wait()
        return TunnelResponse(request_id=request.request_id, result={"event_id": "event-1"})

    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(side_effect=dispatch)
    tunnel = ParlantDashboardTunnelService(
        url="ws://localhost:2500/cloud",
        token="test-token",
        dispatcher=dispatcher,
        tracer=tracer,
        logger=StdoutLogger(LocalTracer()),
    )

    try:
        response = await asyncio.wait_for(
            tunnel._dispatch_request(_create_event_request()),
            timeout=1.0,
        )

        assert response.result == {"event_id": "event-1"}
        assert tracer.get_attribute("rpc.system") is None
        assert tracer.get_attribute("rpc.method") is None
        assert tracer.get_attribute("parlant.tunnel.request_id") is None
        assert background_task is not None
        assert not background_task.done()

        continue_task.set()
        await asyncio.wait_for(background_task, timeout=1.0)

        assert background_attributes == {
            "span_id": "tunnel.request::engine.process",
            "rpc.system": "parlant.tunnel",
            "rpc.method": "sessions.create_event",
            "request_id": "request-1",
        }
    finally:
        continue_task.set()
        if background_task is not None and not background_task.done():
            background_task.cancel()
            with suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(background_task, timeout=1.0)


@pytest.mark.parametrize("enter_error_type", [RuntimeError, asyncio.CancelledError])
async def test_that_composite_entry_failure_restores_context_before_tunnel_dispatch(
    enter_error_type: type[BaseException],
) -> None:
    response = TunnelResponse(request_id="request-1", result={"event_id": "event-1"})
    healthy_tracer = LocalTracer()
    failing_tracer = _FailingSpanTracer(enter_error=enter_error_type("enter failed"))
    tracer = CompositeTracer([healthy_tracer, failing_tracer])
    dispatch_contexts: list[tuple[str, AttributeValue | None]] = []

    async def dispatch(request: TunnelRequest) -> TunnelResponse:
        dispatch_contexts.append(
            (
                tracer.trace_id,
                tracer.get_attribute("parlant.tunnel.request_id"),
            )
        )
        return TunnelResponse(request_id=request.request_id, result=response.result)

    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(side_effect=dispatch)
    tunnel = ParlantDashboardTunnelService(
        "ws://localhost:2500/cloud",
        "test-token",
        dispatcher,
        tracer=tracer,
        logger=StdoutLogger(LocalTracer()),
    )

    first_result = await tunnel._dispatch_request(_create_event_request())

    assert first_result.request_id == response.request_id
    assert first_result.result == response.result
    dispatcher.dispatch.assert_awaited_once()
    (failed_trace_id,) = failing_tracer.enter_trace_ids
    assert failed_trace_id != "<main>"
    assert dispatch_contexts == [("<main>", None)]
    for current_tracer in (tracer, healthy_tracer, failing_tracer):
        assert current_tracer.trace_id == "<main>"
        assert current_tracer.span_id == "<main>"
        assert current_tracer.get_attribute("parlant.tunnel.request_id") is None

    failing_tracer._enter_error = None
    second_result = await tunnel._dispatch_request(_create_event_request("request-2"))

    assert second_result.request_id == "request-2"
    assert dispatcher.dispatch.await_count == 2
    assert len(failing_tracer.enter_trace_ids) == 2
    healthy_trace_id = failing_tracer.enter_trace_ids[1]
    assert healthy_trace_id != failed_trace_id
    assert dispatch_contexts[1] == (healthy_trace_id, "request-2")
    for current_tracer in (tracer, healthy_tracer, failing_tracer):
        assert current_tracer.trace_id == "<main>"
        assert current_tracer.span_id == "<main>"
        assert current_tracer.get_attribute("parlant.tunnel.request_id") is None


def test_that_composite_restores_context_after_body_error() -> None:
    child_tracers = [LocalTracer(), LocalTracer()]
    tracer = CompositeTracer(child_tracers)
    body_error = ValueError("body failed")

    with pytest.raises(ValueError) as exc_info:
        with tracer.span("test.span", {"request_id": "request-1"}):
            raise body_error

    assert exc_info.value is body_error
    for current_tracer in (tracer, *child_tracers):
        assert current_tracer.trace_id == "<main>"
        assert current_tracer.span_id == "<main>"
        assert current_tracer.get_attribute("request_id") is None


@pytest.mark.parametrize("exit_error_type", [RuntimeError, asyncio.CancelledError])
def test_that_composite_restores_context_after_child_exit_error(
    exit_error_type: type[BaseException],
) -> None:
    healthy_tracer = LocalTracer()
    exit_error = exit_error_type("exit failed")
    failing_tracer = _FailingSpanTracer(exit_error=exit_error)
    tracer = CompositeTracer([healthy_tracer, failing_tracer])

    with pytest.raises(exit_error_type) as exc_info:
        with tracer.span("test.span", {"request_id": "request-1"}):
            pass

    assert exc_info.value is exit_error
    for current_tracer in (tracer, healthy_tracer, failing_tracer):
        assert current_tracer.trace_id == "<main>"
        assert current_tracer.span_id == "<main>"
        assert current_tracer.get_attribute("request_id") is None


def test_that_composite_child_exit_precedes_trace_context_reset() -> None:
    child_tracer = _ExitObservingTracer()
    tracer = CompositeTracer([child_tracer])
    child_tracer.composite = tracer

    with tracer.span("test.span"):
        trace_id = tracer.trace_id

    assert child_tracer.trace_context_on_exit == (trace_id, trace_id)
    assert tracer.trace_id == "<main>"
    assert child_tracer.trace_id == "<main>"


def test_that_composite_ignores_non_context_var_private_trace_attribute() -> None:
    child_tracer = _NonContextVarTraceTracer()
    tracer = CompositeTracer([child_tracer])

    with tracer.span("test.span"):
        assert tracer.trace_id != "<main>"

    assert child_tracer.entered
    assert child_tracer.private_trace_context.set_calls == 0
    assert tracer._shared_trace_id.get() == ""


async def test_that_tunnel_dispatches_once_when_tracer_span_enter_fails() -> None:
    response = TunnelResponse(request_id="request-1", result={"event_id": "event-1"})
    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(return_value=response)
    tunnel = ParlantDashboardTunnelService(
        "ws://localhost:2500/cloud",
        "test-token",
        dispatcher,
        tracer=_FailingSpanTracer(enter_error=RuntimeError("enter failed")),
        logger=StdoutLogger(LocalTracer()),
    )

    result = await tunnel._dispatch_request(_create_event_request())

    assert result is response
    dispatcher.dispatch.assert_awaited_once()


async def test_that_tunnel_dispatches_once_when_tracer_span_enter_is_cancelled() -> None:
    response = TunnelResponse(request_id="request-1", result={"event_id": "event-1"})
    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(return_value=response)
    tunnel = ParlantDashboardTunnelService(
        "ws://localhost:2500/cloud",
        "test-token",
        dispatcher,
        tracer=_FailingSpanTracer(enter_error=asyncio.CancelledError()),
        logger=StdoutLogger(LocalTracer()),
    )

    result = await tunnel._dispatch_request(_create_event_request())

    assert result is response
    dispatcher.dispatch.assert_awaited_once()


async def test_that_tunnel_returns_response_when_tracer_span_exit_fails() -> None:
    response = TunnelResponse(request_id="request-1", result={"event_id": "event-1"})
    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(return_value=response)
    tunnel = ParlantDashboardTunnelService(
        "ws://localhost:2500/cloud",
        "test-token",
        dispatcher,
        tracer=_FailingSpanTracer(exit_error=RuntimeError("exit failed")),
        logger=StdoutLogger(LocalTracer()),
    )

    result = await tunnel._dispatch_request(_create_event_request())

    assert result is response
    dispatcher.dispatch.assert_awaited_once()


async def test_that_tunnel_returns_response_when_tracer_span_exit_is_cancelled() -> None:
    response = TunnelResponse(request_id="request-1", result={"event_id": "event-1"})
    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(return_value=response)
    tunnel = ParlantDashboardTunnelService(
        "ws://localhost:2500/cloud",
        "test-token",
        dispatcher,
        tracer=_FailingSpanTracer(exit_error=asyncio.CancelledError()),
        logger=StdoutLogger(LocalTracer()),
    )

    result = await tunnel._dispatch_request(_create_event_request())

    assert result is response
    dispatcher.dispatch.assert_awaited_once()


async def test_that_dispatcher_error_survives_tracer_exit_failure_without_redispatch() -> None:
    dispatcher_error = ValueError("dispatch failed")
    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(side_effect=dispatcher_error)
    tunnel = ParlantDashboardTunnelService(
        "ws://localhost:2500/cloud",
        "test-token",
        dispatcher,
        tracer=_FailingSpanTracer(exit_error=RuntimeError("exit failed")),
        logger=StdoutLogger(LocalTracer()),
    )

    with pytest.raises(ValueError) as exc_info:
        await tunnel._dispatch_request(_create_event_request())

    assert exc_info.value is dispatcher_error
    dispatcher.dispatch.assert_awaited_once()


async def test_that_dispatcher_error_survives_cancelled_tracer_exit_without_redispatch() -> None:
    dispatcher_error = ValueError("dispatch failed")
    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(side_effect=dispatcher_error)
    tunnel = ParlantDashboardTunnelService(
        "ws://localhost:2500/cloud",
        "test-token",
        dispatcher,
        tracer=_FailingSpanTracer(exit_error=asyncio.CancelledError()),
        logger=StdoutLogger(LocalTracer()),
    )

    with pytest.raises(ValueError) as exc_info:
        await tunnel._dispatch_request(_create_event_request())

    assert exc_info.value is dispatcher_error
    dispatcher.dispatch.assert_awaited_once()


async def test_that_dispatcher_cancellation_survives_cancelled_tracer_exit() -> None:
    dispatcher_error = asyncio.CancelledError()
    dispatcher = AsyncMock(spec=TunnelRequestDispatcher)
    dispatcher.dispatch = AsyncMock(side_effect=dispatcher_error)
    tunnel = ParlantDashboardTunnelService(
        "ws://localhost:2500/cloud",
        "test-token",
        dispatcher,
        tracer=_FailingSpanTracer(exit_error=asyncio.CancelledError()),
        logger=StdoutLogger(LocalTracer()),
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await tunnel._dispatch_request(_create_event_request())

    assert exc_info.value is dispatcher_error
    dispatcher.dispatch.assert_awaited_once()


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

        tunnel = ParlantDashboardTunnelService(
            url=url,
            token="test-token",
            dispatcher=dispatcher,
            logger=StdoutLogger(LocalTracer()),
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


async def test_that_tunnel_streams_dispatcher_events_as_stream_frames() -> None:
    received_frames: list[dict[str, Any]] = []

    async def mock_platform_handler(websocket: websockets.asyncio.server.ServerConnection) -> None:
        await websocket.send(
            json.dumps(
                {
                    "request_id": "stream-1",
                    "type": "stream",
                    "method": "sessions.stream_events",
                    "params": {"session_id": "sess-1"},
                }
            )
        )

        received_frames.append(json.loads(await websocket.recv()))
        received_frames.append(json.loads(await websocket.recv()))
        await websocket.wait_closed()

    class StreamingDispatcher:
        async def dispatch_stream(self, request: Any) -> Any:
            assert request.request_id == "stream-1"
            assert request.method == "sessions.stream_events"
            yield {"id": "evt-1", "offset": 0}

    async with websockets.asyncio.server.serve(mock_platform_handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://localhost:{port}"

        tunnel = ParlantDashboardTunnelService(
            url=url,
            token="test-token",
            dispatcher=StreamingDispatcher(),  # type: ignore[arg-type]
            logger=StdoutLogger(LocalTracer()),
            initial_reconnect_delay=10.0,
        )

        task = asyncio.create_task(tunnel.start())

        for _ in range(50):
            if len(received_frames) >= 2:
                break
            await asyncio.sleep(0.1)

        await tunnel.stop()
        await asyncio.wait_for(task, timeout=1.0)

    assert received_frames == [
        {
            "request_id": "stream-1",
            "type": "stream_data",
            "event": "message",
            "data": {"id": "evt-1", "offset": 0},
        },
        {
            "request_id": "stream-1",
            "type": "stream_end",
        },
    ]


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

        tunnel = ParlantDashboardTunnelService(
            url=url,
            token="test-token",
            dispatcher=dispatcher,
            logger=StdoutLogger(LocalTracer()),
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

        tunnel = ParlantDashboardTunnelService(
            url=url,
            token="test-token",
            dispatcher=dispatcher,
            logger=StdoutLogger(LocalTracer()),
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

        tunnel = ParlantDashboardTunnelService(
            url=url,
            token="test-token",
            dispatcher=dispatcher,
            logger=StdoutLogger(LocalTracer()),
            initial_reconnect_delay=10.0,
        )

        task = asyncio.create_task(tunnel.start())

        for _ in range(50):
            if connection_count >= 1:
                break
            await asyncio.sleep(0.1)

        await tunnel.stop()
        await asyncio.wait_for(task, timeout=1.0)
