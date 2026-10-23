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

"""WebSocket tunnel between Parlant Dashboard and a local Parlant Server."""

import asyncio
from collections.abc import Mapping
from contextlib import AsyncExitStack, contextmanager, suppress
from collections.abc import Generator
import faulthandler
import json as _json_mod
import os
import signal
import time
from typing import Any

import websockets

from parlant.core.application import Application
from parlant.core.health import HealthReporter
from parlant.core.health.tunnel_view import TunnelState
from parlant.core.loggers import ScopedLogger, Logger
from parlant.core.tracer import AttributeValue, LocalTracer, Tracer
from parlant.core.tunnels import (
    TunnelRequest,
    TunnelRequestDispatcher,
    TunnelResponse,
    TunnelService,
)

from parlant.adapters.modules.parlant_dashboard.config import _get_dashboard_tunnel_url

# A disconnected tunnel is retried on a steady cadence rather than backing off
# into minute-long gaps: the dashboard is usually a moment away from being
# reachable again (a restart, a route coming up), and a minute of silence reads
# as a tunnel that is simply down.
_RECONNECT_DELAY = 5.0
_MAX_RECONNECT_DELAY = _RECONNECT_DELAY


@contextmanager
def _best_effort_tracing_span(
    tracer: Tracer,
    span_id: str,
    attributes: Mapping[str, AttributeValue],
) -> Generator[None, None, None]:
    # Telemetry must never replace the dispatch result: a tracer that fails to
    # enter, exit, or gets cancelled is dropped, and the body's own outcome
    # (result, error, or cancellation) is preserved untouched.
    span_context = None
    try:
        span_context = tracer.span(span_id, attributes)
        span_context.__enter__()
    except (Exception, asyncio.CancelledError):
        span_context = None

    if span_context is None:
        yield
        return

    try:
        yield
    except BaseException as dispatch_error:
        with suppress(Exception, asyncio.CancelledError):
            span_context.__exit__(
                type(dispatch_error),
                dispatch_error,
                dispatch_error.__traceback__,
            )
        raise
    else:
        with suppress(Exception, asyncio.CancelledError):
            span_context.__exit__(None, None, None)


# Keep the receive loop free to drain frames (including heartbeat traffic) while
# bounding admitted work and concurrent handlers on each runtime connection.
_MAX_CONCURRENT_UNARY_REQUESTS = 16
_MAX_PENDING_UNARY_REQUESTS = 128


class ParlantDashboardTunnelService(TunnelService):
    """Tunnel that connects to the platform via WebSocket."""

    def __init__(
        self,
        url: str,
        token: str,
        dispatcher: TunnelRequestDispatcher,
        logger: Logger,
        initial_reconnect_delay: float = _RECONNECT_DELAY,
        *,
        tracer: Tracer | None = None,
    ) -> None:
        self._url = url
        self._token = token
        self._dispatcher = dispatcher
        self._tracer = tracer if tracer is not None else LocalTracer()
        self._initial_reconnect_delay = initial_reconnect_delay
        self._running = False
        self._stop_event: asyncio.Event | None = None
        self._websocket: Any | None = None
        self._logger = ScopedLogger(logger, "ParlantDashboardTunnelService")
        self._send_lock = asyncio.Lock()
        self._stream_tasks: dict[str, asyncio.Task[None]] = {}
        self._unary_tasks: set[asyncio.Task[None]] = set()
        self._unary_slots = asyncio.Semaphore(_MAX_CONCURRENT_UNARY_REQUESTS)
        self._session_requests: dict[str, tuple[asyncio.Lock, int]] = {}
        # State for the `tunnel` view on /healthz. A tunnel that is down looks
        # exactly like a tunnel that is quiet -- success logs at trace, failure
        # at warning -- so the service has to be able to answer for itself.
        # Reported on demand rather than logged: the answer belongs where health
        # is already read, not in a periodic line nobody asked for.
        self._tunnel_state = "starting"
        self._last_error: str | None = None
        self._reconnect_attempts = 0
        self._connected_since: float | None = None
        self._last_connected_at: float | None = None

    async def start(self) -> None:
        if not self._token:
            raise ValueError(
                "PARLANT_DASHBOARD_PROJECT_TOKEN is required to start the tunnel. "
                "Set it in your environment to connect to Parlant Dashboard."
            )

        self._running = True
        self._stop_event = asyncio.Event()
        reconnect_delay = self._initial_reconnect_delay

        self._install_stack_dump_handler()

        while self._running:
            try:
                self._tunnel_state = "connecting"
                await self._connect_and_listen()
                # A clean return means the socket closed without raising, which
                # is a different story from a failed connect and worth naming.
                self._tunnel_state = "disconnected"
                self._connected_since = None
                reconnect_delay = self._initial_reconnect_delay
            except asyncio.CancelledError:
                self._tunnel_state = "cancelled"
                await self.stop()
                return
            except Exception as e:
                if not self._running:
                    return
                self._tunnel_state = "reconnecting"
                self._last_error = f"{type(e).__name__}: {e}"
                self._reconnect_attempts += 1
                self._connected_since = None
                self._logger.warning(
                    f"Tunnel connection failed: {e}. Reconnecting in {reconnect_delay:.1f}s..."
                )
                await self._wait_for_reconnect_or_stop(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_DELAY)

        self._tunnel_state = "stopped"

    def tunnel_state(self) -> TunnelState:
        """Answer for the tunnel's own condition, for the `tunnel` health view.

        This used to be a log line every thirty seconds, in every project, for
        the life of the process -- the overwhelming majority of it announcing
        that nothing had changed. The state was worth having; broadcasting it
        on a timer was not. Reported here instead, it is read when someone
        asks, which is what /healthz is already for.
        """
        now = time.monotonic()

        return TunnelState(
            state=self._tunnel_state,
            connected_for_seconds=(
                round(now - self._connected_since, 1) if self._connected_since is not None else None
            ),
            seconds_since_connected=(
                round(now - self._last_connected_at, 1)
                if self._connected_since is None and self._last_connected_at is not None
                else None
            ),
            reconnect_attempts=self._reconnect_attempts,
            last_error=self._last_error,
        )

    def _install_stack_dump_handler(self) -> None:
        """Let an operator ask a wedged runtime what its threads are doing.

        `kill -USR1 <pid>` dumps every thread's stack to stderr, which lands in
        the allocation log. Costs nothing until used, and is the difference
        between "the tunnel is stuck" and knowing exactly where.
        """
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
        except Exception as e:  # pragma: no cover - platform dependent
            self._logger.debug(f"Could not install stack dump handler: {e}")

    async def stop(self) -> None:
        self._running = False
        self._tunnel_state = "stopped"
        self._connected_since = None
        if self._stop_event:
            self._stop_event.set()

        await self._cancel_unary_tasks()
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
            self._tunnel_state = "connected"
            self._connected_since = time.monotonic()
            self._last_connected_at = self._connected_since
            self._reconnect_attempts = 0
            self._last_error = None
            self._logger.trace(f"Tunnel connected to {self._url}")

            try:
                async for raw_message in ws:
                    if not self._running:
                        # Drain queued frames while stop closes the socket. Breaking
                        # here can leave receive backpressure hiding the close reply.
                        continue

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

                        if not self._start_unary_task(ws, request):
                            await ws.close(code=1013, reason="Too many pending tunnel requests")
                            break

                    except Exception as e:
                        self._logger.error(f"Error processing tunnel message: {e}")
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
                await self._cancel_unary_tasks()
                await self._cancel_stream_tasks()
                self._websocket = None

    def _start_unary_task(self, ws: Any, request: TunnelRequest) -> bool:
        if len(self._unary_tasks) >= _MAX_PENDING_UNARY_REQUESTS:
            return False

        # Preserve arrival order for operations on an existing session without
        # serializing independent session creation or other sessions' traffic.
        session_id = request.params.get("session_id")
        session_lock: asyncio.Lock | None = None
        if isinstance(session_id, str):
            session_lock, count = self._session_requests.get(session_id, (asyncio.Lock(), 0))
            self._session_requests[session_id] = (session_lock, count + 1)

        task = asyncio.create_task(self._handle_unary_request(ws, request, session_lock))
        self._unary_tasks.add(task)

        def finished(done: asyncio.Task[None]) -> None:
            self._unary_tasks.discard(done)
            if isinstance(session_id, str) and session_id in self._session_requests:
                lock, remaining = self._session_requests[session_id]
                if remaining == 1:
                    del self._session_requests[session_id]
                else:
                    self._session_requests[session_id] = (lock, remaining - 1)

        task.add_done_callback(finished)
        return True

    async def _handle_unary_request(
        self, ws: Any, request: TunnelRequest, session_lock: asyncio.Lock | None
    ) -> None:
        try:
            async with AsyncExitStack() as resources:
                if session_lock is not None:
                    await resources.enter_async_context(session_lock)
                await resources.enter_async_context(self._unary_slots)
                response = await self._dispatch_request(request)
                await self._send_json(ws, response.to_dict())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._logger.error(f"Error processing tunnel message: {error}")
            with suppress(Exception):
                await self._send_json(
                    ws, TunnelResponse(request_id=request.request_id, error=str(error)).to_dict()
                )

    async def _cancel_unary_tasks(self) -> None:
        tasks = list(self._unary_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._unary_tasks.clear()
        self._session_requests.clear()

    async def _dispatch_request(self, request: TunnelRequest) -> TunnelResponse:
        if request.method != "sessions.create_event":
            return await self._dispatcher.dispatch(request)

        with _best_effort_tracing_span(
            self._tracer,
            "tunnel.request",
            {
                "rpc.system": "parlant.tunnel",
                "rpc.method": request.method,
                "parlant.tunnel.request_id": request.request_id,
            },
        ):
            return await self._dispatcher.dispatch(request)

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
            self._logger.error(f"Error processing tunnel stream: {e}")
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


def create_tunnel_service(
    application: Application,
    health_reporter: HealthReporter,
    logger: Logger,
    tracer: Tracer | None = None,
) -> ParlantDashboardTunnelService | None:
    """Create a tunnel service if PARLANT_DASHBOARD_PROJECT_TOKEN is set."""
    token = os.environ.get("PARLANT_DASHBOARD_PROJECT_TOKEN", "")

    if not token:
        return None

    dispatcher = TunnelRequestDispatcher(
        application=application,
        health_reporter=health_reporter,
        logger=logger,
    )

    return ParlantDashboardTunnelService(
        url=_get_dashboard_tunnel_url(),
        token=token,
        dispatcher=dispatcher,
        logger=logger,
        tracer=tracer,
    )
