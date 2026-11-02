# Copyright 2026 Emcie Co Ltd.
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

import asyncio
import time

from fastapi import APIRouter, FastAPI, status
from fastapi.routing import APIRoute
from pytest import mark
from starlette.requests import Request
from parlant.core.container import Container
import httpx

from parlant.api.app import _resolve_operation_id, create_api_app
from parlant.api.authorization import AuthorizationPolicy, HeaderApiKeyAuthorizationPolicy
from parlant.core.event_loop_monitor import EventLoopMonitor, EventLoopHealth


PROJECT_TOKEN_HEADER = "X-Parlant-Project-Token"
PROJECT_TOKEN = "strict-project-token"


async def test_health_check_returns_status_and_event_loop_check(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/healthz")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "degraded", "unhealthy")
    assert "checks" in data
    assert "event_loop" in data["checks"]
    assert "status" in data["checks"]["event_loop"]
    assert "latency_ms" in data["checks"]["event_loop"]


async def test_health_check_remains_public_with_strict_authorization(
    container: Container,
) -> None:
    strict_container = container.clone()
    strict_container[AuthorizationPolicy] = HeaderApiKeyAuthorizationPolicy(
        api_key=PROJECT_TOKEN,
        header_name=PROJECT_TOKEN_HEADER,
    )
    app = await create_api_app(strict_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == status.HTTP_200_OK


async def test_event_loop_monitor_reports_healthy_under_normal_conditions(
    container: Container,
) -> None:
    monitor = container[EventLoopMonitor]

    # Give the monitor a couple of ticks to measure
    await asyncio.sleep(1)

    s = monitor.status
    assert s.health == EventLoopHealth.HEALTHY
    assert s.latency_ms < 100  # Well under degraded threshold


async def test_event_loop_monitor_detects_degraded_loop() -> None:
    # A 10s window, so the one stall below is more than 1% of it.
    async with EventLoopMonitor(interval=0.1, window=10.0) as monitor:
        # Let it establish a healthy baseline
        await asyncio.sleep(1)
        assert monitor.status.health == EventLoopHealth.HEALTHY

        # Block the event loop with a synchronous sleep.
        blocking_done = asyncio.Event()

        async def block_then_signal() -> None:
            await asyncio.sleep(0.05)
            time.sleep(0.3)
            blocking_done.set()

        asyncio.create_task(block_then_signal())
        await blocking_done.wait()

        # Yield to let the monitor's tick complete and record the overshoot.
        await asyncio.sleep(5)

        s = monitor.status
        assert s.health == EventLoopHealth.DEGRADED, (
            f"Expected DEGRADED after 300ms block, got {s.health} with latency {s.latency_ms}ms"
        )
        assert s.latency_ms >= 200


async def test_event_loop_monitor_detects_unhealthy_loop() -> None:
    # A 10s window, so the one stall below is more than 1% of it.
    async with EventLoopMonitor(interval=0.1, window=10.0) as monitor:
        # Let it establish a healthy baseline
        await asyncio.sleep(1)
        assert monitor.status.health == EventLoopHealth.HEALTHY

        # Block the event loop with a synchronous sleep.
        blocking_done = asyncio.Event()

        async def block_then_signal() -> None:
            await asyncio.sleep(0.05)
            time.sleep(3)
            blocking_done.set()

        asyncio.create_task(block_then_signal())
        await blocking_done.wait()

        # Yield to let the monitor's tick complete and record the overshoot.
        await asyncio.sleep(5)

        s = monitor.status
        assert s.health == EventLoopHealth.UNHEALTHY, (
            f"Expected UNHEALTHY after 3s block, got {s.health} with latency {s.latency_ms}ms"
        )
        assert s.latency_ms >= 500


@mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/api/projects/project-1/sessions/session-1/events", "create_event"),
        ("GET", "/api/projects/project-1/sessions/session-1/events", "list_events"),
        ("DELETE", "/api/projects/project-1/sessions/session-1/events", None),
        ("POST", "/sessions/session-1/events", None),
        ("POST", "/api/projects/project-1/missing", None),
    ],
)
def test_that_operation_resolution_matches_nested_router_prefixes_and_methods(
    method: str, path: str, expected: str | None
) -> None:
    app = FastAPI()
    sessions = APIRouter()

    @sessions.post("/sessions/{session_id}/events", operation_id="create_event")
    async def create_event(session_id: str) -> dict[str, str]:
        return {"session_id": session_id}

    @sessions.get("/sessions/{session_id}/events", operation_id="list_events")
    async def list_events(session_id: str) -> dict[str, str]:
        return {"session_id": session_id}

    projects = APIRouter()
    projects.include_router(sessions, prefix="/projects/{project_id}")
    app.include_router(projects, prefix="/api")
    request = Request(
        {
            "type": "http",
            "app": app,
            "method": method,
            "path": path,
            "root_path": "",
            "headers": [],
            "query_string": b"",
        }
    )

    # Middleware runs before routing has populated scope['route'].
    assert "route" not in request.scope
    assert _resolve_operation_id(request) == expected


def test_that_operation_resolution_preserves_an_already_matched_api_route() -> None:
    async def endpoint() -> None:
        pass

    route = APIRoute("/events", endpoint, methods=["POST"], operation_id="create_event")
    request = Request({"type": "http", "route": route})
    assert _resolve_operation_id(request) == "create_event"
