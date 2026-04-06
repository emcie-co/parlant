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

from fastapi import status
from lagom import Container
import httpx

from parlant.core.event_loop_monitor import EventLoopMonitor, EventLoopHealth


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
    async with EventLoopMonitor(interval=0.1) as monitor:
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
    async with EventLoopMonitor(interval=0.1) as monitor:
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
