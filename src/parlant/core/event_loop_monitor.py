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

"""Event loop health monitor.

Runs a periodic task that schedules a callback on the event loop and measures
the delay between the expected and actual execution time. A healthy event loop
processes the callback promptly; a blocked one introduces measurable latency.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum


class EventLoopHealth(str, Enum):
    """Event loop health status following incident.io conventions."""

    HEALTHY = "healthy"
    """Latency is within acceptable bounds (< 200ms)."""

    DEGRADED = "degraded"
    """Latency is elevated (200ms–500ms). The event loop is slow but functional."""

    UNHEALTHY = "unhealthy"
    """Latency exceeds acceptable bounds (> 500ms). The event loop is blocked."""


@dataclass
class EventLoopStatus:
    """Snapshot of event loop health."""

    health: EventLoopHealth
    latency_ms: float
    """Measured delay in milliseconds between expected and actual callback execution."""


# Thresholds in seconds
_DEGRADED_THRESHOLD = 0.2  # 200ms
_UNHEALTHY_THRESHOLD = 0.5  # 500ms


def _classify_event_loop_latency(latency_seconds: float) -> EventLoopHealth:
    if latency_seconds >= _UNHEALTHY_THRESHOLD:
        return EventLoopHealth.UNHEALTHY
    if latency_seconds >= _DEGRADED_THRESHOLD:
        return EventLoopHealth.DEGRADED
    return EventLoopHealth.HEALTHY


class EventLoopMonitor:
    """Monitors event loop responsiveness by periodically measuring callback latency.

    Use as an async context manager for automatic lifecycle management::

        async with EventLoopMonitor() as monitor:
            print(monitor.status)

    In production, register with an ``AsyncExitStack`` so it's stopped on shutdown::

        container[EventLoopMonitor] = await exit_stack.enter_async_context(
            EventLoopMonitor()
        )
    """

    def __init__(self, interval: float = 0.1, window: float = 60.0) -> None:
        self._interval = interval
        self._window = window
        self._window_start: float = 0.0
        self._window_peak: float = 0.0
        self._status = EventLoopStatus(
            health=EventLoopHealth.HEALTHY,
            latency_ms=0.0,
        )
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "EventLoopMonitor":
        self._start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        await self._stop()

    @property
    def status(self) -> EventLoopStatus:
        """Return the most recent health snapshot."""
        return self._status

    def _start(self) -> None:
        self._task = asyncio.create_task(self._monitor_loop(), name="EventLoopMonitor")

    async def _stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        """Periodically measure event loop latency.

        Measures the overshoot of ``asyncio.sleep(interval)`` — if the event
        loop was blocked, the sleep returns late and the overshoot reflects
        the blocking duration. The reported status reflects the max latency
        within the configured window.
        """
        loop = asyncio.get_running_loop()
        self._window_start = loop.time()
        self._window_peak = 0.0

        while True:
            expected = loop.time() + self._interval
            await asyncio.sleep(self._interval)
            now = loop.time()

            latency = max(0.0, now - expected)

            # Reset the window if it has expired.
            if now - self._window_start >= self._window:
                self._window_start = now
                self._window_peak = 0.0

            self._window_peak = max(self._window_peak, latency)

            self._status = EventLoopStatus(
                health=_classify_event_loop_latency(self._window_peak),
                latency_ms=round(self._window_peak * 1000, 2),
            )
