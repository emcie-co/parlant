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

Health follows how much of a rolling window the loop spent lagging, not the
single worst lag in it. A loop that stalls once for half a second is fine and
reports that as its peak; one whose stalls add up to a share of the window is
not. A blocked loop can't take measurements, so each late measurement counts
for as long as it was late: one long stall weighs as much as many short ones.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import Enum


class EventLoopHealth(str, Enum):
    """Event loop health status following incident.io conventions."""

    HEALTHY = "healthy"
    """Lag stayed within bounds for all but a small share of the window."""

    DEGRADED = "degraded"
    """Lags of 200ms or more add up to 1% of the window. The loop is slow but functional."""

    UNHEALTHY = "unhealthy"
    """Lags of 500ms or more add up to 1% of the window. The loop is being blocked."""


@dataclass
class EventLoopStatus:
    """Snapshot of event loop health."""

    health: EventLoopHealth
    latency_ms: float
    """The peak lag in the window, in milliseconds."""
    lag_share: float = 0.0
    """The share of the window, from 0 to 1, spent in lags at or above the
    degraded cutoff."""


@dataclass(frozen=True)
class EventLoopHealthThresholds:
    """Cutoffs for the ``event_loop`` check.

    The loop is degraded or unhealthy when the lags at or above that status's
    cutoff, in milliseconds, add up to at least ``lag_share`` of the monitor's
    rolling window (five minutes by default). The default 1% is three seconds
    of lag in five minutes, so a single brief stall shows as the peak without
    changing the status. Set ``lag_share`` to 0 to have any single lag at a
    cutoff count, as the check did before.
    """

    degraded_lag_ms: float = 200.0
    unhealthy_lag_ms: float = 500.0
    lag_share: float = 0.01


@dataclass(frozen=True)
class _Lag:
    at: float
    seconds: float


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

    def __init__(
        self,
        interval: float = 0.1,
        window: float = 300.0,
        thresholds: EventLoopHealthThresholds | None = None,
    ) -> None:
        self._interval = interval
        self._window = window
        self._thresholds = thresholds if thresholds is not None else EventLoopHealthThresholds()
        # Both oldest first, and kept small so each measurement costs the
        # loop it measures next to nothing: the peak candidates in the window
        # (each larger than every later one), and the lags at the degraded
        # cutoff or above.
        self._peaks: deque[_Lag] = deque()
        self._lagging: deque[_Lag] = deque()
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
        the blocking duration. The reported status reflects the lag within
        the last ``window`` seconds.
        """
        loop = asyncio.get_running_loop()

        while True:
            expected = loop.time() + self._interval
            await asyncio.sleep(self._interval)
            now = loop.time()

            self._record(_Lag(at=now, seconds=max(0.0, now - expected)))
            self._status = self._measure()

    def _record(self, lag: _Lag) -> None:
        while self._peaks and self._peaks[-1].seconds <= lag.seconds:
            self._peaks.pop()
        self._peaks.append(lag)

        if lag.seconds * 1000 >= self._thresholds.degraded_lag_ms:
            self._lagging.append(lag)

        oldest = lag.at - self._window
        for lags in (self._peaks, self._lagging):
            while lags and lags[0].at <= oldest:
                lags.popleft()

    def _measure(self) -> EventLoopStatus:
        unhealthy_cutoff = self._thresholds.unhealthy_lag_ms / 1000

        degraded_seconds = sum(lag.seconds for lag in self._lagging)
        unhealthy_seconds = sum(
            lag.seconds for lag in self._lagging if lag.seconds >= unhealthy_cutoff
        )

        return EventLoopStatus(
            health=self._classify(degraded_seconds, unhealthy_seconds),
            latency_ms=round(self._peaks[0].seconds * 1000, 2),
            lag_share=round(degraded_seconds / self._window, 4),
        )

    def _classify(self, degraded_seconds: float, unhealthy_seconds: float) -> EventLoopHealth:
        # Strictly positive, so a zero share means "any lag at the cutoff".
        def reaches_share(seconds: float) -> bool:
            return seconds > 0 and seconds >= self._thresholds.lag_share * self._window

        if reaches_share(unhealthy_seconds):
            return EventLoopHealth.UNHEALTHY
        if reaches_share(degraded_seconds):
            return EventLoopHealth.DEGRADED
        return EventLoopHealth.HEALTHY
