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

"""The event loop monitor classifies lag against thresholds it is given,
not against fixed module constants.

Health follows how much of a rolling window the loop spent lagging, not the
single worst lag in it: one brief spike shows as the peak but leaves the loop
healthy, while lag that adds up to the configured share of the window does not.
"""

import asyncio
import time

import pytest

from parlant.core.event_loop_monitor import EventLoopHealth, EventLoopMonitor
from parlant.core.health.thresholds import EventLoopHealthThresholds


async def _block_loop_and_settle(seconds: float, interval: float) -> None:
    # Let the monitor take its first measurement, then stall the loop so the
    # next measurement overshoots by about ``seconds``.
    await asyncio.sleep(interval * 2)
    time.sleep(seconds)
    await asyncio.sleep(interval * 3)


@pytest.mark.parametrize(
    "degraded_lag_ms,unhealthy_lag_ms,expected",
    [
        (50.0, 5_000.0, EventLoopHealth.DEGRADED),
        (10.0, 50.0, EventLoopHealth.UNHEALTHY),
        (5_000.0, 10_000.0, EventLoopHealth.HEALTHY),
    ],
)
async def test_that_event_loop_lag_is_classified_against_the_configured_thresholds(
    degraded_lag_ms: float,
    unhealthy_lag_ms: float,
    expected: EventLoopHealth,
) -> None:
    thresholds = EventLoopHealthThresholds(
        degraded_lag_ms=degraded_lag_ms,
        unhealthy_lag_ms=unhealthy_lag_ms,
    )

    # A short window, so the one stall below is well over 1% of it.
    async with EventLoopMonitor(interval=0.02, window=1.0, thresholds=thresholds) as monitor:
        await _block_loop_and_settle(seconds=0.15, interval=0.02)

        assert monitor.status.latency_ms >= 100
        assert monitor.status.health == expected


async def test_that_the_default_thresholds_need_lag_to_add_up_to_one_percent_of_the_window() -> (
    None
):
    assert EventLoopHealthThresholds() == EventLoopHealthThresholds(
        degraded_lag_ms=200.0,
        unhealthy_lag_ms=500.0,
        lag_share=0.01,
    )


async def test_that_a_single_brief_spike_shows_as_the_peak_but_leaves_the_loop_healthy() -> None:
    # 0.6s of lag is 0.2% of the default five-minute window.
    async with EventLoopMonitor(interval=0.02) as monitor:
        await _block_loop_and_settle(seconds=0.6, interval=0.02)

        assert monitor.status.latency_ms >= 500
        assert monitor.status.health == EventLoopHealth.HEALTHY


async def test_that_one_long_stall_counts_for_its_whole_duration_and_makes_the_loop_unhealthy() -> (
    None
):
    # One stall is one late measurement, but 0.6s of lag is 6% of a 10s window.
    async with EventLoopMonitor(interval=0.02, window=10.0) as monitor:
        await _block_loop_and_settle(seconds=0.6, interval=0.02)

        assert monitor.status.health == EventLoopHealth.UNHEALTHY
        assert monitor.status.lag_share >= 0.05


async def test_that_lag_under_the_unhealthy_cutoff_adding_up_to_the_share_degrades_the_loop() -> (
    None
):
    # 0.3s of lag: over the 200ms degraded cutoff, under the 500ms unhealthy one.
    async with EventLoopMonitor(interval=0.02, window=10.0) as monitor:
        await _block_loop_and_settle(seconds=0.3, interval=0.02)

        assert monitor.status.health == EventLoopHealth.DEGRADED


async def test_that_lag_is_counted_over_a_rolling_window_until_it_is_older_than_the_window() -> (
    None
):
    window = 1.0

    async with EventLoopMonitor(interval=0.02, window=window) as monitor:
        await asyncio.sleep(0.2)
        time.sleep(0.6)
        # A window that restarted every second would have dropped the stall by
        # now; the last second still holds it.
        await asyncio.sleep(0.4)

        assert monitor.status.health == EventLoopHealth.UNHEALTHY

        await asyncio.sleep(window)

        later = monitor.status
        assert later.health == EventLoopHealth.HEALTHY
        assert later.latency_ms < 500


async def test_that_a_zero_lag_share_makes_any_single_lag_over_a_cutoff_count() -> None:
    thresholds = EventLoopHealthThresholds(lag_share=0.0)

    async with EventLoopMonitor(interval=0.02, thresholds=thresholds) as monitor:
        await _block_loop_and_settle(seconds=0.6, interval=0.02)

        assert monitor.status.health == EventLoopHealth.UNHEALTHY
