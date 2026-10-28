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

"""``configure_healthz`` builds the ``/healthz`` views from the thresholds
bound on the container, so a deployment can tune them without replacing the
views themselves."""

import asyncio
import time
from datetime import timedelta

from parlant.api.health import configure_healthz
from parlant.core.container import Container
from parlant.core.application_context import ApplicationContext
from parlant.core.event_loop_monitor import EventLoopMonitor
from parlant.core.health import (
    ENGINE_TURN_KIND,
    NLP_REQUEST_KIND,
    HealthReporter,
    RouteThresholds,
)
from parlant.core.health.thresholds import (
    EngineHealthThresholds,
    EventLoopHealthThresholds,
    HealthThresholds,
    NLPHealthThresholds,
)


def _container(thresholds: HealthThresholds, monitor: EventLoopMonitor) -> Container:
    container = Container()
    container[ApplicationContext] = ApplicationContext(instance_id="healthz-thresholds")
    container[HealthReporter] = HealthReporter(
        container[ApplicationContext],
        snapshot_cache_ttl=timedelta(0),
    )
    container[EventLoopMonitor] = monitor
    container[HealthThresholds] = thresholds
    return container


def _report_turn(reporter: HealthReporter, latency_ms: float) -> None:
    reporter.report(
        ENGINE_TURN_KIND,
        {"success": True, "latency_ms": latency_ms, "error_class": None},
    )


def _report_request(reporter: HealthReporter, route: str, latency_ms: float) -> None:
    reporter.report(
        NLP_REQUEST_KIND,
        {
            "route": route,
            "agent_id": "agent-1",
            "schema": "Schema",
            "model": "test-model",
            "success": True,
            "latency_ms": latency_ms,
            "error_class": None,
        },
    )


async def test_that_configure_healthz_builds_its_views_from_the_container_thresholds() -> None:
    thresholds = HealthThresholds(
        engine=EngineHealthThresholds(degraded_p95_ms=5.0, unhealthy_p95_ms=1_000.0),
        nlp=NLPHealthThresholds(
            routes={"matching.rank": RouteThresholds(unhealthy_p95_ms=5.0)},
        ),
    )

    async with EventLoopMonitor() as monitor:
        container = _container(thresholds, monitor)
        configure_healthz(container)

        reporter = container[HealthReporter]
        _report_turn(reporter, latency_ms=10.0)
        _report_request(reporter, route="matching.rank", latency_ms=10.0)
        _report_request(reporter, route="generation.message", latency_ms=10.0)

        snapshot = reporter.snapshot()

    engine = snapshot["checks"]["engine"]
    assert engine["status"] == "degraded"
    assert engine["reasons"] == [
        {
            "code": "high_turn_latency",
            "metric": "p95_latency_ms",
            "severity": "degraded",
            "observed_value": 10.0,
            "threshold": 5.0,
            "comparison": "gte",
        }
    ]

    nlp = snapshot["checks"]["nlp"]
    assert nlp["routes"]["matching.rank"]["status"] == "unhealthy"
    assert nlp["routes"]["generation.message"]["status"] == "healthy"

    assert snapshot["status"] == "unhealthy"


async def test_that_healthz_views_stay_healthy_under_the_default_thresholds() -> None:
    async with EventLoopMonitor() as monitor:
        container = _container(HealthThresholds(), monitor)
        configure_healthz(container)

        reporter = container[HealthReporter]
        _report_turn(reporter, latency_ms=10.0)
        _report_request(reporter, route="matching.rank", latency_ms=10.0)

        snapshot = reporter.snapshot()

    assert snapshot["status"] == "healthy"
    assert snapshot["checks"]["engine"]["reasons"] == []


async def test_that_the_event_loop_check_reports_its_peak_lag_and_the_share_of_the_window_spent_lagging() -> (
    None
):
    async with EventLoopMonitor(interval=0.02, window=10.0) as monitor:
        container = _container(HealthThresholds(), monitor)
        configure_healthz(container)

        await asyncio.sleep(0.04)
        time.sleep(0.3)
        await asyncio.sleep(0.06)

        event_loop = container[HealthReporter].snapshot()["checks"]["event_loop"]

    assert event_loop["status"] == "degraded"
    assert event_loop["latency_ms"] >= 300
    assert event_loop["lag_share"] >= 0.03


def test_that_default_thresholds_match_the_previous_hardcoded_values() -> None:
    thresholds = HealthThresholds()

    assert thresholds.engine == EngineHealthThresholds(
        degraded_below_success_rate=0.95,
        unhealthy_below_success_rate=0.80,
        degraded_p95_ms=30_000.0,
        unhealthy_p95_ms=60_000.0,
    )

    assert thresholds.event_loop == EventLoopHealthThresholds(
        degraded_lag_ms=200.0,
        unhealthy_lag_ms=500.0,
        lag_share=0.01,
    )

    assert thresholds.nlp.degraded_below_success_rate == 0.95
    assert thresholds.nlp.unhealthy_below_success_rate == 0.80
    assert thresholds.nlp.degraded_p50_ms == 20_000.0
    assert thresholds.nlp.unhealthy_p50_ms == 30_000.0
    assert thresholds.nlp.degraded_p95_ms == 45_000.0
    assert thresholds.nlp.unhealthy_p95_ms == 60_000.0

    matching = RouteThresholds(
        degraded_p50_ms=8_000.0,
        unhealthy_p50_ms=12_000.0,
        degraded_p95_ms=15_000.0,
        unhealthy_p95_ms=20_000.0,
    )
    review = RouteThresholds(
        degraded_p50_ms=10_000.0,
        unhealthy_p50_ms=15_000.0,
        degraded_p95_ms=20_000.0,
        unhealthy_p95_ms=25_000.0,
    )
    assert dict(thresholds.nlp.routes) == {
        "matching.rank": matching,
        "matching.distill": matching,
        "review.message": review,
        "review.tools": review,
    }
