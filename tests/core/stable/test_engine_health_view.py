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

"""Engine verdicts include every cause and the actual configured comparison.

These tests use real turn reports and the production view/reporter. The additive
reason contract must not change existing thresholds or rounded metric fields.
"""

from datetime import datetime, timedelta, timezone
import json

import pytest

from parlant.core.application_context import ApplicationContext
from parlant.core.health.engine_view import ENGINE_TTFM_KIND, ENGINE_TURN_KIND, EngineHealthView
from parlant.core.health.reporter import (
    HealthReport,
    HealthReporter,
    OverallHealth,
    ReportRetention,
)


def _turns(successes: int, total: int, latency_ms: float = 1000.0) -> list[HealthReport]:
    now = datetime.now(timezone.utc)
    return [
        HealthReport(
            kind=ENGINE_TURN_KIND,
            timestamp=now,
            attributes={
                "success": index < successes,
                "latency_ms": latency_ms,
                "error_class": None if index < successes else "TimeoutError",
            },
        )
        for index in range(total)
    ]


def _reason(
    metric: str, severity: str, observed_value: float, threshold: float
) -> dict[str, str | float]:
    return {
        "code": "low_success_rate" if metric == "success_rate" else "high_turn_latency",
        "metric": metric,
        "severity": severity,
        "observed_value": observed_value,
        "threshold": threshold,
        "comparison": "lt" if metric == "success_rate" else "gte",
    }


@pytest.mark.parametrize("sample_kind", ["empty", "healthy-turns", "ttfm-only"])
def test_that_healthy_and_empty_engine_reports_have_no_health_reasons(sample_kind: str) -> None:
    reports = {ENGINE_TURN_KIND: _turns(3, 3)} if sample_kind == "healthy-turns" else {}
    if sample_kind == "ttfm-only":
        reports[ENGINE_TTFM_KIND] = [
            HealthReport(
                kind=ENGINE_TTFM_KIND,
                timestamp=datetime.now(timezone.utc),
                attributes={"ttfm_ms": 90_000.0},
            )
        ]
    snapshot = EngineHealthView().render(reports)
    assert snapshot.status is OverallHealth.HEALTHY
    assert snapshot.body["reasons"] == []


@pytest.mark.parametrize(
    "latency,warning,critical,severity,threshold",
    [
        (34_200.0, 30_000.0, 60_000.0, "degraded", 30_000.0),
        (61_000.0, 30_000.0, 60_000.0, "unhealthy", 60_000.0),
        (1500.0, 1000.0, 2000.0, "degraded", 1000.0),
        (2500.0, 1000.0, 2000.0, "unhealthy", 2000.0),
    ],
)
def test_that_slow_turns_report_the_latency_reason_and_configured_threshold(
    latency: float, warning: float, critical: float, severity: str, threshold: float
) -> None:
    snapshot = EngineHealthView(degraded_p95_ms=warning, unhealthy_p95_ms=critical).render(
        {ENGINE_TURN_KIND: _turns(20, 20, latency)}
    )
    assert snapshot.status.value == severity
    assert snapshot.body["success_rate"] == 1.0
    assert snapshot.body["recent_errors"] == {}
    assert snapshot.body["reasons"] == [_reason("p95_latency_ms", severity, latency, threshold)]


@pytest.mark.parametrize(
    "successes,warning,critical,severity,threshold",
    [
        (18, 0.95, 0.80, "degraded", 0.95),
        (15, 0.95, 0.80, "unhealthy", 0.80),
        (19, 0.99, 0.97, "unhealthy", 0.97),
        (18, 0.99, 0.85, "degraded", 0.99),
    ],
)
def test_that_failed_turns_report_the_success_rate_reason_and_configured_threshold(
    successes: int, warning: float, critical: float, severity: str, threshold: float
) -> None:
    snapshot = EngineHealthView(
        degraded_below_success_rate=warning, unhealthy_below_success_rate=critical
    ).render({ENGINE_TURN_KIND: _turns(successes, 20)})
    assert snapshot.status.value == severity
    assert snapshot.body["recent_errors"] == {"TimeoutError": 20 - successes}
    assert snapshot.body["reasons"] == [
        _reason("success_rate", severity, successes / 20, threshold)
    ]


@pytest.mark.parametrize(
    "successes,latency,success_severity,latency_severity",
    [
        (18, 34_200.0, "degraded", "degraded"),
        (15, 61_000.0, "unhealthy", "unhealthy"),
        (18, 61_000.0, "degraded", "unhealthy"),
        (15, 34_200.0, "unhealthy", "degraded"),
    ],
)
def test_that_all_contributing_metrics_are_reported_with_their_own_severity(
    successes: int, latency: float, success_severity: str, latency_severity: str
) -> None:
    snapshot = EngineHealthView().render({ENGINE_TURN_KIND: _turns(successes, 20, latency)})
    assert snapshot.status.value == (
        "unhealthy" if "unhealthy" in (success_severity, latency_severity) else "degraded"
    )
    assert sorted(snapshot.body["reasons"], key=lambda reason: reason["metric"]) == sorted(
        [
            _reason(
                "success_rate",
                success_severity,
                successes / 20,
                0.8 if success_severity == "unhealthy" else 0.95,
            ),
            _reason(
                "p95_latency_ms",
                latency_severity,
                latency,
                60_000.0 if latency_severity == "unhealthy" else 30_000.0,
            ),
        ],
        key=lambda reason: reason["metric"],
    )


@pytest.mark.parametrize(
    "successes,latency,status,metric,threshold",
    [
        (19, 1000.0, "healthy", None, None),
        (18, 1000.0, "degraded", "success_rate", 0.95),
        (16, 1000.0, "degraded", "success_rate", 0.95),
        (15, 1000.0, "unhealthy", "success_rate", 0.80),
        (20, 29_999.0, "healthy", None, None),
        (20, 30_000.0, "degraded", "p95_latency_ms", 30_000.0),
        (20, 59_999.0, "degraded", "p95_latency_ms", 30_000.0),
        (20, 60_000.0, "unhealthy", "p95_latency_ms", 60_000.0),
    ],
)
def test_that_threshold_boundaries_preserve_existing_health_classification(
    successes: int, latency: float, status: str, metric: str | None, threshold: float | None
) -> None:
    snapshot = EngineHealthView().render({ENGINE_TURN_KIND: _turns(successes, 20, latency)})
    assert snapshot.status.value == status
    expected = []
    if metric is not None and threshold is not None:
        expected = [
            _reason(
                metric, status, successes / 20 if metric == "success_rate" else latency, threshold
            )
        ]
    assert snapshot.body["reasons"] == expected


@pytest.mark.parametrize("metric", ["success_rate", "p95_latency_ms"])
def test_that_health_reasons_use_unrounded_values_for_classification(metric: str) -> None:
    if metric == "success_rate":
        snapshot = EngineHealthView(
            degraded_below_success_rate=0.66668, unhealthy_below_success_rate=0.5
        ).render({ENGINE_TURN_KIND: _turns(2, 3)})
        assert snapshot.status is OverallHealth.DEGRADED
        # The existing rounded metric looks ABOVE the warning threshold, but the
        # actual ratio is below it. The reason must retain the actual observation.
        assert snapshot.body["success_rate"] == 0.6667
        assert snapshot.body["reasons"] == [_reason("success_rate", "degraded", 2 / 3, 0.66668)]
    else:
        snapshot = EngineHealthView().render({ENGINE_TURN_KIND: _turns(3, 3, 29_999.999)})
        # Rounding produces 30 seconds; the actual duration is below the threshold.
        assert snapshot.body["p95_latency_ms"] == 30_000.0
        assert snapshot.status is OverallHealth.HEALTHY
        assert snapshot.body["reasons"] == []


def test_that_engine_health_reasons_survive_reporter_snapshot_serialization() -> None:
    reporter = HealthReporter(
        ApplicationContext(instance_id="engine-health-reasons"), snapshot_cache_ttl=timedelta(0)
    )
    reporter.configure_retention(
        ENGINE_TURN_KIND, ReportRetention(window=timedelta(minutes=10), max_count=100)
    )
    reporter.register_view(EngineHealthView(health_reporter=reporter))
    for turn in _turns(18, 20, 61_000.0):
        reporter.report(ENGINE_TURN_KIND, turn.attributes)

    snapshot = json.loads(json.dumps(reporter.snapshot()))
    assert snapshot["status"] == "unhealthy"
    engine = snapshot["checks"]["engine"]
    assert engine["status"] == "unhealthy"
    assert engine["sample_count"] == 20
    assert len(engine["reasons"]) == 2
    assert _reason("success_rate", "degraded", 0.9, 0.95) in engine["reasons"]
    assert _reason("p95_latency_ms", "unhealthy", 61_000.0, 60_000.0) in engine["reasons"]
