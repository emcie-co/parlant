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

"""HealthView for the engine processing layer.

Aggregates ``engine.turn`` (one per processed turn) and ``engine.ttfm``
(one per first-message emission) into success rate, latency percentiles,
TTFM percentiles, recent errors, and turns-per-minute rate windows.
"""

from collections import Counter
from datetime import timedelta
from typing import Any, Mapping, Sequence

from parlant.core.health.reporter import (
    StatusCriticality,
    HealthReport,
    HealthReporter,
    OverallHealth,
    ViewSnapshot,
)


ENGINE_TURN_KIND = "engine.turn"
ENGINE_TTFM_KIND = "engine.ttfm"

ENGINE_TURNS_COUNTER = "engine.turns"

_RATE_WINDOWS: tuple[tuple[str, timedelta], ...] = (
    ("1m", timedelta(minutes=1)),
    ("5m", timedelta(minutes=5)),
    ("1h", timedelta(hours=1)),
    ("1d", timedelta(days=1)),
)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[idx]


class EngineHealthView:
    """Renders engine processing health from turn outcomes and TTFM samples."""

    name = "engine"
    criticality = StatusCriticality.CRITICAL
    kinds: tuple[str, ...] = (ENGINE_TURN_KIND, ENGINE_TTFM_KIND)

    # Report attribute keys — producers and the renderer share these.
    ATTR_SUCCESS = "success"
    ATTR_LATENCY_MS = "latency_ms"
    ATTR_ERROR_CLASS = "error_class"
    ATTR_TTFM_MS = "ttfm_ms"

    def __init__(
        self,
        *,
        health_reporter: HealthReporter | None = None,
        degraded_below_success_rate: float = 0.95,
        unhealthy_below_success_rate: float = 0.80,
        degraded_p95_ms: float = 30_000.0,
        unhealthy_p95_ms: float = 60_000.0,
        recent_errors_top_k: int = 5,
    ) -> None:
        self._health_reporter = health_reporter
        self._degraded_below_success_rate = degraded_below_success_rate
        self._unhealthy_below_success_rate = unhealthy_below_success_rate
        self._degraded_p95_ms = degraded_p95_ms
        self._unhealthy_p95_ms = unhealthy_p95_ms
        self._recent_errors_top_k = recent_errors_top_k

    def render(
        self,
        reports_by_kind: Mapping[str, Sequence[HealthReport]],
    ) -> ViewSnapshot:
        turn_reports = list(reports_by_kind.get(ENGINE_TURN_KIND, ()))
        ttfm_reports = list(reports_by_kind.get(ENGINE_TTFM_KIND, ()))

        rate_block = self._render_rate_block()

        if not turn_reports and not ttfm_reports:
            return ViewSnapshot(
                status=OverallHealth.HEALTHY,
                body={
                    "sample_count": 0,
                    "success_rate": 1.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p50_ttfm_ms": 0.0,
                    "p95_ttfm_ms": 0.0,
                    "recent_errors": {},
                    **rate_block,
                },
            )

        successes = sum(1 for r in turn_reports if bool(r.attributes.get(self.ATTR_SUCCESS)))
        sample_count = len(turn_reports)
        success_rate = successes / sample_count if sample_count else 1.0

        latencies = [
            float(r.attributes[self.ATTR_LATENCY_MS])
            for r in turn_reports
            if self.ATTR_LATENCY_MS in r.attributes
            and r.attributes[self.ATTR_LATENCY_MS] is not None
        ]
        p50_latency = _percentile(latencies, 0.5)
        p95_latency = _percentile(latencies, 0.95)

        ttfms = [
            float(r.attributes[self.ATTR_TTFM_MS])
            for r in ttfm_reports
            if self.ATTR_TTFM_MS in r.attributes and r.attributes[self.ATTR_TTFM_MS] is not None
        ]
        p50_ttfm = _percentile(ttfms, 0.5)
        p95_ttfm = _percentile(ttfms, 0.95)

        error_counter: Counter[str] = Counter()
        for r in turn_reports:
            if r.attributes.get(self.ATTR_SUCCESS):
                continue
            error_class = str(r.attributes.get(self.ATTR_ERROR_CLASS) or "Unknown")
            error_counter[error_class] += 1
        recent_errors = dict(error_counter.most_common(self._recent_errors_top_k))

        status = self._classify(success_rate=success_rate, p95_ms=p95_latency)

        return ViewSnapshot(
            status=status,
            body={
                "sample_count": sample_count,
                "success_rate": round(success_rate, 4),
                "p50_latency_ms": round(p50_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "p50_ttfm_ms": round(p50_ttfm, 2),
                "p95_ttfm_ms": round(p95_ttfm, 2),
                "recent_errors": recent_errors,
                **rate_block,
            },
        )

    def _render_rate_block(self) -> dict[str, Any]:
        if self._health_reporter is None:
            return {}
        try:
            return {
                "turns_per_minute": {
                    label: round(
                        self._health_reporter.counter_per_minute(ENGINE_TURNS_COUNTER, window),
                        4,
                    )
                    for label, window in _RATE_WINDOWS
                },
            }
        except KeyError:
            return {}

    def _classify(self, *, success_rate: float, p95_ms: float) -> OverallHealth:
        if success_rate < self._unhealthy_below_success_rate or p95_ms >= self._unhealthy_p95_ms:
            return OverallHealth.UNHEALTHY
        if success_rate < self._degraded_below_success_rate or p95_ms >= self._degraded_p95_ms:
            return OverallHealth.DEGRADED
        return OverallHealth.HEALTHY
