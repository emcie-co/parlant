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

"""HealthView for the NLP layer.

Aggregates ``nlp.request`` and ``nlp.embed`` reports into a per-ROUTE
status with success rate, latency percentiles, a recent-error breakdown,
and a per-agent drill-down inside each route. Routes are the stable model
configuration vocabulary, and agents can define their own route configs,
so health is attributed to (route, agent) at the model-routing layer.
Reports from server-scope (agentless) calls group under the ``server``
bucket. The view's status is the worst across all observed routes; a
route's status is the worst across its agents.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from parlant.core.health.reporter import (
    StatusCriticality,
    HealthReport,
    HealthReporter,
    OverallHealth,
    ViewSnapshot,
)


@dataclass(frozen=True)
class PercentileResult:
    """Result of a per-route percentile query.

    ``is_stale`` is ``True`` when the result is served from a frozen
    snapshot taken the last time a query found live data — the live
    buffer is currently empty for the requested route. Callers
    making time-sensitive decisions (e.g. request hedging) should
    inspect ``is_stale`` and ``sample_count`` before trusting the
    values.
    """

    values: dict[float, float]
    sample_count: int
    captured_at: datetime
    is_stale: bool


@dataclass(frozen=True)
class RouteThresholds:
    """Per-route overrides for ``NLPHealthView`` classification.

    Any field left as ``None`` falls back to the view's default for that
    threshold, so callers only need to specify the thresholds they care
    about for a given route.
    """

    degraded_p50_ms: float | None = None
    unhealthy_p50_ms: float | None = None
    degraded_p95_ms: float | None = None
    unhealthy_p95_ms: float | None = None
    degraded_below_success_rate: float | None = None
    unhealthy_below_success_rate: float | None = None


NLP_REQUEST_KIND = "nlp.request"
NLP_EMBED_KIND = "nlp.embed"

NLP_REQUESTS_COUNTER = "nlp.requests"
NLP_TOKENS_COUNTER = "nlp.tokens"

# Rendered bucket for reports with no agent attribution (server-scope calls:
# indexing, evaluation, background jobs).
SERVER_AGENT_BUCKET = "server"

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


class NLPHealthView:
    """Renders NLP health grouped by route, then agent.

    Status thresholds are constructor parameters so deployments can tune
    them without changing the view itself. Defaults err toward leniency
    so transient blips don't pull a pod out of rotation.
    """

    name = "nlp"
    criticality = StatusCriticality.CRITICAL
    kinds: tuple[str, ...] = (NLP_REQUEST_KIND, NLP_EMBED_KIND)

    # Report attribute keys — producers and the renderer share these.
    ATTR_ROUTE = "route"
    ATTR_AGENT = "agent_id"
    ATTR_SCHEMA = "schema"
    ATTR_MODEL = "model"
    ATTR_SUCCESS = "success"
    ATTR_LATENCY_MS = "latency_ms"
    ATTR_ERROR_CLASS = "error_class"

    def __init__(
        self,
        *,
        health_reporter: HealthReporter | None = None,
        degraded_below_success_rate: float = 0.95,
        unhealthy_below_success_rate: float = 0.80,
        degraded_p50_ms: float | None = 20_000.0,
        unhealthy_p50_ms: float | None = 30_000.0,
        degraded_p95_ms: float = 45_000.0,
        unhealthy_p95_ms: float = 60_000.0,
        recent_errors_top_k: int = 5,
        route_thresholds: Mapping[str, RouteThresholds] | None = None,
    ) -> None:
        self._health_reporter = health_reporter
        self._degraded_below_success_rate = degraded_below_success_rate
        self._unhealthy_below_success_rate = unhealthy_below_success_rate
        self._degraded_p50_ms = degraded_p50_ms
        self._unhealthy_p50_ms = unhealthy_p50_ms
        self._degraded_p95_ms = degraded_p95_ms
        self._unhealthy_p95_ms = unhealthy_p95_ms
        self._recent_errors_top_k = recent_errors_top_k
        self._route_thresholds: Mapping[str, RouteThresholds] = dict(route_thresholds or {})
        self._frozen_per_route: dict[str, PercentileResult] = {}

    def percentiles_for_route(
        self,
        route: str,
        ps: Sequence[float] = (0.5, 0.95, 0.99),
        value_attribute: str = "latency_ms",
    ) -> PercentileResult:
        """Return p50/p95/p99 (or custom ``ps``) for ``route``.

        Reads the current ``NLP_REQUEST_KIND`` buffer (after age-pruning).
        If matching entries exist, computes live percentiles and freezes
        them as the route's last-known snapshot. If no matching entries
        exist, returns the frozen snapshot (with ``is_stale=True``) if
        one was ever captured for this route, otherwise a zero-filled
        ``is_stale=True`` result.
        """
        if self._health_reporter is None:
            return PercentileResult(
                values={p: 0.0 for p in ps},
                sample_count=0,
                captured_at=datetime.now(timezone.utc),
                is_stale=True,
            )

        reports = self._health_reporter.read_reports(NLP_REQUEST_KIND)

        values: list[float] = []
        for r in reports:
            if r.attributes.get(self.ATTR_ROUTE) != route:
                continue
            raw = r.attributes.get(value_attribute)
            if raw is None:
                continue
            values.append(float(raw))

        now = datetime.now(timezone.utc)

        if values:
            live = PercentileResult(
                values={p: _percentile(values, p) for p in ps},
                sample_count=len(values),
                captured_at=now,
                is_stale=False,
            )
            self._frozen_per_route[route] = live
            return live

        frozen = self._frozen_per_route.get(route)
        if frozen is not None:
            return PercentileResult(
                values=frozen.values,
                sample_count=frozen.sample_count,
                captured_at=frozen.captured_at,
                is_stale=True,
            )

        return PercentileResult(
            values={p: 0.0 for p in ps},
            sample_count=0,
            captured_at=now,
            is_stale=True,
        )

    def render(
        self,
        reports_by_kind: Mapping[str, Sequence[HealthReport]],
    ) -> ViewSnapshot:
        reports = [
            *reports_by_kind.get(NLP_REQUEST_KIND, ()),
            *reports_by_kind.get(NLP_EMBED_KIND, ()),
        ]

        rate_blocks = self._render_rate_blocks()

        by_route: dict[str, list[HealthReport]] = {}
        for report in reports:
            route = str(report.attributes.get(self.ATTR_ROUTE) or "<unknown>")
            by_route.setdefault(route, []).append(report)

        worst = OverallHealth.HEALTHY
        routes_body: dict[str, Any] = {}
        for route, route_reports in by_route.items():
            entry, status = self._render_route(route, route_reports)
            routes_body[route] = entry
            if _rank(status) > _rank(worst):
                worst = status

        return ViewSnapshot(
            status=worst,
            body={
                "sample_count": len(reports),
                "routes": routes_body,
                **rate_blocks,
            },
        )

    def _render_route(
        self,
        route: str,
        reports: list[HealthReport],
    ) -> tuple[dict[str, Any], OverallHealth]:
        by_agent: dict[str, list[HealthReport]] = {}
        for report in reports:
            agent = report.attributes.get(self.ATTR_AGENT)
            bucket = str(agent) if agent else SERVER_AGENT_BUCKET
            by_agent.setdefault(bucket, []).append(report)

        agents_body: dict[str, Any] = {}
        worst = OverallHealth.HEALTHY
        for bucket, agent_reports in by_agent.items():
            metrics = self._compute_metrics(agent_reports)
            status = self._classify(route=route, **metrics)
            agents_body[bucket] = self._render_metrics(metrics, status, len(agent_reports))
            if _rank(status) > _rank(worst):
                worst = status

        # Route-level metrics span all the route's reports; the route STATUS is
        # the worst of its agents so one failing agent can't hide in the rollup.
        route_metrics = self._compute_metrics(reports)
        entry = self._render_metrics(route_metrics, worst, len(reports))
        entry["agents"] = agents_body
        return entry, worst

    def _compute_metrics(self, reports: Sequence[HealthReport]) -> dict[str, Any]:
        successes = sum(1 for r in reports if bool(r.attributes.get(self.ATTR_SUCCESS)))
        sample_count = len(reports)
        success_rate = successes / sample_count if sample_count else 1.0

        latencies = [
            float(r.attributes[self.ATTR_LATENCY_MS])
            for r in reports
            if self.ATTR_LATENCY_MS in r.attributes
            and r.attributes[self.ATTR_LATENCY_MS] is not None
        ]

        error_counter: Counter[str] = Counter()
        for r in reports:
            if r.attributes.get(self.ATTR_SUCCESS):
                continue
            error_class = str(r.attributes.get(self.ATTR_ERROR_CLASS) or "Unknown")
            error_counter[error_class] += 1

        return {
            "success_rate": success_rate,
            "p50_ms": _percentile(latencies, 0.5),
            "p95_ms": _percentile(latencies, 0.95),
            "recent_errors": dict(error_counter.most_common(self._recent_errors_top_k)),
        }

    @staticmethod
    def _render_metrics(
        metrics: Mapping[str, Any],
        status: OverallHealth,
        sample_count: int,
    ) -> dict[str, Any]:
        return {
            "status": status.value,
            "success_rate": round(metrics["success_rate"], 4),
            "p50_latency_ms": round(metrics["p50_ms"], 2),
            "p95_latency_ms": round(metrics["p95_ms"], 2),
            "recent_errors": metrics["recent_errors"],
            "sample_count": sample_count,
        }

    def _render_rate_blocks(self) -> dict[str, Any]:
        if self._health_reporter is None:
            return {}
        try:
            return {
                "requests_per_minute": {
                    label: round(
                        self._health_reporter.counter_per_minute(NLP_REQUESTS_COUNTER, window),
                        4,
                    )
                    for label, window in _RATE_WINDOWS
                },
                "tokens_per_minute": {
                    label: round(
                        self._health_reporter.counter_per_minute(NLP_TOKENS_COUNTER, window),
                        4,
                    )
                    for label, window in _RATE_WINDOWS
                },
            }
        except KeyError:
            return {}

    def _classify(
        self,
        *,
        route: str,
        success_rate: float,
        p50_ms: float,
        p95_ms: float,
        recent_errors: Mapping[str, int] | None = None,
    ) -> OverallHealth:
        overrides = self._route_thresholds.get(route, RouteThresholds())

        deg_sr = (
            overrides.degraded_below_success_rate
            if overrides.degraded_below_success_rate is not None
            else self._degraded_below_success_rate
        )
        unh_sr = (
            overrides.unhealthy_below_success_rate
            if overrides.unhealthy_below_success_rate is not None
            else self._unhealthy_below_success_rate
        )
        deg_p50 = (
            overrides.degraded_p50_ms
            if overrides.degraded_p50_ms is not None
            else self._degraded_p50_ms
        )
        unh_p50 = (
            overrides.unhealthy_p50_ms
            if overrides.unhealthy_p50_ms is not None
            else self._unhealthy_p50_ms
        )
        deg_p95 = (
            overrides.degraded_p95_ms
            if overrides.degraded_p95_ms is not None
            else self._degraded_p95_ms
        )
        unh_p95 = (
            overrides.unhealthy_p95_ms
            if overrides.unhealthy_p95_ms is not None
            else self._unhealthy_p95_ms
        )

        if (
            success_rate < unh_sr
            or p95_ms >= unh_p95
            or (unh_p50 is not None and p50_ms >= unh_p50)
        ):
            return OverallHealth.UNHEALTHY
        if (
            success_rate < deg_sr
            or p95_ms >= deg_p95
            or (deg_p50 is not None and p50_ms >= deg_p50)
        ):
            return OverallHealth.DEGRADED
        return OverallHealth.HEALTHY


def _rank(status: OverallHealth) -> int:
    return {
        OverallHealth.HEALTHY: 0,
        OverallHealth.DEGRADED: 1,
        OverallHealth.UNHEALTHY: 2,
    }[status]
