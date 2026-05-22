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

Aggregates ``nlp.request`` and ``nlp.embed`` reports into a per-schema
status with success rate, latency percentiles, and a recent-error
breakdown. The view's status is the worst across all observed schemas.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, Sequence

from parlant.core.health.reporter import (
    StatusCriticality,
    HealthReport,
    HealthReporter,
    OverallHealth,
    ViewSnapshot,
)


@dataclass(frozen=True)
class SchemaThresholds:
    """Per-schema overrides for ``NLPHealthView`` classification.

    Any field left as ``None`` falls back to the view's default for that
    threshold, so callers only need to specify the thresholds they care
    about for a given schema.
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
    """Renders NLP health grouped by schema.

    Status thresholds are constructor parameters so deployments can tune
    them without changing the view itself. Defaults err toward leniency
    so transient blips don't pull a pod out of rotation.
    """

    name = "nlp"
    criticality = StatusCriticality.CRITICAL
    kinds: tuple[str, ...] = (NLP_REQUEST_KIND, NLP_EMBED_KIND)

    # Report attribute keys — producers and the renderer share these.
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
        schema_thresholds: Mapping[str, SchemaThresholds] | None = None,
    ) -> None:
        self._health_reporter = health_reporter
        self._degraded_below_success_rate = degraded_below_success_rate
        self._unhealthy_below_success_rate = unhealthy_below_success_rate
        self._degraded_p50_ms = degraded_p50_ms
        self._unhealthy_p50_ms = unhealthy_p50_ms
        self._degraded_p95_ms = degraded_p95_ms
        self._unhealthy_p95_ms = unhealthy_p95_ms
        self._recent_errors_top_k = recent_errors_top_k
        self._schema_thresholds: Mapping[str, SchemaThresholds] = dict(schema_thresholds or {})

    def render(
        self,
        reports_by_kind: Mapping[str, Sequence[HealthReport]],
    ) -> ViewSnapshot:
        request_reports = list(reports_by_kind.get(NLP_REQUEST_KIND, ()))
        embed_reports = list(reports_by_kind.get(NLP_EMBED_KIND, ()))

        rate_blocks = self._render_rate_blocks()
        total_count = len(request_reports) + len(embed_reports)

        worst = OverallHealth.HEALTHY

        schemas_body, worst_schema = self._render_grouped(
            request_reports,
            classify=lambda name, **kw: self._classify(schema=name, **kw),
        )
        if _rank(worst_schema) > _rank(worst):
            worst = worst_schema

        embedders_body, worst_embedder = self._render_grouped(
            embed_reports,
            classify=lambda name, **kw: self._classify(schema=None, **kw),
        )
        if _rank(worst_embedder) > _rank(worst):
            worst = worst_embedder

        return ViewSnapshot(
            status=worst,
            body={
                "sample_count": total_count,
                "schemas": schemas_body,
                "embedders": embedders_body,
                **rate_blocks,
            },
        )

    def _render_grouped(
        self,
        reports: Sequence[HealthReport],
        *,
        classify: Any,
    ) -> tuple[dict[str, Any], OverallHealth]:
        groups: dict[str, list[HealthReport]] = {}
        for report in reports:
            name = str(report.attributes.get(self.ATTR_SCHEMA, "<unknown>"))
            groups.setdefault(name, []).append(report)

        body: dict[str, Any] = {}
        worst = OverallHealth.HEALTHY
        for name, group_reports in groups.items():
            entry, status = self._render_group(name, group_reports, classify=classify)
            body[name] = entry
            if _rank(status) > _rank(worst):
                worst = status
        return body, worst

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

    def _render_group(
        self,
        name: str,
        reports: list[HealthReport],
        *,
        classify: Any,
    ) -> tuple[dict[str, Any], OverallHealth]:
        successes = sum(1 for r in reports if bool(r.attributes.get(self.ATTR_SUCCESS)))
        sample_count = len(reports)
        success_rate = successes / sample_count if sample_count else 1.0

        latencies = [
            float(r.attributes[self.ATTR_LATENCY_MS])
            for r in reports
            if self.ATTR_LATENCY_MS in r.attributes
            and r.attributes[self.ATTR_LATENCY_MS] is not None
        ]
        p50 = _percentile(latencies, 0.5)
        p95 = _percentile(latencies, 0.95)

        error_counter: Counter[str] = Counter()
        for r in reports:
            if r.attributes.get(self.ATTR_SUCCESS):
                continue
            error_class = str(r.attributes.get(self.ATTR_ERROR_CLASS) or "Unknown")
            error_counter[error_class] += 1
        recent_errors = dict(error_counter.most_common(self._recent_errors_top_k))

        status = classify(name, success_rate=success_rate, p50_ms=p50, p95_ms=p95)

        return {
            "status": status.value,
            "success_rate": round(success_rate, 4),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "recent_errors": recent_errors,
            "sample_count": sample_count,
        }, status

    def _classify(
        self,
        *,
        schema: str | None,
        success_rate: float,
        p50_ms: float,
        p95_ms: float,
    ) -> OverallHealth:
        overrides = (
            self._schema_thresholds.get(schema, SchemaThresholds())
            if schema
            else SchemaThresholds()
        )

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
