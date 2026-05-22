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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from parlant.core.application_context import ApplicationContext
from parlant.core.health import (
    StatusCriticality,
    HealthReport,
    HealthReporter,
    NLP_EMBED_KIND,
    NLP_REQUEST_KIND,
    NLPHealthView,
    OverallHealth,
    ReportRetention,
    ViewSnapshot,
)
from parlant.core.health.nlp_view import SchemaThresholds
from parlant.core.health.reporter import RollingCounter

_TEST_APP_CONTEXT = ApplicationContext(instance_id="test-instance")


@dataclass
class _RecordingView:
    """Minimal HealthView used to inspect what the reporter delivers."""

    name: str
    kinds: tuple[str, ...]
    criticality: StatusCriticality = StatusCriticality.CRITICAL
    fixed_status: OverallHealth = OverallHealth.HEALTHY
    last_seen: dict[str, Sequence[HealthReport]] = field(default_factory=dict)

    def render(
        self,
        reports_by_kind: Mapping[str, Sequence[HealthReport]],
    ) -> ViewSnapshot:
        self.last_seen = {k: list(v) for k, v in reports_by_kind.items()}
        return ViewSnapshot(
            status=self.fixed_status,
            body={"sample_count": sum(len(v) for v in reports_by_kind.values())},
        )


def _retention(seconds: float = 60.0, max_count: int = 1000) -> ReportRetention:
    return ReportRetention(window=timedelta(seconds=seconds), max_count=max_count)


async def test_that_a_reported_kind_is_visible_to_views_consuming_that_kind() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    reporter.configure_retention("nlp.request", _retention())
    view = _RecordingView(name="nlp", kinds=("nlp.request",))
    reporter.register_view(view)

    reporter.report("nlp.request", {"schema": "S", "success": True, "latency_ms": 10.0})
    reporter.report("nlp.request", {"schema": "S", "success": True, "latency_ms": 20.0})

    snapshot = reporter.snapshot()

    assert "nlp" in snapshot["checks"]
    assert snapshot["checks"]["nlp"]["sample_count"] == 2
    assert len(view.last_seen["nlp.request"]) == 2


async def test_that_a_view_does_not_see_kinds_it_did_not_declare() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    reporter.configure_retention("nlp.request", _retention())
    reporter.configure_retention("doc_store.query", _retention())

    nlp_view = _RecordingView(name="nlp", kinds=("nlp.request",))
    reporter.register_view(nlp_view)

    reporter.report("nlp.request", {"x": 1})
    reporter.report("doc_store.query", {"y": 2})

    reporter.snapshot()

    assert list(nlp_view.last_seen.keys()) == ["nlp.request"]
    assert len(nlp_view.last_seen["nlp.request"]) == 1


async def test_that_reports_older_than_the_window_are_pruned_on_next_write() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    reporter.configure_retention(
        "k", ReportRetention(window=timedelta(milliseconds=50), max_count=1000)
    )
    view = _RecordingView(name="v", kinds=("k",))
    reporter.register_view(view)

    reporter.report("k", {"i": 1})
    reporter.report("k", {"i": 2})

    await asyncio.sleep(0.12)

    reporter.report("k", {"i": 3})

    reporter.snapshot()

    seen = view.last_seen["k"]
    assert [r.attributes["i"] for r in seen] == [3]


async def test_that_reports_beyond_max_count_are_pruned_oldest_first() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    reporter.configure_retention("k", ReportRetention(window=timedelta(seconds=60), max_count=3))
    view = _RecordingView(name="v", kinds=("k",))
    reporter.register_view(view)

    for i in range(5):
        reporter.report("k", {"i": i})

    reporter.snapshot()

    seen = view.last_seen["k"]
    assert [r.attributes["i"] for r in seen] == [2, 3, 4]


async def test_that_each_kind_has_independent_retention() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    reporter.configure_retention("a", ReportRetention(window=timedelta(seconds=60), max_count=2))
    reporter.configure_retention("b", ReportRetention(window=timedelta(seconds=60), max_count=10))
    view = _RecordingView(name="v", kinds=("a", "b"))
    reporter.register_view(view)

    for i in range(5):
        reporter.report("a", {"i": i})
    for i in range(5):
        reporter.report("b", {"i": i})

    reporter.snapshot()

    assert [r.attributes["i"] for r in view.last_seen["a"]] == [3, 4]
    assert [r.attributes["i"] for r in view.last_seen["b"]] == [0, 1, 2, 3, 4]


async def test_that_overall_status_is_worst_of_critical_views() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT, snapshot_cache_ttl=timedelta(0))
    reporter.configure_retention("k", _retention())

    healthy = _RecordingView(name="healthy", kinds=("k",), fixed_status=OverallHealth.HEALTHY)
    degraded = _RecordingView(name="degraded", kinds=("k",), fixed_status=OverallHealth.DEGRADED)
    unhealthy = _RecordingView(name="unhealthy", kinds=("k",), fixed_status=OverallHealth.UNHEALTHY)

    reporter.register_view(healthy)
    reporter.register_view(degraded)
    snapshot = reporter.snapshot()
    assert snapshot["status"] == "degraded"

    reporter.register_view(unhealthy)
    snapshot = reporter.snapshot()
    assert snapshot["status"] == "unhealthy"


async def test_that_snapshot_is_cached_within_the_cache_ttl() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT, snapshot_cache_ttl=timedelta(seconds=60))
    reporter.configure_retention("k", _retention())
    view = _RecordingView(name="v", kinds=("k",), fixed_status=OverallHealth.HEALTHY)
    reporter.register_view(view)

    reporter.report("k", {"i": 1})
    first = reporter.snapshot()
    first_render_count = sum(len(v) for v in view.last_seen.values())

    # A second report and a re-fetch within the cache window must NOT trigger
    # a re-render, and the cached body must be returned verbatim.
    reporter.report("k", {"i": 2})
    second = reporter.snapshot()
    second_render_count = sum(len(v) for v in view.last_seen.values())

    assert second is first
    assert second_render_count == first_render_count


async def test_that_snapshot_recomputes_after_cache_ttl_expires() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT, snapshot_cache_ttl=timedelta(milliseconds=50))
    reporter.configure_retention("k", _retention())
    view = _RecordingView(name="v", kinds=("k",), fixed_status=OverallHealth.HEALTHY)
    reporter.register_view(view)

    reporter.report("k", {"i": 1})
    first = reporter.snapshot()
    assert first["checks"]["v"]["sample_count"] == 1

    reporter.report("k", {"i": 2})
    await asyncio.sleep(0.12)

    second = reporter.snapshot()
    assert second is not first
    assert second["checks"]["v"]["sample_count"] == 2


async def test_that_informational_views_appear_in_body_but_do_not_affect_overall_status() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    reporter.configure_retention("k", _retention())

    critical_healthy = _RecordingView(
        name="critical_healthy",
        kinds=("k",),
        criticality=StatusCriticality.CRITICAL,
        fixed_status=OverallHealth.HEALTHY,
    )
    informational_unhealthy = _RecordingView(
        name="cost",
        kinds=("k",),
        criticality=StatusCriticality.INFORMATIONAL,
        fixed_status=OverallHealth.UNHEALTHY,
    )

    reporter.register_view(critical_healthy)
    reporter.register_view(informational_unhealthy)

    snapshot = reporter.snapshot()

    assert snapshot["status"] == "healthy"
    assert snapshot["checks"]["cost"]["status"] == "unhealthy"


async def test_that_concurrent_reports_from_many_coroutines_are_all_recorded() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    reporter.configure_retention(
        "k", ReportRetention(window=timedelta(seconds=60), max_count=10_000)
    )
    view = _RecordingView(name="v", kinds=("k",))
    reporter.register_view(view)

    async def writer(start: int, count: int) -> None:
        for i in range(start, start + count):
            reporter.report("k", {"i": i})
            await asyncio.sleep(0)

    await asyncio.gather(*(writer(w * 100, 100) for w in range(10)))

    reporter.snapshot()

    seen_values = {r.attributes["i"] for r in view.last_seen["k"]}
    assert len(seen_values) == 1000
    assert seen_values == set(range(1000))


async def test_that_a_kind_without_configured_retention_raises_on_report() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)

    raised: type[BaseException] | None = None
    try:
        reporter.report("unconfigured", {"x": 1})
    except Exception as e:  # noqa: BLE001
        raised = type(e)

    assert raised is not None, "expected reporting an unconfigured kind to raise"


def _attrs(reports: Sequence[HealthReport]) -> list[Mapping[str, Any]]:
    return [r.attributes for r in reports]


def test_that_a_rolling_counter_returns_zero_when_no_increments_have_happened() -> None:
    counter = RollingCounter(retention=timedelta(days=1))
    assert counter.sum_in_window(timedelta(minutes=1)) == 0
    assert counter.per_minute(timedelta(minutes=5)) == 0.0


def test_that_a_rolling_counter_sums_increments_within_the_query_window() -> None:
    counter = RollingCounter(retention=timedelta(days=1))
    counter.increment(100)
    counter.increment(50)
    counter.increment(25)
    assert counter.sum_in_window(timedelta(minutes=1)) == 175


def test_that_a_rolling_counter_excludes_increments_older_than_the_query_window() -> None:
    counter = RollingCounter(retention=timedelta(days=1))
    now = datetime.now(timezone.utc)

    counter.increment(1000, at=now - timedelta(minutes=10))
    counter.increment(7, at=now - timedelta(seconds=30))

    assert counter.sum_in_window(timedelta(minutes=1), now=now) == 7
    assert counter.sum_in_window(timedelta(minutes=15), now=now) == 1007


def test_that_a_rolling_counter_prunes_buckets_older_than_its_retention() -> None:
    counter = RollingCounter(retention=timedelta(minutes=5))
    now = datetime.now(timezone.utc)

    counter.increment(999, at=now - timedelta(minutes=30))
    counter.increment(3, at=now)

    assert counter.sum_in_window(timedelta(days=1), now=now) == 3


def test_that_a_rolling_counter_per_minute_normalizes_by_window_minutes() -> None:
    counter = RollingCounter(retention=timedelta(days=1))
    now = datetime.now(timezone.utc)

    for i in range(60):
        counter.increment(10, at=now - timedelta(seconds=i))

    rate = counter.per_minute(timedelta(minutes=1), now=now)
    assert rate == 600.0


def test_that_health_reporter_exposes_configured_counters() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    reporter.configure_counter("nlp.tokens", retention=timedelta(days=1))

    reporter.increment_counter("nlp.tokens", 1234)
    reporter.increment_counter("nlp.tokens", 66)

    assert reporter.counter_sum("nlp.tokens", timedelta(minutes=1)) == 1300
    assert reporter.counter_per_minute("nlp.tokens", timedelta(minutes=1)) == 1300.0


def test_that_incrementing_an_unconfigured_counter_raises() -> None:
    reporter = HealthReporter(_TEST_APP_CONTEXT)
    raised: type[BaseException] | None = None
    try:
        reporter.increment_counter("ghost", 1)
    except Exception as e:  # noqa: BLE001
        raised = type(e)
    assert raised is not None


def _nlp_request_report(
    schema: str,
    *,
    success: bool = True,
    latency_ms: float = 10.0,
) -> HealthReport:
    return HealthReport(
        kind=NLP_REQUEST_KIND,
        timestamp=datetime.now(timezone.utc),
        attributes={
            NLPHealthView.ATTR_SCHEMA: schema,
            NLPHealthView.ATTR_MODEL: "test",
            NLPHealthView.ATTR_SUCCESS: success,
            NLPHealthView.ATTR_LATENCY_MS: latency_ms,
            NLPHealthView.ATTR_ERROR_CLASS: None,
        },
    )


def _nlp_embed_report(
    embedder: str,
    *,
    success: bool = True,
    latency_ms: float = 5.0,
) -> HealthReport:
    return HealthReport(
        kind=NLP_EMBED_KIND,
        timestamp=datetime.now(timezone.utc),
        attributes={
            NLPHealthView.ATTR_SCHEMA: embedder,
            NLPHealthView.ATTR_MODEL: "test",
            NLPHealthView.ATTR_SUCCESS: success,
            NLPHealthView.ATTR_LATENCY_MS: latency_ms,
            NLPHealthView.ATTR_ERROR_CLASS: None,
        },
    )


def test_that_nlp_view_separates_embedder_reports_from_schemas() -> None:
    view = NLPHealthView()

    snapshot = view.render(
        {
            NLP_REQUEST_KIND: [_nlp_request_report("MySchema")],
            NLP_EMBED_KIND: [_nlp_embed_report("MyEmbedder")],
        }
    )

    assert "schemas" in snapshot.body
    assert "embedders" in snapshot.body
    assert "MySchema" in snapshot.body["schemas"]
    assert "MyEmbedder" in snapshot.body["embedders"]
    assert "MyEmbedder" not in snapshot.body["schemas"]
    assert "MySchema" not in snapshot.body["embedders"]


def test_that_nlp_view_uses_per_schema_p95_thresholds_when_configured() -> None:
    view = NLPHealthView(
        schema_thresholds={
            "FastSchema": SchemaThresholds(
                degraded_p95_ms=100.0,
                unhealthy_p95_ms=300.0,
            ),
        }
    )

    # Latencies whose p95 exceeds the per-schema 100ms degraded threshold
    # but stays below the 300ms unhealthy threshold.
    reports = [_nlp_request_report("FastSchema", latency_ms=ms) for ms in [10, 20, 30, 40, 200]]

    snapshot = view.render({NLP_REQUEST_KIND: reports, NLP_EMBED_KIND: []})

    assert snapshot.body["schemas"]["FastSchema"]["status"] == "degraded"


def test_that_nlp_view_uses_per_schema_p50_thresholds_when_configured() -> None:
    view = NLPHealthView(
        schema_thresholds={
            "ChattySchema": SchemaThresholds(
                degraded_p50_ms=50.0,
                unhealthy_p50_ms=200.0,
            ),
        }
    )

    # Median latency = 75ms, which crosses the per-schema 50ms p50-degraded threshold.
    reports = [_nlp_request_report("ChattySchema", latency_ms=ms) for ms in [70, 75, 80]]

    snapshot = view.render({NLP_REQUEST_KIND: reports, NLP_EMBED_KIND: []})

    assert snapshot.body["schemas"]["ChattySchema"]["status"] == "degraded"


def test_that_nlp_view_falls_back_to_default_thresholds_for_unconfigured_schemas() -> None:
    view = NLPHealthView(
        degraded_p95_ms=10_000.0,
        unhealthy_p95_ms=20_000.0,
        schema_thresholds={
            "ConfiguredSchema": SchemaThresholds(degraded_p95_ms=50.0, unhealthy_p95_ms=100.0),
        },
    )

    # 1500ms p95 is well below the 10s default degraded threshold,
    # so an unconfigured schema should remain healthy.
    reports = [_nlp_request_report("UnconfiguredSchema", latency_ms=ms) for ms in [100, 200, 1500]]

    snapshot = view.render({NLP_REQUEST_KIND: reports, NLP_EMBED_KIND: []})

    assert snapshot.body["schemas"]["UnconfiguredSchema"]["status"] == "healthy"
