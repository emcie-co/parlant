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

"""Generic health reporting infrastructure.

The ``HealthReporter`` collects timestamped reports keyed by ``kind`` and
exposes them to registered ``HealthView`` objects, which interpret reports
into snapshot sections served by the ``/healthz`` endpoint.

The write side is deliberately generic: any subsystem can call
``report(kind, attributes)`` without knowing how the data will be rolled
up. The read side is structured: each registered view declares which kinds
it consumes and how to render them, including a status that contributes
to the overall worst-of rollup when the view is marked as critical.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Mapping, Protocol, Sequence

from parlant.core.application_context import ApplicationContext


class StatusCriticality(str, Enum):
    """Whether a view's status contributes to the overall ``/healthz`` rollup."""

    CRITICAL = "critical"
    INFORMATIONAL = "informational"


class OverallHealth(str, Enum):
    """Health status convention used across views and the top-level rollup."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


_RANK = {
    OverallHealth.HEALTHY: 0,
    OverallHealth.DEGRADED: 1,
    OverallHealth.UNHEALTHY: 2,
}


@dataclass(frozen=True)
class HealthReport:
    """A single timestamped observation of some kind."""

    kind: str
    timestamp: datetime
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class ReportRetention:
    """Per-kind retention policy.

    ``window`` bounds entries by age. ``max_count`` bounds entries by total
    number, providing a memory ceiling regardless of write rate.
    """

    window: timedelta
    max_count: int


@dataclass(frozen=True)
class ViewSnapshot:
    """The output of rendering a view at one point in time."""

    status: OverallHealth
    body: Mapping[str, Any]


class HealthView(Protocol):
    """A read-side view that interprets reports of one or more kinds."""

    name: str
    criticality: StatusCriticality
    kinds: tuple[str, ...]

    def render(
        self,
        reports_by_kind: Mapping[str, Sequence[HealthReport]],
    ) -> ViewSnapshot: ...


_DEFAULT_SNAPSHOT_CACHE_TTL = timedelta(minutes=5)

_COUNTER_BUCKET_SECONDS = 10


class RollingCounter:
    """Per-bucket rolling counter for cheap windowed sums.

    Pre-aggregates increments into 10-second buckets so windowed-sum queries
    cost O(buckets-in-window) rather than O(events-in-window). Suitable for
    request-rate / token-rate telemetry over windows up to one day.
    """

    def __init__(self, retention: timedelta) -> None:
        self._retention = retention
        self._buckets: dict[int, int] = {}
        self._lock = Lock()

    def increment(self, amount: int, *, at: datetime | None = None) -> None:
        ts = at if at is not None else datetime.now(timezone.utc)
        bucket_epoch = (int(ts.timestamp()) // _COUNTER_BUCKET_SECONDS) * _COUNTER_BUCKET_SECONDS
        with self._lock:
            self._buckets[bucket_epoch] = self._buckets.get(bucket_epoch, 0) + amount
            self._prune(ts)

    def sum_in_window(self, window: timedelta, *, now: datetime | None = None) -> int:
        ts = now if now is not None else datetime.now(timezone.utc)
        cutoff = int(ts.timestamp()) - int(window.total_seconds())
        with self._lock:
            self._prune(ts)
            return sum(
                count
                for epoch, count in self._buckets.items()
                if epoch + _COUNTER_BUCKET_SECONDS > cutoff
            )

    def per_minute(self, window: timedelta, *, now: datetime | None = None) -> float:
        minutes = window.total_seconds() / 60.0
        if minutes <= 0:
            return 0.0
        return self.sum_in_window(window, now=now) / minutes

    def _prune(self, now_ts: datetime) -> None:
        cutoff = int(now_ts.timestamp()) - int(self._retention.total_seconds())
        old_keys = [k for k in self._buckets if k + _COUNTER_BUCKET_SECONDS <= cutoff]
        for k in old_keys:
            del self._buckets[k]


class HealthReporter:
    """Collects health reports and renders them through registered views.

    ``snapshot()`` results are cached for ``snapshot_cache_ttl`` so that
    high-frequency probes against ``/healthz`` don't pay the rendering
    cost on every call. Reports continue to be recorded during the cache
    window — only the rollup computation is deferred.
    """

    def __init__(
        self,
        application_context: ApplicationContext,
        *,
        snapshot_cache_ttl: timedelta = _DEFAULT_SNAPSHOT_CACHE_TTL,
    ) -> None:
        self._application_context = application_context
        self._retention: dict[str, ReportRetention] = {}
        self._buffers: dict[str, deque[HealthReport]] = defaultdict(deque)
        self._counters: dict[str, RollingCounter] = {}
        self._views: list[HealthView] = []
        self._lock = Lock()
        self._snapshot_cache_ttl = snapshot_cache_ttl
        self._cached_snapshot: dict[str, Any] | None = None
        self._cached_snapshot_at: datetime | None = None

    def configure_retention(self, kind: str, retention: ReportRetention) -> None:
        """Configure how long and how many reports of ``kind`` to retain."""
        with self._lock:
            self._retention[kind] = retention

    def configure_counter(self, name: str, retention: timedelta) -> None:
        """Configure a named rolling counter with the given retention window."""
        with self._lock:
            self._counters[name] = RollingCounter(retention=retention)

    def increment_counter(self, name: str, amount: int) -> None:
        """Increment a configured counter. Raises if the counter is unknown."""
        with self._lock:
            counter = self._counters.get(name)
            if counter is None:
                raise KeyError(f"No counter configured named '{name}'")
        counter.increment(amount)

    def counter_sum(self, name: str, window: timedelta) -> int:
        with self._lock:
            counter = self._counters.get(name)
            if counter is None:
                raise KeyError(f"No counter configured named '{name}'")
        return counter.sum_in_window(window)

    def counter_per_minute(self, name: str, window: timedelta) -> float:
        with self._lock:
            counter = self._counters.get(name)
            if counter is None:
                raise KeyError(f"No counter configured named '{name}'")
        return counter.per_minute(window)

    def register_view(self, view: HealthView) -> None:
        """Register a view that participates in ``snapshot()`` output."""
        with self._lock:
            self._views.append(view)

    def report(self, kind: str, attributes: Mapping[str, Any]) -> None:
        """Record an observation of ``kind``.

        Raises ``KeyError`` if no retention has been configured for ``kind`` —
        misconfigured callers should fail loudly rather than silently writing
        to an unbounded buffer.
        """
        report = HealthReport(
            kind=kind,
            timestamp=datetime.now(timezone.utc),
            attributes=dict(attributes),
        )

        with self._lock:
            if kind not in self._retention:
                raise KeyError(f"No retention configured for health report kind '{kind}'")

            buffer = self._buffers[kind]
            buffer.append(report)
            self._prune(kind, buffer)

    def snapshot(self) -> dict[str, Any]:
        """Render every view and compute the overall worst-of-critical status.

        Results are cached for ``snapshot_cache_ttl``. While the cached
        snapshot is fresh, this method returns it without touching the
        report buffers, so repeated polling of ``/healthz`` does not
        impose rollup cost on the running process.
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            if (
                self._cached_snapshot is not None
                and self._cached_snapshot_at is not None
                and now - self._cached_snapshot_at < self._snapshot_cache_ttl
            ):
                return self._cached_snapshot

            self._prune_all_for_age()
            views = list(self._views)
            buffers_by_view: list[tuple[HealthView, dict[str, list[HealthReport]]]] = []
            for view in views:
                by_kind = {kind: list(self._buffers.get(kind, ())) for kind in view.kinds}
                buffers_by_view.append((view, by_kind))

        overall = OverallHealth.HEALTHY
        checks: dict[str, Any] = {}
        for view, by_kind in buffers_by_view:
            rendered = view.render(by_kind)
            checks[view.name] = {"status": rendered.status.value, **rendered.body}
            if view.criticality is StatusCriticality.CRITICAL:
                if _RANK[rendered.status] > _RANK[overall]:
                    overall = rendered.status

        result = {
            "instance_id": self._application_context.instance_id,
            "status": overall.value,
            "checks": checks,
        }

        with self._lock:
            self._cached_snapshot = result
            self._cached_snapshot_at = now

        return result

    def _prune(self, kind: str, buffer: deque[HealthReport]) -> None:
        retention = self._retention[kind]
        cutoff = datetime.now(timezone.utc) - retention.window
        while buffer and buffer[0].timestamp < cutoff:
            buffer.popleft()
        while len(buffer) > retention.max_count:
            buffer.popleft()

    def _prune_all_for_age(self) -> None:
        for kind, buffer in self._buffers.items():
            if kind in self._retention:
                self._prune(kind, buffer)


class NullHealthReporter(HealthReporter):
    """No-op HealthReporter for contexts that don't need observability.

    Accepts every ``report`` / ``increment_counter`` call regardless of
    configuration and discards it. Useful in tests that exercise NLP /
    engine paths without caring about ``/healthz``: it lets the producer
    code run unmodified while keeping the production reporter strict.
    """

    def report(self, kind: str, attributes: Mapping[str, Any]) -> None:
        return None

    def increment_counter(self, name: str, amount: int) -> None:
        return None
