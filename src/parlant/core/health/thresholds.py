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

"""The thresholds behind the ``/healthz`` status.

Every check that can turn the top-level status ``degraded`` or ``unhealthy``
reads its cutoffs from one ``HealthThresholds`` value bound on the container.
A deployment tunes them by binding its own value in ``configure_container``;
the defaults reproduce Parlant's built-in behavior exactly.
"""

from dataclasses import dataclass, field
from typing import Mapping

from parlant.core.event_loop_monitor import EventLoopHealthThresholds
from parlant.core.health.nlp_view import RouteThresholds

__all__ = [
    "EngineHealthThresholds",
    "EventLoopHealthThresholds",
    "HealthThresholds",
    "NLPHealthThresholds",
    "default_nlp_route_thresholds",
]


def _seconds(p50_deg: float, p50_unh: float, p95_deg: float, p95_unh: float) -> RouteThresholds:
    """Build RouteThresholds from seconds for readability."""
    return RouteThresholds(
        degraded_p50_ms=p50_deg * 1000,
        unhealthy_p50_ms=p50_unh * 1000,
        degraded_p95_ms=p95_deg * 1000,
        unhealthy_p95_ms=p95_unh * 1000,
    )


def default_nlp_route_thresholds() -> Mapping[str, RouteThresholds]:
    """Per-route latency cutoffs for the model routes on the response path.

    Routes not listed here (compaction, indexing, and other background work)
    fall back to the lenient view-wide defaults.
    """
    return {
        # Rule matching (hot path)
        "matching.rank": _seconds(p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20),
        "matching.distill": _seconds(p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20),
        # Message/tool review sits between generation and delivery
        "review.message": _seconds(p50_deg=10, p50_unh=15, p95_deg=20, p95_unh=25),
        "review.tools": _seconds(p50_deg=10, p50_unh=15, p95_deg=20, p95_unh=25),
    }


@dataclass(frozen=True)
class EngineHealthThresholds:
    """Cutoffs for the ``engine`` check, measured over recent turns.

    Success rate is the share of turns that completed without error; latency
    is the p95 of turn duration in milliseconds.
    """

    degraded_below_success_rate: float = 0.95
    unhealthy_below_success_rate: float = 0.80
    degraded_p95_ms: float = 30_000.0
    unhealthy_p95_ms: float = 60_000.0


@dataclass(frozen=True)
class NLPHealthThresholds:
    """Cutoffs for the ``nlp`` check, measured per model route.

    The view-wide values apply to every route. ``routes`` overrides them for
    named routes; a field left ``None`` in an override keeps the view-wide
    value. A ``None`` p50 cutoff disables that comparison.
    """

    degraded_below_success_rate: float = 0.95
    unhealthy_below_success_rate: float = 0.80
    degraded_p50_ms: float | None = 20_000.0
    unhealthy_p50_ms: float | None = 30_000.0
    degraded_p95_ms: float = 45_000.0
    unhealthy_p95_ms: float = 60_000.0
    routes: Mapping[str, RouteThresholds] = field(default_factory=default_nlp_route_thresholds)


@dataclass(frozen=True)
class HealthThresholds:
    """Everything that decides the ``/healthz`` status, in one bindable value."""

    engine: EngineHealthThresholds = EngineHealthThresholds()
    nlp: NLPHealthThresholds = NLPHealthThresholds()
    event_loop: EventLoopHealthThresholds = EventLoopHealthThresholds()
