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

"""HTTP-API view configuration for /healthz.

Owns the data shape of the ``/healthz`` response: which kinds are buffered
and for how long, which counters are tracked, and which views interpret the
data. ``HealthReporter`` itself is created at the binary layer and made
available through the container; everything else flows through here.
"""

from datetime import timedelta
from typing import Mapping

from lagom import Container

from parlant.core.event_loop_monitor import EventLoopMonitor
from parlant.core.health import (
    ENGINE_TTFM_KIND,
    ENGINE_TURN_KIND,
    ENGINE_TURNS_COUNTER,
    NLP_EMBED_KIND,
    NLP_REQUEST_KIND,
    NLP_REQUESTS_COUNTER,
    NLP_TOKENS_COUNTER,
    EngineHealthView,
    EventLoopHealthView,
    HealthReporter,
    NLPHealthView,
    ReportRetention,
    SchemaThresholds,
)


def _t(p50_deg: float, p50_unh: float, p95_deg: float, p95_unh: float) -> SchemaThresholds:
    """Build SchemaThresholds from seconds for readability."""
    return SchemaThresholds(
        degraded_p50_ms=p50_deg * 1000,
        unhealthy_p50_ms=p50_unh * 1000,
        degraded_p95_ms=p95_deg * 1000,
        unhealthy_p95_ms=p95_unh * 1000,
    )


DEFAULT_NLP_SCHEMA_THRESHOLDS: Mapping[str, SchemaThresholds] = {
    # Canned response (hot path)
    "CannedResponsePreambleSchema": _t(p50_deg=3, p50_unh=5, p95_deg=8, p95_unh=15),
    "CannedResponseDraftSchema": _t(p50_deg=10, p50_unh=15, p95_deg=25, p95_unh=45),
    "CannedResponseRevisionSchema": _t(p50_deg=10, p50_unh=15, p95_deg=25, p95_unh=45),
    "CannedResponseSelectionSchema": _t(p50_deg=5, p50_unh=10, p95_deg=15, p95_unh=20),
    "FollowUpCannedResponseSelectionSchema": _t(p50_deg=5, p50_unh=10, p95_deg=15, p95_unh=20),
    "CannedResponseFieldExtractionSchema": _t(p50_deg=5, p50_unh=10, p95_deg=15, p95_unh=20),
    "StreamingText": _t(p50_deg=5, p50_unh=8, p95_deg=10, p95_unh=15),
    # Guideline matching
    "GenericActionableGuidelineMatchesSchema": _t(p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20),
    "GenericLowCriticalityGuidelineMatchesSchema": _t(
        p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20
    ),
    "GenericObservationalGuidelineMatchesSchema": _t(p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20),
    "GenericPreviouslyAppliedActionableGuidelineMatchesSchema": _t(
        p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20
    ),
    "GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema": _t(
        p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20
    ),
    "DisambiguationGuidelineMatchesSchema": _t(p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20),
    # Tool calling
    "SingleToolBatchSchema": _t(p50_deg=15, p50_unh=20, p95_deg=25, p95_unh=30),
    "NonConsequentialToolBatchSchema": _t(p50_deg=10, p50_unh=15, p95_deg=20, p95_unh=25),
    "OverlappingToolsBatchSchema": _t(p50_deg=20, p50_unh=25, p95_deg=30, p95_unh=35),
    # Journey
    "JourneyNextStepSelectionSchema": _t(p50_deg=5, p50_unh=8, p95_deg=10, p95_unh=15),
    "JourneyBacktrackCheckSchema": _t(p50_deg=5, p50_unh=8, p95_deg=10, p95_unh=15),
    "JourneyBacktrackNodeSelectionSchema": _t(p50_deg=8, p50_unh=12, p95_deg=15, p95_unh=20),
    # Response analysis
    "GenericResponseAnalysisSchema": _t(p50_deg=10, p50_unh=15, p95_deg=20, p95_unh=25),
}


def configure_healthz(container: Container) -> None:
    """Configure the ``HealthReporter`` for the ``/healthz`` response.

    Sets up retention buffers and rate counters for the kinds the API
    surface cares about, then registers the views that interpret them.
    Expects ``HealthReporter`` and ``EventLoopMonitor`` to already be
    available on the container.
    """
    health_reporter = container[HealthReporter]

    health_reporter.configure_retention(
        NLP_REQUEST_KIND,
        ReportRetention(window=timedelta(minutes=10), max_count=10_000),
    )
    health_reporter.configure_retention(
        NLP_EMBED_KIND,
        ReportRetention(window=timedelta(minutes=10), max_count=10_000),
    )
    health_reporter.configure_counter(NLP_REQUESTS_COUNTER, retention=timedelta(days=1))
    health_reporter.configure_counter(NLP_TOKENS_COUNTER, retention=timedelta(days=1))

    health_reporter.configure_retention(
        ENGINE_TURN_KIND,
        ReportRetention(window=timedelta(minutes=10), max_count=10_000),
    )
    health_reporter.configure_retention(
        ENGINE_TTFM_KIND,
        ReportRetention(window=timedelta(minutes=10), max_count=10_000),
    )
    health_reporter.configure_counter(ENGINE_TURNS_COUNTER, retention=timedelta(days=1))

    health_reporter.register_view(
        NLPHealthView(
            health_reporter=health_reporter,
            schema_thresholds=DEFAULT_NLP_SCHEMA_THRESHOLDS,
        )
    )
    health_reporter.register_view(EngineHealthView(health_reporter=health_reporter))
    health_reporter.register_view(EventLoopHealthView(container[EventLoopMonitor]))
