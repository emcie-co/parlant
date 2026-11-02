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
from parlant.core.container import Container

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
    TunnelHealthView,
)
from parlant.core.health.thresholds import HealthThresholds
from parlant.core.tunnels import TunnelService


def configure_healthz(container: Container) -> None:
    """Configure the ``HealthReporter`` for the ``/healthz`` response.

    Sets up retention buffers and rate counters for the kinds the API
    surface cares about, then registers the views that interpret them.
    Expects ``HealthReporter``, ``EventLoopMonitor`` and ``HealthThresholds``
    to already be available on the container.
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

    thresholds = container[HealthThresholds]

    health_reporter.register_view(
        NLPHealthView(
            health_reporter=health_reporter,
            degraded_below_success_rate=thresholds.nlp.degraded_below_success_rate,
            unhealthy_below_success_rate=thresholds.nlp.unhealthy_below_success_rate,
            degraded_p50_ms=thresholds.nlp.degraded_p50_ms,
            unhealthy_p50_ms=thresholds.nlp.unhealthy_p50_ms,
            degraded_p95_ms=thresholds.nlp.degraded_p95_ms,
            unhealthy_p95_ms=thresholds.nlp.unhealthy_p95_ms,
            route_thresholds=thresholds.nlp.routes,
        )
    )
    health_reporter.register_view(
        EngineHealthView(
            health_reporter=health_reporter,
            degraded_below_success_rate=thresholds.engine.degraded_below_success_rate,
            unhealthy_below_success_rate=thresholds.engine.unhealthy_below_success_rate,
            degraded_p95_ms=thresholds.engine.degraded_p95_ms,
            unhealthy_p95_ms=thresholds.engine.unhealthy_p95_ms,
        )
    )
    health_reporter.register_view(EventLoopHealthView(container[EventLoopMonitor]))

    # Resolved on every render rather than once here. The tunnel is bound
    # during `initialize_container`, which runs after this, so a service
    # captured now would always be absent -- and every managed project would
    # report its tunnel permanently off, which is worse than not reporting it.
    def read_tunnel() -> TunnelService | None:
        if TunnelService not in container.defined_types:
            return None
        return container[TunnelService]

    health_reporter.register_view(TunnelHealthView(read_tunnel))
