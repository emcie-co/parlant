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

"""HealthView wrapper around ``EventLoopMonitor``."""

from typing import Mapping, Sequence

from parlant.core.event_loop_monitor import EventLoopHealth, EventLoopMonitor
from parlant.core.health.reporter import (
    StatusCriticality,
    HealthReport,
    OverallHealth,
    ViewSnapshot,
)


_HEALTH_TO_OVERALL = {
    EventLoopHealth.HEALTHY: OverallHealth.HEALTHY,
    EventLoopHealth.DEGRADED: OverallHealth.DEGRADED,
    EventLoopHealth.UNHEALTHY: OverallHealth.UNHEALTHY,
}


class EventLoopHealthView:
    """Live view that reads ``EventLoopMonitor.status`` directly.

    Doesn't consume any reports — the monitor maintains its own state,
    and this view just adapts that state into the HealthReporter rollup.
    """

    name = "event_loop"
    criticality = StatusCriticality.CRITICAL
    kinds: tuple[str, ...] = ()

    def __init__(self, monitor: EventLoopMonitor) -> None:
        self._monitor = monitor

    def render(
        self,
        reports_by_kind: Mapping[str, Sequence[HealthReport]],
    ) -> ViewSnapshot:
        status = self._monitor.status
        return ViewSnapshot(
            status=_HEALTH_TO_OVERALL[status.health],
            body={"latency_ms": status.latency_ms},
        )
