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

"""HealthView over the platform tunnel.

A managed project can hold a perfectly healthy runtime while nobody can reach
it, and until now nothing said so. The tunnel logs success at trace level and
failure at warning, so at the default level a working tunnel and one that never
started are equally silent — a stuck project and a healthy one produced byte
identical output.

Asking the service for its own state settles that, and ``/healthz`` is the
right place to put the answer: it is already scraped per project, so the fact
arrives wherever health is read without a second channel to wire up or a stream
of log lines to sift.

The view is CRITICAL. A managed project nobody can reach is not healthy, and
saying otherwise is the blindness this exists to remove. Two states are
deliberately not failures: a project with no tunnel at all (self-hosted and
local, which never had one) and a tunnel still on its way up.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from parlant.core.health.reporter import (
    HealthReport,
    OverallHealth,
    StatusCriticality,
    ViewSnapshot,
)


@dataclass(frozen=True)
class TunnelState:
    """What the tunnel says about itself, independent of how it is rendered."""

    state: str
    connected_for_seconds: float | None = None
    seconds_since_connected: float | None = None
    reconnect_attempts: int = 0
    last_error: str | None = None


class TunnelStateSource(Protocol):
    """Anything that can answer for a tunnel's condition.

    The port lives here, with the consumer, rather than in ``core.tunnels``:
    ``core.tunnels`` depends on ``core.health`` for its ``HealthReporter``, so
    pointing this at ``TunnelService`` would close a cycle. It is also the
    narrower dependency — the view needs one question answered, not a tunnel.
    """

    def tunnel_state(self) -> TunnelState: ...


# Anything not named here is treated as not-yet-working rather than broken:
# an unrecognised state is a reason to look, not a reason to declare an outage.
_HEALTHY_STATES = frozenset({"connected"})
_FAILING_STATES = frozenset({"reconnecting", "disconnected", "stopped"})


class TunnelHealthView:
    """Reports whether the tunnel is on, and if so whether it is working.

    Reads live state rather than consuming reports, in the same shape as
    ``EventLoopHealthView``: the tunnel already knows its own condition, so
    there is nothing to buffer.
    """

    name = "tunnel"
    criticality = StatusCriticality.CRITICAL
    kinds: tuple[str, ...] = ()

    def __init__(self, read_tunnel: Callable[[], TunnelStateSource | None]) -> None:
        # A callable, not the service: the tunnel is constructed after health
        # is configured, and for self-hosted projects it is never constructed
        # at all.
        self._read_tunnel = read_tunnel

    def render(
        self,
        reports_by_kind: Mapping[str, Sequence[HealthReport]],
    ) -> ViewSnapshot:
        try:
            tunnel = self._read_tunnel()
        except Exception as exc:
            # /healthz never turns callers away, so a view that raised would
            # take the endpoint down and hide every other check with it.
            return ViewSnapshot(
                status=OverallHealth.DEGRADED,
                body={"state": "unknown", "reason": f"{type(exc).__name__}: {exc}"},
            )

        if tunnel is None:
            return ViewSnapshot(
                status=OverallHealth.HEALTHY,
                body={"state": "off"},
            )

        state = tunnel.tunnel_state()
        body: dict[str, Any] = {"state": "on", "status": state.state}

        if state.connected_for_seconds is not None:
            body["connected_for_seconds"] = state.connected_for_seconds
        if state.seconds_since_connected is not None:
            body["seconds_since_connected"] = state.seconds_since_connected
        if state.reconnect_attempts:
            body["reconnect_attempts"] = state.reconnect_attempts
        if state.last_error:
            body["last_error"] = state.last_error

        if state.state in _HEALTHY_STATES:
            return ViewSnapshot(status=OverallHealth.HEALTHY, body=body)
        if state.state in _FAILING_STATES:
            return ViewSnapshot(status=OverallHealth.UNHEALTHY, body=body)
        return ViewSnapshot(status=OverallHealth.DEGRADED, body=body)
