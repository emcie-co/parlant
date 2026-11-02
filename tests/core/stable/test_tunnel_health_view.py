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

"""``/healthz`` answers whether the tunnel is on, and if so whether it works.

A managed project reports itself running while nobody can reach it, and the
logs cannot tell the two apart: the tunnel logs success at trace level and
failure at warning, so a healthy tunnel and one that never started are both
silent. Asking the service its own state removes that ambiguity, and puts the
answer somewhere already scraped rather than in more log noise.
"""

from parlant.core.health import (
    OverallHealth,
    StatusCriticality,
    TunnelHealthView,
    TunnelState,
)


class _StubTunnel:
    def __init__(self, state: str, **kwargs: object) -> None:
        self._state = TunnelState(
            state=state,
            connected_for_seconds=kwargs.get("connected_for_seconds"),  # type: ignore[arg-type]
            seconds_since_connected=kwargs.get("seconds_since_connected"),  # type: ignore[arg-type]
            reconnect_attempts=int(kwargs.get("reconnect_attempts", 0)),  # type: ignore[call-overload]
            last_error=kwargs.get("last_error"),  # type: ignore[arg-type]
        )

    def tunnel_state(self) -> TunnelState:
        return self._state


def test_that_a_project_without_a_tunnel_is_healthy_and_says_it_is_off() -> None:
    # Self-hosted and local projects have no tunnel. Reporting them unhealthy
    # for lacking one would make every such deployment look broken.
    view = TunnelHealthView(lambda: None)

    snapshot = view.render({})

    assert snapshot.status is OverallHealth.HEALTHY
    assert snapshot.body["state"] == "off"


def test_that_a_connected_tunnel_is_healthy_and_reports_how_long() -> None:
    view = TunnelHealthView(lambda: _StubTunnel("connected", connected_for_seconds=42.0))

    snapshot = view.render({})

    assert snapshot.status is OverallHealth.HEALTHY
    assert snapshot.body["state"] == "on"
    assert snapshot.body["status"] == "connected"
    assert snapshot.body["connected_for_seconds"] == 42.0


def test_that_a_tunnel_that_cannot_connect_makes_the_project_unhealthy() -> None:
    # The whole point: a managed project nobody can reach must not report
    # healthy. This is what makes the failure visible to everything that
    # already reads /healthz.
    view = TunnelHealthView(
        lambda: _StubTunnel(
            "reconnecting",
            reconnect_attempts=7,
            seconds_since_connected=300.0,
            last_error="gaierror: [Errno -3] Temporary failure in name resolution",
        )
    )

    snapshot = view.render({})

    assert snapshot.status is OverallHealth.UNHEALTHY
    assert snapshot.body["state"] == "on"
    assert snapshot.body["status"] == "reconnecting"
    assert snapshot.body["reconnect_attempts"] == 7
    assert "name resolution" in str(snapshot.body["last_error"])


def test_that_a_tunnel_still_starting_is_degraded_rather_than_unhealthy() -> None:
    # A project a second into its life has not failed. Calling that unhealthy
    # makes every deploy flap, and most "unreachable" readings under load were
    # projects still coming up.
    view = TunnelHealthView(lambda: _StubTunnel("connecting"))

    snapshot = view.render({})

    assert snapshot.status is OverallHealth.DEGRADED
    assert snapshot.body["status"] == "connecting"


def test_that_the_view_folds_into_the_overall_rollup() -> None:
    assert TunnelHealthView.criticality is StatusCriticality.CRITICAL
    assert TunnelHealthView.name == "tunnel"
    assert TunnelHealthView.kinds == ()


def test_that_a_tunnel_that_raises_is_reported_rather_than_crashing_healthz() -> None:
    # /healthz never turns callers away; a view that throws would take the
    # whole endpoint down and hide every other check with it.
    def explode() -> None:
        raise RuntimeError("tunnel unavailable")

    view = TunnelHealthView(explode)

    snapshot = view.render({})

    assert snapshot.status is OverallHealth.DEGRADED
    assert snapshot.body["state"] == "unknown"


def test_that_a_project_whose_tunnel_binds_late_reports_off_until_it_does() -> None:
    # The tunnel is bound during `initialize_container`, which runs after
    # `configure_healthz`. A view that resolved the service once at wiring time
    # would capture the absence permanently and report every managed project as
    # tunnel-less forever, which is the failure this whole view exists to catch.
    bound: list[_StubTunnel] = []

    view = TunnelHealthView(lambda: bound[0] if bound else None)

    assert view.render({}).body["state"] == "off"

    bound.append(_StubTunnel("connected", connected_for_seconds=1.0))

    later = view.render({})
    assert later.status is OverallHealth.HEALTHY
    assert later.body["state"] == "on"
