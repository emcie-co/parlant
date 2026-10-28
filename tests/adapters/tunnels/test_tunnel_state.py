"""The tunnel says what state it is in, so an outage can be diagnosed.

A tunnel that is down looks exactly like a tunnel that is quiet: success logs
at trace level, failure at warning, so at the default level a working tunnel
and one that never started produce identical output. These tests pin the
states apart, because telling them apart afterwards is the whole point -- a
failed connect names its error, a live connection reports how long it has been
up, and a connect loop that has stopped turning shows a gap since it was last
connected rather than silence.

This was briefly a log line every thirty seconds in every project. The state
was worth having; announcing it on a timer was not. It is now answered on
demand, and read through the `tunnel` view on /healthz.
"""

import asyncio
from unittest.mock import AsyncMock

from parlant.adapters.modules.parlant_dashboard import ParlantDashboardTunnelService
from parlant.core.loggers import StdoutLogger
from parlant.core.tracer import LocalTracer


def _service() -> ParlantDashboardTunnelService:
    return ParlantDashboardTunnelService(
        url="ws://example.invalid/tunnel",
        token="token",
        dispatcher=AsyncMock(),
        logger=StdoutLogger(LocalTracer()),
    )


async def test_that_a_connected_tunnel_reports_how_long_it_has_been_up() -> None:
    service = _service()
    service._tunnel_state = "connected"
    service._connected_since = asyncio.get_running_loop().time() - 5.0

    state = service.tunnel_state()

    assert state.state == "connected"
    assert state.connected_for_seconds is not None
    assert state.connected_for_seconds > 0


async def test_that_the_state_names_the_error_behind_a_failed_connect() -> None:
    # Without the error, a reconnect loop is indistinguishable from any other
    # kind of failure. The error is what points at DNS vs TLS vs auth without
    # needing to guess.
    service = _service()
    service._tunnel_state = "reconnecting"
    service._last_error = "gaierror: [Errno -3] Temporary failure in name resolution"
    service._reconnect_attempts = 7

    state = service.tunnel_state()

    assert state.state == "reconnecting"
    assert state.reconnect_attempts == 7
    assert state.last_error is not None
    assert "Temporary failure in name resolution" in state.last_error


async def test_that_a_tunnel_that_dropped_reports_the_gap_since_it_last_worked() -> None:
    # The case with no evidence today. A tunnel stuck reconnecting logs nothing
    # at the default level, and silence reads exactly like a healthy idle
    # tunnel. The gap since it was last connected is what identifies how long
    # the project has actually been unreachable.
    service = _service()
    service._tunnel_state = "reconnecting"
    service._connected_since = None
    service._last_connected_at = asyncio.get_running_loop().time() - 900.0

    state = service.tunnel_state()

    assert state.connected_for_seconds is None
    assert state.seconds_since_connected is not None
    assert state.seconds_since_connected > 100.0


async def test_that_a_tunnel_that_never_connected_reports_neither_duration() -> None:
    service = _service()

    state = service.tunnel_state()

    assert state.state == "starting"
    assert state.connected_for_seconds is None
    assert state.seconds_since_connected is None


async def test_that_stopping_reports_the_tunnel_as_stopped() -> None:
    service = _service()
    service._running = True
    service._tunnel_state = "connected"
    service._connected_since = asyncio.get_running_loop().time()

    await service.stop()

    state = service.tunnel_state()
    assert state.state == "stopped"
    assert state.connected_for_seconds is None
