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

"""The engine's TURN cost-control choke point: a denied turn is acknowledged and
terminated with status events only — a terminal `ready` carrying a namespaced
`cost_control` payload, and no message event — so clients waiting for turn
completion terminate cleanly and frontends decide the presentation."""

from datetime import datetime, timezone
from typing import cast

from parlant.core.cost_control import CostContext, CostVerdict, WorkKind
from parlant.core.emissions import EmittedEvent
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.loggers import Logger
from parlant.core.sessions import EventKind, EventSource, StatusEventData
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import create_engine_context


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append(message)


class _FakePolicy:
    def __init__(self, verdict: CostVerdict | Exception) -> None:
        self._verdict = verdict
        self.check_calls: list[tuple[CostContext, WorkKind]] = []

    async def check(self, context: CostContext, work: WorkKind) -> CostVerdict:
        self.check_calls.append((context, work))
        if isinstance(self._verdict, Exception):
            raise self._verdict
        return self._verdict

    def report(self, trace_id: str, model: str, usage: object) -> None:
        pass


def _make_engine(policy: _FakePolicy, logger: _FakeLogger | None = None) -> CompassEngine:
    engine = object.__new__(CompassEngine)
    engine._cost_control_policy = policy  # type: ignore[assignment]
    engine._tracer = LocalTracer()
    engine._logger = cast(Logger, logger or _FakeLogger())
    return engine


def _context() -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState()
    return context


def _emitted_events(context: EngineContext) -> list[EmittedEvent]:
    return cast(EventBuffer, context.session_event_emitter).events


def _cost_control_payload(context: EngineContext) -> dict[str, object]:
    status_data = cast(StatusEventData, _emitted_events(context)[0].data)
    envelope = cast(dict[str, object], status_data["data"])
    return cast(dict[str, object], envelope["cost_control"])


async def test_that_a_denied_turn_emits_a_terminal_ready_status_with_the_cost_control_payload() -> (
    None
):
    retry_at = datetime(2026, 7, 6, 12, 34, 56, tzinfo=timezone.utc)
    policy = _FakePolicy(
        CostVerdict(
            allowed=False,
            retry_after_utc=retry_at,
            reason="session cooldown",
            scope="session",
        )
    )
    engine = _make_engine(policy)
    context = _context()

    allowed = await engine._gate_turn_with_cost_control(context)

    assert allowed is False

    events = _emitted_events(context)
    assert len(events) == 1
    assert events[0].kind == EventKind.STATUS

    status_data = cast(StatusEventData, events[0].data)
    assert status_data["status"] == "ready"

    payload = _cost_control_payload(context)
    assert payload["circuit_breaker"] == "open"
    assert payload["scope"] == "session"
    assert payload["retry_after_utc"] == retry_at.isoformat()
    assert payload["reason"] == "session cooldown"

    # Crucially: no message event whatsoever.
    assert all(e.kind != EventKind.MESSAGE for e in events)


async def test_that_a_denial_without_a_retry_hint_omits_the_field() -> None:
    policy = _FakePolicy(CostVerdict(allowed=False))
    engine = _make_engine(policy)
    context = _context()

    await engine._gate_turn_with_cost_control(context)

    payload = _cost_control_payload(context)
    assert "retry_after_utc" not in payload
    assert payload["scope"] == "session"  # default scope when the policy names none


async def test_that_an_allowed_turn_proceeds_without_emitting_anything() -> None:
    policy = _FakePolicy(CostVerdict(allowed=True))
    engine = _make_engine(policy)
    context = _context()

    allowed = await engine._gate_turn_with_cost_control(context)

    assert allowed is True
    assert _emitted_events(context) == []


async def test_that_the_turn_gate_fails_open_when_the_policy_raises() -> None:
    logger = _FakeLogger()
    policy = _FakePolicy(RuntimeError("policy store is down"))
    engine = _make_engine(policy, logger)
    context = _context()

    allowed = await engine._gate_turn_with_cost_control(context)

    assert allowed is True
    assert _emitted_events(context) == []
    assert any("policy store is down" in w for w in logger.warnings)


async def test_that_advisory_warnings_are_logged_without_blocking_the_turn() -> None:
    logger = _FakeLogger()
    policy = _FakePolicy(
        CostVerdict(allowed=True, warnings=("session crossed the advisory threshold",))
    )
    engine = _make_engine(policy, logger)
    context = _context()

    allowed = await engine._gate_turn_with_cost_control(context)

    assert allowed is True
    assert _emitted_events(context) == []
    assert any("advisory threshold" in w for w in logger.warnings)


async def test_that_the_turn_gate_binds_the_full_cost_context() -> None:
    policy = _FakePolicy(CostVerdict(allowed=True))
    engine = _make_engine(policy)
    context = _context()

    await engine._gate_turn_with_cost_control(context)

    assert len(policy.check_calls) == 1
    cost_context, work = policy.check_calls[0]
    assert work == WorkKind.TURN
    assert cost_context.agent_id == context.agent.id
    assert cost_context.session_id == context.session.id
    assert cost_context.customer_id == context.customer.id
    assert cost_context.trace_id == engine._tracer.trace_id


# --- STEP gate (response loop) ----------------------------------------------------
#
# Checked at step boundaries: a deny stops iterating after the current step —
# in-flight streamed text is never truncated — and the loop's normal terminal
# path (the `ready {stage: completed}` emission) still runs.

from parlant.core.engines.compass.loop.blocking_loop import BlockingLoop  # noqa: E402


def _make_loop(policy: _FakePolicy, logger: _FakeLogger | None = None) -> BlockingLoop:
    loop = object.__new__(BlockingLoop)
    loop._cost_control_policy = policy  # type: ignore[assignment]
    loop._logger = cast(Logger, logger or _FakeLogger())
    return loop


async def test_that_a_denied_step_stops_the_loop_without_emitting_events() -> None:
    logger = _FakeLogger()
    policy = _FakePolicy(CostVerdict(allowed=False, reason="mid-turn breach"))
    loop = _make_loop(policy, logger)
    context = _context()

    allowed = await loop._gate_step(context)

    assert allowed is False
    assert _emitted_events(context) == []  # the loop's own terminal path handles status
    assert any("mid-turn breach" in w for w in logger.warnings)


async def test_that_an_allowed_step_proceeds() -> None:
    policy = _FakePolicy(CostVerdict(allowed=True))
    loop = _make_loop(policy)
    context = _context()

    assert await loop._gate_step(context) is True


async def test_that_the_step_gate_fails_open_when_the_policy_raises() -> None:
    logger = _FakeLogger()
    policy = _FakePolicy(RuntimeError("policy exploded"))
    loop = _make_loop(policy, logger)
    context = _context()

    assert await loop._gate_step(context) is True
    assert any("policy exploded" in w for w in logger.warnings)


async def test_that_the_step_gate_binds_the_step_work_kind() -> None:
    policy = _FakePolicy(CostVerdict(allowed=True))
    loop = _make_loop(policy)
    context = _context()

    await loop._gate_step(context)

    cost_context, work = policy.check_calls[0]
    assert work == WorkKind.STEP
    assert cost_context.session_id == context.session.id
    assert cost_context.trace_id == context.tracer.trace_id


# --- BACKGROUND gate (post-response finalization) -----------------------------------
#
# Gates pure-spend work with no visible turn (cache warm-ups, session pruning).
# Compaction is exempt by design — it reduces future cost.


async def test_that_a_denied_background_check_skips_finalization_work() -> None:
    logger = _FakeLogger()
    policy = _FakePolicy(CostVerdict(allowed=False, reason="over budget"))
    engine = _make_engine(policy, logger)
    context = _context()

    allowed = await engine._gate_background(context)

    assert allowed is False
    assert _emitted_events(context) == []  # background work has no client protocol
    assert any("over budget" in w for w in logger.warnings)


async def test_that_the_background_gate_fails_open_when_the_policy_raises() -> None:
    policy = _FakePolicy(RuntimeError("store down"))
    engine = _make_engine(policy)
    context = _context()

    assert await engine._gate_background(context) is True


async def test_that_the_background_gate_binds_the_background_work_kind() -> None:
    policy = _FakePolicy(CostVerdict(allowed=True))
    engine = _make_engine(policy)
    context = _context()

    await engine._gate_background(context)

    cost_context, work = policy.check_calls[0]
    assert work == WorkKind.BACKGROUND
    assert cost_context.agent_id == context.agent.id
