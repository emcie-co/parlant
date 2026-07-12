import asyncio
from typing import Any, cast

import pytest

from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.types import Context
from parlant.core.loggers import StdoutLogger
from parlant.core.meter import Meter
from parlant.core.sessions import Event, EventKind, SessionId
from parlant.core.cost_control import AdvisoryCostControlPolicy
from parlant.core.usage_reporter import UsageReporter

from tests.core.stable.engines.compass.matching.utils import (
    RecordedEvent,
    RecordingTracer,
    create_agent,
    create_customer,
    create_session,
)


class _FakeEntityQueries:
    def __init__(self) -> None:
        self.agent = create_agent()
        self.customer = create_customer()
        self.session = create_session(self.agent, self.customer)

    async def read_agent(self, agent_id: str) -> Any:
        return self.agent

    async def read_session(self, session_id: SessionId) -> Any:
        return self.session

    async def read_customer(self, customer_id: str) -> Any:
        return self.customer

    async def find_events(self, session_id: SessionId) -> list[Event]:
        return []


class _CancellableMatcher:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def preload(self, context: Any) -> None:
        self.started.set()
        await asyncio.Future[None]()

    async def fill(self, context: Any) -> None:
        raise AssertionError("fill should not be reached")


class _FakeHooks:
    async def call_on_acknowledging(self, context: Any) -> bool:
        return True

    async def call_on_acknowledged(self, context: Any) -> bool:
        return True

    async def call_on_preparing(self, context: Any) -> bool:
        return True

    async def call_on_error(self, context: Any, exc: Exception) -> bool:
        return True


@pytest.mark.parametrize(
    ("cancel_message", "expected_cause"),
    [
        ("Forced cancellation by BackgroundTaskService [reason: cancelled_by_api]", "stop"),
        ("Restarting task 'process-session(session-1)'", "send_as_interrupt"),
    ],
)
async def test_process_cancellation_during_preparation_emits_turn_interrupted(
    cancel_message: str,
    expected_cause: str,
) -> None:
    tracer = RecordingTracer()
    entity_queries = _FakeEntityQueries()
    matcher = _CancellableMatcher()
    event_emitter = EventBuffer(entity_queries.agent)

    engine = CompassEngine(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        meter=cast(Meter, object()),
        matcher=cast(Any, matcher),
        responder=cast(Any, object()),
        compacter=cast(Any, object()),
        entity_queries=cast(Any, entity_queries),
        entity_commands=cast(Any, object()),
        hooks=cast(Any, _FakeHooks()),
        usage_reporter=UsageReporter(tracer),
        cost_control_policy=AdvisoryCostControlPolicy(UsageReporter(tracer)),
    )

    task = asyncio.create_task(
        engine.process(
            Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
            event_emitter,
        )
    )

    await asyncio.wait_for(matcher.started.wait(), timeout=1)
    task.cancel(cancel_message)

    with pytest.raises(asyncio.CancelledError):
        await task

    status_events = [event for event in event_emitter.events if event.kind == EventKind.STATUS]

    assert [cast(dict[str, Any], event.data)["status"] for event in status_events] == [
        "acknowledged",
        "processing",
        "cancelled",
        "ready",
    ]
    assert tracer.events == [
        RecordedEvent(
            name="turn.interrupted",
            attributes={
                "cause": expected_cause,
                "session_id": entity_queries.session.id,
                "agent_id": entity_queries.agent.id,
            },
            span_id="engine.process",
        )
    ]
