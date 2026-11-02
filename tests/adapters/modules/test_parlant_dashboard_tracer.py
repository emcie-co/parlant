# Copyright 2026 Parlant (Emcie Co Ltd.)
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

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from typing import cast
from unittest.mock import Mock

import httpx
from fastapi import APIRouter, FastAPI
from parlant.api.app import create_api_app
from parlant.core.container import Container
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span
from pytest import MonkeyPatch, fixture, raises, mark

from parlant.adapters.modules.parlant_dashboard import tracer as dashboard_tracer_module
from parlant.adapters.modules.parlant_dashboard.tracer import ParlantDashboardTracer
from parlant.core.application import Application
from parlant.core.engines.types import UtteranceRationale, UtteranceRequest
from parlant.core.sessions import SessionId
from parlant.core.tracer import Tracer, TURN_TRACE_ID_ATTRIBUTE, SESSION_OPERATION_ATTRIBUTE
from tests.test_utilities import create_agent, create_session


class CapturingSpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


async def test_that_a_follow_up_exports_a_native_span_link_to_its_origin(
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    with tracer.span("origin", {"session_id": "session"}):
        original = tracer.current_span_context
        assert original is not None
        root = tracer.trace_id
        assert original.trace_id == root
        assert original.span_id != tracer.span_id
        with tracer.fresh_trace(
            "follow_up",
            {TURN_TRACE_ID_ATTRIBUTE: root, SESSION_OPERATION_ATTRIBUTE: "utter"},
            links=[original],
        ):
            follow_up = tracer.trace_id
            assert follow_up != root
            with tracer.span("nested"):
                assert tracer.get_attribute(TURN_TRACE_ID_ATTRIBUTE) == root
        assert tracer.current_span_context == original
    tracer.flush()
    exported = next(span for span in span_exporter.spans if span.name == "follow_up")
    assert f"{exported.context.trace_id:032x}" == follow_up
    assert exported.attributes is not None
    assert exported.attributes[TURN_TRACE_ID_ATTRIBUTE] == root
    assert len(exported.links) == 1
    assert f"{exported.links[0].context.trace_id:032x}" == root
    assert f"{exported.links[0].context.span_id:016x}" == original.span_id


async def test_that_a_follow_up_exports_a_distinct_trace_with_the_original_turn_reference(
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    released = asyncio.Event()
    with tracer.span("origin", {"session_id": "session", TURN_TRACE_ID_ATTRIBUTE: "origin-turn"}):
        original = tracer.current_span_context
        assert original is not None

        async def finish() -> str:
            await released.wait()
            with tracer.fresh_trace(
                "follow_up",
                {TURN_TRACE_ID_ATTRIBUTE: "origin-turn", SESSION_OPERATION_ATTRIBUTE: "utter"},
                links=[original],
            ):
                return tracer.trace_id

        task = asyncio.create_task(finish())
    with tracer.span("newer_turn"):
        newer = tracer.trace_id
        released.set()
        follow_up = await task
        assert tracer.trace_id == newer
    tracer.flush()
    exported = next(span for span in span_exporter.spans if span.name == "follow_up")
    assert follow_up not in (original.trace_id, newer)
    assert exported.attributes is not None
    assert exported.attributes[TURN_TRACE_ID_ATTRIBUTE] == "origin-turn"
    assert exported.links[0].context.trace_id == int(original.trace_id, 16)


def exported_event_names(spans: Sequence[ReadableSpan]) -> list[str]:
    return [event.name for span in spans for event in span.events]


@fixture
def span_exporter() -> CapturingSpanExporter:
    return CapturingSpanExporter()


@fixture
async def session_id(container: Container) -> SessionId:
    agent = await create_agent(container, name="test-agent")
    session = await create_session(container, agent_id=agent.id)
    return session.id


@fixture
async def tracer(
    monkeypatch: MonkeyPatch,
    span_exporter: CapturingSpanExporter,
) -> AsyncIterator[Tracer]:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setattr(
        dashboard_tracer_module,
        "HTTPOTLPSpanExporter",
        lambda **_: span_exporter,
    )
    monkeypatch.setattr(dashboard_tracer_module, "set_tracer_provider", lambda _: None)

    async with ParlantDashboardTracer(project_id="test-project") as dashboard_tracer:
        yield dashboard_tracer


@mark.parametrize("fresh", [False, True])
def test_that_start_span_failure_restores_context_and_next_span_is_fresh(
    monkeypatch: MonkeyPatch,
    fresh: bool,
) -> None:
    tracer = ParlantDashboardTracer()
    failed_trace_id = "11111111-1111-1111-1111-111111111111"
    healthy_trace_id = "22222222-2222-2222-2222-222222222222"
    generated_trace_ids = iter([failed_trace_id, healthy_trace_id])
    monkeypatch.setattr(tracer, "_generate_trace_id", lambda: next(generated_trace_ids))
    start_error = RuntimeError("start_span failed")
    failing_otel_tracer = Mock()

    def fail_start_span(**_: object) -> Span:
        assert tracer.trace_id == failed_trace_id
        assert tracer.span_id == "failing.span"
        assert tracer.get_attribute("request_id") == "request-1"
        assert tracer._current_span.get() is None
        assert tracer._otel_context.get() is not None
        raise start_error

    failing_otel_tracer.start_span.side_effect = fail_start_span
    monkeypatch.setattr(tracer, "_otel_tracer", failing_otel_tracer, raising=False)

    with raises(RuntimeError) as exc_info:
        scope = tracer.fresh_trace if fresh else tracer.span
        with scope("failing.span", {"request_id": "request-1"}):
            pass

    assert exc_info.value is start_error
    assert tracer.trace_id == "<main>"
    assert tracer.span_id == "<main>"
    assert tracer.get_attribute("request_id") is None
    assert tracer._current_span.get() is None
    assert tracer._otel_context.get() is None

    provider = TracerProvider()
    monkeypatch.setattr(tracer, "_otel_tracer", provider.get_tracer(__name__))
    try:
        with tracer.span("healthy.span", {"request_id": "request-2"}):
            assert tracer.trace_id == healthy_trace_id
            assert tracer.span_id == "healthy.span"
            assert tracer.get_attribute("request_id") == "request-2"
            assert tracer._current_span.get() is not None
            assert tracer._otel_context.get() is not None

        assert tracer.trace_id == "<main>"
        assert tracer.span_id == "<main>"
        assert tracer.get_attribute("request_id") is None
        assert tracer._current_span.get() is None
        assert tracer._otel_context.get() is None
    finally:
        provider.shutdown()


def test_that_span_is_ended_once_when_setup_fails_after_start(
    monkeypatch: MonkeyPatch,
) -> None:
    tracer = ParlantDashboardTracer()
    monkeypatch.setattr(
        tracer,
        "_generate_trace_id",
        lambda: "11111111-1111-1111-1111-111111111111",
    )
    span_mock = Mock(spec=Span)
    created_span = cast(Span, span_mock)
    otel_tracer = Mock()
    otel_tracer.start_span.return_value = created_span
    monkeypatch.setattr(tracer, "_otel_tracer", otel_tracer, raising=False)
    setup_error = RuntimeError("context setup failed")
    original_set_span_in_context = otel_trace.set_span_in_context

    def fail_for_created_span(
        span: Span,
        ctx: otel_context.Context | None = None,
    ) -> otel_context.Context:
        if span is created_span:
            raise setup_error
        return original_set_span_in_context(span, ctx)

    monkeypatch.setattr(otel_trace, "set_span_in_context", fail_for_created_span)

    with raises(RuntimeError) as exc_info:
        with tracer.span("failing.span", {"request_id": "request-1"}):
            pass

    assert exc_info.value is setup_error
    span_mock.record_exception.assert_called_once_with(setup_error)
    span_mock.set_status.assert_called_once()
    recorded_status = cast(otel_trace.Status, span_mock.set_status.call_args.args[0])
    assert recorded_status.status_code is otel_trace.StatusCode.ERROR
    assert recorded_status.description == str(setup_error)
    span_mock.end.assert_called_once_with()
    assert tracer.trace_id == "<main>"
    assert tracer.span_id == "<main>"
    assert tracer.get_attribute("request_id") is None
    assert tracer._current_span.get() is None
    assert tracer._otel_context.get() is None


def test_that_attributes_setup_failure_restores_prior_attributes() -> None:
    tracer = ParlantDashboardTracer()
    span_mock = Mock(spec=Span)
    span_mock.is_recording.return_value = True
    attribute_error = RuntimeError("set_attribute failed")
    span_mock.set_attribute.side_effect = attribute_error
    attributes_reset_token = tracer._attributes.set({"prior": "kept"})
    span_reset_token = tracer._current_span.set(cast(Span, span_mock))

    try:
        with raises(RuntimeError) as exc_info:
            with tracer.attributes({"new": "value"}):
                pass

        assert exc_info.value is attribute_error
        assert tracer._attributes.get() == {"prior": "kept"}
        assert tracer.get_attribute("new") is None
    finally:
        tracer._current_span.reset(span_reset_token)
        tracer._attributes.reset(attributes_reset_token)


async def test_that_span_body_exception_is_exported_once_with_error_status(
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    body_error = RuntimeError("body failed")

    with raises(RuntimeError) as exc_info:
        with tracer.span(
            "failing.span",
            {"http.request.operation": "create_event"},
        ):
            raise body_error

    assert exc_info.value is body_error
    tracer.flush()

    (exported_span,) = span_exporter.spans
    exception_events = [event for event in exported_span.events if event.name == "exception"]
    assert len(exception_events) == 1
    assert exported_span.status.status_code is otel_trace.StatusCode.ERROR


async def test_that_event_is_exportable_before_parent_span_ends(
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    with tracer.span(
        "engine.process",
        {
            "http.request.operation": "create_event",
            "session_id": "session-1",
            "agent_id": "agent-1",
        },
    ):
        tracer.add_event("tool.called", {"tool_call_id": "call-1"})
        tracer.flush()

        assert [span.name for span in span_exporter.spans] == ["parlant.activity.event"]
        assert exported_event_names(span_exporter.spans) == ["tool.called"]
        (event,) = span_exporter.spans[0].events
        assert event.attributes is not None
        assert event.attributes["tool_call_id"] == "call-1"


async def test_that_event_envelope_preserves_trace_parent_and_request_attributes(
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    with tracer.span(
        "loop.step",
        {
            "http.request.operation": "create_event",
            "session_id": "session-1",
            "agent_id": "agent-1",
            "step_index": 2,
        },
    ):
        parent_context = otel_trace.get_current_span().get_span_context()
        tracer.add_event("loop.reasoning.started")
        tracer.flush()

        (event_span,) = span_exporter.spans
        assert event_span.context is not None
        assert event_span.parent is not None
        assert parent_context.is_valid
        assert event_span.context.is_valid
        assert event_span.parent.is_valid
        assert event_span.context.trace_id == parent_context.trace_id
        assert event_span.parent.trace_id == parent_context.trace_id
        assert event_span.parent.span_id == parent_context.span_id
        assert event_span.attributes is not None
        assert event_span.attributes["http.request.operation"] == "create_event"
        assert event_span.attributes["session_id"] == "session-1"
        assert event_span.attributes["agent_id"] == "agent-1"
        assert event_span.attributes["step_index"] == 2


async def test_that_tunnel_create_event_activity_is_exported_with_request_correlation(
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    with tracer.span(
        "tunnel.request",
        {
            "rpc.system": "parlant.tunnel",
            "rpc.method": "sessions.create_event",
            "parlant.tunnel.request_id": "request-1",
        },
    ):
        with tracer.span("engine.process"):
            tracer.add_event("loop.reasoning.started")

    tracer.flush()

    assert exported_event_names(span_exporter.spans) == ["loop.reasoning.started"]
    tunnel_span = next(span for span in span_exporter.spans if span.name == "tunnel.request")
    engine_span = next(span for span in span_exporter.spans if span.name == "engine.process")
    event_span = next(span for span in span_exporter.spans if span.name == "parlant.activity.event")
    assert tunnel_span.context is not None
    assert engine_span.context is not None
    assert engine_span.parent is not None
    assert event_span.context is not None
    assert event_span.parent is not None
    assert engine_span.context.trace_id == tunnel_span.context.trace_id
    assert engine_span.parent.span_id == tunnel_span.context.span_id
    assert event_span.context.trace_id == tunnel_span.context.trace_id
    assert event_span.parent.span_id == engine_span.context.span_id
    assert event_span.attributes is not None
    assert event_span.attributes["rpc.system"] == "parlant.tunnel"
    assert event_span.attributes["rpc.method"] == "sessions.create_event"
    assert event_span.attributes["parlant.tunnel.request_id"] == "request-1"


async def test_that_tunnel_activity_task_preserves_context_after_request_span_ends(
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    continue_task = asyncio.Event()
    background_task: asyncio.Task[None] | None = None

    async def emit_activity() -> None:
        await continue_task.wait()
        with tracer.span("engine.process"):
            tracer.add_event("loop.reasoning.started")

    try:
        with tracer.span(
            "tunnel.request",
            {
                "rpc.system": "parlant.tunnel",
                "rpc.method": "sessions.create_event",
                "parlant.tunnel.request_id": "request-1",
            },
        ):
            background_task = asyncio.create_task(emit_activity())

        assert background_task is not None
        continue_task.set()
        await asyncio.wait_for(background_task, timeout=1.0)
    finally:
        continue_task.set()
        if background_task is not None and not background_task.done():
            background_task.cancel()
            with suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(background_task, timeout=1.0)

    tracer.flush()

    assert exported_event_names(span_exporter.spans) == ["loop.reasoning.started"]
    tunnel_span = next(span for span in span_exporter.spans if span.name == "tunnel.request")
    engine_span = next(span for span in span_exporter.spans if span.name == "engine.process")
    event_span = next(span for span in span_exporter.spans if span.name == "parlant.activity.event")
    assert tunnel_span.context is not None
    assert tunnel_span.end_time is not None
    assert engine_span.context is not None
    assert engine_span.parent is not None
    assert engine_span.start_time is not None
    assert event_span.context is not None
    assert event_span.parent is not None
    assert tunnel_span.end_time <= engine_span.start_time
    assert engine_span.context.trace_id == tunnel_span.context.trace_id
    assert engine_span.parent.span_id == tunnel_span.context.span_id
    assert event_span.context.trace_id == tunnel_span.context.trace_id
    assert event_span.parent.span_id == engine_span.context.span_id
    for span in (engine_span, event_span):
        assert span.attributes is not None
        assert span.attributes["rpc.system"] == "parlant.tunnel"
        assert span.attributes["rpc.method"] == "sessions.create_event"
        assert span.attributes["parlant.tunnel.request_id"] == "request-1"


async def test_that_event_is_not_duplicated_when_parent_span_ends(
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    with tracer.span(
        "engine.process",
        {"http.request.operation": "create_event"},
    ):
        tracer.add_event("tool.called", {"tool_call_id": "call-1"})

    tracer.flush()

    assert exported_event_names(span_exporter.spans) == ["tool.called"]
    event_spans = [span for span in span_exporter.spans if span.name == "parlant.activity.event"]
    assert len(event_spans) == 1
    parent_span = next(span for span in span_exporter.spans if span.name == "engine.process")
    assert list(parent_span.events) == []


async def test_that_the_dashboard_tracer_uses_live_event_batch_schedule(
    monkeypatch: MonkeyPatch,
    span_exporter: CapturingSpanExporter,
) -> None:
    schedule_delays: list[int] = []
    original_batch_span_processor = BatchSpanProcessor

    def capturing_batch_span_processor(
        span_exporter: SpanExporter,
        schedule_delay_millis: int,
        max_queue_size: int,
        max_export_batch_size: int,
    ) -> BatchSpanProcessor:
        schedule_delays.append(schedule_delay_millis)
        return original_batch_span_processor(
            span_exporter=span_exporter,
            schedule_delay_millis=schedule_delay_millis,
            max_queue_size=max_queue_size,
            max_export_batch_size=max_export_batch_size,
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setattr(
        dashboard_tracer_module, "BatchSpanProcessor", capturing_batch_span_processor
    )
    monkeypatch.setattr(
        dashboard_tracer_module,
        "HTTPOTLPSpanExporter",
        lambda **_: span_exporter,
    )
    monkeypatch.setattr(dashboard_tracer_module, "set_tracer_provider", lambda _: None)

    async with ParlantDashboardTracer(project_id="test-project"):
        pass

    assert schedule_delays == [200]
    assert dashboard_tracer_module._EXPORT_SCHEDULE_DELAY_MILLIS == 200


async def test_that_only_create_event_request_spans_are_exported(
    async_client: httpx.AsyncClient,
    session_id: SessionId,
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    response = await async_client.post(
        f"/sessions/{session_id}/events",
        json={
            "kind": "status",
            "source": "human_agent",
            "status": "processing",
            "data": {},
        },
    )
    response.raise_for_status()

    (await async_client.get(f"/sessions/{session_id}")).raise_for_status()

    with tracer.span(
        "tunnel.request",
        {
            "rpc.system": "parlant.tunnel",
            "rpc.method": "sessions.list_events",
            "parlant.tunnel.request_id": "request-2",
        },
    ):
        with tracer.span("engine.process"):
            tracer.add_event("turn.interrupted", {"cause": "stop"})

    cast(ParlantDashboardTracer, tracer).flush()

    assert span_exporter.spans
    assert {span.name for span in span_exporter.spans} == {
        "http.request",
        "session.create_event",
    }
    assert all(
        span.attributes and span.attributes.get("http.request.operation") == "create_event"
        for span in span_exporter.spans
    )


async def test_that_an_utterance_outside_any_request_is_exported(
    container: Container,
    session_id: SessionId,
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    # An agent can speak with no customer request around it, e.g. delivering a
    # background lookup's answer with Agent.utter after its turn ended. Its
    # message carries the utterance's trace, so that trace must reach the
    # Dashboard or the message shows no activity.
    # (This module's fixture sets a placeholder model key, so the engine's
    # model call fails and the utterance ends in an error status; that still
    # runs, and traces, the whole utterance.)
    event = await container[Application].sessions.utter(
        session_id,
        [
            UtteranceRequest(
                action="Tell the customer their order shipped yesterday.",
                rationale=UtteranceRationale.UNSPECIFIED,
            )
        ],
    )

    cast(ParlantDashboardTracer, tracer).flush()

    exported = [span for span in span_exporter.spans if span.name == "sessions.utter"]
    assert exported, "the utterance's spans were not exported"
    assert all(
        f"{span.context.trace_id:032x}" == event.trace_id.replace("-", "") for span in exported
    )


async def test_that_nested_event_routes_export_activity_and_exclude_unrelated_requests(
    container: Container,
    tracer: Tracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    async def configure(app: FastAPI) -> FastAPI:
        events = APIRouter()

        @events.post("/sessions/{session_id}/events", operation_id="create_event")
        async def create_event(session_id: str) -> dict[str, str]:
            with tracer.span("engine.process"):
                tracer.add_event("turn.started", {"session_id": session_id})
            return {"session_id": session_id}

        @events.get("/sessions/{session_id}", operation_id="read_session")
        async def read_session(session_id: str) -> dict[str, str]:
            with tracer.span("session.read"):
                tracer.add_event("session.read", {"session_id": session_id})
            return {"session_id": session_id}

        projects = APIRouter()
        projects.include_router(events, prefix="/projects/{project_id}")
        app.include_router(projects, prefix="/regression")
        return app

    app = await create_api_app(container, configure)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        prefix = "/regression/projects/project-1/sessions/session-1"
        (await client.post(f"{prefix}/events")).raise_for_status()
        (await client.get(prefix)).raise_for_status()

    tracer.flush()
    assert exported_event_names(span_exporter.spans) == ["turn.started"]
    assert {span.name for span in span_exporter.spans} == {
        "http.request",
        "engine.process",
        "parlant.activity.event",
    }
    assert len({span.context.trace_id for span in span_exporter.spans if span.context}) == 1
    assert all(
        span.attributes and span.attributes.get("http.request.operation") == "create_event"
        for span in span_exporter.spans
    )


_TURN_ATTRIBUTES = {
    "rpc.system": "parlant.tunnel",
    "rpc.method": "sessions.create_event",
    "parlant.tunnel.request_id": "request-1",
    "session_id": "session-1",
}


@fixture
async def keepalive_tracer(
    monkeypatch: MonkeyPatch,
    span_exporter: CapturingSpanExporter,
) -> AsyncIterator[ParlantDashboardTracer]:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setattr(
        dashboard_tracer_module,
        "HTTPOTLPSpanExporter",
        lambda **_: span_exporter,
    )
    monkeypatch.setattr(dashboard_tracer_module, "set_tracer_provider", lambda _: None)

    async with ParlantDashboardTracer(
        project_id="test-project", keepalive_interval_seconds=0.05
    ) as dashboard_tracer:
        yield dashboard_tracer


def keepalive_spans(spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
    return [span for span in spans if span.name == "parlant.trace.keepalive"]


async def test_that_an_open_engine_process_sends_keep_alive_spans_while_the_turn_runs(
    keepalive_tracer: ParlantDashboardTracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    # The collector finalizes a trace after 30s without new spans; a long tool call
    # emits none, so the turn must keep its trace alive while the engine works.
    with keepalive_tracer.span("engine.process", _TURN_ATTRIBUTES):
        engine_context = otel_trace.get_current_span().get_span_context()
        await asyncio.sleep(0.2)
        keepalive_tracer.flush()

        beats = keepalive_spans(span_exporter.spans)
        assert len(beats) >= 2
        for beat in beats:
            assert beat.context is not None and beat.parent is not None
            assert beat.context.trace_id == engine_context.trace_id
            assert beat.parent.span_id == engine_context.span_id
            assert beat.attributes is not None
            assert beat.attributes["session_id"] == "session-1"
            # No events, so the collector adds nothing to the activity it shows.
            assert list(beat.events) == []


async def test_that_keep_alive_spans_stop_when_the_engine_process_ends(
    keepalive_tracer: ParlantDashboardTracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    with keepalive_tracer.span("compass.engine.process", _TURN_ATTRIBUTES):
        await asyncio.sleep(0.12)

    keepalive_tracer.flush()
    beats_at_end = len(keepalive_spans(span_exporter.spans))
    await asyncio.sleep(0.2)
    keepalive_tracer.flush()

    assert beats_at_end >= 1
    assert len(keepalive_spans(span_exporter.spans)) == beats_at_end


async def test_that_a_span_that_is_not_an_engine_process_sends_no_keep_alive(
    keepalive_tracer: ParlantDashboardTracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    with keepalive_tracer.span("tunnel.request", _TURN_ATTRIBUTES):
        await asyncio.sleep(0.15)

    keepalive_tracer.flush()

    assert keepalive_spans(span_exporter.spans) == []


async def test_that_a_failing_keep_alive_never_breaks_the_turn(
    monkeypatch: MonkeyPatch,
    keepalive_tracer: ParlantDashboardTracer,
    span_exporter: CapturingSpanExporter,
) -> None:
    original_start_span = keepalive_tracer._otel_tracer.start_span

    def start_span_failing_for_keep_alive(name: str, **kwargs: object) -> Span:
        if name == "parlant.trace.keepalive":
            raise RuntimeError("exporter unavailable")
        return original_start_span(name, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        keepalive_tracer._otel_tracer, "start_span", start_span_failing_for_keep_alive
    )

    with keepalive_tracer.span("engine.process", _TURN_ATTRIBUTES):
        await asyncio.sleep(0.15)
        keepalive_tracer.add_event("tool.called", {"tool_call_id": "call-1"})

    keepalive_tracer.flush()

    assert exported_event_names(span_exporter.spans) == ["tool.called"]
    assert any(span.name == "engine.process" for span in span_exporter.spans)
