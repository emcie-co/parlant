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

from collections.abc import Generator
import asyncio
import contextvars
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing_extensions import override

from parlant.adapters.tracing.opentelemetry import OpenTelemetryTracer
from parlant.core.engines.compass.tracing import (
    CompassTracer,
    format_json_attr,
    normalize_attrs,
)
from parlant.core.groups import GroupId
from parlant.core.nlp.react import ToolCallPart
from parlant.core.sessions import ToolEventData
from parlant.core.tracer import AttributeValue, Tracer
from parlant.core.tracer import LocalTracer
from tests.core.stable.engines.compass.matching.utils import create_rule


@dataclass(frozen=True)
class RecordedEvent:
    name: str
    attributes: Mapping[str, AttributeValue]
    span_id: str


@dataclass(frozen=True)
class RecordedSpan:
    name: str
    attributes: Mapping[str, AttributeValue]


class RecordingTracer(Tracer):
    def __init__(self) -> None:
        self.started_spans: list[RecordedSpan] = []
        self.finished_spans: list[str] = []
        self.events: list[RecordedEvent] = []
        self._attributes: dict[str, AttributeValue] = {}
        self._span_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "test_compass_tracer_span_id",
            default="<main>",
        )

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]:
        self.started_spans.append(RecordedSpan(name=span_id, attributes=dict(attributes)))
        token = self._span_id_var.set(span_id)

        try:
            yield
        finally:
            self._span_id_var.reset(token)
            self.finished_spans.append(span_id)

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Generator[None, None, None]:
        previous_attributes = dict(self._attributes)
        self._attributes.update(attributes)

        try:
            yield
        finally:
            self._attributes = previous_attributes

    @property
    @override
    def trace_id(self) -> str:
        return "trace-id"

    @property
    @override
    def span_id(self) -> str:
        return self._span_id_var.get()

    @override
    def get_attribute(self, name: str) -> AttributeValue | None:
        return self._attributes.get(name)

    @override
    def set_attribute(self, name: str, value: AttributeValue) -> None:
        self._attributes[name] = value

    @override
    def add_event(self, name: str, attributes: Mapping[str, AttributeValue] = {}) -> None:
        self.events.append(
            RecordedEvent(name=name, attributes=dict(attributes), span_id=self.span_id)
        )

    @override
    def flush(self) -> None:
        pass


class _FailingTelemetryTracer(RecordingTracer):
    @override
    def set_attribute(self, name: str, value: AttributeValue) -> None:
        raise RuntimeError("attribute export failed")

    @override
    def add_event(self, name: str, attributes: Mapping[str, AttributeValue] = {}) -> None:
        raise RuntimeError("event export failed")


class _CancelledTelemetryTracer(RecordingTracer):
    @override
    def set_attribute(self, name: str, value: AttributeValue) -> None:
        raise asyncio.CancelledError("attribute export cancelled")

    @override
    def add_event(self, name: str, attributes: Mapping[str, AttributeValue] = {}) -> None:
        raise asyncio.CancelledError("event export cancelled")


class _FailingSpanTracer(RecordingTracer):
    def __init__(self, *, fail_on: str) -> None:
        super().__init__()
        self._fail_on = fail_on

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]:
        if self._fail_on == "entry":
            raise RuntimeError("span entry failed")

        yield

        if self._fail_on == "exit":
            raise RuntimeError("span exit failed")


def _bounded_otel_tracer() -> tuple[OpenTelemetryTracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        span_limits=SpanLimits(max_events=128),
        shutdown_on_exit=False,
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OpenTelemetryTracer()
    tracer._tracer_provider = provider
    tracer._tracer = provider.get_tracer(__name__)
    return tracer, exporter


def _component_references(
    spans: tuple[ReadableSpan, ...],
) -> list[tuple[str, str, str]]:
    return [
        (
            str(event.attributes["entity_type"]),
            str(event.attributes["parlant_id"]),
            str(event.attributes["last_modified_utc"]),
        )
        for span in spans
        for event in span.events
        if event.name == "loaded.entity" and event.attributes is not None
    ]


def test_that_json_attrs_serialize_nested_values_and_fall_back_to_str() -> None:
    value = {
        "text": "שלום",
        "nested": {"numbers": [1, 2]},
        "timestamp": datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
    }

    assert (
        format_json_attr(value)
        == '{"text": "שלום", "nested": {"numbers": [1, 2]}, "timestamp": "2026-07-05 12:00:00+00:00"}'
    )


def test_that_attrs_are_normalized_for_tracer_attribute_values() -> None:
    assert normalize_attrs(
        {
            "none": None,
            "text": "value",
            "flag": True,
            "count": 3,
            "ratio": 1.5,
            "strings": ["a", "b"],
            "bools": [True, False],
            "ints": [1, 2],
            "floats": [1.5, 2.5],
            "nested": {"answer": 42},
            "mixed": ["a", 1],
        }
    ) == {
        "text": "value",
        "flag": True,
        "count": 3,
        "ratio": 1.5,
        "strings": ["a", "b"],
        "bools": [True, False],
        "ints": [1, 2],
        "floats": [1.5, 2.5],
        "nested": '{"answer": 42}',
        "mixed": '["a", 1]',
    }


def test_that_events_are_recorded_with_normalized_attributes() -> None:
    tracer = RecordingTracer()

    CompassTracer(tracer).event(
        "compass.response.ready",
        attributes={
            "stage": "review",
            "skipped": None,
            "metadata": {"labels": ["urgent"]},
        },
    )

    assert tracer.events == [
        RecordedEvent(
            name="compass.response.ready",
            attributes={
                "stage": "review",
                "metadata": '{"labels": ["urgent"]}',
            },
            span_id="<main>",
        )
    ]


def test_that_component_loaded_emits_the_canonical_entity_reference() -> None:
    tracer = RecordingTracer()

    CompassTracer(tracer).component_loaded(
        entity_type="context_variable",
        parlant_id="variable-1",
        last_modified_utc="2026-08-09T10:00:00+00:00",
    )

    assert [(event.name, dict(event.attributes)) for event in tracer.events] == [
        (
            "loaded.entity",
            {
                "entity_type": "context_variable",
                "parlant_id": "variable-1",
                "last_modified_utc": "2026-08-09T10:00:00+00:00",
            },
        )
    ]


def test_that_component_loaded_is_fail_open() -> None:
    CompassTracer(_FailingTelemetryTracer()).component_loaded(
        entity_type="rule",
        parlant_id="rule-1",
        last_modified_utc="2026-08-09T10:00:00+00:00",
    )


def test_that_rule_batch_keeps_all_component_refs_with_bounded_otel_events() -> None:
    shared_group = GroupId("shared-policy-group")
    rules = [
        create_rule(
            condition=f"condition {index}",
            action=f"action {index}",
            groups=[shared_group],
        )
        for index in range(70)
    ]
    tracer, exporter = _bounded_otel_tracer()

    with tracer.span("match.preload"):
        CompassTracer(tracer).rules_loaded([*rules, rules[0]])

    spans = exporter.get_finished_spans()
    component_spans = tuple(span for span in spans if span.name == "compass.engine.load_components")
    operation_span = next(span for span in spans if span.name == "match.preload")
    references = _component_references(spans)
    expected_references = {
        ("rule", str(rule.id), rule.modified_utc.isoformat()) for rule in rules
    } | {("group", str(shared_group), "")}

    assert len(references) == 71
    assert set(references) == expected_references
    assert len(set(references)) == len(references)
    assert len(component_spans) == 1
    assert len(component_spans[0].events) == 71
    assert component_spans[0].context.trace_id == operation_span.context.trace_id
    assert component_spans[0].parent is not None
    assert component_spans[0].parent.span_id == operation_span.context.span_id
    assert [event.name for event in operation_span.events] == ["loaded.rule"] * 70


def test_that_large_rule_batch_chunks_component_refs_below_otel_event_limit() -> None:
    shared_group = GroupId("shared-policy-group")
    rules = [
        create_rule(
            condition=f"condition {index}",
            action=f"action {index}",
            groups=[shared_group],
        )
        for index in range(250)
    ]
    tracer, exporter = _bounded_otel_tracer()

    with tracer.span("match.preload"):
        CompassTracer(tracer).rules_loaded([*rules, rules[0]])

    spans = exporter.get_finished_spans()
    component_spans = tuple(span for span in spans if span.name == "compass.engine.load_components")
    operation_span = next(span for span in spans if span.name == "match.preload")
    references = _component_references(spans)
    expected_references = {
        ("rule", str(rule.id), rule.modified_utc.isoformat()) for rule in rules
    } | {("group", str(shared_group), "")}

    assert len(references) == 251
    assert set(references) == expected_references
    assert len(set(references)) == len(references)
    assert len(component_spans) == 3
    assert [len(span.events) for span in component_spans] == [100, 100, 51]
    assert all(len(span.events) <= 100 for span in component_spans)
    assert all(span.context.trace_id == operation_span.context.trace_id for span in component_spans)
    assert all(
        span.parent is not None and span.parent.span_id == operation_span.context.span_id
        for span in component_spans
    )
    assert [event.name for event in operation_span.events] == ["loaded.rule"] * 128


def test_that_tool_review_lifecycle_events_share_review_ref_without_private_text() -> None:
    tracer = RecordingTracer()
    compass_tracer = CompassTracer(tracer)
    tool_call = ToolCallPart(id="call-1", name="lookup_account", args={"account_id": "A-1"})
    result = SimpleNamespace(
        todo="Keep this private",
        adjusted_reasoning="Keep this private too",
    )

    compass_tracer.tool_review_started("review-1", [tool_call])
    compass_tracer.tool_review(
        "review-1",
        result,
        [tool_call],
        [cast(Any, SimpleNamespace(name="lookup_account", consequential=True))],
        rejected=True,
    )

    assert tracer.events == [
        RecordedEvent(
            name="review.started",
            attributes={
                "review_ref": "review-1",
                "tool_count": 1,
                "tool_names": ["lookup_account"],
            },
            span_id="<main>",
        ),
        RecordedEvent(
            name="review.rejected",
            attributes={
                "review_ref": "review-1",
                "tool_count": 1,
                "tool_names": ["lookup_account"],
                "approved": False,
                "has_todo": True,
                "has_adjusted_reasoning": True,
            },
            span_id="<main>",
        ),
        RecordedEvent(
            name="tool.reviewed",
            attributes={
                "tool_call_id": "call-1",
                "tool_name": "lookup_account",
                "arguments": '{"account_id": "A-1"}',
                "review_status": "rejected",
                "approved": False,
                "consequential": True,
                "has_todo": True,
                "has_adjusted_reasoning": True,
            },
            span_id="<main>",
        ),
    ]


def test_that_compass_telemetry_failures_do_not_escape_into_engine_execution() -> None:
    tracer = CompassTracer(_FailingTelemetryTracer())

    tracer.event("tool.called", {"tool_name": "lookup"})
    tracer.process_resolved(
        outcome="completed",
        stage="engine.process.finalize",
        reply_emitted=True,
    )


def test_that_cancelled_error_from_add_event_does_not_escape_compass_tracer() -> None:
    CompassTracer(_CancelledTelemetryTracer()).event(
        "tool.called",
        {"tool_name": "lookup"},
    )


def test_that_cancelled_error_from_set_attribute_does_not_escape_compass_tracer() -> None:
    CompassTracer(_CancelledTelemetryTracer()).process_resolved(
        outcome="completed",
        stage="engine.process.finalize",
        reply_emitted=True,
    )


def test_that_span_entry_failure_does_not_skip_engine_work() -> None:
    executed = False

    with CompassTracer(_FailingSpanTracer(fail_on="entry")).span("match.fill"):
        executed = True

    assert executed is True


def test_that_span_exit_failure_does_not_replace_successful_engine_work() -> None:
    with CompassTracer(_FailingSpanTracer(fail_on="exit")).span("match.fill"):
        pass


def test_that_span_exit_failure_preserves_the_engine_error() -> None:
    engine_error = ValueError("matching failed")

    with pytest.raises(ValueError) as exc_info:
        with CompassTracer(_FailingSpanTracer(fail_on="exit")).span("match.fill"):
            raise engine_error

    assert exc_info.value is engine_error


def test_that_final_rule_decisions_use_rule_first_event_names_and_revision_data() -> None:
    tracer = RecordingTracer()
    rule = create_rule(
        condition="the customer asks for a refund",
        action="explain the refund policy",
        title="Refunds",
    )

    CompassTracer(tracer).rule_decision(
        rule,
        selected=True,
        rationale="The customer requested a refund.",
    )
    CompassTracer(tracer).rule_decision(
        rule,
        selected=False,
        rationale="The customer changed topics.",
    )

    common_attributes = {
        "rule_id": str(rule.id),
        "title": "Refunds",
        "last_modified": rule.modified_utc.isoformat(),
    }
    assert tracer.events == [
        RecordedEvent(
            name="rule.selected",
            attributes={
                **common_attributes,
                "rationale": "The customer requested a refund.",
            },
            span_id="<main>",
        ),
        RecordedEvent(
            name="rule.ruled_out",
            attributes={
                **common_attributes,
                "rationale": "The customer changed topics.",
            },
            span_id="<main>",
        ),
    ]


def test_that_aggregate_tool_events_correlate_repeated_tool_names_by_call_id() -> None:
    tracer = RecordingTracer()
    data = ToolEventData(
        tool_calls=[
            {
                "tool_id": "local:lookup",
                "rationale": "first lookup",
                "arguments": {"query": "first"},
                "result": {
                    "data": "first result",
                    "metadata": {},
                    "control": {},
                    "canned_responses": [],
                    "canned_response_fields": {},
                },
            },
            {
                "tool_id": "local:lookup",
                "rationale": "second lookup",
                "arguments": {"query": "second"},
                "result": {
                    "data": "second result",
                    "metadata": {},
                    "control": {},
                    "canned_responses": [],
                    "canned_response_fields": {},
                },
            },
        ]
    )

    CompassTracer(tracer).loop_tool_persistent(data, ["call-1", "call-2"])

    event = tracer.events[-1]
    assert event.attributes["tool_call_ids"] == ["call-1", "call-2"]
    assert [call["tool_call_id"] for call in json.loads(str(event.attributes["tool_calls"]))] == [
        "call-1",
        "call-2",
    ]


def test_that_local_tracer_restores_span_and_attributes_after_an_error() -> None:
    tracer = LocalTracer()

    with pytest.raises(RuntimeError, match="failed"):
        with tracer.span("failing", {"scope": "span"}):
            with tracer.attributes({"temporary": True}):
                raise RuntimeError("failed")

    assert tracer.span_id == "<main>"
    assert tracer.get_attribute("scope") is None
    assert tracer.get_attribute("temporary") is None
