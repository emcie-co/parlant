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

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Mapping

from typing_extensions import override

from parlant.core.engines.compass.tracing import (
    CompassTracer,
    format_json_attr,
    normalize_attrs,
)
from parlant.core.tracer import AttributeValue, Tracer


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
    ) -> Iterator[None]:
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
    ) -> Iterator[None]:
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
