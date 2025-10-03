# Copyright 2025 Emcie Co Ltd.
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
import os
from contextlib import contextmanager
from types import TracebackType
from typing import Iterator, Mapping
from typing_extensions import override, Self

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode, SpanContext

from parlant.core.common import AttributeValue, generate_id
from parlant.core.tracer import Tracer


class OtelTracer(Tracer):
    @staticmethod
    def is_environment_set() -> bool:
        """Check if the required OpenTelemetry environment variables are set."""
        return bool(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))

    def __init__(self) -> None:
        self._service_name = os.getenv("OTEL_SERVICE_NAME", "parlant")

        self._tracer_provider: TracerProvider
        self._span_processor: BatchSpanProcessor
        self._span_exporter: OTLPSpanExporter
        self._tracer: trace.Tracer

        self._spans = contextvars.ContextVar[str](
            "otel_tracer_spans",
            default="",
        )

        self._attributes = contextvars.ContextVar[Mapping[str, AttributeValue]](
            "otel_tracer_attributes",
            default={},
        )

        self._trace_id = contextvars.ContextVar[str](
            "otel_tracer_trace_id",
            default="",
        )

        self._current_span = contextvars.ContextVar[trace.Span | None](
            "otel_tracer_current_span",
            default=None,
        )

    async def __aenter__(self) -> Self:
        resource = Resource.create({"service.name": self._service_name})

        endpoint = "http://host.docker.internal:4317"
        insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "false").lower() == "true"

        self._span_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            insecure=insecure,
        )

        self._span_processor = BatchSpanProcessor(
            span_exporter=self._span_exporter,
            schedule_delay_millis=2000,
        )

        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(self._span_processor)

        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(__name__)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._tracer_provider.force_flush()
        self._tracer_provider.shutdown()

        return False

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        current_spans = self._spans.get()

        if not current_spans:
            new_trace_id = generate_id({"strategy": "uuid4"})
            new_spans = span_id
            trace_id_reset_token = self._trace_id.set(new_trace_id)

            span_context = SpanContext(
                # UUID string → 128-bit integer (mask ensures it fits in 128 bits)
                trace_id=int(new_trace_id, 16) & ((1 << 128) - 1),
                # UUID string → 64-bit integer (mask ensures it fits in 64 bits)
                span_id=int(generate_id({"strategy": "uuid4"})[:16], 16) & ((1 << 64) - 1),
                is_remote=False,
            )
            parent_context = trace.set_span_in_context(trace.NonRecordingSpan(span_context))
        else:
            new_spans = current_spans + f"::{span_id}"
            trace_id_reset_token = None
            parent_context = None

        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        spans_reset_token = self._spans.set(new_spans)
        attributes_reset_token = self._attributes.set(new_attributes)

        span = self._tracer.start_span(
            name=span_id,
            attributes=new_attributes,
            context=parent_context,
        )

        span_token = self._current_span.set(span)

        try:
            with trace.use_span(span, end_on_exit=True):
                yield
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            self._spans.reset(spans_reset_token)
            self._attributes.reset(attributes_reset_token)
            self._current_span.reset(span_token)
            if trace_id_reset_token is not None:
                self._trace_id.reset(trace_id_reset_token)

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Iterator[None]:
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        attributes_reset_token = self._attributes.set(new_attributes)

        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            current_span.set_attributes(attributes)

        try:
            yield
        finally:
            self._attributes.reset(attributes_reset_token)

    @property
    @override
    def trace_id(self) -> str:
        if trace_id := self._trace_id.get():
            return trace_id

        return "<main>"

    @property
    @override
    def span_id(self) -> str:
        if spans := self._spans.get():
            return spans

        return "<main>"

    @override
    def get_attribute(
        self,
        name: str,
    ) -> AttributeValue | None:
        attributes = self._attributes.get()
        return attributes.get(name, None)

    @override
    def set_attribute(
        self,
        name: str,
        value: AttributeValue,
    ) -> None:
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, name: value}
        self._attributes.set(new_attributes)

        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            current_span.set_attribute(name, value)

    @override
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes)

    @override
    def flush(self) -> None:
        if hasattr(self, "_tracer_provider") and self._tracer_provider:
            self._tracer_provider.force_flush()
