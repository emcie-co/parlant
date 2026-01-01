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
import os
from contextlib import contextmanager
from types import TracebackType
from typing import Iterator, Mapping
from typing_extensions import override, Self

from opentelemetry import trace, context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpOTLPSpanExporter,
)
from opentelemetry.trace import Status, StatusCode, SpanContext, TraceFlags
from opentelemetry.trace.span import TraceState

from parlant.core.common import generate_id
from parlant.core.tracer import Tracer, AttributeValue


class OpenTelemetryTracer(Tracer):
    def __init__(self) -> None:
        self._service_name = os.getenv("OTEL_SERVICE_NAME", "parlant")

        self._tracer_provider: TracerProvider
        self._span_processor: BatchSpanProcessor
        self._span_exporter: GrpcOTLPSpanExporter | HttpOTLPSpanExporter
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

        self._tracer_provider = TracerProvider(resource=resource)

        # Add console exporter for debugging (using BatchSpanProcessor)
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(
            span_exporter=console_exporter,
            schedule_delay_millis=1000,
        )
        self._tracer_provider.add_span_processor(console_processor)

        # Add OTLP exporter if endpoint is configured
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        if endpoint:
            insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "false").lower() == "true"
            protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()

            match protocol:
                case "http/protobuf":
                    self._span_exporter = HttpOTLPSpanExporter(endpoint=endpoint)
                case "http/json":
                    raise ValueError(
                        "http/json protocol is not supported for traces exporter. please use http/protobuf or grpc."
                    )
                case "grpc":
                    self._span_exporter = GrpcOTLPSpanExporter(
                        endpoint=endpoint,
                        insecure=insecure,
                    )
                case _:
                    raise ValueError(f"Unsupported OTLP protocol: {protocol}")

            self._span_processor = BatchSpanProcessor(
                span_exporter=self._span_exporter,
                schedule_delay_millis=2000,
            )
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
        # Use standard OpenTelemetry span creation
        current_spans = self._spans.get()

        # Prepare attributes first
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        if not current_spans:
            new_spans = span_id
            custom_trace_id = generate_id({"strategy": "uuid4"})
            trace_id_reset_token = self._trace_id.set(custom_trace_id)

            # Convert UUID hex to proper OpenTelemetry format
            # Ensure exactly 32 hex chars (128 bits) for trace ID
            trace_id_hex = str(custom_trace_id)[:32]
            trace_id_int = int(trace_id_hex, 16)

            # Ensure trace ID is non-zero (OpenTelemetry requirement)
            if trace_id_int == 0:
                trace_id_int = 1

            # Generate 64-bit span ID (16 hex chars)
            span_uuid = generate_id({"strategy": "uuid4"})
            span_id_hex = str(span_uuid)[:16]
            span_id_int = int(span_id_hex, 16)

            # Ensure span ID is non-zero (OpenTelemetry requirement)
            if span_id_int == 0:
                span_id_int = 1

            span_context = SpanContext(
                trace_id=trace_id_int,
                span_id=span_id_int,
                is_remote=False,
                trace_flags=TraceFlags(0x01),
                trace_state=TraceState(),
            )

            # For root spans, create a completely isolated context
            # We'll create the span with our custom context after setting up the isolated context
            isolated_ctx = context.Context()
            ctx = isolated_ctx
        else:
            new_spans = current_spans + f"::{span_id}"
            trace_id_reset_token = None
            ctx = context.get_current()

        spans_reset_token = self._spans.set(new_spans)
        attributes_reset_token = self._attributes.set(new_attributes)

        # Create span with the prepared context
        if not current_spans:
            # For root spans, we need to manually create a span with our custom context
            # Start the span normally first
            span = self._tracer.start_span(name=span_id, attributes=new_attributes, context=ctx)
            # Then update its context with our custom IDs (this is a workaround)
            if hasattr(span, "_context"):
                span._context = span_context
        else:
            # For child spans, create normally
            span = self._tracer.start_span(name=span_id, attributes=new_attributes, context=ctx)

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
