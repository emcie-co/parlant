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
from contextlib import contextmanager, ExitStack
from collections.abc import Generator
from types import TracebackType
from typing import Mapping, Sequence
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
from opentelemetry.trace import Status, StatusCode, SpanContext, TraceFlags, Link
from opentelemetry.trace.span import TraceState

from parlant.core.common import generate_id
from parlant.core.tracer import Tracer, AttributeValue, TraceContext


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

        self._root_links = contextvars.ContextVar[Sequence[TraceContext]](
            "otel_trace_links", default=()
        )

        # Store the OpenTelemetry context for this tracer to maintain isolation
        self._otel_context = contextvars.ContextVar[context.Context | None](
            "otel_tracer_otel_context",
            default=None,
        )

    @contextmanager
    @override
    def fresh_trace(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
        *,
        links: Sequence[TraceContext] = (),
        trace_id: str | None = None,
    ) -> Generator[None, None, None]:
        with ExitStack() as stack:
            stack.callback(
                self._trace_id.reset, self._trace_id.set(trace_id or self._generate_trace_id())
            )
            stack.callback(self._spans.reset, self._spans.set(""))
            stack.callback(self._attributes.reset, self._attributes.set({}))
            stack.callback(self._current_span.reset, self._current_span.set(None))
            stack.callback(self._otel_context.reset, self._otel_context.set(None))
            stack.callback(self._root_links.reset, self._root_links.set(tuple(links)))
            with self.span(span_id, attributes):
                yield

    @property
    @override
    def current_span_context(self) -> TraceContext | None:
        span = self._current_span.get()
        native = span.get_span_context() if span else None
        if native is None or not native.is_valid:
            return None
        return TraceContext(f"{native.trace_id:032x}", f"{native.span_id:016x}")

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
        self._tracer = self._tracer_provider.get_tracer(__name__)

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
    ) -> Generator[None, None, None]:
        with ExitStack() as stack:
            parents = self._spans.get()
            new_attributes = {**self._attributes.get(), **attributes}
            if parents:
                native_context = self._otel_context.get() or context.Context()
            else:
                custom_id = self._trace_id.get() or self._generate_trace_id()
                stack.callback(self._trace_id.reset, self._trace_id.set(custom_id))
                seed = SpanContext(
                    trace_id=int(custom_id.replace("-", ""), 16),
                    span_id=int(str(generate_id({"strategy": "uuid4"}))[:16], 16) or 1,
                    is_remote=True,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                    trace_state=TraceState(),
                )
                native_context = trace.set_span_in_context(
                    trace.NonRecordingSpan(seed), context.Context()
                )
            stack.callback(
                self._spans.reset, self._spans.set(f"{parents}::{span_id}" if parents else span_id)
            )
            stack.callback(self._attributes.reset, self._attributes.set(new_attributes))
            links = (
                [
                    Link(
                        SpanContext(
                            trace_id=int(link.trace_id.replace("-", ""), 16),
                            span_id=int(link.span_id, 16),
                            is_remote=True,
                            trace_flags=TraceFlags(TraceFlags.SAMPLED),
                        )
                    )
                    for link in self._root_links.get()
                ]
                if not parents
                else []
            )
            span = self._tracer.start_span(
                name=span_id, attributes=new_attributes, context=native_context, links=links
            )
            stack.callback(span.end)
            stack.callback(
                self._otel_context.reset,
                self._otel_context.set(trace.set_span_in_context(span, native_context)),
            )
            stack.callback(self._current_span.reset, self._current_span.set(span))
            try:
                with trace.use_span(
                    span, end_on_exit=False, record_exception=False, set_status_on_exception=False
                ):
                    yield
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                raise

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Generator[None, None, None]:
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
