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

"""ParlantCloudTracer — OpenTelemetry-backed tracer for Parlant Cloud."""

import contextvars
import logging
import os
from contextlib import contextmanager
from types import TracebackType
from typing import Iterator, Mapping

from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, set_tracer_provider
from typing_extensions import Self, override

from parlant.core.tracer import AttributeValue, Tracer

from .config import _get_cloud_otel_url

_logger = logging.getLogger(__name__)


class ParlantCloudTracer(Tracer):
    def __init__(self, project_id: str = "") -> None:
        self._project_id = project_id

        self._endpoint = os.getenv(
            "PARLANT_CLOUD_OTEL_ENDPOINT",
            f"{_get_cloud_otel_url()}/v1/traces",
        )

        self._project_token = os.getenv("PARLANT_CLOUD_PROJECT_TOKEN", "")

        self._spans = contextvars.ContextVar[str](
            "tracer_spans",
            default="",
        )

        self._attributes = contextvars.ContextVar[Mapping[str, AttributeValue]](
            "tracer_attributes",
            default={},
        )

        self._trace_id = contextvars.ContextVar[str](
            "tracer_trace_id",
            default="",
        )

        self._current_span = contextvars.ContextVar[Span | None](
            "tracer_current_span",
            default=None,
        )

        self._otel_context = contextvars.ContextVar[context.Context | None](
            "parlant_cloud_tracer_otel_context",
            default=None,
        )

    async def __aenter__(self) -> Self:
        headers = {}
        if self._project_token:
            headers["authorization"] = f"Bearer {self._project_token}"
        else:
            _logger.info(
                "Parlant Cloud tracing is not configured. Learn more at https://parlant.io/cloud"
            )

        span_exporter = HTTPOTLPSpanExporter(
            endpoint=self._endpoint,
            headers=headers,
        )

        processor = BatchSpanProcessor(
            span_exporter=span_exporter,
            schedule_delay_millis=1000,
            max_queue_size=1000,
            max_export_batch_size=100,
        )

        original_on_end = processor.on_end

        def filtered_on_end(span: ReadableSpan) -> None:
            attributes = dict(span.attributes) if span.attributes else {}
            if attributes.get("http.request.operation") == "create_event":
                original_on_end(span)

        setattr(processor, "on_end", filtered_on_end)

        resource_attributes: dict[str, str] = {
            "service.name": "parlant-cloud-tracer",
        }
        if self._project_id:
            resource_attributes["project_id"] = self._project_id

        resource = Resource.create(resource_attributes)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)
        set_tracer_provider(provider)
        self._tracer_provider = provider
        self._otel_tracer = provider.get_tracer(__name__)
        self._processor = processor
        self._initialized = True

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._processor:
            try:
                self._processor.force_flush()
                self._processor.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                _logger.warning(f"Error during ParlantCloudTracer shutdown: {e}")

        return False

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        current_span_chain = self._spans.get()
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        if not current_span_chain:
            new_span_chain = span_id
            # Use existing trace_id if set by CompositeTracer, otherwise generate
            existing = self._trace_id.get()
            if existing:
                custom_trace_id = existing
            else:
                custom_trace_id = self._generate_trace_id()
            trace_id_reset_token = self._trace_id.set(custom_trace_id)

            # Inject Parlant trace_id into the OTEL context so the SDK uses
            # the same ID (UUID hex → 128-bit int). Must use is_remote=True
            # and a valid span_id so the SDK treats it as a real parent.
            import random

            otel_trace_id = int(custom_trace_id.replace("-", ""), 16)
            span_ctx = trace.SpanContext(
                trace_id=otel_trace_id,
                span_id=random.getrandbits(64),
                is_remote=True,
                trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
            )
            seeded_span = trace.NonRecordingSpan(span_ctx)
            isolated_ctx = trace.set_span_in_context(seeded_span, context.Context())
            otel_context_reset_token = self._otel_context.set(isolated_ctx)
        else:
            new_span_chain = current_span_chain + f"::{span_id}"
            trace_id_reset_token = None
            stored_ctx = self._otel_context.get()
            if stored_ctx is None:
                isolated_ctx = context.Context()
            else:
                isolated_ctx = stored_ctx
            otel_context_reset_token = None

        spans_reset_token = self._spans.set(new_span_chain)
        attributes_reset_token = self._attributes.set(new_attributes)

        otel_tracer = self._otel_tracer

        otel_span = otel_tracer.start_span(
            name=span_id,
            context=isolated_ctx,
            attributes=dict(new_attributes),
        )

        new_ctx = trace.set_span_in_context(otel_span, isolated_ctx)
        ctx_token = self._otel_context.set(new_ctx)

        span_reset_token = self._current_span.set(otel_span)

        try:
            with trace.use_span(otel_span, end_on_exit=True):
                yield
        except Exception as e:
            otel_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            otel_span.record_exception(e)
            raise
        finally:
            self._spans.reset(spans_reset_token)
            self._attributes.reset(attributes_reset_token)
            self._current_span.reset(span_reset_token)
            self._otel_context.reset(ctx_token)
            if trace_id_reset_token is not None:
                self._trace_id.reset(trace_id_reset_token)
            if otel_context_reset_token is not None:
                self._otel_context.reset(otel_context_reset_token)

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
            for key, value in attributes.items():
                current_span.set_attribute(key, value)

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
        transformed_attributes = dict(attributes)

        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            current_span.add_event(name, transformed_attributes)

    @override
    def flush(self) -> None:
        if hasattr(self, "_processor") and self._processor:
            try:
                self._processor.force_flush()
            except Exception as e:
                _logger.warning(f"Failed to flush spans: {e}")
