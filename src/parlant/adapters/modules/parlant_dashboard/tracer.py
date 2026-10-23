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

"""ParlantDashboardTracer — OpenTelemetry-backed tracer for Parlant Dashboard."""

import asyncio
import contextvars
import os
from contextlib import ExitStack, contextmanager
from collections.abc import Generator
from types import TracebackType
from typing import Mapping

from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, set_tracer_provider
from typing_extensions import Self, override

from parlant.core.loggers import Logger
from parlant.core.tracer import AttributeValue, Tracer

from parlant.adapters.modules.parlant_dashboard.config import _get_dashboard_otel_url

_ACTIVITY_EVENT_SPAN_NAME = "parlant.activity.event"
_EXPORT_SCHEDULE_DELAY_MILLIS = 200

# The Dashboard's collector finalizes a trace after 30s without new spans, and a
# turn emits none while a tool runs. While an engine process is open, an
# event-less span every few seconds keeps its trace live. The collector builds
# activity only from span events, so these add nothing to what it shows.
_KEEPALIVE_SPAN_NAME = "parlant.trace.keepalive"
_DEFAULT_KEEPALIVE_INTERVAL_SECONDS = 10.0
# The span names the collector treats as a turn's engine process (its root).
_ENGINE_PROCESS_SPAN_NAMES = frozenset({"engine.process", "compass.engine.process", "process"})


def _is_exportable_request(attributes: Mapping[str, object]) -> bool:
    return attributes.get("http.request.operation") == "create_event" or (
        attributes.get("rpc.system") == "parlant.tunnel"
        and attributes.get("rpc.method") == "sessions.create_event"
    )


class ParlantDashboardTracer(Tracer):
    def __init__(
        self,
        project_id: str = "",
        logger: Logger | None = None,
        keepalive_interval_seconds: float = _DEFAULT_KEEPALIVE_INTERVAL_SECONDS,
    ) -> None:
        """keepalive_interval_seconds: how often an open engine process sends a
        keep-alive span, so the Dashboard keeps showing a long turn live. Keep it
        well under the collector's 30s idle timeout; 0 turns keep-alives off."""
        self._logger = logger
        self._project_id = project_id
        self._keepalive_interval_seconds = keepalive_interval_seconds

        self._endpoint = os.getenv(
            "PARLANT_DASHBOARD_OTEL_ENDPOINT",
            f"{_get_dashboard_otel_url()}/v1/traces",
        )

        self._project_token = os.getenv("PARLANT_DASHBOARD_PROJECT_TOKEN", "")

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
            "parlant_dashboard_tracer_otel_context",
            default=None,
        )

    async def __aenter__(self) -> Self:
        headers = {}
        if self._project_token:
            headers["authorization"] = f"Bearer {self._project_token}"
        else:
            self._log_info(
                "Parlant Dashboard tracing is not configured. Learn more at https://dashboard.parlant.io"
            )

        span_exporter = HTTPOTLPSpanExporter(
            endpoint=self._endpoint,
            headers=headers,
        )

        processor = BatchSpanProcessor(
            span_exporter=span_exporter,
            schedule_delay_millis=_EXPORT_SCHEDULE_DELAY_MILLIS,
            max_queue_size=1000,
            max_export_batch_size=100,
        )

        original_on_end = processor.on_end

        def filtered_on_end(span: ReadableSpan) -> None:
            attributes = dict(span.attributes) if span.attributes else {}
            if _is_exportable_request(attributes):
                original_on_end(span)

        setattr(processor, "on_end", filtered_on_end)

        resource_attributes: dict[str, str] = {
            "service.name": "parlant-dashboard-tracer",
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
                self._log_warning(f"Error during ParlantDashboardTracer shutdown: {e}")

        return False

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]:
        with ExitStack() as context_stack:
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
                context_stack.callback(self._trace_id.reset, trace_id_reset_token)

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
                context_stack.callback(self._otel_context.reset, otel_context_reset_token)
            else:
                new_span_chain = current_span_chain + f"::{span_id}"
                stored_ctx = self._otel_context.get()
                if stored_ctx is None:
                    isolated_ctx = context.Context()
                else:
                    isolated_ctx = stored_ctx

            spans_reset_token = self._spans.set(new_span_chain)
            context_stack.callback(self._spans.reset, spans_reset_token)
            attributes_reset_token = self._attributes.set(new_attributes)
            context_stack.callback(self._attributes.reset, attributes_reset_token)

            otel_span = self._otel_tracer.start_span(
                name=span_id,
                context=isolated_ctx,
                attributes=dict(new_attributes),
            )

            with ExitStack() as span_stack:
                span_stack.callback(otel_span.end)

                if span_id in _ENGINE_PROCESS_SPAN_NAMES:
                    keepalive = self._start_keepalive(otel_span, isolated_ctx)
                    if keepalive is not None:
                        span_stack.callback(keepalive.cancel)

                try:
                    new_ctx = trace.set_span_in_context(otel_span, isolated_ctx)
                    ctx_token = self._otel_context.set(new_ctx)
                    context_stack.callback(self._otel_context.reset, ctx_token)

                    span_reset_token = self._current_span.set(otel_span)
                    context_stack.callback(self._current_span.reset, span_reset_token)

                    with trace.use_span(
                        otel_span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        yield
                except Exception as e:
                    otel_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    otel_span.record_exception(e)
                    raise

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Generator[None, None, None]:
        with ExitStack() as stack:
            current_attributes = self._attributes.get()
            new_attributes = {**current_attributes, **attributes}

            attributes_reset_token = self._attributes.set(new_attributes)
            stack.callback(self._attributes.reset, attributes_reset_token)

            current_span = self._current_span.get()
            if current_span and current_span.is_recording():
                for key, value in attributes.items():
                    current_span.set_attribute(key, value)

            yield

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
        current_context = self._otel_context.get()
        if not current_span or not current_span.is_recording() or current_context is None:
            return

        event_span = self._otel_tracer.start_span(
            name=_ACTIVITY_EVENT_SPAN_NAME,
            context=current_context,
            attributes=dict(self._attributes.get()),
        )
        try:
            event_span.add_event(name, dict(attributes))
        finally:
            event_span.end()

    def _start_keepalive(
        self,
        process_span: Span,
        parent_context: context.Context,
    ) -> asyncio.Task[None] | None:
        """Keep an open engine process's trace live until the span ends. Fail-open:
        tracing must never affect the turn it traces."""
        interval = self._keepalive_interval_seconds
        if interval <= 0:
            return None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None

        span_context = trace.set_span_in_context(process_span, parent_context)

        async def beat() -> None:
            while True:
                await asyncio.sleep(interval)
                if not process_span.is_recording():
                    return
                try:
                    attributes = dict(getattr(process_span, "attributes", None) or {})
                    if not _is_exportable_request(attributes):
                        continue
                    self._otel_tracer.start_span(
                        name=_KEEPALIVE_SPAN_NAME,
                        context=span_context,
                        attributes=attributes,
                    ).end()
                except Exception as e:
                    self._log_warning(f"Failed to send a trace keep-alive: {e}")

        try:
            return loop.create_task(beat())
        except Exception as e:
            self._log_warning(f"Failed to start trace keep-alives: {e}")
            return None

    @override
    def flush(self) -> None:
        if hasattr(self, "_processor") and self._processor:
            try:
                self._processor.force_flush()
            except Exception as e:
                self._log_warning(f"Failed to flush spans: {e}")

    def _log_info(self, message: str) -> None:
        if self._logger:
            self._logger.info(message)

    def _log_warning(self, message: str) -> None:
        if self._logger:
            self._logger.warning(message)
