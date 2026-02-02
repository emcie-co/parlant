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
import logging
import os
from contextlib import contextmanager
from types import TracebackType
from typing import Iterator, Mapping
from typing_extensions import override, Self

from opentelemetry import trace, context
from opentelemetry.trace import Span, set_tracer_provider
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)

from parlant.core.tracer import Tracer, AttributeValue

logger = logging.getLogger(__name__)


class EmcieTracer(Tracer):
    def __init__(self) -> None:
        # Use gRPC endpoint format (host:port instead of HTTP URL)
        self._endpoint = f"{os.getenv('EMCIE_BASE_URL', 'https://api.emcie.co')}/v1/traces"

        self._api_key = os.getenv(
            "EMCIE_API_KEY", "sk-mc-qtSvL271_LWbJQoDFrtRFYTyl2lsFFDA2qx3Amb_0wn4Gt-bIQ"
        )

        # Context variables for tracking spans (same as before)
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

        # Store the OpenTelemetry context for this tracer to maintain isolation
        self._otel_context = contextvars.ContextVar[context.Context | None](
            "emcie_tracer_otel_context",
            default=None,
        )

    async def __aenter__(self) -> Self:
        # Setup headers for API authentication
        headers = {}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
            logger.debug("API key configured for OTLP export")
        else:
            logger.warning("No API key provided for OTLP export")

        # Create OTLP exporter
        span_exporter = HTTPOTLPSpanExporter(
            endpoint=self._endpoint,
            headers=headers,
        )

        # Create processor with custom filtering
        processor = BatchSpanProcessor(
            span_exporter=span_exporter,
            schedule_delay_millis=1000,  # Export every 1 second
            max_queue_size=1000,
            max_export_batch_size=100,
        )

        # Override the on_end method to add filtering
        original_on_end = processor.on_end

        def filtered_on_end(span: ReadableSpan) -> None:
            """Custom on_end with filtering logic."""
            attributes = dict(span.attributes) if span.attributes else {}

            if attributes.get("http.request.operation") == "create_event":
                original_on_end(span)

        setattr(processor, "on_end", filtered_on_end)

        resource_attributes = {
            "service.name": "parlant-emcie-tracer",
            "api_key": self._api_key,
        }
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
        # Force flush on exit
        if self._processor:
            try:
                self._processor.force_flush()
                self._processor.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Error during EmcieTracer shutdown: {e}")

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

        # Determine if this is a root span
        if not current_span_chain:
            new_span_chain = span_id
            custom_trace_id = self._generate_trace_id()
            trace_id_reset_token = self._trace_id.set(custom_trace_id)
            # Create isolated context for root span - don't use global context
            isolated_ctx = context.Context()
            otel_context_reset_token = self._otel_context.set(isolated_ctx)
        else:
            new_span_chain = current_span_chain + f"::{span_id}"
            trace_id_reset_token = None
            # Use the stored context from parent span
            stored_ctx = self._otel_context.get()
            if stored_ctx is None:
                # Fallback to isolated context if something went wrong
                isolated_ctx = context.Context()
            else:
                isolated_ctx = stored_ctx
            otel_context_reset_token = None

        spans_reset_token = self._spans.set(new_span_chain)
        attributes_reset_token = self._attributes.set(new_attributes)

        otel_tracer = self._otel_tracer

        # Use start_span with explicit context instead of start_as_current_span
        # This prevents picking up spans from other tracers in the global context
        otel_span = otel_tracer.start_span(name=span_id, context=isolated_ctx)

        # Set attributes on the span
        for key, value in new_attributes.items():
            otel_span.set_attribute(key, value)

        # Update the context with this span for child spans to use
        new_ctx = trace.set_span_in_context(otel_span, isolated_ctx)
        ctx_token = self._otel_context.set(new_ctx)

        span_reset_token = self._current_span.set(otel_span)

        try:
            # Use the span without setting it in global context
            with trace.use_span(otel_span, end_on_exit=True):
                yield
        except Exception as e:
            otel_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            otel_span.record_exception(e)
            raise
        finally:
            # Reset context variables
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

        # Update current span with new attributes
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
        if name in ["journey.state.activate", "journey.state.skip"]:
            allowed_keys = {"edge_id", "node_id", "journey_id", "sub_journey_id", "journey_path"}
            transformed_attributes = {k: v for k, v in attributes.items() if k in allowed_keys}
        elif name in ["gm.activate", "gm.skip"]:
            allowed_keys = {"guideline_id", "rationale"}
            transformed_attributes = {k: v for k, v in attributes.items() if k in allowed_keys}
        elif name == "tc":
            allowed_keys = {"tool_id", "rationale", "arguments", "result"}
            transformed_attributes = {k: v for k, v in attributes.items() if k in allowed_keys}
        else:
            transformed_attributes = dict(attributes)

        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            current_span.add_event(name, transformed_attributes)

    @override
    def flush(self) -> None:
        """Flush pending spans immediately."""
        if hasattr(self, "_processor") and self._processor:
            try:
                self._processor.force_flush()
            except Exception as e:
                logger.warning(f"Failed to flush spans: {e}")
