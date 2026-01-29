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

from opentelemetry.trace import Span
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)

from parlant.core.tracer import Tracer, AttributeValue

logger = logging.getLogger(__name__)


class EmcieSpanProcessor(BatchSpanProcessor):
    """Custom span processor that filters and transforms spans before export."""

    def __init__(self, endpoint: str, api_key: str | None = None):
        headers = {}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
            logger.debug("API key configured for OTLP export")
        else:
            logger.warning("No API key provided for OTLP export")

        # Use gRPC for all endpoints (production ready)
        logger.debug(f"Creating OTLPSpanExporter (gRPC) with headers: {list(headers.keys())}")
        span_exporter = HTTPOTLPSpanExporter(
            endpoint=endpoint,
            headers=headers,
        )

        # Initialize BatchSpanProcessor with our exporter
        super().__init__(
            span_exporter=span_exporter,
            schedule_delay_millis=1000,  # Export every 1 second
            max_queue_size=1000,
            max_export_batch_size=100,
        )
        logger.debug("EmcieSpanProcessor initialized successfully")

        # Rate limiting for log messages to avoid spam
        self._last_error_log = 0.0
        self._error_log_interval = 60.0

    def _should_export_span(self, span: ReadableSpan) -> bool:
        """Determine if a span should be exported based on our filtering rules."""
        attributes = dict(span.attributes) if span.attributes else {}

        if attributes.get("http.request.operation") == "create_event":
            return True

        return False


class EmcieTracer(Tracer):
    """Tracer that exports selected traces to Emcie via OTLP gRPC."""

    def __init__(self) -> None:
        # Use gRPC endpoint format (host:port instead of HTTP URL)
        self._endpoint = os.getenv("EMCIE_OTEL_URL", "https://api.emcie.xyz/v1/traces")

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

        self._processor: EmcieSpanProcessor

    async def __aenter__(self) -> Self:
        try:
            self._processor = EmcieSpanProcessor(
                endpoint=self._endpoint,
                api_key=self._api_key,
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize EmcieTracer processor: {e}. Continuing without export."
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        logger.debug("EmcieTracer shutting down")
        # Force flush on exit
        if self._processor:
            try:
                self._processor.force_flush()
                self._processor.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Error during EmcieTracer shutdown: {e}")

        return False

    def _queue_for_export(self, span: ReadableSpan) -> None:
        """Send ReadableSpan to processor for export."""
        self._processor.on_end(span)

    def _should_export_span(
        self, span_name: str, attributes: Mapping[str, AttributeValue] | None = None
    ) -> bool:
        # Only export http.request spans with specific operation
        if span_name == "http.request":
            if not attributes:
                return False
            operation = attributes.get("http.request.operation")
            return operation == "create_event"

        # Reject all other span types
        return False

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        current_span_chain = self._spans.get()
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        # Determine if this is a root span
        if not current_span_chain:
            new_span_chain = span_id
            custom_trace_id = self._generate_trace_id()
            trace_id_reset_token = self._trace_id.set(custom_trace_id)
        else:
            new_span_chain = current_span_chain + f"::{span_id}"
            trace_id_reset_token = None

        spans_reset_token = self._spans.set(new_span_chain)
        attributes_reset_token = self._attributes.set(new_attributes)

        # Create resource with service name and API key
        resource_attributes = {
            "service.name": "parlant-emcie-tracer",
        }
        resource_attributes["api_key"] = self._api_key

        resource = Resource.create(resource_attributes)
        tracer_provider = TracerProvider(resource=resource)
        otel_tracer = tracer_provider.get_tracer(__name__)

        with otel_tracer.start_as_current_span(span_id) as otel_span:
            # Set attributes on the real span
            for key, value in new_attributes.items():
                otel_span.set_attribute(key, value)

            span_reset_token = self._current_span.set(otel_span)

            try:
                yield
            except Exception as e:
                otel_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                # Reset context variables
                self._spans.reset(spans_reset_token)
                self._attributes.reset(attributes_reset_token)
                self._current_span.reset(span_reset_token)
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
        if self._processor:
            try:
                self._processor.force_flush()
            except Exception as e:
                logger.warning(f"Failed to flush spans: {e}")
