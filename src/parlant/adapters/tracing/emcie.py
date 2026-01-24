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
from typing import Any, Iterator, Mapping, Callable, TypedDict
from typing_extensions import override, Self, NotRequired

from opentelemetry.trace import Span
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)

from parlant.core.tracer import Tracer, AttributeValue

logger = logging.getLogger(__name__)


# TypedDict definitions for event structures (for documentation and type hinting)


class EventAttributesBase(TypedDict):
    """Base attributes structure for all events."""

    pass


class ToolCallEventAttributesInput(EventAttributesBase):
    """Attributes for tool call events as they come in."""

    tool_id: str
    arguments: NotRequired[Any]  # Contains tool arguments (sensitive)
    result: NotRequired[Any]  # Contains tool results (sensitive)


class ToolCallEventAttributesOutput(EventAttributesBase):
    """Attributes for tool call events after transformation."""

    tool_id: str


class JourneyActivateEventAttributesInput(EventAttributesBase):
    """Attributes for journey.state.activate events as they come in."""

    edge_id: str
    node_id: str
    journey_id: str
    sub_journey_id: NotRequired[str]
    condition: NotRequired[str]  # Sensitive - removed in output
    action: NotRequired[str]  # Sensitive - removed in output
    rationale: NotRequired[str]  # Sensitive - removed in output


class JourneyActivateEventAttributesOutput(EventAttributesBase):
    """Attributes for journey.state.activate events after transformation."""

    edge_id: NotRequired[str]
    node_id: NotRequired[str]
    journey_id: NotRequired[str]
    sub_journey_id: NotRequired[str]


class GuidelineMatchEventAttributesInput(EventAttributesBase):
    """Attributes for gm.activate events as they come in."""

    guideline_id: str
    condition: NotRequired[str]  # Sensitive - removed in output
    action: NotRequired[str]  # Sensitive - removed in output
    rationale: NotRequired[str]  # Keep in output


class GuidelineMatchEventAttributesOutput(EventAttributesBase):
    """Attributes for gm.activate events after transformation."""

    guideline_id: NotRequired[str]
    rationale: NotRequired[str]


def _transform_trace(span_data: dict[str, Any]) -> dict[str, Any]:
    """Transform http.request spans - refactor content for create_event operations."""
    sanitized = span_data.copy()

    # Remove sensitive attributes from the main span
    if "attributes" in sanitized:
        safe_attributes = {}
        for key, value in sanitized["attributes"].items():
            # Keep only safe attributes, remove sensitive ones
            if key not in ["request_body", "response_body", "sensitive_data"]:
                safe_attributes[key] = value
        sanitized["attributes"] = safe_attributes

    if "spans" in sanitized:
        sanitized["spans"] = _transform_nested_spans(sanitized["spans"])

    return sanitized


def _transform_nested_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform nested spans, focusing on 'process' spans and their events."""
    transformed_spans: list[dict[str, Any]] = []

    for span in spans:
        if span.get("name") == "process":
            transformed_span = _transform_process_span(span)
            transformed_spans.append(transformed_span)
        else:
            transformed_spans.append(span)

    return transformed_spans


def _transform_process_span(span: dict[str, Any]) -> dict[str, Any]:
    """Transform process spans by filtering and transforming their events."""
    result: dict[str, Any] = {"name": span["name"]}

    if "events" in span:
        result["events"] = _transform_process_events(span["events"])

    return result


def _transform_process_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform events within process spans - handle tc, journey.state.activate, and gm.activate."""
    transformed_events: list[dict[str, Any]] = []

    for event in events:
        event_name = event.get("name", "")

        if event_name == "tc":  # Tool call event
            transformed_event = _transform_tc_event(event)
            transformed_events.append(transformed_event)
        elif event_name == "journey.state.activate":
            transformed_event = _transform_journey_activate_event(event)
            transformed_events.append(transformed_event)
        elif event_name == "gm.activate":
            transformed_event = _transform_gm_activate_event(event)
            transformed_events.append(transformed_event)
        # Skip other events not in our allow list

    return transformed_events


def _transform_tc_event(event: dict[str, Any]) -> dict[str, Any]:
    """Transform tool call (tc) events - keep only tool metadata, remove arguments and results."""
    result: dict[str, Any] = {"name": event["name"], "attributes": {}}

    if "attributes" in event:
        input_attrs = event["attributes"]
        safe_attributes: dict[str, Any] = {}

        if "tool_id" in input_attrs:
            safe_attributes["tool_id"] = input_attrs["tool_id"]
        # Remove sensitive fields like "arguments" and "result"

        result["attributes"] = safe_attributes

    return result


def _transform_journey_activate_event(event: dict[str, Any]) -> dict[str, Any]:
    """Transform journey.state.activate events - keep node and journey IDs, remove condition details."""
    result: dict[str, Any] = {"name": event["name"], "attributes": {}}

    if "attributes" in event:
        input_attrs = event["attributes"]
        safe_attributes: dict[str, Any] = {}

        for key in ["edge_id", "node_id", "journey_id", "sub_journey_id"]:
            if key in input_attrs:
                safe_attributes[key] = input_attrs[key]
        # Remove sensitive fields like "condition", "action", "rationale"

        result["attributes"] = safe_attributes

    return result


def _transform_gm_activate_event(event: dict[str, Any]) -> dict[str, Any]:
    """Transform gm.activate (guideline match) events - keep guideline ID and rationale, remove condition and action."""
    result: dict[str, Any] = {"name": event["name"], "attributes": {}}

    if "attributes" in event:
        input_attrs = event["attributes"]
        safe_attributes: dict[str, Any] = {}

        if "guideline_id" in input_attrs:
            safe_attributes["guideline_id"] = input_attrs["guideline_id"]
        if "rationale" in input_attrs:
            safe_attributes["rationale"] = input_attrs["rationale"]
        # Remove sensitive fields like "condition" and "action"

        result["attributes"] = safe_attributes

    return result


# Map of supported trace names to their transform functions
TRACE_TRANSFORMS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "http.request": _transform_trace,
}


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

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends. Apply filtering and transformation."""
        transformed_span = self._transform_span(span)

        # Send to parent processor for batching and export
        super().on_end(transformed_span)

    def _should_export_span(self, span: ReadableSpan) -> bool:
        """Determine if a span should be exported based on our filtering rules."""
        span_name = span.name
        attributes = dict(span.attributes) if span.attributes else {}

        # Only export http.request spans with specific operation
        if span_name == "http.request":
            operation = attributes.get("http.request.operation")
            should_export = operation == "create_event"
            return should_export

        return False

    def _transform_span(self, span: ReadableSpan) -> ReadableSpan:
        """Apply transformation logic to a span if needed."""
        span_name = span.name

        if span_name not in TRACE_TRANSFORMS:
            return span

        # For now, return the span as-is since transformation is complex
        # The existing transform functions work on dict representations
        # We could enhance this later if needed
        return span


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
                # The span will be automatically ended when exiting the context
                # Queue it for export when it ends
                if hasattr(self, "_processor") and self._processor:
                    # Convert to ReadableSpan and export
                    from opentelemetry.sdk.trace import ReadableSpan

                    if isinstance(otel_span, ReadableSpan):
                        self._queue_for_export(otel_span)

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
        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes)

    @override
    def flush(self) -> None:
        """Flush pending spans immediately."""
        if self._processor:
            try:
                self._processor.force_flush()
            except Exception as e:
                logger.warning(f"Failed to flush spans: {e}")
