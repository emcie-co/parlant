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

import asyncio
import contextvars
import logging
import os
import time
from collections import deque
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Iterator, Mapping, Optional, Callable, TypedDict
from typing_extensions import override, Self, NotRequired

import grpc
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceStub
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from opentelemetry.trace import StatusCode

from parlant.core.common import generate_id
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


# Note: Using dict[str, Any] in function signatures for compatibility with existing code
# The TypedDict definitions above serve as documentation of the expected structure


def _transform_trace(span_data: dict[str, Any]) -> dict[str, Any]:
    """Transform http.request spans - refactor content for create_event operations."""
    sanitized = span_data.copy()

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

        for key in ["node_id", "journey_id", "sub_journey_id"]:
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


class EmcieSpanData:
    """Internal representation of a span for OTLP export."""

    def __init__(
        self,
        span_name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str = "",
        attributes: Mapping[str, AttributeValue] | None = None,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
        status_code: StatusCode = StatusCode.UNSET,
        status_message: str = "",
    ):
        self.span_name = span_name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.attributes = attributes or {}
        self.start_time_ns = start_time_ns or time.time_ns()
        self.end_time_ns = end_time_ns
        self.status_code = status_code
        self.status_message = status_message


class EmcieExporter:
    """Handles exporting spans to Emcie via OTLP gRPC."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        insecure: bool = False,
        timeout: float = 5.0,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.insecure = insecure
        self.timeout = timeout
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "parlant")

        # Rate limiting for log messages to avoid spam
        self._last_error_log = 0.0
        self._error_log_interval = 60.0  # Log errors at most once per minute

    def _create_channel(self) -> grpc.Channel:
        """Create gRPC channel with appropriate security settings."""
        if self.insecure:
            return grpc.insecure_channel(self.endpoint)
        else:
            credentials = grpc.ssl_channel_credentials()
            return grpc.secure_channel(self.endpoint, credentials)

    def _convert_attribute_value(self, value: AttributeValue) -> AnyValue:
        """Convert Parlant AttributeValue to OTLP AnyValue."""
        any_value = AnyValue()

        if isinstance(value, str):
            any_value.string_value = value
        elif isinstance(value, bool):
            any_value.bool_value = value
        elif isinstance(value, int):
            any_value.int_value = value
        elif isinstance(value, float):
            any_value.double_value = value
        elif isinstance(value, list):
            # Handle sequences
            if value and isinstance(value[0], str):
                any_value.array_value.values.extend([AnyValue(string_value=v) for v in value])
            elif value and isinstance(value[0], bool):
                any_value.array_value.values.extend([AnyValue(bool_value=v) for v in value])
            elif value and isinstance(value[0], int):
                any_value.array_value.values.extend([AnyValue(int_value=v) for v in value])
            elif value and isinstance(value[0], float):
                any_value.array_value.values.extend([AnyValue(double_value=v) for v in value])
        else:
            # Fallback to string representation
            any_value.string_value = str(value)

        return any_value

    def _convert_span_to_otlp(self, span_data: EmcieSpanData) -> Span:
        """Convert EmcieSpanData to OTLP Span."""
        span = Span()

        # Convert trace_id and span_id from hex string to bytes
        span.trace_id = bytes.fromhex(span_data.trace_id.replace("-", "")[:32].ljust(32, "0"))
        span.span_id = bytes.fromhex(span_data.span_id.replace("-", "")[:16].ljust(16, "0"))

        if span_data.parent_span_id:
            span.parent_span_id = bytes.fromhex(
                span_data.parent_span_id.replace("-", "")[:16].ljust(16, "0")
            )

        span.name = span_data.span_name
        span.start_time_unix_nano = span_data.start_time_ns

        if span_data.end_time_ns:
            span.end_time_unix_nano = span_data.end_time_ns
        else:
            span.end_time_unix_nano = time.time_ns()

        # Convert attributes
        for key, value in span_data.attributes.items():
            kv = KeyValue()
            kv.key = key
            kv.value.CopyFrom(self._convert_attribute_value(value))
            span.attributes.append(kv)

        # Set status
        if span_data.status_code != StatusCode.UNSET:
            span.status.code = span_data.status_code.value
            if span_data.status_message:
                span.status.message = span_data.status_message

        return span

    async def export_spans(self, spans: list[EmcieSpanData]) -> bool:
        """Export spans via OTLP gRPC. Returns True on success."""
        if not spans:
            return True

        try:
            # Run the blocking gRPC call in a thread pool
            return await asyncio.to_thread(self._export_spans_sync, spans)
        except Exception as e:
            self._log_error_rate_limited(f"Failed to export spans: {e}")
            return False

    def _export_spans_sync(self, spans: list[EmcieSpanData]) -> bool:
        """Synchronous span export implementation."""
        try:
            # Create request
            request = ExportTraceServiceRequest()
            traces_data = request.resource_spans.add()

            # Set resource
            traces_data.resource.CopyFrom(
                Resource(
                    attributes=[
                        KeyValue(key="service.name", value=AnyValue(string_value=self.service_name))
                    ]
                )
            )

            # Add spans to scope spans
            scope_spans = traces_data.scope_spans.add()
            for span_data in spans:
                otlp_span = self._convert_span_to_otlp(span_data)
                scope_spans.spans.append(otlp_span)

            # Create channel and stub
            with self._create_channel() as channel:
                stub = TraceServiceStub(channel)  # type: ignore[no-untyped-call]

                # Prepare metadata
                metadata = []
                if self.api_key:
                    metadata.append(("authorization", f"Bearer {self.api_key}"))

                # Make the call
                stub.Export(request, timeout=self.timeout, metadata=metadata)

                return True

        except Exception as e:
            self._log_error_rate_limited(f"gRPC export failed: {e}")
            return False

    def _log_error_rate_limited(self, message: str) -> None:
        """Log error messages with rate limiting to avoid spam."""
        now = time.time()
        if now - self._last_error_log >= self._error_log_interval:
            logger.warning(message)
            self._last_error_log = now


class EmcieTracer(Tracer):
    """Tracer that exports selected traces to Emcie via OTLP gRPC."""

    def __init__(self) -> None:
        # Configuration from environment
        self._enabled = os.getenv("EMCIE_TRACE_EXPORT_ENABLED", "true").lower() == "true"
        self._endpoint = os.getenv("EMCIE_OTEL_URL", "")
        self._api_key = os.getenv("EMCIE_API_KEY")
        self._insecure = os.getenv("EMCIE_OTEL_INSECURE", "false").lower() == "true"
        self._timeout = float(os.getenv("EMCIE_OTEL_TIMEOUT_SECONDS", "5"))

        # Context variables for tracking spans
        self._spans = contextvars.ContextVar[str](
            "emcie_tracer_spans",
            default="",
        )

        self._attributes = contextvars.ContextVar[Mapping[str, AttributeValue]](
            "emcie_tracer_attributes",
            default={},
        )

        self._trace_id = contextvars.ContextVar[str](
            "emcie_tracer_trace_id",
            default="",
        )

        # Internal state for tracking active spans
        self._active_spans = contextvars.ContextVar[dict[str, EmcieSpanData]](
            "emcie_active_spans",
            default={},
        )

        # Export infrastructure
        self._exporter: Optional[EmcieExporter] = None
        self._export_queue: deque[EmcieSpanData] = deque(maxlen=1000)  # Bounded queue
        self._dropped_spans_count = 0

        if self._enabled and self._endpoint:
            self._exporter = EmcieExporter(
                endpoint=self._endpoint,
                api_key=self._api_key,
                insecure=self._insecure,
                timeout=self._timeout,
            )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        await self._flush_async()
        return False

    def _should_export_span(
        self, span_name: str, attributes: Mapping[str, AttributeValue] | None = None
    ) -> bool:
        # For http.request spans, also check for the specific operation attribute
        if span_name == "http.request":
            if not attributes:
                return False
            operation = attributes.get("http.request.operation")
            return operation == "create_event"

        return True

    def _transform_span_data(self, span_name: str, span_data: EmcieSpanData) -> EmcieSpanData:
        """Apply transform function for the span based on its name."""
        if span_name not in TRACE_TRANSFORMS:
            return span_data

        transform_func = TRACE_TRANSFORMS[span_name]

        # Convert to dict for transformation
        span_dict = {
            "span_name": span_data.span_name,
            "trace_id": span_data.trace_id,
            "span_id": span_data.span_id,
            "parent_span_id": span_data.parent_span_id,
            "attributes": dict(span_data.attributes),
            "start_time_ns": span_data.start_time_ns,
            "end_time_ns": span_data.end_time_ns,
            "status_code": span_data.status_code,
            "status_message": span_data.status_message,
        }

        try:
            transformed_dict = transform_func(span_dict)

            # Create new span data with transformed attributes
            return EmcieSpanData(
                span_name=transformed_dict["span_name"],
                trace_id=transformed_dict["trace_id"],
                span_id=transformed_dict["span_id"],
                parent_span_id=transformed_dict["parent_span_id"],
                attributes=transformed_dict["attributes"],
                start_time_ns=transformed_dict["start_time_ns"],
                end_time_ns=transformed_dict["end_time_ns"],
                status_code=transformed_dict["status_code"],
                status_message=transformed_dict["status_message"],
            )
        except Exception as e:
            logger.warning(f"Transform failed for span {span_name}: {e}")
            return span_data

    def _queue_for_export(self, span_data: EmcieSpanData) -> None:
        """Add span to export queue if export is enabled."""
        if not self._enabled or not self._exporter:
            return

        if not self._should_export_span(span_data.span_name, span_data.attributes):
            return

        # Apply transform
        transformed_span = self._transform_span_data(span_data.span_name, span_data)

        # Add to queue (bounded)
        try:
            self._export_queue.append(transformed_span)
        except Exception:
            # Queue is full, increment counter
            self._dropped_spans_count += 1

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        current_spans = self._spans.get()
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        # Determine if this is a root span
        if not current_spans:
            new_spans = span_id
            custom_trace_id = generate_id({"strategy": "uuid4"})
            trace_id_reset_token = self._trace_id.set(custom_trace_id)
            parent_span_id = ""
        else:
            new_spans = current_spans + f"::{span_id}"
            custom_trace_id = self._trace_id.get()
            trace_id_reset_token = None
            # Get parent span ID from current span chain
            parent_span_id = (
                current_spans.split("::")[-1] if "::" in current_spans else current_spans
            )

        # Create span data
        span_data = EmcieSpanData(
            span_name=span_id,
            trace_id=custom_trace_id,
            span_id=generate_id({"strategy": "uuid4"}),
            parent_span_id=parent_span_id,
            attributes=new_attributes,
            start_time_ns=time.time_ns(),
        )

        # Track active span
        active_spans = self._active_spans.get()
        new_active_spans = {**active_spans, span_id: span_data}

        spans_reset_token = self._spans.set(new_spans)
        attributes_reset_token = self._attributes.set(new_attributes)
        active_spans_reset_token = self._active_spans.set(new_active_spans)

        try:
            yield
        except Exception as e:
            span_data.status_code = StatusCode.ERROR
            span_data.status_message = str(e)
            raise
        finally:
            # Finalize span
            span_data.end_time_ns = time.time_ns()

            # Queue for export
            self._queue_for_export(span_data)

            # Reset context variables
            self._spans.reset(spans_reset_token)
            self._attributes.reset(attributes_reset_token)
            self._active_spans.reset(active_spans_reset_token)
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

        # Update active spans with new attributes
        active_spans = self._active_spans.get()
        for span_data in active_spans.values():
            span_data.attributes = {**span_data.attributes, **attributes}

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

        # Update active spans
        active_spans = self._active_spans.get()
        for span_data in active_spans.values():
            span_data.attributes = {**span_data.attributes, name: value}

    @override
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        # Events are not implemented for Emcie export, but we don't error
        pass

    @override
    def flush(self) -> None:
        """Synchronous flush - starts async flush but doesn't wait."""
        if self._enabled and self._exporter and self._export_queue:
            # Schedule async flush
            asyncio.create_task(self._flush_async())  # type: ignore[no-untyped-call]

    async def _flush_async(self) -> None:
        """Flush pending spans asynchronously."""
        if not self._enabled or not self._exporter:
            return

        # Collect all queued spans
        spans_to_export = []
        while self._export_queue:
            try:
                spans_to_export.append(self._export_queue.popleft())
            except IndexError:
                break

        if spans_to_export:
            success = await self._exporter.export_spans(spans_to_export)
            if not success and self._dropped_spans_count > 0:
                logger.warning(f"Export failed. Total dropped spans: {self._dropped_spans_count}")
