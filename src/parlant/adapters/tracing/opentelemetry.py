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
# WITHOUT WARRANTIES OR CONDITIONS OF AttributeValue KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextvars
from types import TracebackType
from typing import Mapping
from typing_extensions import Literal, Self, override

from opentelemetry import trace, metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.trace import Span as OTelSpanAPI, Tracer as OTelTracer
from opentelemetry.metrics import Meter as OTelMeterAPI, Counter, Histogram
from parlant.core.common import generate_id
from parlant.core.tracing import AttributeValue, Meter, Span, Tracer


class OpenTelemetryMeter(Meter):
    def __init__(self, otel_meter: OTelMeterAPI) -> None:
        self._otel_meter = otel_meter
        self._counters: dict[str, Counter] = {}
        self._hists: dict[str, Histogram] = {}

    def _counter(self, name: str) -> Counter:
        c = self._counters.get(name)

        if c is None:
            c = self._otel_meter.create_counter(name)
            self._counters[name] = c

        return c

    def _hist(self, name: str) -> Histogram:
        h = self._hists.get(name)

        if h is None:
            h = self._otel_meter.create_histogram(name)
            self._hists[name] = h

        return h

    @override
    async def record_counter(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        self._counter(name).add(value, attributes or {})

    @override
    async def record_histogram(
        self,
        name: str,
        value: float,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        self._hist(name).record(value, attributes or {})


class OpenTelemetrySpan(Span):
    def __init__(self, otel_span: OTelSpanAPI) -> None:
        self._otel_span = otel_span

    @override
    async def set_attribute(
        self,
        key: str,
        value: AttributeValue,
    ) -> None:
        self._otel_span.set_attribute(key, value)

    @override
    async def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        self._otel_span.add_event(name, attributes=attributes)

    @override
    async def end(
        self,
        error: Exception | None = None,
    ) -> None:
        if error is not None:
            self._otel_span.record_exception(error)
            self._otel_span.set_status(status=trace.StatusCode.ERROR, description=str(error))

        self._otel_span.end()


class OpenTelemetryTracer(Tracer):
    def __init__(
        self,
        service_name: str = "parlant",
        exporter: Literal["remote", "local"] = "local",
        otlp_headers: Mapping[str, str] | None = None,
        otlp_insecure: bool = False,
        otlp_endpoint: str | None = None,
    ):
        self._service_name = service_name
        self._otlp_endpoint = otlp_endpoint
        self._otlp_headers = dict(otlp_headers) if otlp_headers else {}
        self._otlp_insecure = otlp_insecure

        self._tracer_provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._tracer: OTelTracer
        self._meter: OpenTelemetryMeter  # wrapper

        self._instance_id = generate_id()
        self._span_stack: contextvars.ContextVar[tuple[OpenTelemetrySpan, ...]] = (
            contextvars.ContextVar(
                f"otel_span_stack_{self._instance_id}",
                default=(),
            )
        )

        if exporter == "local":
            self._span_exporter: SpanExporter = ConsoleSpanExporter()
            self._metric_exporter: MetricExporter = ConsoleMetricExporter()  # type: ignore[name-defined]
        else:
            self._span_exporter = OTLPSpanExporter(
                endpoint=self._otlp_endpoint,
                headers=self._otlp_headers,
                insecure=self._otlp_insecure,
            )
            self._metric_exporter = OTLPMetricExporter(
                endpoint=self._otlp_endpoint,
                headers=self._otlp_headers,
                insecure=self._otlp_insecure,
            )

    async def __aenter__(self) -> Self:
        resource = Resource.create(
            {"service.name": self._service_name, "service.instance.id": self._instance_id}
        )

        # Traces
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(BatchSpanProcessor(self._span_exporter))
        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(self._service_name)

        # Metrics
        reader = PeriodicExportingMetricReader(self._metric_exporter)
        self._meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(self._meter_provider)

        otel_meter = self._meter_provider.get_meter(self._service_name)
        self._meter = OpenTelemetryMeter(otel_meter)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._tracer_provider is not None:
            # ensure export before shutdown
            self._tracer_provider.force_flush()
            self._tracer_provider.shutdown()  # type: ignore[no-untyped-call]

        if self._meter_provider is not None:
            self._meter_provider.force_flush()
            self._meter_provider.shutdown()

        return False

    @property
    @override
    def meter(self) -> Meter:
        return self._meter

    @override
    async def start_span(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Span:
        otel_span = self._tracer.start_span(name, attributes=attributes)
        span = OpenTelemetrySpan(otel_span)

        stack = self._span_stack.get()
        self._span_stack.set(stack + (span,))
        return span

    @override
    async def end_span(self, error: Exception | None = None) -> None:
        stack = self._span_stack.get()

        if not stack:
            return

        top: OpenTelemetrySpan = stack[-1]
        await top.end(error)
        self._span_stack.set(stack[:-1])
