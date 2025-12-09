from __future__ import annotations
import asyncio
import contextvars
import os
from types import TracebackType
from typing import AsyncGenerator, Mapping
from typing_extensions import override, Self
from contextlib import asynccontextmanager

from opentelemetry import metrics
from opentelemetry.metrics import Counter as OTelCounter, Histogram as OTelHistogram
from opentelemetry.sdk.metrics import (
    MeterProvider,
)
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as GrpcOTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as HttpOTLPMetricExporter,
)
from opentelemetry.util.re import parse_env_headers

from parlant.core.meter import Counter, DurationHistogram, Meter


class OpenTelemetryCounter(Counter):
    def __init__(self, otel_counter: OTelCounter) -> None:
        self._otel_counter = otel_counter

    @override
    async def increment(
        self,
        value: int,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self._otel_counter.add(value, attributes)


class OpenTelemetryHistogram(DurationHistogram):
    def __init__(self, otel_histogram: OTelHistogram) -> None:
        self._otel_histogram = otel_histogram
        # Context variable to store start time for recording
        self._start_time: contextvars.ContextVar[float | None] = contextvars.ContextVar(
            "histogram_start_time", default=None
        )

    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self._otel_histogram.record(value, attributes)

    @override
    @asynccontextmanager
    async def measure(
        self,
        attributes: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[None, None]:
        start_time = asyncio.get_running_loop().time()
        try:
            yield
        finally:
            duration = (
                asyncio.get_running_loop().time() - start_time
            ) * 1000  # Convert to milliseconds
            await self.record(duration, attributes)

    @override
    async def start_record(
        self,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        start_time = asyncio.get_running_loop().time()
        self._start_time.set(start_time)
        print(f"DEBUG: start_record set time {start_time}, context: {id(asyncio.current_task())}")

    @override
    async def end_record(
        self,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        start_time = self._start_time.get()
        print(f"DEBUG: end_record got time {start_time}, context: {id(asyncio.current_task())}")

        if start_time is None:
            raise ValueError("No start time recorded. Call start_record first.")

        end_time = asyncio.get_running_loop().time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds

        # Clear the start time
        self._start_time.set(None)

        await self.record(duration, attributes)


class OpenTelemetryMeter(Meter):
    def __init__(self) -> None:
        self._service_name = os.getenv("OTEL_SERVICE_NAME", "parlant")

        self._meter: metrics.Meter
        self._metric_exporter: GrpcOTLPMetricExporter | HttpOTLPMetricExporter
        self._meter_provider: MeterProvider
        self._histograms: dict[str, OpenTelemetryHistogram] = {}

    async def __aenter__(self) -> Self:
        resource = Resource.create({"service.name": self._service_name})

        endpoint = os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"]
        insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "false").lower() == "true"
        protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()

        match protocol:
            case "http/protobuf":
                self._metric_exporter = HttpOTLPMetricExporter(endpoint=endpoint)
            case "http/json":
                headers_string = os.environ.get(
                    "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
                    os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", ""),
                )
                headers = {
                    "Content-Type": "application/json",
                    **parse_env_headers(headers_string, liberal=True),
                }

                self._metric_exporter = HttpOTLPMetricExporter(
                    endpoint=endpoint,
                    headers=headers,
                )
            case "grpc":
                self._metric_exporter = GrpcOTLPMetricExporter(
                    endpoint=endpoint,
                    insecure=insecure,
                )
            case _:
                raise ValueError(f"Unsupported OTLP protocol: {protocol}")

        metric_reader = PeriodicExportingMetricReader(
            exporter=self._metric_exporter,
            export_interval_millis=3000,
        )
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        metrics.set_meter_provider(self._meter_provider)

        self._meter = metrics.get_meter(__name__)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._meter_provider.force_flush()
        self._meter_provider.shutdown()

        return False

    @override
    def create_counter(
        self,
        name: str,
        description: str,
    ) -> Counter:
        otel_counter = self._meter.create_counter(
            name=name,
            description=description,
        )

        return OpenTelemetryCounter(otel_counter)

    @override
    def create_custom_histogram(
        self,
        name: str,
        description: str,
        unit: str,
    ) -> OpenTelemetryHistogram:
        otel_histogram = self._meter.create_histogram(
            name=name,
            description=description,
            unit=unit,
        )
        histogram = OpenTelemetryHistogram(otel_histogram)
        self._histograms[name] = histogram
        return histogram

    @override
    def create_duration_histogram(
        self,
        name: str,
        description: str,
    ) -> OpenTelemetryHistogram:
        histogram = self.create_custom_histogram(name, description, "ms")
        self._histograms[name] = histogram

        return histogram

    @override
    def get_or_create_duration_histogram(
        self,
        name: str,
    ) -> OpenTelemetryHistogram:
        if name in self._histograms:
            return self._histograms[name]

        return self.create_duration_histogram(
            name=name,
            description=f"Duration histogram for {name}",
        )
