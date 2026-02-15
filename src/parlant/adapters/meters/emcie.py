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
from contextlib import asynccontextmanager
import logging
import os
from types import TracebackType
from typing import AsyncGenerator, Mapping
from typing_extensions import Self, override

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.metrics import Counter as OTELCounter
from opentelemetry.metrics import Histogram as OTELHistogram
from opentelemetry.metrics import Meter as OTELMeter

from parlant.core.meter import Meter, Counter, Histogram, DurationHistogram

logger = logging.getLogger(__name__)


class EmcieCounter(Counter):
    """Counter implementation using OpenTelemetry."""

    def __init__(self, otel_counter: OTELCounter) -> None:
        self._otel_counter = otel_counter

    @override
    async def increment(
        self,
        value: int,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        """Increment the counter by the given value."""
        self._otel_counter.add(value, attributes or {})


class EmcieHistogram(Histogram):
    """Histogram implementation using OpenTelemetry."""

    def __init__(self, otel_histogram: OTELHistogram) -> None:
        self._otel_histogram = otel_histogram

    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        """Record a value in the histogram."""
        self._otel_histogram.record(value, attributes or {})


class EmcieDurationHistogram(DurationHistogram):
    """Duration histogram implementation using OpenTelemetry."""

    def __init__(self, otel_histogram: OTELHistogram) -> None:
        self._otel_histogram = otel_histogram

    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        """Record a duration value in the histogram."""
        self._otel_histogram.record(value, attributes or {})

    @override
    @asynccontextmanager
    async def measure(
        self,
        attributes: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[None, None]:
        """Measure the duration of a code block."""
        start_time = asyncio.get_running_loop().time()
        try:
            yield
        finally:
            duration = asyncio.get_running_loop().time() - start_time
            await self.record(duration, attributes)


class EmcieMeter(Meter):
    """A meter that sends metrics to Emcie backend using OpenTelemetry OTLP."""

    def __init__(self) -> None:
        self._endpoint = f"{os.getenv('EMCIE_BASE_URL', 'https://api.emcie.co')}/v1/metrics"
        self._api_key = os.getenv("EMCIE_API_KEY", "")

        self._meter_provider: MeterProvider | None = None
        self._metric_exporter: OTLPMetricExporter | None = None
        self._metric_reader: PeriodicExportingMetricReader | None = None
        self._otel_meter: OTELMeter | None = None

    async def __aenter__(self) -> Self:
        """Initialize the OpenTelemetry metrics infrastructure."""
        resource = Resource.create({"service.name": "parlant-emcie-meter"})

        # Setup headers for API authentication
        headers = {}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
            logger.debug("API key configured for OTLP metric export")
        else:
            logger.warning("No API key provided for OTLP metric export")

        # Create OTLP metric exporter
        self._metric_exporter = OTLPMetricExporter(
            endpoint=self._endpoint,
            headers=headers,
        )

        # Create metric reader with periodic export
        self._metric_reader = PeriodicExportingMetricReader(
            exporter=self._metric_exporter,
            export_interval_millis=1000,  # Export every 1 second
        )

        # Create meter provider
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[self._metric_reader],
        )

        # Get the meter instance
        self._otel_meter = self._meter_provider.get_meter(__name__)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Shutdown the meter and flush pending metrics."""
        if self._metric_reader:
            try:
                self._metric_reader.force_flush()
                self._metric_reader.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Error during EmcieMeter shutdown: {e}")

        return False

    @override
    def create_counter(
        self,
        name: str,
        description: str,
    ) -> Counter:
        """Create a counter metric."""
        if not self._otel_meter:
            raise RuntimeError("EmcieMeter must be used as an async context manager")

        otel_counter = self._otel_meter.create_counter(
            name=name,
            description=description,
        )
        return EmcieCounter(otel_counter)

    @override
    def create_custom_histogram(
        self,
        name: str,
        description: str,
        unit: str,
    ) -> Histogram:
        """Create a custom histogram metric."""
        if not self._otel_meter:
            raise RuntimeError("EmcieMeter must be used as an async context manager")

        otel_histogram = self._otel_meter.create_histogram(
            name=name,
            description=description,
            unit=unit,
        )
        return EmcieHistogram(otel_histogram)

    @override
    def create_duration_histogram(
        self,
        name: str,
        description: str,
    ) -> DurationHistogram:
        """Create a duration histogram metric."""
        if not self._otel_meter:
            raise RuntimeError("EmcieMeter must be used as an async context manager")

        otel_histogram = self._otel_meter.create_histogram(
            name=name,
            description=description,
            unit="s",  # Duration in seconds
        )
        return EmcieDurationHistogram(otel_histogram)
