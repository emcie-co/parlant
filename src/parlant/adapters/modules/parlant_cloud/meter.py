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

"""ParlantCloudMeter — OTLP-backed metrics for Parlant Cloud."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from types import TracebackType
from typing import AsyncGenerator, Mapping

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Counter as OTELCounter
from opentelemetry.metrics import Histogram as OTELHistogram
from opentelemetry.metrics import Meter as OTELMeter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from typing_extensions import Self, override

from parlant.core.meter import Counter, DurationHistogram, Histogram, Meter

from .config import _get_cloud_base_url

_logger = logging.getLogger(__name__)


class ParlantCloudCounter(Counter):
    def __init__(self, otel_counter: OTELCounter) -> None:
        self._otel_counter = otel_counter

    @override
    async def increment(
        self,
        value: int,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self._otel_counter.add(value, attributes or {})


class ParlantCloudHistogram(Histogram):
    def __init__(self, otel_histogram: OTELHistogram) -> None:
        self._otel_histogram = otel_histogram

    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self._otel_histogram.record(value, attributes or {})


class ParlantCloudDurationHistogram(DurationHistogram):
    def __init__(self, otel_histogram: OTELHistogram) -> None:
        self._otel_histogram = otel_histogram

    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self._otel_histogram.record(value, attributes or {})

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
            duration = asyncio.get_running_loop().time() - start_time
            await self.record(duration, attributes)


class ParlantCloudMeter(Meter):
    def __init__(self, project_id: str = "") -> None:
        self._project_id = project_id
        self._endpoint = f"{_get_cloud_base_url()}/v1/metrics"
        self._project_token = os.getenv("PARLANT_CLOUD_PROJECT_TOKEN", "")

        self._meter_provider: MeterProvider | None = None
        self._metric_exporter: OTLPMetricExporter | None = None
        self._metric_reader: PeriodicExportingMetricReader | None = None
        self._otel_meter: OTELMeter | None = None

    async def __aenter__(self) -> Self:
        resource_attributes: dict[str, str] = {
            "service.name": "parlant-cloud-meter",
        }
        if self._project_id:
            resource_attributes["project_id"] = self._project_id

        resource = Resource.create(resource_attributes)

        headers = {}
        if self._project_token:
            headers["authorization"] = f"Bearer {self._project_token}"

        self._metric_exporter = OTLPMetricExporter(
            endpoint=self._endpoint,
            headers=headers,
        )

        self._metric_reader = PeriodicExportingMetricReader(
            exporter=self._metric_exporter,
            export_interval_millis=1000,
        )

        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[self._metric_reader],
        )

        self._otel_meter = self._meter_provider.get_meter(__name__)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._metric_reader:
            try:
                self._metric_reader.force_flush()
                self._metric_reader.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                _logger.warning(f"Error during ParlantCloudMeter shutdown: {e}")

        return False

    @override
    def create_counter(
        self,
        name: str,
        description: str,
    ) -> Counter:
        if not self._otel_meter:
            raise RuntimeError("ParlantCloudMeter must be used as an async context manager")

        otel_counter = self._otel_meter.create_counter(
            name=name,
            description=description,
        )
        return ParlantCloudCounter(otel_counter)

    @override
    def create_custom_histogram(
        self,
        name: str,
        description: str,
        unit: str,
    ) -> Histogram:
        if not self._otel_meter:
            raise RuntimeError("ParlantCloudMeter must be used as an async context manager")

        otel_histogram = self._otel_meter.create_histogram(
            name=name,
            description=description,
            unit=unit,
        )
        return ParlantCloudHistogram(otel_histogram)

    @override
    def create_duration_histogram(
        self,
        name: str,
        description: str,
    ) -> DurationHistogram:
        if not self._otel_meter:
            raise RuntimeError("ParlantCloudMeter must be used as an async context manager")

        otel_histogram = self._otel_meter.create_histogram(
            name=name,
            description=description,
            unit="s",
        )
        return ParlantCloudDurationHistogram(otel_histogram)
