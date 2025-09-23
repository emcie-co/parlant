import asyncio
import contextvars
import os
from types import TracebackType
from typing import Any, AsyncGenerator, Mapping
from typing_extensions import override, Self
from contextlib import asynccontextmanager

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from parlant.core.common import AttributeValue
from parlant.core.meter import Meter


class OtelMeter(Meter):
    @staticmethod
    def is_environment_set() -> bool:
        if not os.environ.get("OTEL_METRICS_EXPORTER") and not os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        ):
            return False

        return True

    def __init__(self) -> None:
        self._service_name = os.getenv("OTEL_SERVICE_NAME", "parlant")

        self._meter: metrics.Meter
        self._metrics_exporter = os.environ["OTEL_METRICS_EXPORTER"]
        self._meter_provider: MeterProvider

        self._insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "false").lower() == "true"

        self._scopes: contextvars.ContextVar[str] = contextvars.ContextVar(
            f"otel_meter_scopes_{id(self)}",
            default="",
        )

        # Instrument caches (name -> instrument)
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

    async def __aenter__(self) -> Self:
        resource = Resource.create({"service.name": self._service_name})

        metric_exporter = OTLPMetricExporter(insecure=self._insecure)
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=5000,
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
    async def increment(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(name)

        attrs = attributes or {}
        self._counters[name].add(value, attrs)

    @override
    async def record_histogram(
        self,
        name: str,
        value: float,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        if name not in self._histograms:
            self._histograms[name] = self._meter.create_histogram(name)

        attrs = attributes or {}
        self._histograms[name].record(value, attrs)

    @override
    @asynccontextmanager
    async def measure(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> AsyncGenerator[None, None]:
        """
        Measure the duration of a block of code.
        Usage:
            async with meter.measure("my_duration"):
                # Code to measure
        """
        token = self._push_scope(name)
        start_time = asyncio.get_running_loop().time()
        try:
            yield
        finally:
            duration = asyncio.get_running_loop().time() - start_time
            await self.record_histogram(token.var.get(), duration, attributes)
            self._pop_scope(token)

    def _push_scope(self, segment: str) -> contextvars.Token[str]:
        current = self._scopes.get()
        new_scope = f"{current}.{segment}" if current else segment
        return self._scopes.set(new_scope)

    def _pop_scope(self, token: contextvars.Token[str]) -> None:
        self._scopes.reset(token)
