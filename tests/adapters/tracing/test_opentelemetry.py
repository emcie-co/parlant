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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pytest import raises
from pytest_asyncio import fixture

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics.export import MetricExporter, MetricsData, MetricExportResult

from parlant.adapters.tracing.opentelemetry import OpenTelemetryTracer, OpenTelemetrySpan


class CollectingMetricExporter(MetricExporter):
    def __init__(self) -> None:
        super().__init__()
        self.exported: list[MetricsData] = []

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 60,
        **kwargs: Any,
    ) -> MetricExportResult:
        self.exported.append(metrics_data)
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 60, **kwargs: Any) -> bool:
        return True

    def shutdown(self, timeout_millis: float = 60, **kwargs: Any) -> None:
        return


@dataclass
class TracerContext:
    tracer: OpenTelemetryTracer
    span_exporter: InMemorySpanExporter
    metric_exporter: CollectingMetricExporter


@fixture
async def tracer_context() -> AsyncIterator[TracerContext]:
    span_exporter = InMemorySpanExporter()
    metric_exporter = CollectingMetricExporter()

    tracer = OpenTelemetryTracer(exporter="local")
    tracer._span_exporter = span_exporter
    tracer._metric_exporter = metric_exporter

    async with tracer:
        original_end_span = tracer.end_span

        async def end_span_and_flush(error: Exception | None = None) -> None:
            await original_end_span(error)
            if tracer._tracer_provider is not None:
                tracer._tracer_provider.force_flush()
            if tracer._meter_provider is not None:
                tracer._meter_provider.force_flush()

        tracer.end_span = end_span_and_flush  # type: ignore[assignment]

        yield TracerContext(tracer, span_exporter, metric_exporter)

        if tracer._tracer_provider is not None:
            tracer._tracer_provider.force_flush()
        if tracer._meter_provider is not None:
            tracer._meter_provider.force_flush()


def collected_metrics(metric_exp: CollectingMetricExporter) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}
    for md in metric_exp.exported:
        for rm in md.resource_metrics:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    values: list[float] = []
                    # Sum/Counter and Histogram both surface via .data.data_points
                    if hasattr(m.data, "data_points"):
                        for dp in m.data.data_points:
                            # SumPointData has .value; Histogram has .sum/.count
                            if hasattr(dp, "value"):
                                values.append(float(dp.value))
                            elif hasattr(dp, "sum"):
                                values.append(float(dp.sum))
                            else:
                                raise TypeError(f"Unsupported data point type: {type(dp)}")
                    results.setdefault(m.name, []).extend(values)
    return results


def _exported_span_names(span_exp: InMemorySpanExporter) -> list[str]:
    return [s.name for s in span_exp.get_finished_spans()]


async def test_that_metrics_recording_works_via_span(tracer_context: TracerContext) -> None:
    async with tracer_context.tracer.span("metrics") as s:
        await s.record_counter("requests_total", 1, {"endpoint": "/api/test"})
        await s.record_histogram("response_time_ms", 150.5, {"method": "GET"})
        await s.record_counter("errors_total", 2)
        await s.record_histogram("processing_duration", 0.8)

    metrics = collected_metrics(tracer_context.metric_exporter)

    assert set(metrics.keys()) == {
        "requests_total",
        "response_time_ms",
        "errors_total",
        "processing_duration",
    }
    assert metrics["requests_total"] == [1]
    assert metrics["errors_total"] == [2]
    assert metrics["response_time_ms"] == [150.5]
    assert metrics["processing_duration"] == [0.8]


async def test_that_context_manager_yields_a_span_instance(tracer_context: TracerContext) -> None:
    async with tracer_context.tracer.span("op", {"user_id": "123"}) as s:
        assert isinstance(s, OpenTelemetrySpan)

    finished = tracer_context.span_exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "op"
    assert finished[0].attributes and finished[0].attributes.get("user_id") == "123"


async def test_that_span_lifecycle_pushes_and_pops_stack_correctly(
    tracer_context: TracerContext,
) -> None:
    assert tracer_context.tracer._span_stack.get() == ()

    async with tracer_context.tracer.span("op1") as span1:
        assert isinstance(span1, OpenTelemetrySpan)
        assert len(tracer_context.tracer._span_stack.get()) == 1

        async with tracer_context.tracer.span("op2") as span2:
            assert isinstance(span2, OpenTelemetrySpan)
            assert len(tracer_context.tracer._span_stack.get()) == 2
            assert tracer_context.tracer._span_stack.get()[-1] is span2

        assert len(tracer_context.tracer._span_stack.get()) == 1
        assert tracer_context.tracer._span_stack.get()[-1] is span1

    assert tracer_context.tracer._span_stack.get() == ()

    names = _exported_span_names(tracer_context.span_exporter)
    assert set(names) == {"op1", "op2"}
    # LIFO close order: outer "op1" finishes last
    assert names[-1] == "op1"


async def test_that_span_lifecycle_records_attributes_and_events(
    tracer_context: TracerContext,
) -> None:
    async with tracer_context.tracer.span("op") as s:
        await s.set_attribute("status", "processing")
        await s.add_event("checkpoint", {"step": 1})

    finished = tracer_context.span_exporter.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]

    assert span.attributes and span.attributes.get("status") == "processing"

    event_names = [e.name for e in span.events]
    assert "checkpoint" in event_names
    checkpoint = next(e for e in span.events if e.name == "checkpoint")
    assert checkpoint.attributes and checkpoint.attributes.get("step") == 1


async def test_that_context_manager_sets_error_status_when_exception_is_raised(
    tracer_context: TracerContext,
) -> None:
    with raises(RuntimeError, match="boom"):
        async with tracer_context.tracer.span("error-op"):
            raise RuntimeError("boom")

    assert tracer_context.tracer._span_stack.get() == ()

    finished = tracer_context.span_exporter.get_finished_spans()
    assert len(finished) == 1

    span = finished[0]
    assert span.name == "error-op"
    assert any(e.name == "exception" for e in span.events)


async def test_that_concurrent_requests_isolate_span_stacks(tracer_context: TracerContext) -> None:
    async def worker(delay: float) -> None:
        async with tracer_context.tracer.span("process"):
            await asyncio.sleep(delay)

    t1 = asyncio.create_task(worker(0.10), name="ReqA")
    t2 = asyncio.create_task(worker(0.01), name="ReqB")
    await asyncio.gather(t1, t2)

    assert tracer_context.tracer._span_stack.get() == ()

    finished = tracer_context.span_exporter.get_finished_spans()
    assert len(finished) == 2
    name_counts = Counter(s.name for s in finished)
    assert name_counts == Counter({"process": 2})
