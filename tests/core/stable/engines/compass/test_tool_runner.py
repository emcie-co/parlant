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
import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable, Mapping, cast

import pytest

from parlant.core.common import JSONSerializable
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.compass.tool_runner import (
    DEFAULT_TOOL_TIMEOUT,
    ToolRunner,
    stringify_tool_result,
)
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core import metadata
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.nlp.tokenization import ZeroEstimatingTokenizer
from parlant.core.tools import ToolContext, ToolId, ToolResult
from parlant.core.tracer import LocalTracer
from parlant.core.pricing import PricingRegistry
from parlant.core.usage_reporter import UsageReporter


class _FakeService:
    def __init__(self, behavior: Callable[[str, Mapping[str, Any]], ToolResult]) -> None:
        self._behavior = behavior

    async def resolve_tool(self, name: str, context: ToolContext) -> Any:
        return SimpleNamespace(narration=None)

    async def call_tool(
        self, name: str, context: ToolContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        return self._behavior(name, arguments)


class _FakeEntityQueries:
    def __init__(self, service: _FakeService) -> None:
        self._service = service

    async def read_tool_service(self, service_name: str) -> _FakeService:
        return self._service


def _engine_context() -> EngineContext:
    return cast(
        EngineContext,
        SimpleNamespace(
            agent=SimpleNamespace(id="agent-1"),
            session=SimpleNamespace(id="session-1"),
            customer=SimpleNamespace(id="customer-1"),
            state=SimpleNamespace(model_route_qualifiers={}),
        ),
    )


class _FakeModelRouter:
    async def get_estimating_tokenizer(self, **kwargs: Any) -> ZeroEstimatingTokenizer:
        return ZeroEstimatingTokenizer()


def _runner(
    service: _FakeService,
    logger: Logger | None = None,
    tracer: LocalTracer | None = None,
) -> ToolRunner:
    tracer = tracer or LocalTracer()
    return ToolRunner(
        logger or StdoutLogger(tracer),
        tracer,
        cast(EntityQueries, _FakeEntityQueries(service)),
        cast(ModelRouter, _FakeModelRouter()),
        PricingRegistry(UsageReporter(tracer)),
    )


def test_that_tool_results_preserve_unicode_for_prompts() -> None:
    assert stringify_tool_result({"message": "مرحبا"}) == '{"message": "مرحبا"}'


async def test_that_run_tool_executes_against_the_service_and_returns_its_result() -> None:
    runner = _runner(
        _FakeService(lambda name, args: ToolResult(data={"called": name, "args": args}))
    )

    result = await runner.run_tool(
        _engine_context(), ToolId("svc", "echo"), cast(Mapping[str, JSONSerializable], {"x": 1})
    )

    assert result.data == {"called": "echo", "args": {"x": 1}}


async def test_that_run_tool_captures_a_failure_as_an_error_result() -> None:
    def boom(name: str, args: Mapping[str, Any]) -> ToolResult:
        raise RuntimeError("kaboom")

    runner = _runner(_FakeService(boom))

    result = await runner.run_tool(_engine_context(), ToolId("svc", "boom"), {})

    assert "kaboom" in str(metadata.internal_value(result.metadata, ("error_details",)))


class _HangingService:
    async def call_tool(
        self, name: str, context: ToolContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult(data="done")


async def test_that_run_tool_times_out_and_returns_an_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_TOOL_TIMEOUT", "0.05")
    runner = _runner(cast(_FakeService, _HangingService()))

    result = await runner.run_tool(_engine_context(), ToolId("svc", "slow"), {})

    assert "timed out" in str(metadata.internal_value(result.metadata, ("error_details",)))


def test_that_default_tool_timeout_is_180_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARLANT_TOOL_TIMEOUT", raising=False)
    runner = _runner(_FakeService(lambda name, args: ToolResult(data="ok")))

    assert DEFAULT_TOOL_TIMEOUT == 180.0
    assert runner._resolve_timeout() == 180.0


class _EventRecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_events: list[tuple[str, dict[str, Any]]] = []

    def add_event(self, name: str, attributes: Any = {}) -> None:
        self.recorded_events.append((name, dict(attributes)))


async def test_that_a_successful_tool_run_emits_called_and_result_events() -> None:
    tracer = _EventRecordingTracer()
    runner = _runner(
        _FakeService(lambda name, args: ToolResult(data={"price": 3})),
        tracer=tracer,
    )

    compass_tracer = CompassTracer(tracer)
    with compass_tracer.tool_call_scope("call-77"):
        await runner.run_tool(_engine_context(), ToolId("svc", "echo"), {"item": "eggplant"})

    called = [a for n, a in tracer.recorded_events if n == "tool.called"]
    results = [a for n, a in tracer.recorded_events if n == "tool.result"]

    assert len(called) == 1
    assert called[0]["tool_id"] == "svc:echo"
    assert called[0]["tool_call_id"] == "call-77"

    assert len(results) == 1
    assert results[0]["tool_call_id"] == "call-77"
    assert results[0]["is_error"] is False


async def test_that_a_failing_tool_run_emits_a_tool_error_event() -> None:
    def boom(name: str, args: Mapping[str, Any]) -> ToolResult:
        raise RuntimeError("kaboom")

    tracer = _EventRecordingTracer()
    runner = _runner(_FakeService(boom), tracer=tracer)

    await runner.run_tool(_engine_context(), ToolId("svc", "boom"), {})

    errors = [a for n, a in tracer.recorded_events if n == "tool.error"]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["timed_out"] is False
    assert not any(n == "tool.result" for n, _ in tracer.recorded_events)


async def test_that_a_timed_out_tool_run_emits_a_timed_out_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_TOOL_TIMEOUT", "0.05")
    tracer = _EventRecordingTracer()
    runner = _runner(cast(_FakeService, _HangingService()), tracer=tracer)

    await runner.run_tool(_engine_context(), ToolId("svc", "slow"), {})

    errors = [a for n, a in tracer.recorded_events if n == "tool.error"]
    assert len(errors) == 1
    assert errors[0]["timed_out"] is True


async def test_that_normal_tool_execution_records_category_costs_in_the_trace() -> None:
    tracer = _EventRecordingTracer()
    runner = _runner(
        _FakeService(
            lambda name, args: ToolResult(data="Ready", costs={"Search": Decimal("0.02")})
        ),
        tracer=tracer,
    )
    with tracer.span("turn"):
        await runner.run_tool(_engine_context(), ToolId("svc", "lookup"), {})
        costs, error = runner._pricing_registry.snapshot_trace(tracer.trace_id)
    assert error is None and costs is not None
    assert costs.total == Decimal("0.02")
    assert costs.per_tool == {"Search": Decimal("0.02")}
    events = [attrs for name, attrs in tracer.recorded_events if name == "tool.result"]
    assert len(events) == 1
    assert json.loads(events[0]["costs"]) == {"Search": "0.02"}
