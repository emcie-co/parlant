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
from types import SimpleNamespace
from typing import Any, Callable, Mapping, cast

import pytest

from parlant.core.common import JSONSerializable
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.compass.tracing import format_json_attr
from parlant.core.engines.compass.tool_runner import ToolRunner
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.nlp.tokenization import ZeroEstimatingTokenizer
from parlant.core.tools import ToolContext, ToolId, ToolResult
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    RecordedEvent,
    RecordedSpan,
    RecordingTracer,
)


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
        ),
    )


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
        ZeroEstimatingTokenizer(),
    )


async def test_that_run_tool_executes_against_the_service_and_returns_its_result() -> None:
    runner = _runner(
        _FakeService(lambda name, args: ToolResult(data={"called": name, "args": args}))
    )

    result = await runner.run_tool(
        _engine_context(), ToolId("svc", "echo"), cast(Mapping[str, JSONSerializable], {"x": 1})
    )

    assert result.data == {"called": "echo", "args": {"x": 1}}


async def test_that_run_tool_records_call_and_result_events() -> None:
    tracer = RecordingTracer()
    runner = _runner(
        _FakeService(lambda name, args: ToolResult(data={"called": name, "args": args})),
        tracer=tracer,
    )
    tool_id = ToolId("svc", "echo")

    await runner.run_tool(
        _engine_context(),
        tool_id,
        cast(Mapping[str, JSONSerializable], {"x": 1}),
    )

    assert tracer.started_spans == [
        RecordedSpan(
            name="tools.call",
            attributes={
                "tool_id": "svc:echo",
                "tool_name": "echo",
                "service_name": "svc",
            },
        )
    ]
    assert tracer.events == [
        RecordedEvent(
            name="tool.called",
            attributes={
                "tool_id": "svc:echo",
                "tool_name": "echo",
                "service_name": "svc",
                "arguments": format_json_attr({"x": 1}),
            },
            span_id="tools.call",
        ),
        RecordedEvent(
            name="tool.result",
            attributes={
                "tool_id": "svc:echo",
                "tool_name": "echo",
                "service_name": "svc",
                "is_error": False,
                "result": format_json_attr({"called": "echo", "args": {"x": 1}}),
                "metadata": format_json_attr({}),
                "control": format_json_attr({}),
            },
            span_id="tools.call",
        ),
    ]


async def test_that_run_tool_captures_a_failure_as_an_error_result() -> None:
    def boom(name: str, args: Mapping[str, Any]) -> ToolResult:
        raise RuntimeError("kaboom")

    runner = _runner(_FakeService(boom))

    result = await runner.run_tool(_engine_context(), ToolId("svc", "boom"), {})

    assert "error_details" in result.metadata
    assert "kaboom" in result.metadata["error_details"]


async def test_that_run_tool_records_error_event() -> None:
    def boom(name: str, args: Mapping[str, Any]) -> ToolResult:
        raise RuntimeError("kaboom")

    tracer = RecordingTracer()
    runner = _runner(_FakeService(boom), tracer=tracer)

    await runner.run_tool(_engine_context(), ToolId("svc", "boom"), {})

    assert [event.name for event in tracer.events] == [
        "tool.called",
        "tool.result",
        "tool.error",
    ]
    assert tracer.events[-1] == RecordedEvent(
        name="tool.error",
        attributes={
            "tool_id": "svc:boom",
            "tool_name": "boom",
            "service_name": "svc",
            "error_details": "kaboom",
        },
        span_id="tools.call",
    )


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

    assert "error_details" in result.metadata
    assert "timed out" in result.metadata["error_details"]
