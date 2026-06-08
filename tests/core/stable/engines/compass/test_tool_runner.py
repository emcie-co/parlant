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

from types import SimpleNamespace
from typing import Any, Callable, Mapping, cast

from parlant.core.common import JSONSerializable
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.compass.tool_runner import ToolRunner
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import StdoutLogger
from parlant.core.tools import ToolContext, ToolId, ToolResult
from parlant.core.tracer import LocalTracer


class _FakeService:
    def __init__(self, behavior: Callable[[str, Mapping[str, Any]], ToolResult]) -> None:
        self._behavior = behavior

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


def _runner(service: _FakeService) -> ToolRunner:
    return ToolRunner(
        StdoutLogger(LocalTracer()),
        cast(EntityQueries, _FakeEntityQueries(service)),
    )


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

    assert "error_details" in result.metadata
    assert "kaboom" in result.metadata["error_details"]
