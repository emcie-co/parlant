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

from dataclasses import replace
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional
import asyncio

import pytest

from parlant.core.agents import AgentId
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableStore,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.engines.compass.variable_loader import (
    VariableLoader,
    load_fresh_context_variable_value,
)
from parlant.core.loggers import Logger
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.sessions import EventSource
from parlant.core.groups import GroupIds, GroupId
from parlant.core.tools import ToolContext, ToolId, ToolResult
from parlant.core.loggers import StdoutLogger
from parlant.core.tracer import AttributeValue, LocalTracer

from tests.core.stable.engines.compass.matching.utils import create_engine_context


def _variable(
    name: str = "profile",
    *,
    tool_id: Optional[ToolId] = None,
    freshness_rules: Optional[str] = None,
) -> ContextVariable:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ContextVariable(
        id=ContextVariableId(f"var-{name}"),
        name=name,
        description=None,
        creation_utc=now,
        modified_utc=now,
        tool_id=tool_id,
        freshness_rules=freshness_rules,
        groups=[],
    )


def _value(data: JSONSerializable, *, modified: datetime | None = None) -> ContextVariableValue:
    return ContextVariableValue(
        id=ContextVariableValueId(f"value-{data}"),
        modified_utc=modified or datetime(2026, 1, 1, tzinfo=timezone.utc),
        data=data,
    )


class _FakeToolService:
    def __init__(self, data: JSONSerializable) -> None:
        self.data = data
        self.calls: list[tuple[str, ToolContext, dict[str, Any]]] = []

    async def call_tool(
        self,
        name: str,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append((name, context, arguments))
        return ToolResult(data=self.data)


class _FakeEntityQueries:
    def __init__(
        self,
        variables: list[ContextVariable],
        values: dict[tuple[ContextVariableId, str], ContextVariableValue],
        services: dict[str, _FakeToolService] | None = None,
        delay_reads: bool = False,
    ) -> None:
        self.variables = variables
        self.values = values
        self.services = services or {}
        self.delay_reads = delay_reads
        self.value_reads: list[tuple[ContextVariableId, str]] = []
        self.active_value_reads = 0
        self.max_active_value_reads = 0

    async def find_context_variables_for_context(
        self,
        agent_id: AgentId,
    ) -> list[ContextVariable]:
        return self.variables

    async def read_context_variable_value(
        self,
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValue]:
        if self.delay_reads:
            self.active_value_reads += 1
            self.max_active_value_reads = max(
                self.max_active_value_reads,
                self.active_value_reads,
            )
            await asyncio.sleep(0.01)
            self.active_value_reads -= 1

        self.value_reads.append((variable_id, key))
        return self.values.get((variable_id, key))

    async def read_tool_service(self, service_name: str) -> _FakeToolService:
        return self.services[service_name]


class _FakeEntityCommands:
    def __init__(self) -> None:
        self.updates: list[tuple[ContextVariableId, str, JSONSerializable]] = []

    async def update_context_variable_value(
        self,
        variable_id: ContextVariableId,
        key: str,
        data: JSONSerializable,
    ) -> ContextVariableValue:
        self.updates.append((variable_id, key, data))
        return _value(data)


class _RecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.started_spans: list[str] = []
        self.events: list[tuple[str, Mapping[str, AttributeValue]]] = []

    @contextmanager
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        self.started_spans.append(span_id)
        with super().span(span_id, attributes):
            yield

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        self.events.append((name, dict(attributes)))


def _loader(
    queries: _FakeEntityQueries,
    commands: _FakeEntityCommands,
    logger: Logger | None = None,
    tokenizer: EstimatingTokenizer | None = None,
) -> VariableLoader:
    return VariableLoader(
        logger=logger or StdoutLogger(LocalTracer()),
        entity_queries=queries,  # type: ignore[arg-type]
        entity_commands=commands,  # type: ignore[arg-type]
        estimating_tokenizer=tokenizer or ZeroEstimatingTokenizer(),
    )


@pytest.mark.asyncio
async def test_that_loader_prefers_customer_specific_value_over_other_keys() -> None:
    variable = _variable()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.customer = replace(context.customer, groups=[GroupId("vip")])
    context.state = ResponseState(agent_effort=context.agent.effort)

    queries = _FakeEntityQueries(
        variables=[variable],
        values={
            (variable.id, context.customer.id): _value("customer"),
            (variable.id, "group:vip"): _value("group"),
            (variable.id, GroupIds.for_agent_id(context.agent.id)): _value("agent"),
            (variable.id, ContextVariableStore.GLOBAL_KEY): _value("global"),
        },
    )
    commands = _FakeEntityCommands()

    loaded = await _loader(queries, commands).load(context)

    assert loaded == [(variable, _value("customer"))]
    assert queries.value_reads == [(variable.id, context.customer.id)]


@pytest.mark.asyncio
async def test_that_loader_falls_back_from_customer_to_tag_agent_and_global_keys() -> None:
    variable = _variable()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.customer = replace(context.customer, groups=[GroupId("vip")])
    context.state = ResponseState(agent_effort=context.agent.effort)

    queries = _FakeEntityQueries(
        variables=[variable],
        values={(variable.id, GroupIds.for_agent_id(context.agent.id)): _value("agent")},
    )
    commands = _FakeEntityCommands()

    loaded = await _loader(queries, commands).load(context)

    assert loaded == [(variable, _value("agent"))]
    assert queries.value_reads == [
        (variable.id, context.customer.id),
        (variable.id, "group:vip"),
        (variable.id, GroupIds.for_agent_id(context.agent.id)),
    ]


@pytest.mark.asyncio
async def test_that_loader_skips_variable_when_no_key_has_a_value() -> None:
    variable = _variable()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(agent_effort=context.agent.effort)

    queries = _FakeEntityQueries(variables=[variable], values={})
    commands = _FakeEntityCommands()

    loaded = await _loader(queries, commands).load(context)

    assert loaded == []
    assert queries.value_reads == [
        (variable.id, context.customer.id),
        (variable.id, GroupIds.for_agent_id(context.agent.id)),
        (variable.id, ContextVariableStore.GLOBAL_KEY),
    ]


@pytest.mark.asyncio
async def test_that_loader_loads_multiple_variables_concurrently() -> None:
    first_variable = _variable("first")
    second_variable = _variable("second")
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(agent_effort=context.agent.effort)

    queries = _FakeEntityQueries(
        variables=[first_variable, second_variable],
        values={
            (first_variable.id, context.customer.id): _value("first-value"),
            (second_variable.id, context.customer.id): _value("second-value"),
        },
        delay_reads=True,
    )
    commands = _FakeEntityCommands()

    loaded = await _loader(queries, commands).load(context)

    assert loaded == [
        (first_variable, _value("first-value")),
        (second_variable, _value("second-value")),
    ]
    assert queries.max_active_value_reads == 2


@pytest.mark.asyncio
async def test_that_loader_emits_variable_trace_event_with_value_payload() -> None:
    variable = _variable("profile")
    value = _value({"tier": "gold", "region": "emea"})
    tracer = _RecordingTracer()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.tracer = tracer
    context.state = ResponseState(agent_effort=context.agent.effort)

    queries = _FakeEntityQueries(
        variables=[variable],
        values={(variable.id, context.customer.id): value},
    )
    commands = _FakeEntityCommands()

    loaded = await _loader(queries, commands).load(context)

    assert loaded == [(variable, value)]
    assert "load.variables" in tracer.started_spans
    assert tracer.events == [
        (
            "loaded.variable",
            {
                "variable_id": variable.id,
                "name": "profile",
                "value_type": "dict",
                "value_size_chars": 34,
                "value": '{"tier": "gold", "region": "emea"}',
                "modified_utc": "2026-01-01T00:00:00+00:00",
            },
        )
    ]


@pytest.mark.asyncio
async def test_that_freshness_loader_returns_stored_value_for_non_tool_variable() -> None:
    variable = _variable()
    stored_value = _value("stored")
    queries = _FakeEntityQueries(variables=[variable], values={(variable.id, "key"): stored_value})
    commands = _FakeEntityCommands()
    session = create_engine_context(conversation=[]).session

    loaded = await load_fresh_context_variable_value(
        entity_queries=queries,  # type: ignore[arg-type]
        entity_commands=commands,  # type: ignore[arg-type]
        agent_id=AgentId("agent"),
        session=session,
        variable=variable,
        key="key",
    )

    assert loaded == stored_value
    assert commands.updates == []


@pytest.mark.asyncio
async def test_that_fresh_tool_backed_value_is_reused_without_calling_tool() -> None:
    tool_id = ToolId(service_name="svc", tool_name="refresh")
    variable = _variable(tool_id=tool_id, freshness_rules="* * * * *")
    current_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    stored_value = _value("fresh", modified=current_time)
    service = _FakeToolService(data="updated")
    queries = _FakeEntityQueries(
        variables=[variable],
        values={(variable.id, "key"): stored_value},
        services={"svc": service},
    )
    commands = _FakeEntityCommands()
    session = create_engine_context(conversation=[]).session

    loaded = await load_fresh_context_variable_value(
        entity_queries=queries,  # type: ignore[arg-type]
        entity_commands=commands,  # type: ignore[arg-type]
        agent_id=AgentId("agent"),
        session=session,
        variable=variable,
        key="key",
        current_time=current_time,
    )

    assert loaded == stored_value
    assert service.calls == []
    assert commands.updates == []


@pytest.mark.asyncio
async def test_that_stale_tool_backed_value_is_refreshed_and_persisted() -> None:
    tool_id = ToolId(service_name="svc", tool_name="refresh")
    variable = _variable(tool_id=tool_id, freshness_rules="* * * * *")
    current_time = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)
    stale_value = _value("stale", modified=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    service = _FakeToolService(data={"profile": "updated"})
    queries = _FakeEntityQueries(
        variables=[variable],
        values={(variable.id, "key"): stale_value},
        services={"svc": service},
    )
    commands = _FakeEntityCommands()
    session = create_engine_context(conversation=[]).session

    loaded = await load_fresh_context_variable_value(
        entity_queries=queries,  # type: ignore[arg-type]
        entity_commands=commands,  # type: ignore[arg-type]
        agent_id=AgentId("agent"),
        session=session,
        variable=variable,
        key="key",
        current_time=current_time,
    )

    assert loaded == _value({"profile": "updated"})
    assert [(name, args) for name, _, args in service.calls] == [("refresh", {})]
    assert commands.updates == [(variable.id, "key", {"profile": "updated"})]
