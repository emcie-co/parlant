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
from typing import Mapping, Sequence

import pytest

from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.compass.matching.tool_recaller import ToolRecaller
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.rule_tool_associations import (
    RuleToolAssociation,
    RuleToolAssociationId,
)
from parlant.core.rules import Rule
from parlant.core.tools import Tool, ToolContext, ToolId, ToolOverlap, ToolResult, ToolService

from tests.core.stable.engines.compass.matching.utils import (
    RecordedSpan,
    RecordingTracer,
    create_engine_context,
    create_rule,
)


def _tool(name: str, description: str | None = None) -> Tool:
    return Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description=description or f"{name} tool",
        metadata={},
        parameters={},
        required=[],
        consequential=False,
        overlap=ToolOverlap.NONE,
    )


def _association(rule: Rule, tool_id: ToolId) -> RuleToolAssociation:
    return RuleToolAssociation(
        id=RuleToolAssociationId(f"{rule.id}:{tool_id.to_string()}"),
        creation_utc=datetime.now(timezone.utc),
        rule_id=rule.id,
        tool_id=tool_id,
    )


class _FakeToolService(ToolService):
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    async def list_tools(self) -> Sequence[Tool]:
        return list(self._tools.values())

    async def read_tool(self, name: str) -> Tool:
        return self._tools[name]

    async def resolve_tool(
        self,
        name: str,
        context: ToolContext,
    ) -> Tool:
        return self._tools[name]

    async def call_tool(
        self,
        name: str,
        context: ToolContext,
        arguments: Mapping[str, object],
    ) -> ToolResult:
        return ToolResult(data={})


class _FakeEntityQueries:
    def __init__(
        self,
        associations: Sequence[RuleToolAssociation],
        tools: Mapping[ToolId, Tool],
    ) -> None:
        self._associations = list(associations)
        self._services = {
            service_name: _FakeToolService(
                [tool for tool_id, tool in tools.items() if tool_id.service_name == service_name]
            )
            for service_name in {tool_id.service_name for tool_id in tools}
        }

    async def find_rule_tool_associations(self) -> Sequence[RuleToolAssociation]:
        return self._associations

    async def read_tool_service(self, service_name: str) -> ToolService:
        return self._services[service_name]


class _ScoredToolRecaller(ToolRecaller):
    def __init__(
        self,
        associations: Sequence[RuleToolAssociation],
        tools: Mapping[ToolId, Tool],
        scores: Mapping[ToolId, float],
        *,
        max_tools: int,
    ) -> None:
        super().__init__(
            entity_queries=_FakeEntityQueries(associations, tools),
            nlp_service=None,
            embedding_cache=None,
            max_tools=max_tools,
        )
        self._scores = dict(scores)

    async def _score_tools(
        self,
        context: EngineContext,
        tools_by_id: Mapping[ToolId, Tool],
    ) -> Mapping[ToolId, float]:
        return {tool_id: self._scores.get(tool_id, 0.0) for tool_id in tools_by_id}


def _context_with_rules(
    *rules: Rule,
    session_rules: set[Rule] | None = None,
) -> EngineContext:
    context = create_engine_context(conversation=[])
    context.state = ResponseState(
        usable_rules=list(rules),
        session_rules=session_rules or set(),
    )
    return context


async def _prepare_and_select(
    recaller: ToolRecaller,
    context: EngineContext,
) -> list[str]:
    await recaller.prepare(context)
    await recaller.select(context)
    return [tool.name for tool in context.state.available_tools]


@pytest.mark.asyncio
async def test_that_prepare_records_tool_recall_span_with_counts() -> None:
    rule = create_rule("customer needs account lookup")
    tool_id = ToolId("local", "lookup_account")
    tracer = RecordingTracer()
    recaller = ToolRecaller(
        entity_queries=_FakeEntityQueries(
            [_association(rule, tool_id)],
            {tool_id: _tool("lookup_account")},
        ),
        nlp_service=None,
        embedding_cache=None,
    )
    context = replace(_context_with_rules(rule), tracer=tracer)

    await recaller.prepare(context)

    assert tracer.started_spans == [RecordedSpan(name="match.tool.recall", attributes={})]
    assert tracer.get_attribute("candidate_count") == 1
    assert tracer.get_attribute("scored_count") == 1


@pytest.mark.asyncio
async def test_that_turn_matched_tools_take_priority_over_session_and_remaining_tools() -> None:
    turn_rule = create_rule("customer needs account lookup")
    session_rule = create_rule("customer is managing billing")
    remaining_rule = create_rule("customer may need support")
    turn_tool = ToolId("local", "turn_lookup")
    session_tool = ToolId("local", "session_billing")
    remaining_tool = ToolId("local", "remaining_support")
    tools = {
        turn_tool: _tool("turn_lookup"),
        session_tool: _tool("session_billing"),
        remaining_tool: _tool("remaining_support"),
    }
    associations = [
        _association(turn_rule, turn_tool),
        _association(session_rule, session_tool),
        _association(remaining_rule, remaining_tool),
    ]
    recaller = _ScoredToolRecaller(
        associations,
        tools,
        scores={turn_tool: 0.1, session_tool: 0.9, remaining_tool: 1.0},
        max_tools=1,
    )
    context = _context_with_rules(
        turn_rule,
        session_rule,
        remaining_rule,
        session_rules={session_rule},
    )
    context.state.tool_enabled_rule_matches = {
        RuleMatch(rule=turn_rule, rationale="Relevant."): [turn_tool]
    }

    selected = await _prepare_and_select(recaller, context)

    assert selected == ["turn_lookup"]


@pytest.mark.asyncio
async def test_that_session_tools_take_priority_over_higher_scored_remaining_tools() -> None:
    session_rule = create_rule("customer is managing billing")
    remaining_rule = create_rule("customer may need support")
    session_tool = ToolId("local", "session_billing")
    remaining_tool = ToolId("local", "remaining_support")
    tools = {
        session_tool: _tool("session_billing"),
        remaining_tool: _tool("remaining_support"),
    }
    associations = [
        _association(session_rule, session_tool),
        _association(remaining_rule, remaining_tool),
    ]
    recaller = _ScoredToolRecaller(
        associations,
        tools,
        scores={session_tool: 0.1, remaining_tool: 1.0},
        max_tools=1,
    )
    context = _context_with_rules(
        session_rule,
        remaining_rule,
        session_rules={session_rule},
    )

    selected = await _prepare_and_select(recaller, context)

    assert selected == ["session_billing"]


@pytest.mark.asyncio
async def test_that_over_cap_session_tools_are_trimmed_by_individual_tool_score() -> None:
    rule = create_rule("customer is managing a booking")
    low_tool = ToolId("local", "low_session")
    high_tool = ToolId("local", "high_session")
    middle_tool = ToolId("local", "middle_session")
    tools = {
        low_tool: _tool("low_session"),
        high_tool: _tool("high_session"),
        middle_tool: _tool("middle_session"),
    }
    associations = [
        _association(rule, low_tool),
        _association(rule, high_tool),
        _association(rule, middle_tool),
    ]
    recaller = _ScoredToolRecaller(
        associations,
        tools,
        scores={low_tool: 0.1, high_tool: 0.9, middle_tool: 0.5},
        max_tools=2,
    )
    context = _context_with_rules(rule, session_rules={rule})

    selected = await _prepare_and_select(recaller, context)

    assert selected == ["high_session", "middle_session"]


@pytest.mark.asyncio
async def test_that_remaining_tools_fill_the_cap_after_turn_and_session_tools() -> None:
    turn_rule = create_rule("customer needs account lookup")
    session_rule = create_rule("customer is managing billing")
    remaining_rule = create_rule("customer may need support")
    turn_tool = ToolId("local", "turn_lookup")
    session_tool = ToolId("local", "session_billing")
    best_remaining_tool = ToolId("local", "best_remaining")
    other_remaining_tool = ToolId("local", "other_remaining")
    tools = {
        turn_tool: _tool("turn_lookup"),
        session_tool: _tool("session_billing"),
        best_remaining_tool: _tool("best_remaining"),
        other_remaining_tool: _tool("other_remaining"),
    }
    associations = [
        _association(turn_rule, turn_tool),
        _association(session_rule, session_tool),
        _association(remaining_rule, best_remaining_tool),
        _association(remaining_rule, other_remaining_tool),
    ]
    recaller = _ScoredToolRecaller(
        associations,
        tools,
        scores={best_remaining_tool: 0.9, other_remaining_tool: 0.1},
        max_tools=3,
    )
    context = _context_with_rules(
        turn_rule,
        session_rule,
        remaining_rule,
        session_rules={session_rule},
    )
    context.state.tool_enabled_rule_matches = {
        RuleMatch(rule=turn_rule, rationale="Relevant."): [turn_tool]
    }

    selected = await _prepare_and_select(recaller, context)

    assert selected == ["best_remaining", "session_billing", "turn_lookup"]


@pytest.mark.asyncio
async def test_that_over_cap_turn_matched_tools_are_trimmed_by_individual_tool_score() -> None:
    rule = create_rule("customer needs a complex workflow")
    low_tool = ToolId("local", "low_turn")
    high_tool = ToolId("local", "high_turn")
    middle_tool = ToolId("local", "middle_turn")
    tools = {
        low_tool: _tool("low_turn"),
        high_tool: _tool("high_turn"),
        middle_tool: _tool("middle_turn"),
    }
    associations = [
        _association(rule, low_tool),
        _association(rule, high_tool),
        _association(rule, middle_tool),
    ]
    recaller = _ScoredToolRecaller(
        associations,
        tools,
        scores={low_tool: 0.1, high_tool: 0.9, middle_tool: 0.5},
        max_tools=2,
    )
    context = _context_with_rules(rule)
    context.state.tool_enabled_rule_matches = {
        RuleMatch(rule=rule, rationale="Relevant."): [
            low_tool,
            high_tool,
            middle_tool,
        ]
    }

    selected = await _prepare_and_select(recaller, context)

    assert selected == ["high_turn", "middle_turn"]
