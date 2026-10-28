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

# mypy: ignore-errors

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, cast

import pytest
from parlant.core.container import Container

from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.engines.compass.matching.tool_selection import ToolSelector
from parlant.core.engines.compass.matching.tool_recaller import ToolRecaller
from parlant.core.engines.compass.context import EngineContext, ResponseState
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.rule_tool_associations import (
    RuleToolAssociation,
    RuleToolAssociationId,
)
from parlant.core.rules import Rule
from parlant.core.sessions import EventSource
from parlant.core.tools import Tool, ToolContext, ToolId, ToolOverlap, ToolResult, ToolService

from tests.core.stable.engines.compass.matching.utils import create_engine_context, create_rule


def test_that_the_tool_recaller_implements_the_tool_selector_port() -> None:
    assert issubclass(ToolRecaller, ToolSelector)


def test_that_the_container_binds_the_tool_selector_to_the_tool_recaller(
    container: Container,
) -> None:
    selector = container[ToolSelector]

    assert isinstance(selector, ToolRecaller)
    assert selector is container[ToolSelector]


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
            model_router=None,
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
    discovered_rules: set[Rule] | None = None,
) -> EngineContext:
    context = create_engine_context(conversation=[])
    context.state = ResponseState(
        usable_rules=list(rules),
        discovered_rules=discovered_rules or set(),
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
        discovered_rules={session_rule},
    )
    context.state.tool_enabled_rule_matches = {
        RuleEvaluation(rule=turn_rule, reasoning="Relevant."): [turn_tool]
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
        discovered_rules={session_rule},
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
    context = _context_with_rules(rule, discovered_rules={rule})

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
        discovered_rules={session_rule},
    )
    context.state.tool_enabled_rule_matches = {
        RuleEvaluation(rule=turn_rule, reasoning="Relevant."): [turn_tool]
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
        RuleEvaluation(rule=rule, reasoning="Relevant."): [
            low_tool,
            high_tool,
            middle_tool,
        ]
    }

    selected = await _prepare_and_select(recaller, context)

    assert selected == ["high_turn", "middle_turn"]


class _WordCountTokenizer(EstimatingTokenizer):
    async def estimate_token_count(self, prompt: str) -> int:
        return len(prompt.split())


class _BudgetedMarkerEmbedder(Embedder):
    """Deterministic marker-based embedder that mimics the embedding API's hard
    input cap (word-count tokenization), raising on over-limit input."""

    def __init__(self, max_tokens: int) -> None:
        self.embed_calls: list[list[str]] = []
        self._max_tokens = max_tokens
        self._tokenizer = _WordCountTokenizer()

    async def embed(self, texts: list[str], hints: Mapping[str, Any] = {}) -> EmbeddingResult:
        for text in texts:
            if len(text.split()) > self._max_tokens:
                raise RuntimeError(
                    f"embedding input of {len(text.split())} tokens exceeds "
                    f"the {self._max_tokens}-token model limit"
                )
        self.embed_calls.append(list(texts))
        return EmbeddingResult(vectors=[self._vector_for_text(text) for text in texts])

    @property
    def id(self) -> str:
        return "budgeted-marker-embedder"

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    def dimensions(self) -> int:
        return 2

    def _vector_for_text(self, text: str) -> list[float]:
        lowered = text.lower()

        if any(term in lowered for term in ("refund", "money")):
            return [1.0, 0.0]

        if any(term in lowered for term in ("hours", "open")):
            return [-1.0, 0.0]

        return [0.0, -1.0]


class _FakeModelRouter:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def get_embedder(self, **kwargs: Any) -> Embedder:
        return self._embedder


@pytest.mark.asyncio
async def test_that_a_cumulative_tool_query_exceeding_the_embedder_token_budget_is_chunked_and_scoring_still_succeeds() -> (
    None
):
    refund_tool_id = ToolId("local", "refund_flow")
    hours_tool_id = ToolId("local", "store_hours")
    tools = {
        refund_tool_id: _tool("refund_flow", description="Start a refund for the customer"),
        hours_tool_id: _tool("store_hours", description="Tell the customer our opening hours"),
    }

    embedder = _BudgetedMarkerEmbedder(max_tokens=32)

    recaller = ToolRecaller(
        entity_queries=_FakeEntityQueries([], tools),
        model_router=cast(Any, _FakeModelRouter(embedder)),
        embedding_cache=None,
        max_tools=16,
    )

    filler = "the weather is nice today and the sky looks blue"
    conversation = [(EventSource.CUSTOMER, filler) for _ in range(6)] + [
        (EventSource.CUSTOMER, "I want my money back")
    ]
    context = create_engine_context(conversation=conversation)
    context.state = ResponseState()

    scores = await recaller._score_tools(context, tools)

    assert scores[refund_tool_id] > scores[hours_tool_id]
    assert all(len(text.split()) <= 32 for call in embedder.embed_calls for text in call)
