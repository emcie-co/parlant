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

from typing import Any, Sequence, cast

from parlant.core.container import Container

from parlant.core.agents import Agent, AgentStore
import pytest

from parlant.core.entity_cq import (
    EntityQueries,
    TurnCannedResponses,
    select_turn_canned_responses,
)
from parlant.core.glossary import GlossaryStore
from parlant.core.rules import RuleStore
from parlant.core.canned_responses import (
    CannedResponse,
    CannedResponseRelevanceResult,
)
from parlant.core.groups import GroupIds, GroupId
from parlant.core.context_variables import ContextVariableStore
from parlant.core.canned_responses import CannedResponseStore
from tests.core.stable.query_instrumentation import instrument_queries


async def test_that_list_rules_with_mutual_agent_group_are_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    agent_store = container[AgentStore]
    rule_store = container[RuleStore]

    await agent_store.upsert_group(
        agent_id=agent.id,
        group_id=GroupId("tag_1"),
    )

    first_rule = await rule_store.create_rule(
        condition="condition 1",
        content="action 1",
    )

    second_rule = await rule_store.create_rule(
        condition="condition 2",
        content="action 2",
    )

    await rule_store.upsert_group(
        rule_id=first_rule.id,
        group_id=GroupId("tag_1"),
    )

    await rule_store.upsert_group(
        rule_id=second_rule.id,
        group_id=GroupId("tag_2"),
    )

    result = await entity_queries.find_rules_for_context(agent.id)

    assert len(result) == 1
    assert result[0].id == first_rule.id


async def test_that_list_rules_global_rule_is_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]

    global_rule = await rule_store.create_rule(
        condition="condition 1",
        content="action 1",
    )

    result = await entity_queries.find_rules_for_context(agent.id)

    assert len(result) == 1
    assert result[0].id == global_rule.id


async def test_that_rule_with_not_hierarchy_tag_is_not_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]

    first_rule = await rule_store.create_rule(
        condition="condition 1",
        content="action 1",
    )

    second_rule = await rule_store.create_rule(
        condition="condition 2",
        content="action 2",
    )

    await rule_store.upsert_group(
        rule_id=first_rule.id,
        group_id=GroupIds.for_agent_id(agent.id),
    )

    await rule_store.upsert_group(
        rule_id=second_rule.id,
        group_id=GroupId("tag_2"),
    )

    result = await entity_queries.find_rules_for_context(agent.id)

    assert len(result) == 1
    assert result[0].id == first_rule.id


async def test_that_find_glossary_terms_for_agent_returns_all_when_no_tags(
    container: Container,
    agent: Agent,
) -> None:
    glossary_store = container[GlossaryStore]
    entity_queries = container[EntityQueries]

    untagged_term = await glossary_store.create_term(
        name="Hello world",
        description="A greeting",
        groups=[],
    )

    group = GroupId("group1")
    await glossary_store.create_term(
        name="Grouped term",
        description="A grouped glossary entry",
        groups=[group],
    )

    results = await entity_queries.find_glossary_terms_for_context(agent_id=agent.id, query="Hello")
    assert len(results) == 1
    assert results[0].id == untagged_term.id


async def test_that_find_glossary_terms_for_agent_returns_none_for_non_matching_tag(
    container: Container,
    agent: Agent,
) -> None:
    glossary_store = container[GlossaryStore]
    entity_queries = container[EntityQueries]

    group1 = GroupId("group1")
    await glossary_store.create_term(
        name="Grouped term",
        description="A grouped glossary entry",
        groups=[group1],
    )

    await container[AgentStore].upsert_group(
        agent_id=agent.id, group_id=GroupId("non_matching_tag")
    )

    results = await entity_queries.find_glossary_terms_for_context(
        agent_id=agent.id, query="Grouped"
    )
    assert len(results) == 0


class _RankByOrderStore:
    """Fake store whose relevance ranking is just the order it's given the
    candidates, capped at ``max_count``. Enough to observe how the precedence
    tiers are filled without embeddings."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[CannedResponse], int]] = []

    async def filter_relevant_canned_responses(
        self,
        query: str,
        available_canned_responses: Sequence[CannedResponse],
        max_count: int,
    ) -> Sequence[CannedResponseRelevanceResult]:
        candidates = list(available_canned_responses)
        self.calls.append((candidates, max_count))
        return [
            CannedResponseRelevanceResult(canned_response=c, score=1.0)
            for c in candidates[:max_count]
        ]


class _RecordingLogger:
    """Keeps the warnings it is given, so a test can read them back."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _logger() -> Any:
    return cast(Any, _RecordingLogger())


def _transient_canrep(value: str) -> CannedResponse:
    return CannedResponse.create_transient(value=value)


async def test_that_rule_scoped_canned_responses_take_the_slots_before_agent_wide_ones() -> None:
    store = cast(Any, _RankByOrderStore())
    rule = _transient_canrep("rule response")
    agent_a = _transient_canrep("agent response A")
    agent_b = _transient_canrep("agent response B")
    turn = TurnCannedResponses(rule_scoped=[rule], agent_wide=[agent_a, agent_b])

    # A single slot goes to the rule-scoped response; agent-wide ones are excluded.
    only_one = await select_turn_canned_responses(
        store, "query", turn, max_count=1, logger=_logger()
    )
    assert [r.canned_response.value for r in only_one] == ["rule response"]

    # With room to spare, the rule-scoped response still comes first and agent-wide
    # ones fill the remaining slots in relevance order.
    with_room = await select_turn_canned_responses(
        store, "query", turn, max_count=3, logger=_logger()
    )
    assert [r.canned_response.value for r in with_room] == [
        "rule response",
        "agent response A",
        "agent response B",
    ]


async def test_that_agent_wide_canned_responses_are_used_when_no_rules_contribute() -> None:
    store = cast(Any, _RankByOrderStore())
    agent_a = _transient_canrep("agent response A")
    turn = TurnCannedResponses(rule_scoped=[], agent_wide=[agent_a])

    result = await select_turn_canned_responses(store, "query", turn, max_count=5, logger=_logger())
    assert [r.canned_response.value for r in result] == ["agent response A"]


async def test_that_empty_canned_response_tiers_are_not_queried() -> None:
    store = _RankByOrderStore()
    turn = TurnCannedResponses(rule_scoped=[], agent_wide=[])

    result = await select_turn_canned_responses(
        cast(Any, store), "query", turn, max_count=5, logger=_logger()
    )

    assert result == []
    assert store.calls == []


async def test_that_unranked_selection_returns_every_eligible_canned_response() -> None:
    store = _RankByOrderStore()
    rule = _transient_canrep("rule response")
    agent_a = _transient_canrep("agent response A")
    agent_b = _transient_canrep("agent response B")
    turn = TurnCannedResponses(rule_scoped=[rule], agent_wide=[agent_a, agent_b])

    result = await select_turn_canned_responses(
        cast(Any, store),
        "query",
        turn,
        max_count=20,
        logger=_logger(),
        rank=False,
    )

    assert [r.canned_response.value for r in result] == [
        "rule response",
        "agent response A",
        "agent response B",
    ]


async def test_that_unranked_selection_never_queries_the_store() -> None:
    # Ranking is what embeds the customer's message and searches the vector
    # store. Low-effort turns must do neither.
    store = _RankByOrderStore()
    turn = TurnCannedResponses(
        rule_scoped=[_transient_canrep("rule response")],
        agent_wide=[_transient_canrep("agent response")],
    )

    await select_turn_canned_responses(
        cast(Any, store),
        "query",
        turn,
        max_count=20,
        logger=_logger(),
        rank=False,
    )

    assert store.calls == []


async def test_that_unranked_selection_keeps_precedence_order_when_it_clips() -> None:
    store = _RankByOrderStore()
    tool_provided = [_transient_canrep("tool response")]
    turn = TurnCannedResponses(
        rule_scoped=[_transient_canrep("rule response")],
        agent_wide=[_transient_canrep("agent response A"), _transient_canrep("agent response B")],
    )

    result = await select_turn_canned_responses(
        cast(Any, store),
        "query",
        turn,
        max_count=2,
        logger=_logger(),
        tool_provided=tool_provided,
        rank=False,
    )

    assert [r.canned_response.value for r in result] == ["tool response", "rule response"]


async def test_that_clipping_warns_with_the_number_left_out() -> None:
    logger = _RecordingLogger()
    turn = TurnCannedResponses(
        rule_scoped=[],
        agent_wide=[_transient_canrep(f"response {i}") for i in range(5)],
    )

    await select_turn_canned_responses(
        cast(Any, _RankByOrderStore()),
        "query",
        turn,
        max_count=3,
        logger=cast(Any, logger),
        rank=False,
    )

    assert len(logger.warnings) == 1
    assert "2 were left" in logger.warnings[0]


async def test_that_a_set_that_fits_the_cap_is_not_warned_about() -> None:
    logger = _RecordingLogger()
    turn = TurnCannedResponses(rule_scoped=[], agent_wide=[_transient_canrep("response")])

    await select_turn_canned_responses(
        cast(Any, _RankByOrderStore()),
        "query",
        turn,
        max_count=3,
        logger=cast(Any, logger),
        rank=False,
    )

    assert logger.warnings == []


async def test_that_ranked_selection_still_queries_the_store_for_every_tier() -> None:
    # Guards the default path: MEDIUM and above must keep ranking exactly as
    # they do today.
    store = _RankByOrderStore()
    turn = TurnCannedResponses(
        rule_scoped=[_transient_canrep("rule response")],
        agent_wide=[_transient_canrep("agent response")],
    )

    result = await select_turn_canned_responses(
        cast(Any, store),
        "query",
        turn,
        max_count=5,
        logger=_logger(),
    )

    assert [r.canned_response.value for r in result] == ["rule response", "agent response"]
    assert len(store.calls) == 2


# ──────────────────────────── concurrent loading ────────────────────────────
#
# Each of these queries several stores for one turn. The calls don't depend on
# each other (bar the agent read that the group query needs), so they should go
# out together rather than one after another. Only the query boundary shows the
# difference — the results are identical either way. See query_instrumentation.


async def _agent_with_a_group(container: Container, agent: Agent) -> Agent:
    await container[AgentStore].upsert_group(agent_id=agent.id, group_id=GroupId("shared"))
    return await container[AgentStore].read_agent(agent.id)


async def test_that_rules_for_context_are_loaded_concurrently(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _agent_with_a_group(container, agent)
    await container[RuleStore].create_rule(content="global rule")
    await container[RuleStore].create_rule(
        content="agent rule",
        groups=[GroupIds.for_agent_id(agent.id)],
    )

    log = instrument_queries(monkeypatch)

    await container[EntityQueries].find_rules_for_context(agent.id)

    assert log.max_queries_in_flight >= 2


async def test_that_context_variables_for_context_are_loaded_concurrently(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _agent_with_a_group(container, agent)
    await container[ContextVariableStore].create_variable(name="global variable")

    log = instrument_queries(monkeypatch)

    await container[EntityQueries].find_context_variables_for_context(agent.id)

    assert log.max_queries_in_flight >= 2


async def test_that_glossary_terms_for_context_are_loaded_concurrently(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _agent_with_a_group(container, agent)
    await container[GlossaryStore].create_term(name="term", description="a term")

    log = instrument_queries(monkeypatch)

    await container[EntityQueries].list_glossary_terms_for_context(agent.id)

    assert log.max_queries_in_flight >= 2


async def test_that_turn_canned_responses_are_loaded_concurrently(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule = await container[RuleStore].create_rule(content="a rule")

    await container[CannedResponseStore].create_canned_response(
        value="rule response",
        groups=[GroupIds.for_rule_id(rule.id)],
    )
    await container[CannedResponseStore].create_canned_response(
        value="agent response",
        groups=[GroupIds.for_agent_id(agent.id)],
    )

    log = instrument_queries(monkeypatch)

    await container[EntityQueries].find_canned_responses_for_turn([rule], agent.id)

    assert log.max_queries_in_flight >= 2


async def test_that_an_agent_without_groups_skips_the_redundant_global_query(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An agent in no groups makes the group query `list_rules(groups=[])`, which
    # is the global query already issued alongside it. Asking twice is pure
    # waste, and the union is identical either way.
    assert not agent.groups

    await container[RuleStore].create_rule(content="global rule")
    await container[RuleStore].create_rule(
        content="agent rule",
        groups=[GroupIds.for_agent_id(agent.id)],
    )

    log = instrument_queries(monkeypatch)

    await container[EntityQueries].find_rules_for_context(agent.id)

    assert log.count_for("rules") == 2


async def test_that_a_failing_store_still_fails_the_whole_concurrent_load(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Gathering must not swallow or defer an error: a caller that used to see a
    # store failure has to keep seeing it.
    async def failing_read_agent(*args: Any, **kwargs: Any) -> Agent:
        raise RuntimeError("store is down")

    monkeypatch.setattr(
        container[AgentStore],
        "read_agent",
        failing_read_agent,
    )

    with pytest.raises(RuntimeError, match="store is down"):
        await container[EntityQueries].find_rules_for_context(agent.id)
