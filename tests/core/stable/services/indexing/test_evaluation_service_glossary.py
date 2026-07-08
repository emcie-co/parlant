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

"""Glossary-term fetching for rule evaluation prompts: the signal and title
proposers should receive the terms RELEVANT to the rule under evaluation
(ranked by the rule's own query), not the agent's entire glossary."""

from datetime import datetime, timezone
from typing import Any, Sequence, cast
from unittest.mock import AsyncMock

from parlant.core.agents import AgentId
from parlant.core.common import generate_id
from parlant.core.evaluations import RulePayload, PayloadOperation
from parlant.core.glossary import Term, TermId
from parlant.core.rules import RuleContent
from parlant.core.services.indexing.evaluation_service import RuleEvaluator


def _term(name: str, description: str) -> Term:
    now = datetime.now(timezone.utc)
    return Term(
        id=TermId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        name=name,
        description=description,
        synonyms=[],
        groups=[],
    )


class _FakeEntityQueries:
    def __init__(self, relevant_terms: Sequence[Term]) -> None:
        self.relevant_terms = list(relevant_terms)
        self.find_glossary_calls: list[tuple[str, int]] = []

    async def read_agent(self, agent_id: AgentId) -> object:
        return object()

    async def find_glossary_terms_for_context(
        self,
        agent_id: AgentId,
        query: str,
        max_terms: int = 20,
    ) -> Sequence[Term]:
        self.find_glossary_calls.append((query, max_terms))
        return list(self.relevant_terms)


def _make_evaluator(
    entity_queries: _FakeEntityQueries,
) -> tuple[RuleEvaluator, AsyncMock, AsyncMock]:
    signal_proposer = AsyncMock()
    title_proposer = AsyncMock()
    evaluator = object.__new__(RuleEvaluator)
    evaluator._entity_queries = cast(Any, entity_queries)
    evaluator._rule_signal_proposer = cast(Any, signal_proposer)
    evaluator._rule_title_proposer = cast(Any, title_proposer)
    return evaluator, signal_proposer, title_proposer


def _payload(
    *,
    signal_proposition: bool = False,
    title_proposition: bool = False,
    agent_id: AgentId | None = AgentId("test-agent"),
) -> RulePayload:
    return RulePayload(
        content=RuleContent(
            condition="the customer reports PRS",
            action="escalate to a specialist",
            description=None,
        ),
        tool_ids=[],
        operation=PayloadOperation.ADD,
        action_proposition=False,
        properties_proposition=False,
        journey_node_proposition=False,
        signal_proposition=signal_proposition,
        title_proposition=title_proposition,
        title="PRS Escalation",
        agent_id=agent_id,
    )


async def test_that_signal_proposal_fetches_terms_relevant_to_the_rule() -> None:
    term = _term("PRS", "Pinewood Rash Syndrome - an allergy to pinewood.")
    entity_queries = _FakeEntityQueries([term])
    evaluator, signal_proposer, _ = _make_evaluator(entity_queries)

    await evaluator._propose_signals([_payload(signal_proposition=True)], [None])

    assert len(entity_queries.find_glossary_calls) == 1
    query, k = entity_queries.find_glossary_calls[0]
    assert "the customer reports PRS" in query
    assert k == RuleEvaluator._TERMS_PER_RULE

    signal_proposer.propose_signals.assert_awaited_once()
    kwargs = signal_proposer.propose_signals.await_args.kwargs
    assert kwargs["glossary_terms"] == [term]


async def test_that_title_proposal_fetches_terms_relevant_to_the_rule() -> None:
    term = _term("PRS", "Pinewood Rash Syndrome - an allergy to pinewood.")
    entity_queries = _FakeEntityQueries([term])
    evaluator, _, title_proposer = _make_evaluator(entity_queries)

    await evaluator._propose_titles([_payload(title_proposition=True)], [None])

    assert len(entity_queries.find_glossary_calls) == 1
    query, _query_k = entity_queries.find_glossary_calls[0]
    assert "the customer reports PRS" in query

    title_proposer.propose_title.assert_awaited_once()
    kwargs = title_proposer.propose_title.await_args.kwargs
    assert kwargs["glossary_terms"] == [term]


async def test_that_evaluation_without_an_agent_fetches_no_terms() -> None:
    entity_queries = _FakeEntityQueries([_term("PRS", "irrelevant here")])
    evaluator, signal_proposer, _ = _make_evaluator(entity_queries)

    await evaluator._propose_signals([_payload(signal_proposition=True, agent_id=None)], [None])

    assert entity_queries.find_glossary_calls == []
    kwargs = signal_proposer.propose_signals.await_args.kwargs
    assert kwargs["glossary_terms"] == []
