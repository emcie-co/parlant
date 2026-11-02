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

"""Glossary-term fetching for rule indexing prompts: the signal and title
proposers should receive the terms RELEVANT to the rule being indexed
(ranked by the rule's own query), not the agent's entire glossary."""

from datetime import datetime, timezone
from typing import Any, Sequence, cast

from parlant.core.agents import AgentId
from parlant.core.common import generate_id
from parlant.core.indexing import RulePayload, PayloadOperation
from parlant.core.glossary import Term, TermId
from parlant.core.loggers import LogLevel, StdoutLogger
from parlant.core.services.indexing.indexing_service import RuleIndexingProcessor
from parlant.core.services.indexing.rule_relativity_proposer import RuleRelativityProposition
from parlant.core.services.indexing.rule_signal_proposer import RuleSignalProposition
from parlant.core.services.indexing.rule_title_proposer import RuleTitleProposition
from parlant.core.tracer import LocalTracer


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


class _StubRelativityProposer:
    def __init__(
        self,
        proposition: RuleRelativityProposition | None = None,
    ) -> None:
        self.proposition = proposition or RuleRelativityProposition(references_other_policies=False)
        self.calls: list[dict[str, Any]] = []

    async def propose_relativity(self, **kwargs: Any) -> RuleRelativityProposition:
        self.calls.append(kwargs)
        return self.proposition


class _StubSignalProposer:
    def __init__(
        self,
        proposition: RuleSignalProposition | None = None,
    ) -> None:
        self.proposition = proposition or RuleSignalProposition(
            signals=[],
            anti_signals=[],
            rationale="Stub",
        )
        self.calls: list[dict[str, Any]] = []

    async def propose_signals(self, **kwargs: Any) -> RuleSignalProposition:
        self.calls.append(kwargs)
        return self.proposition


class _StubTitleProposer:
    def __init__(
        self,
        proposition: RuleTitleProposition | None = None,
    ) -> None:
        self.proposition = proposition or RuleTitleProposition(title="Stub", rationale="Stub")
        self.calls: list[dict[str, Any]] = []

    async def propose_title(self, **kwargs: Any) -> RuleTitleProposition:
        self.calls.append(kwargs)
        return self.proposition


def _make_processor(
    entity_queries: _FakeEntityQueries,
    relativity_proposer: _StubRelativityProposer | None = None,
    signal_proposer: _StubSignalProposer | None = None,
    title_proposer: _StubTitleProposer | None = None,
) -> tuple[RuleIndexingProcessor, _StubRelativityProposer, _StubSignalProposer, _StubTitleProposer]:
    relativity_proposer = relativity_proposer or _StubRelativityProposer()
    signal_proposer = signal_proposer or _StubSignalProposer()
    title_proposer = title_proposer or _StubTitleProposer()
    processor = RuleIndexingProcessor(
        logger=StdoutLogger(tracer=LocalTracer(), log_level=LogLevel.INFO),
        entity_queries=cast(Any, entity_queries),
        rule_signal_proposer=cast(Any, signal_proposer),
        rule_title_proposer=cast(Any, title_proposer),
        rule_relativity_proposer=cast(Any, relativity_proposer),
    )
    return processor, relativity_proposer, signal_proposer, title_proposer


def _payload(
    *,
    signal_proposition: bool = False,
    title_proposition: bool = False,
    agent_id: AgentId | None = AgentId("test-agent"),
) -> RulePayload:
    return RulePayload(
        condition="the customer reports PRS",
        content="escalate to a specialist",
        operation=PayloadOperation.ADD,
        signal_proposition=signal_proposition,
        title_proposition=title_proposition,
        title="PRS Escalation",
        agent_id=agent_id,
    )


async def test_that_signal_proposal_fetches_terms_relevant_to_the_rule() -> None:
    term = _term("PRS", "Pinewood Rash Syndrome - an allergy to pinewood.")
    entity_queries = _FakeEntityQueries([term])
    processor, _, signal_proposer, _ = _make_processor(entity_queries)

    await processor._propose_signals([_payload(signal_proposition=True)])

    assert len(entity_queries.find_glossary_calls) == 1
    query, k = entity_queries.find_glossary_calls[0]
    assert "the customer reports PRS" in query
    assert k == RuleIndexingProcessor._TERMS_PER_RULE

    assert len(signal_proposer.calls) == 1
    assert signal_proposer.calls[0]["glossary_terms"] == [term]


async def test_that_title_proposal_fetches_terms_relevant_to_the_rule() -> None:
    term = _term("PRS", "Pinewood Rash Syndrome - an allergy to pinewood.")
    entity_queries = _FakeEntityQueries([term])
    processor, _, _, title_proposer = _make_processor(entity_queries)

    await processor._propose_titles([_payload(title_proposition=True)])

    assert len(entity_queries.find_glossary_calls) == 1
    query, _query_k = entity_queries.find_glossary_calls[0]
    assert "the customer reports PRS" in query

    assert len(title_proposer.calls) == 1
    assert title_proposer.calls[0]["glossary_terms"] == [term]


async def test_that_indexing_without_an_agent_fetches_no_terms() -> None:
    entity_queries = _FakeEntityQueries([_term("PRS", "irrelevant here")])
    processor, _, signal_proposer, _ = _make_processor(entity_queries)

    await processor._propose_signals([_payload(signal_proposition=True, agent_id=None)])

    assert entity_queries.find_glossary_calls == []
    assert len(signal_proposer.calls) == 1
    assert signal_proposer.calls[0]["glossary_terms"] == []


async def test_that_rule_indexing_only_runs_the_remaining_propositions() -> None:
    entity_queries = _FakeEntityQueries([])
    processor, relativity_proposer, signal_proposer, title_proposer = _make_processor(
        entity_queries,
        relativity_proposer=_StubRelativityProposer(
            RuleRelativityProposition(references_other_policies=True)
        ),
        signal_proposer=_StubSignalProposer(
            RuleSignalProposition(
                signals=["customer reports PRS"],
                anti_signals=["unrelated allergy question"],
                rationale="Test",
            )
        ),
        title_proposer=_StubTitleProposer(
            RuleTitleProposition(
                title="PRS Escalation",
                rationale="Test",
            )
        ),
    )

    enabled = RulePayload(
        condition="customer reports PRS",
        content="escalate",
        operation=PayloadOperation.ADD,
        relativity_proposition=True,
        signal_proposition=True,
        title_proposition=True,
        agent_id=None,
    )
    disabled = RulePayload(
        condition="customer says goodbye",
        content="say goodbye",
        operation=PayloadOperation.ADD,
    )

    enabled_result, disabled_result = await processor.process([enabled, disabled])

    assert enabled_result.references_other_policies_proposition is True
    assert enabled_result.signals_proposition == ["customer reports PRS"]
    assert enabled_result.anti_signals_proposition == ["unrelated allergy question"]
    assert enabled_result.title_proposition == "PRS Escalation"
    assert disabled_result.references_other_policies_proposition is None
    assert disabled_result.signals_proposition is None
    assert disabled_result.anti_signals_proposition is None
    assert disabled_result.title_proposition is None
    assert len(relativity_proposer.calls) == 1
    assert len(signal_proposer.calls) == 1
    assert len(title_proposer.calls) == 1
