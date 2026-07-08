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

from datetime import datetime, timezone
from dataclasses import replace
from typing import Any, Mapping, Sequence, cast
from unittest.mock import AsyncMock

from lagom import Container
from pytest import fixture

from parlant.core.agents import AgentId
from parlant.core.common import JSONSerializable, generate_id
from parlant.core.engines.compass.matching.glossary_recaller import (
    GlossaryRecaller,
    _SESSION_TERMS_METADATA_KEY,
)
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.glossary import Term, TermId
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    RecordedEvent,
    RecordedSpan,
    RecordingTracer,
    create_engine_context,
)


@fixture
def glossary_recaller(container: Container) -> GlossaryRecaller:
    return container[GlossaryRecaller]


def test_that_a_glossary_recaller_can_be_created(glossary_recaller: GlossaryRecaller) -> None:
    assert glossary_recaller is not None


# --- Fakes ---------------------------------------------------------------------
#
# The utils' create_term gives every term the same TermId("-"), which is useless
# for id-keyed session metadata — so terms here carry unique ids.


def _term(name: str, description: str = "", synonyms: list[str] = []) -> Term:
    now = datetime.now(timezone.utc)
    return Term(
        id=TermId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        name=name,
        description=description or f"the meaning of {name}",
        synonyms=list(synonyms),
        groups=[],
    )


def _fillers(count: int) -> list[Term]:
    return [_term(f"filler term {i:03d}") for i in range(count)]


class _FakeEntityQueries:
    """Inventory listing plus keyword-driven relevance search: a query containing
    a registered keyword "finds" that keyword's terms (stands in for the vector
    search, keeping tests deterministic with no embedder)."""

    def __init__(
        self,
        inventory: Sequence[Term],
        relevant_by_query_keyword: Mapping[str, Sequence[Term]] = {},
    ) -> None:
        self.inventory = list(inventory)
        self.relevant_by_query_keyword = dict(relevant_by_query_keyword)
        self.find_calls: list[tuple[str, int]] = []

    async def list_glossary_terms_for_context(self, agent_id: AgentId) -> Sequence[Term]:
        return list(self.inventory)

    async def find_glossary_terms_for_context(
        self,
        agent_id: AgentId,
        query: str,
        max_terms: int = 20,
    ) -> Sequence[Term]:
        self.find_calls.append((query, max_terms))
        hits: list[Term] = []
        for keyword, terms in self.relevant_by_query_keyword.items():
            if keyword.lower() in query.lower():
                hits.extend(terms)
        return hits[:max_terms]


class _FakeEntityCommands:
    def __init__(self) -> None:
        self.update_session = AsyncMock()


def _make_recaller(
    queries: _FakeEntityQueries,
    commands: _FakeEntityCommands | None = None,
    tracer: LocalTracer | None = None,
) -> GlossaryRecaller:
    tracer = tracer or LocalTracer()
    return GlossaryRecaller(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        entity_queries=cast(Any, queries),
        entity_commands=cast(Any, commands or _FakeEntityCommands()),
    )


def _context(
    conversation: list[tuple[EventSource, str]],
    session_terms: Mapping[str, int] | None = None,
) -> EngineContext:
    """``session_terms`` set (even to {}) marks the session as already seeded;
    None leaves the metadata key absent (a fresh/legacy session)."""
    context = create_engine_context(conversation=conversation)
    if session_terms is not None:
        metadata: dict[str, JSONSerializable] = {_SESSION_TERMS_METADATA_KEY: dict(session_terms)}
        context.session = replace(context.session, metadata=metadata)
    context.state = ResponseState()
    return context


def _persisted_session_terms(commands: _FakeEntityCommands) -> Mapping[str, int]:
    assert commands.update_session.await_args is not None
    metadata = commands.update_session.await_args.args[1]["metadata"]
    return cast(Mapping[str, int], metadata[_SESSION_TERMS_METADATA_KEY])


# --- Fast path: small glossaries -----------------------------------------------


async def test_that_a_small_glossary_is_always_fully_loaded() -> None:
    inventory = _fillers(GlossaryRecaller._MAX_GLOSSARY_TERMS)
    queries = _FakeEntityQueries(inventory)
    commands = _FakeEntityCommands()
    recaller = _make_recaller(queries, commands)
    context = _context([(EventSource.CUSTOMER, "hello there")])

    await recaller.recall(context)

    assert context.state.glossary_terms == set(inventory)
    assert queries.find_calls == []  # no search at all
    commands.update_session.assert_not_awaited()  # no discovery bookkeeping


async def test_that_recall_records_span_and_loaded_glossary_events() -> None:
    account = _term("Account", description="customer account")
    refund = _term("Refund", description="money returned to customer")
    span_tracer = RecordingTracer()
    context_tracer = RecordingTracer()
    recaller = _make_recaller(
        _FakeEntityQueries([refund, account]),
        tracer=span_tracer,
    )
    context = replace(
        _context([(EventSource.CUSTOMER, "hello there")]),
        tracer=context_tracer,
    )

    await recaller.recall(context)

    assert span_tracer.started_spans == [RecordedSpan(name="match.glossary.recall", attributes={})]
    assert context_tracer.events == [
        RecordedEvent(
            name="loaded.glossary",
            attributes={
                "term_id": str(account.id),
                "name": "Account",
                "last_modified": account.modified_utc.isoformat(),
            },
            span_id="<main>",
        ),
        RecordedEvent(
            name="loaded.glossary",
            attributes={
                "term_id": str(refund.id),
                "name": "Refund",
                "last_modified": refund.modified_utc.isoformat(),
            },
            span_id="<main>",
        ),
    ]


# --- Lexical discovery ----------------------------------------------------------


async def test_that_a_term_lexically_mentioned_in_a_new_message_is_discovered() -> None:
    prs = _term("PRS", description="Pinewood Rash Syndrome, an allergy to pinewood")
    inventory = [*_fillers(31), prs]
    queries = _FakeEntityQueries(inventory)
    recaller = _make_recaller(queries)
    context = _context(
        [(EventSource.CUSTOMER, "I have PRS - will the trail be a problem for me?")],
        session_terms={},
    )

    await recaller.recall(context)

    assert prs in context.state.glossary_terms


async def test_that_a_synonym_mention_discovers_the_term() -> None:
    prs = _term(
        "Pinewood Rash Syndrome",
        description="an allergy to pinewood trees",
        synonyms=["Pine Rash", "PRS"],
    )
    inventory = [*_fillers(31), prs]
    recaller = _make_recaller(_FakeEntityQueries(inventory))
    context = _context(
        [(EventSource.CUSTOMER, "my son suffers from pine rash, is that an issue?")],
        session_terms={},
    )

    await recaller.recall(context)

    assert prs in context.state.glossary_terms


async def test_that_lexical_matching_respects_word_boundaries() -> None:
    prs = _term("PRS", description="Pinewood Rash Syndrome")
    inventory = [*_fillers(31), prs]
    recaller = _make_recaller(_FakeEntityQueries(inventory))
    context = _context(
        [(EventSource.CUSTOMER, "what a surprise that was!")],  # 'prs' inside 'surprise'
        session_terms={},
    )

    await recaller.recall(context)

    assert prs not in context.state.glossary_terms


async def test_that_multi_word_synonyms_match_as_phrases() -> None:
    guarantee = _term(
        "Satisfaction Guarantee",
        description="our refund promise",
        synonyms=["money back guarantee"],
    )
    inventory = [*_fillers(31), guarantee]
    recaller = _make_recaller(_FakeEntityQueries(inventory))
    context = _context(
        [(EventSource.CUSTOMER, "do you offer a Money Back Guarantee on this?")],
        session_terms={},
    )

    await recaller.recall(context)

    assert guarantee in context.state.glossary_terms


# --- Embedding discovery --------------------------------------------------------


async def test_that_embedding_hits_from_new_messages_are_discovered() -> None:
    thirst = _term("Thirst Protocol", description="how to serve thirsty customers")
    inventory = [*_fillers(31), thirst]
    queries = _FakeEntityQueries(inventory, {"thirsty": [thirst]})
    recaller = _make_recaller(queries)
    context = _context(
        [(EventSource.CUSTOMER, "I'm feeling really thirsty, what do you have?")],
        session_terms={},
    )

    await recaller.recall(context)

    assert thirst in context.state.glossary_terms


async def test_that_discovery_only_queries_messages_after_the_last_agent_reply() -> None:
    warranty = _term("Warranty", description="our coverage policy")
    inventory = [*_fillers(31), warranty]
    queries = _FakeEntityQueries(inventory, {"warranty": [warranty]})
    recaller = _make_recaller(queries)
    context = _context(
        [
            (EventSource.CUSTOMER, "tell me about the warranty"),  # BEFORE the agent's reply
            (EventSource.AI_AGENT, "sure, our warranty covers two years."),
            (EventSource.CUSTOMER, "and what about shipping times?"),  # the only new message
        ],
        session_terms={},
    )

    await recaller.recall(context)

    # The old message is never re-queried (neither lexically nor by embedding) —
    # this is the structural no-flapping property.
    assert all("warranty" not in query.lower() for query, _ in queries.find_calls)
    assert any("shipping" in query.lower() for query, _ in queries.find_calls)
    assert warranty not in context.state.glossary_terms


# --- Stickiness, offsets & persistence -------------------------------------------


async def test_that_discovered_terms_are_persisted_to_session_metadata() -> None:
    prs = _term("PRS", description="Pinewood Rash Syndrome")
    inventory = [*_fillers(31), prs]
    commands = _FakeEntityCommands()
    recaller = _make_recaller(_FakeEntityQueries(inventory), commands)
    context = _context(
        [
            (EventSource.CUSTOMER, "hello"),  # offset 0
            (EventSource.AI_AGENT, "hi, how can I help?"),  # offset 1
            (EventSource.CUSTOMER, "I have PRS, is that a problem?"),  # offset 2
        ],
        session_terms={},
    )

    await recaller.recall(context)

    commands.update_session.assert_awaited_once()
    persisted = _persisted_session_terms(commands)
    assert persisted[str(prs.id)] == 2  # the offset of the message that hit


async def test_that_the_session_term_set_is_sticky_across_turns() -> None:
    prs = _term("PRS", description="Pinewood Rash Syndrome")
    inventory = [*_fillers(31), prs]
    recaller = _make_recaller(_FakeEntityQueries(inventory))
    # Discovered on an earlier turn (present in metadata); the new message is
    # entirely unrelated.
    context = _context(
        [
            (EventSource.CUSTOMER, "I have PRS"),  # offset 0 (already processed)
            (EventSource.AI_AGENT, "noted!"),  # offset 1
            (EventSource.CUSTOMER, "what are your opening hours?"),  # offset 2
        ],
        session_terms={str(prs.id): 0},
    )

    await recaller.recall(context)

    assert prs in context.state.glossary_terms


async def test_that_a_repeat_mention_refreshes_the_terms_last_hit_offset() -> None:
    prs = _term("PRS", description="Pinewood Rash Syndrome")
    inventory = [*_fillers(31), prs]
    commands = _FakeEntityCommands()
    recaller = _make_recaller(_FakeEntityQueries(inventory), commands)
    context = _context(
        [
            (EventSource.CUSTOMER, "I have PRS"),  # offset 0 (already processed)
            (EventSource.AI_AGENT, "noted!"),  # offset 1
            (EventSource.CUSTOMER, "so does PRS affect my booking?"),  # offset 2
        ],
        session_terms={str(prs.id): 0},
    )

    await recaller.recall(context)

    commands.update_session.assert_awaited_once()
    assert _persisted_session_terms(commands)[str(prs.id)] == 2


async def test_that_metadata_is_not_written_when_nothing_changed() -> None:
    prs = _term("PRS", description="Pinewood Rash Syndrome")
    inventory = [*_fillers(31), prs]
    commands = _FakeEntityCommands()
    recaller = _make_recaller(_FakeEntityQueries(inventory), commands)
    context = _context(
        [
            (EventSource.CUSTOMER, "I have PRS"),  # offset 0 (already processed)
            (EventSource.AI_AGENT, "noted!"),  # offset 1
            (EventSource.CUSTOMER, "what are your opening hours?"),  # offset 2, no hits
        ],
        session_terms={str(prs.id): 0},
    )

    await recaller.recall(context)

    commands.update_session.assert_not_awaited()


async def test_that_session_terms_absent_from_the_inventory_are_dropped() -> None:
    # A term deleted from the store must not linger in the working set.
    prs = _term("PRS", description="Pinewood Rash Syndrome")
    inventory = [*_fillers(31), prs]
    recaller = _make_recaller(_FakeEntityQueries(inventory))
    context = _context(
        [(EventSource.CUSTOMER, "hello")],
        session_terms={str(prs.id): 0, "deleted-term-id": 0},
    )

    await recaller.recall(context)

    assert context.state.glossary_terms == {prs}


# --- Seeding (fresh or legacy session, over-cap inventory) -----------------------


async def test_that_a_session_without_metadata_is_seeded_from_the_whole_conversation() -> None:
    warranty = _term("Warranty", description="our coverage policy")
    inventory = [*_fillers(31), warranty]
    commands = _FakeEntityCommands()
    queries = _FakeEntityQueries(inventory, {"warranty": [warranty]})
    recaller = _make_recaller(queries, commands)
    context = _context(
        [
            (EventSource.CUSTOMER, "tell me about the warranty"),
            (EventSource.AI_AGENT, "sure, two years of coverage."),
            (EventSource.CUSTOMER, "great, thanks!"),
        ],
        session_terms=None,  # metadata key absent
    )

    await recaller.recall(context)

    # The seed query spans the WHOLE conversation (not just the tail) with the
    # full cap, so the earlier warranty mention is picked up exactly once.
    seed_calls = [
        (query, k) for query, k in queries.find_calls if k == GlossaryRecaller._MAX_GLOSSARY_TERMS
    ]
    assert len(seed_calls) == 1
    assert "warranty" in seed_calls[0][0].lower()

    assert warranty in context.state.glossary_terms
    commands.update_session.assert_awaited_once()
    assert str(warranty.id) in _persisted_session_terms(commands)


# --- Pruning ---------------------------------------------------------------------


async def test_that_pruning_is_skipped_at_or_below_the_high_water_mark() -> None:
    inventory = _fillers(GlossaryRecaller._HIGH_WATER_MARK + 5)
    commands = _FakeEntityCommands()
    recaller = _make_recaller(_FakeEntityQueries(inventory), commands)
    session_terms = {
        str(term.id): i for i, term in enumerate(inventory[: GlossaryRecaller._HIGH_WATER_MARK])
    }
    context = _context([(EventSource.CUSTOMER, "hello")], session_terms=session_terms)

    await recaller.prune(context)

    commands.update_session.assert_not_awaited()


async def test_that_prune_records_span() -> None:
    inventory = _fillers(GlossaryRecaller._HIGH_WATER_MARK)
    tracer = RecordingTracer()
    recaller = _make_recaller(_FakeEntityQueries(inventory), tracer=tracer)
    session_terms = {str(term.id): i for i, term in enumerate(inventory)}
    context = _context([(EventSource.CUSTOMER, "hello")], session_terms=session_terms)

    await recaller.prune(context)

    assert tracer.started_spans == [RecordedSpan(name="match.glossary.prune", attributes={})]


async def test_that_pruning_evicts_least_recently_hit_terms_down_to_the_target() -> None:
    count = GlossaryRecaller._HIGH_WATER_MARK + 1
    inventory = _fillers(count + 5)
    commands = _FakeEntityCommands()
    recaller = _make_recaller(_FakeEntityQueries(inventory), commands)
    # Distinct last-hit offsets: term i was last hit at offset i.
    session_terms = {str(term.id): i for i, term in enumerate(inventory[:count])}
    context = _context([(EventSource.CUSTOMER, "hello")], session_terms=session_terms)

    await recaller.prune(context)

    commands.update_session.assert_awaited_once()
    persisted = _persisted_session_terms(commands)
    assert len(persisted) == GlossaryRecaller._MAX_GLOSSARY_TERMS

    # Exactly the most recently hit terms survive.
    expected_survivors = {
        str(term.id) for term in inventory[count - GlossaryRecaller._MAX_GLOSSARY_TERMS : count]
    }
    assert set(persisted) == expected_survivors


async def test_that_an_evicted_term_is_rediscovered_on_a_fresh_mention() -> None:
    prs = _term("PRS", description="Pinewood Rash Syndrome")
    inventory = [*_fillers(31), prs]
    commands = _FakeEntityCommands()
    recaller = _make_recaller(_FakeEntityQueries(inventory), commands)
    # prs was evicted (absent from metadata); the customer mentions it afresh.
    context = _context(
        [
            (EventSource.CUSTOMER, "hi there"),  # offset 0
            (EventSource.AI_AGENT, "hello!"),  # offset 1
            (EventSource.CUSTOMER, "one more thing - my PRS flared up"),  # offset 2
        ],
        session_terms={},
    )

    await recaller.recall(context)

    assert prs in context.state.glossary_terms
    assert _persisted_session_terms(commands)[str(prs.id)] == 2


async def test_that_an_evicted_term_stays_out_without_a_new_mention() -> None:
    prs = _term("PRS", description="Pinewood Rash Syndrome")
    inventory = [*_fillers(31), prs]
    commands = _FakeEntityCommands()
    recaller = _make_recaller(_FakeEntityQueries(inventory), commands)
    # prs was mentioned at offset 0 but has since been evicted; the tail is
    # unrelated. Because only new messages are ever queried, it cannot return.
    context = _context(
        [
            (EventSource.CUSTOMER, "my PRS is acting up"),  # offset 0 (history)
            (EventSource.AI_AGENT, "sorry to hear that!"),  # offset 1
            (EventSource.CUSTOMER, "anyway, what are your opening hours?"),  # offset 2
        ],
        session_terms={},
    )

    await recaller.recall(context)

    assert prs not in context.state.glossary_terms
    commands.update_session.assert_not_awaited()
