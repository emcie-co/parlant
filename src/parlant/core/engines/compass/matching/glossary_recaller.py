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

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from parlant.core.async_utils import safe_gather
from parlant.core.common import JSONSerializable
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.glossary import Term
from parlant.core.loggers import Logger
from parlant.core.sessions import EventKind, EventSource, MessageEventData
from parlant.core.tracer import Tracer


# Sticky per-session working set of glossary terms:
# {term id: event offset of the term's last discovery hit}.
_SESSION_TERMS_METADATA_KEY = "compass.session_terms"


class GlossaryRecaller:
    """Discovers and remembers the glossary terms in play for a session.

    Instead of re-ranking the whole glossary against the whole conversation every
    turn (whose top-k drifts as the conversation grows, churning every cached
    prompt prefix the glossary is rendered into), terms are DISCOVERED
    incrementally: each turn, only the customer messages that arrived since the
    agent's last reply are examined — lexically (a term's name or synonym is
    literally mentioned: guaranteed, rank-free recall) and by embedding relevance
    (top-k per message: paraphrases). Hits join a sticky, session-persisted
    working set, so the union over turns is equivalent to max-pooling over the
    entire history without ever re-scoring it.

    Precision self-corrects rather than being decided per query: a spurious hit
    costs one prompt line and is eventually LRU-evicted for never firing again
    (see :meth:`prune`), while an evicted term structurally cannot flap back —
    old messages are never re-queried, so readmission requires a fresh mention.

    Small glossaries opt out entirely: when the inventory fits under the cap,
    every term is always present (no discovery, no metadata), preserving the
    long-standing behavior.
    """

    # The working-set target size — also the small-glossary fast-path cap.
    _MAX_GLOSSARY_TERMS = 30
    # Pruning fires only above this, so evictions (cached-prefix changes) are
    # batched and rare.
    _HIGH_WATER_MARK = 40
    # Embedding hits per new customer message. Deliberately generous: in a sticky
    # working set, over-inclusion is cheap (LRU-evicted later) while a miss can
    # only be recovered by a future mention.
    _K_PER_MESSAGE = 10

    # Message sources counted as "the agent's reply" — the boundary the per-turn
    # tail starts after (same convention as the turn evaluators).
    _AGENT_MESSAGE_SOURCES = (
        EventSource.AI_AGENT,
        EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
        EventSource.HUMAN_AGENT,
    )

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        entity_queries: EntityQueries,
        entity_commands: EntityCommands,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._entity_queries = entity_queries
        self._entity_commands = entity_commands

    async def recall(self, context: EngineContext) -> None:
        """Load the turn's glossary terms into ``context.state.glossary_terms``,
        discovering new ones from the turn's new customer messages and persisting
        the working set to session metadata."""
        with self._tracer.span("match.glossary.recall"):
            inventory = await self._entity_queries.list_glossary_terms_for_context(context.agent.id)

            # Small glossary: everything is always present, no discovery needed.
            if len(inventory) <= self._MAX_GLOSSARY_TERMS:
                context.state.glossary_terms = set(inventory)
                self._emit_loaded_glossary(context)
                return

            terms_by_id = {str(term.id): term for term in inventory}

            stored = cast(
                Mapping[str, int] | None,
                context.session.metadata.get(_SESSION_TERMS_METADATA_KEY),
            )

            if stored is None:
                # Fresh or legacy session: seed once from the whole conversation
                # (today's loading semantics, exactly once), then go incremental.
                session_terms = await self._seed(context)
            else:
                # Drop terms deleted from the inventory since they were discovered.
                session_terms = {
                    term_id: int(offset)
                    for term_id, offset in stored.items()
                    if term_id in terms_by_id
                }

            new_messages = self._new_customer_messages(context)

            lexical = self._lexical_hits(new_messages, inventory)
            embedding = await self._embedding_hits(context, new_messages)

            for term, offset in [*lexical, *embedding]:
                term_id = str(term.id)
                if term_id not in terms_by_id:
                    continue
                # New terms join; existing members get their recency refreshed —
                # this feeds the LRU eviction in `prune`.
                session_terms[term_id] = max(offset, session_terms.get(term_id, -1))

            if stored is None or session_terms != dict(stored):
                await self._persist(context, session_terms)

            context.state.glossary_terms = {terms_by_id[term_id] for term_id in session_terms}
            self._emit_loaded_glossary(context)

    async def prune(self, context: EngineContext) -> None:
        """Cap the working set: above the high-water mark, keep only the most
        recently hit terms. Pure bookkeeping (no LLM); runs post-response so the
        cached-prefix change lands between turns. Evicted terms cannot flap back:
        only new messages are ever queried, so readmission requires a fresh
        mention."""
        with self._tracer.span("match.glossary.prune"):
            stored = cast(
                Mapping[str, int],
                context.session.metadata.get(_SESSION_TERMS_METADATA_KEY, {}),
            )

            if len(stored) <= self._HIGH_WATER_MARK:
                return

            survivors = dict(
                sorted(stored.items(), key=lambda entry: (entry[1], entry[0]))[
                    -self._MAX_GLOSSARY_TERMS :
                ]
            )

            await self._persist(context, survivors)

            self._logger.debug(
                f"{self.__class__.__name__} evicted {len(stored) - len(survivors)} of "
                f"{len(stored)} session glossary terms (target {self._MAX_GLOSSARY_TERMS})"
            )

    async def _seed(self, context: EngineContext) -> dict[str, int]:
        query_lines: list[str] = []

        if context.state.session_summary:
            query_lines.append(f"Session summary: {context.state.session_summary}")

        query_lines.extend(f"{m.source}: {m.content}" for m in context.interaction.messages)

        if not query_lines:
            # No conversation yet (the initialize-time warm-up). Seed against a
            # neutral greeting so the warmed prefix includes a glossary section.
            query_lines.append("User: Hello")

        seeded = await self._entity_queries.find_glossary_terms_for_context(
            context.agent.id,
            query=str(query_lines),
            max_terms=self._MAX_GLOSSARY_TERMS,
        )

        last_offset = max((e.offset for e in context.interaction.events), default=0)
        return {str(term.id): last_offset for term in seeded}

    def _new_customer_messages(self, context: EngineContext) -> list[tuple[str, int]]:
        """The per-turn tail: customer messages after the agent's last reply.
        Only these are ever queried — history participates solely through what it
        already deposited into the working set."""
        events = context.interaction.events

        cutoff = 0
        for index, event in enumerate(events):
            if event.kind == EventKind.MESSAGE and event.source in self._AGENT_MESSAGE_SOURCES:
                cutoff = index + 1

        return [
            (cast(MessageEventData, event.data)["message"], event.offset)
            for event in events[cutoff:]
            if event.kind == EventKind.MESSAGE and event.source == EventSource.CUSTOMER
        ]

    def _lexical_hits(
        self,
        messages: Sequence[tuple[str, int]],
        inventory: Sequence[Term],
    ) -> list[tuple[Term, int]]:
        """Terms whose name or a synonym is literally mentioned in a new message.
        Rank-free and guaranteed: an explicit mention always fires, no matter how
        the embedding ranking would have placed it."""
        if not messages:
            return []

        patterns = [
            (term, pattern)
            for term in inventory
            if (pattern := self._term_mention_pattern(term)) is not None
        ]

        hits: list[tuple[Term, int]] = []
        for text, offset in messages:
            for term, pattern in patterns:
                if pattern.search(text):
                    hits.append((term, offset))
        return hits

    def _term_mention_pattern(self, term: Term) -> re.Pattern[str] | None:
        # Whole-word, case-insensitive match of the name or any synonym;
        # multi-word surfaces match as whitespace-flexible phrases.
        alternatives = []
        for surface in (term.name, *term.synonyms):
            parts = [re.escape(part) for part in surface.split()]
            if parts:
                alternatives.append(r"\s+".join(parts))

        if not alternatives:
            return None

        return re.compile(
            r"\b(?:" + "|".join(alternatives) + r")\b",
            flags=re.IGNORECASE,
        )

    async def _embedding_hits(
        self,
        context: EngineContext,
        messages: Sequence[tuple[str, int]],
    ) -> list[tuple[Term, int]]:
        """Top-k nearest terms per new message — the paraphrase net. Pure rank
        (the same vector search the glossary always used), just with a short,
        undiluted query."""
        if not messages:
            return []

        results = await safe_gather(
            *(
                self._entity_queries.find_glossary_terms_for_context(
                    context.agent.id,
                    query=text,
                    max_terms=self._K_PER_MESSAGE,
                )
                for text, _ in messages
            )
        )

        return [(term, offset) for (_, offset), found in zip(messages, results) for term in found]

    async def _persist(
        self,
        context: EngineContext,
        session_terms: Mapping[str, int],
    ) -> None:
        metadata = dict(context.session.metadata)
        metadata[_SESSION_TERMS_METADATA_KEY] = cast(JSONSerializable, dict(session_terms))

        await self._entity_commands.update_session(context.session.id, {"metadata": metadata})
        context.session = replace(context.session, metadata=metadata)

    def _emit_loaded_glossary(self, context: EngineContext) -> None:
        CompassTracer(context.tracer).glossary_loaded(context.state.glossary_terms)
