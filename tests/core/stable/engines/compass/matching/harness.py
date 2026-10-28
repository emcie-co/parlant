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

"""The classical matching harness.

Builds a REAL ``Matcher`` from the test container — real stores, real
recaller, real scheduler, real relational resolver — and fakes ONLY the two
model boundaries, wired through the real ``ModelConfigurationService`` path:

- :class:`ScriptedRankGenerator` scripts the ranking LLM: it finds the
  ``RANK_<NAME>`` marker in the (real) prompt it receives and answers with a
  scripted relevance score for that rule.
- :class:`MarkerEmbedder` scripts the embedding model: ``RECALL_<NAME>``
  markers embed to orthogonal axes, so only an exact marker in the
  conversation can produce discovery.

Tests assert on observable outcomes — response state, session persistence —
or, where the contract is about the model boundary itself, on the fakes'
recorded calls. This is the house style for matcher tests per the classical
(non-mockist) testing instruction in AGENTS.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import re
from typing import Any, cast

from parlant.core.container import Container, Singleton

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.core.agents import AgentStore, Effort
from parlant.core.common import IdGenerator, JSONSerializable, generate_id
from parlant.core.customers import CustomerStore
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.engine_context import (
    EngineContext as CoreEngineContext,
    Interaction,
)
from parlant.core.engines.session_metadata_workspace import SessionMetadataWorkspace
from parlant.core.engines.compass.hooks import CompassHooks
from parlant.core.engines.rule_evaluator_registry import RuleEvaluatorRegistry
from parlant.core.engines.compass.matcher import (
    Matcher,
    _DISCOVERED_RULE_IDS_PATH,
    _RULES_EVER_MATCHED_IN_SESSION_PATH,
)
from parlant.core.engines.compass.matching.evaluation_scheduler import (
    EvaluationScheduler,
    EvaluationSchedulingPolicy,
)
from parlant.core.engines.compass.matching.rule_discovery import RuleDiscoverer
from parlant.core.engines.compass.matching.rule_ranker import RuleRankSchema
from parlant.core.engines.compass.matching.rule_recaller import RuleRecaller
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.engines.compass.context import CompassDependencies, EngineContext, ResponseState
from parlant.core.engines.types import Context
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.glossary import GlossaryStore, GlossaryVectorStore
from parlant.core.groups import GroupId
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.decision import DecisionMaker
from parlant.core.nlp.embedding import Embedder, EmbeddingCache, EmbeddingResult
from parlant.core.nlp.generation import SchematicGenerationResult, SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.model_routing import (
    ModelConfig,
    ModelConfigurationService,
    ModelRouter,
    ModelScope,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.loggers import Logger
from parlant.core import metadata
from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.rules import Rule
from parlant.core.sessions import Event, EventSource, SessionStore, SessionUpdateParams
from parlant.core.store_provider import BasicStoreProvider, StoreProvider
from parlant.core.tracer import Tracer

from tests.core.common.utils import create_event_message
from tests.core.stable.engines.compass.matching.utils import create_rule


RANK_MARKER = re.compile(r"\bRANK_([A-Z0-9_]+)\b")
RECALL_MARKER = re.compile(r"\bRECALL_([A-Z0-9_]+)\b")


class ScriptedRankGenerator(SchematicGenerator[RuleRankSchema]):
    """Scripts the ranking model boundary.

    Serves every ``matching.rank`` route consumer (the ranker AND the pruner —
    both send one prompt per rule containing that rule's ``RANK_<NAME>``
    marker). ``relevant_rules`` scores 5-vs-1; ``scores_by_name`` scripts an
    exact 1-5 score per rule where a test needs graded verdicts (pruning).
    """

    def __init__(
        self,
        relevant_rules: set[str],
        *,
        scores_by_name: Mapping[str, int] | None = None,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._relevant_rules = relevant_rules
        self._scores_by_name = dict(scores_by_name or {})
        self._started = started
        self._release = release
        self.calls: list[str] = []
        self.prompt_texts: list[str] = []

    @property
    def schema(self) -> type[RuleRankSchema]:
        return RuleRankSchema

    async def generate(
        self,
        prompt: Any,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[RuleRankSchema]:
        del hints
        prompt_text = prompt.build() if hasattr(prompt, "build") else str(prompt)
        matches = RANK_MARKER.findall(prompt_text)
        assert matches, "The real prompt did not contain a test rule marker"
        rule_name = matches[-1]
        self.calls.append(rule_name)
        self.prompt_texts.append(prompt_text)

        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()

        score = self._scores_by_name.get(
            rule_name,
            5 if rule_name in self._relevant_rules else 1,
        )
        return SchematicGenerationResult(
            content=RuleRankSchema(
                tldr=f"{rule_name} scripted at the model boundary",
                s=score,
            ),
            info=GenerationInfo(
                schema_name=RuleRankSchema.__name__,
                model=self.id,
                duration=0.001,
                usage=UsageInfo(input_tokens=1, output_tokens=1),
            ),
        )

    @property
    def id(self) -> str:
        return "test/scripted-ranker"

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()


class MarkerEmbedder(Embedder):
    """Deterministic external embedding boundary used by the real recaller.

    Each RECALL_* marker is orthogonal. Rules use pinned signals, and the
    recaller is configured with a disabled classifier margin, so only an exact
    marker in the conversation can produce discovery. Texts with no marker all
    share one axis — mutually similar, which keeps glossary lookups trivial.
    """

    def __init__(
        self,
        rule_names: Sequence[str],
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._indices = {name: index for index, name in enumerate(rule_names)}
        self._id = f"test/marker-embedder/{generate_id()}"
        self._started = started
        self._release = release
        self.calls: list[list[str]] = []

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        del hints
        self.calls.append(texts)
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()

        return EmbeddingResult(vectors=[self._vector(text) for text in texts])

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        markers = RECALL_MARKER.findall(text)
        if not markers:
            vector[-1] = 1.0
            return vector

        for marker in markers:
            if marker in self._indices:
                vector[self._indices[marker]] = 1.0
        return vector

    @property
    def id(self) -> str:
        return self._id

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()

    @property
    def dimensions(self) -> int:
        return len(self._indices) + 1


class RecordingEntityQueries:
    """Observation-only proxy over the REAL ``EntityQueries``.

    Delegates everything; records glossary lookups so tests of the per-rule
    term memo can observe the lookup seam (the embedding cache makes the
    embedder boundary blind to repeated identical queries, so the contract
    is only observable here). No behavior is replaced.
    """

    def __init__(self, inner: EntityQueries) -> None:
        self._inner = inner
        self.find_glossary_calls: list[tuple[str, int]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def find_glossary_terms_for_context(
        self,
        agent_id: Any,
        query: str,
        max_terms: int = 20,
    ) -> Any:
        self.find_glossary_calls.append((query, max_terms))
        return await self._inner.find_glossary_terms_for_context(
            agent_id, query=query, max_terms=max_terms
        )


@dataclass
class MatchingHarness:
    matcher: Matcher
    context: EngineContext
    rank_generator: ScriptedRankGenerator
    embedder: MarkerEmbedder
    glossary_embedder: MarkerEmbedder
    glossary_store: GlossaryStore
    evaluator_registry: RuleEvaluatorRegistry
    relationship_store: RelationshipStore
    session_store: SessionStore
    hooks: CompassHooks
    decision_lookups: list[int] = field(default_factory=list)
    entity_queries: RecordingEntityQueries = field(init=False)

    def __post_init__(self) -> None:
        # Wrap the matcher's real EntityQueries with the recording proxy so
        # glossary-lookup seam tests have an observation point.
        self.entity_queries = RecordingEntityQueries(self.matcher._entity_queries)
        self.matcher._entity_queries = cast(EntityQueries, self.entity_queries)

    async def depend(
        self,
        source: Rule,
        target: Rule,
        *,
        kind: RelationshipKind = RelationshipKind.DEPENDENCY,
        group_id: str | None = None,
    ) -> None:
        await self.relationship_store.create_relationship(
            source=RelationshipEntity(source.id, RelationshipEntityKind.RULE),
            target=RelationshipEntity(target.id, RelationshipEntityKind.RULE),
            kind=kind,
            group_id=group_id,
        )

    async def depend_on_group(
        self,
        source: Rule,
        target: GroupId,
        *,
        target_kind: RelationshipEntityKind,
    ) -> None:
        await self.relationship_store.create_relationship(
            source=RelationshipEntity(source.id, RelationshipEntityKind.RULE),
            target=RelationshipEntity(target, target_kind),
            kind=RelationshipKind.DEPENDENCY,
        )

    async def match(self) -> None:
        await self.matcher._match(self.context)

    async def staged_discovered_rule_ids(self) -> set[str]:
        return set(
            cast(
                Sequence[str],
                metadata.internal_value(
                    self.context.session.metadata, _DISCOVERED_RULE_IDS_PATH, []
                ),
            )
        )

    async def staged_rules_ever_matched(self) -> set[str]:
        return set(
            cast(
                Sequence[str],
                metadata.internal_value(
                    self.context.session.metadata, _RULES_EVER_MATCHED_IN_SESSION_PATH, []
                ),
            )
        )

    async def staged_metadata(self) -> Mapping[str, JSONSerializable]:
        return self.context.session.metadata

    def rule_query_embeds(self, marker: str) -> list[str]:
        """Glossary-boundary queries that carried a given rule marker."""
        return [text for batch in self.glossary_embedder.calls for text in batch if marker in text]


def make_rule(name: str, *, extra_condition: str = "") -> Rule:
    condition = f"RANK_{name}"
    if extra_condition:
        condition = f"{condition} {extra_condition}"
    return replace(
        create_rule(condition=condition, action=f"apply {name}"),
        signals=[f"[__pin__] RECALL_{name}"],
    )


class _PrefillThresholdPolicy(CompassOptimizationPolicy):
    """The stock policy with a test-chosen prefill threshold."""

    def __init__(self, minimum: int) -> None:
        self._minimum = minimum

    def get_min_prefill_fan_out(self, hints: Mapping[str, Any] = {}) -> int:
        return self._minimum


async def matching_harness(
    container: Container,
    rules: Sequence[Rule],
    *,
    rank_relevant: set[str],
    scores_by_name: Mapping[str, int] | None = None,
    conversation: str = "No discovery marker is present.",
    conversation_events: Sequence[Event] | None = None,
    discovered_rules: set[Rule] | None = None,
    rules_ever_matched: set[Rule] | None = None,
    session_metadata: Mapping[object, JSONSerializable] | None = None,
    effort: Effort = Effort.HIGH,
    # Prefill's own threshold, kept at 1 so routing tests observe which
    # evaluators would be warmed without the economics getting in the way.
    min_prefill_fan_out: int = 1,
    policy: EvaluationSchedulingPolicy = EvaluationSchedulingPolicy(),
    matcher_type: type[Matcher] = Matcher,
    decision_maker: DecisionMaker | None = None,
    rank_started: asyncio.Event | None = None,
    rank_release: asyncio.Event | None = None,
    embedding_started: asyncio.Event | None = None,
    embedding_release: asyncio.Event | None = None,
) -> MatchingHarness:
    base_container = container
    container = container.clone()
    rule_names = [RANK_MARKER.search(rule.condition or "").group(1) for rule in rules]  # type: ignore[union-attr]
    rank_generator = ScriptedRankGenerator(
        rank_relevant,
        scores_by_name=scores_by_name,
        started=rank_started,
        release=rank_release,
    )
    embedder = MarkerEmbedder(
        rule_names,
        started=embedding_started,
        release=embedding_release,
    )
    glossary_embedder = MarkerEmbedder(rule_names)

    async def provide_ranker(
        _: CoreEngineContext[Any, Any] | None,
    ) -> SchematicGenerator[RuleRankSchema]:
        return rank_generator

    async def provide_embedder(_: CoreEngineContext[Any, Any] | None) -> Embedder:
        return embedder

    async def provide_glossary_embedder(_: CoreEngineContext[Any, Any] | None) -> Embedder:
        return glossary_embedder

    # Counts every resolution of the decision route, so a test can tell a
    # per-batch lookup from a per-rule one.
    decision_lookups: list[int] = []

    async def provide_decision_maker(
        _: CoreEngineContext[Any, Any] | None,
    ) -> DecisionMaker:
        assert decision_maker is not None
        decision_lookups.append(1)
        return decision_maker

    routes: dict[str, ModelConfig] = {
        "matching.rank": ModelConfig(provide_ranker),
        "embedding.rules": ModelConfig(provide_embedder),
        "embedding.glossary": ModelConfig(provide_glossary_embedder),
    }

    if decision_maker is not None:
        # A decision model has to be named explicitly — the bare route above
        # serves schematic ranking, and decisions never ride a wildcard.
        routes["matching.rank[capability=decision]"] = ModelConfig(provide_decision_maker)

    await base_container[ModelConfigurationService].configure(
        ModelScope.server(),
        routes,
    )

    async def provide_test_glossary_embedder() -> Embedder:
        return glossary_embedder

    glossary_store = await GlossaryVectorStore(
        id_generator=container[IdGenerator],
        vector_db=TransientVectorDatabase(
            container[Logger],
            container[Tracer],
            lambda: container[EmbeddingCache],
        ),
        document_db=TransientDocumentDatabase(),
        embedder_provider=provide_test_glossary_embedder,
    ).__aenter__()
    container[GlossaryStore] = glossary_store
    container[StoreProvider] = BasicStoreProvider(lambda: container)
    container[EntityQueries] = Singleton(EntityQueries)
    container[EntityCommands] = Singleton(EntityCommands)

    evaluator_registry = RuleEvaluatorRegistry()
    container[RuleEvaluatorRegistry] = evaluator_registry
    container[EvaluationSchedulingPolicy] = policy
    container[EvaluationScheduler] = Singleton(EvaluationScheduler)

    # Fresh hooks per harness: the base container's hooks object is shared
    # across tests, and handler registration is an attachment signal.
    hooks = CompassHooks()
    container[CompassHooks] = hooks

    recaller = RuleRecaller(
        model_router=container[ModelRouter],
        tracer=container[Tracer],
        embedding_cache=container[EmbeddingCache],
        recall_margin=-100.0,
    )
    container[RuleRecaller] = recaller
    container[RuleDiscoverer] = recaller

    agent = await container[AgentStore].create_agent(
        name="Matching harness agent",
        effort=effort,
    )
    customer = await container[CustomerStore].create_customer(name="Test customer")
    session_store = container[SessionStore]
    session = await session_store.create_session(
        customer_id=customer.id,
        agent_id=agent.id,
    )
    if session_metadata is not None:
        normalized_metadata: dict[str, JSONSerializable] = {}
        for key, value in session_metadata.items():
            if isinstance(key, tuple):
                normalized_metadata = metadata.set_internal_value(normalized_metadata, key, value)
            else:
                normalized_metadata[str(key)] = value
        session = await session_store.update_session(
            session.id,
            cast(SessionUpdateParams, {"metadata": normalized_metadata}),
        )
    events = (
        list(conversation_events)
        if conversation_events is not None
        else [create_event_message(offset=0, source=EventSource.CUSTOMER, message=conversation)]
    )
    workspace = SessionMetadataWorkspace(session, session_store)
    session = workspace.session
    context = EngineContext(
        info=Context(session_id=session.id, agent_id=agent.id),
        logger=container[Logger],
        tracer=container[Tracer],
        agent=agent,
        customer=customer,
        session=session,
        session_event_emitter=EventBuffer(emitting_agent=agent),
        response_event_emitter=EventBuffer(emitting_agent=agent),
        interaction=Interaction(events=events),
        state=ResponseState(
            agent_effort=effort,
            usable_rules=list(rules),
            discovered_rules=discovered_rules or set(),
            # The session's matched-rule ledger; production loads it from
            # session metadata in preload, which harness.match() bypasses.
            rules_ever_matched_in_session={rule.id for rule in (rules_ever_matched or set())},
        ),
        dependencies=replace(
            container[CompassDependencies],
            optimization_policy=_PrefillThresholdPolicy(min_prefill_fan_out),
        ),
        session_metadata=workspace,
    )
    matcher = container[matcher_type]

    return MatchingHarness(
        matcher=matcher,
        hooks=hooks,
        context=context,
        rank_generator=rank_generator,
        embedder=embedder,
        glossary_embedder=glossary_embedder,
        glossary_store=glossary_store,
        evaluator_registry=evaluator_registry,
        relationship_store=container[RelationshipStore],
        session_store=session_store,
        decision_lookups=decision_lookups,
    )
