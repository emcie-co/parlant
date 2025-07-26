# Copyright 2025 Emcie Co Ltd.
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

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
import enum
from hashlib import md5
import importlib.util
from pathlib import Path
from rich.console import Group
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
    TaskID,
    TextColumn,
)
from rich.live import Live
from types import TracebackType
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    TypeAlias,
    TypedDict,
    cast,
)
from typing_extensions import overload
from lagom import Container


from parlant.adapters.db.json_file import JSONFileDocumentCollection, JSONFileDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.nlp.openai_service import OpenAIService
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.core import async_utils
from parlant.core.agents import (
    Agent as _Agent,
    AgentId,
    AgentStore,
    AgentUpdateParams,
    CompositionMode as _CompositionMode,
)
from parlant.core.application import Application
from parlant.core.async_utils import Timeout, default_done_callback
from parlant.core.capabilities import CapabilityId, CapabilityStore, CapabilityVectorStore
from parlant.core.common import IdGenerator, ItemNotFoundError, JSONSerializable, UniqueId, Version
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableDocumentStore,
    ContextVariableId,
    ContextVariableStore,
)
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.customers import (
    Customer as _Customer,
    CustomerDocumentStore,
    CustomerId,
    CustomerStore,
)
from parlant.core.emissions import EmittedEvent, EventEmitterFactory
from parlant.core.engines.alpha.hooks import EngineHook, EngineHookResult, EngineHooks
from parlant.core.engines.alpha.loaded_context import LoadedContext, Interaction, InteractionMessage
from parlant.core.glossary import GlossaryStore, GlossaryVectorStore, TermId
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCache,
    EmbeddingResult,
)
from parlant.core.nlp.generation import (
    FallbackSchematicGenerator,
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import DocumentDatabase, identity_loader_for
from parlant.core.relationships import (
    RelationshipKind,
    RelationshipDocumentStore,
    RelationshipEntity,
    RelationshipEntityId,
    RelationshipEntityKind,
    RelationshipId,
    RelationshipStore,
)
from parlant.core.services.indexing.behavioral_change_evaluation import BehavioralChangeEvaluator
from parlant.core.services.tools.service_registry import ServiceDocumentRegistry, ServiceRegistry
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    Session,
    SessionId,
    SessionDocumentStore,
    SessionStore,
    StatusEventData,
    ToolCall as _SessionToolCall,
    ToolEventData,
    ToolResult as _SessionToolResult,
)
from parlant.core.canned_responses import (
    CannedResponse,
    CannedResponseVectorStore,
    CannedResponseId,
    CannedResponseStore,
)
from parlant.core.evaluations import (
    EvaluationDocumentStore,
    EvaluationStatus,
    EvaluationStore,
    GuidelinePayload,
    InvoiceGuidelineData,
    InvoiceJourneyData,
    JourneyPayload,
    PayloadOperation,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.guidelines import (
    GuidelineContent,
    GuidelineDocumentStore,
    GuidelineId,
    GuidelineStore,
)
from parlant.core.journeys import (
    JourneyEdgeId,
    JourneyId,
    JourneyNodeId,
    JourneyStore,
    JourneyVectorStore,
)
from parlant.core.loggers import LogLevel, Logger
from parlant.core.nlp.service import NLPService
from parlant.bin.server import PARLANT_HOME_DIR, start_parlant, StartupParameters
from parlant.core.services.tools.plugins import PluginServer, ToolEntry, tool
from parlant.core.tags import Tag as _Tag, TagDocumentStore, TagId, TagStore
from parlant.core.tools import (
    ControlOptions,
    SessionMode,
    SessionStatus,
    Tool,
    ToolContext,
    ToolId,
    ToolParameterDescriptor,
    ToolParameterOptions,
    ToolParameterType,
    ToolResult,
)
from parlant.core.version import VERSION

INTEGRATED_TOOL_SERVICE_NAME = "built-in"

T = TypeVar("T")


JourneyStateId: TypeAlias = JourneyNodeId
JourneyTransitionId: TypeAlias = JourneyEdgeId


class SDKError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _load_openai(container: Container) -> NLPService:
    return OpenAIService(container[Logger])


class _CachedGuidelineEvaluation(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    properties: dict[str, JSONSerializable]


class _CachedJourneyEvaluation(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    node_properties: dict[JourneyStateId, dict[str, JSONSerializable]]
    edge_properties: dict[JourneyTransitionId, dict[str, JSONSerializable]]


class _CachedEvaluator:
    @dataclass(frozen=True)
    class JourneyEvaluation:
        node_properties: dict[JourneyStateId, dict[str, JSONSerializable]]
        edge_properties: dict[JourneyTransitionId, dict[str, JSONSerializable]]

    @dataclass(frozen=True)
    class GuidelineEvaluation:
        properties: dict[str, JSONSerializable]

    def __init__(
        self,
        db: JSONFileDocumentDatabase,
        container: Container,
    ) -> None:
        self._db: JSONFileDocumentDatabase = db
        self._guideline_collection: JSONFileDocumentCollection[_CachedGuidelineEvaluation]
        self._journey_collection: JSONFileDocumentCollection[_CachedJourneyEvaluation]

        self._container = container
        self._logger = container[Logger]
        self._exit_stack = AsyncExitStack()
        self._progress: dict[str, float] = {}

    def _set_progress(self, key: str, pct: float) -> None:
        self._progress[key] = max(0.0, min(pct, 100.0))

    def _progress_for(self, key: str) -> float:
        return self._progress.get(key, 0.0)

    async def __aenter__(self) -> _CachedEvaluator:
        await self._exit_stack.enter_async_context(self._db)

        self._guideline_collection = await self._db.get_or_create_collection(
            name="guideline_evaluations",
            schema=_CachedGuidelineEvaluation,
            document_loader=identity_loader_for(_CachedGuidelineEvaluation),
        )

        self._journey_collection = await self._db.get_or_create_collection(
            name="journey_evaluations",
            schema=_CachedJourneyEvaluation,
            document_loader=identity_loader_for(_CachedJourneyEvaluation),
        )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        await self._exit_stack.aclose()
        return False

    def _hash_guideline_evaluation_request(
        self,
        g: GuidelineContent,
        tool_ids: Sequence[ToolId],
        journey_state_propositions: bool,
    ) -> str:
        """Generate a hash for the guideline evaluation request."""
        tool_ids_str = ",".join(str(tool_id) for tool_id in tool_ids) if tool_ids else ""

        return md5(
            f"{g.condition or ''}:{g.action or ''}:{tool_ids_str}:{journey_state_propositions}".encode()
        ).hexdigest()

    def _hash_journey_evaluation_request(
        self,
        journey: Journey,
    ) -> str:
        """Generate a hash for the journey evaluation request."""
        node_ids_str = ",".join(str(node.id) for node in journey.states) if journey.states else ""
        edge_ids_str = (
            ",".join(str(edge.id) for edge in journey.transitions) if journey.transitions else ""
        )

        return md5(f"{journey.id}:{node_ids_str}:{edge_ids_str}".encode()).hexdigest()

    async def evaluate_state(
        self,
        entity_id: JourneyStateId,
        g: GuidelineContent,
        tool_ids: Sequence[ToolId] = [],
    ) -> _CachedEvaluator.GuidelineEvaluation:
        return await self._evaluate_guideline(
            entity_id=entity_id,
            g=g,
            tool_ids=tool_ids,
            journey_state_proposition=True,
        )

    async def evaluate_guideline(
        self,
        entity_id: GuidelineId,
        g: GuidelineContent,
        tool_ids: Sequence[ToolId] = [],
    ) -> _CachedEvaluator.GuidelineEvaluation:
        return await self._evaluate_guideline(
            entity_id=entity_id,
            g=g,
            tool_ids=tool_ids,
        )

    async def _evaluate_guideline(
        self,
        entity_id: GuidelineId | JourneyStateId,
        g: GuidelineContent,
        tool_ids: Sequence[ToolId] = [],
        action_proposition: bool = True,
        journey_state_proposition: bool = False,
    ) -> _CachedEvaluator.GuidelineEvaluation:
        # First check if we have a cached evaluation for this guideline
        _hash = self._hash_guideline_evaluation_request(
            g=g,
            tool_ids=tool_ids,
            journey_state_propositions=journey_state_proposition,
        )

        if cached_evaluation := await self._guideline_collection.find_one({"id": {"$eq": _hash}}):
            # Check if the cached evaluation is based on our current runtime version.
            # This is important as the required evaluation data can change between versions.
            if cached_evaluation["version"] == VERSION:
                self._logger.trace(
                    f"Using cached evaluation for guideline: Condition: {g.condition or 'None'}; Action: {g.action or 'None'}"
                )

                return self.GuidelineEvaluation(
                    properties=cached_evaluation["properties"],
                )
            else:
                self._logger.trace(
                    f"Deleting outdated cached evaluation for guideline: {g.condition or 'None'}"
                )

                await self._guideline_collection.delete_one(
                    {"id": {"$eq": cached_evaluation["id"]}}
                )

        self._logger.trace(
            f"Evaluating guideline: Condition: {g.condition or 'None'}, Action: {g.action or 'None'}"
        )

        evaluation_id = await self._container[BehavioralChangeEvaluator].create_evaluation_task(
            payload_descriptors=[
                PayloadDescriptor(
                    PayloadKind.GUIDELINE,
                    GuidelinePayload(
                        content=GuidelineContent(
                            condition=g.condition,
                            action=g.action,
                        ),
                        tool_ids=tool_ids,
                        operation=PayloadOperation.ADD,
                        coherence_check=False,  # Legacy and will be removed in the future
                        connection_proposition=False,  # Legacy and will be removed in the future
                        action_proposition=action_proposition,
                        properties_proposition=True,
                        journey_node_proposition=journey_state_proposition,
                    ),
                )
            ],
        )

        while True:
            evaluation = await self._container[EvaluationStore].read_evaluation(
                evaluation_id=evaluation_id,
            )

            self._set_progress(entity_id, evaluation.progress)

            if evaluation.status in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
                await asyncio.sleep(0.5)
                continue
            elif evaluation.status == EvaluationStatus.FAILED:
                raise SDKError(f"Evaluation failed: {evaluation.error}")
            elif evaluation.status == EvaluationStatus.COMPLETED:
                if not evaluation.invoices:
                    raise SDKError("Evaluation completed with no invoices.")
                if not evaluation.invoices[0].approved:
                    raise SDKError("Evaluation completed with unapproved invoice.")

                invoice = evaluation.invoices[0]

                if not invoice.data:
                    raise SDKError(
                        "Evaluation completed with no properties_proposition in the invoice."
                    )

            assert invoice.data

            # Cache the evaluation result
            await self._guideline_collection.insert_one(
                {
                    "id": ObjectId(_hash),
                    "version": Version.String(VERSION),
                    "properties": cast(InvoiceGuidelineData, invoice.data).properties_proposition
                    or {},
                }
            )

            # Return the evaluation result
            return self.GuidelineEvaluation(
                properties=cast(InvoiceGuidelineData, invoice.data).properties_proposition or {},
            )

    async def evaluate_journey(
        self,
        journey: Journey,
    ) -> _CachedEvaluator.JourneyEvaluation:
        # First check if we have a cached evaluation for this journey
        _hash = self._hash_journey_evaluation_request(
            journey=journey,
        )

        if cached_evaluation := await self._journey_collection.find_one({"id": {"$eq": _hash}}):
            # Check if the cached evaluation is based on our current runtime version.
            # This is important as the required evaluation data can change between versions.
            if cached_evaluation["version"] == VERSION:
                self._logger.trace(
                    f"Using cached evaluation for journey: Title: {journey.title or 'None'};"
                )

                return self.JourneyEvaluation(
                    node_properties=cached_evaluation["node_properties"],
                    edge_properties=cached_evaluation["edge_properties"],
                )
            else:
                self._logger.info(
                    f"Deleting outdated cached evaluation for journey: {journey.title or 'None'}"
                )

                await self._guideline_collection.delete_one(
                    {"id": {"$eq": cached_evaluation["id"]}}
                )

        self._logger.trace(f"Evaluating journey: Title: {journey.title or 'None'}")

        evaluation_id = await self._container[BehavioralChangeEvaluator].create_evaluation_task(
            payload_descriptors=[
                PayloadDescriptor(
                    PayloadKind.JOURNEY,
                    JourneyPayload(
                        journey_id=journey.id,
                        operation=PayloadOperation.ADD,
                    ),
                )
            ],
        )

        while True:
            evaluation = await self._container[EvaluationStore].read_evaluation(
                evaluation_id=evaluation_id,
            )

            self._set_progress(journey.id, evaluation.progress)

            if evaluation.status in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
                await asyncio.sleep(0.5)
                continue
            elif evaluation.status == EvaluationStatus.FAILED:
                raise SDKError(f"Journey Evaluation failed: {evaluation.error}")
            elif evaluation.status == EvaluationStatus.COMPLETED:
                if not evaluation.invoices:
                    raise SDKError("Journey Evaluation completed with no invoices.")
                if not evaluation.invoices[0].approved:
                    raise SDKError("Journey Evaluation completed with unapproved invoice.")

                invoice = evaluation.invoices[0]

                if not invoice.data:
                    raise SDKError("Journey Evaluation completed with no data in the invoice.")

            assert invoice.data

            # Cache the evaluation result
            await self._journey_collection.insert_one(
                {
                    "id": ObjectId(_hash),
                    "version": Version.String(VERSION),
                    "node_properties": cast(
                        InvoiceJourneyData, invoice.data
                    ).node_properties_proposition,
                    "edge_properties": cast(
                        InvoiceJourneyData, invoice.data
                    ).edge_properties_proposition
                    or {},
                }
            )

            # Return the evaluation result
            return self.JourneyEvaluation(
                node_properties=cast(InvoiceJourneyData, invoice.data).node_properties_proposition
                or {},
                edge_properties=cast(InvoiceJourneyData, invoice.data).edge_properties_proposition
                or {},
            )


class _SdkAgentStore(AgentStore):
    """This is a minimal in-memory implementation of AgentStore for SDK purposes.
    The reason we use this and not any of the other implementations is that it
    uses the agent's name as the ID, which is convenient for SDK usage.

    This is because an SDK file would be re-run multiple times within the same testing session,
    and Parlant's integrated web UI would likely stay running in the background between runs.

    Now, if the agent's ID changed between runs, the UI would not be able to find the agent
    and would essentially lose context in the sessions it displays.

    Incidentally, this is also why we support using a non-transient session store in the SDK."""

    def __init__(self) -> None:
        self._agents: dict[AgentId, _Agent] = {}

    async def create_agent(
        self,
        name: str,
        description: str | None = None,
        creation_utc: datetime | None = None,
        max_engine_iterations: int | None = None,
        composition_mode: _CompositionMode | None = None,
        tags: Sequence[TagId] | None = None,
    ) -> _Agent:
        agent = _Agent(
            id=AgentId(name),
            name=name,
            description=description,
            creation_utc=creation_utc or datetime.now(timezone.utc),
            max_engine_iterations=max_engine_iterations or 1,
            tags=tags or [],
            composition_mode=composition_mode or _CompositionMode.CANNED_FLUID,
        )

        self._agents[agent.id] = agent

        return agent

    async def list_agents(self) -> Sequence[_Agent]:
        return list(self._agents.values())

    async def read_agent(self, agent_id: AgentId) -> _Agent:
        if agent_id not in self._agents:
            raise ItemNotFoundError(UniqueId(agent_id), "Agent not found")
        return self._agents[agent_id]

    async def update_agent(self, agent_id: AgentId, params: AgentUpdateParams) -> _Agent:
        raise NotImplementedError

    async def delete_agent(self, agent_id: AgentId) -> None:
        raise NotImplementedError

    async def upsert_tag(
        self,
        agent_id: AgentId,
        tag_id: TagId,
        creation_utc: datetime | None = None,
    ) -> bool:
        raise NotImplementedError

    async def remove_tag(self, agent_id: AgentId, tag_id: TagId) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Tag:
    @staticmethod
    def preamble() -> TagId:
        return _Tag.preamble()

    id: TagId
    name: str


@dataclass(frozen=True)
class Relationship:
    id: RelationshipId
    kind: RelationshipKind
    source: RelationshipEntityId
    target: RelationshipEntityId


@dataclass(frozen=True)
class Guideline:
    id: GuidelineId
    condition: str
    action: str | None
    tags: Sequence[TagId]
    metadata: Mapping[str, JSONSerializable]

    _server: Server
    _container: Container

    async def prioritize_over(self, guideline: Guideline) -> Relationship:
        return await self._create_relationship(
            guideline=guideline,
            kind=RelationshipKind.PRIORITY,
            direction="source",
        )

    async def entail(self, guideline: Guideline) -> Relationship:
        return await self._create_relationship(
            guideline=guideline,
            kind=RelationshipKind.ENTAILMENT,
            direction="source",
        )

    async def depend_on(self, guideline: Guideline) -> Relationship:
        return await self._create_relationship(
            guideline=guideline,
            kind=RelationshipKind.DEPENDENCY,
            direction="source",
        )

    async def disambiguate(self, targets: Sequence[Guideline]) -> Sequence[Relationship]:
        if len(targets) < 2:
            raise SDKError(
                f"At least two targets are required for disambiguation (got {len(targets)})."
            )

        return [
            await self._create_relationship(
                guideline=t,
                kind=RelationshipKind.DISAMBIGUATION,
                direction="source",
            )
            for t in targets
        ]

    async def reevaluate_after(self, tool: ToolEntry) -> Relationship:
        relationship = await self._container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name),
                kind=RelationshipEntityKind.TOOL,
            ),
            target=RelationshipEntity(
                id=self.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            kind=RelationshipKind.REEVALUATION,
        )

        return Relationship(
            id=relationship.id,
            kind=relationship.kind,
            source=relationship.source.id,
            target=relationship.target.id,
        )

    async def _create_relationship(
        self,
        guideline: Guideline,
        kind: RelationshipKind,
        direction: Literal["source", "target"],
    ) -> Relationship:
        if direction == "source":
            source = RelationshipEntity(id=self.id, kind=RelationshipEntityKind.GUIDELINE)
            target = RelationshipEntity(id=guideline.id, kind=RelationshipEntityKind.GUIDELINE)
        else:
            source = RelationshipEntity(id=guideline.id, kind=RelationshipEntityKind.GUIDELINE)
            target = RelationshipEntity(id=self.id, kind=RelationshipEntityKind.GUIDELINE)

        relationship = await self._container[RelationshipStore].create_relationship(
            source=source,
            target=target,
            kind=kind,
        )

        return Relationship(
            id=relationship.id,
            kind=relationship.kind,
            source=relationship.source.id,
            target=relationship.target.id,
        )


TState = TypeVar("TState", bound="JourneyState")


@dataclass(frozen=True)
class JourneyTransition(Generic[TState]):
    id: JourneyTransitionId
    condition: str | None
    source: JourneyState
    target: TState
    metadata: Mapping[str, JSONSerializable]


@dataclass(frozen=True)
class JourneyState:
    id: JourneyStateId
    action: str | None
    tools: Sequence[ToolEntry]
    metadata: Mapping[str, JSONSerializable]

    _journey: Journey | None

    @property
    def internal_action(self) -> str | None:
        return self.action or cast(str | None, self.metadata.get("internal_action"))

    async def _fork(self) -> JourneyTransition[ForkJourneyState]:
        return cast(
            JourneyTransition[ForkJourneyState],
            await self._transition(
                condition=None,
                state=None,
                action=None,
                tools=[],
                fork=True,
            ),
        )

    async def _transition(
        self,
        *,
        condition: str | None = None,
        state: TState | None = None,
        action: str | None = None,
        tools: Sequence[ToolEntry] = [],
        fork: bool = False,
    ) -> JourneyTransition[JourneyState]:
        if not self._journey:
            raise SDKError("EndState cannot be connected to any other states.")

        actual_state: JourneyState | None = None

        if state is not None:
            actual_state = state
        elif tools:
            actual_state = await self._journey._create_state(
                ToolJourneyState,
                action=action,
                tools=tools,
            )

            [
                await self._journey._container[RelationshipStore].create_relationship(
                    source=RelationshipEntity(
                        id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name),
                        kind=RelationshipEntityKind.TOOL,
                    ),
                    target=RelationshipEntity(
                        id=_Tag.for_journey_node_id(actual_state.id),
                        kind=RelationshipEntityKind.TAG,
                    ),
                    kind=RelationshipKind.REEVALUATION,
                )
                for t in tools
            ]

        elif action:
            actual_state = await self._journey._create_state(
                ChatJourneyState,
                action=action,
                tools=[],
            )
        elif fork:
            actual_state = await self._journey._create_state(
                ForkJourneyState,
            )

        transitions = [t for t in self._journey.transitions if t.source == self]

        if len(transitions) > 0 and (not condition or any(not e.condition for e in transitions)):
            raise SDKError(
                "Cannot connect a new state without a condition if there are already connected states without conditions."
            )

        transition = await self._journey.create_transition(
            condition=condition, source=self, target=actual_state or END_JOURNEY
        )

        if actual_state:
            cast(list[JourneyState], self._journey.states).append(actual_state)

        cast(list[JourneyTransition[JourneyState]], self._journey.transitions).append(transition)

        return transition


END_JOURNEY = JourneyState(
    id=JourneyStore.END_NODE_ID,
    action=None,
    tools=[],
    metadata={},
    _journey=None,
)


class InitialJourneyState(JourneyState):
    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        state: TState,
    ) -> JourneyTransition[TState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str,
    ) -> JourneyTransition[ChatJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: ToolEntry,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: Sequence[ToolEntry],
    ) -> JourneyTransition[ToolJourneyState]: ...

    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str | None = None,
        tool_instruction: str | None = None,
        state: TState | None = None,
        tool_state: ToolEntry | Sequence[ToolEntry] = [],
    ) -> JourneyTransition[Any]:
        return await self._transition(
            condition=condition,
            state=state,
            action=chat_state or tool_instruction,
            tools=[tool_state] if isinstance(tool_state, ToolEntry) else tool_state,
        )


class ToolJourneyState(JourneyState):
    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        state: TState,
    ) -> JourneyTransition[TState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str,
    ) -> JourneyTransition[ChatJourneyState]: ...

    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str | None = None,
        state: TState | None = None,
    ) -> JourneyTransition[Any]:
        return await self._transition(
            condition=condition,
            state=state,
            action=chat_state,
        )

    async def fork(self) -> JourneyTransition[ForkJourneyState]:
        return await super()._fork()


class ChatJourneyState(JourneyState):
    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        state: TState,
    ) -> JourneyTransition[TState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str,
    ) -> JourneyTransition[ChatJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: ToolEntry,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: Sequence[ToolEntry],
    ) -> JourneyTransition[ToolJourneyState]: ...

    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str | None = None,
        tool_instruction: str | None = None,
        state: TState | None = None,
        tool_state: ToolEntry | Sequence[ToolEntry] = [],
    ) -> JourneyTransition[Any]:
        return await self._transition(
            condition=condition,
            state=state,
            action=chat_state or tool_instruction,
            tools=[tool_state] if isinstance(tool_state, ToolEntry) else tool_state,
        )

    async def fork(self) -> JourneyTransition[ForkJourneyState]:
        return await super()._fork()


class ForkJourneyState(JourneyState):
    @overload
    async def transition_to(
        self,
        *,
        condition: str,
        state: TState,
    ) -> JourneyTransition[TState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str,
        chat_state: str,
    ) -> JourneyTransition[ChatJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str,
        tool_instruction: str | None = None,
        tool_state: ToolEntry,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str,
        tool_instruction: str | None = None,
        tool_state: Sequence[ToolEntry],
    ) -> JourneyTransition[ToolJourneyState]: ...

    async def transition_to(
        self,
        *,
        condition: str,
        chat_state: str | None = None,
        tool_instruction: str | None = None,
        state: TState | None = None,
        tool_state: ToolEntry | Sequence[ToolEntry] = [],
    ) -> JourneyTransition[Any]:
        return await self._transition(
            condition=condition,
            state=state,
            action=chat_state or tool_instruction,
            tools=[tool_state] if isinstance(tool_state, ToolEntry) else tool_state,
        )


@dataclass(frozen=True)
class Journey:
    id: JourneyId
    title: str
    description: str
    conditions: list[Guideline]
    states: Sequence[JourneyState]
    transitions: Sequence[JourneyTransition[JourneyState]]
    tags: Sequence[TagId]

    _start_state_id: JourneyStateId
    _server: Server
    _container: Container

    @property
    def initial_state(self) -> InitialJourneyState:
        return cast(
            InitialJourneyState, next(n for n in self.states if n.id == self._start_state_id)
        )

    async def _create_state(
        self,
        state_type: type[TState],
        action: str | None = None,
        tools: Sequence[ToolEntry] = [],
    ) -> TState:
        metadata_type = {
            ForkJourneyState: "fork",
            ToolJourneyState: "tool",
            ChatJourneyState: "chat",
        }[state_type]

        for t in list(tools):
            await self._server._plugin_server.enable_tool(t)

        if len(tools) == 1 and not action:
            action = f"Use the tool {tools[0].tool.name}"

        node = await self._container[JourneyStore].create_node(
            journey_id=self.id,
            action=action,
            tools=[
                ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name)
                for t in tools
            ],
        )

        node = await self._container[JourneyStore].set_node_metadata(
            node_id=node.id,
            key="journey_node",
            value={"kind": metadata_type},
        )

        return state_type(
            id=node.id,
            action=action,
            tools=tools,
            metadata=node.metadata,
            _journey=self,
        )

    async def create_transition(
        self,
        condition: str | None,
        source: JourneyState,
        target: TState,
    ) -> JourneyTransition[TState]:
        self._server._advance_creation_progress()

        if target is not None and target.id != END_JOURNEY.id:
            target_tool_ids = {
                t.tool.name: ToolId(
                    service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name
                )
                for t in target.tools
            }

            self._server._add_state_evaluation(
                target.id,
                GuidelineContent(condition=condition or "", action=target.internal_action),
                list(target_tool_ids.values()),
            )

        transition = await self._container[JourneyStore].create_edge(
            journey_id=self.id,
            source=source.id,
            target=target.id if target else END_JOURNEY.id,
            condition=condition,
        )

        return JourneyTransition[TState](
            id=transition.id,
            condition=condition,
            source=source,
            target=target,
            metadata=transition.metadata,
        )

    async def create_guideline(
        self,
        condition: str,
        action: str | None = None,
        tools: Iterable[ToolEntry] = [],
        metadata: dict[str, JSONSerializable] = {},
    ) -> Guideline:
        self._server._advance_creation_progress()

        tool_ids = [
            ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name) for t in tools
        ]

        for t in list(tools):
            await self._server._plugin_server.enable_tool(t)

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=action,
        )

        self._server._add_guideline_evaluation(
            guideline.id,
            GuidelineContent(condition=condition, action=action),
            tool_ids,
        )

        await self._container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=guideline.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=_Tag.for_journey_id(self.id),
                kind=RelationshipEntityKind.TAG,
            ),
            kind=RelationshipKind.DEPENDENCY,
        )

        for t in list(tools):
            await self._container[GuidelineToolAssociationStore].create_association(
                guideline_id=guideline.id,
                tool_id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name),
            )

        return Guideline(
            id=guideline.id,
            condition=condition,
            action=action,
            tags=guideline.tags,
            metadata=guideline.metadata,
            _server=self._server,
            _container=self._container,
        )

    async def attach_tool(
        self,
        tool: ToolEntry,
        condition: str,
    ) -> GuidelineId:
        await self._server._plugin_server.enable_tool(tool)

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=None,
        )

        self._server._add_guideline_evaluation(
            guideline.id,
            GuidelineContent(condition=condition, action=None),
            [ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name)],
        )

        await self._container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=guideline.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=_Tag.for_journey_id(self.id),
                kind=RelationshipEntityKind.TAG,
            ),
            kind=RelationshipKind.DEPENDENCY,
        )

        await self._container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name),
        )

        return guideline.id

    async def create_canned_response(
        self,
        template: str,
        tags: list[TagId] = [],
        signals: list[str] = [],
    ) -> CannedResponseId:
        self._server._advance_creation_progress()

        can_rep = await self._container[CannedResponseStore].create_can_rep(
            value=template,
            tags=[_Tag.for_journey_id(self.id), *tags],
            fields=[],
            signals=signals,
        )

        return can_rep.id


@dataclass(frozen=True)
class Capability:
    id: CapabilityId
    title: str
    description: str
    signals: Sequence[str]
    tags: Sequence[TagId]


@dataclass(frozen=True)
class Term:
    id: TermId
    name: str
    description: str
    synonyms: Sequence[str]
    tags: Sequence[TagId]


@dataclass(frozen=True)
class Variable:
    id: ContextVariableId
    name: str
    description: str | None
    tool: ToolEntry | None
    freshness_rules: str | None
    tags: Sequence[TagId]
    _server: Server
    _container: Container

    async def set_value_for_customer(self, customer: Customer, value: JSONSerializable) -> None:
        await self._container[ContextVariableStore].update_value(
            variable_id=self.id,
            key=customer.id,
            data=value,
        )

    async def set_value_for_tag(self, tag: TagId, value: JSONSerializable) -> None:
        await self._container[ContextVariableStore].update_value(
            variable_id=self.id,
            key=f"tag:{tag}",
            data=value,
        )

    async def set_global_value(self, value: JSONSerializable) -> None:
        await self._container[ContextVariableStore].update_value(
            variable_id=self.id,
            key=ContextVariableStore.GLOBAL_KEY,
            data=value,
        )

    async def get_value_for_customer(self, customer: Customer) -> JSONSerializable | None:
        value = await self._container[ContextVariableStore].read_value(
            variable_id=self.id,
            key=customer.id,
        )

        return value.data if value else None

    async def get_value_for_tag(self, tag: TagId) -> JSONSerializable | None:
        value = await self._container[ContextVariableStore].read_value(
            variable_id=self.id,
            key=f"tag:{tag}",
        )

        return value.data if value else None

    async def get_global_value(self) -> JSONSerializable | None:
        value = await self._container[ContextVariableStore].read_value(
            variable_id=self.id,
            key=ContextVariableStore.GLOBAL_KEY,
        )

        return value.data if value else None


@dataclass(frozen=True)
class Customer:
    @staticmethod
    def guest() -> Customer:
        return Customer(
            id=CustomerStore.GUEST_ID,
            name="Guest",
            metadata={},
            tags=[],
        )

    id: CustomerId
    name: str
    metadata: Mapping[str, str]
    tags: Sequence[TagId]


@dataclass(frozen=True)
class RetrieverContext:
    correlation_id: str
    session: Session
    customer: Customer
    variables: Mapping[ContextVariableId, JSONSerializable]
    interaction: Interaction


@dataclass(frozen=True)
class RetrieverResult:
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)
    canned_responses: Sequence[str] = field(default_factory=list)
    canned_response_fields: Mapping[str, Any] = field(default_factory=dict)


class CompositionMode(enum.Enum):
    FLUID = _CompositionMode.CANNED_FLUID
    COMPOSITED = _CompositionMode.CANNED_COMPOSITED
    STRICT = _CompositionMode.CANNED_STRICT


@dataclass(frozen=True)
class Agent:
    _server: Server
    _container: Container

    id: AgentId
    name: str
    description: str | None
    max_engine_iterations: int
    composition_mode: CompositionMode
    tags: Sequence[TagId]

    retrievers: Mapping[str, Callable[[RetrieverContext], Awaitable[JSONSerializable]]] = field(
        default_factory=dict
    )

    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: list[str | Guideline],
    ) -> Journey:
        self._server._advance_creation_progress()

        journey = await self._server.create_journey(title, description, conditions)

        await self.attach_journey(journey)

        return Journey(
            id=journey.id,
            title=journey.title,
            description=description,
            conditions=journey.conditions,
            tags=journey.tags,
            states=journey.states,
            transitions=journey.transitions,
            _start_state_id=journey._start_state_id,
            _server=self._server,
            _container=self._container,
        )

    async def attach_journey(self, journey: Journey) -> None:
        await self._container[JourneyStore].upsert_tag(
            journey.id,
            _Tag.for_agent_id(self.id),
        )

    async def create_guideline(
        self,
        condition: str,
        action: str | None = None,
        tools: Iterable[ToolEntry] = [],
        metadata: dict[str, JSONSerializable] = {},
    ) -> Guideline:
        self._server._advance_creation_progress()

        tool_ids = [
            ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name) for t in tools
        ]

        for t in list(tools):
            await self._server._plugin_server.enable_tool(t)

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=action,
            tags=[_Tag.for_agent_id(self.id)],
        )

        self._server._add_guideline_evaluation(
            guideline.id,
            GuidelineContent(condition=condition, action=action),
            tool_ids,
        )

        for t in list(tools):
            await self._container[GuidelineToolAssociationStore].create_association(
                guideline_id=guideline.id,
                tool_id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name),
            )

        return Guideline(
            id=guideline.id,
            condition=condition,
            action=action,
            tags=guideline.tags,
            metadata=guideline.metadata,
            _server=self._server,
            _container=self._container,
        )

    async def create_observation(
        self,
        condition: str,
    ) -> Guideline:
        return await self.create_guideline(condition=condition)

    async def attach_tool(
        self,
        tool: ToolEntry,
        condition: str,
    ) -> GuidelineId:
        await self._server._plugin_server.enable_tool(tool)

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=None,
        )

        self._server._add_guideline_evaluation(
            guideline.id,
            GuidelineContent(condition=condition, action=None),
            [ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name)],
        )

        await self._container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name),
        )

        return guideline.id

    async def create_canned_response(
        self,
        template: str,
        tags: list[TagId] = [],
        signals: list[str] = [],
    ) -> CannedResponseId:
        self._server._advance_creation_progress()

        can_rep = await self._container[CannedResponseStore].create_can_rep(
            value=template,
            tags=[_Tag.for_agent_id(self.id), *tags],
            fields=[],
            signals=signals,
        )

        return can_rep.id

    async def create_capability(
        self,
        title: str,
        description: str,
        signals: Sequence[str] | None = None,
    ) -> Capability:
        self._server._advance_creation_progress()

        capability = await self._container[CapabilityStore].create_capability(
            title=title,
            description=description,
            signals=signals,
            tags=[_Tag.for_agent_id(self.id)],
        )

        return Capability(
            id=capability.id,
            title=capability.title,
            description=capability.description,
            signals=capability.signals,
            tags=capability.tags,
        )

    async def create_term(
        self,
        name: str,
        description: str,
        synonyms: Sequence[str] = [],
    ) -> Term:
        self._server._advance_creation_progress()

        term = await self._container[GlossaryStore].create_term(
            name=name,
            description=description,
            synonyms=synonyms,
            tags=[_Tag.for_agent_id(self.id)],
        )

        return Term(
            id=term.id,
            name=term.name,
            description=term.description,
            synonyms=term.synonyms,
            tags=term.tags,
        )

    async def create_variable(
        self,
        name: str,
        description: str | None = None,
        tool: ToolEntry | None = None,
        freshness_rules: str | None = None,
    ) -> Variable:
        self._server._advance_creation_progress()

        variable = await self._container[ContextVariableStore].create_variable(
            name=name,
            description=description,
            tool_id=ToolId(INTEGRATED_TOOL_SERVICE_NAME, tool.tool.name) if tool else None,
            freshness_rules=freshness_rules,
            tags=[_Tag.for_agent_id(self.id)],
        )

        return Variable(
            id=variable.id,
            name=variable.name,
            description=variable.description,
            tool=tool,
            freshness_rules=variable.freshness_rules,
            tags=variable.tags,
            _server=self._server,
            _container=self._container,
        )

    async def list_variables(self) -> Sequence[Variable]:
        variables = await self._container[ContextVariableStore].list_variables(
            tags=[_Tag.for_agent_id(self.id)]
        )

        return [
            Variable(
                id=variable.id,
                name=variable.name,
                description=variable.description,
                tool=self._server._plugin_server.tools[variable.tool_id.tool_name]
                if variable.tool_id
                else None,
                freshness_rules=variable.freshness_rules,
                tags=variable.tags,
                _server=self._server,
                _container=self._container,
            )
            for variable in variables
        ]

    async def find_variable(
        self,
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> Variable | None:
        if not id and not name:
            raise SDKError("Either id or name must be provided to find a variable.")

        variable: ContextVariable | None = None

        if id:
            try:
                variable = await self._container[ContextVariableStore].read_variable(
                    ContextVariableId(id)
                )
            except ItemNotFoundError:
                return None
        else:
            variable = next(
                (
                    v
                    for v in await self._container[ContextVariableStore].list_variables(
                        tags=[_Tag.for_agent_id(self.id)]
                    )
                    if v.name == name
                ),
                None,
            )

            if not variable:
                return None

        return Variable(
            id=variable.id,
            name=variable.name,
            description=variable.description,
            tool=self._server._plugin_server.tools[variable.tool_id.tool_name]
            if variable.tool_id
            else None,
            freshness_rules=variable.freshness_rules,
            tags=variable.tags,
            _server=self._server,
            _container=self._container,
        )

    async def attach_retriever(
        self,
        retriever: Callable[[RetrieverContext], Awaitable[JSONSerializable | RetrieverResult]],
        id: str | None = None,
    ) -> None:
        if not id:
            id = f"retriever-{len(self.retrievers) + 1}"

        cast(
            dict[str, Callable[[RetrieverContext], Awaitable[JSONSerializable | RetrieverResult]]],
            self.retrievers,
        )[id] = retriever

        self._server._retrievers[self.id][id] = retriever


class ToolContextAccessor:
    def __init__(self, context: ToolContext) -> None:
        self.context = context

    @property
    def server(self) -> Server:
        return cast(Server, self.context.plugin_data["server"])


class Server:
    def __init__(
        self,
        port: int = 8800,
        tool_service_port: int = 8818,
        nlp_service: Callable[[Container], NLPService] = _load_openai,
        session_store: Literal["transient", "local"] | str | SessionStore = "transient",
        customer_store: Literal["transient", "local"] | str | CustomerStore = "transient",
        log_level: LogLevel = LogLevel.INFO,
        modules: list[str] = [],
        migrate: bool = False,
        configure_hooks: Callable[[EngineHooks], Awaitable[EngineHooks]] | None = None,
        configure_container: Callable[[Container], Awaitable[Container]] | None = None,
        initialize: Callable[[Container], Awaitable[None]] | None = None,
    ) -> None:
        self.port = port
        self.tool_service_port = tool_service_port
        self.log_level = log_level
        self.modules = modules

        self._migrate = migrate
        self._nlp_service_func = nlp_service
        self._evaluator: _CachedEvaluator
        self._session_store = session_store
        self._customer_store = customer_store
        self._configure_hooks = configure_hooks
        self._configure_container = configure_container
        self._initialize = initialize
        self._retrievers: dict[
            AgentId,
            dict[str, Callable[[RetrieverContext], Awaitable[JSONSerializable | RetrieverResult]]],
        ] = defaultdict(dict)
        self._exit_stack = AsyncExitStack()

        self._plugin_server: PluginServer
        self._container: Container

        self._guideline_evaluations: dict[
            GuidelineId, Coroutine[Any, Any, _CachedEvaluator.GuidelineEvaluation]
        ] = {}
        self._node_evaluations: dict[
            JourneyStateId, Coroutine[Any, Any, _CachedEvaluator.GuidelineEvaluation]
        ] = {}
        self._journey_evaluations: dict[
            JourneyId, Coroutine[Any, Any, _CachedEvaluator.JourneyEvaluation]
        ] = {}

        self._creation_progress: Progress | None = Progress(
            TextColumn("{task.description}"),
            BarColumn(pulse_style="bold green"),
            TimeElapsedColumn(),
        )
        self._creation_progress_k = 0
        self._creation_progress_task_id: TaskID

    def _advance_creation_progress(self) -> None:
        if self._creation_progress is None:
            return

        self._creation_progress_k += 1

        self._creation_progress.update(
            self._creation_progress_task_id,
            description=f"Caching entity embeddings ({self._creation_progress_k})",
        )

    async def __aenter__(self) -> Server:
        self._startup_context_manager = start_parlant(self._get_startup_params())
        self._container = await self._startup_context_manager.__aenter__()

        assert self._creation_progress
        self._creation_progress = self._creation_progress.__enter__()
        self._creation_progress_task_id = self._creation_progress.add_task(
            "Caching entity embeddings", total=None
        )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        assert self._creation_progress
        self._creation_progress.__exit__(None, None, None)
        self._creation_progress = None

        await self._process_evaluations()
        await self._setup_retrievers()
        await self._startup_context_manager.__aexit__(exc_type, exc_value, tb)
        await self._exit_stack.aclose()
        return False

    def _add_guideline_evaluation(
        self,
        guideline_id: GuidelineId,
        guideline_content: GuidelineContent,
        tool_ids: Sequence[ToolId],
    ) -> None:
        evaluation = self._evaluator.evaluate_guideline(
            guideline_id,
            guideline_content,
            tool_ids,
        )

        self._guideline_evaluations[guideline_id] = evaluation

    def _add_state_evaluation(
        self,
        state_id: JourneyStateId,
        guideline_content: GuidelineContent,
        tools: Sequence[ToolId],
    ) -> None:
        evaluation = self._evaluator.evaluate_state(
            state_id,
            guideline_content,
            tools,
        )

        self._node_evaluations[state_id] = evaluation

    def _add_journey_evaluation(
        self,
        journey: Journey,
    ) -> None:
        evaluation = self._evaluator.evaluate_journey(journey)

        self._journey_evaluations[journey.id] = evaluation

    async def _render_guideline(self, guideline_id: GuidelineId) -> str:
        guideline = await self._container[GuidelineStore].read_guideline(guideline_id)

        return f"When {guideline.content.condition}" + (
            f", then {guideline.content.action }" if guideline.content.action else ""
        )

    async def _render_state(self, state_id: JourneyStateId) -> str:
        state = await self._container[JourneyStore].read_node(state_id)

        return f"State: {state.action}"

    async def _render_journey(self, journey_id: JourneyId) -> str:
        journey = await self._container[JourneyStore].read_journey(journey_id)

        return f"Journey: {journey.title}"

    async def _process_evaluations(self) -> None:
        _render_functions: dict[
            Literal["guideline", "node", "journey"],
            Callable[[GuidelineId | JourneyStateId | JourneyId], Awaitable[str]],
        ] = {
            "guideline": self._render_guideline,  # type: ignore
            "node": self._render_state,  # type: ignore
            "journey": self._render_journey,  # type: ignore
        }

        def create_evaluation_task(
            evaluation: Coroutine[
                Any, Any, _CachedEvaluator.GuidelineEvaluation | _CachedEvaluator.JourneyEvaluation
            ],
            entity_type: Literal["guideline", "node", "journey"],
            entity_id: GuidelineId | JourneyStateId | JourneyId,
        ) -> asyncio.Task[
            tuple[
                Literal["guideline", "node", "journey"],
                GuidelineId | JourneyStateId | JourneyId,
                _CachedEvaluator.GuidelineEvaluation | _CachedEvaluator.JourneyEvaluation,
            ]
        ]:
            async def task_wrapper() -> (
                tuple[
                    Literal["guideline", "node", "journey"],
                    GuidelineId | JourneyStateId | JourneyId,
                    _CachedEvaluator.GuidelineEvaluation | _CachedEvaluator.JourneyEvaluation,
                ]
            ):
                result = await evaluation
                return (entity_type, entity_id, result)

            return asyncio.create_task(task_wrapper(), name=f"{entity_type}_evaluation_{entity_id}")

        tasks: list[
            asyncio.Task[
                tuple[
                    Literal["guideline", "node", "journey"],
                    GuidelineId | JourneyStateId | JourneyId,
                    _CachedEvaluator.GuidelineEvaluation | _CachedEvaluator.JourneyEvaluation,
                ]
            ]
        ] = []

        for guideline_id, evaluation_func in self._guideline_evaluations.items():
            tasks.append((create_evaluation_task(evaluation_func, "guideline", guideline_id)))

        for node_id, evaluation_func in self._node_evaluations.items():
            tasks.append((create_evaluation_task(evaluation_func, "node", node_id)))

        for journey_id, journey_evaluation_func in self._journey_evaluations.items():
            tasks.append((create_evaluation_task(journey_evaluation_func, "journey", journey_id)))

        if self.log_level == LogLevel.TRACE:
            evaluation_results = await async_utils.safe_gather(*tasks)
        else:
            max_visible = 5

            overall_progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                TaskProgressColumn(style="bold blue"),
                "{task.completed}/{task.total}",
                TimeElapsedColumn(),
            )

            entity_progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                TaskProgressColumn(style="bold blue"),
                "{task.completed}/{task.total}",
                TimeElapsedColumn(),
                transient=True,
            )

            with Live(Group(overall_progress, entity_progress), refresh_per_second=10):
                bar_id: dict[str, int] = {}

                for t in tasks:
                    entity_id = cast(
                        GuidelineId | JourneyStateId | JourneyId, t.get_name().split("_")[-1]
                    )
                    entity_type = t.get_name().split("_")[0]
                    description = await _render_functions[
                        cast(Literal["guideline", "node", "journey"], entity_type)
                    ](entity_id)

                    bar_id[entity_id] = entity_progress.add_task(
                        description[:50],
                        total=100,
                    )

                overall = overall_progress.add_task("Evaluating entities", total=100)

                gather = asyncio.create_task(async_utils.safe_gather(*tasks))

                while not gather.done():
                    unfinished: list[tuple[str, float]] = []

                    for _id, rich_id in bar_id.items():
                        pct = self._evaluator._progress_for(_id)
                        entity_progress.update(TaskID(rich_id), completed=pct)

                        if pct < 100.0:
                            unfinished.append((_id, pct))

                    if unfinished:
                        show = {
                            e_id for e_id, _ in sorted(unfinished, key=lambda x: x[1])[:max_visible]
                        }
                    else:
                        show = set()

                    for e_id, rich_id in bar_id.items():
                        entity_progress.update(TaskID(rich_id), visible=(e_id in show))

                    overall_pct = sum(self._evaluator._progress_for(e_id) for e_id in bar_id) / len(
                        bar_id
                    )
                    overall_progress.update(overall, completed=overall_pct)

                    await asyncio.sleep(0.2)

                for e_id, rich_id in bar_id.items():
                    entity_progress.remove_task(
                        TaskID(rich_id),
                    )

                entity_progress.refresh()
                overall_progress.update(overall, completed=100)
                evaluation_results = await gather

        for entity_type, entity_id, result in evaluation_results:
            if entity_type == "guideline":
                guideline = await self._container[GuidelineStore].read_guideline(
                    guideline_id=cast(GuidelineId, entity_id)
                )

                properties = cast(_CachedEvaluator.GuidelineEvaluation, result).properties

                properties_to_add = {
                    k: v for k, v in properties.items() if k not in guideline.metadata
                }

                for key, value in properties_to_add.items():
                    await self._container[GuidelineStore].set_metadata(
                        guideline_id=cast(GuidelineId, entity_id),
                        key=key,
                        value=value,
                    )

            elif entity_type == "node":
                node = await self._container[JourneyStore].read_node(
                    node_id=cast(JourneyStateId, entity_id)
                )
                properties = cast(_CachedEvaluator.GuidelineEvaluation, result).properties

                properties_to_add = {k: v for k, v in properties.items() if k not in node.metadata}

                for key, value in properties_to_add.items():
                    await self._container[JourneyStore].set_node_metadata(
                        node_id=cast(JourneyStateId, entity_id),
                        key=key,
                        value=value,
                    )

            elif entity_type == "journey":
                for node_id, properties in cast(
                    _CachedEvaluator.JourneyEvaluation, result
                ).node_properties.items():
                    node = await self._container[JourneyStore].read_node(node_id)
                    properties_to_add = {
                        k: v
                        for k, v in properties.items()
                        if k not in node.metadata or node.metadata[k] is None
                    }

                    for key, value in properties_to_add.items():
                        await self._container[JourneyStore].set_node_metadata(
                            node_id=node_id,
                            key=key,
                            value=value,
                        )

    async def _setup_retrievers(self) -> None:
        async def setup_retriever(
            c: Container,
            agent_id: AgentId,
            retriever_id: str,
            retriever: Callable[[RetrieverContext], Awaitable[JSONSerializable | RetrieverResult]],
        ) -> None:
            tasks_for_this_retriever: dict[
                str,
                tuple[Timeout, asyncio.Task[JSONSerializable | RetrieverResult]],
            ] = {}

            async def on_message_acknowledged(
                ctx: LoadedContext,
                exc: Optional[Exception],
            ) -> EngineHookResult:
                # First do some garbage collection if needed.
                # This might be needed if tasks were not awaited
                # because of exceptions during engine processing.
                for correlation_id in list(tasks_for_this_retriever.keys()):
                    if tasks_for_this_retriever[correlation_id][0].expired():
                        # Very, very little change that this task is still meant to be running,
                        # or that anyone is still waiting for it. It's 99.999% garbage.
                        try:
                            tasks_for_this_retriever[correlation_id][1].add_done_callback(
                                default_done_callback()
                            )
                            tasks_for_this_retriever[correlation_id][1].cancel()
                            del tasks_for_this_retriever[correlation_id]
                        except BaseException:
                            # If anything went unexpectedly here, whatever. Carry on.
                            pass

                coroutine = retriever(
                    RetrieverContext(
                        correlation_id=ctx.correlation_id,
                        session=ctx.session,
                        customer=Customer(
                            id=ctx.customer.id,
                            name=ctx.customer.name,
                            metadata=ctx.customer.extra,
                            tags=ctx.customer.tags,
                        ),
                        variables={var.id: val.data for var, val in ctx.state.context_variables},
                        interaction=ctx.interaction,
                    )
                )

                c[Logger].trace(
                    f"Starting retriever {retriever_id} for agent {agent_id} with correlation {ctx.correlation_id}"
                )

                tasks_for_this_retriever[ctx.correlation_id] = (
                    Timeout(600),  # Expiration timeout for garbage collection purposes
                    asyncio.create_task(
                        cast(Coroutine[Any, Any, JSONSerializable | RetrieverResult], coroutine),
                        name=f"Retriever {retriever_id} for agent {agent_id}",
                    ),
                )

                return EngineHookResult.CALL_NEXT

            async def on_generating_messages(
                ctx: LoadedContext,
                exc: Optional[Exception],
            ) -> EngineHookResult:
                if timeout_and_task := tasks_for_this_retriever.pop(ctx.correlation_id, None):
                    _, task = timeout_and_task
                    task_result = await task

                    if isinstance(task_result, RetrieverResult):
                        retriever_result = task_result
                    else:
                        retriever_result = RetrieverResult(
                            data=task_result,
                            metadata={},
                            canned_responses=[],
                            canned_response_fields={},
                        )

                    ctx.state.tool_events.append(
                        await ctx.response_event_emitter.emit_tool_event(
                            ctx.correlation_id,
                            ToolEventData(
                                tool_calls=[
                                    _SessionToolCall(
                                        tool_id=ToolId(
                                            service_name=INTEGRATED_TOOL_SERVICE_NAME,
                                            tool_name=retriever_id,
                                        ).to_string(),
                                        arguments={},
                                        result=_SessionToolResult(
                                            data=retriever_result.data,
                                            metadata=retriever_result.metadata,
                                            control={"lifespan": "response"},
                                            canned_responses=[
                                                CannedResponse(
                                                    id=CannedResponse.TRANSIENT_ID,
                                                    creation_utc=datetime.now(timezone.utc),
                                                    value=u,
                                                    fields=[],
                                                    signals=[],
                                                    tags=[],
                                                )
                                                for u in retriever_result.canned_responses
                                            ],
                                            canned_response_fields=retriever_result.canned_response_fields,
                                        ),
                                    )
                                ]
                            ),
                        )
                    )

                return EngineHookResult.CALL_NEXT

            c[EngineHooks].on_acknowledged.append(on_message_acknowledged)
            c[EngineHooks].on_generating_messages.append(on_generating_messages)

        for agent in self._retrievers:
            for retriever_id, retriever in self._retrievers[agent].items():
                await setup_retriever(self._container, agent, retriever_id, retriever)

    async def create_tag(self, name: str) -> Tag:
        self._advance_creation_progress()

        tag = await self._container[TagStore].create_tag(name=name)

        return Tag(
            id=tag.id,
            name=tag.name,
        )

    async def create_agent(
        self,
        name: str,
        description: str,
        composition_mode: CompositionMode = CompositionMode.FLUID,
        max_engine_iterations: int | None = None,
        tags: Sequence[TagId] = [],
    ) -> Agent:
        self._advance_creation_progress()

        agent = await self._container[AgentStore].create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations or 1,
            composition_mode=composition_mode.value,
        )

        return Agent(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=CompositionMode(agent.composition_mode),
            tags=tags,
            _server=self,
            _container=self._container,
        )

    async def list_agents(self) -> Sequence[Agent]:
        agents = await self._container[AgentStore].list_agents()

        return [
            Agent(
                id=a.id,
                name=a.name,
                description=a.description,
                max_engine_iterations=a.max_engine_iterations,
                composition_mode=CompositionMode(a.composition_mode),
                tags=a.tags,
                _server=self,
                _container=self._container,
            )
            for a in agents
        ]

    async def find_agent(self, *, id: str) -> Agent | None:
        try:
            agent = await self._container[AgentStore].read_agent(AgentId(id))

            return Agent(
                id=agent.id,
                name=agent.name,
                description=agent.description,
                max_engine_iterations=agent.max_engine_iterations,
                composition_mode=CompositionMode(agent.composition_mode),
                tags=agent.tags,
                _server=self,
                _container=self._container,
            )
        except ItemNotFoundError:
            return None

    async def get_agent(self, *, id: str) -> Agent:
        if agent := await self.find_agent(id=id):
            return agent
        raise SDKError(f"Agent with id {id} not found.")

    async def create_customer(
        self,
        name: str,
        metadata: Mapping[str, str] = {},
        tags: Sequence[TagId] = [],
    ) -> Customer:
        self._advance_creation_progress()

        customer = await self._container[CustomerStore].create_customer(
            name=name,
            extra=metadata,
            tags=tags,
        )

        return Customer(
            id=customer.id,
            name=customer.name,
            metadata=customer.extra,
            tags=customer.tags,
        )

    async def list_customers(self) -> Sequence[Customer]:
        customers = await self._container[CustomerStore].list_customers()

        return [
            Customer(
                id=c.id,
                name=c.name,
                metadata=c.extra,
                tags=c.tags,
            )
            for c in customers
        ]

    async def find_customer(
        self,
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> Customer | None:
        if not id and not name:
            raise SDKError("Either id or name must be provided to find a customer.")

        customer: _Customer | None = None

        if id:
            try:
                customer = await self._container[CustomerStore].read_customer(CustomerId(id))
            except ItemNotFoundError:
                return None

            return Customer(
                id=customer.id,
                name=customer.name,
                metadata=customer.extra,
                tags=customer.tags,
            )

        if name:
            customers = await self._container[CustomerStore].list_customers()

            if customer := next((c for c in customers if c.name == name), None):
                return Customer(
                    id=customer.id,
                    name=customer.name,
                    metadata=customer.extra,
                    tags=customer.tags,
                )

        return None

    async def get_customer(self, *, id: CustomerId) -> Customer:
        customer = await self._container[CustomerStore].read_customer(id)

        return Customer(
            id=customer.id,
            name=customer.name,
            metadata=customer.extra,
            tags=customer.tags,
        )

    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: list[str | Guideline],
        tags: Sequence[TagId] = [],
    ) -> Journey:
        self._advance_creation_progress()

        condition_guidelines = [c for c in conditions if isinstance(c, Guideline)]

        str_conditions = [c for c in conditions if isinstance(c, str)]

        for str_condition in str_conditions:
            guideline = await self._container[GuidelineStore].create_guideline(
                condition=str_condition,
            )

            self._add_guideline_evaluation(
                guideline.id,
                GuidelineContent(condition=str_condition, action=None),
                tool_ids=[],
            )

            condition_guidelines.append(
                Guideline(
                    id=guideline.id,
                    condition=guideline.content.condition,
                    action=guideline.content.action,
                    tags=guideline.tags,
                    metadata=guideline.metadata,
                    _server=self,
                    _container=self._container,
                )
            )

        stored_journey = await self._container[JourneyStore].create_journey(
            title=title,
            description=description,
            conditions=[c.id for c in condition_guidelines],
            tags=[],
        )

        journey = Journey(
            id=stored_journey.id,
            title=title,
            description=description,
            conditions=condition_guidelines,
            states=[],
            transitions=[],
            tags=tags,
            _start_state_id=stored_journey.root_id,
            _server=self,
            _container=self._container,
        )

        start_state = await self._container[JourneyStore].read_node(node_id=stored_journey.root_id)

        cast(list[JourneyState], journey.states).append(
            InitialJourneyState(
                id=start_state.id,
                action=start_state.action,
                tools=[],
                metadata=start_state.metadata,
                _journey=journey,
            )
        )

        for c in condition_guidelines:
            await self._container[GuidelineStore].upsert_tag(
                guideline_id=c.id,
                tag_id=_Tag.for_journey_id(journey_id=journey.id),
            )

        self._add_journey_evaluation(journey)

        return journey

    def _get_startup_params(self) -> StartupParameters:
        async def override_stores_with_transient_versions(c: Callable[[], Container]) -> None:
            c()[NLPService] = self._nlp_service_func(c())

            c()[AgentStore] = _SdkAgentStore()

            for interface, implementation in [
                (ContextVariableStore, ContextVariableDocumentStore),
                (TagStore, TagDocumentStore),
                (GuidelineStore, GuidelineDocumentStore),
                (GuidelineToolAssociationStore, GuidelineToolAssociationDocumentStore),
                (RelationshipStore, RelationshipDocumentStore),
            ]:
                c()[interface] = await self._exit_stack.enter_async_context(
                    implementation(c()[IdGenerator], TransientDocumentDatabase())  #  type: ignore
                )

            c()[EvaluationStore] = await self._exit_stack.enter_async_context(
                EvaluationDocumentStore(TransientDocumentDatabase())
            )

            def make_transient_db() -> Awaitable[DocumentDatabase]:
                async def shim() -> DocumentDatabase:
                    return TransientDocumentDatabase()

                return shim()

            def make_json_db(file_path: Path) -> Awaitable[DocumentDatabase]:
                return self._exit_stack.enter_async_context(
                    JSONFileDocumentDatabase(
                        c()[Logger],
                        file_path,
                    ),
                )

            mongo_client: object | None = None

            async def make_mongo_db(url: str, name: str) -> DocumentDatabase:
                nonlocal mongo_client

                if importlib.util.find_spec("pymongo") is None:
                    raise SDKError(
                        "MongoDB requires an additional package to be installed. "
                        "Please install parlant[mongo] to use MongoDB."
                    )

                from pymongo import AsyncMongoClient
                from parlant.adapters.db.mongo_db import MongoDocumentDatabase

                if mongo_client is None:
                    mongo_client = await self._exit_stack.enter_async_context(
                        AsyncMongoClient[Any](url)
                    )

                db = await self._exit_stack.enter_async_context(
                    MongoDocumentDatabase(
                        mongo_client=cast(AsyncMongoClient[Any], mongo_client),
                        database_name=f"parlant_{name}",
                        logger=c()[Logger],
                    )
                )

                return db

            async def make_persistable_store(t: type[T], spec: str, name: str, **kwargs: Any) -> T:
                store: T

                if spec in ["transient", "local"]:
                    store = await self._exit_stack.enter_async_context(
                        t(
                            database=await cast(
                                dict[str, Callable[[], Awaitable[DocumentDatabase]]],
                                {
                                    "transient": make_transient_db,
                                    "local": lambda: make_json_db(
                                        PARLANT_HOME_DIR / f"{name}.json"
                                    ),
                                },
                            )[spec](),
                            allow_migration=self._migrate,
                            **kwargs,
                        )  # type: ignore
                    )

                    return store
                elif spec.startswith("mongodb://") or spec.startswith("mongodb+srv://"):
                    store = await self._exit_stack.enter_async_context(
                        t(
                            database=await make_mongo_db(spec, name),
                            allow_migration=self._migrate,
                            **kwargs,
                        )  # type: ignore
                    )

                    return store
                else:
                    raise SDKError(
                        f"Invalid session store type: {self._session_store}. "
                        "Expected 'transient', 'local', or a MongoDB connection string."
                    )

            if isinstance(self._session_store, SessionStore):
                c()[SessionStore] = self._session_store
            else:
                c()[SessionStore] = await make_persistable_store(
                    SessionDocumentStore, self._session_store, "sessions"
                )

            if isinstance(self._customer_store, CustomerStore):
                c()[CustomerStore] = self._customer_store
            else:
                c()[CustomerStore] = await make_persistable_store(
                    CustomerDocumentStore,
                    self._customer_store,
                    "customers",
                    id_generator=c()[IdGenerator],
                )

            c()[ServiceRegistry] = await self._exit_stack.enter_async_context(
                ServiceDocumentRegistry(
                    database=TransientDocumentDatabase(),
                    event_emitter_factory=c()[EventEmitterFactory],
                    logger=c()[Logger],
                    correlator=c()[ContextualCorrelator],
                    nlp_services_provider=lambda: {"__nlp__": c()[NLPService]},
                    allow_migration=False,
                )
            )

            embedder_factory = EmbedderFactory(c())

            async def get_embedder_type() -> type[Embedder]:
                return type(await c()[NLPService].get_embedder())

            for vector_store_interface, vector_store_type in [
                (GlossaryStore, GlossaryVectorStore),
                (CannedResponseStore, CannedResponseVectorStore),
                (CapabilityStore, CapabilityVectorStore),
                (JourneyStore, JourneyVectorStore),
            ]:
                c()[vector_store_interface] = await self._exit_stack.enter_async_context(
                    vector_store_type(
                        id_generator=c()[IdGenerator],
                        vector_db=TransientVectorDatabase(
                            c()[Logger],
                            embedder_factory,
                            lambda: c()[EmbeddingCache],
                        ),
                        document_db=TransientDocumentDatabase(),
                        embedder_factory=embedder_factory,
                        embedder_type_provider=get_embedder_type,
                    )  # type: ignore
                )

            c()[Application] = lambda rc: Application(rc)

        async def configure(c: Container) -> Container:
            latest_container = c

            def get_latest_container() -> Container:
                return latest_container

            await override_stores_with_transient_versions(get_latest_container)

            if self._configure_container:
                latest_container = await self._configure_container(latest_container.clone())

            if self._configure_hooks:
                hooks = await self._configure_hooks(c[EngineHooks])
                latest_container[EngineHooks] = hooks

            return latest_container

        async def async_nlp_service_shim(c: Container) -> NLPService:
            return c[NLPService]

        async def initialize(c: Container) -> None:
            host = "127.0.0.1"
            port = self.tool_service_port

            self._plugin_server = PluginServer(
                tools=[],
                port=port,
                host=host,
                hosted=True,
                plugin_data={
                    "server": self,
                    "container": c,
                },
            )

            await c[ServiceRegistry].update_tool_service(
                name=INTEGRATED_TOOL_SERVICE_NAME,
                kind="sdk",
                url=f"http://{host}:{port}",
                transient=True,
            )

            await self._exit_stack.enter_async_context(self._plugin_server)
            self._exit_stack.push_async_callback(self._plugin_server.shutdown)

            self._evaluator = _CachedEvaluator(
                db=JSONFileDocumentDatabase(c[Logger], PARLANT_HOME_DIR / "evaluation_cache.json"),
                container=c,
            )
            await self._exit_stack.enter_async_context(self._evaluator)

            if self._initialize:
                await self._initialize(c)

        return StartupParameters(
            port=self.port,
            nlp_service=async_nlp_service_shim,
            log_level=self.log_level,
            modules=self.modules,
            migrate=self._migrate,
            configure=configure,
            initialize=initialize,
        )


__all__ = [
    "Agent",
    "AgentId",
    "Capability",
    "CapabilityId",
    "CompositionMode",
    "Container",
    "Customer",
    "CustomerId",
    "Variable",
    "ContextVariableId",
    "ControlOptions",
    "Embedder",
    "EmbedderFactory",
    "EmbeddingResult",
    "EmittedEvent",
    "EngineHook",
    "EngineHookResult",
    "EngineHooks",
    "EstimatingTokenizer",
    "EventKind",
    "EventSource",
    "FallbackSchematicGenerator",
    "Guideline",
    "GuidelineId",
    "Interaction",
    "InteractionMessage",
    "Journey",
    "JourneyId",
    "JourneyState",
    "JourneyStateId",
    "END_JOURNEY",
    "JourneyTransition",
    "JourneyTransitionId",
    "JSONSerializable",
    "LoadedContext",
    "LogLevel",
    "Logger",
    "MessageEventData",
    "NLPService",
    "PluginServer",
    "RelationshipEntity",
    "RelationshipEntityId",
    "RelationshipEntityKind",
    "RelationshipId",
    "RelationshipKind",
    "RetrieverContext",
    "RetrieverResult",
    "SchematicGenerationResult",
    "SchematicGenerator",
    "Server",
    "ServiceRegistry",
    "Session",
    "SessionId",
    "SessionMode",
    "SessionStatus",
    "StatusEventData",
    "Tag",
    "TagId",
    "Term",
    "TermId",
    "Tool",
    "ToolContext",
    "ToolContextAccessor",
    "ToolEntry",
    "ToolEventData",
    "ToolId",
    "ToolParameterDescriptor",
    "ToolParameterOptions",
    "ToolParameterType",
    "ToolResult",
    "CannedResponseId",
    "tool",
]
