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
from hashlib import md5
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    cast,
)
from lagom import Container

from parlant.adapters.db.json_file import JSONFileDocumentCollection, JSONFileDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.nlp.openai_service import OpenAIService
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.core.agents import (
    Agent as _Agent,
    AgentId,
    AgentStore,
    AgentUpdateParams,
    CompositionMode,
)
from parlant.core.application import Application
from parlant.core.async_utils import Timeout, default_done_callback
from parlant.core.capabilities import CapabilityId, CapabilityStore, CapabilityVectorStore
from parlant.core.common import JSONSerializable, Version
from parlant.core.context_variables import (
    ContextVariableDocumentStore,
    ContextVariableId,
    ContextVariableStore,
)
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.customers import CustomerDocumentStore, CustomerId, CustomerStore
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
from parlant.core.utterances import Utterance, UtteranceVectorStore, UtteranceId, UtteranceStore
from parlant.core.evaluations import (
    EvaluationDocumentStore,
    EvaluationStatus,
    EvaluationStore,
    GuidelinePayload,
    GuidelinePayloadOperation,
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


class SDKError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _load_openai(container: Container) -> NLPService:
    return OpenAIService(container[Logger])


class _CachedEvaluation(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    action_proposition: str | None
    properties: dict[str, JSONSerializable]


class _CachedEvaluator:
    @dataclass(frozen=True)
    class GuidelineEvaluation:
        properties: dict[str, JSONSerializable]

    def __init__(
        self,
        db: JSONFileDocumentDatabase,
        container: Container,
    ) -> None:
        self._db: JSONFileDocumentDatabase = db
        self._collection: JSONFileDocumentCollection[_CachedEvaluation]
        self._container = container
        self._logger = container[Logger]
        self._exit_stack = AsyncExitStack()

    async def __aenter__(self) -> _CachedEvaluator:
        await self._exit_stack.enter_async_context(self._db)

        self._collection = await self._db.get_or_create_collection(
            name="guideline_evaluations",
            schema=_CachedEvaluation,
            document_loader=identity_loader_for(_CachedEvaluation),
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

    def _hash_request(
        self,
        g: GuidelineContent,
        tool_ids: Sequence[ToolId],
        journey_step_propositions: bool,
    ) -> str:
        """Generate a hash for the evaluation request."""
        tool_ids_str = ",".join(str(tool_id) for tool_id in tool_ids) if tool_ids else ""

        return md5(
            f"{g.condition or ''}:{g.action or ''}:{tool_ids_str}:{journey_step_propositions}".encode()
        ).hexdigest()

    async def evaluate_guideline(
        self,
        g: GuidelineContent,
        tool_ids: Sequence[ToolId] = [],
        journey_step_propositions: bool = False,
    ) -> _CachedEvaluator.GuidelineEvaluation:
        # First check if we have a cached evaluation for this guideline
        _hash = self._hash_request(
            g=g,
            tool_ids=tool_ids,
            journey_step_propositions=journey_step_propositions,
        )

        if cached_evaluation := await self._collection.find_one({"id": {"$eq": _hash}}):
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
                self._logger.info(
                    f"Deleting outdated cached evaluation for guideline: {g.condition or 'None'}"
                )

                await self._collection.delete_one({"id": {"$eq": cached_evaluation["id"]}})

        self._logger.info(
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
                        operation=GuidelinePayloadOperation.ADD,
                        coherence_check=False,  # Legacy and will be removed in the future
                        connection_proposition=False,  # Legacy and will be removed in the future
                        action_proposition=True,
                        properties_proposition=True,
                        journey_step_proposition=journey_step_propositions,
                    ),
                )
            ]
        )

        while True:
            evaluation = await self._container[EvaluationStore].read_evaluation(
                evaluation_id=evaluation_id,
            )

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
            await self._collection.insert_one(
                {
                    "id": ObjectId(_hash),
                    "version": Version.String(VERSION),
                    "properties": invoice.data.properties_proposition or {},
                }
            )

            # Return the evaluation result
            return self.GuidelineEvaluation(
                properties=invoice.data.properties_proposition or {},
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
        composition_mode: CompositionMode | None = None,
        tags: Sequence[TagId] | None = None,
    ) -> _Agent:
        agent = _Agent(
            id=AgentId(name),
            name=name,
            description=description,
            creation_utc=creation_utc or datetime.now(timezone.utc),
            max_engine_iterations=max_engine_iterations or 1,
            tags=tags or [],
            composition_mode=composition_mode or CompositionMode.FLUID,
        )

        self._agents[agent.id] = agent

        return agent

    async def list_agents(self) -> Sequence[_Agent]:
        return list(self._agents.values())

    async def read_agent(self, agent_id: AgentId) -> _Agent:
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


@dataclass(frozen=True)
class JourneyEdge:
    id: JourneyEdgeId
    condition: str | None
    source: JourneyNode
    target: JourneyNode
    metadata: Mapping[str, JSONSerializable]


@dataclass(frozen=True)
class JourneyNode:
    id: JourneyNodeId
    action: str | None
    tools: Sequence[ToolEntry]
    metadata: Mapping[str, JSONSerializable]

    _server: Server
    _container: Container
    _journey: Journey

    END: str | None = None

    @property
    def internal_action(self) -> str | None:
        return self.action or cast(str | None, self.metadata.get("internal_action"))

    async def connect(
        self,
        node: JourneyNode | None = None,
        condition: str | None = None,
        action: str | None = None,
        tools: Sequence[ToolEntry] = [],
    ) -> JourneyEdge:
        if node is None:
            node = await self._journey.create_node(action=action, tools=tools)

        node_connections = [edge for edge in self._journey.edges if edge.source == self]

        if len(node_connections) > 0 and (
            not condition or any(not e.condition for e in node_connections)
        ):
            raise SDKError(
                "Cannot connect a new node without a condition if there are already connected nodes without conditions."
            )

        edge = await self._journey.create_edge(
            condition=condition,
            source=self,
            target=node,
        )

        cast(list[JourneyNode], self._journey.nodes).append(node)
        cast(list[JourneyEdge], self._journey.edges).append(edge)

        return edge


@dataclass(frozen=True)
class Journey:
    id: JourneyId
    title: str
    description: str
    conditions: list[Guideline]
    tags: Sequence[TagId]
    nodes: Sequence[JourneyNode]
    edges: Sequence[JourneyEdge]

    _server: Server
    _container: Container

    @property
    def root(self) -> JourneyNode:
        return next(n for n in self.nodes if n.id == self._container[JourneyStore].ROOT_NODE_ID)

    async def create_node(
        self,
        action: str | None,
        tools: Sequence[ToolEntry],
    ) -> JourneyNode:
        for t in list(tools):
            await self._server._plugin_server.enable_tool(t)

        node = await self._container[JourneyStore].create_node(
            journey_id=self.id,
            action=action,
            tools=[
                ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name)
                for t in tools
            ],
        )

        return JourneyNode(
            id=node.id,
            action=action,
            tools=tools,
            metadata=node.metadata,
            _server=self._server,
            _container=self._container,
            _journey=self,
        )

    async def create_edge(
        self,
        condition: str | None,
        source: JourneyNode,
        target: JourneyNode,
    ) -> JourneyEdge:
        target_tool_ids = {
            t.tool.name: ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name)
            for t in target.tools
        }

        evaluation = await self._server._evaluator.evaluate_guideline(
            GuidelineContent(condition=condition or "", action=target.internal_action),
            list(target_tool_ids.values()),
            journey_step_propositions=True,
        )

        edge = await self._container[JourneyStore].create_edge(
            journey_id=self.id,
            source=source.id,
            target=target.id,
            condition=condition,
        )

        # Set metadata for the target node.
        # The evaluated properties—`tool_running_only`, `customer_dependent_action`, and `action_proposition`
        # are all tied to the node's action, so they are stored in the node metadata rather than in the edge.
        #
        # The evaluation is performed during edge creation for two reasons:
        # 1. Prior to this point, the node is not yet connected to the graph and thus isn't relevant for evaluation.
        # 2. The edge's condition provides important context for understanding the action during evaluation.
        #
        # We assume that evaluations between multiple edges pointing to the same node should not differ.
        # Therefore, we safely override the target node's metadata with the evaluated properties.
        for key, value in evaluation.properties.items():
            await self._container[JourneyStore].set_node_metadata(target.id, key, value)

            if key == "tool_running_only" and value:
                for entry in target.tools:
                    await self._container[RelationshipStore].create_relationship(
                        source=RelationshipEntity(
                            id=target_tool_ids[entry.tool.name],
                            kind=RelationshipEntityKind.TOOL,
                        ),
                        target=RelationshipEntity(
                            id=_Tag.for_journey_node_id(target.id),
                            kind=RelationshipEntityKind.TAG,
                        ),
                        kind=RelationshipKind.REEVALUATION,
                    )

        cast(dict[str, JSONSerializable], target.metadata).update(evaluation.properties)

        return JourneyEdge(
            id=edge.id,
            condition=condition,
            source=source,
            target=target,
            metadata=edge.metadata,
        )

    async def create_guideline(
        self,
        condition: str,
        action: str | None = None,
        tools: Iterable[ToolEntry] = [],
        metadata: dict[str, JSONSerializable] = {},
    ) -> Guideline:
        tool_ids = [
            ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name) for t in tools
        ]

        evaluation = await self._server._evaluator.evaluate_guideline(
            GuidelineContent(condition=condition, action=action),
            tool_ids,
        )

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=action,
            metadata={**evaluation.properties, **metadata},
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
            await self._server._plugin_server.enable_tool(t)

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

        evaluation = await self._server._evaluator.evaluate_guideline(
            GuidelineContent(condition=condition, action=None),
            [ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name)],
        )

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=None,
            metadata=evaluation.properties,
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

    async def create_utterance(
        self,
        template: str,
        tags: list[TagId] = [],
        queries: list[str] = [],
    ) -> UtteranceId:
        utterance = await self._container[UtteranceStore].create_utterance(
            value=template,
            tags=[_Tag.for_journey_id(self.id), *tags],
            fields=[],
            queries=[],
        )

        return utterance.id


@dataclass(frozen=True)
class Capability:
    id: CapabilityId
    title: str
    description: str
    queries: Sequence[str]
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

    async def set_value_for_tag(self, tag: Tag, value: JSONSerializable) -> None:
        await self._container[ContextVariableStore].update_value(
            variable_id=self.id,
            key=f"tag:{tag.id}",
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
    id: CustomerId
    name: str
    extra: Mapping[str, str]
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
    utterances: Sequence[str] = field(default_factory=list)
    utterance_fields: Mapping[str, Any] = field(default_factory=dict)


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
        journey = await self._server.create_journey(title, description, conditions)

        await self.attach_journey(journey)

        return Journey(
            id=journey.id,
            title=journey.title,
            description=description,
            conditions=journey.conditions,
            tags=journey.tags,
            nodes=journey.nodes,
            edges=journey.edges,
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
        tool_ids = [
            ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name) for t in tools
        ]

        evaluation = await self._server._evaluator.evaluate_guideline(
            GuidelineContent(condition=condition, action=action),
            tool_ids,
        )

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=action,
            metadata={**evaluation.properties, **metadata},
            tags=[_Tag.for_agent_id(self.id)],
        )

        for t in list(tools):
            await self._server._plugin_server.enable_tool(t)

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

        evaluation = await self._server._evaluator.evaluate_guideline(
            GuidelineContent(condition=condition, action=None),
            [ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name)],
        )

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=None,
            metadata=evaluation.properties,
        )

        await self._container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name),
        )

        return guideline.id

    async def create_utterance(
        self,
        template: str,
        tags: list[TagId] = [],
        queries: list[str] = [],
    ) -> UtteranceId:
        utterance = await self._container[UtteranceStore].create_utterance(
            value=template,
            tags=[_Tag.for_agent_id(self.id), *tags],
            fields=[],
            queries=queries,
        )

        return utterance.id

    async def create_capability(
        self,
        title: str,
        description: str,
        queries: Sequence[str] | None = None,
    ) -> Capability:
        capability = await self._container[CapabilityStore].create_capability(
            title=title,
            description=description,
            queries=queries,
            tags=[_Tag.for_agent_id(self.id)],
        )

        return Capability(
            id=capability.id,
            title=capability.title,
            description=capability.description,
            queries=capability.queries,
            tags=capability.tags,
        )

    async def create_term(
        self,
        name: str,
        description: str,
        synonyms: Sequence[str] = [],
    ) -> Term:
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


class Server:
    def __init__(
        self,
        port: int = 8800,
        tool_service_port: int = 8818,
        nlp_service: Callable[[Container], NLPService] = _load_openai,
        session_store: Literal["transient", "local"] | SessionStore = "transient",
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
        self.migrate = migrate

        self._nlp_service_func = nlp_service
        self._evaluator: _CachedEvaluator
        self._session_store = session_store
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

    async def __aenter__(self) -> Server:
        self._startup_context_manager = start_parlant(self._get_startup_params())
        self._container = await self._startup_context_manager.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        await self._setup_retrievers()
        await self._startup_context_manager.__aexit__(exc_type, exc_value, tb)
        await self._exit_stack.aclose()
        return False

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
                            extra=ctx.customer.extra,
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
                            utterances=[],
                            utterance_fields={},
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
                                            utterances=[
                                                Utterance(
                                                    id=Utterance.TRANSIENT_ID,
                                                    creation_utc=datetime.now(timezone.utc),
                                                    value=u,
                                                    fields=[],
                                                    queries=[],
                                                    tags=[],
                                                )
                                                for u in retriever_result.utterances
                                            ],
                                            utterance_fields=retriever_result.utterance_fields,
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
        tag = await self._container[TagStore].create_tag(name=name)

        return Tag(
            id=tag.id,
            name=tag.name,
        )

    async def create_agent(
        self,
        name: str,
        description: str,
        composition_mode: CompositionMode = CompositionMode.FLUID_UTTERANCE,
        max_engine_iterations: int | None = None,
        tags: Sequence[TagId] = [],
    ) -> Agent:
        agent = await self._container[AgentStore].create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations or 1,
            composition_mode=composition_mode,
        )

        return Agent(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=agent.composition_mode,
            tags=tags,
            _server=self,
            _container=self._container,
        )

    async def create_customer(
        self,
        name: str,
        extra: Mapping[str, str] = {},
        tags: Sequence[TagId] = [],
    ) -> Customer:
        customer = await self._container[CustomerStore].create_customer(
            name=name,
            extra=extra,
            tags=tags,
        )

        return Customer(
            id=customer.id,
            name=customer.name,
            extra=customer.extra,
            tags=customer.tags,
        )

    async def list_customers(self) -> Sequence[Customer]:
        customers = await self._container[CustomerStore].list_customers()

        return [
            Customer(
                id=c.id,
                name=c.name,
                extra=c.extra,
                tags=c.tags,
            )
            for c in customers
        ]

    async def find_customer_by_name(self, name: str) -> Customer | None:
        customers = await self._container[CustomerStore].list_customers()

        if customer := next((c for c in customers if c.name == name), None):
            return Customer(
                id=customer.id,
                name=customer.name,
                extra=customer.extra,
                tags=customer.tags,
            )

        return None

    async def read_customer(self, customer_id: CustomerId) -> Customer:
        customer = await self._container[CustomerStore].read_customer(customer_id)

        return Customer(
            id=customer.id,
            name=customer.name,
            extra=customer.extra,
            tags=customer.tags,
        )

    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: list[str | Guideline],
        tags: Sequence[TagId] = [],
    ) -> Journey:
        condition_guidelines = [c for c in conditions if isinstance(c, Guideline)]

        str_conditions = [c for c in conditions if isinstance(c, str)]

        for str_condition in str_conditions:
            evaluation = await self._evaluator.evaluate_guideline(
                GuidelineContent(condition=str_condition, action=None)
            )

            guideline = await self._container[GuidelineStore].create_guideline(
                condition=str_condition,
                metadata=evaluation.properties,
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
            nodes=[],
            edges=[],
            tags=tags,
            _server=self,
            _container=self._container,
        )

        root_node = await self._container[JourneyStore].read_node(
            journey_id=journey.id, node_id=JourneyStore.ROOT_NODE_ID
        )

        cast(list[JourneyNode], journey.nodes).append(
            JourneyNode(
                id=root_node.id,
                action=root_node.action,
                tools=[],
                metadata=root_node.metadata,
                _server=self,
                _container=self._container,
                _journey=journey,
            )
        )

        for c in condition_guidelines:
            await self._container[GuidelineStore].upsert_tag(
                guideline_id=c.id,
                tag_id=_Tag.for_journey_id(journey_id=journey.id),
            )

        return journey

    def _get_startup_params(self) -> StartupParameters:
        async def override_stores_with_transient_versions(c: Callable[[], Container]) -> None:
            c()[NLPService] = self._nlp_service_func(c())

            c()[AgentStore] = _SdkAgentStore()

            for interface, implementation in [
                (ContextVariableStore, ContextVariableDocumentStore),
                (CustomerStore, CustomerDocumentStore),
                (EvaluationStore, EvaluationDocumentStore),
                (TagStore, TagDocumentStore),
                (GuidelineStore, GuidelineDocumentStore),
                (GuidelineToolAssociationStore, GuidelineToolAssociationDocumentStore),
                (RelationshipStore, RelationshipDocumentStore),
            ]:
                c()[interface] = await self._exit_stack.enter_async_context(
                    implementation(TransientDocumentDatabase())  #  type: ignore
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

            if isinstance(self._session_store, SessionStore):
                c()[SessionStore] = self._session_store
            else:
                c()[SessionStore] = await self._exit_stack.enter_async_context(
                    SessionDocumentStore(
                        await cast(
                            dict[str, Callable[[], Awaitable[DocumentDatabase]]],
                            {
                                "transient": lambda: make_transient_db(),
                                "local": lambda: make_json_db(PARLANT_HOME_DIR / "sessions.json"),
                            },
                        )[self._session_store](),
                    )
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

            c()[GlossaryStore] = await self._exit_stack.enter_async_context(
                GlossaryVectorStore(
                    vector_db=TransientVectorDatabase(
                        c()[Logger],
                        embedder_factory,
                        lambda: c()[EmbeddingCache],
                    ),
                    document_db=TransientDocumentDatabase(),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

            c()[UtteranceStore] = await self._exit_stack.enter_async_context(
                UtteranceVectorStore(
                    vector_db=TransientVectorDatabase(
                        c()[Logger],
                        embedder_factory,
                        lambda: c()[EmbeddingCache],
                    ),
                    document_db=TransientDocumentDatabase(),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

            c()[CapabilityStore] = await self._exit_stack.enter_async_context(
                CapabilityVectorStore(
                    vector_db=TransientVectorDatabase(
                        c()[Logger],
                        embedder_factory,
                        lambda: c()[EmbeddingCache],
                    ),
                    document_db=TransientDocumentDatabase(),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

            c()[JourneyStore] = await self._exit_stack.enter_async_context(
                JourneyVectorStore(
                    vector_db=TransientVectorDatabase(
                        c()[Logger],
                        embedder_factory,
                        lambda: c()[EmbeddingCache],
                    ),
                    document_db=TransientDocumentDatabase(),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
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
            migrate=self.migrate,
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
    "JourneyNode",
    "JourneyEdge",
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
    "ToolEntry",
    "ToolEventData",
    "ToolId",
    "ToolParameterDescriptor",
    "ToolParameterOptions",
    "ToolParameterType",
    "ToolResult",
    "UtteranceId",
    "tool",
]
