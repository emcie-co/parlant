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
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from types import TracebackType
from typing import Awaitable, Callable, Iterable, Literal, Sequence, TypedDict, cast
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
from parlant.core.capabilities import CapabilityId, CapabilityStore, CapabilityVectorStore
from parlant.core.common import JSONSerializable, Version
from parlant.core.context_variables import (
    ContextVariableDocumentStore,
    ContextVariableStore,
)
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.customers import CustomerDocumentStore, CustomerStore
from parlant.core.emissions import EmittedEvent, EventEmitterFactory
from parlant.core.engines.alpha.hooks import EngineHook, EngineHookResult, EngineHooks
from parlant.core.engines.alpha.loaded_context import LoadedContext
from parlant.core.glossary import GlossaryStore, GlossaryVectorStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from parlant.core.nlp.embedding import Embedder, EmbedderFactory, EmbeddingResult
from parlant.core.nlp.generation import (
    FallbackSchematicGenerator,
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import DocumentDatabase, identity_loader_for
from parlant.core.relationships import (
    GuidelineRelationshipKind,
    RelationshipDocumentStore,
    RelationshipEntity,
    RelationshipEntityId,
    RelationshipEntityKind,
    RelationshipId,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.services.indexing.behavioral_change_evaluation import BehavioralChangeEvaluator
from parlant.core.services.tools.service_registry import ServiceDocumentRegistry, ServiceRegistry
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    SessionId,
    SessionDocumentStore,
    SessionStore,
    StatusEventData,
    ToolEventData,
)
from parlant.core.utterances import UtteranceVectorStore, UtteranceId, UtteranceStore
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
from parlant.core.journeys import JourneyId, JourneyStore, JourneyVectorStore
from parlant.core.loggers import LogLevel, Logger
from parlant.core.nlp.service import NLPService
from parlant.bin.server import PARLANT_HOME_DIR, start_parlant, StartupParameters
from parlant.core.services.tools.plugins import PluginServer, ToolEntry, tool
from parlant.core.tags import Tag, TagDocumentStore, TagId, TagStore
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

_INTEGRATED_TOOL_SERVICE_NAME = "built-in"


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
        action_proposition: str | None
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

    def _hash_guideline(self, g: GuidelineContent) -> str:
        """Generate a hash for the guideline content."""
        return md5(f"{g.condition or ''}:{g.action or ''}".encode()).hexdigest()

    async def evaluate_guideline(self, g: GuidelineContent) -> _CachedEvaluator.GuidelineEvaluation:
        # First check if we have a cached evaluation for this guideline
        if cached_evaluation := await self._collection.find_one(
            {"id": {"$eq": self._hash_guideline(g)}}
        ):
            # Check if the cached evaluation is based on our current runtime version.
            # This is important as the required evaluation data can change between versions.
            if cached_evaluation["version"] == VERSION:
                self._logger.info(
                    f"Using cached evaluation for guideline: Condition: {g.condition or 'None'}; Action: {g.action or 'None'}"
                )

                return self.GuidelineEvaluation(
                    action_proposition=cached_evaluation.get("action_proposition"),
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
                        tool_ids=[],
                        operation=GuidelinePayloadOperation.ADD,
                        coherence_check=False,  # Legacy and will be removed in the future
                        connection_proposition=False,  # Legacy and will be removed in the future
                        action_proposition=g.action is not None,
                        properties_proposition=True,
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
                    "id": ObjectId(self._hash_guideline(g)),
                    "version": Version.String(VERSION),
                    "properties": invoice.data.properties_proposition or {},
                    "action_proposition": invoice.data.action_proposition or None,
                }
            )

            # Return the evaluation result
            return self.GuidelineEvaluation(
                action_proposition=invoice.data.action_proposition,
                properties=invoice.data.properties_proposition or {},
            )


class _PicoAgentStore(AgentStore):
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
class Relationship:
    id: RelationshipId
    kind: RelationshipKind
    source: RelationshipEntityId
    target: RelationshipEntityId


@dataclass
class Guideline:
    id: GuidelineId
    condition: str
    action: str | None
    tags: Sequence[TagId]

    _parlant: Server
    _container: Container

    async def prioritize_over(self, guideline: Guideline) -> Relationship:
        return await self._create_relationship(
            guideline=guideline,
            kind=GuidelineRelationshipKind.PRIORITY,
            direction="source",
        )

    async def entail(self, guideline: Guideline) -> Relationship:
        return await self._create_relationship(
            guideline=guideline,
            kind=GuidelineRelationshipKind.ENTAILMENT,
            direction="source",
        )

    async def depend_on(self, guideline: Guideline) -> Relationship:
        return await self._create_relationship(
            guideline=guideline,
            kind=GuidelineRelationshipKind.DEPENDENCY,
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
                kind=GuidelineRelationshipKind.DISAMBIGUATION,
                direction="source",
            )
            for t in targets
        ]

    async def _create_relationship(
        self,
        guideline: Guideline,
        kind: GuidelineRelationshipKind,
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


@dataclass
class Journey:
    id: JourneyId
    title: str
    description: str
    conditions: list[Guideline]
    tags: Sequence[TagId]

    _parlant: Server
    _container: Container

    async def create_guideline(
        self,
        condition: str,
        action: str | None = None,
        tools: Iterable[ToolEntry] = [],
        metadata: dict[str, JSONSerializable] = {},
    ) -> Guideline:
        evaluation = await self._parlant._evaluator.evaluate_guideline(
            GuidelineContent(condition=condition, action=action)
        )

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=action or evaluation.action_proposition,
            metadata={**evaluation.properties, **metadata},
        )

        await self._container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=guideline.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=Tag.for_journey_id(self.id),
                kind=RelationshipEntityKind.TAG,
            ),
            kind=GuidelineRelationshipKind.DEPENDENCY,
        )

        for t in list(tools):
            await self._parlant._plugin_server.enable_tool(t)

            await self._container[GuidelineToolAssociationStore].create_association(
                guideline_id=guideline.id,
                tool_id=ToolId(service_name=_INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name),
            )

        return Guideline(
            id=guideline.id,
            condition=condition,
            action=action,
            tags=guideline.tags,
            _parlant=self._parlant,
            _container=self._container,
        )

    async def attach_tool(
        self,
        tool: ToolEntry,
        condition: str,
    ) -> GuidelineId:
        await self._parlant._plugin_server.enable_tool(tool)

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=f"Consider using the tool {tool.tool.name}",
            tags=[],
        )

        await self._container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=guideline.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=Tag.for_journey_id(self.id),
                kind=RelationshipEntityKind.TAG,
            ),
            kind=GuidelineRelationshipKind.DEPENDENCY,
        )

        await self._container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId(service_name=_INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name),
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
            tags=[Tag.for_journey_id(self.id), *tags],
            fields=[],
            queries=[],
        )

        return utterance.id


@dataclass
class Capability:
    id: CapabilityId
    title: str
    description: str
    queries: Sequence[str]
    tags: Sequence[TagId]


@dataclass
class Agent:
    id: AgentId
    name: str
    description: str | None
    max_engine_iterations: int
    composition_mode: CompositionMode
    tags: Sequence[TagId]

    _parlant: Server
    _container: Container

    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: list[str | Guideline],
    ) -> Journey:
        journey = await self._parlant.create_journey(title, description, conditions)

        await self.attach_journey(journey)

        return Journey(
            id=journey.id,
            title=journey.title,
            description=description,
            conditions=journey.conditions,
            tags=journey.tags,
            _parlant=self._parlant,
            _container=self._container,
        )

    async def attach_journey(self, journey: Journey) -> None:
        await self._container[JourneyStore].upsert_tag(
            journey.id,
            Tag.for_agent_id(self.id),
        )

    async def create_guideline(
        self,
        condition: str,
        action: str | None = None,
        tools: Iterable[ToolEntry] = [],
        metadata: dict[str, JSONSerializable] = {},
    ) -> Guideline:
        evaluation = await self._parlant._evaluator.evaluate_guideline(
            GuidelineContent(condition=condition, action=action)
        )

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=action or evaluation.action_proposition,
            metadata={**evaluation.properties, **metadata},
            tags=[Tag.for_agent_id(self.id)],
        )

        for t in list(tools):
            await self._parlant._plugin_server.enable_tool(t)

            await self._container[GuidelineToolAssociationStore].create_association(
                guideline_id=guideline.id,
                tool_id=ToolId(service_name=_INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name),
            )

        return Guideline(
            id=guideline.id,
            condition=condition,
            action=action,
            tags=guideline.tags,
            _parlant=self._parlant,
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
        await self._parlant._plugin_server.enable_tool(tool)

        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=f"Consider using the tool {tool.tool.name}",
            tags=[Tag.for_agent_id(self.id)],
        )

        await self._container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId(service_name=_INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name),
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
            tags=tags,
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
            tags=[Tag.for_agent_id(self.id)],
        )

        return Capability(
            id=capability.id,
            title=capability.title,
            description=capability.description,
            queries=capability.queries,
            tags=capability.tags,
        )


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
        await self._startup_context_manager.__aexit__(exc_type, exc_value, tb)
        await self._exit_stack.aclose()
        return False

    async def create_agent(
        self,
        name: str,
        description: str,
        composition_mode: CompositionMode = CompositionMode.COMPOSITED_UTTERANCE,
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
            _parlant=self,
            _container=self._container,
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
                    _parlant=self,
                    _container=self._container,
                )
            )

        journey = await self._container[JourneyStore].create_journey(
            title,
            description,
            [c.id for c in condition_guidelines],
        )

        for c in condition_guidelines:
            await self._container[GuidelineStore].upsert_tag(
                guideline_id=c.id,
                tag_id=Tag.for_journey_id(journey_id=journey.id),
            )

        return Journey(
            id=journey.id,
            title=journey.title,
            description=description,
            conditions=condition_guidelines,
            tags=tags,
            _container=self._container,
            _parlant=self,
        )

    def _get_startup_params(self) -> StartupParameters:
        async def override_stores_with_transient_versions(c: Container) -> None:
            c[NLPService] = self._nlp_service_func(c)

            c[AgentStore] = _PicoAgentStore()

            for interface, implementation in [
                (ContextVariableStore, ContextVariableDocumentStore),
                (CustomerStore, CustomerDocumentStore),
                (EvaluationStore, EvaluationDocumentStore),
                (TagStore, TagDocumentStore),
                (GuidelineStore, GuidelineDocumentStore),
                (GuidelineToolAssociationStore, GuidelineToolAssociationDocumentStore),
                (RelationshipStore, RelationshipDocumentStore),
            ]:
                c[interface] = await self._exit_stack.enter_async_context(
                    implementation(TransientDocumentDatabase())  #  type: ignore
                )

            def make_transient_db() -> Awaitable[DocumentDatabase]:
                async def shim() -> DocumentDatabase:
                    return TransientDocumentDatabase()

                return shim()

            def make_json_db(file_path: Path) -> Awaitable[DocumentDatabase]:
                return self._exit_stack.enter_async_context(
                    JSONFileDocumentDatabase(
                        c[Logger],
                        file_path,
                    ),
                )

            if isinstance(self._session_store, SessionStore):
                c[SessionStore] = self._session_store
            else:
                c[SessionStore] = await self._exit_stack.enter_async_context(
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

            c[ServiceRegistry] = await self._exit_stack.enter_async_context(
                ServiceDocumentRegistry(
                    database=TransientDocumentDatabase(),
                    event_emitter_factory=c[EventEmitterFactory],
                    logger=c[Logger],
                    correlator=c[ContextualCorrelator],
                    nlp_services_provider=lambda: {"__nlp__": c[NLPService]},
                    allow_migration=False,
                )
            )

            embedder_factory = EmbedderFactory(c)

            async def get_embedder_type() -> type[Embedder]:
                return type(await c[NLPService].get_embedder())

            c[GlossaryStore] = await self._exit_stack.enter_async_context(
                GlossaryVectorStore(
                    vector_db=TransientVectorDatabase(c[Logger], embedder_factory),
                    document_db=TransientDocumentDatabase(),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

            c[UtteranceStore] = await self._exit_stack.enter_async_context(
                UtteranceVectorStore(
                    vector_db=TransientVectorDatabase(c[Logger], embedder_factory),
                    document_db=TransientDocumentDatabase(),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

            c[CapabilityStore] = await self._exit_stack.enter_async_context(
                CapabilityVectorStore(
                    vector_db=TransientVectorDatabase(c[Logger], embedder_factory),
                    document_db=TransientDocumentDatabase(),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

            c[JourneyStore] = await self._exit_stack.enter_async_context(
                JourneyVectorStore(
                    vector_db=TransientVectorDatabase(c[Logger], embedder_factory),
                    document_db=TransientDocumentDatabase(),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

        async def configure(c: Container) -> Container:
            await override_stores_with_transient_versions(c)

            if self._configure_container:
                c = await self._configure_container(c.clone())

            if self._configure_hooks:
                hooks = await self._configure_hooks(c[EngineHooks])
                c[EngineHooks] = hooks

            return c

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
                name=_INTEGRATED_TOOL_SERVICE_NAME,
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
    "Journey",
    "JourneyId",
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
    "SchematicGenerationResult",
    "SchematicGenerator",
    "Server",
    "SessionId",
    "ServiceRegistry",
    "SessionMode",
    "SessionStatus",
    "StatusEventData",
    "TagId",
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
