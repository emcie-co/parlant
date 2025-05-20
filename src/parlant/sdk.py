from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Awaitable, Callable, Iterable, Literal, Sequence, cast
from lagom import Container

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
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
from parlant.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
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
from parlant.core.persistence.document_database import DocumentDatabase
from parlant.core.relationships import (
    GuidelineRelationshipKind,
    RelationshipDocumentStore,
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipId,
    RelationshipStore,
)
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
from parlant.core.utterances import UtteranceDocumentStore, UtteranceId, UtteranceStore
from parlant.core.evaluations import EvaluationDocumentStore, EvaluationStore
from parlant.core.guidelines import GuidelineDocumentStore, GuidelineId, GuidelineStore
from parlant.core.journeys import JourneyDocumentStore, JourneyId, JourneyStore
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

_INTEGRATED_TOOL_SERVICE_NAME = "built-in"


def _load_openai(container: Container) -> NLPService:
    return OpenAIService(container[Logger])


class _PicoAgentStore(AgentStore):
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


@dataclass
class Guideline:
    id: GuidelineId
    condition: str
    action: str | None
    tags: Sequence[TagId]

    _parlant: Server
    _container: Container

    async def prioritize_over(self, guideline: Guideline) -> RelationshipId:
        return await self._create_relationship(
            guideline=guideline,
            kind=GuidelineRelationshipKind.PRIORITY,
            direction="source",
        )

    async def entail(self, guideline: Guideline) -> RelationshipId:
        return await self._create_relationship(
            guideline=guideline,
            kind=GuidelineRelationshipKind.ENTAILMENT,
            direction="source",
        )

    async def depend_on(self, guideline: Guideline) -> RelationshipId:
        return await self._create_relationship(
            guideline=guideline,
            kind=GuidelineRelationshipKind.DEPENDENCY,
            direction="source",
        )

    async def _create_relationship(
        self,
        guideline: Guideline,
        kind: GuidelineRelationshipKind,
        direction: Literal["source", "target"],
    ) -> RelationshipId:
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

        return relationship.id


@dataclass
class Journey:
    id: JourneyId
    description: str
    conditions: list[str]

    _parlant: Server
    _container: Container

    async def create_guideline(
        self,
        condition: str,
        action: str,
        tools: Iterable[ToolEntry] = [],
    ) -> Guideline:
        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=action,
            tags=[Tag.for_journey_id(self.id)],
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
            tags=[Tag.for_journey_id(self.id)],
        )

        await self._container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId(service_name=_INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name),
        )

        return guideline.id


@dataclass
class Agent:
    id: AgentId
    name: str
    description: str | None
    max_engine_iterations: int
    composition_mode: CompositionMode

    _parlant: Server
    _container: Container

    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: list[str],
    ) -> Journey:
        journey = await self._parlant.create_journey(title, description, conditions)
        await self.attach_journey(journey)
        return Journey(
            id=journey.id,
            description=description,
            conditions=conditions,
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
        action: str,
        tools: Iterable[ToolEntry] = [],
    ) -> Guideline:
        guideline = await self._container[GuidelineStore].create_guideline(
            condition=condition,
            action=action,
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
    ) -> UtteranceId:
        utterance = await self._container[UtteranceStore].create_utterance(
            value=template,
            tags=tags,
            fields=[],
        )

        return utterance.id


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
        composition_mode: CompositionMode | None = None,
        max_engine_iterations: int | None = None,
    ) -> Agent:
        agent = await self._container[AgentStore].create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations or 1,
            composition_mode=composition_mode or CompositionMode.FLUID,
        )

        return Agent(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=agent.composition_mode,
            _parlant=self,
            _container=self._container,
        )

    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: list[str],
    ) -> Journey:
        condition_ids = []

        for c in conditions:
            condition_ids.append(
                (await self._container[GuidelineStore].create_guideline(condition=c)).id
            )

        journey = await self._container[JourneyStore].create_journey(
            title,
            description,
            condition_ids,
        )

        return Journey(
            id=journey.id,
            description=description,
            conditions=conditions,
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
                (UtteranceStore, UtteranceDocumentStore),
                (GuidelineStore, GuidelineDocumentStore),
                (GuidelineToolAssociationStore, GuidelineToolAssociationDocumentStore),
                (JourneyStore, JourneyDocumentStore),
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
    "GuidelineId",
    "Journey",
    "JourneyId",
    "LoadedContext",
    "LogLevel",
    "Logger",
    "MessageEventData",
    "NLPService",
    "PluginServer",
    "RelationshipId",
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
