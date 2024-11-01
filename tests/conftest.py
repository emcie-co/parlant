import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
import tempfile
from typing import Any, AsyncIterator, cast
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from lagom import Container, Singleton
from pytest import fixture, Config

from emcie.server.adapters.db.chroma.glossary import GlossaryChromaStore
from emcie.server.adapters.nlp.openai import (
    GPT_4o,
    GPT_4o_Mini,
    OmniModeration,
    OpenAITextEmbedding3Large,
)
from emcie.server.api.app import create_app
from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from emcie.server.core.emission.event_publisher import EventPublisherFactory
from emcie.server.core.emissions import EventEmitterFactory
from emcie.server.core.end_users import EndUserDocumentStore, EndUserStore
from emcie.server.core.evaluations import EvaluationDocumentStore, EvaluationStore
from emcie.server.core.nlp.embedding import EmbedderFactory
from emcie.server.core.nlp.generation import FallbackSchematicGenerator, SchematicGenerator
from emcie.server.core.guideline_connections import (
    GuidelineConnectionDocumentStore,
    GuidelineConnectionStore,
)
from emcie.server.core.guidelines import GuidelineDocumentStore, GuidelineStore
from emcie.server.adapters.db.chroma.database import ChromaDatabase
from emcie.server.adapters.db.transient import TransientDocumentDatabase
from emcie.server.core.services.tools.service_registry import (
    ServiceDocumentRegistry,
    ServiceRegistry,
)
from emcie.server.core.sessions import (
    PollingSessionListener,
    SessionDocumentStore,
    SessionListener,
    SessionStore,
)
from emcie.server.core.engines.alpha.engine import AlphaEngine
from emcie.server.core.glossary import GlossaryStore
from emcie.server.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionsSchema,
)
from emcie.server.core.engines.alpha.message_event_producer import (
    MessageEventProducer,
    MessageEventSchema,
)
from emcie.server.core.engines.alpha.tool_caller import ToolCallInferenceSchema
from emcie.server.core.engines.alpha.tool_event_producer import ToolEventProducer
from emcie.server.core.engines.types import Engine
from emcie.server.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
)
from emcie.server.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    PredicatesEntailmentTestsSchema,
    ActionsContradictionTestsSchema,
)
from emcie.server.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
    GuidelineConnectionPropositionsSchema,
)
from emcie.server.core.logging import Logger, StdoutLogger
from emcie.server.core.mc import MC
from emcie.server.core.agents import AgentDocumentStore, AgentStore
from emcie.server.core.persistence.document_database import (
    DocumentDatabase,
)
from emcie.server.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from emcie.server.core.tools import LocalToolService

from .test_utilities import SyncAwaiter


@fixture
async def sync_await() -> SyncAwaiter:
    return SyncAwaiter(asyncio.get_event_loop())


@fixture
def test_config(pytestconfig: Config) -> dict[str, Any]:
    return {"patience": 10}


@fixture
async def container() -> AsyncIterator[Container]:
    container = Container()

    container[ContextualCorrelator] = Singleton(ContextualCorrelator)
    container[Logger] = StdoutLogger(container[ContextualCorrelator])

    container[SchematicGenerator[GuidelinePropositionsSchema]] = GPT_4o[
        GuidelinePropositionsSchema
    ](logger=container[Logger])

    container[SchematicGenerator[MessageEventSchema]] = GPT_4o[MessageEventSchema](
        logger=container[Logger]
    )
    container[SchematicGenerator[ToolCallInferenceSchema]] = FallbackSchematicGenerator(
        GPT_4o_Mini[ToolCallInferenceSchema](logger=container[Logger]),
        GPT_4o[ToolCallInferenceSchema](logger=container[Logger]),
        logger=container[Logger],
    )
    container[SchematicGenerator[PredicatesEntailmentTestsSchema]] = GPT_4o[
        PredicatesEntailmentTestsSchema
    ](logger=container[Logger])
    container[SchematicGenerator[ActionsContradictionTestsSchema]] = GPT_4o[
        ActionsContradictionTestsSchema
    ](logger=container[Logger])
    container[SchematicGenerator[GuidelineConnectionPropositionsSchema]] = GPT_4o[
        GuidelineConnectionPropositionsSchema
    ](logger=container[Logger])

    container[DocumentDatabase] = TransientDocumentDatabase
    container[AgentStore] = Singleton(AgentDocumentStore)
    container[GuidelineStore] = Singleton(GuidelineDocumentStore)
    container[GuidelineProposer] = Singleton(GuidelineProposer)
    container[GuidelineConnectionStore] = Singleton(GuidelineConnectionDocumentStore)
    container[GuidelineConnectionProposer] = Singleton(GuidelineConnectionProposer)
    container[CoherenceChecker] = Singleton(CoherenceChecker)

    container[SessionStore] = Singleton(SessionDocumentStore)
    container[ContextVariableStore] = Singleton(ContextVariableDocumentStore)
    container[EndUserStore] = Singleton(EndUserDocumentStore)
    container[GuidelineToolAssociationStore] = Singleton(GuidelineToolAssociationDocumentStore)
    container[SessionListener] = PollingSessionListener
    container[EvaluationStore] = Singleton(EvaluationDocumentStore)
    container[BehavioralChangeEvaluator] = BehavioralChangeEvaluator
    container[EventEmitterFactory] = Singleton(EventPublisherFactory)

    async with AsyncExitStack() as stack:
        chroma_temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
        container[GlossaryStore] = GlossaryChromaStore(
            ChromaDatabase(container[Logger], Path(chroma_temp_dir), EmbedderFactory(container)),
            embedder_type=OpenAITextEmbedding3Large,
        )

        container[ServiceRegistry] = await stack.enter_async_context(
            ServiceDocumentRegistry(
                database=container[DocumentDatabase],
                event_emitter_factory=container[EventEmitterFactory],
                correlator=container[ContextualCorrelator],
                moderation_services={"openai": OmniModeration(logger=container[Logger])},
            )
        )
        container[LocalToolService] = cast(
            LocalToolService,
            await container[ServiceRegistry].update_tool_service(
                name="local", kind="local", url=""
            ),
        )

        container[MessageEventProducer] = Singleton(MessageEventProducer)
        container[ToolEventProducer] = Singleton(ToolEventProducer)

        container[Engine] = AlphaEngine

        container[MC] = await stack.enter_async_context(MC(container))

        yield container


@fixture
async def api_app(container: Container) -> FastAPI:
    return await create_app(container)


@fixture
async def client(api_app: FastAPI) -> AsyncIterator[TestClient]:
    with TestClient(api_app) as client:
        yield client


@fixture
async def async_client(api_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app),
        base_url="http://testserver",
    ) as client:
        yield client
