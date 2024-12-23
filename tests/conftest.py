# Copyright 2024 Emcie Co Ltd.
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

import asyncio
from contextlib import AsyncExitStack
import os
import tempfile
from typing import Any, AsyncIterator, cast
from fastapi import FastAPI
import httpx
from lagom import Container, Singleton
from pytest import fixture, Config
import pytest

from parlant.adapters.nlp.openai import OpenAIService
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.api.app import create_api_app, ASGIApplication
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from parlant.core.emission.event_publisher import EventPublisherFactory
from parlant.core.emissions import EventEmitterFactory
from parlant.core.customers import CustomerDocumentStore, CustomerStore
from parlant.core.evaluations import (
    EvaluationListener,
    PollingEvaluationListener,
    EvaluationDocumentStore,
    EvaluationStore,
)
from parlant.core.nlp.embedding import EmbedderFactory
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.guideline_connections import (
    GuidelineConnectionDocumentStore,
    GuidelineConnectionStore,
)
from parlant.core.guidelines import GuidelineDocumentStore, GuidelineStore
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.nlp.service import NLPService
from parlant.core.services.tools.service_registry import (
    ServiceDocumentRegistry,
    ServiceRegistry,
)
from parlant.core.sessions import (
    PollingSessionListener,
    SessionDocumentStore,
    SessionListener,
    SessionStore,
)
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.glossary import GlossaryStore, GlossaryVectorStore
from parlant.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionsSchema,
)
from parlant.core.engines.alpha.message_event_generator import (
    MessageEventGenerator,
    MessageEventSchema,
)
from parlant.core.engines.alpha.tool_caller import ToolCallInferenceSchema
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.types import Engine
from parlant.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
)
from parlant.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    ConditionsEntailmentTestsSchema,
    ActionsContradictionTestsSchema,
)
from parlant.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
    GuidelineConnectionPropositionsSchema,
)
from parlant.core.logging import Logger, StdoutLogger
from parlant.core.application import Application
from parlant.core.agents import AgentDocumentStore, AgentStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from parlant.core.tags import TagDocumentStore, TagStore
from parlant.core.tools import LocalToolService

from .test_utilities import (
    CachedSchematicGenerator,
    SyncAwaiter,
    create_schematic_generation_result_collection,
)


@fixture
async def sync_await() -> SyncAwaiter:
    return SyncAwaiter(asyncio.get_event_loop())


@fixture
def test_config(pytestconfig: Config) -> dict[str, Any]:
    return {"patience": 10}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("caching")
    group.addoption(
        "--use-cache",
        action="store_true",
        dest="use_cache",
        help="Whether to use the cache during the current test suite",
    )


@fixture
async def container(request: pytest.FixtureRequest) -> AsyncIterator[Container]:
    container = Container()

    use_cache = bool(request.config.getoption("use_cache", False))

    container[ContextualCorrelator] = Singleton(ContextualCorrelator)
    container[Logger] = StdoutLogger(container[ContextualCorrelator])

    async with AsyncExitStack() as stack:
        schematic_generation_result_collection = (
            await create_schematic_generation_result_collection(stack, logger=container[Logger])
        )

        temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
        os.environ["PARLANT_HOME"] = temp_dir

        container[BackgroundTaskService] = await stack.enter_async_context(
            BackgroundTaskService(container[Logger])
        )

        container[AgentStore] = await stack.enter_async_context(
            AgentDocumentStore(TransientDocumentDatabase())
        )
        container[GuidelineStore] = await stack.enter_async_context(
            GuidelineDocumentStore(TransientDocumentDatabase())
        )
        container[GuidelineConnectionStore] = await stack.enter_async_context(
            GuidelineConnectionDocumentStore(TransientDocumentDatabase())
        )
        container[SessionStore] = await stack.enter_async_context(
            SessionDocumentStore(TransientDocumentDatabase())
        )
        container[ContextVariableStore] = await stack.enter_async_context(
            ContextVariableDocumentStore(TransientDocumentDatabase())
        )
        container[TagStore] = await stack.enter_async_context(
            TagDocumentStore(TransientDocumentDatabase())
        )
        container[CustomerStore] = await stack.enter_async_context(
            CustomerDocumentStore(TransientDocumentDatabase())
        )
        container[GuidelineToolAssociationStore] = await stack.enter_async_context(
            GuidelineToolAssociationDocumentStore(TransientDocumentDatabase())
        )
        container[SessionListener] = PollingSessionListener
        container[EvaluationStore] = await stack.enter_async_context(
            EvaluationDocumentStore(TransientDocumentDatabase())
        )
        container[EvaluationListener] = PollingEvaluationListener
        container[BehavioralChangeEvaluator] = BehavioralChangeEvaluator
        container[EventEmitterFactory] = Singleton(EventPublisherFactory)

        container[ServiceRegistry] = await stack.enter_async_context(
            ServiceDocumentRegistry(
                database=TransientDocumentDatabase(),
                event_emitter_factory=container[EventEmitterFactory],
                correlator=container[ContextualCorrelator],
                nlp_services={"default": OpenAIService(container[Logger])},
            )
        )

        container[NLPService] = await container[ServiceRegistry].read_nlp_service("default")

        embedder_type = type(await container[NLPService].get_embedder())
        embedder_factory = EmbedderFactory(container)
        container[GlossaryStore] = await stack.enter_async_context(
            GlossaryVectorStore(
                await stack.enter_async_context(
                    TransientVectorDatabase(container[Logger], embedder_factory, embedder_type)
                ),
                embedder_factory=embedder_factory,
                embedder_type=embedder_type,
            )
        )

        container[SchematicGenerator[GuidelinePropositionsSchema]] = CachedSchematicGenerator(
            await container[NLPService].get_schematic_generator(GuidelinePropositionsSchema),
            schematic_generation_result_collection,
            use_cache,
        )
        container[SchematicGenerator[MessageEventSchema]] = CachedSchematicGenerator(
            await container[NLPService].get_schematic_generator(MessageEventSchema),
            schematic_generation_result_collection,
            use_cache,
        )
        container[SchematicGenerator[ToolCallInferenceSchema]] = CachedSchematicGenerator(
            await container[NLPService].get_schematic_generator(ToolCallInferenceSchema),
            schematic_generation_result_collection,
            use_cache,
        )
        container[SchematicGenerator[ConditionsEntailmentTestsSchema]] = CachedSchematicGenerator(
            await container[NLPService].get_schematic_generator(ConditionsEntailmentTestsSchema),
            schematic_generation_result_collection,
            use_cache,
        )

        container[SchematicGenerator[ActionsContradictionTestsSchema]] = CachedSchematicGenerator(
            await container[NLPService].get_schematic_generator(ActionsContradictionTestsSchema),
            schematic_generation_result_collection,
            use_cache,
        )
        container[SchematicGenerator[GuidelineConnectionPropositionsSchema]] = (
            CachedSchematicGenerator(
                await container[NLPService].get_schematic_generator(
                    GuidelineConnectionPropositionsSchema
                ),
                schematic_generation_result_collection,
                use_cache,
            )
        )

        container[GuidelineProposer] = Singleton(GuidelineProposer)
        container[GuidelineConnectionProposer] = Singleton(GuidelineConnectionProposer)
        container[CoherenceChecker] = Singleton(CoherenceChecker)

        container[LocalToolService] = cast(
            LocalToolService,
            await container[ServiceRegistry].update_tool_service(
                name="local", kind="local", url=""
            ),
        )

        container[MessageEventGenerator] = Singleton(MessageEventGenerator)
        container[ToolEventGenerator] = Singleton(ToolEventGenerator)

        container[Engine] = AlphaEngine

        container[Application] = Application(container)

        yield container


@fixture
async def api_app(container: Container) -> ASGIApplication:
    return await create_api_app(container)


@fixture
async def async_client(api_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app),
        base_url="http://testserver",
    ) as client:
        yield client


class NoCachedGenerations:
    pass


@fixture
def no_cache(container: Container) -> None:
    if isinstance(
        container[SchematicGenerator[GuidelinePropositionsSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GuidelinePropositionsSchema],
            container[SchematicGenerator[GuidelinePropositionsSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[MessageEventSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[MessageEventSchema],
            container[SchematicGenerator[MessageEventSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[ToolCallInferenceSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[ToolCallInferenceSchema],
            container[SchematicGenerator[ToolCallInferenceSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[ConditionsEntailmentTestsSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[ConditionsEntailmentTestsSchema],
            container[SchematicGenerator[ConditionsEntailmentTestsSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[ActionsContradictionTestsSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[ActionsContradictionTestsSchema],
            container[SchematicGenerator[ActionsContradictionTestsSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[GuidelineConnectionPropositionsSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GuidelineConnectionPropositionsSchema],
            container[SchematicGenerator[GuidelineConnectionPropositionsSchema]],
        ).use_cache = False
