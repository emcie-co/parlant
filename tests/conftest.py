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

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import os
from typing import Any, AsyncIterator, Iterator, Mapping, cast
from fastapi import FastAPI
import httpx
from parlant.core.container import Container, Singleton
from pytest import fixture, Config
import pytest

from parlant.adapters.db.sqlite import SQLiteDocumentDatabase
from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.adapters.nlp.anthropic_service import register_anthropic_models
from parlant.adapters.nlp.gemini_service import (
    RoutedGeminiSchematicGenerator,
    register_gemini_models,
)
from parlant.adapters.nlp.openai_service import register_openai_models
from parlant.adapters.nlp.openrouter_service import register_openrouter_models
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.api.app import create_api_app, ASGIApplication
from parlant.api.authorization import AuthorizationPolicy, DevelopmentAuthorizationPolicy

from parlant.core.app_modules.request_context import RequestContext
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.store_provider import BasicStoreProvider, StoreProvider
from parlant.core.application_context import ApplicationContext
from parlant.core.health import HealthReporter, HealthThresholds, NullHealthReporter
from parlant.core.common import IdGenerator
from parlant.core.meter import Meter, LocalMeter
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.tracer import LocalTracer, Tracer
from parlant.core.cost_control import CompositeCostControlPolicy, CostControlPolicy
from parlant.core.engines.compass.context import CompassDependencies
from parlant.core.engines.compass.matcher import Matcher as CompassMatcher
from parlant.core.engines.compass.matching.rule_discovery import RuleDiscoverer
from parlant.core.engines.compass.responder.responder import Responder as CompassResponder
from parlant.core.engines.compass.reviewer import BasicReviewer, ReviewSchema
from parlant.core.engines.compass.reviewer import Reviewer as CompassReviewer
from parlant.core.engines.compass.matching.rule_evaluation import TurnEvaluator
from parlant.core.engines.compass.matching.rule_recaller import RuleRecaller
from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.engines.compass.matching.tool_selection import ToolSelector
from parlant.core.engines.compass.matching.tool_recaller import ToolRecaller
from parlant.core.engines.compass.matching.evaluation_scheduler import (
    CrossReferenceEvaluationPreconditionProvider,
    DependencyEvaluationPreconditionProvider,
    EvaluationPreconditionRegistry,
    EvaluationScheduler,
    EvaluationSchedulingPolicy,
)
from parlant.core.usage_reporter import UsageReporter
from parlant.core.pricing import PricingRegistry, SessionBudgetService
from parlant.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from parlant.core.emission.event_buffer import EventBufferFactory
from parlant.core.emission.event_publisher import EventPublisherFactory
from parlant.core.emission.session_tap import SessionEventTap
from parlant.core.emissions import EventEmitterFactory
from parlant.core.customers import CustomerDocumentStore, CustomerStore
from parlant.core.engines.compass.matching.rule_ranker import (
    RuleRankSchema,
)
from parlant.core.engines.compass.hooks import CompassHooks
from parlant.core.event_loop_monitor import EventLoopMonitor
from parlant.core.indexing import (
    IndexingDocumentStore,
    IndexingListener,
    IndexingStore,
    PollingIndexingListener,
)
from parlant.core.services.indexing.rule_relativity_proposer import (
    RuleRelativityProposer,
    RuleRelativityPropositionSchema,
)
from parlant.core.services.indexing.rule_signal_proposer import (
    RuleSignalProposer,
    RuleSignalPropositionSchema,
)
from parlant.core.services.indexing.rule_title_proposer import (
    RuleTitleProposer,
    RuleTitlePropositionSchema,
)
from parlant.core.canned_responses import CannedResponseStore, CannedResponseVectorStore
from parlant.core.nlp.embedding import (
    BasicEmbeddingCache,
    Embedder,
    EmbeddingCache,
    NullEmbeddingCache,
)
from parlant.core.nlp.generation import T, SchematicGenerator
from parlant.core.relationships import (
    RelationshipDocumentStore,
    RelationshipStore,
)
from parlant.core.rules import RuleStore, RuleVectorStore
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.nlp.prompt_routing import PromptConfigurationService
from parlant.core.nlp.model_routing import (
    ModelConfig,
    ModelConfigurationService,
    ModelImplementationRegistry,
    ModelRouter,
    ModelScope,
    NullSecretLoader,
    RuntimeModelRouteSource,
    SecretLoader,
    TransientModelRouteStore,
)
from parlant.core.nlp.rate_limits import RateLimiter, TransientRateLimiter
from parlant.core.persistence.data_collection import DataCollectingSchematicGenerator
from parlant.core.persistence.document_database import DocumentCollection
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
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.compass.variable_loader import VariableLoader
from parlant.core.glossary import GlossaryStore, GlossaryVectorStore

from parlant.core.engines.types import EngineRegistry
from parlant.core.services.indexing.indexing_service import RuleIndexingProcessor


from parlant.core.loggers import LogLevel, Logger, StdoutLogger
from parlant.core.application import Application
from parlant.core.agents import AgentDocumentStore, AgentStore, Effort
from parlant.core.rule_tool_associations import (
    RuleToolAssociationDocumentStore,
    RuleToolAssociationStore,
)
from parlant.core.entity_cq import EntityQueries, EntityCommands
from parlant.core.groups import GroupDocumentStore, GroupStore
from parlant.core.tools import LocalToolService

from .test_utilities import (
    GLOBAL_EMBEDDER_CACHE_FILE,
    CachedSchematicGenerator,
    JournalingEngineHooks,
    SchematicGenerationResultDocument,
    SyncAwaiter,
    create_schematic_generation_result_collection,
    get_random_port,
)


@fixture(scope="session")
def mongo_server_uri() -> Iterator[str]:
    """A base URI for a live MongoDB reachable by tests.

    Uses TEST_MONGO_SERVER when it points at a reachable server; otherwise
    spins up an in-memory mongod for the session and tears it down after.
    """
    configured = os.environ.get("TEST_MONGO_SERVER")

    if configured:
        import pymongo
        from pymongo.errors import PyMongoError

        client: pymongo.MongoClient[Any] = pymongo.MongoClient(
            configured, serverSelectionTimeoutMS=2000
        )
        try:
            client.admin.command("ping")
            yield configured
            return
        except PyMongoError:
            pass  # fall through to the in-memory server
        finally:
            client.close()

    os.environ["PYMONGOIM__MONGOD_PORT"] = str(get_random_port())

    from pymongo_inmemory import Mongod  # type: ignore[import-untyped]
    from pymongo_inmemory.context import Context  # type: ignore[import-untyped]

    with Mongod(Context()) as mongod:
        # connection_string carries a default database path; yield the bare server URI.
        yield str(mongod.connection_string).rsplit("/", 1)[0]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("caching")

    group.addoption(
        "--no-cache",
        action="store_true",
        dest="no_cache",
        default=False,
        help="Whether to avoid using the cache during the current test suite",
    )


@fixture
def tracer(request: pytest.FixtureRequest) -> Iterator[Tracer]:
    tracer = LocalTracer()

    with tracer.attributes({"scope": request.node.name}):
        yield tracer


@fixture
def logger(tracer: Tracer) -> Logger:
    return StdoutLogger(tracer=tracer, log_level=LogLevel.INFO)


@dataclass(frozen=True)
class CacheOptions:
    cache_enabled: bool
    cache_schematic_generation_collection: (
        DocumentCollection[SchematicGenerationResultDocument] | None
    )


@fixture
async def cache_options(
    request: pytest.FixtureRequest,
    logger: Logger,
) -> AsyncIterator[CacheOptions]:
    if not request.config.getoption("no_cache", True):
        logger.warning("*** Cache is enabled")

        async with (
            create_schematic_generation_result_collection(logger=logger) as schematic_collection,
        ):
            yield CacheOptions(
                cache_enabled=True,
                cache_schematic_generation_collection=schematic_collection,
            )

    else:
        yield CacheOptions(
            cache_enabled=False,
            cache_schematic_generation_collection=None,
        )


@fixture
async def sync_await() -> SyncAwaiter:
    return SyncAwaiter(asyncio.get_running_loop())


@fixture
def test_config(pytestconfig: Config) -> dict[str, Any]:
    return {"patience": 10}


async def make_schematic_generator(
    container: Container,
    cache_options: CacheOptions,
    schema: type[T],
) -> SchematicGenerator[T]:
    generator = await container[ModelRouter].get_schematic_generator(
        context=None,
        route="test",
        schema=schema,
    )

    if cache_options.cache_enabled:
        assert cache_options.cache_schematic_generation_collection

        generator = CachedSchematicGenerator[schema](  # type: ignore
            base_generator=generator,
            collection=cache_options.cache_schematic_generation_collection,
            use_cache=True,
        )

    if os.environ.get("PARLANT_DATA_COLLECTION", "false").lower() not in ["false", "no", "0"]:
        generator = DataCollectingSchematicGenerator[schema](  # type: ignore
            generator,
            container[Tracer],
        )

    return generator


@fixture
def compass_engine_bindings() -> dict[type, type]:
    """The port implementations the test container binds for the compass engine.

    Defaults to the open-source implementations, so the base test suite proves
    the OSS composition stands on its own. A plugin test package may override
    this fixture in its own conftest to bind its own implementations — the
    single place that decides which side is exercised, which is why this file
    names none of them."""
    return {
        CompassMatcher: CompassMatcher,
        CompassReviewer: BasicReviewer,
        CompassResponder: CompassResponder,
    }


@fixture
def extra_schematic_schemas() -> tuple[type, ...]:
    """Additional schemas to register schematic generators for. Empty here; a
    plugin test package adds whatever schemas its own evaluators need."""
    return ()


@fixture
async def container(
    tracer: Tracer,
    logger: Logger,
    cache_options: CacheOptions,
    compass_engine_bindings: dict[type, type],
    extra_schematic_schemas: tuple[type, ...],
) -> AsyncIterator[Container]:
    container = Container()

    container[Tracer] = tracer
    container[Logger] = logger
    container[Meter] = Singleton(LocalMeter)
    container[WebSocketLogger] = WebSocketLogger(container[Tracer])

    container[IdGenerator] = Singleton(IdGenerator)

    async with AsyncExitStack() as stack:
        container[BackgroundTaskService] = await stack.enter_async_context(
            BackgroundTaskService(container[Logger])
        )

        await container[BackgroundTaskService].start(
            container[WebSocketLogger].start(), tag="websocket-logger"
        )

        container[EventLoopMonitor] = await stack.enter_async_context(EventLoopMonitor())

        container[ApplicationContext] = ApplicationContext(instance_id="test-instance")
        container[HealthReporter] = NullHealthReporter(container[ApplicationContext])
        container[HealthThresholds] = HealthThresholds()
        container[UsageReporter] = Singleton(UsageReporter)
        container[PricingRegistry] = Singleton(PricingRegistry)
        route_store = TransientModelRouteStore()
        runtime_routes = RuntimeModelRouteSource()
        secret_loader = NullSecretLoader()
        model_implementations = ModelImplementationRegistry(secret_loader)

        async def provide_test_generator(
            model: str,
            schema: type[Any],
            raw_args: Mapping[str, Any],
        ) -> SchematicGenerator[Any]:
            assert not raw_args
            if model == "container":
                return container[SchematicGenerator[schema]]  # type: ignore[valid-type]

            assert model == "bare"
            generator_type = RoutedGeminiSchematicGenerator[schema]  # type: ignore[valid-type]
            return generator_type(
                model_name="gemini-3.6-flash",
                logger=container[Logger],
                tracer=container[Tracer],
                meter=container[Meter],
                usage_reporter=container[UsageReporter],
            )

        model_implementations.register_schematic(
            "test",
            provide_test_generator,
            default="bare",
        )
        model_implementations.register_provider_loader(
            "openai",
            lambda: register_openai_models(model_implementations, container),
        )
        model_implementations.register_provider_loader(
            "gemini",
            lambda: register_gemini_models(model_implementations, container),
        )
        model_implementations.register_provider_loader(
            "anthropic",
            lambda: register_anthropic_models(model_implementations, container),
        )
        model_implementations.register_provider_loader(
            "openrouter",
            lambda: register_openrouter_models(model_implementations, container),
        )
        model_configuration = ModelConfigurationService(
            route_store,
            runtime_routes,
            model_implementations,
        )
        rate_limiter = TransientRateLimiter()
        model_router = ModelRouter(
            route_store,
            runtime_routes,
            model_implementations,
            rate_limiter,
        )
        container[TransientModelRouteStore] = route_store
        container[RuntimeModelRouteSource] = runtime_routes
        container[SecretLoader] = secret_loader
        container[ModelImplementationRegistry] = model_implementations
        container[ModelConfigurationService] = model_configuration
        container[PromptConfigurationService] = PromptConfigurationService()
        container[ModelRouter] = model_router
        container[RateLimiter] = rate_limiter
        await model_configuration.configure(
            ModelScope.server(),
            {"*": "gemini", "test": ModelConfig("test:bare")},
        )
        container[TurnEvaluator] = Singleton(RuleRanker)
        container[DependencyEvaluationPreconditionProvider] = Singleton(
            DependencyEvaluationPreconditionProvider
        )
        container[CrossReferenceEvaluationPreconditionProvider] = Singleton(
            CrossReferenceEvaluationPreconditionProvider
        )
        container[EvaluationPreconditionRegistry] = Singleton(
            lambda c: EvaluationPreconditionRegistry(
                [
                    c[DependencyEvaluationPreconditionProvider],
                    c[CrossReferenceEvaluationPreconditionProvider],
                ]
            )
        )
        container[EvaluationSchedulingPolicy] = EvaluationSchedulingPolicy()
        container[CompassDependencies] = Singleton(CompassDependencies)
        container[EvaluationScheduler] = Singleton(EvaluationScheduler)
        # Alias, not a second registration: retrain and discovery share the frame.
        container[RuleRecaller] = Singleton(RuleRecaller)
        container[RuleDiscoverer] = lambda c: c[RuleRecaller]
        container[ToolSelector] = Singleton(ToolRecaller)

        for interface, implementation in compass_engine_bindings.items():
            container[interface] = Singleton(implementation)
        # SDK-created circuit breakers are added to the composite at runtime.
        container[CostControlPolicy] = CompositeCostControlPolicy()

        container[AgentStore] = await stack.enter_async_context(
            AgentDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[RelationshipStore] = await stack.enter_async_context(
            RelationshipDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[SessionStore] = await stack.enter_async_context(
            SessionDocumentStore(TransientDocumentDatabase())
        )
        container[SessionBudgetService] = Singleton(SessionBudgetService)
        container[ContextVariableStore] = await stack.enter_async_context(
            ContextVariableDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[GroupStore] = await stack.enter_async_context(
            GroupDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[CustomerStore] = await stack.enter_async_context(
            CustomerDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[RuleToolAssociationStore] = await stack.enter_async_context(
            RuleToolAssociationDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[SessionListener] = PollingSessionListener(
            store_provider_factory=lambda: container[StoreProvider]
        )
        container[IndexingStore] = await stack.enter_async_context(
            IndexingDocumentStore(TransientDocumentDatabase())
        )
        container[IndexingListener] = PollingIndexingListener
        container[SessionEventTap] = SessionEventTap()
        container[EventEmitterFactory] = EventPublisherFactory(
            store_provider_factory=lambda: container[StoreProvider],
            tap=container[SessionEventTap],
        )
        container[EventBufferFactory] = EventBufferFactory(
            store_provider_factory=lambda: container[StoreProvider],
        )

        container[ServiceRegistry] = await stack.enter_async_context(
            ServiceDocumentRegistry(
                database=TransientDocumentDatabase(),
                event_emitter_factory=container[EventEmitterFactory],
                logger=container[Logger],
                tracer=container[Tracer],
            )
        )

        container[EstimatingTokenizer] = ZeroEstimatingTokenizer()

        if cache_options.cache_enabled:
            embedding_cache: EmbeddingCache = BasicEmbeddingCache(
                document_database=await stack.enter_async_context(
                    SQLiteDocumentDatabase(
                        logger,
                        GLOBAL_EMBEDDER_CACHE_FILE.with_suffix(".sqlite"),
                    ),
                )
            )
        else:
            embedding_cache = NullEmbeddingCache()

        container[EmbeddingCache] = embedding_cache

        async def get_embedder(route: str) -> Embedder:
            return await container[ModelRouter].get_embedder(context=None, route=route)

        container[GlossaryStore] = await stack.enter_async_context(
            GlossaryVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    container[Tracer],
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_provider=lambda: get_embedder("embedding.glossary"),
            )
        )

        container[CannedResponseStore] = await stack.enter_async_context(
            CannedResponseVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger], container[Tracer], lambda: embedding_cache
                ),
                document_db=TransientDocumentDatabase(),
                embedder_provider=lambda: get_embedder("embedding.canned-responses"),
            )
        )

        container[RuleStore] = await stack.enter_async_context(
            RuleVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    container[Tracer],
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_provider=lambda: get_embedder("embedding.rules"),
            )
        )

        container[StoreProvider] = BasicStoreProvider(lambda: container)

        container[EntityQueries] = Singleton(EntityQueries)
        container[EntityCommands] = Singleton(EntityCommands)

        for generation_schema in (
            RuleRelativityPropositionSchema,
            RuleSignalPropositionSchema,
            RuleTitlePropositionSchema,
            RuleRankSchema,
            ReviewSchema,
            *extra_schematic_schemas,
        ):
            container[SchematicGenerator[generation_schema]] = await make_schematic_generator(  # type: ignore
                container,
                cache_options,
                generation_schema,
            )

        async def provide_rule_ranker(_: object) -> SchematicGenerator[RuleRankSchema]:
            return container[SchematicGenerator[RuleRankSchema]]

        async def provide_basic_reviewer(
            _: object,
        ) -> SchematicGenerator[ReviewSchema]:
            return container[SchematicGenerator[ReviewSchema]]

        extra_generators = {
            schema.__name__: container[SchematicGenerator[schema]]  # type: ignore[valid-type]
            for schema in extra_schematic_schemas
        }

        async def provide_distiller(context: Any) -> SchematicGenerator[Any]:
            schema_name = (
                "HighEffortRuleDistillationSchema"
                if context.state.dynamic_effort_level >= Effort.MEDIUM
                else "LowEffortRuleDistillationSchema"
            )
            return extra_generators[schema_name]

        async def provide_high_effort_reviewer(_: object) -> SchematicGenerator[Any]:
            return extra_generators["HighEffortReviewSchema"]

        async def provide_high_effort_message_reviewer(
            _: object,
        ) -> SchematicGenerator[Any]:
            return extra_generators["HighEffortMessageReviewSchema"]

        await model_configuration.configure(
            ModelScope.server(),
            {
                "indexing": ModelConfig("test:container"),
                "matching.rank": ModelConfig(provide_rule_ranker),
                # Calibration hook: with PARLANT_TEST_DECISION_RANKER set to a
                # provider:model, the ranker's cheap efforts go through a decision
                # model instead of the schematic generator, so the existing
                # ranking suites double as an eval for it.
                **(
                    {
                        "matching.rank[capability=decision]": ModelConfig(
                            os.environ["PARLANT_TEST_DECISION_RANKER"]
                        )
                    }
                    if os.environ.get("PARLANT_TEST_DECISION_RANKER")
                    else {}
                ),
                "review.tools": ModelConfig(provide_basic_reviewer),
                "review.message": ModelConfig(provide_basic_reviewer),
                **(
                    {"matching.distill": ModelConfig(provide_distiller)}
                    if "LowEffortRuleDistillationSchema" in extra_generators
                    else {}
                ),
                **(
                    {
                        "review.tools[effort>=high]": ModelConfig(provide_high_effort_reviewer),
                        "review.message[effort>=high]": ModelConfig(
                            provide_high_effort_message_reviewer
                        ),
                    }
                    if "HighEffortReviewSchema" in extra_generators
                    else {}
                ),
            },
        )

        container[RuleRelativityProposer] = Singleton(RuleRelativityProposer)
        container[RuleSignalProposer] = Singleton(RuleSignalProposer)
        container[RuleTitleProposer] = Singleton(RuleTitleProposer)
        container[LocalToolService] = cast(
            LocalToolService,
            await container[ServiceRegistry].update_tool_service(
                name="local", kind="local", url=""
            ),
        )
        container[RuleIndexingProcessor] = Singleton(RuleIndexingProcessor)

        hooks = JournalingEngineHooks()
        container[JournalingEngineHooks] = hooks
        container[CompassHooks] = hooks

        container[AuthorizationPolicy] = Singleton(DevelopmentAuthorizationPolicy)

        container[VariableLoader] = Singleton(VariableLoader)
        container[CompassEngine] = Singleton(CompassEngine)
        container[EngineRegistry] = EngineRegistry(
            {
                "compass": lambda: container[CompassEngine],
            }
        )

        container[Application] = Singleton(Application)
        container[RequestContext] = Singleton(RequestContext)

        yield container

        await container[BackgroundTaskService].cancel_all()


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
        container[SchematicGenerator[RuleRankSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[RuleRankSchema],
            container[SchematicGenerator[RuleRankSchema]],
        ).use_cache = False
