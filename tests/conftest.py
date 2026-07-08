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
from typing import Any, AsyncIterator, Iterator, cast
from fastapi import FastAPI
import httpx
from lagom import Container, Singleton
from pytest import fixture, Config
import pytest

from parlant.adapters.db.sqlite import SQLiteDocumentDatabase
from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.adapters.nlp.gemini_service import GeminiService
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.api.app import create_api_app, ASGIApplication
from parlant.api.authorization import AuthorizationPolicy, DevelopmentAuthorizationPolicy

from parlant.core.app_modules.request_context import RequestContext
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.store_provider import BasicStoreProvider, StoreProvider
from parlant.core.capabilities import CapabilityStore, CapabilityVectorStore
from parlant.core.application_context import ApplicationContext
from parlant.core.health import HealthReporter, NullHealthReporter
from parlant.core.common import IdGenerator
from parlant.core.engines.alpha.rule_matching.generic.rule_low_criticality_batch import (
    GenericLowCriticalityRuleMatchesSchema,
    GenericLowCriticalityRuleMatching,
)
from parlant.core.engines.alpha.rule_matching.generic.journey.journey_backtrack_check import (
    JourneyBacktrackCheckSchema,
)
from parlant.core.engines.alpha.rule_matching.generic.journey.journey_backtrack_node_selection import (
    JourneyBacktrackNodeSelectionSchema,
)
from parlant.core.engines.alpha.rule_matching.generic.journey.journey_next_step_selection import (
    JourneyNextStepSelectionSchema,
)
from parlant.core.meter import Meter, LocalMeter
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.services.indexing.journey_reachable_nodes_evaluation import (
    ReachableNodesEvaluationSchema,
)
from parlant.core.tracer import LocalTracer, Tracer
from parlant.core.cost_control import AdvisoryCostControlPolicy, CostControlPolicy
from parlant.core.engines.compass.matcher import Matcher as CompassMatcher
from parlant.core.engines.compass.matching.rule_discovery import RuleDiscoverer
from parlant.core.engines.compass.reviewer import BasicReviewer, LowEffortReviewSchema
from parlant.core.engines.compass.reviewer import Reviewer as CompassReviewer
from parlant.core.engines.compass.matching.rule_evaluation import TurnEvaluator
from parlant.core.engines.compass.matching.rule_recaller import RuleRecaller
from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.usage_reporter import UsageReporter
from parlant.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from parlant.core.emission.event_publisher import EventPublisherFactory
from parlant.core.emissions import EventEmitterFactory
from parlant.core.customers import CustomerDocumentStore, CustomerStore
from parlant.core.engines.alpha.rule_matching.generic import (
    observational_batch,
)
from parlant.core.engines.alpha.rule_matching.generic import (
    rule_previously_applied_actionable_batch,
)
from parlant.core.engines.alpha.rule_matching.generic import (
    rule_actionable_batch,
)
from parlant.core.engines.alpha.rule_matching.generic import (
    rule_previously_applied_actionable_customer_dependent_batch,
)
from parlant.core.engines.alpha.rule_matching.generic import (
    response_analysis_batch,
)
from parlant.core.engines.alpha.rule_matching.generic.disambiguation_batch import (
    DisambiguationRuleMatchesSchema,
)
from parlant.core.engines.alpha.rule_matching.generic_rule_matching_strategy_resolver import (
    GenericRuleMatchingStrategyResolver,
)
from parlant.core.engines.alpha.optimization_policy import (
    BasicOptimizationPolicy,
    OptimizationPolicy,
)
from parlant.core.engines.alpha.perceived_performance_policy import (
    NullPerceivedPerformancePolicy,
    PerceivedPerformancePolicy,
)
from parlant.core.engines.alpha.rule_matching.generic.rule_previously_applied_actionable_customer_dependent_batch import (
    GenericPreviouslyAppliedActionableCustomerDependentRuleMatchesSchema,
    GenericPreviouslyAppliedActionableCustomerDependentRuleMatching,
    GenericPreviouslyAppliedActionableCustomerDependentRuleMatchingShot,
)
from parlant.core.engines.alpha.rule_matching.generic.rule_actionable_batch import (
    GenericActionableRuleMatchesSchema,
    GenericActionableRuleMatching,
    GenericActionableRuleRuleMatchingShot,
)
from parlant.core.engines.compass.matching.rule_ranker import (
    RuleRankSchema,
)
from parlant.core.engines.alpha.rule_matching.generic.rule_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableRuleMatchesSchema,
    GenericPreviouslyAppliedActionableRuleMatching,
    GenericPreviouslyAppliedActionableRuleRuleMatchingShot,
)
from parlant.core.engines.alpha.tool_calling import overlapping_tools_batch, single_tool_batch
from parlant.core.engines.alpha.rule_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
    GenericResponseAnalysisShot,
)
from parlant.core.engines.alpha import message_generator
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.alpha.planners import NullPlanner, PlannerProvider
from parlant.core.engines.alpha.relational_resolver import RelationalResolver
from parlant.core.event_loop_monitor import EventLoopMonitor
from parlant.core.engines.alpha.tool_calling.default_tool_call_batcher import DefaultToolCallBatcher
from parlant.core.engines.alpha.canned_response_generator import (
    CannedResponseDraftSchema,
    CannedResponseFieldExtractionSchema,
    CannedResponseFieldExtractor,
    CannedResponsePreambleSchema,
    CannedResponseGenerator,
    CannedResponseSelectionSchema,
    FollowUpCannedResponseSelectionSchema,
    CannedResponseRevisionSchema,
    BasicNoMatchResponseProvider,
    NoMatchResponseProvider,
)
from parlant.core.evaluations import (
    EvaluationListener,
    PollingEvaluationListener,
    EvaluationDocumentStore,
    EvaluationStore,
)
from parlant.core.journey_rule_projection import JourneyRuleProjection
from parlant.core.journeys import JourneyStore, JourneyVectorStore
from parlant.core.services.indexing.customer_dependent_action_detector import (
    CustomerDependentActionDetector,
    CustomerDependentActionSchema,
)
from parlant.core.services.indexing.rule_action_proposer import (
    RuleActionProposer,
    RuleActionPropositionSchema,
)
from parlant.core.services.indexing.rule_agent_intention_proposer import (
    AgentIntentionProposer,
    AgentIntentionProposerSchema,
)
from parlant.core.services.indexing.rule_continuous_proposer import (
    RuleContinuousProposer,
    RuleContinuousPropositionSchema,
)
from parlant.core.services.indexing.rule_signal_proposer import (
    RuleSignalProposer,
    RuleSignalPropositionSchema,
)
from parlant.core.services.indexing.rule_title_proposer import (
    RuleTitleProposer,
    RuleTitlePropositionSchema,
)
from parlant.core.services.indexing.relative_action_proposer import (
    RelativeActionProposer,
    RelativeActionSchema,
)
from parlant.core.services.indexing.tool_running_action_detector import (
    ToolRunningActionDetector,
    ToolRunningActionSchema,
)
from parlant.core.canned_responses import CannedResponseStore, CannedResponseVectorStore
from parlant.core.nlp.embedding import (
    BasicEmbeddingCache,
    Embedder,
    EmbedderFactory,
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
from parlant.core.nlp.service import NLPService
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
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.compass.variable_loader import VariableLoader
from parlant.core.glossary import GlossaryStore, GlossaryVectorStore
from parlant.core.engines.alpha.rule_matching.rule_matcher import (
    RuleMatcher,
    RuleMatchingStrategyResolver,
    ResponseAnalysisBatch,
)

from parlant.core.engines.alpha.rule_matching.generic.observational_batch import (
    GenericObservationalRuleMatchesSchema,
    GenericObservationalRuleMatchingShot,
    ObservationalRuleMatching,
)
from parlant.core.engines.alpha.message_generator import (
    MessageGenerator,
    MessageGeneratorShot,
    MessageSchema,
)
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCallBatcher,
    ToolCaller,
)
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.types import Engine, EngineRegistry
from parlant.core.services.indexing.evaluation_service import (
    RuleEvaluator,
    JourneyEvaluator,
)


from parlant.core.loggers import LogLevel, Logger, StdoutLogger
from parlant.core.application import Application
from parlant.core.agents import AgentDocumentStore, AgentStore
from parlant.core.rule_tool_associations import (
    RuleToolAssociationDocumentStore,
    RuleToolAssociationStore,
)
from parlant.core.shots import ShotCollection
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
)


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
    return SyncAwaiter(asyncio.get_event_loop())


@fixture
def test_config(pytestconfig: Config) -> dict[str, Any]:
    return {"patience": 10}


async def make_schematic_generator(
    container: Container,
    cache_options: CacheOptions,
    schema: type[T],
) -> SchematicGenerator[T]:
    generator = await container[NLPService].get_schematic_generator(schema)

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
async def container(
    tracer: Tracer,
    logger: Logger,
    cache_options: CacheOptions,
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
        container[UsageReporter] = Singleton(UsageReporter)
        container[TurnEvaluator] = Singleton(RuleRanker)
        # Alias, not a second registration: retrain and discovery share the frame.
        container[RuleRecaller] = Singleton(RuleRecaller)
        container[RuleDiscoverer] = lambda c: c[RuleRecaller]

        container[CompassMatcher] = Singleton(CompassMatcher)
        container[CompassReviewer] = Singleton(BasicReviewer)
        # Self-subscribes to the UsageReporter on construction.
        container[CostControlPolicy] = AdvisoryCostControlPolicy(container[UsageReporter])

        container[AgentStore] = await stack.enter_async_context(
            AgentDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[RelationshipStore] = await stack.enter_async_context(
            RelationshipDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[SessionStore] = await stack.enter_async_context(
            SessionDocumentStore(TransientDocumentDatabase())
        )
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
        container[EvaluationStore] = await stack.enter_async_context(
            EvaluationDocumentStore(TransientDocumentDatabase())
        )
        container[EvaluationListener] = PollingEvaluationListener
        container[EventEmitterFactory] = EventPublisherFactory(
            store_provider_factory=lambda: container[StoreProvider]
        )

        container[ServiceRegistry] = await stack.enter_async_context(
            ServiceDocumentRegistry(
                database=TransientDocumentDatabase(),
                event_emitter_factory=container[EventEmitterFactory],
                logger=container[Logger],
                tracer=container[Tracer],
                nlp_services_provider=lambda: {
                    "default": GeminiService(
                        container[Logger],
                        container[Tracer],
                        container[Meter],
                        container[HealthReporter],
                        container[UsageReporter],
                    )
                },
            )
        )

        container[NLPService] = await container[ServiceRegistry].read_nlp_service("default")
        container[EstimatingTokenizer] = (
            container[NLPService]
            if isinstance(container[NLPService], EstimatingTokenizer)
            else ZeroEstimatingTokenizer()
        )

        async def get_embedder_type() -> type[Embedder]:
            return type(await container[NLPService].get_embedder())

        embedder_factory = EmbedderFactory(container)

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

        container[JourneyStore] = await stack.enter_async_context(
            JourneyVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    container[Tracer],
                    embedder_factory,
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[GlossaryStore] = await stack.enter_async_context(
            GlossaryVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    container[Tracer],
                    embedder_factory,
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[CannedResponseStore] = await stack.enter_async_context(
            CannedResponseVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger], container[Tracer], embedder_factory, lambda: embedding_cache
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[CapabilityStore] = await stack.enter_async_context(
            CapabilityVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    container[Tracer],
                    embedder_factory,
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[RuleStore] = await stack.enter_async_context(
            RuleVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    container[Tracer],
                    embedder_factory,
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[StoreProvider] = BasicStoreProvider(lambda: container)

        container[EntityQueries] = Singleton(EntityQueries)
        container[EntityCommands] = Singleton(EntityCommands)

        container[JourneyRuleProjection] = Singleton(JourneyRuleProjection)

        for generation_schema in (
            GenericObservationalRuleMatchesSchema,
            GenericActionableRuleMatchesSchema,
            GenericLowCriticalityRuleMatchesSchema,
            GenericPreviouslyAppliedActionableRuleMatchesSchema,
            GenericPreviouslyAppliedActionableCustomerDependentRuleMatchesSchema,
            MessageSchema,
            CannedResponseDraftSchema,
            CannedResponseSelectionSchema,
            FollowUpCannedResponseSelectionSchema,
            CannedResponsePreambleSchema,
            CannedResponseRevisionSchema,
            CannedResponseFieldExtractionSchema,
            single_tool_batch.SingleToolBatchSchema,
            single_tool_batch.NonConsequentialToolBatchSchema,
            overlapping_tools_batch.OverlappingToolsBatchSchema,
            RuleActionPropositionSchema,
            RuleContinuousPropositionSchema,
            RuleSignalPropositionSchema,
            RuleTitlePropositionSchema,
            CustomerDependentActionSchema,
            ToolRunningActionSchema,
            GenericResponseAnalysisSchema,
            AgentIntentionProposerSchema,
            DisambiguationRuleMatchesSchema,
            JourneyBacktrackNodeSelectionSchema,
            JourneyNextStepSelectionSchema,
            RelativeActionSchema,
            ReachableNodesEvaluationSchema,
            JourneyBacktrackCheckSchema,
            RuleRankSchema,
            LowEffortReviewSchema,
        ):
            container[SchematicGenerator[generation_schema]] = await make_schematic_generator(  # type: ignore
                container,
                cache_options,
                generation_schema,
            )

        container[ShotCollection[GenericPreviouslyAppliedActionableRuleRuleMatchingShot]] = (
            rule_previously_applied_actionable_batch.shot_collection
        )
        container[ShotCollection[GenericActionableRuleRuleMatchingShot]] = (
            rule_actionable_batch.shot_collection
        )
        container[
            ShotCollection[GenericPreviouslyAppliedActionableCustomerDependentRuleMatchingShot]
        ] = rule_previously_applied_actionable_customer_dependent_batch.shot_collection
        container[ShotCollection[GenericObservationalRuleMatchingShot]] = (
            observational_batch.shot_collection
        )
        container[ShotCollection[GenericResponseAnalysisShot]] = (
            response_analysis_batch.shot_collection
        )
        container[ShotCollection[single_tool_batch.SingleToolBatchShot]] = (
            single_tool_batch.consequential_shot_collection
        )
        container[ShotCollection[overlapping_tools_batch.OverlappingToolsBatchShot]] = (
            overlapping_tools_batch.shot_collection
        )
        container[ShotCollection[MessageGeneratorShot]] = message_generator.shot_collection

        container[RuleActionProposer] = Singleton(RuleActionProposer)
        container[RuleContinuousProposer] = Singleton(RuleContinuousProposer)
        container[RuleSignalProposer] = Singleton(RuleSignalProposer)
        container[RuleTitleProposer] = Singleton(RuleTitleProposer)
        container[CustomerDependentActionDetector] = Singleton(CustomerDependentActionDetector)
        container[AgentIntentionProposer] = Singleton(AgentIntentionProposer)
        container[ToolRunningActionDetector] = Singleton(ToolRunningActionDetector)
        container[RelativeActionProposer] = Singleton(RelativeActionProposer)
        container[LocalToolService] = cast(
            LocalToolService,
            await container[ServiceRegistry].update_tool_service(
                name="local", kind="local", url=""
            ),
        )
        container[GenericRuleMatchingStrategyResolver] = Singleton(
            GenericRuleMatchingStrategyResolver
        )
        container[RuleMatchingStrategyResolver] = lambda container: container[
            GenericRuleMatchingStrategyResolver
        ]
        container[ObservationalRuleMatching] = Singleton(ObservationalRuleMatching)
        container[GenericActionableRuleMatching] = Singleton(GenericActionableRuleMatching)
        container[GenericLowCriticalityRuleMatching] = Singleton(GenericLowCriticalityRuleMatching)
        container[GenericPreviouslyAppliedActionableRuleMatching] = Singleton(
            GenericPreviouslyAppliedActionableRuleMatching
        )
        container[GenericPreviouslyAppliedActionableCustomerDependentRuleMatching] = Singleton(
            GenericPreviouslyAppliedActionableCustomerDependentRuleMatching
        )
        container[ResponseAnalysisBatch] = Singleton(GenericResponseAnalysisBatch)
        container[RuleMatcher] = Singleton(RuleMatcher)
        container[RuleEvaluator] = Singleton(RuleEvaluator)
        container[JourneyEvaluator] = Singleton(JourneyEvaluator)

        container[DefaultToolCallBatcher] = Singleton(DefaultToolCallBatcher)
        container[ToolCallBatcher] = lambda container: container[DefaultToolCallBatcher]
        container[ToolCaller] = Singleton(ToolCaller)
        container[RelationalResolver] = Singleton(RelationalResolver)
        container[PlannerProvider] = PlannerProvider(default_planner=NullPlanner())
        container[CannedResponseGenerator] = Singleton(CannedResponseGenerator)
        container[NoMatchResponseProvider] = Singleton(BasicNoMatchResponseProvider)
        container[CannedResponseFieldExtractor] = Singleton(CannedResponseFieldExtractor)
        container[MessageGenerator] = Singleton(MessageGenerator)
        container[ToolEventGenerator] = Singleton(ToolEventGenerator)
        container[PerceivedPerformancePolicy] = NullPerceivedPerformancePolicy
        container[OptimizationPolicy] = Singleton(BasicOptimizationPolicy)

        hooks = JournalingEngineHooks()
        container[JournalingEngineHooks] = hooks
        container[EngineHooks] = hooks

        container[AuthorizationPolicy] = Singleton(DevelopmentAuthorizationPolicy)

        container[Engine] = Singleton(AlphaEngine)
        container[AlphaEngine] = Singleton(AlphaEngine)
        container[VariableLoader] = Singleton(VariableLoader)
        container[CompassEngine] = Singleton(CompassEngine)
        container[EngineRegistry] = EngineRegistry(
            {
                "alpha": lambda: container[AlphaEngine],
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
    if isinstance(
        container[SchematicGenerator[GenericPreviouslyAppliedActionableRuleMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericPreviouslyAppliedActionableRuleMatchesSchema],
            container[SchematicGenerator[GenericPreviouslyAppliedActionableRuleMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[GenericActionableRuleMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericActionableRuleMatchesSchema],
            container[SchematicGenerator[GenericActionableRuleMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[GenericLowCriticalityRuleMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericLowCriticalityRuleMatchesSchema],
            container[SchematicGenerator[GenericLowCriticalityRuleMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[
            SchematicGenerator[GenericPreviouslyAppliedActionableCustomerDependentRuleMatchesSchema]
        ],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[
                GenericPreviouslyAppliedActionableCustomerDependentRuleMatchesSchema
            ],
            container[
                SchematicGenerator[
                    GenericPreviouslyAppliedActionableCustomerDependentRuleMatchesSchema
                ]
            ],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[GenericObservationalRuleMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericObservationalRuleMatchesSchema],
            container[SchematicGenerator[GenericObservationalRuleMatchesSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[MessageSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[MessageSchema],
            container[SchematicGenerator[MessageSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponseDraftSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponseDraftSchema],
            container[SchematicGenerator[CannedResponseDraftSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponseSelectionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponseSelectionSchema],
            container[SchematicGenerator[CannedResponseSelectionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[FollowUpCannedResponseSelectionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[FollowUpCannedResponseSelectionSchema],
            container[SchematicGenerator[FollowUpCannedResponseSelectionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponsePreambleSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponsePreambleSchema],
            container[SchematicGenerator[CannedResponsePreambleSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponseRevisionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponseRevisionSchema],
            container[SchematicGenerator[CannedResponseRevisionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponseFieldExtractionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponseFieldExtractionSchema],
            container[SchematicGenerator[CannedResponseFieldExtractionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[single_tool_batch.SingleToolBatchSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[single_tool_batch.SingleToolBatchSchema],
            container[SchematicGenerator[single_tool_batch.SingleToolBatchSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[DisambiguationRuleMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[DisambiguationRuleMatchesSchema],
            container[SchematicGenerator[DisambiguationRuleMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[JourneyBacktrackNodeSelectionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[JourneyBacktrackNodeSelectionSchema],
            container[SchematicGenerator[JourneyBacktrackNodeSelectionSchema]],
        ).use_cache = False
