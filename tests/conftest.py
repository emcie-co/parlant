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

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.adapters.nlp.emcie_service import EmcieService
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.api.app import create_api_app, ASGIApplication
from parlant.api.authorization import AuthorizationPolicy, DevelopmentAuthorizationPolicy

from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.capabilities import CapabilityStore, CapabilityVectorStore
from parlant.core.common import IdGenerator
from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_check import (
    JourneyBacktrackCheckSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_node_selection import (
    JourneyBacktrackNodeSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_next_step_selection import (
    JourneyNextStepSelectionSchema,
)
from parlant.core.meter import Meter, LocalMeter
from parlant.core.services.indexing.journey_reachable_nodes_evaluation import (
    ReachableNodesEvaluationSchema,
)
from parlant.core.tracer import LocalTracer, Tracer
from parlant.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from parlant.core.emission.event_publisher import EventPublisherFactory
from parlant.core.emissions import EventEmitterFactory
from parlant.core.customers import CustomerDocumentStore, CustomerStore
from parlant.core.engines.alpha.guideline_matching.generic import (
    observational_batch,
)
from parlant.core.engines.alpha.guideline_matching.generic import (
    guideline_previously_applied_actionable_batch,
)
from parlant.core.engines.alpha.guideline_matching.generic import (
    guideline_actionable_batch,
)
from parlant.core.engines.alpha.guideline_matching.generic import (
    guideline_previously_applied_actionable_customer_dependent_batch,
)
from parlant.core.engines.alpha.guideline_matching.generic import (
    response_analysis_batch,
)
from parlant.core.engines.alpha.guideline_matching.generic.disambiguation_batch import (
    DisambiguationGuidelineMatchesSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic_guideline_matching_strategy_resolver import (
    GenericGuidelineMatchingStrategyResolver,
)
from parlant.core.engines.alpha.optimization_policy import (
    BasicOptimizationPolicy,
    OptimizationPolicy,
)
from parlant.core.engines.alpha.perceived_performance_policy import (
    NullPerceivedPerformancePolicy,
    PerceivedPerformancePolicy,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_customer_dependent_batch import (
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_actionable_batch import (
    GenericActionableGuidelineMatchesSchema,
    GenericActionableGuidelineMatching,
    GenericActionableGuidelineGuidelineMatchingShot,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableGuidelineMatching,
    GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot,
)
from parlant.core.engines.alpha.tool_calling import overlapping_tools_batch, single_tool_batch
from parlant.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
    GenericResponseAnalysisShot,
)
from parlant.core.engines.alpha import message_generator
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.alpha.relational_guideline_resolver import RelationalGuidelineResolver
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
from parlant.core.journey_guideline_projection import JourneyGuidelineProjection
from parlant.core.journeys import JourneyStore, JourneyVectorStore
from parlant.core.services.indexing.customer_dependent_action_detector import (
    CustomerDependentActionDetector,
    CustomerDependentActionSchema,
)
from parlant.core.services.indexing.guideline_action_proposer import (
    GuidelineActionProposer,
    GuidelineActionPropositionSchema,
)
from parlant.core.services.indexing.guideline_agent_intention_proposer import (
    AgentIntentionProposer,
    AgentIntentionProposerSchema,
)
from parlant.core.services.indexing.guideline_continuous_proposer import (
    GuidelineContinuousProposer,
    GuidelineContinuousPropositionSchema,
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
from parlant.core.guidelines import GuidelineDocumentStore, GuidelineStore
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
from parlant.core.glossary import GlossaryStore, GlossaryVectorStore
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatcher,
    GuidelineMatchingStrategyResolver,
    ResponseAnalysisBatch,
)

from parlant.core.engines.alpha.guideline_matching.generic.observational_batch import (
    GenericObservationalGuidelineMatchesSchema,
    GenericObservationalGuidelineMatchingShot,
    ObservationalGuidelineMatching,
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
from parlant.core.engines.types import Engine
from parlant.core.services.indexing.behavioral_change_evaluation import (
    GuidelineEvaluator,
    JourneyEvaluator,
)


from parlant.core.loggers import LogLevel, Logger, StdoutLogger
from parlant.core.application import Application
from parlant.core.agents import AgentDocumentStore, AgentStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from parlant.core.shots import ShotCollection
from parlant.core.entity_cq import EntityQueries, EntityCommands
from parlant.core.tags import TagDocumentStore, TagStore
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

        container[AgentStore] = await stack.enter_async_context(
            AgentDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[GuidelineStore] = await stack.enter_async_context(
            GuidelineDocumentStore(container[IdGenerator], TransientDocumentDatabase())
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
        container[TagStore] = await stack.enter_async_context(
            TagDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[CustomerStore] = await stack.enter_async_context(
            CustomerDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[GuidelineToolAssociationStore] = await stack.enter_async_context(
            GuidelineToolAssociationDocumentStore(
                container[IdGenerator], TransientDocumentDatabase()
            )
        )
        container[SessionListener] = PollingSessionListener
        container[EvaluationStore] = await stack.enter_async_context(
            EvaluationDocumentStore(TransientDocumentDatabase())
        )
        container[EvaluationListener] = PollingEvaluationListener
        container[EventEmitterFactory] = Singleton(EventPublisherFactory)

        container[ServiceRegistry] = await stack.enter_async_context(
            ServiceDocumentRegistry(
                database=TransientDocumentDatabase(),
                event_emitter_factory=container[EventEmitterFactory],
                logger=container[Logger],
                tracer=container[Tracer],
                nlp_services_provider=lambda: {
                    "default": EmcieService(
                        container[Logger],
                        container[Tracer],
                        container[Meter],
                        model_tier=os.environ.get("EMCIE_MODEL_TIER", "jackal"),  # type: ignore
                        model_role=os.environ.get("EMCIE_MODEL_ROLE", "teacher"),  # type: ignore
                    )
                },
            )
        )

        container[NLPService] = await container[ServiceRegistry].read_nlp_service("default")

        async def get_embedder_type() -> type[Embedder]:
            return type(await container[NLPService].get_embedder())

        embedder_factory = EmbedderFactory(container)

        if cache_options.cache_enabled:
            embedding_cache: EmbeddingCache = BasicEmbeddingCache(
                document_database=await stack.enter_async_context(
                    JSONFileDocumentDatabase(logger, GLOBAL_EMBEDDER_CACHE_FILE),
                )
            )
        else:
            embedding_cache = NullEmbeddingCache()

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

        container[EntityQueries] = Singleton(EntityQueries)
        container[EntityCommands] = Singleton(EntityCommands)

        container[JourneyGuidelineProjection] = Singleton(JourneyGuidelineProjection)

        for generation_schema in (
            GenericObservationalGuidelineMatchesSchema,
            GenericActionableGuidelineMatchesSchema,
            GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
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
            GuidelineActionPropositionSchema,
            GuidelineContinuousPropositionSchema,
            CustomerDependentActionSchema,
            ToolRunningActionSchema,
            GenericResponseAnalysisSchema,
            AgentIntentionProposerSchema,
            DisambiguationGuidelineMatchesSchema,
            JourneyBacktrackNodeSelectionSchema,
            JourneyNextStepSelectionSchema,
            RelativeActionSchema,
            ReachableNodesEvaluationSchema,
            JourneyBacktrackCheckSchema,
        ):
            container[SchematicGenerator[generation_schema]] = await make_schematic_generator(  # type: ignore
                container,
                cache_options,
                generation_schema,
            )

        container[
            ShotCollection[GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot]
        ] = guideline_previously_applied_actionable_batch.shot_collection
        container[ShotCollection[GenericActionableGuidelineGuidelineMatchingShot]] = (
            guideline_actionable_batch.shot_collection
        )
        container[
            ShotCollection[GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot]
        ] = guideline_previously_applied_actionable_customer_dependent_batch.shot_collection
        container[ShotCollection[GenericObservationalGuidelineMatchingShot]] = (
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

        container[GuidelineActionProposer] = Singleton(GuidelineActionProposer)
        container[GuidelineContinuousProposer] = Singleton(GuidelineContinuousProposer)
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
        container[GenericGuidelineMatchingStrategyResolver] = Singleton(
            GenericGuidelineMatchingStrategyResolver
        )
        container[GuidelineMatchingStrategyResolver] = lambda container: container[
            GenericGuidelineMatchingStrategyResolver
        ]
        container[ObservationalGuidelineMatching] = Singleton(ObservationalGuidelineMatching)
        container[GenericActionableGuidelineMatching] = Singleton(
            GenericActionableGuidelineMatching
        )
        container[GenericPreviouslyAppliedActionableGuidelineMatching] = Singleton(
            GenericPreviouslyAppliedActionableGuidelineMatching
        )
        container[GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching] = Singleton(
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching
        )
        container[ResponseAnalysisBatch] = Singleton(GenericResponseAnalysisBatch)
        container[GuidelineMatcher] = Singleton(GuidelineMatcher)
        container[GuidelineEvaluator] = Singleton(GuidelineEvaluator)
        container[JourneyEvaluator] = Singleton(JourneyEvaluator)

        container[DefaultToolCallBatcher] = Singleton(DefaultToolCallBatcher)
        container[ToolCallBatcher] = lambda container: container[DefaultToolCallBatcher]
        container[ToolCaller] = Singleton(ToolCaller)
        container[RelationalGuidelineResolver] = Singleton(RelationalGuidelineResolver)
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

        container[Application] = Singleton(Application)

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
        container[SchematicGenerator[GenericPreviouslyAppliedActionableGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericPreviouslyAppliedActionableGuidelineMatchesSchema],
            container[SchematicGenerator[GenericPreviouslyAppliedActionableGuidelineMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[GenericActionableGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericActionableGuidelineMatchesSchema],
            container[SchematicGenerator[GenericActionableGuidelineMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[
            SchematicGenerator[
                GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
            ]
        ],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[
                GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
            ],
            container[
                SchematicGenerator[
                    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
                ]
            ],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[GenericObservationalGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericObservationalGuidelineMatchesSchema],
            container[SchematicGenerator[GenericObservationalGuidelineMatchesSchema]],
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
        container[SchematicGenerator[DisambiguationGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[DisambiguationGuidelineMatchesSchema],
            container[SchematicGenerator[DisambiguationGuidelineMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[JourneyBacktrackNodeSelectionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[JourneyBacktrackNodeSelectionSchema],
            container[SchematicGenerator[JourneyBacktrackNodeSelectionSchema]],
        ).use_cache = False
