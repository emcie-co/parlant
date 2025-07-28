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

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, AsyncIterator, cast
from fastapi import FastAPI
import httpx
from lagom import Container, Singleton
from pytest import fixture, Config
import pytest

from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.adapters.nlp.openai_service import OpenAIService
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.api.app import create_api_app, ASGIApplication
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.capabilities import CapabilityStore, CapabilityVectorStore
from parlant.core.contextual_correlator import ContextualCorrelator
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
from parlant.core.engines.alpha.perceived_performance_policy import (
    BasicPerceivedPerformancePolicy,
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
from parlant.core.engines.alpha.utterance_selector import (
    UtteranceDraftSchema,
    UtteranceFieldExtractionSchema,
    UtteranceFieldExtractor,
    UtteranceFluidPreambleSchema,
    UtteranceSelector,
    UtteranceSelectionSchema,
    UtteranceRevisionSchema,
)
from parlant.core.evaluations import (
    EvaluationListener,
    PollingEvaluationListener,
    EvaluationDocumentStore,
    EvaluationStore,
)
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
from parlant.core.utterances import UtteranceStore, UtteranceVectorStore
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.nlp.generation import T, SchematicGenerator
from parlant.core.relationships import (
    RelationshipDocumentStore,
    RelationshipStore,
)
from parlant.core.guidelines import GuidelineDocumentStore, GuidelineStore
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.nlp.service import NLPService
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
    LegacyBehavioralChangeEvaluator,
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
def correlator() -> ContextualCorrelator:
    return ContextualCorrelator()


@fixture
def logger(correlator: ContextualCorrelator) -> Logger:
    return StdoutLogger(correlator=correlator, log_level=LogLevel.INFO)


@dataclass(frozen=True)
class CacheOptions:
    cache_enabled: bool
    cache_collection: DocumentCollection[SchematicGenerationResultDocument] | None


@fixture
async def cache_options(
    request: pytest.FixtureRequest,
    logger: Logger,
) -> AsyncIterator[CacheOptions]:
    if not request.config.getoption("no_cache", True):
        logger.warning("*** Cache is enabled")
        async with create_schematic_generation_result_collection(logger=logger) as collection:
            yield CacheOptions(cache_enabled=True, cache_collection=collection)
    else:
        yield CacheOptions(cache_enabled=False, cache_collection=None)


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
    base_generator = await container[NLPService].get_schematic_generator(schema)

    if cache_options.cache_enabled:
        assert cache_options.cache_collection

        return CachedSchematicGenerator[T](
            base_generator=base_generator,
            collection=cache_options.cache_collection,
            use_cache=True,
        )
    else:
        return base_generator


@fixture
async def container(
    correlator: ContextualCorrelator,
    logger: Logger,
    cache_options: CacheOptions,
) -> AsyncIterator[Container]:
    container = Container()

    container[ContextualCorrelator] = correlator
    container[Logger] = logger
    container[WebSocketLogger] = WebSocketLogger(container[ContextualCorrelator])

    async with AsyncExitStack() as stack:
        container[BackgroundTaskService] = await stack.enter_async_context(
            BackgroundTaskService(container[Logger])
        )

        await container[BackgroundTaskService].start(
            container[WebSocketLogger].start(), tag="websocket-logger"
        )

        container[AgentStore] = await stack.enter_async_context(
            AgentDocumentStore(TransientDocumentDatabase())
        )
        container[GuidelineStore] = await stack.enter_async_context(
            GuidelineDocumentStore(TransientDocumentDatabase())
        )
        container[RelationshipStore] = await stack.enter_async_context(
            RelationshipDocumentStore(TransientDocumentDatabase())
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
        container[LegacyBehavioralChangeEvaluator] = LegacyBehavioralChangeEvaluator
        container[EventEmitterFactory] = Singleton(EventPublisherFactory)

        container[ServiceRegistry] = await stack.enter_async_context(
            ServiceDocumentRegistry(
                database=TransientDocumentDatabase(),
                event_emitter_factory=container[EventEmitterFactory],
                logger=container[Logger],
                correlator=container[ContextualCorrelator],
                nlp_services_provider=lambda: {"default": OpenAIService(container[Logger])},
            )
        )

        container[NLPService] = await container[ServiceRegistry].read_nlp_service("default")

        async def get_embedder_type() -> type[Embedder]:
            return type(await container[NLPService].get_embedder())

        embedder_factory = EmbedderFactory(container)

        container[JourneyStore] = await stack.enter_async_context(
            JourneyVectorStore(
                vector_db=TransientVectorDatabase(container[Logger], embedder_factory),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[GlossaryStore] = await stack.enter_async_context(
            GlossaryVectorStore(
                vector_db=TransientVectorDatabase(container[Logger], embedder_factory),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[UtteranceStore] = await stack.enter_async_context(
            UtteranceVectorStore(
                vector_db=TransientVectorDatabase(container[Logger], embedder_factory),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[CapabilityStore] = await stack.enter_async_context(
            CapabilityVectorStore(
                vector_db=TransientVectorDatabase(container[Logger], embedder_factory),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[EntityQueries] = Singleton(EntityQueries)
        container[EntityCommands] = Singleton(EntityCommands)

        for generation_schema in (
            GenericObservationalGuidelineMatchesSchema,
            GenericActionableGuidelineMatchesSchema,
            GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
            MessageSchema,
            UtteranceDraftSchema,
            UtteranceSelectionSchema,
            UtteranceFluidPreambleSchema,
            UtteranceRevisionSchema,
            UtteranceFieldExtractionSchema,
            single_tool_batch.SingleToolBatchSchema,
            overlapping_tools_batch.OverlappingToolsBatchSchema,
            ConditionsEntailmentTestsSchema,
            ActionsContradictionTestsSchema,
            GuidelineConnectionPropositionsSchema,
            GuidelineActionPropositionSchema,
            GuidelineContinuousPropositionSchema,
            CustomerDependentActionSchema,
            GenericResponseAnalysisSchema,
            AgentIntentionProposerSchema,
            DisambiguationGuidelineMatchesSchema,
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
            single_tool_batch.shot_collection
        )
        container[ShotCollection[overlapping_tools_batch.OverlappingToolsBatchShot]] = (
            overlapping_tools_batch.shot_collection
        )
        container[ShotCollection[MessageGeneratorShot]] = message_generator.shot_collection

        container[GuidelineConnectionProposer] = Singleton(GuidelineConnectionProposer)
        container[CoherenceChecker] = Singleton(CoherenceChecker)
        container[GuidelineActionProposer] = Singleton(GuidelineActionProposer)
        container[GuidelineContinuousProposer] = Singleton(GuidelineContinuousProposer)
        container[CustomerDependentActionDetector] = Singleton(CustomerDependentActionDetector)
        container[AgentIntentionProposer] = Singleton(AgentIntentionProposer)
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

        container[DefaultToolCallBatcher] = Singleton(DefaultToolCallBatcher)
        container[ToolCallBatcher] = lambda container: container[DefaultToolCallBatcher]
        container[ToolCaller] = Singleton(ToolCaller)
        container[RelationalGuidelineResolver] = Singleton(RelationalGuidelineResolver)
        container[UtteranceSelector] = Singleton(UtteranceSelector)
        container[UtteranceFieldExtractor] = Singleton(UtteranceFieldExtractor)
        container[MessageGenerator] = Singleton(MessageGenerator)
        container[ToolEventGenerator] = Singleton(ToolEventGenerator)
        container[PerceivedPerformancePolicy] = Singleton(BasicPerceivedPerformancePolicy)

        hooks = JournalingEngineHooks()
        container[JournalingEngineHooks] = hooks
        container[EngineHooks] = hooks

        container[Engine] = Singleton(AlphaEngine)

        container[Application] = Application(container)

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
        container[SchematicGenerator[UtteranceDraftSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[UtteranceDraftSchema],
            container[SchematicGenerator[UtteranceDraftSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[UtteranceSelectionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[UtteranceSelectionSchema],
            container[SchematicGenerator[UtteranceSelectionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[UtteranceFluidPreambleSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[UtteranceFluidPreambleSchema],
            container[SchematicGenerator[UtteranceFluidPreambleSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[UtteranceRevisionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[UtteranceRevisionSchema],
            container[SchematicGenerator[UtteranceRevisionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[UtteranceFieldExtractionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[UtteranceFieldExtractionSchema],
            container[SchematicGenerator[UtteranceFieldExtractionSchema]],
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
    if isinstance(
        container[SchematicGenerator[DisambiguationGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[DisambiguationGuidelineMatchesSchema],
            container[SchematicGenerator[DisambiguationGuidelineMatchesSchema]],
        ).use_cache = False
