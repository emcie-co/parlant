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

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import AsyncExitStack
import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
import enum
from functools import partial
from hashlib import md5
import importlib.util
from itertools import chain
from pathlib import Path
import sys
import rich
from rich.console import Console, Group
from rich.panel import Panel
import rich.box
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
    TaskID,
    TextColumn,
)
from rich.live import Live
from rich.text import Text
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
    NoReturn,
    Optional,
    Sequence,
    TypeVar,
    TypeAlias,
    TypedDict,
    cast,
)
from typing_extensions import overload
from fastapi import FastAPI
import httpx
from lagom import Container


from parlant.adapters.db.json_file import JSONFileDocumentCollection, JSONFileDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.api.authorization import (
    AuthorizationException,
    Operation,
    AuthorizationPolicy,
    BasicRateLimiter,
    DevelopmentAuthorizationPolicy,
    ProductionAuthorizationPolicy,
    RateLimitExceededException,
    RateLimiter,
)


from parlant.core import async_utils
from parlant.core.agents import (
    AgentDocumentStore,
    AgentId,
    AgentStore,
    CompositionMode as _CompositionMode,
)
from parlant.core.async_utils import Timeout, default_done_callback
from parlant.core.capabilities import CapabilityId, CapabilityStore, CapabilityVectorStore
from parlant.core.common import (
    Criticality,
    DefaultBaseModel,
    IdGenerator,
    ItemNotFoundError,
    JSONSerializable,
    Version,
    classproperty,
)
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableDocumentStore,
    ContextVariableId,
    ContextVariableStore,
)
from parlant.core.engines.alpha.guideline_matching.generic.common import (
    format_journey_node_guideline_id,
)
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.customers import (
    Customer as _Customer,
    CustomerDocumentStore,
    CustomerId,
    CustomerStore,
)
from parlant.core.emissions import EmittedEvent, EventEmitterFactory
from parlant.core.engines.alpha.prompt_builder import PromptBuilder, PromptSection
from parlant.core.engines.alpha.hooks import EngineHook, EngineHookResult, EngineHooks
from parlant.core.engines.alpha.engine_context import (
    EngineContext,
    LoadedContext,  # type: ignore
    Interaction,
    InteractionMessage,
)
from parlant.core.engines.alpha.entity_context import EntityContext
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch as _GuidelineMatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matching_context import (
    GuidelineMatchingContext as _GuidelineMatchingContext,
)
from parlant.core.engines.alpha.guideline_matching.generic_guideline_matching_strategy_resolver import (
    GenericGuidelineMatchingStrategyResolver,
)
from parlant.core.engines.alpha.guideline_matching.custom_guideline_matching_strategy import (
    CustomGuidelineMatchingStrategy,
)
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
    SessionId,
    SessionDocumentStore,
    SessionStore,
    StatusEventData,
    ToolCall as _SessionToolCall,
    ToolEventData,
    ToolResult as _SessionToolResult,
)
from parlant.core.canned_responses import (
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
    Guideline as _Guideline,
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
from parlant.core.nlp.service import (
    EmbedderHints,
    ModelGeneration,
    ModelSize,
    ModelType,
    NLPService,
    SchematicGeneratorHints,
)

from parlant.core.nlp.moderation import (
    CustomerModerationContext,
    ModerationCheck,
    ModerationService,
    ModerationTag,
    NoModeration,
)
from parlant.core.engines.alpha.canned_response_generator import (
    NoMatchResponseProvider,
    BasicNoMatchResponseProvider,
)
from parlant.core.engines.alpha.optimization_policy import (
    OptimizationPolicy,
    BasicOptimizationPolicy,
)
from parlant.core.engines.alpha.perceived_performance_policy import (
    PerceivedPerformancePolicy,
    PerceivedPerformancePolicyProvider,
    NullPerceivedPerformancePolicy,
    BasicPerceivedPerformancePolicy,
    VoiceOptimizedPerceivedPerformancePolicy,
)
from parlant.bin.server import PARLANT_HOME_DIR, start_parlant, StartupParameters
from parlant.core.services.tools.plugins import PluginServer, ToolEntry, tool
from parlant.core.tags import Tag as _Tag, TagDocumentStore, TagId, TagStore
from parlant.core.tools import (
    ControlOptions,
    Lifespan,
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
    """Main class for SDK-related errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class NLPServiceConfigurationError(SDKError):
    """Raised when there is a configuration error with an NLP service."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class NLPServices:
    """A collection of static methods to create built-in NLPService instances for the SDK."""

    @staticmethod
    def emcie(container: Container) -> NLPService:
        """Creates an Azure NLPService instance using the provided container."""
        from parlant.adapters.nlp.emcie_service import EmcieService

        if error := EmcieService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return EmcieService(
            container[Logger],
            container[Tracer],
            container[Meter],
        )

    @staticmethod
    def azure(container: Container) -> NLPService:
        """Creates an Azure NLPService instance using the provided container."""
        from parlant.adapters.nlp.azure_service import AzureService

        if error := AzureService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return AzureService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def openai(container: Container) -> NLPService:
        """Creates an OpenAI NLPService instance using the provided container."""
        from parlant.adapters.nlp.openai_service import OpenAIService

        if error := OpenAIService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return OpenAIService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def anthropic(container: Container) -> NLPService:
        """Creates an Anthropic NLPService instance using the provided container."""
        from parlant.adapters.nlp.anthropic_service import AnthropicService

        if error := AnthropicService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return AnthropicService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def cerebras(container: Container) -> NLPService:
        """Creates a Cerebras NLPService instance using the provided container."""
        from parlant.adapters.nlp.cerebras_service import CerebrasService

        if error := CerebrasService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return CerebrasService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def together(container: Container) -> NLPService:
        """Creates a Together NLPService instance using the provided container."""
        from parlant.adapters.nlp.together_service import TogetherService

        if error := TogetherService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return TogetherService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def gemini(container: Container) -> NLPService:
        """Creates a Gemini NLPService instance using the provided container."""
        from parlant.adapters.nlp.gemini_service import GeminiService

        if error := GeminiService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return GeminiService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def litellm(container: Container) -> NLPService:
        """Creates a Litellm NLPService instance using the provided container."""
        from parlant.adapters.nlp.litellm_service import LiteLLMService

        if error := LiteLLMService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return LiteLLMService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def modelscope(container: Container) -> NLPService:
        """Creates a ModelScope NLPService instance using the provided container."""
        from parlant.adapters.nlp.modelscope_service import ModelScopeService

        if error := ModelScopeService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return ModelScopeService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def vertex(container: Container) -> NLPService:
        """Creates a Vertex NLPService instance using the provided container."""
        from parlant.adapters.nlp.vertex_service import VertexAIService

        if error := VertexAIService.verify_environment():
            raise NLPServiceConfigurationError(error)

        if err := VertexAIService.validate_adc():
            raise NLPServiceConfigurationError(err)

        return VertexAIService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def mistral(container: Container) -> NLPService:
        """Creates a Ollama NLPService instance using the provided container."""
        from parlant.adapters.nlp.mistral_service import MistralService

        if error := MistralService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return MistralService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def ollama(container: Container) -> NLPService:
        """Creates a Ollama NLPService instance using the provided container."""
        from parlant.adapters.nlp.ollama_service import OllamaService

        if error := OllamaService.verify_environment():
            raise NLPServiceConfigurationError(error)

        if err := OllamaService.verify_models():
            raise NLPServiceConfigurationError(err)

        return OllamaService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def glm(container: Container) -> NLPService:
        """Creates a GLM NLPService instance using the provided container."""
        from parlant.adapters.nlp.glm_service import GLMService

        if error := GLMService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return GLMService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def qwen(container: Container) -> NLPService:
        """Creates a Qwen NLPService instance using the provided container."""
        from parlant.adapters.nlp.qwen_service import QwenService

        if error := QwenService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return QwenService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def deepseek(container: Container) -> NLPService:
        """Creates a DeepSeek NLPService instance using the provided container."""
        from parlant.adapters.nlp.deepseek_service import DeepSeekService

        if error := DeepSeekService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return DeepSeekService(container[Logger], container[Tracer], container[Meter])

    @staticmethod
    def snowflake(container: Container) -> NLPService:
        """Creates a SnowflakeCortexService instance using the provided container."""
        from parlant.adapters.nlp.snowflake_cortex_service import SnowflakeCortexService

        if error := SnowflakeCortexService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return SnowflakeCortexService(container[Logger], container[Tracer], container[Meter])

    # @staticmethod
    # def fireworks(container: Container) -> NLPService:
    #     """Creates a Fireworks NLPService instance using the provided container."""
    #     from parlant.adapters.nlp.fireworks_service import FireworksService
    #
    #     if error := FireworksService.verify_environment():
    #         raise SDKError(error)
    #
    #     return FireworksService(container[Logger], container[Meter])
    # NOTE: Fireworks method is temporarily disabled due to fireworks-ai dependency
    # pinning protobuf=5.29.3 which has security vulnerability CVE-2025-4565

    @staticmethod
    def openrouter(
        container: Container | None = None,
    ) -> NLPService | Callable[[Container], NLPService]:
        """
        Returns a callable that creates an OpenRouter NLPService instance using the provided container.
        If container is None, the callable expects the container to be provided later (by the Server).
        All configuration is done via environment variables.
        """
        from parlant.adapters.nlp.openrouter_service import OpenRouterService

        def factory(c: Container) -> NLPService:
            if error := OpenRouterService.verify_environment():
                raise NLPServiceConfigurationError(error)
            return OpenRouterService(
                c[Logger],
                c[Tracer],
                c[Meter],
            )

        if container is not None:
            return factory(container)

        return factory

    @staticmethod
    def zhipu(container: Container) -> NLPService:
        """Creates a Zhipu AI NLPService instance using the provided container."""
        from parlant.adapters.nlp.zhipu_service import ZhipuService

        if error := ZhipuService.verify_environment():
            raise NLPServiceConfigurationError(error)

        return ZhipuService(container[Logger], container[Tracer], container[Meter])


class _CachedGuidelineEvaluation(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    version: Version.String
    properties: dict[str, JSONSerializable]


class _CachedJourneyEvaluation(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
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
            name=f"guideline_evaluations_{VERSION}",
            schema=_CachedGuidelineEvaluation,
            document_loader=identity_loader_for(_CachedGuidelineEvaluation),
        )

        self._journey_collection = await self._db.get_or_create_collection(
            name=f"journey_evaluations_{VERSION}",
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
        properties_proposition: bool,
    ) -> str:
        """Generate a hash for the guideline evaluation request."""
        tool_ids_str = ",".join(str(tool_id) for tool_id in tool_ids) if tool_ids else ""

        return md5(
            f"{g.condition or ''}:{g.action or ''}:{tool_ids_str}:{journey_state_propositions}:{properties_proposition}".encode()
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
        properties_proposition: bool = True,
    ) -> _CachedEvaluator.GuidelineEvaluation:
        # First check if we have a cached evaluation for this guideline
        _hash = self._hash_guideline_evaluation_request(
            g=g,
            tool_ids=tool_ids,
            journey_state_propositions=journey_state_proposition,
            properties_proposition=properties_proposition,
        )

        if cached_evaluation := await self._guideline_collection.find_one({"id": {"$eq": _hash}}):
            self._logger.trace(
                f"Using cached evaluation for guideline: Condition: {g.condition or 'None'}; Action: {g.action or 'None'}"
            )

            return self.GuidelineEvaluation(
                properties=cached_evaluation["properties"],
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
                        action_proposition=action_proposition,
                        properties_proposition=properties_proposition,
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
                    "creation_utc": datetime.now(timezone.utc).isoformat(),
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
            self._logger.trace(
                f"Using cached evaluation for journey: Title: {journey.title or 'None'};"
            )

            return self.JourneyEvaluation(
                node_properties=cached_evaluation["node_properties"],
                edge_properties=cached_evaluation["edge_properties"],
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
                    "creation_utc": datetime.now(timezone.utc).isoformat(),
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


@dataclass(frozen=True)
class Tag:
    """A tag used to categorize and link entities."""

    @staticmethod
    def preamble() -> TagId:
        return _Tag.preamble()

    id: TagId
    name: str


@dataclass(frozen=True)
class Relationship:
    """A relationship between two entities in the system."""

    id: RelationshipId
    kind: RelationshipKind
    source: RelationshipEntityId
    target: RelationshipEntityId


@dataclass(frozen=True)
class GuidelineMatch:
    """Result of a custom guideline matcher."""

    id: GuidelineId
    """The ID of the guideline that was matched."""

    matched: bool
    """Whether the guideline matched the current context."""

    rationale: str
    """Explanation of why the guideline matched or didn't match."""


@dataclass
class GuidelineMatchingContext:
    """Context for custom guideline matchers, providing information about the current interaction."""

    server: Server
    container: Container
    logger: Logger
    tracer: Tracer
    session: Session
    agent: Agent
    customer: Customer
    variables: Mapping[Variable, JSONSerializable]

    @classmethod
    async def _from_core(
        cls,
        core_ctx: _GuidelineMatchingContext,
        server: Server,
        container: Container,
    ) -> GuidelineMatchingContext:
        """Convert a core GuidelineMatchingContext to an SDK GuidelineMatchingContext."""
        agent = await server.get_agent(id=core_ctx.agent.id)
        customer = await server.get_customer(id=core_ctx.customer.id)
        interaction = Interaction(core_ctx.interaction_history)

        return cls(
            server=server,
            container=container,
            logger=container[Logger],
            tracer=container[Tracer],
            session=Session(
                id=core_ctx.session.id,
                interaction=interaction,
            ),
            agent=agent,
            customer=customer,
            variables={
                await agent.get_variable(id=var.id): val.data
                for var, val in core_ctx.context_variables
            },
        )


async def _match_always(ctx: GuidelineMatchingContext, g: Guideline) -> GuidelineMatch:
    return GuidelineMatch(
        id=g.id,
        matched=True,
        rationale="Always relevant",
    )


@dataclass
class JourneyStateMatch:
    """Result of a journey state transition match."""

    state_id: JourneyStateId
    """The ID of the journey state that was matched."""

    transition_id: JourneyTransitionId
    """The ID of the journey transition that was matched."""

    matched: bool
    """Whether the journey state transition matched the current context."""

    rationale: str | None
    """Explanation of why the state transition matched or didn't match."""


@dataclass(frozen=True)
class Guideline:
    """A guideline that defines a condition and an action to be taken."""

    MATCH_ALWAYS = _match_always

    id: GuidelineId
    condition: str
    action: str | None
    tags: Sequence[TagId]
    metadata: Mapping[str, JSONSerializable]

    _server: Server
    _container: Container

    async def entail(self, guideline: Guideline) -> Relationship:
        """Creates an entailment relationship with another guideline."""
        return await self._create_relationship(
            target=guideline,
            kind=RelationshipKind.ENTAILMENT,
            direction="source",
        )

    async def prioritize_over(self, target: Guideline | Journey) -> Relationship:
        """Creates a priority relationship with another guideline or journey."""
        if isinstance(target, Guideline):
            return await self._create_relationship(
                target=target,
                kind=RelationshipKind.PRIORITY,
                direction="source",
            )
        elif isinstance(target, Journey):
            return await self._create_relationship(
                target=target,
                kind=RelationshipKind.PRIORITY,
                direction="source",
            )
        else:
            raise SDKError("Either guideline or journey must be provided for prioritization.")

    async def depend_on(
        self,
        target: Guideline | Journey,
    ) -> Relationship:
        if isinstance(target, Guideline):
            return await self._create_relationship(
                target=target,
                kind=RelationshipKind.DEPENDENCY,
                direction="source",
            )
        elif isinstance(target, Journey):
            return await self._create_relationship(
                target=target,
                kind=RelationshipKind.DEPENDENCY,
                direction="source",
            )
        else:
            raise SDKError("Either guideline or journey must be provided for dependency.")

    async def disambiguate(
        self,
        targets: Sequence[Guideline | Journey],
    ) -> Sequence[Relationship]:
        if len(targets) < 2:
            raise SDKError(
                f"At least two targets are required for disambiguation (got {len(targets)})."
            )

        guideline_targets = [t for t in targets if isinstance(t, Guideline)]
        journey_conditions = list(
            chain.from_iterable([t.conditions for t in targets if isinstance(t, Journey)])
        )

        return [
            await self._create_relationship(
                target=t,
                kind=RelationshipKind.DISAMBIGUATION,
                direction="source",
            )
            for t in guideline_targets + journey_conditions
        ]

    async def reevaluate_after(self, tool: ToolEntry) -> Relationship:
        """Creates a reevaluation relationship with a tool."""
        relationship = await self._container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=self.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=tool.tool.name),
                kind=RelationshipEntityKind.TOOL,
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
        target: Guideline | Journey,
        kind: RelationshipKind,
        direction: Literal["source", "target"],
    ) -> Relationship:
        if direction == "source":
            entity_source = RelationshipEntity(id=self.id, kind=RelationshipEntityKind.GUIDELINE)
            entity_target = (
                RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE)
                if isinstance(target, Guideline)
                else RelationshipEntity(
                    id=_Tag.for_journey_id(target.id), kind=RelationshipEntityKind.TAG
                )
            )
        else:
            entity_source = (
                RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE)
                if isinstance(target, Guideline)
                else RelationshipEntity(
                    id=_Tag.for_journey_id(target.id), kind=RelationshipEntityKind.TAG
                )
            )
            entity_target = RelationshipEntity(id=self.id, kind=RelationshipEntityKind.GUIDELINE)

        relationship = await self._container[RelationshipStore].create_relationship(
            source=entity_source,
            target=entity_target,
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
    """A transition between two states in a journey."""

    id: JourneyTransitionId
    condition: str | None
    source: JourneyState
    target: TState
    metadata: Mapping[str, JSONSerializable]


@dataclass(frozen=True)
class JourneyState:
    """A state in a journey that can be transitioned to or from."""

    id: JourneyStateId
    action: str | None
    tools: Sequence[ToolEntry]
    metadata: Mapping[str, JSONSerializable]
    description: str | None

    _journey: Journey | None

    @property
    def _internal_action(self) -> str | None:
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
        description: str | None = None,
        tools: Sequence[ToolEntry] = [],
        journey: Journey | None = None,
        fork: bool = False,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        composition_mode: CompositionMode | None = None,
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[JourneyState]:
        if not self._journey:
            raise SDKError("EndState cannot be connected to any other states.")

        transitions = [t for t in self._journey.transitions if t.source == self]

        if len(transitions) > 0 and (not condition or any(not e.condition for e in transitions)):
            raise SDKError(
                "Cannot connect a new state without a condition if there are already connected states without conditions."
            )

        actual_state: JourneyState | None = None

        if state is not None:
            actual_state = state
        elif tools:
            actual_state = await self._journey._create_state(
                ToolJourneyState,
                action=action,
                description=description,
                tools=tools,
                metadata=metadata,
                composition_mode=composition_mode,
            )

            [
                await self._journey._container[RelationshipStore].create_relationship(
                    source=RelationshipEntity(
                        id=_Tag.for_journey_node_id(actual_state.id),
                        kind=RelationshipEntityKind.TAG,
                    ),
                    target=RelationshipEntity(
                        id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name),
                        kind=RelationshipEntityKind.TOOL,
                    ),
                    kind=RelationshipKind.REEVALUATION,
                )
                for t in tools
            ]

        elif action:
            actual_state = await self._journey._create_state(
                ChatJourneyState,
                action=action,
                description=description,
                tools=[],
                metadata=metadata,
                composition_mode=composition_mode,
            )
        elif fork:
            actual_state = await self._journey._create_state(
                ForkJourneyState,
                description=description,
                metadata=metadata,
                composition_mode=composition_mode,
            )
        elif journey:
            if canned_responses:
                raise SDKError(
                    "Canned responses cannot be associated when transitioning to a sub-journey."
                )

            return await self._transition_to_sub_journey(
                journey=journey,
                condition=condition,
            )

        transition = await self._journey.create_transition(
            condition=condition,
            source=self,
            target=actual_state or END_JOURNEY,
            on_match=on_match,
            on_message=on_message,
            canned_response_field_provider=canned_response_field_provider,
        )

        if actual_state:
            cast(list[JourneyState], self._journey.states).append(actual_state)

            for id in canned_responses:
                await self._journey._container[CannedResponseStore].upsert_tag(
                    canned_response_id=id, tag_id=_Tag.for_journey_node_id(actual_state.id)
                )

        cast(list[JourneyTransition[JourneyState]], self._journey.transitions).append(transition)

        return transition

    async def _transition_to_sub_journey(
        self,
        journey: Journey,
        condition: str | None = None,
    ) -> JourneyTransition[JourneyState]:
        if self._journey is None:
            raise SDKError(
                "Cannot transition to sub-journey from a state without a parent journey."
            )

        # Create mappings for states and transitions for easy lookup
        state_mapping: dict[JourneyStateId, JourneyState] = {}
        transitions_by_source: dict[JourneyStateId, list[JourneyTransition[JourneyState]]] = (
            defaultdict(list)
        )
        for transition in journey.transitions:
            transitions_by_source[transition.source.id].append(transition)

        # Create merge fork state for leaf nodes
        fork_state = await self._journey._create_state(
            ForkJourneyState,
            metadata={"sub_journey_id": journey.id},
        )

        async def create_mapped_state(state: JourneyState) -> JourneyState:
            assert self._journey  # We already checked this above

            metadata = dict(state.metadata)
            metadata["journey_node"] = {
                **cast(dict[str, JSONSerializable], metadata.get("journey_node", {})),
                "journey_id": self._journey.id,
                "sub_journey_id": journey.id,
            }

            # Create the new state
            if state.tools:
                new_state = cast(
                    JourneyState,
                    await self._journey._create_state(
                        ToolJourneyState,
                        action=state.action,
                        tools=state.tools,
                        metadata=metadata,
                    ),
                )

                [
                    await self._journey._container[RelationshipStore].create_relationship(
                        source=RelationshipEntity(
                            id=_Tag.for_journey_node_id(new_state.id),
                            kind=RelationshipEntityKind.TAG,
                        ),
                        target=RelationshipEntity(
                            id=ToolId(
                                service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name
                            ),
                            kind=RelationshipEntityKind.TOOL,
                        ),
                        kind=RelationshipKind.REEVALUATION,
                    )
                    for t in state.tools
                ]

            elif (
                isinstance(state.metadata.get("journey_node"), dict)
                and cast(dict[str, JSONSerializable], state.metadata.get("journey_node")).get(
                    "kind"
                )
                == "fork"
            ):
                new_state = cast(
                    JourneyState,
                    await self._journey._create_state(
                        ForkJourneyState,
                        metadata=metadata,
                    ),
                )
            else:
                new_state = cast(
                    JourneyState,
                    await self._journey._create_state(
                        ChatJourneyState,
                        action=state.action,
                        tools=[],
                        metadata=metadata,
                    ),
                )

            # Copy canned responses from the original state to the new state
            original_state_tag = _Tag.for_journey_node_id(state.id)
            new_state_tag = _Tag.for_journey_node_id(new_state.id)

            # Get all canned responses associated with the original state
            canned_response_store = self._journey._container[CannedResponseStore]
            canreps = await canned_response_store.list_canned_responses(tags=[original_state_tag])

            # Associate them with the new state
            for canrep in canreps:
                await canned_response_store.upsert_tag(
                    canned_response_id=canrep.id, tag_id=new_state_tag
                )

            return new_state

        # Create entry point - either self directly or via a condition fork
        entry_state: JourneyState

        if condition:
            # Create a fork state for the condition
            entry_fork = await self._journey._create_state(
                ForkJourneyState,
                metadata={"sub_journey_id": journey.id},
            )
            cast(list[JourneyState], self._journey.states).append(entry_fork)

            # Create transition from self to the entry fork with condition
            entry_transition = await self._journey.create_transition(
                condition=condition,
                source=self,
                target=entry_fork,
            )
            cast(list[JourneyTransition[JourneyState]], self._journey.transitions).append(
                cast(JourneyTransition[JourneyState], entry_transition)
            )
            entry_state = entry_fork
        else:
            entry_state = self

        # Traverse the journey starting from the root
        queue: deque[tuple[JourneyStateId, JourneyState | None]] = deque()
        visited: set[JourneyStateId] = set()

        # Skip the root state and go directly to its target states
        root_transitions = transitions_by_source[journey._start_state_id]

        # Process each transition from the root state
        for root_transition in root_transitions:
            target_state_id = root_transition.target.id

            if target_state_id == END_JOURNEY.id:
                # Root transitions directly to END_JOURNEY - connect to fork
                new_transition = await self._journey.create_transition(
                    condition=root_transition.condition,
                    source=entry_state,
                    target=fork_state,
                )
                cast(list[JourneyTransition[JourneyState]], self._journey.transitions).append(
                    cast(JourneyTransition[JourneyState], new_transition)
                )
            else:
                # Create the target state and add it to processing queue
                if target_state := next(
                    (s for s in journey.states if s.id == target_state_id), None
                ):
                    new_state = await create_mapped_state(target_state)
                    state_mapping[target_state_id] = new_state
                    cast(list[JourneyState], self._journey.states).append(new_state)

                    # Create transition from entry_state to the target
                    new_transition = await self._journey.create_transition(
                        condition=root_transition.condition,
                        source=entry_state,
                        target=cast(ForkJourneyState, new_state),
                    )
                    cast(list[JourneyTransition[JourneyState]], self._journey.transitions).append(
                        cast(JourneyTransition[JourneyState], new_transition)
                    )

                    # Add to queue for further processing
                    queue.append((target_state_id, new_state))

        while queue:
            current_state_id, mapped_source_state = queue.popleft()

            if current_state_id in visited:
                continue

            visited.add(current_state_id)

            # Get the current state from the sub-journey
            current_state = next((s for s in journey.states if s.id == current_state_id), None)
            if not current_state:
                continue

            # Check if this state has no outgoing transitions (leaf state)
            state_transitions = transitions_by_source.get(current_state_id, [])
            if (
                not state_transitions
                and mapped_source_state
                and current_state_id != journey._start_state_id
            ):
                # This is a leaf state - connect it to the fork
                new_transition = await self._journey.create_transition(
                    condition=None,
                    source=mapped_source_state,
                    target=fork_state,
                )
                cast(list[JourneyTransition[JourneyState]], self._journey.transitions).append(
                    cast(JourneyTransition[JourneyState], new_transition)
                )
                continue

            # Process all transitions from this state
            for transition in state_transitions:
                target_state_id = transition.target.id

                # Handle END_JOURNEY transitions - connect to fork
                if target_state_id == END_JOURNEY.id:
                    if mapped_source_state:
                        new_transition = await self._journey.create_transition(
                            condition=transition.condition,
                            source=mapped_source_state,
                            target=fork_state,
                        )
                        cast(
                            list[JourneyTransition[JourneyState]], self._journey.transitions
                        ).append(cast(JourneyTransition[JourneyState], new_transition))
                        # Transition to fork created
                    continue

                # Get or create the target state
                if target_state_id not in state_mapping:
                    if target_state := next(
                        (s for s in journey.states if s.id == target_state_id), None
                    ):
                        new_state = await create_mapped_state(target_state)
                        state_mapping[target_state_id] = new_state
                        cast(list[JourneyState], self._journey.states).append(new_state)

                # Create the transition only if target state is in mapping
                if target_state_id in state_mapping:
                    target_mapped_state = state_mapping[target_state_id]
                    if mapped_source_state:
                        new_transition = await self._journey.create_transition(
                            condition=transition.condition,
                            source=mapped_source_state,
                            target=cast(ForkJourneyState, target_mapped_state),
                        )

                        cast(
                            list[JourneyTransition[JourneyState]], self._journey.transitions
                        ).append(cast(JourneyTransition[JourneyState], new_transition))

                    # Add target to queue for further processing
                    queue.append((target_state_id, target_mapped_state))
                else:
                    # Target state not in mapping - this is a transition to another journey
                    # Connect the source state to the fork state to exit this sub-journey
                    if mapped_source_state:
                        new_transition = await self._journey.create_transition(
                            condition=transition.condition,
                            source=mapped_source_state,
                            target=fork_state,
                        )
                        cast(
                            list[JourneyTransition[JourneyState]], self._journey.transitions
                        ).append(cast(JourneyTransition[JourneyState], new_transition))

        # We create a transient transition from self to the fork state to represent the exit point
        result_transition = JourneyTransition[JourneyState](
            id=JourneyTransitionId("transient"),
            condition=condition,
            source=self,
            target=fork_state,
            metadata={},
        )

        return result_transition


END_JOURNEY = JourneyState(
    id=JourneyStore.END_NODE_ID,
    action=None,
    tools=[],
    metadata={},
    description=None,
    _journey=None,
)
"""A special state used to indicate the end of a journey."""


def _validate_transition_parameters(
    *,
    condition: str | None = None,
    chat_state: str | None = None,
    tool_instruction: str | None = None,
    state: Any = None,
    tool_state: Any = None,
    journey: Any = None,
    canned_responses: Sequence[CannedResponseId] = [],
    metadata: Mapping[str, JSONSerializable] = {},
    on_match: Any = None,
    is_fork_state: bool = False,
) -> None:
    """Validate transition parameters against overload signatures."""

    # Determine which target parameter is being used
    target_param = None
    has_tool_state = tool_state and (
        isinstance(tool_state, ToolEntry)
        or (isinstance(tool_state, Sequence) and len(tool_state) > 0)
    )

    if state is not None:
        target_param = "state"
    elif chat_state is not None:
        target_param = "chat_state"
    elif has_tool_state:
        target_param = "tool_state"
    elif journey is not None:
        target_param = "journey"
    else:
        raise SDKError(
            "Must provide at least one target parameter: chat_state, state, tool_state, or journey."
        )

    # Check for multiple target parameters
    target_count = 0
    if state is not None:
        target_count += 1
    if chat_state is not None:
        target_count += 1
    if has_tool_state:
        target_count += 1
    if journey is not None:
        target_count += 1

    if target_count > 1:
        provided = []
        if state is not None:
            provided.append("state")
        if chat_state is not None:
            provided.append("chat_state")
        if has_tool_state:
            provided.append("tool_state")
        if journey is not None:
            provided.append("journey")
        raise SDKError(
            f"Cannot provide multiple target parameters simultaneously: {', '.join(provided)}. "
            "Please specify only one of: chat_state, state, tool_state, or journey."
        )

    # Validate parameter combinations based on overload signatures
    if target_param == "journey":
        # Journey overload: only condition and journey allowed
        invalid_params = []
        if canned_responses:
            invalid_params.append("canned_responses")
        if metadata:
            invalid_params.append("metadata")
        if on_match is not None:
            invalid_params.append("on_match")
        if tool_instruction is not None:
            invalid_params.append("tool_instruction")

        if invalid_params:
            raise SDKError(
                f"Journey transitions do not support the following parameters: {', '.join(invalid_params)}. "
                "Only 'condition' and 'journey' are allowed for journey transitions."
            )

    elif target_param == "tool_state":
        # Tool state overloads: tool_instruction is optional but other params should be allowed
        if tool_instruction is not None:
            # This is valid - tool_instruction + tool_state combination
            pass
        # canned_responses, metadata, on_match are all allowed for tool_state transitions

    elif target_param in ["state", "chat_state"]:
        # State and chat_state overloads: tool_instruction not allowed
        if tool_instruction is not None:
            raise SDKError(
                f"tool_instruction cannot be used with {target_param}. "
                "tool_instruction is only valid when using tool_state."
            )
        # canned_responses, metadata, on_match are all allowed

    # Special validation for ForkJourneyState
    if is_fork_state and target_param != "journey":
        if condition is None:
            raise SDKError(
                "ForkJourneyState requires a condition (except when transition to a journey)."
            )


class InitialJourneyState(JourneyState):
    """A special state used to indicate the initial state of a journey."""

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        state: TState,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[TState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ChatJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: ToolEntry,
        description: str | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: Sequence[ToolEntry],
        description: str | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        journey: Journey,
    ) -> JourneyTransition[ForkJourneyState]: ...

    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str | None = None,
        tool_instruction: str | None = None,
        state: TState | None = None,
        tool_state: ToolEntry | Sequence[ToolEntry] = [],
        journey: Journey | None = None,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[Any]:
        # Validate parameters against overload signatures
        _validate_transition_parameters(
            condition=condition,
            chat_state=chat_state,
            tool_instruction=tool_instruction,
            state=state,
            tool_state=tool_state,
            journey=journey,
            canned_responses=canned_responses,
            metadata=metadata,
            on_match=on_match,
            is_fork_state=False,
        )

        return await self._transition(
            condition=condition,
            state=state,
            action=chat_state or tool_instruction,
            description=description,
            tools=[tool_state] if isinstance(tool_state, ToolEntry) else tool_state,
            journey=journey,
            canned_responses=canned_responses,
            metadata=metadata,
            on_match=on_match,
            on_message=on_message,
            composition_mode=composition_mode,
            canned_response_field_provider=canned_response_field_provider,
        )


class ToolJourneyState(JourneyState):
    """A state in a journey that represents a tool being used."""

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        state: TState,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[TState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ChatJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: ToolEntry,
        description: str | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: Sequence[ToolEntry],
        description: str | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        journey: Journey,
    ) -> JourneyTransition[ForkJourneyState]: ...

    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str | None = None,
        tool_instruction: str | None = None,
        state: TState | None = None,
        tool_state: ToolEntry | Sequence[ToolEntry] = [],
        journey: Journey | None = None,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[Any]:
        # Validate parameters against overload signatures
        _validate_transition_parameters(
            condition=condition,
            chat_state=chat_state,
            tool_instruction=tool_instruction,
            state=state,
            tool_state=tool_state,
            journey=journey,
            canned_responses=canned_responses,
            metadata=metadata,
            on_match=on_match,
            is_fork_state=False,
        )

        return await self._transition(
            condition=condition,
            state=state,
            action=chat_state,
            description=description,
            tools=[tool_state] if isinstance(tool_state, ToolEntry) else tool_state,
            journey=journey,
            canned_responses=canned_responses,
            metadata=metadata,
            on_match=on_match,
            on_message=on_message,
            composition_mode=composition_mode,
            canned_response_field_provider=canned_response_field_provider,
        )

    async def fork(self) -> JourneyTransition[ForkJourneyState]:
        return await super()._fork()


class ChatJourneyState(JourneyState):
    """A state in a journey that represents a chat interaction."""

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        state: TState,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[TState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ChatJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: ToolEntry,
        description: str | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        tool_instruction: str | None = None,
        tool_state: Sequence[ToolEntry],
        description: str | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        journey: Journey,
    ) -> JourneyTransition[ForkJourneyState]: ...

    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str | None = None,
        tool_instruction: str | None = None,
        state: TState | None = None,
        tool_state: ToolEntry | Sequence[ToolEntry] = [],
        journey: Journey | None = None,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[Any]:
        # Validate parameters against overload signatures
        _validate_transition_parameters(
            condition=condition,
            chat_state=chat_state,
            tool_instruction=tool_instruction,
            state=state,
            tool_state=tool_state,
            journey=journey,
            canned_responses=canned_responses,
            metadata=metadata,
            on_match=on_match,
            is_fork_state=False,
        )

        return await self._transition(
            condition=condition,
            state=state,
            action=chat_state or tool_instruction,
            description=description,
            tools=[tool_state] if isinstance(tool_state, ToolEntry) else tool_state,
            journey=journey,
            canned_responses=canned_responses,
            metadata=metadata,
            on_match=on_match,
            on_message=on_message,
            composition_mode=composition_mode,
            canned_response_field_provider=canned_response_field_provider,
        )

    async def fork(self) -> JourneyTransition[ForkJourneyState]:
        return await super()._fork()


class ForkJourneyState(JourneyState):
    """A state in a journey that represents a conditional fork in the journey."""

    @overload
    async def transition_to(
        self,
        *,
        condition: str,
        state: TState,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[TState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str,
        chat_state: str,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ChatJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str,
        tool_instruction: str | None = None,
        tool_state: ToolEntry,
        description: str | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str,
        tool_instruction: str | None = None,
        tool_state: Sequence[ToolEntry],
        description: str | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[ToolJourneyState]: ...

    @overload
    async def transition_to(
        self,
        *,
        condition: str | None = None,
        journey: Journey,
    ) -> JourneyTransition[ForkJourneyState]: ...

    async def transition_to(
        self,
        *,
        condition: str | None = None,
        chat_state: str | None = None,
        tool_instruction: str | None = None,
        state: TState | None = None,
        tool_state: ToolEntry | Sequence[ToolEntry] = [],
        journey: Journey | None = None,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        composition_mode: CompositionMode | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[Any]:
        # Validate parameters against overload signatures
        _validate_transition_parameters(
            condition=condition,
            chat_state=chat_state,
            tool_instruction=tool_instruction,
            state=state,
            tool_state=tool_state,
            journey=journey,
            canned_responses=canned_responses,
            metadata=metadata,
            on_match=on_match,
            is_fork_state=True,
        )

        return await self._transition(
            condition=condition,
            state=state,
            action=chat_state or tool_instruction,
            description=description,
            tools=[tool_state] if isinstance(tool_state, ToolEntry) else tool_state,
            journey=journey,
            canned_responses=canned_responses,
            metadata=metadata,
            on_match=on_match,
            on_message=on_message,
            composition_mode=composition_mode,
            canned_response_field_provider=canned_response_field_provider,
        )


@dataclass(frozen=True)
class Journey:
    """A journey that consists of multiple states and transitions."""

    id: JourneyId
    title: str
    description: str
    conditions: list[Guideline]
    states: Sequence[JourneyState]
    transitions: Sequence[JourneyTransition[JourneyState]]
    tags: Sequence[TagId]
    composition_mode: CompositionMode | None

    _start_state_id: JourneyStateId
    _server: Server
    _container: Container

    @property
    def initial_state(self) -> InitialJourneyState:
        """Returns the initial state of the journey."""
        return cast(
            InitialJourneyState, next(n for n in self.states if n.id == self._start_state_id)
        )

    async def _create_state(
        self,
        state_type: type[TState],
        action: str | None = None,
        description: str | None = None,
        tools: Sequence[ToolEntry] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        composition_mode: CompositionMode | None = None,
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

        # Node-level composition_mode overrides journey-level
        # If no node-level composition_mode provided, inherit from journey
        effective_composition_mode = (
            composition_mode if composition_mode is not None else self.composition_mode
        )

        node = await self._container[JourneyStore].create_node(
            journey_id=self.id,
            action=action,
            tools=[
                ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name)
                for t in tools
            ],
            description=description,
            composition_mode=CompositionMode._to_core_composition_mode(effective_composition_mode),
        )

        node = await self._container[JourneyStore].set_node_metadata(
            node_id=node.id,
            key="journey_node",
            value={"kind": metadata_type},
        )

        for k, v in metadata.items():
            node = await self._container[JourneyStore].set_node_metadata(
                node_id=node.id,
                key=k,
                value=v,
            )

        return state_type(
            id=node.id,
            action=action,
            tools=tools,
            metadata=node.metadata,
            description=node.description,
            _journey=self,
        )

    @staticmethod
    async def _create_journey_state_handler_shim(
        user_callback: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]],
        state_id: JourneyStateId,
        transition_id: JourneyEdgeId,
        core_ctx: EngineContext,
        core_match: _GuidelineMatch,
    ) -> None:
        """Generic shim that translates core types to SDK JourneyStateMatch and calls user callback."""
        sdk_match = JourneyStateMatch(
            state_id=state_id,
            matched=True,
            rationale=core_match.rationale,
            transition_id=transition_id,
        )
        await user_callback(core_ctx, sdk_match)

    async def create_transition(
        self,
        condition: str | None,
        source: JourneyState,
        target: TState,
        on_match: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, JourneyStateMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> JourneyTransition[TState]:
        """Creates a transition between two states in the journey."""

        self._server._advance_creation_progress()

        transition = await self._container[JourneyStore].create_edge(
            journey_id=self.id,
            source=source.id,
            target=target.id if target else END_JOURNEY.id,
            condition=condition,
        )

        # Register handlers if provided
        if target is not None:
            if (
                on_match is not None
                or on_message is not None
                or canned_response_field_provider is not None
            ):
                guideline_id = format_journey_node_guideline_id(target.id, transition.id)
                engine_hooks = self._container[EngineHooks]

                if on_match is not None:
                    shim = partial(
                        Journey._create_journey_state_handler_shim,
                        on_match,
                        target.id,
                        transition.id,
                    )
                    engine_hooks.on_guideline_match_handlers[guideline_id].append(shim)

                if on_message is not None:
                    shim = partial(
                        Journey._create_journey_state_handler_shim,
                        on_message,
                        target.id,
                        transition.id,
                    )
                    engine_hooks.on_guideline_message_handlers[guideline_id].append(shim)

                if canned_response_field_provider is not None:
                    shim = partial(
                        Server._create_field_provider_shim,
                        canned_response_field_provider,
                    )
                    engine_hooks.on_guideline_match_handlers[guideline_id].append(shim)

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
        description: str | None = None,
        tools: Iterable[ToolEntry] = [],
        metadata: dict[str, JSONSerializable] = {},
        canned_responses: Sequence[CannedResponseId] = [],
        criticality: Criticality = Criticality.MEDIUM,
        composition_mode: CompositionMode | None = None,
        matcher: Callable[[GuidelineMatchingContext, Guideline], Awaitable[GuidelineMatch]]
        | None = None,
        on_match: Callable[[EngineContext, GuidelineMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, GuidelineMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
        id: GuidelineId | None = None,
    ) -> Guideline:
        """Creates a guideline with the specified condition and action, as well as (optionally) tools to achieve its task."""
        return await self._server._create_guideline(
            condition=condition,
            action=action,
            description=description,
            tools=tools,
            metadata=metadata,
            canned_responses=canned_responses,
            criticality=criticality,
            composition_mode=composition_mode,
            matcher=matcher,
            on_match=on_match,
            on_message=on_message,
            canned_response_field_provider=canned_response_field_provider,
            tags=None,
            relationship_target_tag_id=_Tag.for_journey_id(self.id),
            id=id,
        )

    async def create_observation(
        self,
        condition: str,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        composition_mode: CompositionMode | None = None,
        on_match: Callable[[EngineContext, GuidelineMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> Guideline:
        """A shorthand for creating an observational guideline with the specified condition."""

        return await self.create_guideline(
            condition=condition,
            description=description,
            canned_responses=canned_responses,
            composition_mode=composition_mode,
            on_match=on_match,
            canned_response_field_provider=canned_response_field_provider,
        )

    async def attach_tool(
        self,
        tool: ToolEntry,
        condition: str,
    ) -> GuidelineId:
        """Attaches a tool to the journey, to be usable by the agent under the specified condition."""

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
        metadata: Mapping[str, JSONSerializable] = {},
    ) -> CannedResponseId:
        """Creates a journey-scoped canned response with the specified template, tags, and signals."""

        self._server._advance_creation_progress()

        canrep = await self._container[CannedResponseStore].create_canned_response(
            value=template,
            tags=[_Tag.for_journey_id(self.id), *tags],
            fields=[],
            signals=signals,
            metadata=metadata,
        )

        return canrep.id

    async def prioritize_over(
        self,
        target: Guideline | Journey,
    ) -> Relationship:
        """Creates a priority relationship with another guideline or journey."""
        if isinstance(target, Guideline):
            return await self._create_relationship(
                target=target,
                kind=RelationshipKind.PRIORITY,
                direction="source",
            )
        else:
            return await self._create_relationship(
                target=target,
                kind=RelationshipKind.PRIORITY,
                direction="source",
            )

    async def depend_on(
        self,
        target: Guideline,
    ) -> Relationship:
        """Creates a dependency relationship with a guideline."""
        return await self._create_relationship(
            target=target,
            kind=RelationshipKind.DEPENDENCY,
            direction="source",
        )

    async def _create_relationship(
        self,
        target: Guideline | Journey,
        kind: RelationshipKind,
        direction: Literal["source", "target"],
    ) -> Relationship:
        if direction == "source":
            entity_source = RelationshipEntity(
                id=_Tag.for_journey_id(self.id), kind=RelationshipEntityKind.TAG
            )
            entity_target = (
                RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE)
                if isinstance(target, Guideline)
                else RelationshipEntity(
                    id=_Tag.for_journey_id(target.id), kind=RelationshipEntityKind.TAG
                )
            )
        else:
            entity_source = (
                RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE)
                if isinstance(target, Guideline)
                else RelationshipEntity(
                    id=_Tag.for_journey_id(target.id), kind=RelationshipEntityKind.TAG
                )
            )
            entity_target = RelationshipEntity(
                id=_Tag.for_journey_id(self.id), kind=RelationshipEntityKind.TAG
            )

        relationship = await self._container[RelationshipStore].create_relationship(
            source=entity_source,
            target=entity_target,
            kind=kind,
        )

        return Relationship(
            id=relationship.id,
            kind=relationship.kind,
            source=relationship.source.id,
            target=relationship.target.id,
        )


@dataclass(frozen=True)
class Capability:
    """A capability informs the agent about a specific functionality it can provide."""

    id: CapabilityId
    title: str
    description: str
    signals: Sequence[str]
    tags: Sequence[TagId]


@dataclass(frozen=True)
class Term:
    """A glossary term defines a specific concept in the agent's domain."""

    id: TermId
    name: str
    description: str
    synonyms: Sequence[str]
    tags: Sequence[TagId]


@dataclass(frozen=True)
class Variable:
    """A variable that can hold values for customers or customer groups."""

    id: ContextVariableId
    name: str
    description: str | None
    tool: ToolEntry | None
    freshness_rules: str | None
    tags: Sequence[TagId]
    _server: Server
    _container: Container

    async def set_value_for_customer(self, customer: Customer, value: JSONSerializable) -> None:
        """Sets the value of the variable for a specific customer."""

        await self._container[ContextVariableStore].update_value(
            variable_id=self.id,
            key=customer.id,
            data=value,
        )

    async def set_value_for_tag(self, tag: TagId, value: JSONSerializable) -> None:
        """Sets the value of the variable for a specific tag (e.g., a customer group tag)."""

        await self._container[ContextVariableStore].update_value(
            variable_id=self.id,
            key=f"tag:{tag}",
            data=value,
        )

    async def set_global_value(self, value: JSONSerializable) -> None:
        """Sets the global value of the variable, which is accessible to all customers by default."""

        await self._container[ContextVariableStore].update_value(
            variable_id=self.id,
            key=ContextVariableStore.GLOBAL_KEY,
            data=value,
        )

    async def get_value_for_customer(self, customer: Customer) -> JSONSerializable | None:
        """Retrieves the value of the variable for a specific customer."""

        value = await self._container[ContextVariableStore].read_value(
            variable_id=self.id,
            key=customer.id,
        )

        return value.data if value else None

    async def get_value_for_tag(self, tag: TagId) -> JSONSerializable | None:
        """Retrieves the value of the variable for a specific tag (e.g., a customer group tag)."""
        value = await self._container[ContextVariableStore].read_value(
            variable_id=self.id,
            key=f"tag:{tag}",
        )

        return value.data if value else None

    async def get_global_value(self) -> JSONSerializable | None:
        """Retrieves the global value of the variable, which is accessible to all customers by default."""

        value = await self._container[ContextVariableStore].read_value(
            variable_id=self.id,
            key=ContextVariableStore.GLOBAL_KEY,
        )

        return value.data if value else None

    async def get_value(self) -> JSONSerializable | None:
        """Retrieves the value of the variable for the current context"""
        value = EntityContext.get_variable_value(self.id)
        return value.data if value else None


@dataclass(frozen=True)
class Customer:
    """A customer represents an individual or entity interacting with the agent."""

    @classproperty
    def guest(cls: Customer) -> Customer:
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

    @classproperty
    def current(cls) -> Customer:
        """Get the current customer from the asyncio task context.

        Returns:
            The current customer as an SDK Customer object

        Raises:
            RuntimeError: If no customer is available in the current context
        """
        core_customer = EntityContext.get_customer()
        if core_customer is None:
            raise RuntimeError("No customer available in current context")

        return Customer(
            id=core_customer.id,
            name=core_customer.name,
            metadata=core_customer.extra,
            tags=core_customer.tags,
        )


@dataclass(frozen=True)
class RetrieverContext:
    """Context for retriever functions, providing helpful information for data retrieval."""

    server: Server
    container: Container
    logger: Logger
    tracer: Tracer
    session: Session
    agent: Agent
    customer: Customer
    variables: Mapping[Variable, JSONSerializable]
    interaction: Interaction

    @property
    def correlator(self) -> Tracer:
        self.logger.warning(
            "`correlator` is deprecated. Please change your code to use the `tracer` property"
        )
        return self.tracer


@dataclass(frozen=True)
class RetrieverResult:
    """Result of a retriever function, containing the retrieved data and metadata, as well (optionally) as canned response information."""

    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)
    canned_responses: Sequence[str] = field(default_factory=list)
    canned_response_fields: Mapping[str, Any] = field(default_factory=dict)


DeferredRetriever: TypeAlias = Callable[[EngineContext], Awaitable[RetrieverResult | None]]
"""A deferred retriever callable that receives a pre-response EngineContext and returns a RetrieverResult or None.

Returning this allows retrievers to start work in parallel during on_acknowledged, but defer the final decision
of what data to return (or whether to return any data at all) until on_generating_messages, when the
full EngineContext including matched guidelines and tool insights is available.
"""

RetrieverFunction: TypeAlias = Callable[
    [RetrieverContext], Awaitable[RetrieverResult | None | DeferredRetriever]
]
"""A retriever function that can either return a result directly, or return a deferred callable.

When a RetrieverResult or None is returned directly, it's used as-is.
When a DeferredRetriever is returned, it will be called later with the EngineContext to get the final result.
"""


class CompositionMode(enum.Enum):
    """Defines the composition mode for the agent, which determines how responses are generated."""

    FLUID = _CompositionMode.CANNED_FLUID
    """Responses are generated fluidly, allowing for dynamic composition of responses."""

    COMPOSITED = _CompositionMode.CANNED_COMPOSITED
    """Responses are generated in such a way as to mimic the style of the provided set of canned responses."""

    STRICT = _CompositionMode.CANNED_STRICT
    """Responses are generated strictly based on the provided canned responses, without fluidity."""

    @staticmethod
    def _to_core_composition_mode(mode: CompositionMode | None) -> _CompositionMode | None:
        if mode is None:
            return None
        return mode.value

    @staticmethod
    def _from_core_composition_mode(mode: _CompositionMode | None) -> CompositionMode | None:
        if mode is None:
            return None

        # Map core modes back to SDK modes
        if mode == _CompositionMode.CANNED_FLUID:
            return CompositionMode.FLUID
        elif mode == _CompositionMode.CANNED_COMPOSITED:
            return CompositionMode.COMPOSITED
        elif mode == _CompositionMode.CANNED_STRICT:
            return CompositionMode.STRICT
        else:
            # FLUID mode is not exposed in SDK, so return None
            return None


class ExperimentalAgentFeatures:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def create_capability(
        self,
        title: str,
        description: str,
        signals: Sequence[str] | None = None,
    ) -> Capability:
        """Creates a capability with the specified title, description, and signals."""

        self._agent._server._advance_creation_progress()

        capability = await self._agent._container[CapabilityStore].create_capability(
            title=title,
            description=description,
            signals=signals,
            tags=[_Tag.for_agent_id(self._agent.id)],
        )

        return Capability(
            id=capability.id,
            title=capability.title,
            description=capability.description,
            signals=capability.signals,
            tags=capability.tags,
        )


@dataclass(frozen=True)
class Agent:
    """An agent represents an entity that can interact with customers, manage journeys, and perform various tasks."""

    _server: Server
    _container: Container

    id: AgentId
    name: str
    description: str | None
    max_engine_iterations: int
    composition_mode: CompositionMode
    tags: Sequence[TagId]

    retrievers: Mapping[str, RetrieverFunction] = field(default_factory=dict)

    @property
    def experimental_features(self) -> ExperimentalAgentFeatures:
        """Provides access to experimental features of the agent."""
        return ExperimentalAgentFeatures(self)

    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: list[str | Guideline],
        id: JourneyId | None = None,
        composition_mode: CompositionMode | None = None,
    ) -> Journey:
        """Creates a new journey with the specified title, description, and conditions."""

        self._server._advance_creation_progress()

        journey = await self._server.create_journey(
            title, description, conditions, id=id, composition_mode=composition_mode
        )

        await self.attach_journey(journey)

        return Journey(
            id=journey.id,
            title=journey.title,
            description=description,
            conditions=journey.conditions,
            tags=journey.tags,
            states=journey.states,
            transitions=journey.transitions,
            composition_mode=journey.composition_mode,
            _start_state_id=journey._start_state_id,
            _server=self._server,
            _container=self._container,
        )

    async def attach_journey(self, journey: Journey) -> None:
        """Attaches an existing journey to the agent, allowing it to be used in interactions."""

        await self._container[JourneyStore].upsert_tag(
            journey.id,
            _Tag.for_agent_id(self.id),
        )

    async def create_guideline(
        self,
        condition: str,
        action: str | None = None,
        description: str | None = None,
        tools: Iterable[ToolEntry] = [],
        metadata: dict[str, JSONSerializable] = {},
        canned_responses: Sequence[CannedResponseId] = [],
        criticality: Criticality = Criticality.MEDIUM,
        composition_mode: CompositionMode | None = None,
        matcher: Callable[[GuidelineMatchingContext, Guideline], Awaitable[GuidelineMatch]]
        | None = None,
        on_match: Callable[[EngineContext, GuidelineMatch], Awaitable[None]] | None = None,
        on_message: Callable[[EngineContext, GuidelineMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
        id: GuidelineId | None = None,
    ) -> Guideline:
        """Creates a guideline with the specified condition and action, as well as (optionally) tools to achieve its task."""
        return await self._server._create_guideline(
            condition=condition,
            action=action,
            description=description,
            tools=tools,
            metadata=metadata,
            canned_responses=canned_responses,
            criticality=criticality,
            composition_mode=composition_mode,
            matcher=matcher,
            on_match=on_match,
            on_message=on_message,
            canned_response_field_provider=canned_response_field_provider,
            tags=[_Tag.for_agent_id(self.id)],
            relationship_target_tag_id=None,
            id=id,
        )

    async def create_observation(
        self,
        condition: str,
        description: str | None = None,
        canned_responses: Sequence[CannedResponseId] = [],
        criticality: Criticality = Criticality.MEDIUM,
        composition_mode: CompositionMode | None = None,
        on_match: Callable[[EngineContext, GuidelineMatch], Awaitable[None]] | None = None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None = None,
    ) -> Guideline:
        """A shorthand for creating an observational guideline with the specified condition."""

        return await self.create_guideline(
            condition=condition,
            description=description,
            canned_responses=canned_responses,
            composition_mode=composition_mode,
            on_match=on_match,
            criticality=criticality,
            canned_response_field_provider=canned_response_field_provider,
        )

    async def attach_tool(
        self,
        tool: ToolEntry,
        condition: str,
    ) -> GuidelineId:
        """Attaches a tool to the agent, to be usable under the specified condition."""

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
        metadata: Mapping[str, JSONSerializable] = {},
    ) -> CannedResponseId:
        """Creates a canned response with the specified template, tags, and signals."""

        self._server._advance_creation_progress()

        canrep = await self._container[CannedResponseStore].create_canned_response(
            value=template,
            tags=[_Tag.for_agent_id(self.id), *tags],
            fields=[],
            signals=signals,
            metadata=metadata,
        )

        return canrep.id

    async def create_term(
        self,
        name: str,
        description: str,
        synonyms: Sequence[str] = [],
        id: Optional[TermId] = None,
    ) -> Term:
        """Creates a glossary term with the specified name, description, and synonyms."""

        self._server._advance_creation_progress()

        term = await self._container[GlossaryStore].create_term(
            name=name,
            description=description,
            synonyms=synonyms,
            tags=[_Tag.for_agent_id(self.id)],
            id=id,
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
        """Creates a variable with the specified name, description, tool, and freshness rules."""

        self._server._advance_creation_progress()

        if tool:
            await self._server._plugin_server.enable_tool(tool)

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
        """Lists all variables associated with the agent."""

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
        """Finds a variable by its ID or name."""

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

    async def get_variable(self, id: ContextVariableId | str) -> Variable:
        """Retrieves a variable by its ID, raising an error if not found."""

        if variable := await self.find_variable(id=id):
            return variable
        raise SDKError(f"Variable with id {id} not found.")

    async def attach_retriever(
        self,
        retriever: RetrieverFunction,
        id: str | None = None,
    ) -> None:
        """Attaches a retriever function to the agent, allowing it to be used in interactions."""

        if not id:
            id = f"retriever-{len(self.retrievers) + 1}"

        cast(
            dict[str, RetrieverFunction],
            self.retrievers,
        )[id] = retriever

        self._server._retrievers[self.id][id] = retriever

    @classproperty
    def current(cls) -> Agent:
        """Get the current agent from the asyncio task context.

        Returns:
            The current agent as an SDK Agent object

        Raises:
            RuntimeError: If no agent is available in the current context
        """
        core_agent = EntityContext.get_agent()
        if core_agent is None:
            raise RuntimeError("No agent available in current context")

        # Get the current server and construct the Agent with the necessary references
        server = Server.current

        # Map core composition mode to SDK composition mode
        composition_mode_map = {
            _CompositionMode.FLUID: CompositionMode.FLUID,
            _CompositionMode.CANNED_FLUID: CompositionMode.FLUID,
            _CompositionMode.CANNED_COMPOSITED: CompositionMode.COMPOSITED,
            _CompositionMode.CANNED_STRICT: CompositionMode.STRICT,
        }

        return Agent(
            _server=server,
            _container=server._container,
            id=core_agent.id,
            name=core_agent.name,
            description=core_agent.description,
            max_engine_iterations=core_agent.max_engine_iterations,
            composition_mode=composition_mode_map[core_agent.composition_mode],
            tags=core_agent.tags,
        )


@dataclass(frozen=True)
class Session:
    """A session represents an ongoing conversation between a customer and an agent."""

    id: SessionId
    """The unique identifier of the session."""

    interaction: Interaction
    """The interaction history of this session."""

    @classproperty
    def current(cls) -> Session:
        """Get the current session from the asyncio task context.

        Returns:
            The current session as an SDK Session object

        Raises:
            RuntimeError: If no session is available in the current context
        """
        core_session = EntityContext.get_session()

        if core_session is None:
            raise RuntimeError("No session available in current context")

        interaction = EntityContext.get_interaction()

        if interaction is None:
            raise RuntimeError("No interaction available in current context")

        return Session(
            id=core_session.id,
            interaction=interaction,
        )


class ToolContextAccessor:
    """A context accessor for tools, providing access to the server and other relevant data."""

    def __init__(self, context: ToolContext) -> None:
        self.context = context

    @property
    def server(self) -> Server:
        """Returns the server associated with the tool context."""
        return cast(Server, self.context.plugin_data["server"])

    @property
    def current_interaction(self) -> Interaction:
        """Returns the engine context associated with the tool context."""
        interaction = EntityContext.get_interaction()

        if interaction is None:
            raise RuntimeError("No interaction available in current context")

        return interaction

    @property
    def logger(self) -> Logger:
        """Returns the logger associated with the context."""
        return self.server._container[Logger]


def _die(message: str, exc: Exception | None) -> NoReturn:
    if exc:
        import traceback

        traceback.print_exception(exc)
    rich.print(Text(message, style="bold red"), file=sys.stderr)
    sys.exit(1)


def _die_nlp_config_error(error: NLPServiceConfigurationError) -> NoReturn:
    console = Console(stderr=True)

    header = Text()
    header.append("🔧 ", style="bold")
    header.append("NLP SERVICE CONFIGURATION ERROR", style="bold white")

    content = Text(str(error), style="white")

    panel = Panel(
        content,
        title=header,
        title_align="left",
        border_style="white",
        box=rich.box.DOUBLE_EDGE,
        padding=(1, 3),
        width=100,
    )

    console.print()
    console.print(panel)
    console.print()
    sys.exit(1)


class Server:
    """The main server class that manages the agent, journeys, tools, and other components.

    This class is responsible for initializing the server, managing the lifecycle of the agent, and providing access to various services and components.

    Args:
        host: The NIC host to which the server will bind.
        port: The port on which the server will run.
        tool_service_port: The port for the integrated tool service.
        nlp_service: A factory function to create an NLP service instance. See `NLPServiceFactories` for available options.
        session_store: The session store to use for managing sessions.
        customer_store: The customer store to use for managing customers.
        log_level: The logging level for the server.
        modules: A list of module names to load for the server.
        migrate: Whether to allow database migrations on startup (if needed).
        configure_hooks: A callable to configure engine hooks.
        configure_container: A callable to configure the dependency injection container.
        initialize_container: A callable to perform additional initialization after the container is set up.
    """

    _current_server_var = contextvars.ContextVar[Optional["Server"]](
        "parlant_current_server", default=None
    )

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8800,
        tool_service_port: int = 8818,
        nlp_service: Callable[[Container], NLPService] = NLPServices.openai,
        session_store: Literal["transient", "local"] | str | SessionStore = "transient",
        customer_store: Literal["transient", "local"] | str | CustomerStore = "transient",
        variable_store: Literal["transient", "local"] | str | ContextVariableStore = "transient",
        log_level: LogLevel = LogLevel.INFO,
        modules: list[str] = [],
        migrate: bool = False,
        configure_hooks: Callable[[EngineHooks], Awaitable[EngineHooks]] | None = None,
        configure_container: Callable[[Container], Awaitable[Container]] | None = None,
        initialize_container: Callable[[Container], Awaitable[None]] | None = None,
        configure_api: Callable[[FastAPI], Awaitable[None]] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.tool_service_port = tool_service_port
        self.log_level = log_level
        self.modules = modules

        self._migrate = migrate
        self._nlp_service_func = nlp_service
        self._evaluator: _CachedEvaluator

        self._session_store = session_store
        self._customer_store = customer_store
        self._context_variable_store = variable_store

        self._configure_hooks = configure_hooks
        self._configure_container = configure_container
        self._initialize = initialize_container
        self._configure_api = configure_api
        self._retrievers: dict[
            AgentId,
            dict[str, RetrieverFunction],
        ] = defaultdict(dict)
        self._exit_stack = AsyncExitStack()

        self._finished_setup = False
        self._ready_event = asyncio.Event()

        self._plugin_server: PluginServer
        self._container: Container

        self._guideline_evaluations: dict[
            GuidelineId,
            tuple[Any, Callable[..., Coroutine[Any, Any, _CachedEvaluator.GuidelineEvaluation]]],
        ] = {}

        self._journey_evaluations: dict[
            JourneyId,
            tuple[Any, Callable[..., Coroutine[Any, Any, _CachedEvaluator.JourneyEvaluation]]],
        ] = {}

        self._creation_progress: Progress | None = Progress(
            TextColumn("{task.description}"),
            BarColumn(pulse_style="bold green"),
            TimeElapsedColumn(),
        )
        self._creation_progress_k = 0
        self._creation_progress_task_id: TaskID

    @property
    def container(self) -> Container:
        """Returns the dependency injection container."""
        return self._container

    @property
    def ready(self) -> asyncio.Event:
        """An asyncio event that is set when the server is ready to accept requests."""
        return self._ready_event

    @property
    def api(self) -> FastAPI:
        """Returns the FastAPI application instance.

        Raises:
            RuntimeError: If the server API is not yet initialized.
        """
        if not self._finished_setup:
            raise RuntimeError("Server API is not yet initialized. Wait for the server to start.")

        return self._container[FastAPI]

    def _advance_creation_progress(self) -> None:
        if self._creation_progress is None:
            return

        self._creation_progress_k += 1

        self._creation_progress.update(
            self._creation_progress_task_id,
            description=f"Caching entity embeddings ({self._creation_progress_k})",
        )

    async def __aenter__(self) -> Server:
        # Set this server instance as the current server in the context
        self._current_server_var.set(self)

        try:
            self._startup_context_manager = start_parlant(self._get_startup_params())
            self._container = await self._startup_context_manager.__aenter__()

            assert self._creation_progress
            self._creation_progress = self._creation_progress.__enter__()
            self._creation_progress_task_id = self._creation_progress.add_task(
                "Caching entity embeddings", total=None
            )

            return self

        except NLPServiceConfigurationError as e:
            _die_nlp_config_error(e)
        except SDKError as e:
            _die(str(e), e)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._finished_setup = True

        assert self._creation_progress
        self._creation_progress.__exit__(None, None, None)
        self._creation_progress = None

        with self._container[Tracer].span(
            "startup.evaluations",
            attributes={"scope": "Evaluations"},
        ):
            await self._process_evaluations()

        await self._setup_retrievers()

        # Start health check polling to set ready event when the server is ready to receive requests
        health_check_task = asyncio.create_task(self._poll_health_endpoint())

        # This actually starts the server
        await self._startup_context_manager.__aexit__(exc_type, exc_value, tb)

        # Wait for health check to complete before cleanup
        await health_check_task

        await self._exit_stack.aclose()
        return False

    # Start background task to poll health endpoint and set ready event
    async def _poll_health_endpoint(self) -> None:
        url = f"http://{self.host if self.host not in ['0.0.0.0', '127.0.0.1'] else 'localhost'}:{self.port}/healthz"

        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.get(url, timeout=30.0)

                    if response.status_code != 200:
                        self._container[Logger].critical("Health check failed.")
                        sys.exit(1)

                    self._ready_event.set()
                    self._container[Logger].info("Server is ready to accept requests.")
                    return
                except (httpx.RequestError, httpx.TimeoutException):
                    await asyncio.sleep(1)

    def _add_guideline_evaluation(
        self,
        guideline_id: GuidelineId,
        guideline_content: GuidelineContent,
        tool_ids: Sequence[ToolId],
    ) -> None:
        self._guideline_evaluations[guideline_id] = (
            (guideline_id, guideline_content, tool_ids),
            self._evaluator.evaluate_guideline,
        )

    def _add_journey_evaluation(
        self,
        journey: Journey,
    ) -> None:
        self._journey_evaluations[journey.id] = ((journey,), self._evaluator.evaluate_journey)

    @staticmethod
    async def _create_guideline_handler_shim(
        user_callback: Callable[[EngineContext, GuidelineMatch], Awaitable[None]],
        guideline_id: GuidelineId,
        core_ctx: EngineContext,
        core_match: _GuidelineMatch,
    ) -> None:
        """Generic shim that translates core types to SDK GuidelineMatch and calls user callback."""
        sdk_match = GuidelineMatch(
            id=guideline_id,
            matched=True,
            rationale=core_match.rationale,
        )
        await user_callback(core_ctx, sdk_match)

    @staticmethod
    async def _create_field_provider_shim(
        provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]],
        core_ctx: EngineContext,
        core_match: _GuidelineMatch,
    ) -> None:
        """Shim that calls a field provider and updates the context with the returned fields."""
        fields = await provider(core_ctx)
        core_ctx.state.additional_canned_response_fields.update(fields)

    async def _create_guideline(
        self,
        condition: str,
        action: str | None,
        description: str | None,
        tools: Iterable[ToolEntry],
        metadata: dict[str, JSONSerializable],
        criticality: Criticality,
        composition_mode: CompositionMode | None,
        canned_responses: Sequence[CannedResponseId],
        matcher: Callable[[GuidelineMatchingContext, Guideline], Awaitable[GuidelineMatch]] | None,
        on_match: Callable[[EngineContext, GuidelineMatch], Awaitable[None]] | None,
        on_message: Callable[[EngineContext, GuidelineMatch], Awaitable[None]] | None,
        canned_response_field_provider: Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
        | None,
        tags: Sequence[TagId] | None,
        relationship_target_tag_id: TagId | None,
        id: GuidelineId | None = None,
    ) -> Guideline:
        """Internal method to create a guideline with common logic."""
        self._advance_creation_progress()

        tool_ids = [
            ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name) for t in tools
        ]

        for t in list(tools):
            await self._plugin_server.enable_tool(t)

        guideline = await self.container[GuidelineStore].create_guideline(
            condition=condition,
            action=action,
            description=description,
            criticality=criticality,
            metadata=metadata,
            composition_mode=CompositionMode._to_core_composition_mode(composition_mode),
            id=id,
            tags=tags,
        )

        if canned_responses:
            tag_id = _Tag.for_guideline_id(guideline.id)

            for canrep_id in canned_responses:
                await self.container[CannedResponseStore].upsert_tag(
                    canned_response_id=canrep_id,
                    tag_id=tag_id,
                )

        # Evaluate what matcher to use if custom matcher isn't specified
        if matcher is None:
            self._add_guideline_evaluation(
                guideline.id,
                GuidelineContent(condition=condition, action=action),
                tool_ids,
            )

        # Create relationship if target tag specified
        if relationship_target_tag_id is not None:
            await self.container[RelationshipStore].create_relationship(
                source=RelationshipEntity(
                    id=guideline.id,
                    kind=RelationshipEntityKind.GUIDELINE,
                ),
                target=RelationshipEntity(
                    id=relationship_target_tag_id,
                    kind=RelationshipEntityKind.TAG,
                ),
                kind=RelationshipKind.DEPENDENCY,
            )

        for t in list(tools):
            await self.container[GuidelineToolAssociationStore].create_association(
                guideline_id=guideline.id,
                tool_id=ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=t.tool.name),
            )

        result_guideline = Guideline(
            id=guideline.id,
            condition=condition,
            action=action,
            tags=guideline.tags,
            metadata=guideline.metadata,
            _server=self,
            _container=self.container,
        )

        if matcher is not None:
            # Create a shim that translates between SDK and core types
            async def shim_matcher(
                core_ctx: _GuidelineMatchingContext, core_guideline: _Guideline
            ) -> _GuidelineMatch:
                sdk_ctx = await GuidelineMatchingContext._from_core(
                    core_ctx=core_ctx,
                    server=self,
                    container=self.container,
                )
                result = await matcher(sdk_ctx, result_guideline)

                return _GuidelineMatch(
                    guideline=core_guideline,
                    score=10 if result.matched else 1,
                    rationale=result.rationale,
                )

            strategy = CustomGuidelineMatchingStrategy(
                guideline=guideline,
                matcher=shim_matcher,
                logger=self.container[Logger],
            )

            self.container[GenericGuidelineMatchingStrategyResolver].guideline_overrides[
                guideline.id
            ] = strategy

        if (
            on_match is not None
            or on_message is not None
            or canned_response_field_provider is not None
        ):
            engine_hooks = self.container[EngineHooks]

            if on_match is not None:
                shim = partial(
                    Server._create_guideline_handler_shim,
                    on_match,
                    guideline.id,
                )
                engine_hooks.on_guideline_match_handlers[guideline.id].append(shim)

            if on_message is not None:
                shim = partial(
                    Server._create_guideline_handler_shim,
                    on_message,
                    guideline.id,
                )
                engine_hooks.on_guideline_message_handlers[guideline.id].append(shim)

            if canned_response_field_provider is not None:
                shim = partial(
                    Server._create_field_provider_shim,
                    canned_response_field_provider,
                )
                engine_hooks.on_guideline_match_handlers[guideline.id].append(shim)

        return result_guideline

    async def _render_guideline(self, guideline_id: GuidelineId) -> str:
        guideline = await self._container[GuidelineStore].read_guideline(guideline_id)

        return f"When {guideline.content.condition}" + (
            f", then {guideline.content.action}" if guideline.content.action else ""
        )

    async def _render_journey(self, journey_id: JourneyId) -> str:
        journey = await self._container[JourneyStore].read_journey(journey_id)

        return f"Journey: {journey.title}"

    async def _process_evaluations(self) -> None:
        _render_functions: dict[
            Literal["guideline", "journey"],
            Callable[[GuidelineId | JourneyId], Awaitable[str]],
        ] = {
            "guideline": self._render_guideline,  # type: ignore
            "journey": self._render_journey,  # type: ignore
        }

        def create_evaluation_task(
            evaluation: Coroutine[
                Any, Any, _CachedEvaluator.GuidelineEvaluation | _CachedEvaluator.JourneyEvaluation
            ],
            entity_type: Literal["guideline", "journey"],
            entity_id: GuidelineId | JourneyId,
        ) -> asyncio.Task[
            tuple[
                Literal["guideline", "journey"],
                GuidelineId | JourneyId,
                _CachedEvaluator.GuidelineEvaluation | _CachedEvaluator.JourneyEvaluation,
            ]
        ]:
            async def task_wrapper() -> tuple[
                Literal["guideline", "journey"],
                GuidelineId | JourneyId,
                _CachedEvaluator.GuidelineEvaluation | _CachedEvaluator.JourneyEvaluation,
            ]:
                result = await evaluation
                return (entity_type, entity_id, result)

            return asyncio.create_task(task_wrapper(), name=f"{entity_type}_evaluation_{entity_id}")

        tasks: list[
            asyncio.Task[
                tuple[
                    Literal["guideline", "journey"],
                    GuidelineId | JourneyId,
                    _CachedEvaluator.GuidelineEvaluation | _CachedEvaluator.JourneyEvaluation,
                ]
            ]
        ] = []

        for guideline_id, (args, func) in self._guideline_evaluations.items():
            tasks.append((create_evaluation_task(func(*args), "guideline", guideline_id)))

        for journey_id, (args, journey_func) in self._journey_evaluations.items():
            tasks.append((create_evaluation_task(journey_func(*args), "journey", journey_id)))

        if not tasks:
            return

        if self.log_level == LogLevel.TRACE:
            evaluation_results = await async_utils.safe_gather(*tasks)
        else:
            max_visible = 5

            overall_progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                TaskProgressColumn(style="bold blue"),
                TimeElapsedColumn(),
            )

            entity_progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                TaskProgressColumn(style="bold blue"),
                TimeElapsedColumn(),
                transient=True,
            )

            with Live(Group(overall_progress, entity_progress), refresh_per_second=10):
                bar_id: dict[str, int] = {}

                for t in tasks:
                    entity_id = cast(GuidelineId | JourneyId, t.get_name().split("_")[-1])
                    entity_type = t.get_name().split("_")[0]
                    description = await _render_functions[
                        cast(Literal["guideline", "journey"], entity_type)
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

            elif entity_type == "journey":
                for node_id, properties in cast(
                    _CachedEvaluator.JourneyEvaluation, result
                ).node_properties.items():
                    if node_id == END_JOURNEY.id:
                        continue

                    node = await self._container[JourneyStore].read_node(node_id)
                    properties_to_add = {
                        k: v
                        for k, v in properties.items()
                        if k not in node.metadata or node.metadata[k] is None
                    }

                    journey_node_properties = {
                        **(
                            cast(dict[str, JSONSerializable], properties.get("journey_node", {}))
                            if properties
                            else {}
                        ),
                        **cast(dict[str, JSONSerializable], node.metadata.get("journey_node", {})),
                    }
                    if journey_node_properties:
                        properties_to_add["journey_node"] = journey_node_properties

                    for key, value in properties_to_add.items():
                        await self._container[JourneyStore].set_node_metadata(
                            node_id=node_id,
                            key=key,
                            value=value,
                        )

        print()

    async def _setup_retrievers(self) -> None:
        async def setup_retriever(
            c: Container,
            agent_id: AgentId,
            retriever_id: str,
            retriever: RetrieverFunction,
        ) -> None:
            tasks_for_this_retriever: dict[
                str,
                tuple[Timeout, asyncio.Task[RetrieverResult | None | DeferredRetriever]],
            ] = {}

            async def on_message_acknowledged(
                ctx: EngineContext,
                payload: Any,
                exc: Optional[Exception],
            ) -> EngineHookResult:
                # First do some garbage collection if needed.
                # This might be needed if tasks were not awaited
                # because of exceptions during engine processing.
                for trace_id in list(tasks_for_this_retriever.keys()):
                    if tasks_for_this_retriever[trace_id][0].expired():
                        # Very, very little change that this task is still meant to be running,
                        # or that anyone is still waiting for it. It's 99.999% garbage.
                        try:
                            tasks_for_this_retriever[trace_id][1].add_done_callback(
                                default_done_callback()
                            )
                            tasks_for_this_retriever[trace_id][1].cancel()
                            del tasks_for_this_retriever[trace_id]
                        except BaseException:
                            # If anything went unexpectedly here, whatever. Carry on.
                            pass

                agent = await self.get_agent(id=ctx.agent.id)
                customer = await self.get_customer(id=ctx.customer.id)

                coroutine = retriever(
                    RetrieverContext(
                        server=self,
                        container=self._container,
                        logger=self._container[Logger],
                        tracer=self._container[Tracer],
                        session=Session(
                            id=ctx.session.id,
                            interaction=ctx.interaction,
                        ),
                        agent=agent,
                        customer=customer,
                        variables={
                            await agent.get_variable(id=var.id): val.data
                            for var, val in ctx.state.context_variables
                        },
                        interaction=ctx.interaction,
                    )
                )

                c[Logger].trace(
                    f"Starting retriever {retriever_id} for agent {agent_id} with trace {ctx.tracer.trace_id}"
                )

                tasks_for_this_retriever[ctx.tracer.trace_id] = (
                    Timeout(600),  # Expiration timeout for garbage collection purposes
                    asyncio.create_task(
                        cast(
                            Coroutine[Any, Any, RetrieverResult | None | DeferredRetriever],
                            coroutine,
                        ),
                        name=f"Retriever {retriever_id} for agent {agent_id}",
                    ),
                )

                return EngineHookResult.CALL_NEXT

            async def on_generating_messages(
                ctx: EngineContext,
                payload: Any,
                exc: Optional[Exception],
            ) -> EngineHookResult:
                if timeout_and_task := tasks_for_this_retriever.pop(ctx.tracer.trace_id, None):
                    _, task = timeout_and_task
                    task_result = await task

                    # Check if the result is a deferred callable
                    if callable(task_result):
                        # Call the deferred callable with the EngineContext
                        final_result = await task_result(ctx)
                        if final_result is None:
                            # Deferred callable decided not to return data
                            return EngineHookResult.CALL_NEXT
                        task_result = final_result

                    # Handle None result
                    if task_result is None:
                        return EngineHookResult.CALL_NEXT

                    # task_result must be a RetrieverResult at this point
                    retriever_result = task_result

                    if not (
                        retriever_result.data
                        or retriever_result.metadata
                        or retriever_result.canned_responses
                        or retriever_result.canned_response_fields
                    ):
                        # No need to emit tool event if nothing was retrieved.
                        return EngineHookResult.CALL_NEXT

                    ctx.state.tool_events.append(
                        await ctx.response_event_emitter.emit_tool_event(
                            ctx.tracer.trace_id,
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
                                                u for u in retriever_result.canned_responses
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
        id: str | None = None,
        perceived_performance_policy: PerceivedPerformancePolicy | None = None,
    ) -> Agent:
        """Creates a new agent with the specified name, description, and composition mode.

        Args:
            name: The agent's name (required).
            description: A description of the agent's purpose and capabilities (required).
            composition_mode: How the agent composes responses. Defaults to FLUID.
                - FLUID: Dynamic response composition
                - CANNED_FLUID: Mix of canned and dynamic responses
                - CANNED_COMPOSITED: Composed from canned responses
                - CANNED_STRICT: Strictly uses canned responses
            max_engine_iterations: Maximum number of engine iterations per turn.
                Defaults to 3 if not specified.
            tags: List of tag IDs to associate with the agent. Defaults to empty list.
            id: Custom agent ID string (optional). If not provided, an ID will be
                automatically generated based on the agent's properties. Custom IDs
                can be any string format and are useful for maintaining consistent
                agent identifiers across deployments or integrations.
            perceived_performance_policy: Optional perceived performance policy for this agent.
                If not specified, the agent will use the default policy (BasicPerceivedPerformancePolicy).

        Returns:
            The created Agent instance.
        """

        self._advance_creation_progress()

        agent = await self._container[AgentStore].create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations or 3,
            composition_mode=composition_mode.value,
            id=AgentId(id) if id is not None else None,
        )

        if perceived_performance_policy is not None:
            self._container[PerceivedPerformancePolicyProvider].set_policy(
                agent.id, perceived_performance_policy
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
        """Lists all agents."""

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
        """Finds an agent by its ID."""

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
        """Retrieves an agent by its ID, raising an error if not found."""

        if agent := await self.find_agent(id=id):
            return agent
        raise SDKError(f"Agent with id {id} not found.")

    async def create_customer(
        self,
        name: str,
        metadata: Mapping[str, str] = {},
        tags: Sequence[TagId] = [],
        id: str | None = None,
    ) -> Customer:
        """Creates a new customer with the specified name and metadata.

        Args:
            name: The customer's name (required). An arbitrary string that
                identifies and/or describes the customer.
            metadata: Key-value pairs to describe the customer. Defaults to
                empty dictionary. This allows you to store arbitrary metadata
                about the customer (e.g., email, VIP status, preferences).
            tags: List of tag IDs to associate with the customer. Defaults to
                empty list. Tags are useful for categorizing and filtering
                customers.
            id: Custom customer ID string (optional). If not provided, an ID
                will be automatically generated based on the customer's
                properties. Custom IDs can be any string format and are useful
                for maintaining consistent customer identifiers across
                deployments or integrations (e.g., matching your internal
                customer IDs).

        Returns:
            The created Customer instance.
        """

        self._advance_creation_progress()

        customer = await self._container[CustomerStore].create_customer(
            name=name,
            extra=metadata,
            tags=tags,
            id=CustomerId(id) if id is not None else None,
        )

        return Customer(
            id=customer.id,
            name=customer.name,
            metadata=customer.extra,
            tags=customer.tags,
        )

    async def list_customers(self) -> Sequence[Customer]:
        """Lists all customers."""

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
        """Finds a customer by its ID or name."""

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
        """Retrieves a customer by its ID, raising an error if not found."""

        if customer := await self.find_customer(id=id):
            return customer
        raise SDKError(f"Customer with id {id} not found.")

    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: list[str | Guideline],
        tags: Sequence[TagId] = [],
        id: JourneyId | None = None,
        composition_mode: CompositionMode | None = None,
    ) -> Journey:
        """Creates a new journey with the specified title, description, and conditions."""

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
            id=id,
            composition_mode=CompositionMode._to_core_composition_mode(composition_mode),
        )

        journey = Journey(
            id=stored_journey.id,
            title=title,
            description=description,
            conditions=condition_guidelines,
            states=[],
            transitions=[],
            tags=tags,
            composition_mode=CompositionMode._from_core_composition_mode(
                stored_journey.composition_mode
            ),
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
                description=start_state.description,
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

    async def create_canned_response(
        self,
        template: str,
        tags: list[TagId] = [],
        signals: list[str] = [],
        metadata: Mapping[str, JSONSerializable] = {},
    ) -> CannedResponseId:
        """Creates a canned response with the specified template, tags, and signals."""

        self._advance_creation_progress()

        canrep = await self._container[CannedResponseStore].create_canned_response(
            value=template,
            tags=tags,
            fields=[],
            signals=signals,
            metadata=metadata,
        )

        return canrep.id

    def _get_startup_params(self) -> StartupParameters:
        async def override_stores_with_transient_versions(c: Callable[[], Container]) -> None:
            c()[NLPService] = self._nlp_service_func(c())

            for interface, implementation in [
                (AgentStore, AgentDocumentStore),
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

            if isinstance(self._context_variable_store, ContextVariableStore):
                c()[ContextVariableStore] = self._context_variable_store
            else:
                c()[ContextVariableStore] = await make_persistable_store(
                    ContextVariableDocumentStore,
                    self._context_variable_store,
                    "context_variables",
                    id_generator=c()[IdGenerator],
                )

            c()[ServiceRegistry] = await self._exit_stack.enter_async_context(
                ServiceDocumentRegistry(
                    database=TransientDocumentDatabase(),
                    event_emitter_factory=c()[EventEmitterFactory],
                    logger=c()[Logger],
                    tracer=c()[Tracer],
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
                            c()[Tracer],
                            embedder_factory,
                            lambda: c()[EmbeddingCache],
                        ),
                        document_db=TransientDocumentDatabase(),
                        embedder_factory=embedder_factory,
                        embedder_type_provider=get_embedder_type,
                    )  # type: ignore
                )

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
            host=self.host,
            port=self.port,
            nlp_service=async_nlp_service_shim,
            log_level=self.log_level,
            modules=self.modules,
            migrate=self._migrate,
            configure=configure,
            initialize=initialize,
            configure_api=self._configure_api,
            contextvar_propagation={
                self._current_server_var: self,
            },
        )

    @classproperty
    def current(cls: Server) -> Server:
        """Get the current server from the asyncio task context.

        Returns:
            The current server instance

        Raises:
            RuntimeError: If no server is available in the current context
        """
        server = cls._current_server_var.get()
        if server is None:
            raise RuntimeError("No server available in current context")
        return server


__all__ = [
    "Agent",
    "AgentId",
    "AuthorizationException",
    "AuthorizationPolicy",
    "BasicNoMatchResponseProvider",
    "BasicOptimizationPolicy",
    "BasicPerceivedPerformancePolicy",
    "BasicRateLimiter",
    "CannedResponseId",
    "Capability",
    "CapabilityId",
    "CompositionMode",
    "Container",
    "ContextVariableId",
    "ContextVariableStore",
    "ControlOptions",
    "Criticality",
    "Customer",
    "CustomerId",
    "CustomerModerationContext",
    "CustomerStore",
    "DefaultBaseModel",
    "DeferredRetriever",
    "DevelopmentAuthorizationPolicy",
    "END_JOURNEY",
    "Embedder",
    "EmbedderFactory",
    "EmbedderHints",
    "EmbeddingResult",
    "EmittedEvent",
    "EngineContext",
    "EngineHook",
    "EngineHookResult",
    "EngineHooks",
    "EstimatingTokenizer",
    "EventKind",
    "EventSource",
    "FallbackSchematicGenerator",
    "Guideline",
    "GuidelineId",
    "GuidelineMatchingContext",
    "Interaction",
    "InteractionMessage",
    "JSONSerializable",
    "Journey",
    "JourneyId",
    "JourneyState",
    "JourneyStateId",
    "JourneyStateMatch",
    "JourneyTransition",
    "JourneyTransitionId",
    "Lifespan",
    "LoadedContext",
    "LogLevel",
    "Logger",
    "MessageEventData",
    "ModelGeneration",
    "ModelSize",
    "ModelType",
    "ModerationCheck",
    "ModerationService",
    "ModerationTag",
    "NLPService",
    "NLPServices",
    "NoMatchResponseProvider",
    "NoModeration",
    "NullPerceivedPerformancePolicy",
    "Operation",
    "OptimizationPolicy",
    "PerceivedPerformancePolicy",
    "PerceivedPerformancePolicyProvider",
    "PluginServer",
    "ProductionAuthorizationPolicy",
    "PromptBuilder",
    "PromptSection",
    "RateLimitExceededException",
    "RateLimiter",
    "RelationshipEntity",
    "RelationshipEntityId",
    "RelationshipEntityKind",
    "RelationshipId",
    "RelationshipKind",
    "RetrieverContext",
    "RetrieverFunction",
    "RetrieverResult",
    "SchematicGenerationResult",
    "SchematicGenerator",
    "SchematicGeneratorHints",
    "Server",
    "ServiceRegistry",
    "Session",
    "SessionId",
    "SessionMode",
    "SessionStatus",
    "SessionStore",
    "StatusEventData",
    "T",
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
    "Tracer",
    "Variable",
    "Variable",
    "VoiceOptimizedPerceivedPerformancePolicy",
    "tool",
]
