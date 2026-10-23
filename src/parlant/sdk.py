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
import os
import uuid
from collections import defaultdict
from contextlib import AsyncExitStack, suppress
import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import partial
import xxhash
import importlib.util
import inspect
from itertools import chain
from pathlib import Path
import sys
import signal
import watchfiles
import rich
from rich.console import Group as RichGroup
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
from types import ModuleType, TracebackType
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Set,
    TypeVar,
    TypeAlias,
    TypedDict,
    cast,
)
from fastapi import FastAPI
import httpx
from parlant.core.container import Container, Singleton


from parlant.adapters.db.json_file import JSONFileDocumentCollection, JSONFileDocumentDatabase
from parlant.adapters.db.sqlite import SQLiteDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.api.authentication import (
    AdminPrincipal,
    AnonymousPrincipal,
    Authenticator,
    AuthenticationError,
    CustomerPrincipal,
    GuestPrincipal,
    Principal,
    TokenAuthenticator,
)
from parlant.core.health.thresholds import (
    EngineHealthThresholds,
    EventLoopHealthThresholds,
    HealthThresholds,
    NLPHealthThresholds,
    default_nlp_route_thresholds,
)
from parlant.core.health.nlp_view import RouteThresholds
from parlant.api.authorization import (
    AuthConfig,
    AuthorizationException,
    Authorizer,
    CompositeAuthorizationPolicy,
    DefaultPermissionGate,
    Operation,
    AuthorizationPolicy,
    BasicRateLimiter,
    DevelopmentAuthorizationPolicy,
    HeaderApiKeyAuthorizationPolicy,
    OwnershipAuthorizer,
    PermissionGate,
    RateLimitExceededException,
    RateLimiter,
    create_composite_authorization_policy,
)


from parlant.core import async_utils
from parlant.core.application import Application as _Application
from parlant.core.agents import (
    Moderation,
    AgentDocumentStore,
    AgentId,
    AgentStore,
    Effort,
    MessageOutputMode as _MessageOutputMode,
)
from parlant.core.async_utils import Timeout, default_done_callback
from parlant.core.common import (
    Weight,
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
from parlant.core.emission.event_publisher import EventPublisherFactory
from parlant.core.emission.session_tap import SessionEventTap
from parlant.core.tracer import Tracer
from parlant.core.pricing import (
    ModelPricing,
    Pricing as _Pricing,
    PricingRegistry,
    SessionBudgetDefinition,
    SessionBudgetService,
)
from parlant.core.customers import (
    Customer as _Customer,
    CustomerDocumentStore,
    CustomerId,
    CustomerKind,
    CustomerStore,
)
from parlant.core.emissions import EmittedEvent, EventEmitter, EventEmitterFactory
from parlant.core.application_context import ApplicationContext
from parlant.core.environment import Environment, resolve_environment
from parlant.core.engines.types import (
    UtteranceRationale as _UtteranceRationale,
    UtteranceRequest as _UtteranceRequest,
)
from parlant.core.engines.compass.prompt_builder import (
    VETO,
    PromptBuilder,
    PromptSection,
    SectionAdaptation,
)
from parlant.core.nlp.prompt_routing import (
    PromptConfigurationService,
    PromptProvider,
    PromptRoutes,
)
from parlant.core.engines.compass.compacter import PromptSections as _CompacterPromptSections
from parlant.core.engines.compass.matching.common import PromptSections as _SharedPromptSections
from parlant.core.engines.compass.matching.rule_pruner import (
    PromptSections as _PrunerPromptSections,
)
from parlant.core.engines.compass.matching.rule_ranker import (
    PromptSections as _RankerPromptSections,
)
from parlant.core.engines.compass.responder.responder import (
    PromptSections as _ResponderPromptSections,
)
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.engines.compass.reviewer import PromptSections as _ReviewerPromptSections


from parlant.core.engines.compass.hooks import (
    Compacted as _CompactedPayload,
    CompassHook,
    CompassHookResult,
    CompassHooks,
    GaveUp as _GaveUpPayload,
    GiveUpKind as _GiveUpKind,
    MessageReviewed as _MessageReviewedPayload,
    ReviewingMessage as _ReviewingMessagePayload,
    ReviewingToolCalls as _ReviewingToolCallsPayload,
    ToolCallOutcome as _ToolCallOutcomePayload,
    ToolCallsReviewed as _ToolCallsReviewedPayload,
)
from parlant.core.engines.engine_context import (
    EngineContext,
    Interaction,
    InteractionMessage,
)
from parlant.core.engines.entity_context import EntityContext
from parlant.core.engines.session_metadata_workspace import (
    SessionMetadataWorkspace,
    WorkspaceState,
)
from parlant.core import metadata as metadata_utils
from parlant.core.engines.rule_evaluation import RuleEvaluation as _RuleEvaluation
from parlant.core.engines.rule_evaluator_registry import (
    RuleEvaluatorPhase,
    RuleEvaluatorRegistry,
)
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.engines.compass.variable_loader import load_fresh_context_variable_value
from parlant.core.glossary import (
    GlossaryStore,
    GlossaryVectorStore,
    Term as _CoreTerm,
    TermId,
)
from parlant.core.rule_tool_associations import (
    RuleToolAssociationDocumentStore,
    RuleToolAssociationStore,
)
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderHints,
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
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.indexing.indexing_service import (
    IndexingService,
    RuleIndexingProcessor,
)
from parlant.core.services.training_service import TrainingService
from parlant.core.services.tools.service_registry import ServiceDocumentRegistry, ServiceRegistry
from parlant.core.sessions import (
    AttachmentRef,
    Event,
    EventKind,
    EventSource,
    MessageEventData,
    SessionId,
    SessionDocumentStore,
    SessionStore,
    SessionUpdateParams as _SessionUpdateParams,
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
from parlant.core.indexing import (
    IndexingDocumentStore,
    IndexingStatus,
    IndexingStore,
    RulePayload,
    PayloadOperation,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.rules import (
    Rule as _Rule,
    RuleVectorStore,
    RuleId,
    RuleStore,
    RuleUpdateParams,
    compose_rule_query,
)
from parlant.core.loggers import LogLevel, Logger
from parlant.core.nlp.common import ModelGeneration, ModelSize, ModelType

from parlant.core.nlp.moderation import (
    CustomerModerationContext,
    ModerationCheck,
    ModerationService,
    ModerationTag,
    NoModeration,
)
from parlant.core.nlp.react import ReasoningConfig as ReasoningConfig
from parlant.core.nlp.model_routing import (
    InvalidModelArgumentsError as InvalidModelArgumentsError,
    InvalidModelProviderResultError as InvalidModelProviderResultError,
    InvalidModelRouteError as InvalidModelRouteError,
    MappingSecretLoader as MappingSecretLoader,
    MissingModelRouteError as MissingModelRouteError,
    ModelConfigurationError as ModelConfigurationError,
    ModelConfigurationService,
    ModelConfig as ModelConfig,
    ReasoningEffort as ReasoningEffort,
    ModelImplementationError as ModelImplementationError,
    ModelProvider as ModelProvider,
    ModelProviderExecutionError as ModelProviderExecutionError,
    ModelProviderUnavailableError as ModelProviderUnavailableError,
    ModelRouter,
    ModelRouteUpdates,
    ModelRouteValue,
    ModelRoutes,
    ModelScope,
    ModelRouteStore,
    SchematicSchemaMismatchError as SchematicSchemaMismatchError,
    SecretLoader as SecretLoader,
    SecretRef as SecretRef,
    SecretValue as SecretValue,
    UnknownModelProviderError as UnknownModelProviderError,
    UnsupportedModelCapabilityError as UnsupportedModelCapabilityError,
    UnsupportedRateLimitCapabilityError as UnsupportedRateLimitCapabilityError,
)
from parlant.core.nlp.rate_limits import (
    RateLimit as RateLimit,
    RateLimitConflictError as RateLimitConflictError,
    RateLimitError as RateLimitError,
    TransientRateLimiter as TransientRateLimiter,
)
from parlant.core.nlp.model_route_store import ModelRouteDocumentStore
from parlant.core.engines.compass.preambles import (
    PreambleConfiguration as PreambleConfiguration,
    PreambleConfigurationProvider as PreambleConfigurationProvider,
    PreambleDecision as PreambleDecision,
    PreambleEmissionPolicy as PreambleEmissionPolicy,
)
from parlant.bin.server import (
    PARLANT_HOME_DIR,
    StartupParameters,
    configure_model_routing,
    start_parlant,
)
from parlant.core.store_provider import BasicStoreProvider, StoreProvider, StoreProviderHints
from parlant.core.services.tools.plugins import PluginServer, ToolEntry, tool
from parlant.core.groups import GroupIds, GroupDocumentStore, GroupId, GroupStore
from parlant.core.tools import (
    ControlOptions,
    Lifespan,
    Narration,
    NarrationFn,
    SessionMode,
    SessionStatus,
    Tool,
    ToolContext,
    TransientRule,
    ToolId,
    ToolOverlap,
    ToolParameterDescriptor,
    ToolParameterOptions,
    ToolParameterType,
    ToolResult,
)
from parlant.core.version import VERSION


class CompassHookPayload:
    """Single roof for the engine-hook payload data classes.

    Engine hooks are called as ``(context, payload, exc)``; the payload's type
    depends on the hook. Annotate handlers with the member matching the hook,
    e.g. ``payload: p.CompassHookPayload.ToolCallsReviewed`` for
    ``on_tool_calls_reviewed``.
    """

    Compacted: TypeAlias = _CompactedPayload
    GaveUp: TypeAlias = _GaveUpPayload
    GiveUpKind: TypeAlias = _GiveUpKind
    MessageReviewed: TypeAlias = _MessageReviewedPayload
    ReviewingMessage: TypeAlias = _ReviewingMessagePayload
    ReviewingToolCalls: TypeAlias = _ReviewingToolCallsPayload
    ToolCallOutcome: TypeAlias = _ToolCallOutcomePayload
    ToolCallsReviewed: TypeAlias = _ToolCallsReviewedPayload


class PromptSections:
    """The compass prompt-section registries, aggregated for discovery.

    Each attribute is the ``PromptSections`` enum a compass component exposes
    beside its prompt — the section names a prompt provider (see the
    ``prompts`` parameter of :class:`Server` and ``create_agent``) can match
    on. Autocompleting these enums is the registry: every name is stable,
    documented API. Additional engine components may expose their own section
    registries alongside their implementations.
    """

    Ranker = _RankerPromptSections
    Pruner = _PrunerPromptSections
    MatchingCommon = _SharedPromptSections
    Compacter = _CompacterPromptSections
    Reviewer = _ReviewerPromptSections
    Responder = _ResponderPromptSections


OutputMode = _MessageOutputMode

INTEGRATED_TOOL_SERVICE_NAME = "built-in"

ToolRef: TypeAlias = ToolEntry | ToolId
"""A reference to a tool: either a ``ToolEntry`` (hosted on the integrated
plugin server) or a ``ToolId`` (hosted on an external tool service)."""

T = TypeVar("T")


def _should_configure_parlant_dashboard() -> bool:
    # The project token is the module's only credential.
    return bool(os.environ.get("PARLANT_DASHBOARD_PROJECT_TOKEN"))


def _tool_ref_to_id(ref: ToolRef) -> ToolId:
    """Convert a ToolRef to a ToolId."""
    if isinstance(ref, ToolId):
        return ref
    return ToolId(service_name=INTEGRATED_TOOL_SERVICE_NAME, tool_name=ref.tool.name)


async def _enable_tool_refs(plugin_server: PluginServer, refs: Iterable[ToolRef]) -> None:
    """Enable only ToolEntry refs on the plugin server; skip ToolId refs."""
    for ref in refs:
        if isinstance(ref, ToolEntry):
            await plugin_server.enable_tool(ref)


class SDKError(Exception):
    """Main class for SDK-related errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class _MongoConnectionSettings:
    uri: str = field(repr=False)
    database_name: str


def _parse_mongo_connection_settings(
    uri: str,
    parse_uri: Callable[[str], Mapping[str, Any]],
) -> _MongoConnectionSettings:
    database_name = cast(Optional[str], parse_uri(uri).get("database")) or "parlant"
    return _MongoConnectionSettings(uri=uri, database_name=database_name)


def _ensure_same_mongo_connection(
    established: _MongoConnectionSettings,
    requested: _MongoConnectionSettings,
) -> None:
    if established != requested:
        raise SDKError("All persisted stores must use the same MongoDB connection and database.")


class _CachedRuleIndexingResult(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    version: Version.String
    references_other_policies: bool | None
    signals: list[str]
    anti_signals: list[str]
    title: str | None


@dataclass(frozen=True)
class _RuleText:
    """A rule's authored text, identity-free — the cached indexer's cache key
    and payload input."""

    content: str | None = None
    condition: str | None = None


class _CachedIndexer:
    _RULE_INDEXING_CACHE_VERSION = "rule-indexing-v7-flat-rules"

    @dataclass(frozen=True)
    class RuleIndexingResult:
        references_other_policies: bool | None
        signals: Sequence[str]
        anti_signals: Sequence[str]
        title: str | None = None

    def __init__(
        self,
        db: JSONFileDocumentDatabase,
        container: Container,
    ) -> None:
        self._db: JSONFileDocumentDatabase = db
        self._rule_collection: JSONFileDocumentCollection[_CachedRuleIndexingResult]

        self._container = container
        self._store_provider = container[StoreProvider]
        self._logger = container[Logger]
        self._exit_stack = AsyncExitStack()
        self._progress: dict[str, float] = {}

    def _set_progress(self, key: str, pct: float) -> None:
        self._progress[key] = max(0.0, min(pct, 100.0))

    def _progress_for(self, key: str) -> float:
        return self._progress.get(key, 0.0)

    async def __aenter__(self) -> _CachedIndexer:
        await self._exit_stack.enter_async_context(self._db)

        self._rule_collection = await self._db.get_or_create_collection(
            name=f"rule_indexing_{VERSION}",
            schema=_CachedRuleIndexingResult,
            document_loader=identity_loader_for(_CachedRuleIndexingResult),
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

    def _hash_rule_indexing_request(
        self,
        g: _RuleText,
        relativity_proposition: bool,
        signal_proposition: bool,
        title_proposition: bool,
        title: str | None,
        agent_id: AgentId | None,
        glossary_fingerprint: str,
    ) -> str:
        """Generate a hash for the rule indexing request."""
        return xxhash.xxh3_128_hexdigest(
            f"{self._RULE_INDEXING_CACHE_VERSION}:{title or ''}:{g.condition or ''}:{g.content or ''}:{relativity_proposition}:{signal_proposition}:{title_proposition}:{agent_id or ''}:{glossary_fingerprint}".encode()
        )

    def _glossary_fingerprint(self, terms: Sequence[_CoreTerm]) -> str:
        """A content fingerprint of the terms an indexing request depends on - folded
        into the request hash so a term added or edited at runtime invalidates
        exactly the cached indexing results it affects."""
        return xxhash.xxh3_128_hexdigest(
            "|".join(
                sorted(
                    f"{term.name}:{term.description}:{','.join(term.synonyms)}" for term in terms
                )
            ).encode()
        )

    def _cached_rule_indexing_is_usable(
        self,
        cached_indexing_result: _CachedRuleIndexingResult,
        *,
        signal_proposition: bool,
    ) -> bool:
        if not signal_proposition:
            return True

        signals = cached_indexing_result.get("signals", [])
        anti_signals = cached_indexing_result.get("anti_signals", [])
        return len(signals) == 10 and len(anti_signals) == 10

    async def index_rule(
        self,
        entity_id: RuleId,
        g: _RuleText,
        title: str | None = None,
        agent_id: AgentId | None = None,
        relativity_proposition: bool = True,
        signal_proposition: bool = True,
        title_proposition: bool = False,
    ) -> _CachedIndexer.RuleIndexingResult:
        return await self._index_rule(
            entity_id=entity_id,
            g=g,
            relativity_proposition=relativity_proposition,
            signal_proposition=signal_proposition,
            title=title,
            agent_id=agent_id,
            title_proposition=title_proposition,
        )

    async def _index_rule(
        self,
        entity_id: RuleId,
        g: _RuleText,
        relativity_proposition: bool = True,
        signal_proposition: bool = True,
        title_proposition: bool = False,
        title: str | None = None,
        agent_id: AgentId | None = None,
    ) -> _CachedIndexer.RuleIndexingResult:
        # The indexing prompts consume the glossary terms relevant to this
        # rule, so the cache must be sensitive to exactly those terms: a term
        # added or edited at runtime invalidates only the indexing results it affects.
        relevant_terms = (
            await self._container[EntityQueries].find_glossary_terms_for_context(
                agent_id,
                query=compose_rule_query(
                    title=title,
                    condition=g.condition,
                    content=g.content,
                ),
                max_terms=RuleIndexingProcessor._TERMS_PER_RULE,
            )
            if agent_id
            else []
        )

        # First check if we have a cached indexing result for this rule
        _hash = self._hash_rule_indexing_request(
            g=g,
            relativity_proposition=relativity_proposition,
            signal_proposition=signal_proposition,
            title_proposition=title_proposition,
            title=title,
            agent_id=agent_id,
            glossary_fingerprint=self._glossary_fingerprint(relevant_terms),
        )

        if cached_indexing_result := await self._rule_collection.find_one({"id": {"$eq": _hash}}):
            if self._cached_rule_indexing_is_usable(
                cached_indexing_result,
                signal_proposition=signal_proposition,
            ):
                self._set_progress(entity_id, 100.0)

                return self.RuleIndexingResult(
                    references_other_policies=cached_indexing_result.get(
                        "references_other_policies"
                    ),
                    signals=cached_indexing_result.get("signals", []),
                    anti_signals=cached_indexing_result.get("anti_signals", []),
                    title=cached_indexing_result.get("title"),
                )

        indexing_task_id = await self._container[IndexingService].create_indexing_task(
            payload_descriptors=[
                PayloadDescriptor(
                    PayloadKind.RULE,
                    RulePayload(
                        content=g.content,
                        condition=g.condition,
                        operation=PayloadOperation.ADD,
                        relativity_proposition=relativity_proposition,
                        signal_proposition=signal_proposition,
                        title_proposition=title_proposition,
                        title=title,
                        agent_id=agent_id,
                    ),
                )
            ],
            hints=StoreProviderHints(call_site="sdk"),
        )

        while True:
            indexing_task = await self._container[IndexingService].read_indexing_task(
                indexing_task_id=indexing_task_id,
                hints=StoreProviderHints(call_site="sdk"),
            )

            self._set_progress(entity_id, indexing_task.progress)

            if indexing_task.status in [IndexingStatus.PENDING, IndexingStatus.RUNNING]:
                await asyncio.sleep(0.5)
                continue
            elif indexing_task.status == IndexingStatus.FAILED:
                raise SDKError(
                    f"Indexing failed: {indexing_task.error}. Indexing calls your model "
                    "provider when rules are created, so check the provider's API key and "
                    "quota first. A free-tier or low quota is a common cause: see "
                    '"Rate Limits" in docs/providers.md to limit indexing\'s request rate.'
                )
            elif indexing_task.status == IndexingStatus.COMPLETED:
                if not indexing_task.invoices:
                    raise SDKError("Indexing completed with no invoices.")
                if not indexing_task.invoices[0].approved:
                    raise SDKError("Indexing completed with unapproved invoice.")

                invoice = indexing_task.invoices[0]

                if not invoice.data:
                    raise SDKError("Indexing completed with no data in the invoice.")

            assert invoice.data

            # Cache the indexing result
            await self._rule_collection.insert_one(
                {
                    "id": ObjectId(_hash),
                    "creation_utc": datetime.now(timezone.utc).isoformat(),
                    "version": Version.String(VERSION),
                    "references_other_policies": invoice.data.references_other_policies_proposition,
                    "signals": list(invoice.data.signals_proposition or []),
                    "anti_signals": list(invoice.data.anti_signals_proposition or []),
                    "title": invoice.data.title_proposition,
                }
            )

            # Return the indexing result
            return self.RuleIndexingResult(
                references_other_policies=invoice.data.references_other_policies_proposition,
                signals=invoice.data.signals_proposition or [],
                anti_signals=invoice.data.anti_signals_proposition or [],
                title=invoice.data.title_proposition,
            )


@dataclass(frozen=True)
class Group:
    """A group used to categorize and link entities."""

    id: GroupId
    name: str
    _server: Optional[Server] = field(default=None, repr=False)

    async def reevaluate_after(self, *tools: ToolRef) -> Sequence[Relationship]:
        """Creates reevaluation relationships between this group and one or more tools.

        When any of the tools is called, all rules grouped with this group
        will be reevaluated."""
        if self._server is None:
            raise SDKError(
                "Group reevaluation can only be performed during the server startup scope."
            )

        if not tools:
            raise SDKError("At least one tool must be provided for reevaluation.")

        results: list[Relationship] = []
        for t in tools:
            relationship = await self._server._container[RelationshipStore].create_relationship(
                source=RelationshipEntity(
                    id=self.id,
                    kind=RelationshipEntityKind.GROUP_ALL,
                ),
                target=RelationshipEntity(
                    id=_tool_ref_to_id(t),
                    kind=RelationshipEntityKind.TOOL,
                ),
                kind=RelationshipKind.REEVALUATION,
            )

            results.append(
                Relationship(
                    id=relationship.id,
                    kind=relationship.kind,
                    source=relationship.source.id,
                    target=relationship.target.id,
                )
            )

        return results

    async def includes_current_customer(
        self, context: RuleEvaluationContext, rule: Rule
    ) -> RuleEvaluation:
        """Custom evaluator: relevant exactly when the turn's customer is a member of this group.

        Pass as an ``evaluation_func`` so a policy or observation applies only to
        customers in this group, with membership decided by exact code rather than
        the model's judgment::

            vip = await agent.create_group(name="vip")
            await agent.create_observation(
                "the customer is a VIP",
                evaluation_func=vip.includes_current_customer,
            )
        """
        return self._customer_membership_evaluation(context, rule, want_member=True)

    async def excludes_current_customer(
        self, context: RuleEvaluationContext, rule: Rule
    ) -> RuleEvaluation:
        """Custom evaluator: relevant exactly when the turn's customer is NOT a member of this group.

        The inverse of :meth:`includes_current_customer`; pass it as an
        ``evaluation_func`` to apply a policy or observation only to customers
        outside this group.
        """
        return self._customer_membership_evaluation(context, rule, want_member=False)

    def _customer_membership_evaluation(
        self, context: RuleEvaluationContext, rule: Rule, want_member: bool
    ) -> RuleEvaluation:
        is_member = self in context.customer.groups
        return RuleEvaluation(
            id=rule.id,
            is_relevant=(is_member == want_member),
            reasoning=(
                f"Customer {'is' if is_member else 'is not'} a member of group '{self.name}'"
            ),
        )

    async def _create_relationship(
        self,
        target: Rule | Group | AnyOf | AllOf,
        kind: RelationshipKind,
        group_id: str | None = None,
    ) -> Relationship:
        server = self._server
        if server is None:
            raise SDKError(
                "Group relationships can only be created during the server startup scope."
            )

        entity_source = RelationshipEntity(id=self.id, kind=RelationshipEntityKind.GROUP_ALL)

        if isinstance(target, Rule):
            entity_target = RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE)
        elif isinstance(target, AnyOf):
            entity_target = RelationshipEntity(
                id=target.group.id, kind=RelationshipEntityKind.GROUP_ANY
            )
        elif isinstance(target, AllOf):
            entity_target = RelationshipEntity(
                id=target.group.id, kind=RelationshipEntityKind.GROUP_ALL
            )
        elif isinstance(target, Group):
            entity_target = RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GROUP_ALL)

        relationship = await server._container[RelationshipStore].create_relationship(
            source=entity_source,
            target=entity_target,
            kind=kind,
            group_id=group_id,
        )

        return Relationship(
            id=relationship.id,
            kind=relationship.kind,
            source=relationship.source.id,
            target=relationship.target.id,
        )

    async def prioritize_over(self, *targets: Rule | Group | AllOf) -> Sequence[Relationship]:
        """Creates priority relationships with other rules or groups."""
        if not targets:
            raise SDKError("At least one target must be provided for prioritization.")

        return [await self._create_relationship(t, RelationshipKind.PRIORITY) for t in targets]

    async def depend_on(self, *targets: Rule | Group | AnyOf | AllOf) -> Sequence[Relationship]:
        """Creates dependency relationships with other rules or groups."""
        if not targets:
            raise SDKError("At least one target must be provided for dependency.")

        return [await self._create_relationship(t, RelationshipKind.DEPENDENCY) for t in targets]

    async def depend_on_any(self, *targets: Rule | Group | AnyOf | AllOf) -> Sequence[Relationship]:
        """Creates OR dependency relationships. At least one target must be active."""
        if not targets:
            raise SDKError("At least one target must be provided for dependency.")

        group_id = str(uuid.uuid4())
        return [
            await self._create_relationship(t, RelationshipKind.DEPENDENCY_ANY, group_id=group_id)
            for t in targets
        ]


@dataclass(frozen=True)
class AnyOf:
    """Wraps a Group to indicate ANY semantics in a dependency relationship.

    When used as a target in ``depend_on()``, the dependency is satisfied if
    at least one entity grouped with the given group is active.
    """

    group: Group


@dataclass(frozen=True)
class AllOf:
    """Wraps a Group to indicate ALL semantics in a dependency relationship.

    When used as a target in ``depend_on()``, ``prioritize_over()``, or ``exclude()``,
    the dependency/priority is evaluated against all entities grouped with the given group.
    This is also the default when a bare ``Group`` is passed.
    """

    group: Group


def _tags_from_ids(group_ids: Sequence[GroupId]) -> list[Group]:
    """Convert a sequence of GroupIds to a list of Group objects, using the ID as the name."""
    return [Group(id=group_id, name=str(group_id)) for group_id in group_ids]


@dataclass(frozen=True)
class Relationship:
    """A relationship between two entities in the system."""

    id: RelationshipId
    kind: RelationshipKind
    source: RelationshipEntityId
    target: RelationshipEntityId


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call by the agent."""

    tool_id: ToolId
    arguments: Mapping[str, JSONSerializable]
    result: ToolResult


@dataclass(frozen=True)
class RuleEvaluation:
    """A custom evaluator's judgment about a rule's relevance."""

    id: RuleId
    """The ID of the rule that was evaluated."""

    is_relevant: bool
    """Whether the rule is relevant to the current context."""

    reasoning: str
    """Explanation of why the rule is or is not relevant."""

    score: float | None = None
    highlights: Sequence[str] = ()
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)


RuleHandler: TypeAlias = (
    Callable[[EngineContext, RuleEvaluation], Awaitable[None]]
    | Callable[[EngineContext], Awaitable[None]]
    | Callable[[], Awaitable[None]]
)
"""An on_discovered/on_matched callback. May take (context, evaluation),
just (context), or no parameters — the SDK passes what it accepts."""

CannedResponseFieldProvider: TypeAlias = (
    Callable[[EngineContext], Awaitable[Mapping[str, Any]]]
    | Callable[[], Awaitable[Mapping[str, Any]]]
)
"""A callback that supplies canned-response fields when a rule matches.

It may take the engine context or no parameters.
"""


@dataclass
class RuleEvaluationContext:
    """Context supplied to custom rule evaluators."""

    server: Server
    container: Container
    logger: Logger
    tracer: Tracer
    session: Session
    agent: Agent
    customer: Customer
    variables: Mapping[Variable, JSONSerializable]
    staged_events: Sequence[EmittedEvent]

    @property
    def staged_tool_calls(self) -> Sequence[ToolCall]:
        """Returns the staged events that are tool calls."""
        core_tool_calls = chain.from_iterable(
            [
                cast(ToolEventData, e.data)["tool_calls"]
                for e in self.staged_events
                if e.kind == EventKind.TOOL
            ]
        )

        return [
            ToolCall(
                tool_id=ToolId.from_string(call["tool_id"]),
                arguments=call["arguments"],
                result=ToolResult(
                    data=call["result"].get("data"),
                    metadata=call["result"].get("metadata"),
                    control=call["result"].get("control"),
                    canned_responses=call["result"].get("canned_responses"),
                    canned_response_fields=call["result"].get("canned_response_fields"),
                    rules=call["result"].get("rules"),
                ),
            )
            for call in core_tool_calls
        ]

    @classmethod
    async def _from_engine_context(
        cls,
        engine_ctx: EngineContext,
        server: Server,
        container: Container,
    ) -> RuleEvaluationContext:
        """Build an SDK evaluation context from an engine context."""
        agent = await server.get_agent(id=engine_ctx.agent.id)
        customer = await server.get_customer(id=engine_ctx.customer.id)

        return cls(
            server=server,
            container=container,
            logger=container[Logger],
            tracer=engine_ctx.tracer,
            session=Session(
                id=engine_ctx.session.id,
                interaction=engine_ctx.interaction,
                metadata=engine_ctx.session.metadata,
                labels=engine_ctx.session.labels,
                customer=customer,
                agent=agent,
                mode=engine_ctx.session.mode,
                title=engine_ctx.session.title,
                _metadata_workspace=engine_ctx.session_metadata,
            ),
            agent=agent,
            customer=customer,
            variables={
                await agent.get_variable(id=var.id): val.data
                for var, val in engine_ctx.state.context_variables
            },
            staged_events=engine_ctx.state.tool_events,
        )


async def MATCH_ALWAYS(context: RuleEvaluationContext, rule: Rule) -> RuleEvaluation:
    """An evaluation function that makes a rule relevant on every turn."""
    return RuleEvaluation(
        id=rule.id,
        is_relevant=True,
        reasoning="Always relevant",
    )


@dataclass(frozen=True)
class Rule:
    """A base class for SDK objects backed by a rule record."""

    id: RuleId
    groups: Sequence[Group]
    metadata: Mapping[str, JSONSerializable]

    _server: Server
    _container: Container
    _store_provider: StoreProvider

    condition: str | None = None
    title: str | None = None
    content: str | None = None
    labels: set[str] = field(default_factory=set)
    trigger_effort: Effort | None = None
    priority: int = 0

    async def entail(self, rule: Rule) -> Relationship:
        """Creates an entailment relationship with another rule."""
        return await self._create_relationship(
            target=rule,
            kind=RelationshipKind.ENTAILMENT,
            direction="source",
        )

    async def prioritize_over(self, *targets: Rule | Group | AllOf) -> Sequence[Relationship]:
        """Creates priority relationships with other rules or groups."""
        if not targets:
            raise SDKError("At least one target must be provided for prioritization.")

        return [
            await self._create_relationship(
                target=t,
                kind=RelationshipKind.PRIORITY,
                direction="source",
            )
            for t in targets
        ]

    async def depend_on(self, *targets: Rule | Group | AnyOf | AllOf) -> Sequence[Relationship]:
        """Creates dependency relationships with other rules or groups."""
        if not targets:
            raise SDKError("At least one target must be provided for dependency.")

        return [
            await self._create_relationship(
                target=t,
                kind=RelationshipKind.DEPENDENCY,
                direction="source",
            )
            for t in targets
        ]

    async def depend_on_any(self, *targets: Rule | Group | AnyOf | AllOf) -> Sequence[Relationship]:
        """Creates OR dependency relationships. At least one target must be active."""
        if not targets:
            raise SDKError("At least one target must be provided for dependency.")

        group_id = str(uuid.uuid4())
        return [
            await self._create_relationship(
                target=t,
                kind=RelationshipKind.DEPENDENCY_ANY,
                direction="source",
                group_id=group_id,
            )
            for t in targets
        ]

    async def reevaluate_after(self, *tools: ToolRef) -> Sequence[Relationship]:
        """Creates reevaluation relationships with one or more tools."""
        if not tools:
            raise SDKError("At least one tool must be provided for reevaluation.")

        results: list[Relationship] = []
        for t in tools:
            relationship = await self._container[RelationshipStore].create_relationship(
                source=RelationshipEntity(
                    id=self.id,
                    kind=RelationshipEntityKind.RULE,
                ),
                target=RelationshipEntity(
                    id=_tool_ref_to_id(t),
                    kind=RelationshipEntityKind.TOOL,
                ),
                kind=RelationshipKind.REEVALUATION,
            )

            results.append(
                Relationship(
                    id=relationship.id,
                    kind=relationship.kind,
                    source=relationship.source.id,
                    target=relationship.target.id,
                )
            )

        return results

    async def attach_retriever(
        self,
        retriever: Callable[[RetrieverContext], Awaitable[RetrieverResult | None]],
        id: str | None = None,
    ) -> None:
        """Attaches a retriever that runs only when this rule is matched."""

        # A rule-attached retriever fires only on a turn-match verdict, so the
        # rule must actually be evaluated each turn: stamp the internal
        # evaluation_method entry (consulted by the matcher's strategy finder)
        # unless the rule already carries one — never downgrade "distill".
        store = self._server._store_provider.get_store(
            RuleStore, StoreProviderHints(call_site="sdk")
        )
        stored_rule = await store.read_rule(self.id)
        if metadata_utils.internal_value(stored_rule.metadata, ("evaluation_method",)) is None:
            updated_metadata = metadata_utils.set_internal_value(
                stored_rule.metadata, ("evaluation_method",), "rank"
            )
            await store.set_metadata(
                self.id,
                metadata_utils.PARLANT_METADATA_KEY,
                updated_metadata[metadata_utils.PARLANT_METADATA_KEY],
            )

        def is_rule_matched(ctx: EngineContext) -> bool:
            return any(m.rule.id == self.id for m in ctx.state.ordinary_rule_matches) or any(
                m.rule.id == self.id for m in ctx.state.tool_enabled_rule_matches
            )

        self._server._attach_conditional_retriever(
            retriever_id=id or f"rule-retriever-{self.id}",
            retriever=retriever,
            should_run=is_rule_matched,
        )

    async def _create_relationship(
        self,
        target: Rule | Group | AnyOf | AllOf,
        kind: RelationshipKind,
        direction: Literal["source", "target"],
        group_id: str | None = None,
    ) -> Relationship:
        if isinstance(target, Rule):
            other_entity = RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE)
        elif isinstance(target, AnyOf):
            other_entity = RelationshipEntity(
                id=target.group.id, kind=RelationshipEntityKind.GROUP_ANY
            )
        elif isinstance(target, AllOf):
            other_entity = RelationshipEntity(
                id=target.group.id, kind=RelationshipEntityKind.GROUP_ALL
            )
        elif isinstance(target, Group):
            other_entity = RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GROUP_ALL)
        else:
            raise SDKError(f"Unsupported relationship target: {target!r}")

        self_entity = RelationshipEntity(id=self.id, kind=RelationshipEntityKind.RULE)

        if direction == "source":
            entity_source = self_entity
            entity_target = other_entity
        else:
            entity_source = other_entity
            entity_target = self_entity

        relationship = await self._store_provider.get_store(
            RelationshipStore, StoreProviderHints(call_site="sdk")
        ).create_relationship(
            source=entity_source,
            target=entity_target,
            kind=kind,
            group_id=group_id,
        )

        return Relationship(
            id=relationship.id,
            kind=relationship.kind,
            source=relationship.source.id,
            target=relationship.target.id,
        )


@dataclass(frozen=True)
class Observation(Rule):
    """An observation that identifies a contextual state."""

    condition: str = ""

    @property
    def note(self) -> str | None:
        """The optional note supplied at creation (stored as the rule's content)."""
        return self.content


@dataclass(frozen=True)
class Policy(Rule):
    """A policy that defines standing guidance."""


@dataclass(frozen=True)
class Term:
    """A glossary term defines a specific concept in the agent's domain."""

    id: TermId
    name: str
    description: str
    synonyms: Sequence[str]
    groups: Sequence[Group]


@dataclass(frozen=True)
class Variable:
    """A variable that can hold values for customers or customer groups."""

    id: ContextVariableId
    name: str
    description: str | None
    tool: ToolRef | None
    freshness_rules: str | None
    groups: Sequence[Group]
    _server: Server
    _container: Container
    _store_provider: StoreProvider

    def __hash__(self) -> int:
        return hash(self.id)

    async def set_value_for_customer(self, customer: Customer, value: JSONSerializable) -> None:
        """Sets the value of the variable for a specific customer."""

        await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).update_value(
            variable_id=self.id,
            key=customer.id,
            data=value,
        )

    async def set_value_for_tag(self, group: GroupId, value: JSONSerializable) -> None:
        """Sets the value of the variable for a specific group (e.g., a customer group group)."""

        await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).update_value(
            variable_id=self.id,
            key=f"group:{group}",
            data=value,
        )

    async def set_value_for_agent(self, agent: Agent, value: JSONSerializable) -> None:
        """Sets the value of the variable for a specific agent."""
        await self._container[ContextVariableStore].update_value(
            variable_id=self.id,
            key=GroupIds.for_agent_id(agent.id),
            data=value,
        )

    async def get_value_for_agent(self, agent: Agent) -> JSONSerializable | None:
        """Retrieves the value of the variable for a specific agent."""
        value = await self._container[ContextVariableStore].read_value(
            variable_id=self.id,
            key=GroupIds.for_agent_id(agent.id),
        )
        return value.data if value else None

    async def set_global_value(self, value: JSONSerializable) -> None:
        """Sets the global value of the variable, which is accessible to all customers by default."""

        await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).update_value(
            variable_id=self.id,
            key=ContextVariableStore.GLOBAL_KEY,
            data=value,
        )

    async def get_value_for_customer(self, customer: Customer) -> JSONSerializable | None:
        """Retrieves the value of the variable for a specific customer."""

        value = await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).read_value(
            variable_id=self.id,
            key=customer.id,
        )

        return value.data if value else None

    async def get_value_for_tag(self, group: GroupId) -> JSONSerializable | None:
        """Retrieves the value of the variable for a specific group (e.g., a customer group group)."""
        value = await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).read_value(
            variable_id=self.id,
            key=f"group:{group}",
        )

        return value.data if value else None

    async def get_global_value(self) -> JSONSerializable | None:
        """Retrieves the global value of the variable, which is accessible to all customers by default."""

        value = await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).read_value(
            variable_id=self.id,
            key=ContextVariableStore.GLOBAL_KEY,
        )

        return value.data if value else None

    async def get_value(self) -> JSONSerializable | None:
        """Retrieves the value of the variable for the current context"""
        if value := EntityContext.get_variable_value(self.id):
            return value.data

        # The engine-loaded variable state may not be populated yet —
        # retrievers deliberately start in parallel with preparation. Resolve
        # the tiers directly instead, in the same order the engine's variable
        # loader uses (customer > customer groups > agent > global), through
        # the same freshness-aware helper (tool-backed variables refresh).
        agent = EntityContext.get_agent()
        customer = EntityContext.get_customer()
        session = EntityContext.get_session()
        if agent is None or customer is None or session is None:
            return None

        variable = await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).read_variable(self.id)

        keys_in_order_of_importance = [
            str(customer.id),
            *[f"group:{group_id}" for group_id in customer.groups],
            str(GroupIds.for_agent_id(agent.id)),
            ContextVariableStore.GLOBAL_KEY,
        ]

        for key in keys_in_order_of_importance:
            value = await load_fresh_context_variable_value(
                entity_queries=self._container[EntityQueries],
                entity_commands=self._container[EntityCommands],
                agent_id=agent.id,
                session=session,
                variable=variable,
                key=key,
            )
            if value is not None:
                return value.data

        return None


class CustomerMetadata:
    """Async-aware metadata accessor for a customer.

    Supports sync reads via ``[]`` and async writes via :meth:`set` / :meth:`delete`.
    Use :meth:`get` for an async read that refreshes from the store.
    """

    def __init__(
        self,
        customer_id: CustomerId,
        data: Mapping[str, str],
        server: Optional[Server] = None,
    ) -> None:
        self._customer_id = customer_id
        self._data = dict(data)
        self._server = server

    def _get_store(self) -> CustomerStore:
        server = self._server if self._server is not None else Server.current
        return server._store_provider.get_store(CustomerStore, StoreProviderHints(call_site="sdk"))

    # -- sync reads ----------------------------------------------------------

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- async operations -----------------------------------------------------

    async def get(self, key: str, default: str | None = None) -> str | None:
        """Read a metadata value, refreshing from the store first."""
        customer = await self._get_store().read_customer(self._customer_id)
        self._data = dict(customer.metadata)
        return self._data.get(key, default)

    def _check_not_guest(self) -> None:
        if self._customer_id == CustomerStore.GUEST_ID:
            raise RuntimeError("Cannot update the guest customer")

    async def set(self, key: str, value: str) -> None:
        """Set a metadata value and persist it to the store."""
        self._check_not_guest()
        await self._get_store().upsert_metadata(self._customer_id, {key: value})
        self._data[key] = value

    async def delete(self, key: str) -> None:
        """Delete a metadata value and persist the removal to the store."""
        self._check_not_guest()
        await self._get_store().remove_metadata(self._customer_id, [key])
        del self._data[key]


class CustomerGroups:
    """Async-aware accessor for a customer's group memberships.

    Supports sync reads (iteration, indexing, ``len``, and ``in`` — which
    accepts a :class:`Group` or a ``GroupId``) over a cached snapshot, and
    async writes via :meth:`add` / :meth:`remove` that persist to the store
    and refresh the snapshot. Use :meth:`get` for an async read that refreshes
    from the store first.
    """

    def __init__(
        self,
        customer_id: CustomerId,
        groups: Sequence[Group],
        server: Optional[Server] = None,
    ) -> None:
        self._customer_id = customer_id
        self._groups = list(groups)
        self._server = server

    def _get_store(self) -> CustomerStore:
        server = self._server if self._server is not None else Server.current
        return server._store_provider.get_store(CustomerStore, StoreProviderHints(call_site="sdk"))

    # -- sync reads ----------------------------------------------------------

    def __iter__(self) -> Iterator[Group]:
        return iter(self._groups)

    def __getitem__(self, index: int) -> Group:
        return self._groups[index]

    def __len__(self) -> int:
        return len(self._groups)

    def __contains__(self, group: object) -> bool:
        group_id = group.id if isinstance(group, Group) else group
        return any(existing.id == group_id for existing in self._groups)

    # -- async operations -----------------------------------------------------

    def _check_not_guest(self) -> None:
        if self._customer_id == CustomerStore.GUEST_ID:
            raise RuntimeError("Cannot update the guest customer")

    async def _refresh(self) -> None:
        customer = await self._get_store().read_customer(self._customer_id)
        self._groups = list(_tags_from_ids(customer.groups))

    async def get(self) -> Sequence[Group]:
        """Return the customer's groups, refreshing from the store first."""
        await self._refresh()
        return list(self._groups)

    async def add(self, *groups: Group) -> None:
        """Add the customer to one or more groups, then refresh. Idempotent."""
        self._check_not_guest()
        store = self._get_store()
        for group in groups:
            await store.upsert_group(self._customer_id, group.id)
        await self._refresh()

    async def remove(self, *groups: Group) -> None:
        """Remove the customer from one or more groups, then refresh. Idempotent."""
        self._check_not_guest()
        store = self._get_store()
        for group in groups:
            await store.remove_group(self._customer_id, group.id)
        await self._refresh()


class Customer:
    """A customer represents an individual or entity interacting with the agent."""

    def __init__(
        self,
        id: CustomerId,
        name: str,
        metadata: Mapping[str, str],
        groups: Sequence[Group],
        kind: CustomerKind = CustomerKind.REAL,
        _server: Optional[Server] = None,
    ) -> None:
        self._id = id
        self._name = name
        self._kind = kind
        self._metadata = CustomerMetadata(
            customer_id=id,
            data=metadata,
            server=_server,
        )
        self._groups = CustomerGroups(
            customer_id=id,
            groups=groups,
            server=_server,
        )
        self._server = _server

    @property
    def id(self) -> CustomerId:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> CustomerMetadata:
        return self._metadata

    @property
    def groups(self) -> CustomerGroups:
        return self._groups

    @property
    def kind(self) -> CustomerKind:
        """Whether this is a real customer or a persona used for testing.

        Fixed when the customer is created. A persona cannot be authenticated
        as, so no end user is ever served as one.
        """
        return self._kind

    @property
    def is_persona(self) -> bool:
        """Whether this customer is a persona used for testing.

        The same answer as comparing :attr:`kind`, in the form a guard wants
        it. False for anything that is not explicitly a persona, so code that
        reads it treats an unknown customer as a real one.
        """
        return self._kind is CustomerKind.PERSONA

    @classproperty
    def guest(cls: Customer) -> Customer:
        return Customer(
            id=CustomerStore.GUEST_ID,
            name="Guest",
            metadata={},
            groups=[],
            # Stated rather than left to the default: guest sessions carry
            # production traffic, so the guest is a real customer.
            kind=CustomerKind.REAL,
        )

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
            metadata=core_customer.metadata,
            groups=_tags_from_ids(core_customer.groups),
            kind=core_customer.kind,
        )

    async def update(
        self,
        *,
        name: str | None = None,
    ) -> None:
        """Updates the customer's information.

        Args:
            name: New name for the customer.

        Raises:
            RuntimeError: If this is the guest customer.
        """
        if self._id == CustomerStore.GUEST_ID:
            raise RuntimeError("Cannot update the guest customer")

        server = self._server if self._server is not None else Server.current
        customer_store = server._store_provider.get_store(
            CustomerStore, StoreProviderHints(call_site="sdk")
        )

        if name is not None:
            await customer_store.update_customer(
                customer_id=self._id,
                params={"name": name},
            )

        updated = await customer_store.read_customer(self._id)
        self._name = updated.name
        self._groups = CustomerGroups(
            customer_id=self._id,
            groups=_tags_from_ids(updated.groups),
            server=self._server,
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


@dataclass(frozen=True)
class RetrieverResult:
    """Result of a retriever function, containing the retrieved data and metadata, as well (optionally) as canned response information."""

    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)
    canned_responses: Sequence[str] = field(default_factory=list)
    canned_response_fields: Mapping[str, Any] = field(default_factory=dict)
    rules: Sequence[TransientRule] = field(default_factory=list)


DeferredRetriever: TypeAlias = Callable[[EngineContext], Awaitable[RetrieverResult | None]]
"""A deferred retriever callable that receives a pre-response EngineContext and returns a RetrieverResult or None.

Returning this allows retrievers to start work in parallel during on_acknowledged, but defer the final decision
of what data to return (or whether to return any data at all) until on_generating_messages, when the
full EngineContext including matched rules and tool insights is available.
"""

RetrieverFunction: TypeAlias = Callable[
    [RetrieverContext], Awaitable[RetrieverResult | None | DeferredRetriever]
]
"""A retriever function that can either return a result directly, or return a deferred callable.

When a RetrieverResult or None is returned directly, it's used as-is.
When a DeferredRetriever is returned, it will be called later with the EngineContext to get the final result.
"""


class ExperimentalAgentFeatures:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent


@dataclass(frozen=True)
class Agent:
    """An agent represents an entity that can interact with customers and perform various tasks."""

    _server: Server
    _container: Container
    _store_provider: StoreProvider

    id: AgentId
    name: str
    instructions: str | None
    max_steps: int
    output_mode: OutputMode
    groups: Sequence[Group]
    models: Mapping[str, ModelRouteValue] = field(default_factory=dict)

    retrievers: Mapping[str, RetrieverFunction] = field(default_factory=dict)

    @property
    def experimental_features(self) -> ExperimentalAgentFeatures:
        """Provides access to experimental features of the agent."""
        return ExperimentalAgentFeatures(self)

    async def update(
        self,
        *,
        models: ModelRouteUpdates,
    ) -> None:
        """Update this agent's model routes without restarting its sessions."""
        configuration = self._container[ModelConfigurationService]
        scope = ModelScope.agent(self.id)
        await configuration.configure(scope, models)
        object.__setattr__(self, "models", await configuration.read(scope))

    async def create_policy(
        self,
        content: str | None = None,
        title: str | None = None,
        tools: Iterable[ToolRef] = [],
        metadata: dict[str, JSONSerializable] = {},
        canned_responses: Sequence[CannedResponseId] = [],
        weight: Weight | None = None,
        trigger_effort: Effort | None = None,
        evaluation_func: Callable[[RuleEvaluationContext, Policy], Awaitable[RuleEvaluation]]
        | None = None,
        on_discovered: RuleHandler | None = None,
        on_matched: RuleHandler | None = None,
        canned_response_field_provider: CannedResponseFieldProvider | None = None,
        groups: Sequence[Group] = [],
        id: RuleId | None = None,
        labels: Iterable[str] = (),
        dependencies: Sequence[Rule] = [],
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
        trigger_message_review: bool = False,
        trigger_tool_review: bool = False,
    ) -> Policy:
        """Creates a policy with the specified title and content.

        Notable parameters:
            evaluation_func: Replaces the engine's semantic matching for this
                policy with your own code (see ``MATCH_ALWAYS`` for an
                always-on policy). Setting it also disables automatic indexing
                for the policy - no recall signals or cross-reference analysis
                are proposed, since your function alone decides applicability.
            on_matched: A callback fired when the policy matches a turn. Note
                that attaching one escalates the policy to a full LLM ranking
                verdict at every effort level (a match verdict is required to
                fire the callback), which adds per-turn evaluation cost.
            id: Fixes the policy's identity across content changes. Without
                it, the id is derived from the authored content, so editing
                the content creates a new identity; pass a stable id to keep
                references (relationships, analytics) intact as you iterate.
        """
        policy = cast(
            Policy,
            await self._server._create_rule(
                condition=None,
                content=content,
                title=title,
                tools=tools,
                metadata=metadata,
                canned_responses=canned_responses,
                weight=weight,
                effort=trigger_effort,
                evaluation_func=evaluation_func,
                on_discovered=on_discovered,
                on_matched=on_matched,
                canned_response_field_provider=canned_response_field_provider,
                groups=[GroupIds.for_agent_id(self.id), *[t.id for t in groups]],
                relationship_target_group_id=None,
                id=id,
                labels=labels,
                priority=priority,
                signals=signals,
                anti_signals=anti_signals,
                message_review=trigger_message_review,
                tool_review=trigger_tool_review,
                agent_id=self.id,
                rule_kind="policy",
            ),
        )

        if dependencies:
            await policy.depend_on(*dependencies)

        return policy

    async def create_observation(
        self,
        condition: str,
        note: str | None = None,
        tools: Iterable[ToolRef] = [],
        canned_responses: Sequence[CannedResponseId] = [],
        weight: Weight | None = None,
        trigger_effort: Effort | None = None,
        evaluation_func: Callable[[RuleEvaluationContext, Observation], Awaitable[RuleEvaluation]]
        | None = None,
        on_discovered: RuleHandler | None = None,
        on_matched: RuleHandler | None = None,
        canned_response_field_provider: CannedResponseFieldProvider | None = None,
        groups: Sequence[Group] = [],
        id: RuleId | None = None,
        title: str | None = None,
        labels: Iterable[str] = (),
        dependencies: Sequence[Rule] = [],
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
        trigger_message_review: bool = False,
        trigger_tool_review: bool = False,
    ) -> Observation:
        """A shorthand for creating an observational rule with the specified condition."""

        if not condition:
            raise SDKError("create_observation requires 'condition' to be provided.")

        observation = cast(
            Observation,
            await self._server._create_rule(
                condition=condition,
                content=note,
                title=title,
                tools=tools,
                metadata={},
                canned_responses=canned_responses,
                weight=weight,
                effort=trigger_effort,
                evaluation_func=evaluation_func,
                on_discovered=on_discovered,
                on_matched=on_matched,
                canned_response_field_provider=canned_response_field_provider,
                groups=[GroupIds.for_agent_id(self.id), *[t.id for t in groups]],
                relationship_target_group_id=None,
                id=id,
                labels=labels,
                priority=priority,
                signals=signals,
                anti_signals=anti_signals,
                message_review=trigger_message_review,
                tool_review=trigger_tool_review,
                agent_id=self.id,
                rule_kind="observation",
            ),
        )

        if dependencies:
            await observation.depend_on(*dependencies)

        return observation

    async def create_canned_response(
        self,
        template: str,
        groups: Sequence[Group] = [],
        signals: Sequence[str] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        field_dependencies: Sequence[str] = (),
        note: str = "",
    ) -> CannedResponseId:
        """Creates a canned response with the specified template, scope, retrieval signals,
        and an optional author note for generation and review."""

        self._server._advance_creation_progress()

        canrep = await self._store_provider.get_store(
            CannedResponseStore, StoreProviderHints(call_site="sdk")
        ).create_canned_response(
            value=template,
            groups=[GroupIds.for_agent_id(self.id), *[t.id for t in groups]],
            fields=[],
            signals=signals,
            metadata=metadata,
            field_dependencies=field_dependencies,
            note=note,
        )

        return canrep.id

    async def create_term(
        self,
        name: str,
        description: str,
        id: Optional[TermId] = None,
        synonyms: Sequence[str] = [],
    ) -> Term:
        """Creates a glossary term with the specified name, description, and synonyms."""

        self._server._advance_creation_progress()

        term = await self._store_provider.get_store(
            GlossaryStore, StoreProviderHints(call_site="sdk")
        ).create_term(
            name=name,
            description=description,
            synonyms=synonyms,
            groups=[GroupIds.for_agent_id(self.id)],
            id=id,
        )

        return Term(
            id=term.id,
            name=term.name,
            description=term.description,
            synonyms=term.synonyms,
            groups=_tags_from_ids(term.groups),
        )

    async def create_variable(
        self,
        name: str,
        description: str | None = None,
        tool: ToolRef | None = None,
        freshness_rules: str | None = None,
    ) -> Variable:
        """Creates a variable with the specified name, description, tool, and freshness rules."""

        self._server._advance_creation_progress()

        if tool:
            await _enable_tool_refs(self._server._plugin_server, [tool])

        variable = await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).create_variable(
            name=name,
            description=description,
            tool_id=_tool_ref_to_id(tool) if tool else None,
            freshness_rules=freshness_rules,
            groups=[GroupIds.for_agent_id(self.id)],
        )

        return Variable(
            id=variable.id,
            name=variable.name,
            description=variable.description,
            tool=tool,
            freshness_rules=variable.freshness_rules,
            groups=_tags_from_ids(variable.groups),
            _server=self._server,
            _container=self._container,
            _store_provider=self._store_provider,
        )

    async def list_variables(self) -> Sequence[Variable]:
        """Lists all variables associated with the agent."""

        variables = await self._store_provider.get_store(
            ContextVariableStore, StoreProviderHints(call_site="sdk")
        ).list_variables(groups=[GroupIds.for_agent_id(self.id)])

        return [
            Variable(
                id=variable.id,
                name=variable.name,
                description=variable.description,
                tool=self._server._plugin_server.tools[variable.tool_id.tool_name]
                if variable.tool_id
                else None,
                freshness_rules=variable.freshness_rules,
                groups=_tags_from_ids(variable.groups),
                _server=self._server,
                _container=self._container,
                _store_provider=self._store_provider,
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
                variable = await self._store_provider.get_store(
                    ContextVariableStore, StoreProviderHints(call_site="sdk")
                ).read_variable(ContextVariableId(id))
            except ItemNotFoundError:
                return None
        else:
            variable = next(
                (
                    v
                    for v in await self._store_provider.get_store(
                        ContextVariableStore, StoreProviderHints(call_site="sdk")
                    ).list_variables(groups=[GroupIds.for_agent_id(self.id)])
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
            groups=_tags_from_ids(variable.groups),
            _server=self._server,
            _container=self._container,
            _store_provider=self._store_provider,
        )

    async def get_variable(
        self,
        *,
        id: ContextVariableId | str | None = None,
        name: str | None = None,
    ) -> Variable:
        """Retrieves a variable by its ID or name, raising an error if not found."""

        if variable := await self.find_variable(id=id, name=name):
            return variable
        raise SDKError(f"Variable with id {id} or name {name} not  found.")

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

    async def utter(
        self,
        session: Session | None = None,
        *,
        rules: Sequence[TransientRule],
    ) -> str:
        """Generate an agent message following turn-scoped instructions.

        This is an in-process SDK operation and does not require an HTTP client.
        Within an active tool or engine hook, ``session`` may be omitted and the
        method will use :attr:`Session.current`. The same applies to child asyncio
        tasks created in that context, because they inherit its context variables.
        Outside an active engine context, pass ``session`` explicitly; otherwise
        resolving :attr:`Session.current` raises ``RuntimeError``.

        Args:
            session: The session in which to generate the message. Defaults to
                the currently active session.
            rules: Turn-scoped instructions whose ``content`` values the agent
                should follow for the generated response.

        Returns:
            The generated message text.
        """
        if session is None:
            session = Session.current

        app = self._container[_Application]
        requests = [
            _UtteranceRequest(
                action=g["content"],
                rationale=_UtteranceRationale.UNSPECIFIED,
            )
            for g in rules
        ]
        event = await app.sessions.utter(session.id, requests)
        return cast(str, cast(dict[str, Any], event.data)["message"])

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

        return Agent(
            _server=server,
            _container=server._container,
            _store_provider=server._store_provider,
            id=core_agent.id,
            name=core_agent.name,
            instructions=core_agent.instructions,
            max_steps=core_agent.max_steps,
            output_mode=core_agent.message_output_mode or OutputMode.BLOCK,
            groups=_tags_from_ids(core_agent.groups),
        )


class _SessionBudgetEvaluator:
    """Decides a rule by comparing the session's spend against a budget.

    One class serves both senses so that budget rules — whichever way they
    read — are recognized at rule creation and scheduled EARLY, before the
    turn spends anything on a model.
    """

    def __init__(self, budget: SessionBudget, *, match_when_exceeded: bool) -> None:
        self._budget = budget
        self._match_when_exceeded = match_when_exceeded

    async def __call__(self, ctx: RuleEvaluationContext, rule: Rule) -> RuleEvaluation:
        measured = await self._budget._measure(ctx.session)
        exceeded = measured > self._budget.value

        return RuleEvaluation(
            id=rule.id,
            is_relevant=exceeded if self._match_when_exceeded else not exceeded,
            reasoning=(
                f"Session cost is {measured} {self._budget.currency}; "
                f"budget is {self._budget.value} {self._budget.currency}"
            ),
        )


class SessionBudget:
    def __init__(self, definition: SessionBudgetDefinition, server: Server) -> None:
        self._definition = definition
        self._server = server
        self._exceeded = _SessionBudgetEvaluator(self, match_when_exceeded=True)
        self._not_exceeded = _SessionBudgetEvaluator(self, match_when_exceeded=False)

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def value(self) -> Decimal:
        return self._definition.value

    @property
    def currency(self) -> str:
        return self._definition.currency

    @property
    def window(self) -> timedelta | None:
        return self._definition.window

    @property
    def exceeded(self) -> _SessionBudgetEvaluator:
        """Matches on turns where the session has spent past this budget."""
        return self._exceeded

    @property
    def not_exceeded(self) -> _SessionBudgetEvaluator:
        """Matches for as long as the session still has budget left.

        The mirror image of ``exceeded`` at the same boundary: on any turn
        exactly one of the two matches. Spending equal to the budget is not
        exceeding it.
        """
        return self._not_exceeded

    async def measure(self) -> Decimal:
        return await self._measure(Session.current)

    async def reset(self) -> None:
        await self._reset(Session.current)

    async def _measure(self, session: Session) -> Decimal:
        return await self._server.container[SessionBudgetService].measure(
            self._definition,
            session.id,
            workspace=session._metadata_workspace,
        )

    async def _reset(self, session: Session) -> None:
        await self._server.container[SessionBudgetService].reset(
            self._definition,
            session.id,
            workspace=session._metadata_workspace,
        )


class Pricing:
    """The server-wide pricing configuration used for reporting and budgets."""

    def __init__(self, pricing: _Pricing, server: Server) -> None:
        self._pricing = pricing
        self._server = server

    @property
    def currency(self) -> str:
        return self._pricing.currency

    async def create_session_budget(
        self,
        *,
        name: str,
        value: Decimal,
        window: timedelta | None = None,
    ) -> SessionBudget:
        if value < 0:
            raise SDKError("Session budget value must be non-negative")
        if window is not None and window <= timedelta(0):
            raise SDKError("Session budget window must be positive")

        definition = SessionBudgetDefinition(
            name=name,
            value=value,
            currency=self.currency,
            window=window,
        )
        self._server.container[PricingRegistry].register_budget(definition)
        return SessionBudget(definition, self._server)


class SessionUsage:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def measure(self, budget: SessionBudget) -> Decimal:
        return await budget._measure(self._session)

    async def reset(self, budget: SessionBudget) -> None:
        await budget._reset(self._session)


class SessionMetadata:
    """Async-aware metadata accessor for a session.

    Supports sync reads via ``[]`` and async writes via :meth:`set` / :meth:`delete`.
    Use :meth:`get` for an async read that refreshes from the store.
    """

    def __init__(
        self,
        session_id: SessionId,
        data: Mapping[str, JSONSerializable],
        server: Optional[Server] = None,
        workspace: SessionMetadataWorkspace | None = None,
    ) -> None:
        self._session_id = session_id
        self._data = dict(data)
        self._server = server
        self._workspace = workspace

    def _current_data(self) -> Mapping[str, JSONSerializable]:
        if self._workspace is not None:
            return self._workspace.view
        return self._data

    def _get_store(self) -> SessionStore:
        server = self._server if self._server is not None else Server.current
        return server._store_provider.get_store(SessionStore, StoreProviderHints(call_site="sdk"))

    # -- sync reads ----------------------------------------------------------

    def __getitem__(self, key: str) -> JSONSerializable:
        return self._current_data()[key]

    def __contains__(self, key: object) -> bool:
        return key in self._current_data()

    def __iter__(self) -> Iterator[str]:
        return iter(self._current_data())

    def __len__(self) -> int:
        return len(self._current_data())

    # -- async operations -----------------------------------------------------

    async def get(self, key: str, default: JSONSerializable = None) -> JSONSerializable:
        """Read a metadata value, refreshing from the store first."""
        session = await self._get_store().read_session(self._session_id)
        if self._workspace is not None and self._workspace.state is WorkspaceState.ACTIVE:
            self._workspace.rebase(session)
            return self._workspace.view.get(key, default)

        self._data = dict(session.metadata)
        return self._data.get(key, default)

    async def set(self, key: str, value: JSONSerializable) -> None:
        """Set a metadata value and persist it to the store."""
        session = await self._get_store().set_metadata(self._session_id, key, value)
        if self._workspace is not None and self._workspace.state is WorkspaceState.ACTIVE:
            self._workspace.rebase(session)
        else:
            self._data = dict(session.metadata)

    async def delete(self, key: str) -> None:
        """Delete a metadata value and persist the removal to the store."""
        session = await self._get_store().unset_metadata(self._session_id, key)
        if self._workspace is not None and self._workspace.state is WorkspaceState.ACTIVE:
            self._workspace.rebase(session)
        else:
            self._data = dict(session.metadata)


class SessionLabels:
    """Async-aware labels accessor for a session.

    Supports sync reads via ``in`` and ``len`` and async writes via :meth:`add` / :meth:`remove`.
    """

    def __init__(
        self,
        session_id: SessionId,
        data: Set[str],
        server: Optional[Server] = None,
    ) -> None:
        self._session_id = session_id
        self._data = set(data)
        self._server = server

    def _get_store(self) -> SessionStore:
        server = self._server if self._server is not None else Server.current
        return server._store_provider.get_store(SessionStore, StoreProviderHints(call_site="sdk"))

    # -- sync reads ----------------------------------------------------------

    def __contains__(self, label: object) -> bool:
        return label in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- async operations -----------------------------------------------------

    async def add(self, label: str) -> None:
        """Add a label and persist it to the store."""
        await self._get_store().upsert_labels(self._session_id, {label})
        self._data.add(label)

    async def remove(self, label: str) -> None:
        """Remove a label and persist the removal to the store."""
        await self._get_store().remove_labels(self._session_id, {label})
        self._data.discard(label)


class Session:
    """A session represents an ongoing conversation between a customer and an agent."""

    def __init__(
        self,
        id: SessionId,
        interaction: Interaction,
        metadata: Mapping[str, JSONSerializable],
        labels: Set[str],
        customer: Customer,
        agent: Agent,
        mode: SessionMode,
        title: str | None = None,
        _server: Optional[Server] = None,
        _metadata_workspace: SessionMetadataWorkspace | None = None,
    ) -> None:
        self._id = id
        self._interaction = interaction
        self._metadata = SessionMetadata(
            session_id=id,
            data=metadata,
            server=_server,
            workspace=_metadata_workspace,
        )
        self._labels = SessionLabels(
            session_id=id,
            data=labels,
            server=_server,
        )
        self._customer = customer
        self._agent = agent
        self._mode = mode
        self._title = title
        self._server = _server
        self._metadata_workspace = _metadata_workspace
        self._usage = SessionUsage(self)

    @property
    def id(self) -> SessionId:
        return self._id

    @property
    def interaction(self) -> Interaction:
        return self._interaction

    @property
    def metadata(self) -> SessionMetadata:
        return self._metadata

    @property
    def labels(self) -> SessionLabels:
        return self._labels

    @property
    def usage(self) -> SessionUsage:
        return self._usage

    @property
    def customer(self) -> Customer:
        """The customer associated with this session."""
        return self._customer

    @property
    def agent(self) -> Agent:
        """The agent associated with this session."""
        return self._agent

    @property
    def mode(self) -> SessionMode:
        return self._mode

    @property
    def effort(self) -> Effort:
        """The current turn's resolved effort level.

        Available inside an active turn (tools, hooks, callbacks); raises
        RuntimeError outside one.
        """
        engine_context = EntityContext.get()

        if engine_context is None or engine_context.session.id != self._id:
            raise RuntimeError("No active turn for this session in the current context")

        return cast(Effort, engine_context.state.dynamic_effort_level)

    @property
    def title(self) -> str | None:
        return self._title

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
            metadata=core_session.metadata,
            labels=core_session.labels,
            customer=Customer.current,
            agent=Agent.current,
            mode=core_session.mode,
            title=core_session.title,
            _metadata_workspace=(
                current_context.session_metadata
                if (current_context := EntityContext.get()) is not None
                else None
            ),
        )

    async def update(
        self,
        *,
        customer: Customer | None = None,
        agent: Agent | None = None,
        mode: SessionMode | None = None,
        title: str | None = None,
    ) -> None:
        """Updates the session's information.

        Args:
            customer: New customer for the session.
            agent: New agent for the session.
            mode: New session mode ("auto" or "manual").
            title: New title for the session.
        """
        server = self._server if self._server is not None else Server.current
        session_store = server._store_provider.get_store(
            SessionStore, StoreProviderHints(call_site="sdk")
        )

        params: _SessionUpdateParams = {}
        if customer is not None:
            params["customer_id"] = customer.id
        if agent is not None:
            params["agent_id"] = agent.id
        if mode is not None:
            params["mode"] = mode
        if title is not None:
            params["title"] = title

        if params:
            await session_store.update_session(
                session_id=self._id,
                params=params,
            )

        updated = await session_store.read_session(self._id)
        self._mode = updated.mode
        self._title = updated.title
        if customer is not None:
            self._customer = customer
        if agent is not None:
            self._agent = agent

    async def _resolve_event_emitter(self) -> tuple[EventEmitter, str]:
        # Inside a turn for this session, ride the turn's own emitter so the
        # event correlates with the current trace; otherwise (admin code, or a
        # different session) build a standalone emitter.
        context = EntityContext.get()
        if context is not None and context.session.id == self._id:
            return context.session_event_emitter, context.tracer.trace_id

        server = self._server if self._server is not None else Server.current
        emitter = await server.container[EventEmitterFactory].create_event_emitter(
            emitting_agent_id=self._agent.id,
            session_id=self._id,
        )
        return emitter, server.container[Tracer].trace_id

    async def emit_message(
        self,
        message: str,
        *,
        source: EventSource = EventSource.AI_AGENT,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> None:
        """Emit a verbatim message event into the session.

        Unlike :meth:`Agent.utter`, no model is involved: the text is posted
        exactly as given. The message becomes part of the session history, so
        later turns treat it as something its ``source`` actually said. Inside
        a turn (e.g. an ``on_matched`` callback) the event joins the current
        trace; outside one, it is emitted standalone — pass
        ``source=p.EventSource.HUMAN_AGENT`` for operator-side messages.
        """
        if source is EventSource.HUMAN_AGENT:
            server = self._server if self._server is not None else Server.current
            await server.container[_Application].sessions.create_human_agent_message_event(
                session_id=self._id,
                message=message,
                participant={"id": None, "display_name": "Operator"},
                metadata=metadata,
            )
            return

        emitter, trace_id = await self._resolve_event_emitter()
        await emitter.emit_message_event(
            trace_id=trace_id,
            data=message,
            metadata=metadata,
            source=source,
        )

    async def emit_status(
        self,
        status: SessionStatus,
        data: JSONSerializable = None,
        *,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> None:
        """Emit a status event (e.g. ``"processing"``, ``"typing"``) into the session."""
        event_data = StatusEventData(status=status)
        if data is not None:
            event_data["data"] = data

        emitter, trace_id = await self._resolve_event_emitter()
        await emitter.emit_status_event(
            trace_id=trace_id,
            data=event_data,
            metadata=metadata,
        )

    async def emit_custom(
        self,
        data: JSONSerializable,
        *,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> None:
        """Emit a custom event into the session, for custom frontends."""
        emitter, trace_id = await self._resolve_event_emitter()
        await emitter.emit_custom_event(
            trace_id=trace_id,
            data=data,
            metadata=metadata,
        )


def _die(message: str, exc: Exception | None) -> NoReturn:
    if exc:
        import traceback

        traceback.print_exception(exc)
    rich.print(Text(message, style="bold red"), file=sys.stderr)
    sys.exit(1)


def _hot_reload_watch_root(main_module: Any | None = None) -> Path | None:
    """The directory hot reload watches: the entry script's parent.

    Returns None — disarming hot reload — when the process has no file-backed
    ``__main__`` (REPLs, notebooks, embedded interpreters): there is nothing
    meaningful to watch, and restarting would re-exec the host process.
    """

    module = main_module if main_module is not None else sys.modules.get("__main__")
    file = getattr(module, "__file__", None)
    if not file:
        return None
    return Path(file).resolve().parent


class Server:
    """The main server class that manages the agent, tools, and other components.

    This class is responsible for initializing the server, managing the lifecycle of the agent, and providing access to various services and components.

    Args:
        host: The NIC host to which the server will bind.
        port: The port on which the server will run.
        tool_service_port: The port for the integrated tool service.
        models: Server-wide model routes used by the Compass engine. Defaults
            to ``{"*": "openrouter"}``.
        prompts: Server-wide prompt-builder providers, keyed by the same
            route-selector syntax as ``models`` (wildcards and effort
            qualifiers included). Each value is a zero-argument callable
            returning a PromptBuilder — typically a subclass — that the
            routed component populates with its sections.
        session_store: The session store to use for managing sessions.
            ``"local"`` (the default) persists to a SQLite file under the
            Parlant home directory; ``"transient"`` keeps sessions in memory;
            a MongoDB connection string routes to MongoDB; or pass a
            ``SessionStore`` instance for full control.
        customer_store: The customer store to use for managing customers.
            Accepts the same options as ``session_store`` and also defaults
            to ``"local"``.
        auth: Authentication/authorization configuration. Pass an ``AuthConfig`` to
            enable the OSS composite policy (guest/customer/admin principals with
            session ownership), a custom ``AuthorizationPolicy`` instance for full
            control, or None to keep the environment-selected default.
        log_level: The logging level for the server.
        diagnostic_reports: Whether engine turns should produce downloadable diagnostic reports.
        hot_reload: Watch the entry script's directory for Python file changes
            and restart the server in place: graceful teardown, then a re-exec
            of the original command line. Defaults to True; disarms itself when
            there is no file-backed entry script (REPL, notebook). A deployment
            whose files never change pays only for an idle watcher; set False
            to skip the watcher entirely.
        modules: A list of module names to load for the server.
        migrate: Whether to allow database migrations on startup (if needed).
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
        models: ModelRoutes | None = None,
        prompts: PromptRoutes | None = None,
        session_store: Literal["transient", "local"] | str | SessionStore = "local",
        customer_store: Literal["transient", "local"] | str | CustomerStore = "local",
        variable_store: Literal["transient", "local"] | str | ContextVariableStore = "transient",
        auth: AuthConfig | AuthorizationPolicy | None = None,
        log_level: LogLevel = LogLevel.INFO,
        diagnostic_reports: bool = True,
        hot_reload: bool = True,
        modules: list[str] = [],
        migrate: bool = False,
        configure_container: Callable[[Container], Awaitable[Container]] | None = None,
        initialize_container: Callable[[Container], Awaitable[None]] | None = None,
        configure_api: Callable[[FastAPI], Awaitable[None]] | None = None,
    ) -> None:
        self.host = host
        self.port = int(os.environ.get("PORT", port))
        self.tool_service_port = int(os.environ.get("TOOL_SERVICE_PORT", tool_service_port))
        self.log_level = log_level
        self.diagnostic_reports = diagnostic_reports
        self.hot_reload = hot_reload
        self._hot_reload_task: asyncio.Task[None] | None = None
        self._restart_requested = False
        self.prompts: PromptRoutes = dict(prompts or {})
        self.modules = modules

        self._migrate = migrate
        self._models: ModelRoutes = {"*": "openrouter", **(models or {})}
        self._cached_indexer: _CachedIndexer

        self._session_store = session_store
        self._customer_store = customer_store
        self._context_variable_store = variable_store
        self._auth = auth

        self._user_configure_container = configure_container
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

        # Known before startup, so configuration can branch on it while the
        # server is still being built. Re-read from the container once that
        # exists, in case initialization overrode the application context.
        self._environment = resolve_environment()

        self._rule_indexing_requests: dict[
            RuleId,
            tuple[Any, Callable[..., Coroutine[Any, Any, _CachedIndexer.RuleIndexingResult]]],
        ] = {}

        # TRACE is firehose mode: raw logs are the progress display, so no
        # Rich bars compete with them (mirrors the rule-indexing bars).
        self._creation_progress: Progress | None = (
            None
            if log_level == LogLevel.TRACE
            else Progress(
                TextColumn("{task.description}"),
                BarColumn(pulse_style="bold green"),
                TimeElapsedColumn(),
            )
        )
        self._creation_progress_k = 0
        self._creation_progress_task_id: TaskID

    async def _configure_container(self, container: Container) -> Container:
        return container

    @property
    def container(self) -> Container:
        """Returns the dependency injection container."""
        return self._container

    @property
    def logger(self) -> Logger:
        """Returns the logger instance from the container."""
        return self._container[Logger]

    @property
    def environment(self) -> Environment:
        """The environment this server runs in, from ``PARLANT_ENV``.

        Every session the server creates records it, and configuration can
        branch on it — binding fake services outside production, for example.
        """
        return self._environment

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

    def _apply_authorization_policy(self, container: Container) -> Container:
        """Binds the server's explicit ``auth=`` policy into the container.

        No-op when no ``auth`` was passed (the environment-selected default —
        development, production env config, or the Dashboard module's own
        binding — stays in place). Under Dashboard mode, the explicit policy
        becomes the INNER policy of the project-token trust wrapper rather
        than replacing it: the platform keeps managing the server while your
        policy governs everyone else.
        """
        if self._auth is None:
            return container

        container = container.clone()
        auth_binding = self._build_authorization_policy_binding(self._auth)

        dashboard_token = os.environ.get("PARLANT_DASHBOARD_PROJECT_TOKEN", "")
        if not (_should_configure_parlant_dashboard() and dashboard_token):
            container[AuthorizationPolicy] = auth_binding
            return container

        from parlant.adapters.modules.parlant_dashboard.auth import (
            ParlantDashboardAuthorizationPolicy,
        )

        if isinstance(auth_binding, AuthorizationPolicy):
            container[AuthorizationPolicy] = ParlantDashboardAuthorizationPolicy(
                project_token=dashboard_token,
                inner=auth_binding,
            )
        else:
            inner_factory = auth_binding
            wrapped_policy: AuthorizationPolicy | None = None

            def _wrapped_auth_factory(c: Container) -> AuthorizationPolicy:
                nonlocal wrapped_policy
                if wrapped_policy is None:
                    wrapped_policy = ParlantDashboardAuthorizationPolicy(
                        project_token=dashboard_token,
                        inner=inner_factory(c),
                    )
                return wrapped_policy

            container[AuthorizationPolicy] = _wrapped_auth_factory

        return container

    def _build_authorization_policy_binding(
        self,
        auth: AuthConfig | AuthorizationPolicy,
    ) -> AuthorizationPolicy | Callable[[Container], AuthorizationPolicy]:
        if isinstance(auth, AuthorizationPolicy):
            return auth

        # An AuthConfig: build the composite policy lazily (and once) so the
        # customer store is resolved only when the API actually stands up.
        policy: AuthorizationPolicy | None = None

        def factory(c: Container) -> AuthorizationPolicy:
            nonlocal policy

            if policy is None:
                if auth.guest_token_secret is None:
                    c[Logger].warning(
                        "AuthConfig.guest_token_secret is not set; generated an ephemeral "
                        "guest token secret. Guest sessions will not survive a server restart."
                    )

                policy = create_composite_authorization_policy(
                    config=auth,
                    customer_store=c[CustomerStore],
                )

            return policy

        return factory

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
            self._store_provider = self._container[StoreProvider]
            self._environment = self._container[ApplicationContext].environment

            if self.hot_reload and (watch_root := _hot_reload_watch_root()) is not None:
                self._hot_reload_task = asyncio.create_task(self._watch_for_hot_reload(watch_root))

            if self._creation_progress is not None:
                self._creation_progress = self._creation_progress.__enter__()
                self._creation_progress_task_id = self._creation_progress.add_task(
                    "Caching entity embeddings", total=None
                )

            return self

        except SDKError as e:
            await self._unwind_partial_startup()
            _die(str(e), e)
            raise
        except BaseException:
            # Startup can fail after resources are already open — the plugin
            # server most of all. Nothing else will close them, and the
            # process would stay alive with nothing serving it.
            await self._unwind_partial_startup()
            raise

    async def _unwind_partial_startup(self) -> None:
        with suppress(Exception):
            self._cancel_hot_reload_watcher()

        with suppress(Exception):
            await self._exit_stack.aclose()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._finished_setup = True

        if self._creation_progress is not None:
            self._creation_progress.__exit__(None, None, None)
            self._creation_progress = None

        if exc_value is not None:
            self._cancel_hot_reload_watcher()
            await self._startup_context_manager.__aexit__(exc_type, exc_value, tb)
            await self._exit_stack.aclose()
            return False

        try:
            with self._container[Tracer].span(
                "startup.indexing.enrichment",
                attributes={"scope": "Indexing enrichment"},
            ):
                await self._process_indexing_enrichment()

            # Train each agent's recall discriminants now that indexing has populated
            # the rules' signals.
            with self._container[Tracer].span(
                "startup.training",
                attributes={"scope": "Training"},
            ):
                await self._process_training()

            print()

            await self._setup_retrievers()
        except BaseException:
            # Interrupted before the server began serving — Ctrl-C during
            # indexing, most often. Tear down what startup opened: leaving the
            # startup context manager suspended lets the event loop's
            # async-generator finalization race a half-unwound exit stack, and
            # the process then hangs on a thread nobody closed until a second
            # Ctrl-C arrives.
            with suppress(Exception):
                await self._startup_context_manager.__aexit__(*sys.exc_info())

            await self._unwind_partial_startup()

            raise

        # Start health check polling to set ready event when the server is ready to receive requests
        health_check_task = asyncio.create_task(self._poll_health_endpoint())

        try:
            # This actually starts the server
            await self._startup_context_manager.__aexit__(None, None, None)
        except BaseException:
            health_check_task.cancel()
            raise
        finally:
            self._cancel_hot_reload_watcher()
            # Wait for health check to complete before cleanup
            await health_check_task
            await self._exit_stack.aclose()

        if self._restart_requested:
            # Everything is torn down and the port is free: replace this
            # process image with a fresh run of the original command line.
            print("Hot reload: restarting...")
            os.execv(sys.executable, [sys.executable, *sys.argv])

        return False

    async def _watch_for_hot_reload(self, root: Path) -> None:
        """Request an in-place server restart on the first Python file change.

        The restart reuses the regular shutdown path end to end: raising
        SIGINT lets uvicorn drain and the exit stack tear down exactly as a
        Ctrl-C would, and only then does ``__aexit__`` re-exec the original
        command line. One shot by design — after the first change the process
        is on its way out, and the reborn process arms a fresh watcher.
        """

        async for changes in watchfiles.awatch(root, watch_filter=watchfiles.PythonFilter()):
            changed = ", ".join(sorted({path for _, path in changes}))
            self._container[Logger].info(
                f"Hot reload: change detected ({changed}); restarting the server"
            )
            self._restart_requested = True
            signal.raise_signal(signal.SIGINT)
            return

    def _cancel_hot_reload_watcher(self) -> None:
        if self._hot_reload_task is not None and not self._hot_reload_task.done():
            self._hot_reload_task.cancel()
        self._hot_reload_task = None

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

    def _add_rule_indexing_request(
        self,
        rule_id: RuleId,
        rule_content: _RuleText,
        title: str | None = None,
        agent_id: AgentId | None = None,
        title_proposition: bool = False,
        relativity_proposition: bool = True,
        signal_proposition: bool = True,
    ) -> None:
        self._rule_indexing_requests[rule_id] = (
            (
                rule_id,
                rule_content,
                title,
                agent_id,
                relativity_proposition,
                signal_proposition,
                title_proposition,
            ),
            self._cached_indexer.index_rule,
        )

    @staticmethod
    async def _create_rule_handler_shim(
        user_callback: RuleHandler,
        rule_id: RuleId,
        core_ctx: EngineContext,
        core_evaluation: _RuleEvaluation,
    ) -> None:
        """Translate a core evaluation and invoke an SDK rule callback.

        The callback's arity is adapted here: it may accept
        ``(context, evaluation)``, just ``(context)``, or no parameters at
        all, and receives as many of the two arguments as it takes.
        """
        sdk_evaluation = RuleEvaluation(
            id=rule_id,
            is_relevant=core_evaluation.is_relevant,
            reasoning=core_evaluation.reasoning,
            score=core_evaluation.score,
            highlights=core_evaluation.highlights,
            metadata=core_evaluation.metadata,
        )

        arguments: tuple[Any, ...] = (core_ctx, sdk_evaluation)
        parameters = inspect.signature(user_callback).parameters.values()

        if not any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
            positional_count = sum(
                1
                for parameter in parameters
                if parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
            arguments = arguments[: min(positional_count, len(arguments))]

        await cast(Callable[..., Awaitable[None]], user_callback)(*arguments)

    @staticmethod
    async def _create_field_provider_shim(
        provider: CannedResponseFieldProvider,
        core_ctx: EngineContext,
        core_evaluation: _RuleEvaluation,
    ) -> None:
        """Invoke a context-aware or parameterless field provider and stage its fields."""
        arguments: tuple[Any, ...] = (core_ctx,)
        parameters = inspect.signature(provider).parameters.values()

        if not any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
            positional_count = sum(
                1
                for parameter in parameters
                if parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
            arguments = arguments[: min(positional_count, len(arguments))]

        fields = await cast(Callable[..., Awaitable[Mapping[str, Any]]], provider)(*arguments)
        core_ctx.state.additional_canned_response_fields.update(fields)

    async def _create_rule(
        self,
        condition: str | None,
        content: str | None,
        title: str | None,
        tools: Iterable[ToolRef],
        metadata: dict[str, JSONSerializable],
        weight: Weight | None,
        effort: Effort | None,
        canned_responses: Sequence[CannedResponseId],
        evaluation_func: Callable[[RuleEvaluationContext, Any], Awaitable[RuleEvaluation]] | None,
        on_discovered: RuleHandler | None,
        on_matched: RuleHandler | None,
        canned_response_field_provider: CannedResponseFieldProvider | None,
        groups: Sequence[GroupId] | None,
        relationship_target_group_id: GroupId | None,
        id: RuleId | None = None,
        labels: Iterable[str] = (),
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
        message_review: bool = False,
        tool_review: bool = False,
        agent_id: AgentId | None = None,
        *,
        rule_kind: Literal["observation", "policy"],
    ) -> Rule:
        """Internal method to create a rule with common logic."""
        self._advance_creation_progress()

        tools_list = list(tools)

        await _enable_tool_refs(self._plugin_server, tools_list)

        rule = await self._store_provider.get_store(
            RuleStore, StoreProviderHints(call_site="sdk")
        ).create_rule(
            content=content,
            condition=condition,
            title=title,
            weight=weight,
            metadata=metadata,
            effort=effort,
            id=id,
            groups=groups,
            labels=set(labels) if labels else None,
            priority=priority,
            signals=signals,
            anti_signals=anti_signals,
            message_review=message_review,
            tool_review=tool_review,
        )

        if canned_responses:
            group_id = GroupIds.for_rule_id(rule.id)

            for canrep_id in canned_responses:
                await self._store_provider.get_store(
                    CannedResponseStore, StoreProviderHints(call_site="sdk")
                ).upsert_group(
                    canned_response_id=canrep_id,
                    group_id=group_id,
                )

        # Evaluate what evaluator to use if a custom evaluation function isn't specified.
        needs_title_proposition = rule_kind == "policy" and not title

        if evaluation_func is None or needs_title_proposition:
            self._add_rule_indexing_request(
                rule.id,
                _RuleText(content=content, condition=condition),
                title=title,
                agent_id=agent_id,
                title_proposition=needs_title_proposition,
                relativity_proposition=evaluation_func is None,
                signal_proposition=evaluation_func is None,
            )

        # Create relationship if target group specified
        if relationship_target_group_id is not None:
            await self._store_provider.get_store(
                RelationshipStore, StoreProviderHints(call_site="sdk")
            ).create_relationship(
                source=RelationshipEntity(
                    id=rule.id,
                    kind=RelationshipEntityKind.RULE,
                ),
                target=RelationshipEntity(
                    id=relationship_target_group_id,
                    kind=RelationshipEntityKind.GROUP_ALL,
                ),
                kind=RelationshipKind.DEPENDENCY,
            )

        for t in list(tools):
            await self._store_provider.get_store(
                RuleToolAssociationStore, StoreProviderHints(call_site="sdk")
            ).create_association(
                rule_id=rule.id,
                tool_id=_tool_ref_to_id(t),
            )

        rule_common = {
            "id": rule.id,
            "groups": _tags_from_ids(rule.groups),
            "metadata": rule.metadata,
            "title": rule.title,
            "content": rule.content,
            "labels": rule.labels,
            "trigger_effort": rule.effort,
            "priority": rule.priority,
            "_server": self,
            "_container": self.container,
            "_store_provider": self._store_provider,
        }

        match rule_kind:
            case "observation":
                result_rule: Rule = Observation(
                    **rule_common,  # type: ignore
                    condition=rule.condition or "",
                )
            case "policy":
                result_rule = Policy(**rule_common)  # type: ignore

        if evaluation_func is not None:
            # Register the evaluator in the engine-agnostic registry, wrapped in a
            # shim that translates between the (neutral) EngineContext and the
            # SDK's RuleEvaluationContext. Whichever engine runs consumes the
            # registry — the SDK doesn't depend on any engine's matching internals.
            async def shim_evaluator(
                engine_ctx: EngineContext, core_rule: _Rule
            ) -> _RuleEvaluation:
                sdk_ctx = await RuleEvaluationContext._from_engine_context(
                    engine_ctx=engine_ctx,
                    server=self,
                    container=self.container,
                )
                result = await evaluation_func(sdk_ctx, result_rule)

                return _RuleEvaluation(
                    rule=core_rule,
                    reasoning=result.reasoning,
                    is_relevant=result.is_relevant,
                    score=result.score,
                    highlights=result.highlights,
                    metadata=result.metadata,
                )

            self.container[RuleEvaluatorRegistry].register(
                rule.id,
                shim_evaluator,
                phase=(
                    RuleEvaluatorPhase.EARLY
                    if isinstance(evaluation_func, _SessionBudgetEvaluator)
                    else RuleEvaluatorPhase.NORMAL
                ),
            )

        if (
            on_discovered is not None
            or on_matched is not None
            or canned_response_field_provider is not None
        ):
            engine_hooks = self.container[CompassHooks]

            if on_discovered is not None:
                shim = partial(
                    Server._create_rule_handler_shim,
                    on_discovered,
                    rule.id,
                )
                engine_hooks.on_rule_discovered_handlers[rule.id].append(shim)

            if on_matched is not None:
                shim = partial(
                    Server._create_rule_handler_shim,
                    on_matched,
                    rule.id,
                )
                engine_hooks.on_rule_matched_handlers[rule.id].append(shim)

            if canned_response_field_provider is not None:
                shim = partial(
                    Server._create_field_provider_shim,
                    canned_response_field_provider,
                )
                engine_hooks.on_rule_matched_handlers[rule.id].append(shim)

        return result_rule

    async def _render_rule(self, rule_id: RuleId) -> str:
        rule = await self._store_provider.get_store(
            RuleStore, StoreProviderHints(call_site="sdk")
        ).read_rule(rule_id)

        if rule.title:
            return rule.title

        if rule.condition:
            return f"When {rule.condition}"

        return rule.content[:50] + "..." if rule.content else "Untitled rule"

    def _attach_conditional_retriever(
        self,
        retriever_id: str,
        retriever: Callable[[RetrieverContext], Awaitable[RetrieverResult | None]],
        should_run: Callable[[EngineContext], bool],
    ) -> None:
        """Register a retriever that fires only when the condition is met.

        Args:
            retriever_id: Unique identifier for this retriever.
            retriever: The retriever function to call.
            should_run: A function that takes EngineContext and returns True if the retriever should run.
        """

        async def on_generating_messages(
            ctx: EngineContext,
            payload: Any,
            exc: Optional[Exception],
        ) -> CompassHookResult:
            if not should_run(ctx):
                return CompassHookResult.CALL_NEXT

            # Build RetrieverContext
            agent = await self.get_agent(id=ctx.agent.id)
            customer = await self.get_customer(id=ctx.customer.id)

            retriever_context = RetrieverContext(
                server=self,
                container=self._container,
                logger=self._container[Logger],
                tracer=ctx.tracer,
                session=Session(
                    id=ctx.session.id,
                    interaction=ctx.interaction,
                    metadata=ctx.session.metadata,
                    labels=ctx.session.labels,
                    customer=customer,
                    agent=agent,
                    mode=ctx.session.mode,
                    title=ctx.session.title,
                    _metadata_workspace=ctx.session_metadata,
                ),
                agent=agent,
                customer=customer,
                variables={
                    await agent.get_variable(id=var.id): val.data
                    for var, val in ctx.state.context_variables
                },
                interaction=ctx.interaction,
            )

            # Call retriever
            result = await retriever(retriever_context)

            if result is None:
                return CompassHookResult.CALL_NEXT

            if not (
                result.data
                or result.metadata
                or result.canned_responses
                or result.canned_response_fields
            ):
                # No need to emit tool event if nothing was retrieved.
                return CompassHookResult.CALL_NEXT

            # Build the tool result
            tool_result = _SessionToolResult(
                data=result.data,
                metadata=result.metadata,
                control={"lifespan": "response"},
                canned_responses=[u for u in result.canned_responses],
                canned_response_fields=result.canned_response_fields,
            )

            if result.rules:
                tool_result["rules"] = list(result.rules)

            # Emit tool event with retriever data
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
                                rationale="Retriever result",
                                result=tool_result,
                            )
                        ]
                    ),
                )
            )

            return CompassHookResult.CALL_NEXT

        self._container[CompassHooks].on_generating_messages.append(on_generating_messages)

    async def _process_indexing_enrichment(self) -> None:
        def create_indexing_task(
            indexing: Coroutine[Any, Any, _CachedIndexer.RuleIndexingResult],
            entity_id: RuleId,
        ) -> asyncio.Task[tuple[RuleId, _CachedIndexer.RuleIndexingResult]]:
            async def task_wrapper() -> tuple[RuleId, _CachedIndexer.RuleIndexingResult]:
                result = await indexing
                return (entity_id, result)

            return asyncio.create_task(task_wrapper(), name=f"rule_indexing_{entity_id}")

        tasks: list[asyncio.Task[tuple[RuleId, _CachedIndexer.RuleIndexingResult]]] = []

        for rule_id, (args, func) in self._rule_indexing_requests.items():
            tasks.append(create_indexing_task(func(*args), rule_id))

        if not tasks:
            return

        if self.log_level == LogLevel.TRACE:
            indexing_results = await async_utils.safe_gather(*tasks)
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

            with Live(RichGroup(overall_progress, entity_progress), refresh_per_second=10):
                bar_id: dict[str, int] = {}

                for t in tasks:
                    entity_id = RuleId(t.get_name().split("_")[-1])
                    rendered_rule = await self._render_rule(entity_id)

                    bar_id[entity_id] = entity_progress.add_task(
                        rendered_rule[:50],
                        total=100,
                    )

                overall = overall_progress.add_task("Enriching entity metadata", total=100)

                gather = asyncio.create_task(async_utils.safe_gather(*tasks))

                while not gather.done():
                    unfinished: list[tuple[str, float]] = []

                    for _id, rich_id in bar_id.items():
                        pct = self._cached_indexer._progress_for(_id)
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

                    overall_pct = sum(
                        self._cached_indexer._progress_for(e_id) for e_id in bar_id
                    ) / len(bar_id)
                    overall_progress.update(overall, completed=overall_pct)

                    await asyncio.sleep(0.2)

                for e_id, rich_id in bar_id.items():
                    entity_progress.remove_task(
                        TaskID(rich_id),
                    )

                entity_progress.refresh()
                overall_progress.update(overall, completed=100)
                indexing_results = await gather

        total_results = len(indexing_results)

        if self.log_level == LogLevel.TRACE or total_results == 0:
            await self._apply_indexing_results(indexing_results)
        else:
            applying_progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                TaskProgressColumn(style="bold blue"),
                TimeElapsedColumn(),
            )

            with applying_progress:
                bar = applying_progress.add_task("Applying indexing results", total=total_results)

                # For UX: Stretch the bar to ~1s total so it's visible even when
                # processing is fast; pure no-op once natural work exceeds it.
                per_item_budget = 1.0 / total_results

                for i, (entity_id, result) in enumerate(indexing_results):
                    item_started = asyncio.get_running_loop().time()
                    await self._apply_single_indexing_result(entity_id, result)
                    item_ended = asyncio.get_running_loop().time()

                    applying_progress.update(bar, completed=i + 1)

                    # If processing was very fast, wait a bit to meet the allotted budget
                    if (remaining := per_item_budget - (item_ended - item_started)) > 0:
                        await asyncio.sleep(remaining)

    async def _process_training(self) -> None:
        training_service = self._container[TrainingService]

        if self.log_level == LogLevel.TRACE:
            await training_service.train()
            return

        training_progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(style="bold blue"),
            TimeElapsedColumn(),
        )

        with training_progress:
            bar = training_progress.add_task("Training classifiers", total=100)

            async def callback(pct: float) -> None:
                training_progress.update(bar, completed=pct)

            await training_service.train(progress_report=ProgressReport(callback))
            training_progress.update(bar, completed=100)

    async def _apply_indexing_results(
        self,
        indexing_results: Sequence[tuple[RuleId, _CachedIndexer.RuleIndexingResult]],
    ) -> None:
        for entity_id, result in indexing_results:
            await self._apply_single_indexing_result(entity_id, result)

    async def _apply_single_indexing_result(
        self,
        entity_id: RuleId,
        result: _CachedIndexer.RuleIndexingResult,
    ) -> None:
        store = self._store_provider.get_store(RuleStore, StoreProviderHints(call_site="sdk"))
        rule = await store.read_rule(rule_id=entity_id)
        params: RuleUpdateParams = {}

        if result.references_other_policies is not None:
            # set_internal_value stamps the schema_version the strict internal-
            # metadata reader requires; hand-rolling the root would write an
            # unversioned block that fails validation at engine preload.
            updated_metadata = metadata_utils.set_internal_value(
                rule.metadata,
                ("references_other_policies",),
                result.references_other_policies,
            )
            if updated_metadata != rule.metadata:
                await store.set_metadata(
                    rule_id=entity_id,
                    key=metadata_utils.PARLANT_METADATA_KEY,
                    value=updated_metadata[metadata_utils.PARLANT_METADATA_KEY],
                )

        generated_signals = list(result.signals)
        if generated_signals:
            merged_signals: list[str] = []
            seen_signals: set[str] = set()

            for signal in [*rule.signals, *generated_signals]:
                if not (clean := signal.strip()):
                    continue

                key = clean.casefold()
                if key in seen_signals:
                    continue

                seen_signals.add(key)
                merged_signals.append(clean)

            if list(rule.signals) != merged_signals:
                params["signals"] = merged_signals

        generated_anti_signals = list(result.anti_signals)
        if generated_anti_signals:
            merged_anti_signals: list[str] = []
            seen_anti_signals: set[str] = set()

            for anti_signal in [*rule.anti_signals, *generated_anti_signals]:
                if not (clean := anti_signal.strip()):
                    continue

                key = clean.casefold()
                if key in seen_anti_signals:
                    continue

                seen_anti_signals.add(key)
                merged_anti_signals.append(clean)

            if list(rule.anti_signals) != merged_anti_signals:
                params["anti_signals"] = merged_anti_signals

        if result.title and not rule.title:
            params["title"] = result.title

        if params:
            await store.update_rule(rule_id=entity_id, params=params)

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
            ) -> CompassHookResult:
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
                            metadata=ctx.session.metadata,
                            labels=ctx.session.labels,
                            customer=customer,
                            agent=agent,
                            mode=ctx.session.mode,
                            title=ctx.session.title,
                            _metadata_workspace=ctx.session_metadata,
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

                task = asyncio.create_task(
                    cast(
                        Coroutine[Any, Any, RetrieverResult | None | DeferredRetriever],
                        coroutine,
                    ),
                    name=f"Retriever {retriever_id} for agent {agent_id}",
                )
                if (managed_tasks := getattr(ctx.state, "background_tasks", None)) is not None:
                    managed_tasks.add(task)
                tasks_for_this_retriever[ctx.tracer.trace_id] = (
                    Timeout(600),  # Expiration timeout for garbage collection purposes
                    task,
                )

                return CompassHookResult.CALL_NEXT

            async def on_generating_messages(
                ctx: EngineContext,
                payload: Any,
                exc: Optional[Exception],
            ) -> CompassHookResult:
                if timeout_and_task := tasks_for_this_retriever.pop(ctx.tracer.trace_id, None):
                    _, task = timeout_and_task
                    try:
                        task_result = await task
                    finally:
                        if (
                            managed_tasks := getattr(ctx.state, "background_tasks", None)
                        ) is not None:
                            managed_tasks.discard(task)

                    # Check if the result is a deferred callable
                    if callable(task_result):
                        # Call the deferred callable with the EngineContext
                        final_result = await task_result(ctx)
                        if final_result is None:
                            # Deferred callable decided not to return data
                            return CompassHookResult.CALL_NEXT
                        task_result = final_result

                    # Handle None result
                    if task_result is None:
                        return CompassHookResult.CALL_NEXT

                    # task_result must be a RetrieverResult at this point
                    retriever_result = task_result

                    if not (
                        retriever_result.data
                        or retriever_result.metadata
                        or retriever_result.canned_responses
                        or retriever_result.canned_response_fields
                        or retriever_result.rules
                    ):
                        # No need to emit tool event if nothing was retrieved.
                        return CompassHookResult.CALL_NEXT

                    # Build the tool result
                    tool_result = _SessionToolResult(
                        data=retriever_result.data,
                        metadata=retriever_result.metadata,
                        control={"lifespan": "response"},
                        canned_responses=[u for u in retriever_result.canned_responses],
                        canned_response_fields=retriever_result.canned_response_fields,
                    )

                    if retriever_result.rules:
                        tool_result["rules"] = list(retriever_result.rules)

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
                                        rationale="Retriever",
                                        arguments={},
                                        result=tool_result,
                                    )
                                ]
                            ),
                        )
                    )

                return CompassHookResult.CALL_NEXT

            c[CompassHooks].on_preparing.append(on_message_acknowledged)
            c[CompassHooks].on_generating_messages.append(on_generating_messages)

        for agent in self._retrievers:
            for retriever_id, retriever in self._retrievers[agent].items():
                await setup_retriever(self._container, agent, retriever_id, retriever)

    async def get_group(
        self,
        *,
        id: GroupId | None = None,
        name: str | None = None,
    ) -> Group:
        if (id is None) == (name is None):
            raise SDKError("Exactly one of 'id' or 'name' must be provided.")

        if id is not None:
            group = await self._container[GroupStore].read_group(group_id=id)
        else:
            assert name is not None
            groups = await self._container[GroupStore].list_groups(name=name)
            if not groups:
                raise SDKError(f"Group with name '{name}' not found.")
            group = groups[0]

        return Group(
            id=group.id,
            name=group.name,
            _server=self,
        )

    async def create_session_budget(
        self,
        *,
        name: str,
        value: Decimal,
        window: timedelta | None = None,
    ) -> SessionBudget:
        """Create a session budget, with or without a configured price table.

        The budget is denominated in the server's effective pricing currency:
        the price table's currency when one was set via :meth:`set_pricing`,
        and otherwise USD — the denomination of provider-reported charges
        (OpenRouter, LiteLLM), which is what a table-less budget measures. If
        a turn includes an operation whose cost is neither provider-reported
        nor covered by the price table, measuring the budget raises a
        ``PricingConfigurationError`` rather than under-counting.
        """
        if value < 0:
            raise SDKError("Session budget value must be non-negative")
        if window is not None and window <= timedelta(0):
            raise SDKError("Session budget window must be positive")

        configured_pricing = self.container[PricingRegistry].pricing
        definition = SessionBudgetDefinition(
            name=name,
            value=value,
            currency=configured_pricing.currency if configured_pricing else "USD",
            window=window,
        )
        self.container[PricingRegistry].register_budget(definition)
        return SessionBudget(definition, self)

    async def set_pricing(
        self,
        models: Mapping[str, ModelPricing],
        *,
        currency: str = "USD",
    ) -> Pricing:
        """Set the server-wide model prices used for reporting and budgets.

        Prices are expressed per one million usage units. A key is either a
        bare model id (``"gpt-5.5"``), the model's price under any provider, or
        ``"provider:model"`` (``"azure-eu:gpt-5.5"``), its price when served by
        that one provider; the exact key wins where both apply. Replacing the
        pricing affects future traces only; traces already in flight retain the
        immutable pricing snapshot they bound on their first usage report.
        """
        if not currency:
            raise SDKError("Pricing currency cannot be empty")

        pricing = _Pricing(currency=currency, models=dict(models))
        self.container[PricingRegistry].set_pricing(pricing)
        return Pricing(pricing, self)

    async def get_session(self, *, id: SessionId | str) -> Session:
        session_id = SessionId(id)
        core_session = await self._store_provider.get_store(
            SessionStore, StoreProviderHints(call_site="sdk")
        ).read_session(session_id)
        agent = await self.get_agent(id=core_session.agent_id)
        customer = await self.get_customer(id=core_session.customer_id)
        events = await self.container[EntityQueries].find_events(session_id)
        return Session(
            id=session_id,
            interaction=Interaction(events),
            metadata=core_session.metadata,
            labels=core_session.labels,
            customer=customer,
            agent=agent,
            mode=core_session.mode,
            title=core_session.title,
            _server=self,
        )

    async def create_group(self, name: str) -> Group:
        self._advance_creation_progress()

        group = await self._store_provider.get_store(
            GroupStore, StoreProviderHints(call_site="sdk")
        ).create_group(name=name)

        return Group(
            id=group.id,
            name=group.name,
            _server=self,
        )

    async def create_agent(
        self,
        name: str,
        instructions: str,
        *,
        id: str,
        output_mode: OutputMode = OutputMode.STREAM,
        max_steps: int | None = None,
        groups: Sequence[GroupId] = [],
        preamble_config: PreambleConfiguration | None = None,
        engine: Literal["compass"] = "compass",
        effort: Effort = Effort.MEDIUM,
        moderation: Moderation = Moderation.AUTO,
        models: ModelRoutes | None = None,
        prompts: PromptRoutes | None = None,
        message_review: bool = False,
        tool_review: bool = False,
    ) -> Agent:
        """Creates a new agent with the specified name and instructions.

        Args:
            name: The agent's name (required).
            instructions: The agent's instructions (required).
            output_mode: How the agent delivers responses. Defaults to STREAM.
                - BLOCK: Complete response delivered after generation finishes
                - STREAM: Response streamed progressively as generated
            max_steps: Maximum number of engine steps per turn.
                Defaults to 15 if not specified.
            groups: List of group IDs to associate with the agent. Defaults to empty list.
            id: The agent's ID (required, keyword-only). An explicit ID keeps the
                agent's identity stable across restarts and content edits, which
                everything durable (sessions, discovered rules, model routes) keys
                on. IDs can be any string format and are useful for maintaining consistent
                agent identifiers across deployments or integrations.
            preamble_config: Optional preamble configuration for this agent.
                Controls how the Compass engine handles text immediately before tool
                calls. Use ``PreambleConfiguration.default()``, ``encourage()``, or
                ``discourage()``. Only supported by the Compass engine.
            engine: Which engine this agent runs on ("compass").
            effort: How much effort the agent invests in processing. Defaults to MEDIUM.
            models: Optional model-route overrides for this agent.
            prompts: Optional prompt-builder providers for this agent,
                overriding server-wide ``prompts`` for the same routes.
            message_review: Review every drafted message before delivery (compass engine).
                Matched rules may additionally request message review per turn via
                ``trigger_message_review``.
            tool_review: Review every proposed tool call before execution (compass engine).
                Matched rules (via ``trigger_tool_review``) and consequential tools may
                additionally request tool review.

        Returns:
            The created Agent instance.
        """

        self._advance_creation_progress()

        agent = await self._store_provider.get_store(
            AgentStore, StoreProviderHints(call_site="sdk")
        ).create_agent(
            name=name,
            instructions=instructions,
            max_steps=max_steps or 15,
            message_output_mode=output_mode,
            engine=engine,
            effort=effort,
            moderation=moderation,
            id=AgentId(id),
            message_review=message_review,
            tool_review=tool_review,
        )

        if models:
            try:
                await self._container[ModelConfigurationService].configure(
                    ModelScope.agent(agent.id),
                    models,
                )
            except Exception:
                await self._store_provider.get_store(
                    AgentStore, StoreProviderHints(call_site="sdk")
                ).delete_agent(agent.id)
                raise

        if prompts:
            try:
                await self._container[PromptConfigurationService].configure(
                    ModelScope.agent(agent.id),
                    prompts,
                )
            except Exception:
                await self._store_provider.get_store(
                    AgentStore, StoreProviderHints(call_site="sdk")
                ).delete_agent(agent.id)
                raise

        if preamble_config is not None:
            self._container[PreambleConfigurationProvider].set(agent.id, preamble_config)

        return Agent(
            id=agent.id,
            name=agent.name,
            instructions=agent.instructions,
            max_steps=agent.max_steps,
            output_mode=agent.message_output_mode or OutputMode.BLOCK,
            groups=_tags_from_ids(groups),
            models=await self._container[ModelConfigurationService].read(
                ModelScope.agent(agent.id)
            ),
            _server=self,
            _container=self._container,
            _store_provider=self._store_provider,
        )

    async def list_agents(self) -> Sequence[Agent]:
        """Lists all agents."""

        agents = await self._store_provider.get_store(
            AgentStore, StoreProviderHints(call_site="sdk")
        ).list_agents()

        return [
            Agent(
                id=a.id,
                name=a.name,
                instructions=a.instructions,
                max_steps=a.max_steps,
                output_mode=a.message_output_mode or OutputMode.BLOCK,
                groups=_tags_from_ids(a.groups),
                models=await self._container[ModelConfigurationService].read(
                    ModelScope.agent(a.id)
                ),
                _server=self,
                _container=self._container,
                _store_provider=self._store_provider,
            )
            for a in agents
        ]

    async def find_agent(self, *, id: str) -> Agent | None:
        """Finds an agent by its ID."""

        try:
            agent = await self._store_provider.get_store(
                AgentStore, StoreProviderHints(call_site="sdk")
            ).read_agent(AgentId(id))

            return Agent(
                id=agent.id,
                name=agent.name,
                instructions=agent.instructions,
                max_steps=agent.max_steps,
                output_mode=agent.message_output_mode or OutputMode.BLOCK,
                groups=_tags_from_ids(agent.groups),
                models=await self._container[ModelConfigurationService].read(
                    ModelScope.agent(agent.id)
                ),
                _server=self,
                _container=self._container,
                _store_provider=self._store_provider,
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
        groups: Sequence[Group] = [],
        kind: CustomerKind = CustomerKind.REAL,
        *,
        id: str,
    ) -> Customer:
        """Creates a new customer with the specified name and metadata.

        Args:
            name: The customer's name (required). An arbitrary string that
                identifies and/or describes the customer.
            metadata: Key-value pairs to describe the customer. Defaults to
                empty dictionary. This allows you to store arbitrary metadata
                about the customer (e.g., email, VIP status, preferences).
            groups: Groups to associate with the customer (see
                ``server.create_group`` / ``server.get_group``). Defaults to
                empty. Groups are useful for categorizing and filtering
                customers.
            id: The customer's ID (required). Can be any string format; use a
                stable identifier from your own systems (e.g., your internal
                CRM customer ID) so customers keep consistent identities
                across deployments and integrations. Auto-provisioned
                customers (AuthConfig.auto_provision_customers) likewise use
                the customer ID carried by their JWT.
            kind: Whether this is a real customer or a persona used for
                testing. Defaults to ``CustomerKind.REAL``. A persona cannot
                be authenticated as, so no end user is ever served as one,
                and only an admin may open a session with it. The kind is
                fixed at creation: to change it, delete the customer and
                create it again.

        Returns:
            The created Customer instance.
        """

        self._advance_creation_progress()

        customer = await self._store_provider.get_store(
            CustomerStore, StoreProviderHints(call_site="sdk")
        ).create_customer(
            name=name,
            metadata=metadata,
            groups=[group.id for group in groups],
            id=CustomerId(id),
            kind=kind,
        )

        return Customer(
            id=customer.id,
            name=customer.name,
            metadata=customer.metadata,
            groups=_tags_from_ids(customer.groups),
            kind=customer.kind,
            _server=self,
        )

    async def list_customers(self) -> Sequence[Customer]:
        """Lists all customers."""

        customers = await self._store_provider.get_store(
            CustomerStore, StoreProviderHints(call_site="sdk")
        ).list_customers()

        return [
            Customer(
                id=c.id,
                name=c.name,
                metadata=c.metadata,
                groups=_tags_from_ids(c.groups),
                kind=c.kind,
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
                customer = await self._store_provider.get_store(
                    CustomerStore, StoreProviderHints(call_site="sdk")
                ).read_customer(CustomerId(id))
            except ItemNotFoundError:
                return None

            return Customer(
                id=customer.id,
                name=customer.name,
                metadata=customer.metadata,
                groups=_tags_from_ids(customer.groups),
                kind=customer.kind,
            )

        if name:
            customers = await self._store_provider.get_store(
                CustomerStore, StoreProviderHints(call_site="sdk")
            ).list_customers()

            if customer := next((c for c in customers if c.name == name), None):
                return Customer(
                    id=customer.id,
                    name=customer.name,
                    metadata=customer.metadata,
                    groups=_tags_from_ids(customer.groups),
                    kind=customer.kind,
                )

        return None

    async def get_customer(self, *, id: CustomerId) -> Customer:
        """Retrieves a customer by its ID, raising an error if not found."""

        if customer := await self.find_customer(id=id):
            return customer
        raise SDKError(f"Customer with id {id} not found.")

    async def create_canned_response(
        self,
        template: str,
        groups: Sequence[Group] = [],
        signals: Sequence[str] = [],
        metadata: Mapping[str, JSONSerializable] = {},
        field_dependencies: Sequence[str] = (),
        note: str = "",
    ) -> CannedResponseId:
        """Creates a canned response with the specified template, scope, retrieval signals,
        and an optional author note for generation and review."""

        self._advance_creation_progress()

        canrep = await self._store_provider.get_store(
            CannedResponseStore, StoreProviderHints(call_site="sdk")
        ).create_canned_response(
            value=template,
            groups=[t.id for t in groups],
            fields=[],
            signals=signals,
            metadata=metadata,
            field_dependencies=field_dependencies,
            note=note,
        )

        return canrep.id

    def _get_startup_params(self) -> StartupParameters:
        async def override_stores_with_transient_versions(c: Callable[[], Container]) -> None:
            for interface, implementation in [
                (AgentStore, AgentDocumentStore),
                (GroupStore, GroupDocumentStore),
                (ModelRouteStore, ModelRouteDocumentStore),
                (RuleToolAssociationStore, RuleToolAssociationDocumentStore),
                (RelationshipStore, RelationshipDocumentStore),
            ]:
                if interface not in c().defined_types:
                    c()[interface] = await self._exit_stack.enter_async_context(
                        implementation(c()[IdGenerator], TransientDocumentDatabase())  #  type: ignore
                    )

            c()[IndexingStore] = await self._exit_stack.enter_async_context(
                IndexingDocumentStore(TransientDocumentDatabase())
            )

            await configure_model_routing(c(), self._models)

            if self.prompts:
                await c()[PromptConfigurationService].configure(
                    ModelScope.server(),
                    self.prompts,
                )

            def make_transient_db() -> Awaitable[DocumentDatabase]:
                async def shim() -> DocumentDatabase:
                    return TransientDocumentDatabase()

                return shim()

            def make_sqlite_db(file_path: Path) -> Awaitable[DocumentDatabase]:
                return self._exit_stack.enter_async_context(
                    SQLiteDocumentDatabase(
                        c()[Logger],
                        file_path,
                    ),
                )

            mongo_client: object | None = None
            mongo_db: DocumentDatabase | None = None
            mongo_connection_settings: _MongoConnectionSettings | None = None

            async def make_mongo_db(url: str) -> DocumentDatabase:
                nonlocal mongo_client
                nonlocal mongo_db
                nonlocal mongo_connection_settings

                if importlib.util.find_spec("pymongo") is None:
                    raise SDKError(
                        "MongoDB requires an additional package to be installed. "
                        "Please install parlant[mongo] to use MongoDB."
                    )

                from pymongo import AsyncMongoClient
                from pymongo.uri_parser import parse_uri
                from parlant.adapters.db.mongo_db import MongoDocumentDatabase

                requested_settings = _parse_mongo_connection_settings(url, parse_uri)
                if mongo_connection_settings is None:
                    mongo_connection_settings = requested_settings
                else:
                    _ensure_same_mongo_connection(
                        mongo_connection_settings,
                        requested_settings,
                    )

                if mongo_client is None:
                    mongo_client = await self._exit_stack.enter_async_context(
                        AsyncMongoClient[Any](url)
                    )

                if mongo_db is None:
                    mongo_db = await self._exit_stack.enter_async_context(
                        MongoDocumentDatabase(
                            mongo_client=cast(AsyncMongoClient[Any], mongo_client),
                            database_name=requested_settings.database_name,
                            logger=c()[Logger],
                        )
                    )

                assert mongo_db

                return mongo_db

            async def make_persistable_store(t: type[T], spec: str, name: str, **kwargs: Any) -> T:
                store: T

                if spec in ["transient", "local"]:
                    store = await self._exit_stack.enter_async_context(
                        t(
                            database=await cast(
                                dict[str, Callable[[], Awaitable[DocumentDatabase]]],
                                {
                                    "transient": make_transient_db,
                                    "local": lambda: make_sqlite_db(
                                        PARLANT_HOME_DIR / f"{name}.db"
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
                            database=await make_mongo_db(spec),
                            allow_migration=self._migrate,
                            collections_prefix=name,
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
            elif SessionStore not in c().defined_types:
                c()[SessionStore] = await make_persistable_store(
                    SessionDocumentStore, self._session_store, "sessions"
                )

            if isinstance(self._customer_store, CustomerStore):
                c()[CustomerStore] = self._customer_store
            elif CustomerStore not in c().defined_types:
                c()[CustomerStore] = await make_persistable_store(
                    CustomerDocumentStore,
                    self._customer_store,
                    "customers",
                    id_generator=c()[IdGenerator],
                )

            if isinstance(self._context_variable_store, ContextVariableStore):
                c()[ContextVariableStore] = self._context_variable_store
            elif ContextVariableStore not in c().defined_types:
                c()[ContextVariableStore] = await make_persistable_store(
                    ContextVariableDocumentStore,
                    self._context_variable_store,
                    "context_variables",
                    id_generator=c()[IdGenerator],
                )

            for vector_store_interface, vector_store_type, route in [
                (GlossaryStore, GlossaryVectorStore, "embedding.glossary"),
                (
                    CannedResponseStore,
                    CannedResponseVectorStore,
                    "embedding.canned-responses",
                ),
                (RuleStore, RuleVectorStore, "embedding.rules"),
            ]:
                if vector_store_interface not in c().defined_types:

                    async def get_embedder(route: str = route) -> Embedder:
                        return await c()[ModelRouter].get_embedder(context=None, route=route)

                    c()[vector_store_interface] = await self._exit_stack.enter_async_context(
                        vector_store_type(
                            id_generator=c()[IdGenerator],
                            vector_db=TransientVectorDatabase(
                                c()[Logger],
                                c()[Tracer],
                                lambda: c()[EmbeddingCache],
                            ),
                            document_db=TransientDocumentDatabase(),
                            embedder_provider=get_embedder,
                        )  # type: ignore
                    )

        def get_env_based_module() -> ModuleType | None:
            if env_module_name := os.getenv("PARLANT_SDK_MODULE"):
                try:
                    return importlib.import_module(env_module_name)
                except ImportError as e:
                    raise SDKError(
                        f"Failed to import module '{env_module_name}' specified in PARLANT_SDK_MODULE environment variable."
                    ) from e
            return None

        # The SDK's stores and providers resolve dependencies lazily through
        # get_latest_container(). The container object is REPLACED (not
        # mutated) at several points after configure returns — the server
        # clones it for diagnostic-report wiring, modules may return fresh ones — so
        # `latest_container` must track the newest one (initialize rebinds it
        # to the final container), or closures built here resolve against a
        # stale pre-clone object that never received later registrations
        # (e.g. EmbeddingCache, defined during server initialization).
        latest_container: Container | None = None

        def get_latest_container() -> Container:
            assert latest_container is not None
            return latest_container

        async def configure(c: Container) -> Container:
            nonlocal latest_container
            latest_container = c
            c[StoreProvider] = BasicStoreProvider(get_latest_container)

            # Emission and tool-service infrastructure is registered BEFORE the
            # user's configure_container hook, so the hook can interact with it
            # (e.g. container[ServiceRegistry].update_tool_service(...)). The
            # defined_types guards keep a user-supplied binding authoritative.
            if SessionEventTap not in c.defined_types:
                c[SessionEventTap] = SessionEventTap()

            c[EventEmitterFactory] = EventPublisherFactory(
                store_provider_factory=lambda: get_latest_container()[StoreProvider],
                tap=c[SessionEventTap],
            )

            if ServiceRegistry not in c.defined_types:
                c[ServiceRegistry] = await self._exit_stack.enter_async_context(
                    ServiceDocumentRegistry(
                        database=TransientDocumentDatabase(),
                        event_emitter_factory=c[EventEmitterFactory],
                        logger=c[Logger],
                        tracer=c[Tracer],
                        allow_migration=False,
                    )
                )

            latest_container = await self._configure_container(latest_container.clone())

            if self._user_configure_container:
                latest_container = await self._user_configure_container(latest_container.clone())

            await override_stores_with_transient_versions(get_latest_container)

            if env_based_module := get_env_based_module():
                if module_func := getattr(env_based_module, "configure_container", None):
                    latest_container = await module_func(latest_container.clone())

            if _should_configure_parlant_dashboard():
                from parlant.adapters.modules.parlant_dashboard import (
                    configure_container as dashboard_configure,
                )

                latest_container = await dashboard_configure(latest_container)

            latest_container = self._apply_authorization_policy(latest_container)

            return latest_container

        async def initialize(c: Container) -> None:
            # The server hands this hook the FINAL container — after any
            # diagnostic-report or module cloning and after full initialization.
            # Rebind so every get_latest_container() closure resolves
            # against reality from here on.
            nonlocal latest_container
            latest_container = c

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
                context_vars={
                    self._current_server_var: self,
                },
                embedder=await c[ModelRouter].get_embedder(
                    context=None,
                    route="embedding.tools",
                ),
                embedding_cache_provider=lambda: c[EmbeddingCache],
            )

            await c[ServiceRegistry].update_tool_service(
                name=INTEGRATED_TOOL_SERVICE_NAME,
                kind="sdk",
                url=f"http://{host}:{port}",
                transient=True,
            )

            await self._exit_stack.enter_async_context(self._plugin_server)
            self._exit_stack.push_async_callback(self._plugin_server.shutdown)

            self._cached_indexer = _CachedIndexer(
                db=JSONFileDocumentDatabase(c[Logger], PARLANT_HOME_DIR / "indexing_cache.json"),
                container=c,
            )
            await self._exit_stack.enter_async_context(self._cached_indexer)

            if self._initialize:
                await self._initialize(c)

            if env_based_module := get_env_based_module():
                if module_func := getattr(env_based_module, "initialize_container", None):
                    await module_func(c.clone())

            if _should_configure_parlant_dashboard():
                from parlant.adapters.modules.parlant_dashboard import (
                    initialize_container as dashboard_initialize,
                )

                await dashboard_initialize(c)

        async def configure_api(app: FastAPI) -> FastAPI:
            if self._configure_api:
                await self._configure_api(app)

            if env_based_module := get_env_based_module():
                if module_func := getattr(env_based_module, "configure_api", None):
                    if new_app := await module_func(app):
                        app = new_app

            return app

        return StartupParameters(
            host=self.host,
            port=self.port,
            models=self._models,
            log_level=self.log_level,
            modules=self.modules,
            migrate=self._migrate,
            diagnostic_reports=self.diagnostic_reports,
            configure=configure,
            initialize=initialize,
            configure_api=configure_api,
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
    "AdminPrincipal",
    "Agent",
    "Moderation",
    "AgentId",
    "AllOf",
    "AnonymousPrincipal",
    "AnyOf",
    "AuthConfig",
    "EngineHealthThresholds",
    "EventLoopHealthThresholds",
    "HealthThresholds",
    "NLPHealthThresholds",
    "RouteThresholds",
    "default_nlp_route_thresholds",
    "AuthenticationError",
    "Authenticator",
    "AuthorizationException",
    "AuthorizationPolicy",
    "Authorizer",
    "BasicRateLimiter",
    "CannedResponseId",
    "CompassOptimizationPolicy",
    "CompositeAuthorizationPolicy",
    "Container",
    "ContextVariableId",
    "ContextVariableStore",
    "ControlOptions",
    "Weight",
    "Customer",
    "CustomerGroups",
    "CustomerKind",
    "CustomerMetadata",
    "CustomerId",
    "CustomerModerationContext",
    "CustomerPrincipal",
    "CustomerStore",
    "Environment",
    "resolve_environment",
    "DefaultBaseModel",
    "DefaultPermissionGate",
    "DeferredRetriever",
    "DevelopmentAuthorizationPolicy",
    "HeaderApiKeyAuthorizationPolicy",
    "GuestPrincipal",
    "Embedder",
    "EmbedderHints",
    "EmbeddingResult",
    "EmittedEvent",
    "EngineContext",
    "CompassHook",
    "CompassHookPayload",
    "CompassHookResult",
    "CompassHooks",
    "EstimatingTokenizer",
    "Event",
    "EventKind",
    "EventSource",
    "FallbackSchematicGenerator",
    "Rule",
    "RuleId",
    "Interaction",
    "InteractionMessage",
    "InvalidModelArgumentsError",
    "InvalidModelProviderResultError",
    "InvalidModelRouteError",
    "JSONSerializable",
    "Lifespan",
    "Effort",
    "LogLevel",
    "AttachmentRef",
    "Logger",
    "MessageEventData",
    "MissingModelRouteError",
    "ModelConfigurationError",
    "ModelConfig",
    "ModelGeneration",
    "ReasoningConfig",
    "ReasoningEffort",
    "ModelImplementationError",
    "ModelPricing",
    "ModelProvider",
    "ModelProviderExecutionError",
    "ModelProviderUnavailableError",
    "ModelSize",
    "ModelType",
    "RateLimit",
    "RateLimitConflictError",
    "RateLimitError",
    "TransientRateLimiter",
    "ModerationCheck",
    "ModerationService",
    "ModerationTag",
    "NoModeration",
    "Operation",
    "OwnershipAuthorizer",
    "PermissionGate",
    "Principal",
    "OutputMode",
    "Observation",
    "PluginServer",
    "Policy",
    "Pricing",
    "PreambleConfiguration",
    "PreambleConfigurationProvider",
    "PreambleDecision",
    "PreambleEmissionPolicy",
    "PromptBuilder",
    "PromptProvider",
    "PromptSections",
    "SectionAdaptation",
    "VETO",
    "PromptRoutes",
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
    "RuleEvaluation",
    "RuleEvaluationContext",
    "RuleHandler",
    "SchematicGenerationResult",
    "SchematicGenerator",
    "SchematicSchemaMismatchError",
    "SecretLoader",
    "MappingSecretLoader",
    "MATCH_ALWAYS",
    "SecretRef",
    "SecretValue",
    "Server",
    "ServiceRegistry",
    "Session",
    "SessionBudget",
    "SessionId",
    "SessionLabels",
    "SessionMetadata",
    "SessionUsage",
    "SessionMode",
    "SessionStatus",
    "SessionStore",
    "Singleton",
    "StatusEventData",
    "T",
    "Group",
    "GroupId",
    "Term",
    "TermId",
    "Narration",
    "NarrationFn",
    "TokenAuthenticator",
    "Tool",
    "ToolContext",
    "ToolEntry",
    "ToolEventData",
    "ToolRef",
    "TransientRule",
    "ToolId",
    "ToolParameterDescriptor",
    "ToolOverlap",
    "ToolParameterOptions",
    "ToolParameterType",
    "ToolResult",
    "Tracer",
    "UnknownModelProviderError",
    "UnsupportedModelCapabilityError",
    "UnsupportedRateLimitCapabilityError",
    "Variable",
    "tool",
]
