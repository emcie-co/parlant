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

# mypy: disable-error-code=import-untyped

import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from dataclasses import dataclass, field
import importlib
import inspect
import os
import socket
import traceback
import warnings
from fastapi import FastAPI
from parlant.core.container import Container, Singleton
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TextIO,
)
import rich
import toml
from typing_extensions import NoReturn
from pathlib import Path
import sys
import uvicorn


from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.adapters.vector_db.transient import TransientVectorDatabase
from parlant.api.authorization import (
    AuthConfig,
    AuthorizationPolicy,
    DevelopmentAuthorizationPolicy,
    create_composite_authorization_policy,
)

from parlant.core.app_modules.agents import validate_moderation_configuration
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.application_context import ApplicationContext
from parlant.core.environment import Environment, resolve_environment
from parlant.core.common import IdGenerator, generate_id
from parlant.core.engines.compass.hooks import CompassHooks
from parlant.core.engines.rule_evaluator_registry import RuleEvaluatorRegistry
from parlant.core.event_loop_monitor import EventLoopMonitor
from parlant.core.health import HealthReporter
from parlant.core.health.thresholds import HealthThresholds
from parlant.core.meter import Meter, LocalMeter
from parlant.core.persistence.vector_database import VectorDatabase
from parlant.core.services.indexing.rule_relativity_proposer import (
    RuleRelativityProposer,
)
from parlant.core.services.indexing.rule_signal_proposer import (
    RuleSignalProposer,
)
from parlant.core.services.indexing.rule_title_proposer import (
    RuleTitleProposer,
)
from parlant.core.canned_responses import CannedResponseStore, CannedResponseVectorStore
from parlant.core.nlp.model_route_store import ModelRouteDocumentStore
from parlant.core.nlp.prompt_routing import PromptConfigurationService
from parlant.core.nlp.null_provider import load_null_models
from parlant.core.nlp.model_routing import (
    MissingProviderCredentialsError,
    ModelConfigurationService,
    ModelImplementationRegistry,
    ModelProviderUnavailableError,
    ModelRoutes,
    ModelRouteStore,
    ModelRouter,
    ModelScope,
    RuntimeModelRouteSource,
    NullSecretLoader,
    SecretLoader,
)
from parlant.core.nlp.rate_limits import RateLimiter, TransientRateLimiter
from parlant.core.persistence.common import MigrationRequired, ServerOutdated
from parlant.core.groups import GroupDocumentStore, GroupStore
from parlant.api.app import create_api_app, ASGIApplication
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.tracer import LocalTracer, Tracer
from parlant.core.diagnostic_reports import (
    DiagnosticReportRepository,
    DiagnosticReportService,
    DiagnosticStore,
    DiagnosticTraceHistory,
    NOOP_DIAGNOSTIC_REPORT_SERVICE,
    RecordingTracer,
    TraceLogBuffer,
    resolve_error_history_size,
    resolve_trace_history_size,
)
from parlant.core.cost_control import CompositeCostControlPolicy, CostControlPolicy
from parlant.core.usage_reporter import UsageReporter
from parlant.core.pricing import PricingRegistry, SessionBudgetService
from parlant.core.pricing_report import PricingReportService
from parlant.core.runtime_paths import resolve_log_rotation_config, resolve_runtime_paths
from parlant.core.agents import AgentDocumentStore, AgentStore
from parlant.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from parlant.core.emission.event_publisher import EventPublisherFactory
from parlant.core.emission.session_tap import SessionEventTap
from parlant.core.emissions import EventEmitterFactory
from parlant.core.customers import CustomerDocumentStore, CustomerStore
from parlant.core.indexing import (
    IndexingDocumentStore,
    IndexingListener,
    IndexingStatus,
    IndexingStore,
    PollingIndexingListener,
)
from parlant.core.entity_cq import EntityQueries, EntityCommands
from parlant.core.relationships import (
    RelationshipDocumentStore,
    RelationshipStore,
)
from parlant.core.rules import (
    RuleVectorStore,
    RuleStore,
)
from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.db.sqlite import SQLiteDocumentDatabase
from parlant.core.nlp.embedding import (
    BasicEmbeddingCache,
    Embedder,
    EmbedderProvider,
    EmbeddingCache,
    NullEmbeddingCache,
)
from parlant.core.services.tools.service_registry import (
    ServiceRegistry,
    ServiceDocumentRegistry,
)
from parlant.core.sessions import (
    PollingSessionListener,
    SessionDocumentStore,
    SessionListener,
    SessionStore,
)
from parlant.core.glossary import GlossaryStore, GlossaryVectorStore
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.compass.context import CompassDependencies
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.engines.compass.preambles import PreambleConfigurationProvider
from parlant.core.engines.compass.matcher import Matcher as CompassMatcher
from parlant.core.engines.compass.matching.rule_evaluation import TurnEvaluator
from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.engines.compass.matching.evaluation_scheduler import (
    CrossReferenceEvaluationPreconditionProvider,
    DependencyEvaluationPreconditionProvider,
    EvaluationPreconditionRegistry,
    EvaluationScheduler,
    EvaluationSchedulingPolicy,
)
from parlant.core.engines.compass.responder.responder import Responder as CompassResponder
from parlant.core.engines.compass.relational_resolver import (
    RelationalResolver as CompassRelationalResolver,
)
from parlant.core.engines.compass.matching.rule_discovery import RuleDiscoverer
from parlant.core.engines.compass.matching.rule_recaller import RuleRecaller
from parlant.core.engines.compass.matching.tool_selection import ToolSelector
from parlant.core.engines.compass.matching.tool_recaller import ToolRecaller
from parlant.core.engines.compass.reviewer import (
    BasicReviewer,
    Reviewer as CompassReviewer,
)
from parlant.core.engines.compass.variable_loader import VariableLoader
from parlant.core.rule_tool_associations import (
    RuleToolAssociationDocumentStore,
    RuleToolAssociationStore,
)


from parlant.core.engines.types import EngineRegistry
from parlant.core.services.indexing.indexing_service import IndexingService
from parlant.core.services.training_service import TrainingService
from parlant.core.loggers import CompositeLogger, FileLogger, LogLevel, Logger
from parlant.core.application import Application
from parlant.core.store_provider import BasicStoreProvider, StoreProvider, StoreProviderHints
from parlant.core.version import VERSION


# Suppress known deprecation warnings from websockets library used by dependencies
# This must be done early to catch warnings from FastAPI/uvicorn websocket handling
def _suppress_websocket_deprecation_warnings(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Optional[TextIO] = None,
    line: Optional[str] = None,
) -> None:
    if category is DeprecationWarning:
        msg_str = str(message)
        if any(
            keyword in msg_str
            for keyword in ["ws_handler", "websockets", "WebSocketServerProtocol"]
        ):
            return  # Suppress websocket-related deprecation warnings
    # Let all other warnings through
    original_showwarning = getattr(warnings, "_default_showwarning", warnings.showwarning)
    original_showwarning(message, category, filename, lineno, file, line)


if not hasattr(warnings, "_default_showwarning"):
    warnings._default_showwarning = warnings.showwarning  # type: ignore[attr-defined]
warnings.showwarning = _suppress_websocket_deprecation_warnings


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8800
SERVER_ADDRESS = "https://localhost"
CONFIG_FILE_PATH = Path("parlant.toml")

DEFAULT_HOME_DIR = "parlant-data"
RUNTIME_PATHS = resolve_runtime_paths(DEFAULT_HOME_DIR)
LOG_ROTATION_CONFIG = resolve_log_rotation_config()
PARLANT_HOME_DIR = RUNTIME_PATHS.home_dir
PARLANT_HOME_DIR.mkdir(parents=True, exist_ok=True)

EXIT_STACK: AsyncExitStack

DEFAULT_AGENT_NAME = "Default Agent"

sys.path.append(PARLANT_HOME_DIR.as_posix())
sys.path.append(".")

TRACER = LocalTracer()
LOGGER = FileLogger(
    RUNTIME_PATHS.log_file,
    TRACER,
    LogLevel.INFO,
    max_bytes=LOG_ROTATION_CONFIG.max_bytes,
    backup_count=LOG_ROTATION_CONFIG.backup_count,
)


class StartupError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class StartupParameters:
    host: str
    port: int
    models: ModelRoutes
    log_level: str | LogLevel
    modules: list[str]
    migrate: bool
    diagnostic_reports: bool = True
    configure: Callable[[Container], Awaitable[Container]] | None = None
    initialize: Callable[[Container], Awaitable[None]] | None = None
    configure_api: Callable[[FastAPI], Awaitable[FastAPI | None]] | None = None
    contextvar_propagation: Mapping[ContextVar[Any], Any] = field(default_factory=dict)


def _load_model_implementations(
    registry: ModelImplementationRegistry,
    container: Container,
    *,
    provider: str,
    display_name: str,
    module_path: str,
    registrar_name: str,
    installation_command: str,
) -> None:
    try:
        module = importlib.import_module(module_path)
        registrar = getattr(module, registrar_name)
        registrar(registry, container)
    except ModuleNotFoundError as exc:
        raise ModelProviderUnavailableError(
            f"{display_name} model support is not installed. "
            f"Install it with: {installation_command}",
            provider=provider,
        ) from exc


def load_openai_models(
    registry: ModelImplementationRegistry,
    container: Container,
) -> None:
    _load_model_implementations(
        registry,
        container,
        provider="openai",
        display_name="OpenAI",
        module_path="parlant.adapters.nlp.openai_service",
        registrar_name="register_openai_models",
        installation_command="pip install parlant",
    )


def load_gemini_models(
    registry: ModelImplementationRegistry,
    container: Container,
) -> None:
    _load_model_implementations(
        registry,
        container,
        provider="gemini",
        display_name="Gemini",
        module_path="parlant.adapters.nlp.gemini_service",
        registrar_name="register_gemini_models",
        installation_command="pip install 'parlant[gemini]'",
    )


def load_anthropic_models(
    registry: ModelImplementationRegistry,
    container: Container,
) -> None:
    _load_model_implementations(
        registry,
        container,
        provider="anthropic",
        display_name="Anthropic",
        module_path="parlant.adapters.nlp.anthropic_service",
        registrar_name="register_anthropic_models",
        installation_command="pip install 'parlant[anthropic]'",
    )


def load_openrouter_models(
    registry: ModelImplementationRegistry,
    container: Container,
) -> None:
    _load_model_implementations(
        registry,
        container,
        provider="openrouter",
        display_name="OpenRouter",
        module_path="parlant.adapters.nlp.openrouter_service",
        registrar_name="register_openrouter_models",
        installation_command="pip install parlant",
    )


def load_litellm_models(
    registry: ModelImplementationRegistry,
    container: Container,
) -> None:
    _load_model_implementations(
        registry,
        container,
        provider="litellm",
        display_name="LiteLLM",
        module_path="parlant.adapters.nlp.litellm_service",
        registrar_name="register_litellm_models",
        installation_command="pip install 'parlant[litellm]'",
    )


def register_model_implementation_loaders(
    registry: ModelImplementationRegistry,
    container: Container,
) -> None:
    registry.register_provider_loader(
        "openai",
        lambda: load_openai_models(registry, container),
    )
    registry.register_provider_loader(
        "gemini",
        lambda: load_gemini_models(registry, container),
    )
    registry.register_provider_loader(
        "anthropic",
        lambda: load_anthropic_models(registry, container),
    )
    registry.register_provider_loader(
        "openrouter",
        lambda: load_openrouter_models(registry, container),
    )
    registry.register_provider_loader(
        "litellm",
        lambda: load_litellm_models(registry, container),
    )
    registry.register_provider_loader(
        "null",
        lambda: load_null_models(registry),
    )


async def configure_model_routing(
    container: Container,
    model_routes: ModelRoutes,
) -> None:
    if SecretLoader not in container.defined_types:
        container[SecretLoader] = NullSecretLoader()
    if RateLimiter not in container.defined_types:
        container[RateLimiter] = TransientRateLimiter()

    if ModelRouter not in container.defined_types:
        runtime_routes = RuntimeModelRouteSource()
        model_implementations = ModelImplementationRegistry(container[SecretLoader])
        register_model_implementation_loaders(model_implementations, container)
        model_configuration = ModelConfigurationService(
            container[ModelRouteStore],
            runtime_routes,
            model_implementations,
        )
        container[RuntimeModelRouteSource] = runtime_routes
        container[ModelImplementationRegistry] = model_implementations
        container[ModelConfigurationService] = model_configuration
        container[PromptConfigurationService] = PromptConfigurationService()
        container[ModelRouter] = ModelRouter(
            container[ModelRouteStore],
            runtime_routes,
            model_implementations,
            container[RateLimiter],
            health_reporter=container[HealthReporter],
        )

    if PricingReportService not in container.defined_types:
        pricing_report = PricingReportService(
            container[ModelConfigurationService],
            container[ModelImplementationRegistry],
            container[PricingRegistry],
            container[AgentStore],
            container[Logger],
        )
        container[PricingReportService] = pricing_report
        container[HealthReporter].register_admin_detail("pricing", pricing_report.read)

    if model_routes:
        await container[ModelConfigurationService].configure(
            ModelScope.server(),
            model_routes,
        )


async def create_agent_if_absent(agent_store: AgentStore) -> None:
    agents = await agent_store.list_agents()
    if not agents:
        await agent_store.create_agent(name=DEFAULT_AGENT_NAME)


async def get_module_list_from_config(file_path: Path) -> list[str]:
    if file_path.exists():
        config = toml.load(file_path)
        # Expecting the following toml structure:
        #
        # [parlant]
        # modules = ["module_1", "module_2"]
        return list(config.get("parlant", {}).get("modules", []))

    return []


@asynccontextmanager
async def load_modules(
    container: Container,
    modules: Iterable[str],
) -> AsyncGenerator[
    tuple[Container, Sequence[tuple[str, Callable[[Container], Awaitable[None]]]]], None
]:
    imported_modules = []
    initializers: list[tuple[str, Callable[[Container], Awaitable[None]]]] = []

    for module_path in modules:
        module = importlib.import_module(module_path)
        imported_modules.append(module)

        if configure_module := getattr(module, "configure_module", None):
            LOGGER.info(f"Configuring module '{module.__name__}'")
            if new_container := await configure_module(container.clone()):
                container = new_container

        if initialize_module := getattr(module, "initialize_module", None):
            initializers.append((module.__name__, initialize_module))

    try:
        yield container, initializers
    finally:
        for m in reversed(imported_modules):
            if shutdown_module := getattr(module, "shutdown_module", None):
                LOGGER.info(f"Shutting down module '{m.__name__}'")
                await shutdown_module()


async def _define_logger(container: Container) -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"):
        from parlant.adapters.loggers.opentelemetry import OpenTelemetryLogger

        print("OpenTelemetry logging is enabled.")
        container[Logger] = CompositeLogger(
            [
                await EXIT_STACK.enter_async_context(OpenTelemetryLogger(container[Tracer])),
                container[WebSocketLogger],
            ]
        )

    else:
        container[Logger] = CompositeLogger([LOGGER, container[WebSocketLogger]])


async def _define_tracer(container: Container) -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        from parlant.adapters.tracing.opentelemetry import OpenTelemetryTracer

        print("OpenTelemetry tracing is enabled.")
        container[Tracer] = await EXIT_STACK.enter_async_context(OpenTelemetryTracer())

    else:
        # Reuse the module tracer that LOGGER (the stdout/file sink) was built
        # around: spans open on the container tracer, and with two separate
        # LocalTracer instances the terminal and log file would never carry a
        # trace id.
        container[Tracer] = TRACER


async def _define_meter(container: Container) -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
        from parlant.adapters.meter.opentelemetry import OpenTelemetryMeter

        print("OpenTelemetry metrics is enabled.")
        container[Meter] = await EXIT_STACK.enter_async_context(OpenTelemetryMeter())

    else:
        _define_singleton(container, Meter, LocalMeter)


def _configure_diagnostic_reports(
    container: Container,
    enabled: bool,
) -> Container:
    if not enabled:
        _define_singleton_value(
            container,
            DiagnosticReportService,
            NOOP_DIAGNOSTIC_REPORT_SERVICE,
        )
        return container

    try:
        trace_history_size = resolve_trace_history_size()
        error_history_size = resolve_error_history_size()
    except ValueError as exc:
        raise StartupError(str(exc)) from exc

    container = container.clone()

    if DiagnosticReportRepository in container.defined_types:
        repository = container[DiagnosticReportRepository]
    else:
        repository = DiagnosticReportRepository(
            error_report_dir=RUNTIME_PATHS.error_reports_dir,
            trace_report_dir=RUNTIME_PATHS.trace_reports_dir,
            error_history_size=error_history_size,
            logger=container[Logger],
        )
        _define_singleton_value(container, DiagnosticReportRepository, repository)
    repository.prune()

    if DiagnosticTraceHistory in container.defined_types:
        history = container[DiagnosticTraceHistory]
    else:
        history = DiagnosticTraceHistory(trace_history_size)
        _define_singleton_value(container, DiagnosticTraceHistory, history)

    if history.capacity == 0 and repository.error_history_size == 0:
        _define_singleton_value(
            container,
            DiagnosticReportService,
            NOOP_DIAGNOSTIC_REPORT_SERVICE,
        )
        return container

    store = DiagnosticStore()
    service = DiagnosticReportService(
        store=store,
        repository=repository,
        history=history,
        logger=container[Logger],
    )
    tracer = RecordingTracer(container[Tracer], service)
    trace_log_buffer = TraceLogBuffer(tracer=tracer, store=store)

    _define_singleton_value(container, DiagnosticStore, store)
    _define_singleton_value(container, DiagnosticReportService, service)
    _define_singleton_value(container, Tracer, tracer)

    logger = container[Logger]
    assert isinstance(logger, CompositeLogger)  # base setup always builds one
    logger.append(trace_log_buffer)

    # The service's own diagnostics must not loop back into the trace buffer
    # that feeds its store: log through a view of the composite without it.
    service._logger = logger.excluding(trace_log_buffer)
    repository._logger = service._logger

    return container


def _define_singleton(container: Container, interface: type, implementation: type) -> None:
    try:
        container[implementation] = Singleton(implementation)

        if interface != implementation:
            container[interface] = lambda c: c[implementation]
    except BaseException:
        rich.print(
            rich.text.Text(
                f"Error adding {implementation} as implementation for {interface}",
                style="bold red",
            )
        )
        raise


def _load_auth_config_from_env() -> AuthConfig:
    admin_api_key = os.environ.get("PARLANT_ADMIN_API_KEY")

    if not admin_api_key:
        raise StartupError(
            "PARLANT_ENV=production requires PARLANT_ADMIN_API_KEY to be set. "
            "Optionally also set PARLANT_CUSTOMER_JWT_SECRET or PARLANT_CUSTOMER_JWKS_URL "
            "(to enable customer authentication), PARLANT_GUEST_TOKEN_SECRET (to keep guest "
            "tokens valid across restarts), and PARLANT_AUTH_AUTO_PROVISION_CUSTOMERS."
        )

    customer_jwt_secret = os.environ.get("PARLANT_CUSTOMER_JWT_SECRET")
    customer_jwks_url = os.environ.get("PARLANT_CUSTOMER_JWKS_URL")

    if customer_jwt_secret and customer_jwks_url:
        raise StartupError(
            "PARLANT_CUSTOMER_JWT_SECRET and PARLANT_CUSTOMER_JWKS_URL are mutually "
            "exclusive; set exactly one customer token verification method."
        )

    return AuthConfig(
        admin_api_key=admin_api_key,
        customer_jwt_secret=customer_jwt_secret,
        customer_jwks_url=customer_jwks_url,
        customer_jwt_audience=os.environ.get("PARLANT_CUSTOMER_JWT_AUDIENCE"),
        customer_jwt_issuer=os.environ.get("PARLANT_CUSTOMER_JWT_ISSUER"),
        require_customer_jwt_expiry=os.environ.get(
            "PARLANT_CUSTOMER_JWT_REQUIRE_EXPIRY", ""
        ).lower()
        in ("1", "true", "yes"),
        guest_token_secret=os.environ.get("PARLANT_GUEST_TOKEN_SECRET"),
        auto_provision_customers=os.environ.get("PARLANT_AUTH_AUTO_PROVISION_CUSTOMERS", "").lower()
        in ("1", "true", "yes"),
        trusted_proxies=_parse_trusted_proxies(os.environ.get("PARLANT_TRUSTED_PROXIES")),
    )


def _parse_trusted_proxies(value: str | None) -> frozenset[str]:
    """Parse the comma-separated PARLANT_TRUSTED_PROXIES env var into a set of
    peer addresses whose forwarding headers (X-Forwarded-For, etc.) may be
    trusted for rate-limit identity. Empty/unset means trust none."""
    if not value:
        return frozenset()
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def _production_authorization_policy_factory() -> Callable[[Container], AuthorizationPolicy]:
    # Deferred to first resolution so the config is validated once the server
    # actually stands up its API (and so an SDK-provided policy binding can
    # supersede this one without requiring the production env vars).
    policy: AuthorizationPolicy | None = None

    def factory(c: Container) -> AuthorizationPolicy:
        nonlocal policy

        if policy is None:
            config = _load_auth_config_from_env()

            if config.guest_token_secret is None:
                LOGGER.warning(
                    "PARLANT_GUEST_TOKEN_SECRET is not set; generated an ephemeral guest "
                    "token secret. Guest sessions will not survive a server restart."
                )

            policy = create_composite_authorization_policy(
                config=config,
                customer_store=c[CustomerStore],
            )

        return policy

    return factory


def _define_singleton_value(container: Container, interface: type, implementation: Any) -> None:
    implementation_type = getattr(implementation, "__orig_class__", type(implementation))

    try:
        container[implementation_type] = implementation

        if interface != implementation_type:
            container[interface] = lambda c: c[implementation_type]
    except BaseException:
        rich.print(
            rich.text.Text(
                f"Error adding {implementation_type} instance as implementation for {interface}",
                style="bold red",
            )
        )
        raise


@asynccontextmanager
async def setup_container() -> AsyncGenerator[Container, None]:
    c = Container()

    await _define_tracer(c)
    web_socket_logger = WebSocketLogger(c[Tracer], LogLevel.INFO)
    c[WebSocketLogger] = web_socket_logger

    await _define_logger(c)
    await _define_meter(c)
    _define_singleton(c, BackgroundTaskService, BackgroundTaskService)

    _define_singleton(c, IdGenerator, IdGenerator)

    _define_singleton_value(c, CompassHooks, CompassHooks())
    _define_singleton_value(c, RuleEvaluatorRegistry, RuleEvaluatorRegistry())

    _define_singleton_value(c, SessionEventTap, SessionEventTap())
    _define_singleton_value(
        c,
        EventEmitterFactory,
        EventPublisherFactory(
            store_provider_factory=lambda: c[StoreProvider],
            tap=c[SessionEventTap],
        ),
    )

    _define_singleton(c, EntityQueries, EntityQueries)
    _define_singleton(c, EntityCommands, EntityCommands)

    _define_singleton(c, RuleRelativityProposer, RuleRelativityProposer)
    _define_singleton(c, RuleSignalProposer, RuleSignalProposer)
    _define_singleton(c, RuleTitleProposer, RuleTitleProposer)

    _define_singleton(c, IndexingService, IndexingService)
    _define_singleton(c, TrainingService, TrainingService)
    _define_singleton(c, IndexingListener, PollingIndexingListener)

    # Must be a singleton: retrain() (via TrainingService / SDK startup) and the
    # discovery path (via the compass Matcher) have to share the same trained frame,
    # so the port binding aliases the concrete recaller registration.
    _define_singleton(c, RuleDiscoverer, RuleRecaller)
    _define_singleton(c, ToolSelector, ToolRecaller)

    if resolve_environment() is Environment.PRODUCTION:
        c[AuthorizationPolicy] = _production_authorization_policy_factory()
    else:
        _define_singleton(c, AuthorizationPolicy, DevelopmentAuthorizationPolicy)

    _define_singleton(c, VariableLoader, VariableLoader)
    _define_singleton(c, CompassRelationalResolver, CompassRelationalResolver)
    _define_singleton(c, TurnEvaluator, RuleRanker)
    _define_singleton(
        c,
        DependencyEvaluationPreconditionProvider,
        DependencyEvaluationPreconditionProvider,
    )
    # Evaluation scheduling is extensible through providers. Register any new
    # prerequisite source here, then include its instance in this registry;
    # Matcher and EvaluationScheduler remain unaware of concrete provider types.
    _define_singleton(
        c,
        CrossReferenceEvaluationPreconditionProvider,
        CrossReferenceEvaluationPreconditionProvider,
    )
    c[EvaluationPreconditionRegistry] = Singleton(
        lambda container: EvaluationPreconditionRegistry(
            [
                container[DependencyEvaluationPreconditionProvider],
                container[CrossReferenceEvaluationPreconditionProvider],
            ]
        )
    )
    _define_singleton_value(
        c,
        EvaluationSchedulingPolicy,
        EvaluationSchedulingPolicy(),
    )
    _define_singleton(c, CompassOptimizationPolicy, CompassOptimizationPolicy)
    _define_singleton(c, PreambleConfigurationProvider, PreambleConfigurationProvider)
    _define_singleton(c, CompassDependencies, CompassDependencies)
    _define_singleton(c, EvaluationScheduler, EvaluationScheduler)
    _define_singleton(c, CompassMatcher, CompassMatcher)
    _define_singleton(c, CompassReviewer, BasicReviewer)
    _define_singleton(c, CompassResponder, CompassResponder)
    _define_singleton(c, CompassEngine, CompassEngine)

    _define_singleton_value(
        c,
        ApplicationContext,
        ApplicationContext(instance_id=generate_id(), environment=resolve_environment()),
    )
    _define_singleton_value(c, HealthThresholds, HealthThresholds())
    # Resolved lazily, after `configure_container`, so a rebound
    # HealthThresholds reaches the monitor.
    c[EventLoopMonitor] = Singleton(
        lambda container: EventLoopMonitor(thresholds=container[HealthThresholds].event_loop)
    )
    _define_singleton(c, HealthReporter, HealthReporter)
    _define_singleton(c, UsageReporter, UsageReporter)
    _define_singleton(c, PricingRegistry, PricingRegistry)
    _define_singleton(c, SessionBudgetService, SessionBudgetService)
    _define_singleton(c, CompositeCostControlPolicy, CompositeCostControlPolicy)
    c[CostControlPolicy] = lambda container: container[CompositeCostControlPolicy]

    _define_singleton(c, Application, Application)
    _define_singleton(c, RequestContext, RequestContext)

    c[StoreProvider] = BasicStoreProvider(lambda: c)

    yield c


async def initialize_container(
    c: Container,
    model_routes: ModelRoutes,
    log_level: str | LogLevel,
    migrate: bool,
) -> None:
    def try_define(t: type, value: object) -> None:
        if t not in c.defined_types:
            if isinstance(value, type):
                _define_singleton(c, t, value)
            else:
                _define_singleton_value(c, t, value)

    async def try_define_func(
        t: type,
        value_func: Callable[[], Awaitable[object]],
    ) -> None:
        if t not in c.defined_types:
            c[t] = await value_func()

    async def try_define_document_store(
        store_interface: type,
        store_implementation: type,
        filename: str,
    ) -> None:
        if store_interface not in c.defined_types:
            db = await EXIT_STACK.enter_async_context(
                JSONFileDocumentDatabase(
                    c[Logger],
                    PARLANT_HOME_DIR / filename,
                )
            )

            sig = inspect.signature(store_implementation)
            params = list(sig.parameters.keys())

            # Remove 'self' from parameters list
            if "self" in params:
                params.remove("self")

            # Build arguments based on what the constructor accepts
            args: list[Any] = []

            if "id_generator" in params:
                args.append(c[IdGenerator])

            args.extend([db, migrate])

            c[store_implementation] = await EXIT_STACK.enter_async_context(
                store_implementation(*args)
            )
            c[store_interface] = lambda _c: c[store_implementation]

    async def try_define_vector_store(
        store_interface: type,
        store_implementation: type,
        vector_db_factory: Callable[[], Awaitable[VectorDatabase]],
        document_db_filename: str,
        embedder_provider: EmbedderProvider,
    ) -> None:
        if store_interface not in c.defined_types:
            vector_db = await vector_db_factory()
            document_db = await EXIT_STACK.enter_async_context(
                JSONFileDocumentDatabase(
                    c[Logger],
                    PARLANT_HOME_DIR / document_db_filename,
                )
            )
            c[store_implementation] = await EXIT_STACK.enter_async_context(
                store_implementation(
                    id_generator=c[IdGenerator],
                    vector_db=vector_db,
                    document_db=document_db,
                    embedder_provider=embedder_provider,
                )
            )
            c[store_interface] = lambda _c: c[store_implementation]

    c[EngineRegistry] = EngineRegistry(
        {
            "compass": lambda: c[CompassEngine],
        }
    )

    await EXIT_STACK.enter_async_context(c[BackgroundTaskService])

    c[Logger].set_level(
        log_level
        if isinstance(log_level, LogLevel)
        else {
            "info": LogLevel.INFO,
            "debug": LogLevel.DEBUG,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL,
        }[log_level],
    )

    await c[BackgroundTaskService].start(c[WebSocketLogger].start(), tag="websocket-logger")

    try_define(
        SessionListener, PollingSessionListener(store_provider_factory=lambda: c[StoreProvider])
    )

    try:
        for interface, implementation, filename in [
            (AgentStore, AgentDocumentStore, "agents.json"),
            (ContextVariableStore, ContextVariableDocumentStore, "context_variables.json"),
            (CustomerStore, CustomerDocumentStore, "customers.json"),
            (IndexingStore, IndexingDocumentStore, "indexing.json"),
            (GroupStore, GroupDocumentStore, "groups.json"),
            (
                RuleToolAssociationStore,
                RuleToolAssociationDocumentStore,
                "rule_tool_associations.json",
            ),
            (RelationshipStore, RelationshipDocumentStore, "relationships.json"),
            (SessionStore, SessionDocumentStore, "sessions.json"),
            (ModelRouteStore, ModelRouteDocumentStore, "model_routes.json"),
        ]:
            await try_define_document_store(interface, implementation, filename)

        await configure_model_routing(c, model_routes)

        async def make_service_document_registry() -> ServiceRegistry:
            db = await EXIT_STACK.enter_async_context(
                JSONFileDocumentDatabase(
                    c[Logger],
                    PARLANT_HOME_DIR / "services.json",
                )
            )

            return await EXIT_STACK.enter_async_context(
                ServiceDocumentRegistry(
                    database=db,
                    event_emitter_factory=c[EventEmitterFactory],
                    logger=c[Logger],
                    tracer=c[Tracer],
                    allow_migration=migrate,
                )
            )

        await try_define_func(ServiceRegistry, make_service_document_registry)

        if c[CompassOptimizationPolicy].use_embedding_cache():
            c[EmbeddingCache] = BasicEmbeddingCache(
                await EXIT_STACK.enter_async_context(
                    SQLiteDocumentDatabase(
                        c[Logger],
                        PARLANT_HOME_DIR / "embedding_cache.sqlite",
                    )
                )
            )
        else:
            c[EmbeddingCache] = NullEmbeddingCache()

        async def get_transient_vector_db() -> VectorDatabase:
            return TransientVectorDatabase(
                c[Logger],
                c[Tracer],
                lambda: c[EmbeddingCache],
            )

        for store_interface, store_implementation, document_db_filename, route in [
            (GlossaryStore, GlossaryVectorStore, "glossary_tags.json", "embedding.glossary"),
            (
                CannedResponseStore,
                CannedResponseVectorStore,
                "canned_responses.json",
                "embedding.canned-responses",
            ),
            (RuleStore, RuleVectorStore, "rules.json", "embedding.rules"),
        ]:

            async def get_embedder(route: str = route) -> Embedder:
                return await c[ModelRouter].get_embedder(context=None, route=route)

            await try_define_vector_store(
                store_interface,
                store_implementation,
                lambda: get_transient_vector_db(),
                document_db_filename,
                get_embedder,
            )

    except MigrationRequired as e:
        c[Logger].critical(str(e))
        die("Please re-run with `--migrate` to migrate your data to the new version.")
    except ServerOutdated as e:
        c[Logger].critical(str(e))
        die(
            "Your runtime data came from a higher server version and is not supported.\nPlease upgrade to the latest version of Parlant."
        )


async def recover_server_tasks(
    indexing_service: IndexingService,
) -> None:
    hints = StoreProviderHints(call_site="engine")
    indexing_tasks = await indexing_service.list_indexing_tasks(hints=hints)
    for indexing_task in indexing_tasks:
        if indexing_task.status in [IndexingStatus.PENDING, IndexingStatus.RUNNING]:
            LOGGER.info(f"Recovering indexing task: '{indexing_task.id}'")
            await indexing_service.run_indexing_task(indexing_task, hints=hints)


@asynccontextmanager
async def load_app(
    params: StartupParameters,
) -> AsyncGenerator[tuple[ASGIApplication, Container], None]:
    global EXIT_STACK

    EXIT_STACK = AsyncExitStack()

    async with (
        setup_container() as base_container,
        EXIT_STACK,
    ):
        modules = list(
            dict.fromkeys((await get_module_list_from_config(CONFIG_FILE_PATH)) + params.modules)
        )

        if modules:
            # Allow modules to return a different container
            actual_container, module_initializers = await EXIT_STACK.enter_async_context(
                load_modules(base_container, modules),
            )
        else:
            actual_container, module_initializers = base_container, []
            LOGGER.info("No external modules selected")

        if params.configure:
            actual_container = await params.configure(actual_container.clone())

        actual_container = _configure_diagnostic_reports(
            actual_container,
            params.diagnostic_reports,
        )

        await initialize_container(
            actual_container,
            params.models,
            params.log_level,
            params.migrate,
        )

        for module_name, initializer in module_initializers:
            LOGGER.info(f"Initializing module '{module_name}'")
            await initializer(actual_container)

        if params.initialize:
            await params.initialize(actual_container)

        await recover_server_tasks(
            indexing_service=actual_container[IndexingService],
        )

        if not params.configure:
            # Running in non-SDK mode
            await create_agent_if_absent(actual_container[AgentStore])

        # moderation=on agents require their moderation routes at startup;
        # failing here beats silently skipping a required safety layer later.
        for agent in await actual_container[AgentStore].list_agents():
            await validate_moderation_configuration(
                actual_container[ModelRouter],
                agent,
                actual_container[Logger],
            )

        _print_startup_banner()

        yield (
            await create_api_app(
                actual_container,
                params.configure_api,
                params.contextvar_propagation,
            ),
            actual_container,
        )


def _print_startup_banner() -> None:
    ascii_logo = rf"""
                           ..
                        :=++++=-
                      :+***+++**+.
                    .=*****++++*+=:.
                   .=+++*******-
           ..:::::...  .::::=++
       .-+***#####**+=-..=+=:.
     :+######***********. =***=.
    =####**###**********+ .*****-
   =#******###** v{VERSION[:3]} **+ .******-
  :#*******#######****=. =********:
  .*#******#*:---=-::..-*********+
   -##*##***. -----=++*******++**:
    :*###**: =****###**********+:
      -+*#- -****************+-
        .: .*******++++++==-.
          .****+=:.
          =+=:.
         ..
    """.strip("\n")

    ascii_logo = "\n".join([f"  {line}" for line in ascii_logo.splitlines()])
    ascii_logo = f"\n{ascii_logo}\n"

    rich.print(rich.text.Text(ascii_logo, style="bold #0e8766"))


class _HealthMonitoredServer(uvicorn.Server):
    def __init__(
        self,
        config: uvicorn.Config,
        monitor: EventLoopMonitor,
        exit_stack: AsyncExitStack,
        pricing_report: PricingReportService | None = None,
    ) -> None:
        super().__init__(config)
        self._monitor = monitor
        self._exit_stack = exit_stack
        self._pricing_report = pricing_report

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            if self._pricing_report is not None:
                await self._pricing_report.warn_missing()
            # SDK declarations, indexing and ASGI lifespan initialization are
            # complete. Measure serving responsiveness rather than startup work.
            await self._exit_stack.enter_async_context(self._monitor)


async def serve_app(
    container: Container,
    app: ASGIApplication,
    host: str,
    port: int,
) -> None:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="critical",
        timeout_graceful_shutdown=1,
        ws="wsproto",
        root_path=os.environ.get("ROOT_PATH", ""),
    )
    host_txt = "localhost" if host in ["127.0.0.1", "0.0.0.0"] else host

    try:
        LOGGER.info(".-----------------------------------------.")
        LOGGER.info("| Server is ready for some serious action |")
        LOGGER.info("'-----------------------------------------'")
        LOGGER.info(f"Server authorization policy: {container[AuthorizationPolicy].name}")

        if isinstance(container[AuthorizationPolicy], DevelopmentAuthorizationPolicy):
            LOGGER.info(f"Try the Sandbox UI at http://{host_txt}:{port}")
        else:
            LOGGER.info(f"Server address: http://{host_txt}:{port}")

        async with AsyncExitStack() as monitor_stack:
            server = _HealthMonitoredServer(
                config,
                container[EventLoopMonitor],
                monitor_stack,
                container[PricingReportService]
                if PricingReportService in container.defined_types
                else None,
            )
            await server.serve()
        await asyncio.sleep(0)  # Required to trigger the possible cancellation error
    except (KeyboardInterrupt, asyncio.CancelledError):
        await container[BackgroundTaskService].cancel_all(reason="Server shutting down")
    except BaseException as e:
        LOGGER.critical(traceback.format_exc())
        LOGGER.critical(e.__class__.__name__ + ": " + str(e))
        sys.exit(1)


def die(message: str) -> NoReturn:
    rich.print(rich.text.Text(message, style="bold red"), file=sys.stderr)
    sys.exit(1)


def require_env_keys(keys: list[str]) -> None:
    if missing_keys := [k for k in keys if not os.environ.get(k)]:
        die(f"The following environment variables are missing:\n{', '.join(missing_keys)}")


@asynccontextmanager
async def start_parlant(params: StartupParameters) -> AsyncGenerator[Container, None]:
    LOGGER.set_level(
        params.log_level
        if isinstance(params.log_level, LogLevel)
        else {
            "info": LogLevel.INFO,
            "debug": LogLevel.DEBUG,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL,
        }[params.log_level],
    )

    LOGGER.info(f"Server version: {VERSION}")
    LOGGER.info(f"Runtime data: '{PARLANT_HOME_DIR.absolute()}'")
    LOGGER.info(f"Log file: '{RUNTIME_PATHS.log_file.absolute()}'")

    async with AsyncExitStack() as stack:
        try:
            app, container = await stack.enter_async_context(load_app(params))
        except MissingProviderCredentialsError as exc:
            # A missing API key is a setup mistake, not a crash. Name the
            # provider and the variable, and stop: a traceback here would
            # bury the only line that helps.
            die("\n".join([f"{exc.display_name} needs an API key.", "", *exc.suggestions]))

        logger = container[Logger]
        logger.trace(f"Using error report directory '{RUNTIME_PATHS.error_reports_dir.absolute()}'")
        logger.trace(f"Using trace export directory '{RUNTIME_PATHS.trace_reports_dir.absolute()}'")
        if LOG_ROTATION_CONFIG.max_bytes:
            logger.trace(
                "Log rotation is enabled: "
                f"max_bytes={LOG_ROTATION_CONFIG.max_bytes}, "
                f"backup_count={LOG_ROTATION_CONFIG.backup_count}"
            )
        else:
            logger.trace("Log rotation is disabled")

        yield container

        await serve_app(
            container,
            app,
            params.host,
            params.port,
        )
