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

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, cast
from uuid import uuid4

from parlant.core.agents import AgentId, Effort
from parlant.core.common import JSONSerializable
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.engines.engine_context import EngineContext
from parlant.core.async_utils import Stopwatch
from parlant.core.health import (
    NLP_EMBED_KIND,
    NLP_REQUEST_KIND,
    NLP_REQUESTS_COUNTER,
    NLP_TOKENS_COUNTER,
    HealthReporter,
    NLPHealthView,
)
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.decision import DecisionMaker
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    SchematicGenerationResult,
    SchematicGenerator,
    T,
)
from parlant.core.nlp.moderation import ModerationService
from parlant.core.nlp.rate_limits import (
    RateLimit,
    RateLimiter,
    rate_limit_react_generator,
    rate_limit_schematic_generator,
)
from parlant.core.nlp.react import ReactGenerator, ReasoningConfig
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.sessions import SessionId


# Shorthand for ``ModelConfig.reasoning``: just the reasoning depth. Expands to a
# full ``ReasoningConfig`` (default visibility) in ``ModelConfig.__post_init__``.
ReasoningEffort: TypeAlias = Literal["minimal", "low", "medium", "high"]


class ModelConfigurationError(ValueError):
    """Base class for actionable model-routing configuration failures."""

    code = "model_configuration_error"
    http_status = 422

    def __init__(
        self,
        message: str,
        *,
        route: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        scope: str | None = None,
        capability: str | None = None,
        suggestions: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.route = route
        self.provider = provider
        self.model = model
        self.scope = scope
        self.capability = capability
        self.suggestions = suggestions

    def add_context(
        self,
        *,
        route: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        scope: str | None = None,
        capability: str | None = None,
    ) -> ModelConfigurationError:
        self.route = self.route or route
        self.provider = self.provider or provider
        self.model = self.model or model
        self.scope = self.scope or scope
        self.capability = self.capability or capability
        return self

    def diagnostic(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            **({"route": self.route} if self.route else {}),
            **({"provider": self.provider} if self.provider else {}),
            **({"model": self.model} if self.model else {}),
            **({"scope": self.scope} if self.scope else {}),
            **({"capability": self.capability} if self.capability else {}),
            "suggestions": list(self.suggestions),
        }


class MissingProviderCredentialsError(ModelConfigurationError):
    """A model provider was selected without the credentials it needs."""

    code = "missing_provider_credentials"

    def __init__(self, provider: str, env_var: str) -> None:
        super().__init__(
            f"{provider} needs an API key, and none was found.",
            provider=provider.lower(),
            suggestions=(
                f"Set {env_var} in your environment, or review the provider's documentation for how to configure credentials in code.\n",
                "If your key is in a .env file, load it before starting the server.",
            ),
        )

        self.display_name = provider
        self.env_var = env_var


class ModelRoutingError(ModelConfigurationError):
    """Compatibility base for model route resolution errors."""

    code = "model_routing_error"


class InvalidModelRouteError(ModelRoutingError):
    code = "invalid_model_route"


class UnknownModelProviderError(ModelRoutingError):
    code = "unknown_model_provider"


class ModelProviderUnavailableError(ModelRoutingError):
    code = "model_provider_unavailable"


class MissingModelRouteError(ModelRoutingError):
    code = "missing_model_route"


class UnsupportedModelCapabilityError(ModelRoutingError):
    code = "unsupported_model_capability"


class UnsupportedRateLimitCapabilityError(ModelRoutingError):
    code = "unsupported_rate_limit_capability"


class InvalidModelArgumentsError(ModelRoutingError):
    code = "invalid_model_arguments"


class InvalidModelProviderResultError(ModelRoutingError):
    code = "invalid_model_provider_result"


class SchematicSchemaMismatchError(InvalidModelProviderResultError):
    code = "schematic_schema_mismatch"


class ModelProviderExecutionError(ModelRoutingError):
    code = "model_provider_execution_error"
    http_status = 502


class ModelImplementationError(ModelRoutingError):
    code = "model_implementation_error"
    http_status = 502


class ModelCapability(str, Enum):
    SCHEMATIC = "schematic"
    DECISION = "decision"
    REACT = "react"
    EMBEDDING = "embedding"
    MODERATION = "moderation"
    TOKENIZER = "tokenizer"


@dataclass(frozen=True)
class SecretRef:
    """An opaque reference resolved by the application's SecretLoader."""

    key: str = field(repr=False)

    def __repr__(self) -> str:
        return "SecretRef(key='[REDACTED]')"


@dataclass(frozen=True)
class SecretValue:
    """A process-local secret value that must never be persisted."""

    value: str = field(repr=False)


ModelArgument: TypeAlias = (
    JSONSerializable
    | SecretRef
    | SecretValue
    | Mapping[str, "ModelArgument"]
    # Sequence rather than list/tuple: list is invariant, so a literal like
    # [SecretRef("k"), None, 2] infers as list[SecretRef | int | None] and
    # would not type-check against list["ModelArgument"].
    | Sequence["ModelArgument"]
)


RoutedModel: TypeAlias = (
    SchematicGenerator[Any]
    | ReactGenerator
    | Embedder
    | ModerationService
    | EstimatingTokenizer
    | DecisionMaker
)
ModelProvider: TypeAlias = Callable[[EngineContext[Any, Any] | None], Awaitable[RoutedModel]]


@dataclass(frozen=True, repr=False)
class ModelConfig:
    """A model source plus optional constructor arguments and request limits.

    ``reasoning`` controls reasoning depth for ReAct and schematic routes. Like
    the model itself, it follows the resolved route — per agent, per effort
    qualifier. Accepts a full ``ReasoningConfig`` or the depth shorthand
    (``"minimal"``/``"low"``/``"medium"``/``"high"``). Schematic generators use
    only its ``effort``; ``visibility`` remains ReAct-specific. Left ``None``, the
    engine's built-in effort→reasoning default is used, so existing routes are
    unaffected.
    """

    model: str | ModelProvider
    args: Mapping[str, ModelArgument] = field(default_factory=dict)
    rate_limit: RateLimit | None = None
    reasoning: ReasoningConfig | ReasoningEffort | None = None

    def __post_init__(self) -> None:
        if isinstance(self.reasoning, str):
            object.__setattr__(self, "reasoning", ReasoningConfig(effort=self.reasoning))

        if isinstance(self.model, str):
            if self.model == "null":
                # The null provider ignores model names entirely, so the bare
                # provider string is enough to say "deliberately waived".
                object.__setattr__(self, "model", "null:null")

            provider, separator, model = self.model.partition(":")
            if not separator or not provider or not model:
                raise InvalidModelRouteError(
                    f"Model '{self.model}' must be provider-qualified, "
                    "for example 'provider:model'",
                    model=self.model,
                )
        elif callable(self.model):
            if self.args:
                raise InvalidModelArgumentsError(
                    "ModelConfig.args can only be used with a provider-qualified model string"
                )
        else:
            raise InvalidModelRouteError(
                "ModelConfig.model must be a provider-qualified model string or ModelProvider"
            )

        if self.rate_limit is not None and not isinstance(self.rate_limit, RateLimit):
            raise InvalidModelRouteError("ModelConfig.rate_limit must be a RateLimit")

    def __repr__(self) -> str:
        source = (
            self.model
            if isinstance(self.model, str)
            else getattr(self.model, "__qualname__", type(self.model).__name__)
        )
        redacted_args = {name: "[REDACTED]" for name in self.args}
        return (
            f"ModelConfig(model={source!r}, args={redacted_args!r}, rate_limit={self.rate_limit!r})"
        )


ModelRouteValue: TypeAlias = ModelConfig | str
ModelRoutes: TypeAlias = Mapping[str, ModelRouteValue]
ModelRouteUpdates: TypeAlias = Mapping[str, ModelRouteValue | None]
ModelRouteQualifiers: TypeAlias = Mapping[str, str]


@dataclass(frozen=True)
class _ModelRouteSelector:
    route: str
    effort_values: frozenset[Effort] | None = None
    qualifier_clauses: tuple[str, ...] = ()
    # None means "this route's primary capability" — see ModelRouter.
    capability: ModelCapability | None = None

    @property
    def canonical(self) -> str:
        if not self.qualifier_clauses:
            return self.route
        return f"{self.route}[{','.join(self.qualifier_clauses)}]"

    @property
    def qualifier_specificity(self) -> int:
        return len(self.effort_values) if self.effort_values is not None else len(Effort) + 1

    def matches(self, effort: Effort | None) -> bool:
        if self.effort_values is None:
            return True
        return effort is not None and effort in self.effort_values

    def serves(self, capability: ModelCapability, primary: ModelCapability | None) -> bool:
        """Whether this selector may answer a request for ``capability``.

        A selector that names a capability serves only that one. A selector that
        names a route with a primary capability serves that primary alone — never
        a secondary one, because the model it points at was chosen for the primary
        job and would be the wrong implementation for any other.

        The bare provider wildcard (``"*"``, whose route has no primary) is the
        exception: "use this provider for everything" serves whatever capability
        the provider can supply. Decisions are additionally gated on the provider
        having registered a decision default — see ``_find_binding``."""
        if self.capability is not None:
            return self.capability == capability
        return primary is None or primary == capability


class ModelScopeKind(str, Enum):
    SERVER = "server"
    AGENT = "agent"
    SESSION = "session"
    TURN = "turn"


@dataclass(frozen=True)
class ModelScope:
    kind: ModelScopeKind
    id: str | None = None

    @classmethod
    def server(cls) -> ModelScope:
        return cls(ModelScopeKind.SERVER)

    @classmethod
    def agent(cls, agent_id: AgentId | str) -> ModelScope:
        return cls(ModelScopeKind.AGENT, str(agent_id))

    @classmethod
    def session(cls, session_id: SessionId | str) -> ModelScope:
        return cls(ModelScopeKind.SESSION, str(session_id))

    @classmethod
    def turn(cls, turn_id: str) -> ModelScope:
        return cls(ModelScopeKind.TURN, turn_id)

    def __post_init__(self) -> None:
        if self.kind == ModelScopeKind.SERVER and self.id is not None:
            raise InvalidModelRouteError("The server model scope cannot have an ID")
        if self.kind != ModelScopeKind.SERVER and not self.id:
            raise InvalidModelRouteError(f"The {self.kind.value} model scope requires an ID")


@dataclass(frozen=True)
class ModelRouteBinding:
    target: ModelRouteValue
    revision: str = field(default_factory=lambda: uuid4().hex)


class SecretLoader(ABC):
    @abstractmethod
    async def load(self, ref: SecretRef) -> str: ...


class NullSecretLoader(SecretLoader):
    """Rejects secret references until the application configures a loader."""

    async def load(self, ref: SecretRef) -> str:
        raise ModelRoutingError(
            f"Secret reference '{ref.key}' cannot be loaded because no SecretLoader is configured"
        )


class MappingSecretLoader(SecretLoader):
    """Resolves references from an application-owned in-memory mapping."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)

    async def load(self, ref: SecretRef) -> str:
        try:
            return self._values[ref.key]
        except KeyError as exc:
            raise ModelRoutingError(
                f"Secret reference '{ref.key}' is unavailable in the configured SecretLoader"
            ) from exc


class ModelRouteStore(ABC):
    @abstractmethod
    async def read_all_routes(self) -> Mapping[ModelScope, Mapping[str, ModelRouteBinding]]: ...

    @abstractmethod
    async def read_routes(
        self,
        scope: ModelScope,
    ) -> Mapping[str, ModelRouteBinding]: ...

    @abstractmethod
    async def update_routes(
        self,
        scope: ModelScope,
        updates: Mapping[str, ModelRouteBinding | None],
    ) -> None: ...


class TransientModelRouteStore(ModelRouteStore):
    def __init__(self) -> None:
        self._routes: dict[ModelScope, dict[str, ModelRouteBinding]] = {}
        self._lock = asyncio.Lock()

    async def read_all_routes(self) -> Mapping[ModelScope, Mapping[str, ModelRouteBinding]]:
        async with self._lock:
            return {scope: dict(routes) for scope, routes in self._routes.items()}

    async def read_routes(self, scope: ModelScope) -> Mapping[str, ModelRouteBinding]:
        async with self._lock:
            return dict(self._routes.get(scope, {}))

    async def update_routes(
        self,
        scope: ModelScope,
        updates: Mapping[str, ModelRouteBinding | None],
    ) -> None:
        async with self._lock:
            routes = self._routes.setdefault(scope, {})
            for route, binding in updates.items():
                if binding is None:
                    routes.pop(route, None)
                else:
                    routes[route] = binding


class RuntimeModelRouteSource:
    """Process-local route bindings for server configuration and callbacks."""

    def __init__(self) -> None:
        self._routes: dict[ModelScope, dict[str, ModelRouteBinding]] = {}
        self._lock = asyncio.Lock()

    async def read_all_routes(self) -> Mapping[ModelScope, Mapping[str, ModelRouteBinding]]:
        async with self._lock:
            return {scope: dict(routes) for scope, routes in self._routes.items()}

    async def read_routes(self, scope: ModelScope) -> Mapping[str, ModelRouteBinding]:
        async with self._lock:
            return dict(self._routes.get(scope, {}))

    async def update_routes(
        self,
        scope: ModelScope,
        updates: Mapping[str, ModelRouteBinding | None],
    ) -> None:
        async with self._lock:
            routes = self._routes.setdefault(scope, {})
            for route, binding in updates.items():
                if binding is None:
                    routes.pop(route, None)
                else:
                    routes[route] = binding


def validate_model_route(route: str, value: ModelRouteValue | None = None) -> None:
    selector = ModelRouter.parse_route_selector(route)

    if selector.route == "*":
        if value is None:
            return
        if not isinstance(value, str) or ":" in value or not value:
            raise InvalidModelRouteError(
                f"Provider route '{selector.canonical}' must contain one bare provider name",
                route=route,
            )
        return

    if value is None:
        return

    if isinstance(value, str):
        raise InvalidModelRouteError(
            f"Route '{route}' requires ModelConfig; "
            "bare provider names are only valid for route '*'",
            route=route,
        )

    if isinstance(value, ModelConfig):
        return
    else:
        raise InvalidModelRouteError(
            f"Unsupported model target for route '{route}'; wrap provider callables in ModelConfig",
            route=route,
        )


def validate_requested_route(route: str) -> None:
    selector = ModelRouter.parse_route_selector(route)
    if "*" in selector.route or selector.qualifier_clauses:
        raise InvalidModelRouteError(
            f"Consumers must request a concrete unqualified model route, received '{route}'",
            route=route,
        )


def _contains_secret_value(value: ModelArgument) -> bool:
    if isinstance(value, SecretValue):
        return True
    if isinstance(value, Mapping):
        return any(_contains_secret_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_value(cast(ModelArgument, item)) for item in value)
    return False


def _config_contains_secret_value(config: ModelConfig) -> bool:
    return any(_contains_secret_value(value) for value in config.args.values())


class ModelConfigurationService:
    """Coordinates durable routes and process-local runtime bindings."""

    def __init__(
        self,
        route_store: ModelRouteStore,
        runtime_source: RuntimeModelRouteSource,
        implementations: ModelImplementationRegistry,
    ) -> None:
        self._route_store = route_store
        self._runtime_source = runtime_source
        self._implementations = implementations

    async def configure(
        self,
        scope: ModelScope,
        updates: ModelRouteUpdates,
    ) -> None:
        durable_updates: dict[str, ModelRouteBinding | None] = {}
        runtime_updates: dict[str, ModelRouteBinding | None] = {}
        normalized_updates: dict[str, ModelRouteValue | None] = {}

        for route, target in updates.items():
            validate_model_route(route, target)
            selector = ModelRouter.parse_route_selector(route)
            canonical_route = selector.canonical
            if canonical_route in normalized_updates:
                raise InvalidModelRouteError(
                    f"Model route '{route}' duplicates canonical route '{canonical_route}'",
                    route=route,
                )
            normalized_updates[canonical_route] = target

            if selector.route == "*" and isinstance(target, str):
                self._validate_provider(target, canonical_route)
            elif isinstance(target, ModelConfig) and isinstance(target.model, str):
                provider, _, _ = target.model.partition(":")
                self._validate_provider(provider, canonical_route)

        durable_routes, runtime_routes = await asyncio.gather(
            self._route_store.read_routes(scope),
            self._runtime_source.read_routes(scope),
        )
        resulting_routes = {
            *durable_routes.keys(),
            *runtime_routes.keys(),
        }
        for route, target in normalized_updates.items():
            if target is None:
                resulting_routes.discard(route)
            else:
                resulting_routes.add(route)
        ModelRouter.validate_route_selectors(resulting_routes)

        for route, target in normalized_updates.items():
            if target is None:
                durable_updates[route] = None
                runtime_updates[route] = None
                continue

            runtime_only = scope.kind == ModelScopeKind.SERVER or (
                isinstance(target, ModelConfig)
                and (callable(target.model) or _config_contains_secret_value(target))
            )
            if runtime_only:
                runtime_updates[route] = ModelRouteBinding(target)
                if scope.kind != ModelScopeKind.SERVER:
                    durable_updates[route] = None
                continue

            durable_updates[route] = ModelRouteBinding(target)
            runtime_updates[route] = None

        if durable_updates:
            await self._route_store.update_routes(scope, durable_updates)
        if runtime_updates:
            await self._runtime_source.update_routes(scope, runtime_updates)

    async def read_all(self) -> Mapping[ModelScope, Mapping[str, ModelRouteValue]]:
        durable, runtime = await asyncio.gather(
            self._route_store.read_all_routes(), self._runtime_source.read_all_routes()
        )
        return {
            scope: {
                route: binding.target
                for route, binding in {**durable.get(scope, {}), **runtime.get(scope, {})}.items()
            }
            for scope in set(durable) | set(runtime)
        }

    async def read(self, scope: ModelScope) -> Mapping[str, ModelRouteValue]:
        durable, runtime = await asyncio.gather(
            self._route_store.read_routes(scope),
            self._runtime_source.read_routes(scope),
        )
        return {route: binding.target for route, binding in {**durable, **runtime}.items()}

    def _validate_provider(self, provider: str, route: str) -> None:
        if provider not in self._implementations.providers:
            suggestion = _closest(provider, self._implementations.providers)
            suffix = f" Did you mean '{suggestion}'?" if suggestion else ""
            raise UnknownModelProviderError(
                f"Unknown model provider '{provider}'.{suffix}",
                route=route,
                provider=provider,
                suggestions=(suggestion,) if suggestion else (),
            )
        try:
            self._implementations.ensure_provider(provider)
        except ModelConfigurationError as exc:
            raise exc.add_context(route=route, provider=provider)


SchematicFactory: TypeAlias = Callable[
    [str, type[Any], Mapping[str, Any]],
    Awaitable[SchematicGenerator[Any]],
]
ModelFactory: TypeAlias = Callable[
    [str, Mapping[str, Any]],
    Awaitable[Any],
]
DefaultModel: TypeAlias = str | ModelConfig
DefaultModelResolver: TypeAlias = Callable[
    [str, ModelRouteQualifiers],
    DefaultModel | None,
]
ModelImplementationLoader: TypeAlias = Callable[[], None]


class ModelImplementationRegistry:
    """Capability-oriented registry for concrete provider implementations."""

    def __init__(self, secret_loader: SecretLoader) -> None:
        self._secret_loader = secret_loader
        self._schematic_factories: dict[str, SchematicFactory] = {}
        self._factories: dict[tuple[str, ModelCapability], ModelFactory] = {}
        self._provider_loaders: dict[str, ModelImplementationLoader] = {}
        self._loaded_providers: set[str] = set()
        self._loading_providers: set[str] = set()
        self._defaults: dict[
            tuple[str, ModelCapability],
            DefaultModelResolver,
        ] = {}
        self._cache: dict[
            tuple[str, str, str, ModelCapability, type[Any] | None],
            Any,
        ] = {}
        self._lock = asyncio.Lock()

    @property
    def providers(self) -> set[str]:
        return {
            *self._provider_loaders.keys(),
            *self._schematic_factories.keys(),
            *(provider for provider, _ in self._factories.keys()),
        }

    def register_provider_loader(
        self,
        provider: str,
        loader: ModelImplementationLoader,
    ) -> None:
        if provider in self._provider_loaders:
            raise ValueError(
                f"A model implementation loader is already registered for '{provider}'"
            )
        self._provider_loaders[provider] = loader

    def ensure_provider(self, provider: str) -> None:
        if provider in self._loaded_providers or provider not in self._provider_loaders:
            return
        if provider in self._loading_providers:
            raise ModelProviderUnavailableError(
                f"Model provider '{provider}' recursively attempted to load itself",
                provider=provider,
            )

        self._loading_providers.add(provider)
        try:
            self._provider_loaders[provider]()
        except ModelConfigurationError:
            raise
        except Exception as exc:
            raise ModelProviderUnavailableError(
                f"Could not load model provider '{provider}'",
                provider=provider,
            ) from exc
        finally:
            self._loading_providers.remove(provider)

        self._loaded_providers.add(provider)

    def register_schematic(
        self,
        provider: str,
        factory: SchematicFactory,
        *,
        default: DefaultModel | DefaultModelResolver | None = None,
    ) -> None:
        self._schematic_factories[provider] = factory
        self._register_default(provider, ModelCapability.SCHEMATIC, default)

    def register(
        self,
        provider: str,
        capability: ModelCapability,
        factory: ModelFactory,
        *,
        default: DefaultModel | DefaultModelResolver | None = None,
    ) -> None:
        if capability == ModelCapability.SCHEMATIC:
            raise ValueError("Use register_schematic() for schematic generators")
        self._factories[(provider, capability)] = factory
        self._register_default(provider, capability, default)

    def _register_default(
        self,
        provider: str,
        capability: ModelCapability,
        default: DefaultModel | DefaultModelResolver | None,
    ) -> None:
        if isinstance(default, (str, ModelConfig)):
            default_model = default

            def resolve_default(
                route: str,
                qualifiers: ModelRouteQualifiers,
                model: DefaultModel = default_model,
            ) -> DefaultModel:
                return model

            self._defaults[(provider, capability)] = resolve_default
        elif default is not None:
            self._defaults[(provider, capability)] = default

    def has_default(self, provider: str, capability: ModelCapability) -> bool:
        """Whether ``provider`` registered a default model for ``capability``.

        Used to decide whether a bare provider wildcard (``{"*": provider}``) may
        serve a capability it was not asked for by name — e.g. decisions, which a
        provider opts into by registering a default decision model."""
        return (provider, capability) in self._defaults

    def default_config(
        self,
        provider: str,
        capability: ModelCapability,
        route: str,
        qualifiers: ModelRouteQualifiers,
    ) -> ModelConfig:
        try:
            self.ensure_provider(provider)
        except ModelConfigurationError as exc:
            raise exc.add_context(
                route=route,
                provider=provider,
                capability=capability.value,
            )
        try:
            resolver = self._defaults[(provider, capability)]
        except KeyError as exc:
            raise UnsupportedModelCapabilityError(
                f"Provider route '*' selected provider '{provider}' for route '{route}', "
                f"but provider '{provider}' has no default {capability.value} model",
                route=route,
                provider=provider,
                capability=capability.value,
            ) from exc

        if not (default := resolver(route, qualifiers)):
            raise UnsupportedModelCapabilityError(
                f"Provider '{provider}' has no default model for route '{route}'",
                route=route,
                provider=provider,
                capability=capability.value,
            )

        if isinstance(default, ModelConfig):
            if not isinstance(default.model, str):
                raise InvalidModelRouteError(
                    f"Provider '{provider}' default for route '{route}' must resolve to "
                    "a provider-qualified model string",
                    route=route,
                    provider=provider,
                    capability=capability.value,
                )

            default_provider, _, _ = default.model.partition(":")
            if default_provider != provider:
                raise InvalidModelRouteError(
                    f"Provider '{provider}' default for route '{route}' resolved to a model "
                    f"owned by provider '{default_provider}'",
                    route=route,
                    provider=provider,
                    model=default.model,
                    capability=capability.value,
                )
            return default

        return ModelConfig(f"{provider}:{default}")

    async def construct(
        self,
        *,
        config: ModelConfig,
        capability: ModelCapability,
        route: str,
        scope: str,
        schema: type[Any] | None = None,
    ) -> Any:
        assert isinstance(config.model, str)
        provider, _, model = config.model.partition(":")
        try:
            self.ensure_provider(provider)
        except ModelConfigurationError as exc:
            raise exc.add_context(
                route=route,
                provider=provider,
                model=config.model,
                scope=scope,
                capability=capability.value,
            )
        cache_key = (
            provider,
            model,
            _model_arguments_fingerprint(config.args),
            capability,
            schema,
        )

        async with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

            if capability == ModelCapability.SCHEMATIC:
                if schema is None:
                    raise InvalidModelRouteError(
                        f"Schematic route '{route}' requires an output schema",
                        route=route,
                        model=config.model,
                        scope=scope,
                        capability=capability.value,
                    )
                try:
                    schematic_factory = self._schematic_factories[provider]
                except KeyError as exc:
                    raise self._unsupported(
                        config,
                        provider,
                        capability,
                        route,
                        scope,
                    ) from exc
            else:
                try:
                    factory = self._factories[(provider, capability)]
                except KeyError as exc:
                    raise self._unsupported(
                        config,
                        provider,
                        capability,
                        route,
                        scope,
                    ) from exc

            try:
                resolved_args = cast(
                    Mapping[str, Any],
                    {
                        name: await self._resolve_secrets(value)
                        for name, value in config.args.items()
                    },
                )

                if capability == ModelCapability.SCHEMATIC:
                    assert schema is not None
                    instance = await schematic_factory(
                        model,
                        schema,
                        resolved_args,
                    )
                else:
                    instance = await factory(model, resolved_args)
            except ModelConfigurationError as exc:
                raise exc.add_context(
                    route=route,
                    provider=provider,
                    model=config.model,
                    scope=scope,
                    capability=capability.value,
                )
            except Exception as exc:
                raise ModelImplementationError(
                    f"Could not construct model implementation '{config.model}' "
                    f"for route '{route}' in {scope}: {exc}",
                    route=route,
                    provider=provider,
                    model=config.model,
                    scope=scope,
                    capability=capability.value,
                ) from exc

            self._cache[cache_key] = instance
            return instance

    async def _resolve_secrets(self, value: ModelArgument) -> Any:
        if isinstance(value, SecretRef):
            return await self._secret_loader.load(value)
        if isinstance(value, SecretValue):
            return value.value
        if isinstance(value, Mapping):
            return {key: await self._resolve_secrets(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [await self._resolve_secrets(cast(ModelArgument, item)) for item in value]
        return value

    def _unsupported(
        self,
        config: ModelConfig,
        provider: str,
        capability: ModelCapability,
        route: str,
        scope: str,
    ) -> UnsupportedModelCapabilityError:
        assert isinstance(config.model, str)
        return UnsupportedModelCapabilityError(
            f"Model config '{config.model}' cannot satisfy route '{route}': "
            f"provider '{provider}' does not implement {capability.value}",
            route=route,
            provider=provider,
            model=config.model,
            scope=scope,
            capability=capability.value,
        )


class _AttributedSchematicGenerator(SchematicGenerator[T]):
    """Reports NLP health attributed to the resolved (route, agent).

    A featherweight per-resolution view over the shared, cached generator:
    the inner instance (HTTP client, tokenizer) stays shared across routes
    and agents, while this wrapper — created where both the route and the
    calling agent are known — observes each call from the outside and is
    the ONLY place NLP request health is reported.
    """

    def __init__(
        self,
        inner: SchematicGenerator[T],
        *,
        health_reporter: HealthReporter,
        route: str,
        agent_id: AgentId | None,
    ) -> None:
        self._inner = inner
        self._health_reporter = health_reporter
        self._route = route
        self._agent_id = agent_id

    @property
    def schema(self) -> type[T]:
        return self._inner.schema

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def max_tokens(self) -> int:
        return self._inner.max_tokens

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._inner.tokenizer

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        start = Stopwatch.start()
        try:
            result = await self._inner.generate(prompt, hints)
        except Exception as exc:
            self._report(start.elapsed, success=False, error=exc)
            raise
        self._report(start.elapsed, success=True, error=None, usage=result.info.usage)
        return result

    def _report(
        self,
        duration_seconds: float,
        *,
        success: bool,
        error: BaseException | None,
        usage: UsageInfo | None = None,
    ) -> None:
        self._health_reporter.report(
            NLP_REQUEST_KIND,
            {
                NLPHealthView.ATTR_ROUTE: self._route,
                NLPHealthView.ATTR_AGENT: str(self._agent_id) if self._agent_id else None,
                NLPHealthView.ATTR_SCHEMA: self._inner.schema.__name__,
                NLPHealthView.ATTR_MODEL: self._inner.id,
                NLPHealthView.ATTR_SUCCESS: success,
                NLPHealthView.ATTR_LATENCY_MS: duration_seconds * 1000.0,
                NLPHealthView.ATTR_ERROR_CLASS: type(error).__name__ if error else None,
            },
        )
        self._health_reporter.increment_counter(NLP_REQUESTS_COUNTER, 1)
        if usage is not None:
            self._health_reporter.increment_counter(
                NLP_TOKENS_COUNTER, usage.input_tokens + usage.output_tokens
            )


class _AttributedEmbedder(Embedder):
    """The embedding counterpart of ``_AttributedSchematicGenerator``."""

    def __init__(
        self,
        inner: Embedder,
        *,
        health_reporter: HealthReporter,
        route: str,
        agent_id: AgentId | None,
    ) -> None:
        self._inner = inner
        self._health_reporter = health_reporter
        self._route = route
        self._agent_id = agent_id

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def max_tokens(self) -> int:
        return self._inner.max_tokens

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._inner.tokenizer

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        start = Stopwatch.start()
        try:
            result = await self._inner.embed(texts, hints)
        except Exception as exc:
            self._report(start.elapsed, success=False, error=exc)
            raise
        self._report(start.elapsed, success=True, error=None)
        return result

    def _report(
        self,
        duration_seconds: float,
        *,
        success: bool,
        error: BaseException | None,
    ) -> None:
        self._health_reporter.report(
            NLP_EMBED_KIND,
            {
                NLPHealthView.ATTR_ROUTE: self._route,
                NLPHealthView.ATTR_AGENT: str(self._agent_id) if self._agent_id else None,
                NLPHealthView.ATTR_MODEL: self._inner.id,
                NLPHealthView.ATTR_SUCCESS: success,
                NLPHealthView.ATTR_LATENCY_MS: duration_seconds * 1000.0,
                NLPHealthView.ATTR_ERROR_CLASS: type(error).__name__ if error else None,
            },
        )
        self._health_reporter.increment_counter(NLP_REQUESTS_COUNTER, 1)


class ModelRouter:
    """Resolves scoped model routes immediately before each model operation."""

    _SELECTOR_PATTERN = re.compile(r"^(?P<route>[^\[\]]+?)(?:\[(?P<qualifiers>[^\[\]]+)\])?$")
    _QUALIFIER_PATTERN = re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9_-]*)\s*"
        r"(?P<operator><=|>=|!=|=|<|>)\s*"
        r"(?P<value>[A-Za-z0-9_.-]+)$"
    )

    def __init__(
        self,
        route_store: ModelRouteStore,
        runtime_source: RuntimeModelRouteSource,
        implementations: ModelImplementationRegistry,
        rate_limiter: RateLimiter,
        health_reporter: HealthReporter | None = None,
    ) -> None:
        self._route_store = route_store
        self._runtime_source = runtime_source
        self._implementations = implementations
        self._rate_limiter = rate_limiter
        self._health_reporter = health_reporter

    @classmethod
    def parse_route_selector(cls, raw_selector: str) -> _ModelRouteSelector:
        """Parse and canonicalize a configured route selector.

        The selector grammar is intentionally owned by the router because matching
        semantics and supported qualifiers are part of route resolution, not storage.
        """
        if not raw_selector:
            raise InvalidModelRouteError("Model route cannot be empty", route=raw_selector)

        match = cls._SELECTOR_PATTERN.fullmatch(raw_selector.strip())
        if not match:
            raise InvalidModelRouteError(
                f"Invalid model route selector '{raw_selector}'",
                route=raw_selector,
            )

        route = match.group("route").strip()
        if not route:
            raise InvalidModelRouteError("Model route cannot be empty", route=raw_selector)
        if "*" in route and not (route == "*" or (route.endswith(".*") and route.count("*") == 1)):
            raise InvalidModelRouteError(
                f"Invalid model route '{route}'; only terminal namespace wildcards "
                "and the global provider route are supported",
                route=raw_selector,
            )

        raw_qualifiers = match.group("qualifiers")
        if raw_qualifiers is None:
            return _ModelRouteSelector(route=route)

        clauses = [clause.strip() for clause in raw_qualifiers.split(",")]
        if not clauses or any(not clause for clause in clauses):
            raise InvalidModelRouteError(
                f"Invalid model route qualifiers in '{raw_selector}'",
                route=raw_selector,
            )

        effort_values: frozenset[Effort] | None = None
        capability: ModelCapability | None = None
        canonical_clauses: list[str] = []
        for clause in clauses:
            qualifier_match = cls._QUALIFIER_PATTERN.fullmatch(clause)
            if not qualifier_match:
                qualifier = "effort" if clause.startswith("effort") else "model route"
                raise InvalidModelRouteError(
                    f"Invalid {qualifier} qualifier '{clause}'",
                    route=raw_selector,
                )

            name = qualifier_match.group("name")
            operator = qualifier_match.group("operator")
            value = qualifier_match.group("value")
            if name not in ("effort", "capability"):
                suggestion = _closest(name, {"effort", "capability"})
                suffix = f" Did you mean '{suggestion}'?" if suggestion else ""
                raise InvalidModelRouteError(
                    f"Unsupported model route qualifier '{name}'.{suffix}",
                    route=raw_selector,
                    suggestions=(suggestion,) if suggestion else (),
                )

            if name == "capability":
                if operator != "=":
                    raise InvalidModelRouteError(
                        f"A capability qualifier takes '=', not '{operator}'",
                        route=raw_selector,
                    )
                if capability is not None:
                    raise InvalidModelRouteError(
                        f"Model route '{raw_selector}' names more than one capability",
                        route=raw_selector,
                    )
                capability = cls._parse_capability_qualifier(value, raw_selector)
                canonical_clauses.append(f"capability={value}")
                continue

            clause_values = cls._parse_effort_qualifier(operator, value, raw_selector)
            effort_values = (
                clause_values
                if effort_values is None
                else effort_values.intersection(clause_values)
            )
            canonical_clauses.append(f"effort{operator}{value}")

        if effort_values is not None and not effort_values:
            raise InvalidModelRouteError(
                f"Effort qualifiers in '{raw_selector}' do not match any effort level",
                route=raw_selector,
            )

        return _ModelRouteSelector(
            route=route,
            effort_values=effort_values,
            qualifier_clauses=tuple(sorted(canonical_clauses)),
            capability=capability,
        )

    @staticmethod
    def _parse_capability_qualifier(
        raw_value: str,
        route: str,
    ) -> ModelCapability:
        try:
            return ModelCapability(raw_value)
        except ValueError as exc:
            suggestion = _closest(raw_value, {c.value for c in ModelCapability})
            suffix = f" Did you mean '{suggestion}'?" if suggestion else ""
            raise InvalidModelRouteError(
                f"Unknown capability '{raw_value}'.{suffix}",
                route=route,
                suggestions=(suggestion,) if suggestion else (),
            ) from exc

    @staticmethod
    def _parse_effort_qualifier(
        operator: str,
        raw_value: str,
        route: str,
    ) -> frozenset[Effort]:
        try:
            expected = Effort(raw_value)
        except ValueError as exc:
            options = {effort.value for effort in Effort}
            suggestion = _closest(raw_value, options)
            suffix = f" Did you mean '{suggestion}'?" if suggestion else ""
            raise InvalidModelRouteError(
                f"Unknown effort level '{raw_value}'.{suffix}",
                route=route,
                suggestions=(suggestion,) if suggestion else (),
            ) from exc

        comparisons: Mapping[str, Callable[[Effort], bool]] = {
            "=": lambda actual: actual == expected,
            "!=": lambda actual: actual != expected,
            "<": lambda actual: actual < expected,
            "<=": lambda actual: actual <= expected,
            ">": lambda actual: actual > expected,
            ">=": lambda actual: actual >= expected,
        }
        try:
            comparison = comparisons[operator]
        except KeyError as exc:
            raise InvalidModelRouteError(
                f"Invalid effort qualifier operator '{operator}'",
                route=route,
            ) from exc

        matched = frozenset(effort for effort in Effort if comparison(effort))
        if not matched:
            raise InvalidModelRouteError(
                f"Effort qualifier 'effort{operator}{raw_value}' does not match any effort level",
                route=route,
            )
        return matched

    @classmethod
    def validate_route_selectors(cls, raw_selectors: set[str]) -> None:
        selectors = [cls.parse_route_selector(selector) for selector in raw_selectors]
        for index, left in enumerate(selectors):
            if left.effort_values is None:
                continue
            for right in selectors[index + 1 :]:
                if (
                    right.route != left.route
                    or right.effort_values is None
                    or len(right.effort_values) != len(left.effort_values)
                    or not right.effort_values.intersection(left.effort_values)
                ):
                    continue
                raise InvalidModelRouteError(
                    f"Model route selectors '{left.canonical}' and '{right.canonical}' "
                    "are ambiguous because they overlap with equal specificity",
                    route=left.canonical,
                )

    async def get_schematic_generator(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        schema: type[T],
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> SchematicGenerator[T]:
        generator = await self._get_schematic_generator(
            context=context,
            route=route,
            schema=schema,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )
        if self._health_reporter is None:
            return generator
        return _AttributedSchematicGenerator(
            generator,
            health_reporter=self._health_reporter,
            route=route,
            agent_id=self._resolve_agent_id(context, agent_id),
        )

    async def _get_schematic_generator(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        schema: type[T],
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> SchematicGenerator[T]:
        model = await self._resolve(
            context=context,
            route=route,
            capability=ModelCapability.SCHEMATIC,
            schema=schema,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )
        if not isinstance(model, SchematicGenerator):
            raise self._wrong_provider_result(route, "SchematicGenerator", model)
        try:
            if model.schema is not schema:
                raise SchematicSchemaMismatchError(
                    f"ModelProvider for route '{route}' returned a schematic generator "
                    f"for '{model.schema.__name__}', expected '{schema.__name__}'",
                    route=route,
                    capability=ModelCapability.SCHEMATIC.value,
                )
        except AttributeError as exc:
            raise SchematicSchemaMismatchError(
                f"ModelProvider for route '{route}' returned a schematic generator "
                "without a discoverable schema",
                route=route,
                capability=ModelCapability.SCHEMATIC.value,
            ) from exc
        return cast(SchematicGenerator[T], model)

    async def get_react_generator(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> ReactGenerator:
        model = await self._resolve(
            context=context,
            route=route,
            capability=ModelCapability.REACT,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )
        if not isinstance(model, ReactGenerator):
            raise self._wrong_provider_result(route, "ReactGenerator", model)
        return model

    async def get_embedder(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> Embedder:
        model = await self._resolve(
            context=context,
            route=route,
            capability=ModelCapability.EMBEDDING,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )
        if not isinstance(model, Embedder):
            raise self._wrong_provider_result(route, "Embedder", model)
        if self._health_reporter is None:
            return model
        return _AttributedEmbedder(
            model,
            health_reporter=self._health_reporter,
            route=route,
            agent_id=self._resolve_agent_id(context, agent_id),
        )

    async def get_moderation_service(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str = "moderation.customer",
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> ModerationService:
        model = await self._resolve(
            context=context,
            route=route,
            capability=ModelCapability.MODERATION,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )
        if not isinstance(model, ModerationService):
            raise self._wrong_provider_result(route, "ModerationService", model)
        return model

    async def get_decision_maker(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> DecisionMaker | None:
        """The decision model configured for a route, or None when there is none.

        Unlike the other getters, an absent route is not an error here. A decision
        model is an optional accelerator: a caller asks whether one is configured
        and keeps its own path when the answer is no. Every other failure —
        an unknown provider, a provider that cannot serve decisions, a bad
        argument — still raises, because those are misconfigurations rather than
        an absence."""
        try:
            model = await self._resolve(
                context=context,
                route=route,
                capability=ModelCapability.DECISION,
                agent_id=agent_id,
                qualifiers=qualifiers,
            )
        except MissingModelRouteError:
            return None

        if not isinstance(model, DecisionMaker):
            raise self._wrong_provider_result(route, "DecisionMaker", model)
        return model

    async def get_estimating_tokenizer(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str = "tokenizer",
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> EstimatingTokenizer:
        model = await self._resolve(
            context=context,
            route=route,
            capability=ModelCapability.TOKENIZER,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )
        if not isinstance(model, EstimatingTokenizer):
            raise self._wrong_provider_result(route, "EstimatingTokenizer", model)
        return model

    async def _resolve_config(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        capability: ModelCapability,
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> tuple[ModelConfig, str, _ModelRouteSelector, ModelRouteBinding]:
        """Resolve a route+capability+qualifiers to its winning ModelConfig,
        without constructing the implementation. Shared by ``_resolve`` (which
        goes on to build the generator) and ``resolve_react_reasoning`` (which
        only needs the config's reasoning)."""
        validate_requested_route(route)
        self._validate_builtin_capability(route, capability)
        effort = self._parse_requested_effort(qualifiers, route)

        resolved_agent_id = self._resolve_agent_id(context, agent_id)
        resolved_binding = await self._find_binding(resolved_agent_id, route, effort, capability)
        if resolved_binding is None:
            agent_text = f" for agent '{resolved_agent_id}'" if resolved_agent_id else ""
            candidates = ", ".join(self._route_candidates(route))
            raise MissingModelRouteError(
                f"No model route{agent_text} can satisfy '{route}' ({capability.value}). "
                f"Checked: {candidates}, *",
                route=route,
                scope=f"agent:{resolved_agent_id}" if resolved_agent_id else "server",
                capability=capability.value,
                suggestions=tuple([*self._route_candidates(route)[1:], "*"]),
            )

        selected_scope, selected_selector, binding = resolved_binding
        scope = (
            selected_scope.kind.value
            if selected_scope.id is None
            else f"{selected_scope.kind.value}:{selected_scope.id}"
        )
        target = binding.target
        if isinstance(target, str):
            config = self._implementations.default_config(
                target,
                capability,
                route,
                qualifiers,
            )
        else:
            config = target

        return config, scope, selected_selector, binding

    async def resolve_react_reasoning(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> ReasoningConfig | None:
        """The reasoning configured on the resolved ReAct route (per agent, per
        effort qualifier), or ``None`` to fall back to the engine's built-in
        effort→reasoning default."""
        config, _scope, _selector, _binding = await self._resolve_config(
            context=context,
            route=route,
            capability=ModelCapability.REACT,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )
        return config.reasoning if isinstance(config.reasoning, ReasoningConfig) else None

    async def resolve_schematic_reasoning_effort(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> ReasoningEffort | None:
        """The reasoning effort configured on the resolved schematic route, or
        ``None`` to fall back to the engine component's built-in default.

        Schematic generation has no reasoning-visibility channel, so only the
        effort field of a full ``ReasoningConfig`` applies.
        """
        config, _scope, _selector, _binding = await self._resolve_config(
            context=context,
            route=route,
            capability=ModelCapability.SCHEMATIC,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )
        return config.reasoning.effort if isinstance(config.reasoning, ReasoningConfig) else None

    async def _resolve(
        self,
        *,
        context: EngineContext[Any, Any] | None,
        route: str,
        capability: ModelCapability,
        schema: type[Any] | None = None,
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
    ) -> Any:
        config, scope, selected_selector, binding = await self._resolve_config(
            context=context,
            route=route,
            capability=capability,
            agent_id=agent_id,
            qualifiers=qualifiers,
        )

        if callable(config.model):
            provider = config.model
            provider_name = getattr(provider, "__name__", type(provider).__name__)
            try:
                model = await provider(context)
            except ModelConfigurationError as exc:
                raise exc.add_context(
                    route=route,
                    provider=provider_name,
                    scope=scope,
                    capability=capability.value,
                )
            except Exception as exc:
                raise ModelProviderExecutionError(
                    f"Model provider '{provider_name}' failed while resolving route "
                    f"'{route}' in {scope}",
                    route=route,
                    provider=provider_name,
                    scope=scope,
                    capability=capability.value,
                ) from exc
            self._validate_model_provider_result(
                model=model,
                provider_name=provider_name,
                route=route,
                scope=scope,
                capability=capability,
                schema=schema,
            )
        else:
            model = await self._implementations.construct(
                config=config,
                capability=capability,
                route=route,
                scope=scope,
                schema=schema,
            )

        if config.rate_limit is None:
            return model

        bucket = config.rate_limit.bucket or (
            f"model-route:{scope}:{selected_selector.canonical}:{binding.revision}"
        )
        if capability == ModelCapability.SCHEMATIC:
            return rate_limit_schematic_generator(
                cast(SchematicGenerator[Any], model),
                limiter=self._rate_limiter,
                limit=config.rate_limit,
                bucket=bucket,
            )
        if capability == ModelCapability.REACT:
            return rate_limit_react_generator(
                cast(ReactGenerator, model),
                limiter=self._rate_limiter,
                limit=config.rate_limit,
                bucket=bucket,
            )
        raise UnsupportedRateLimitCapabilityError(
            f"Rate limits are supported for schematic and React generators, not {capability.value}",
            route=route,
            scope=scope,
            capability=capability.value,
        )

    async def _find_binding(
        self,
        agent_id: AgentId | None,
        route: str,
        effort: Effort | None,
        capability: ModelCapability,
    ) -> tuple[ModelScope, _ModelRouteSelector, ModelRouteBinding] | None:
        scopes = [ModelScope.agent(agent_id)] if agent_id else []
        scopes.append(ModelScope.server())
        candidates = [*self._route_candidates(route), "*"]

        for scope in scopes:
            runtime_routes, durable_routes = await asyncio.gather(
                self._runtime_source.read_routes(scope),
                self._route_store.read_routes(scope),
            )
            routes = {**durable_routes, **runtime_routes}
            matches: list[tuple[tuple[int, int], _ModelRouteSelector, ModelRouteBinding]] = []
            for raw_selector, binding in routes.items():
                selector = self.parse_route_selector(raw_selector)
                if selector.route not in candidates or not selector.matches(effort):
                    continue
                # Capability is a filter, never a tiebreaker: a decision binding
                # must not out-specify its way into serving a schematic request.
                if not selector.serves(capability, self.primary_capability(selector.route)):
                    continue
                # A bare provider wildcard serves decisions only when that provider
                # registered a decision default; otherwise decisions stay off and the
                # caller (e.g. the ranker) keeps its own schematic path. An explicit
                # [capability=decision] selector, or a specific model, is unaffected.
                if (
                    capability == ModelCapability.DECISION
                    and selector.capability is None
                    and not (
                        isinstance(binding.target, str)
                        and self._implementations.has_default(
                            binding.target, ModelCapability.DECISION
                        )
                    )
                ):
                    continue
                matches.append(
                    (
                        (
                            candidates.index(selector.route),
                            selector.qualifier_specificity,
                        ),
                        selector,
                        binding,
                    )
                )

            if not matches:
                continue

            matches.sort(key=lambda item: item[0])
            best_specificity = matches[0][0]
            best = [match for match in matches if match[0] == best_specificity]
            if len(best) > 1:
                selectors = ", ".join(f"'{match[1].canonical}'" for match in best)
                raise InvalidModelRouteError(
                    f"Model route selectors {selectors} are ambiguous for route '{route}'",
                    route=route,
                    scope=(
                        scope.kind.value if scope.id is None else f"{scope.kind.value}:{scope.id}"
                    ),
                )
            return scope, best[0][1], best[0][2]

        return None

    @classmethod
    def _parse_requested_effort(
        cls,
        qualifiers: ModelRouteQualifiers,
        route: str,
    ) -> Effort | None:
        unsupported = set(qualifiers).difference({"effort"})
        if unsupported:
            name = sorted(unsupported)[0]
            suggestion = _closest(name, {"effort"})
            suffix = f" Did you mean '{suggestion}'?" if suggestion else ""
            raise InvalidModelRouteError(
                f"Unsupported requested model route qualifier '{name}'.{suffix}",
                route=route,
                suggestions=(suggestion,) if suggestion else (),
            )

        if "effort" not in qualifiers:
            return None

        raw_effort = qualifiers["effort"]
        matched = cls._parse_effort_qualifier("=", raw_effort, route)
        return next(iter(matched))

    def _resolve_agent_id(
        self,
        context: EngineContext[Any, Any] | None,
        agent_id: AgentId | None,
    ) -> AgentId | None:
        if context is None:
            return agent_id
        if agent_id is not None and context.agent.id != agent_id:
            raise InvalidModelRouteError(
                f"Model routing agent '{agent_id}' does not match engine context agent "
                f"'{context.agent.id}'",
                scope=f"agent:{agent_id}",
            )
        return context.agent.id

    def _validate_model_provider_result(
        self,
        *,
        model: object,
        provider_name: str,
        route: str,
        scope: str,
        capability: ModelCapability,
        schema: type[Any] | None,
    ) -> None:
        expected_types: Mapping[ModelCapability, type[Any]] = {
            ModelCapability.SCHEMATIC: SchematicGenerator,
            ModelCapability.REACT: ReactGenerator,
            ModelCapability.EMBEDDING: Embedder,
            ModelCapability.MODERATION: ModerationService,
            ModelCapability.TOKENIZER: EstimatingTokenizer,
            ModelCapability.DECISION: DecisionMaker,
        }
        expected = expected_types[capability]
        if not isinstance(model, expected):
            raise InvalidModelProviderResultError(
                f"Model provider '{provider_name}' returned {type(model).__name__} "
                f"for route '{route}'; expected {expected.__name__}",
                route=route,
                provider=provider_name,
                scope=scope,
                capability=capability.value,
            )

        if capability == ModelCapability.SCHEMATIC and schema is not None:
            actual_schema = cast(SchematicGenerator[Any], model).schema
            if actual_schema is not schema:
                raise SchematicSchemaMismatchError(
                    f"Model provider '{provider_name}' returned a schematic generator "
                    f"for '{actual_schema.__name__}' on route '{route}', expected "
                    f"'{schema.__name__}'",
                    route=route,
                    provider=provider_name,
                    scope=scope,
                    capability=capability.value,
                )

    def _route_candidates(self, route: str) -> list[str]:
        parts = route.split(".")
        candidates = [
            route,
            *(f"{'.'.join(parts[:length])}.*" for length in range(len(parts) - 1, 0, -1)),
        ]
        if route.startswith("embedding."):
            candidates.append("embedding")
        return candidates

    @staticmethod
    def primary_capability(route: str) -> ModelCapability | None:
        """The capability a route serves when its selector names none.

        A capability-less selector means this one, which is what keeps every
        existing route configuration meaning exactly what it did before
        capabilities became selectable."""
        if route == "react" or route.startswith("react."):
            return ModelCapability.REACT
        if route.startswith(("matching.", "review.", "indexing.")) or route in {
            "compaction",
            "indexing",
        }:
            return ModelCapability.SCHEMATIC
        if route == "embedding" or route.startswith("embedding."):
            return ModelCapability.EMBEDDING
        if route == "moderation" or route.startswith("moderation."):
            # moderation.customer and moderation.agent resolve independently -
            # they will typically route to different models.
            return ModelCapability.MODERATION
        if route == "tokenizer":
            return ModelCapability.TOKENIZER
        return None

    @staticmethod
    def _permitted_capabilities(route: str) -> frozenset[ModelCapability] | None:
        """Everything a route may serve, beyond its primary capability.

        A judgment a schematic generator can make, a decision model can often make
        instead — cheaper, and in one request for a whole batch. So the matching
        and review routes accept either, and a caller asks for the one it wants."""
        primary = ModelRouter.primary_capability(route)
        if primary is None:
            return None
        if primary == ModelCapability.SCHEMATIC and route.startswith(("matching.", "review.")):
            return frozenset({ModelCapability.SCHEMATIC, ModelCapability.DECISION})
        return frozenset({primary})

    def _validate_builtin_capability(
        self,
        route: str,
        capability: ModelCapability,
    ) -> None:
        permitted = self._permitted_capabilities(route)

        if permitted is not None and capability not in permitted:
            expected = ", ".join(sorted(c.value for c in permitted))
            raise UnsupportedModelCapabilityError(
                f"Route '{route}' requires {expected}, not {capability.value}",
                route=route,
                capability=capability.value,
            )

    def _wrong_provider_result(
        self,
        route: str,
        expected: str,
        actual: object,
    ) -> InvalidModelProviderResultError:
        return InvalidModelProviderResultError(
            f"ModelProvider for route '{route}' returned {type(actual).__name__}; "
            f"expected {expected}",
            route=route,
        )


def _closest(value: str, options: set[str]) -> str | None:
    if not options:
        return None

    def distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[j] + 1,
                        previous[j - 1] + (left_char != right_char),
                    )
                )
            previous = current
        return previous[-1]

    match = min(options, key=lambda option: distance(value, option))
    return match if distance(value, match) <= max(2, len(value) // 3) else None


def _model_arguments_fingerprint(args: Mapping[str, ModelArgument]) -> str:
    def sanitize(value: ModelArgument) -> JSONSerializable:
        if isinstance(value, SecretRef):
            return {
                "$secret_ref_sha256": hashlib.sha256(value.key.encode()).hexdigest(),
            }
        if isinstance(value, SecretValue):
            return {
                "$secret_value_sha256": hashlib.sha256(value.value.encode()).hexdigest(),
            }
        if isinstance(value, Mapping):
            return {
                str(key): sanitize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [sanitize(cast(ModelArgument, item)) for item in value]
        return cast(JSONSerializable, value)

    serialized = json.dumps(
        sanitize(cast(ModelArgument, args)),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
