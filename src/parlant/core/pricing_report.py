"""Live pricing coverage, separate from health and from historic cost records."""

from collections.abc import Mapping
from typing import Any

from parlant.core.agents import AgentStore, Effort
from parlant.core.loggers import Logger
from parlant.core.nlp.model_routing import (
    ModelCapability,
    ModelConfig,
    ModelConfigurationError,
    ModelConfigurationService,
    ModelImplementationRegistry,
    ModelRoutes,
    ModelRouter,
    ModelScope,
    ModelScopeKind,
)
from parlant.core.pricing import PricingRegistry, split_usage_key

# Current built-in billable route families. Explicit application routes are read
# directly, and opaque factories appear only when they report an actual usage key.
# Resolve provider defaults across efforts without constructing/calling models.
_DEFAULT_ROUTES = (
    ("react", ModelCapability.REACT),
    ("matching.rank", ModelCapability.SCHEMATIC),
    ("matching.rank", ModelCapability.DECISION),
    ("matching.distill", ModelCapability.SCHEMATIC),
    ("review.message", ModelCapability.SCHEMATIC),
    ("review.tools", ModelCapability.SCHEMATIC),
    ("compaction", ModelCapability.SCHEMATIC),
    ("indexing.create", ModelCapability.SCHEMATIC),
    ("indexing.retrieve", ModelCapability.SCHEMATIC),
    ("embedding.rules", ModelCapability.EMBEDDING),
    ("embedding.tools", ModelCapability.EMBEDDING),
    ("embedding.tag", ModelCapability.EMBEDDING),
    ("moderation.customer", ModelCapability.MODERATION),
    ("moderation.agent", ModelCapability.MODERATION),
)


class PricingReportService:
    def __init__(
        self,
        configuration: ModelConfigurationService,
        implementations: ModelImplementationRegistry,
        pricing_registry: PricingRegistry,
        agent_store: AgentStore,
        logger: Logger,
        router: ModelRouter,
    ) -> None:
        self._configuration = configuration
        self._implementations = implementations
        self._registry = pricing_registry
        self._agents = agent_store
        self._logger = logger
        self._router = router

    async def read(self) -> Mapping[str, Any]:
        # Bulk reads avoid one request per agent. Exclude stale/deleted agent
        # bindings, plus transient session/turn overrides that aren't configured
        # server routes (their actual usage is included through observations).
        agents = {str(agent.id) for agent in await self._agents.list_agents()}
        routes = {
            scope: bindings
            for scope, bindings in (await self._configuration.read_all()).items()
            if scope.kind == ModelScopeKind.SERVER
            or (scope.kind == ModelScopeKind.AGENT and scope.id in agents)
        }
        configured = self._configured_models(routes)
        observed = self._registry.observed_models
        pricing = self._registry.pricing
        models: list[dict[str, str]] = []
        for key in sorted(configured | set(observed)):
            parts = split_usage_key(key)
            if parts is None or parts[0] == "null":
                continue
            if parts[0] in {"openrouter", "litellm"} or observed.get(key, False):
                status = "provider_reported"
            elif pricing is not None and pricing.covers_model(key):
                status = "table"
            else:
                status = "missing"
            models.append(
                {
                    "key": key,
                    "price": status,
                    "source": "configured" if key in configured else "observed",
                }
            )
        return {
            "price_table": self._registry.explicit_pricing,
            "currency": pricing.currency
            if pricing is not None and self._registry.explicit_pricing
            else None,
            "models": models,
        }

    def _configured_models(self, snapshot: Mapping[ModelScope, ModelRoutes]) -> set[str]:
        keys: set[str] = set()
        server_scope = ModelScope.server()
        contexts: list[tuple[ModelScope, ...]] = [(server_scope,)] + [
            (scope, server_scope) for scope in snapshot if scope.kind == ModelScopeKind.AGENT
        ]
        for scopes in contexts:
            for route, capability in _DEFAULT_ROUTES:
                for effort in Effort:
                    for scope in scopes:
                        routes = snapshot.get(scope, {})
                        selected = self._router.select_route(
                            scope=scope,
                            routes=routes,
                            route=route,
                            capability=capability,
                            effort=effort,
                        )
                        if selected is None:
                            continue
                        target = routes[selected]
                        if isinstance(target, str):
                            try:
                                target = self._implementations.default_config(
                                    target, capability, route, {"effort": effort.value}
                                )
                            except ModelConfigurationError:
                                # Unknown custom defaults remain unknown. A
                                # selected binding never falls through to a
                                # different provider in a less-specific scope.
                                break
                        if isinstance(target.model, str):
                            keys.add(target.model)
                        break

        # Explicit application-only routes aren't in the built-in inventory.
        # Keep their concrete models discoverable as before; don't re-add
        # overridden targets from known built-in routes or namespace wildcards.
        for scope, routes in snapshot.items():
            for name, target in routes.items():
                if not isinstance(target, ModelConfig) or not isinstance(target.model, str):
                    continue
                # Use routing's own matching rules, including capability and
                # effort, to distinguish built-in coverage from application
                # routes/capabilities outside this discovery inventory.
                builtin = any(
                    self._router.select_route(
                        scope=scope,
                        routes={name: target},
                        route=route,
                        capability=capability,
                        effort=effort,
                    )
                    is not None
                    for route, capability in _DEFAULT_ROUTES
                    for effort in Effort
                )
                if not builtin:
                    keys.add(target.model)
        return keys

    async def warn_missing(self) -> None:
        try:
            report = await self.read()
        except Exception:
            self._logger.warning("Model pricing coverage could not be checked at startup.")
            return
        missing = [model["key"] for model in report["models"] if model["price"] == "missing"]
        if missing:
            self._logger.warning(
                "Model prices are missing for: "
                + ", ".join(missing)
                + ". Add server.set_pricing(...) to report monetary costs; "
                "without it usage remains tokens only. Prices apply to future turns."
            )
