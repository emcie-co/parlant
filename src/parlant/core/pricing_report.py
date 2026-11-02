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
    ModelRouter,
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
    ) -> None:
        self._configuration = configuration
        self._implementations = implementations
        self._registry = pricing_registry
        self._agents = agent_store
        self._logger = logger

    async def read(self) -> Mapping[str, Any]:
        # Bulk reads avoid one request per agent. Exclude stale/deleted agent
        # bindings, plus transient session/turn overrides that aren't configured
        # server routes (their actual usage is included through observations).
        agents = {str(agent.id) for agent in await self._agents.list_agents()}
        configured: set[str] = set()
        for scope, routes in (await self._configuration.read_all()).items():
            if scope.kind == ModelScopeKind.SERVER or (
                scope.kind == ModelScopeKind.AGENT and scope.id in agents
            ):
                for selector_name, target in routes.items():
                    if isinstance(target, ModelConfig) and isinstance(target.model, str):
                        configured.add(target.model)
                    elif isinstance(target, str):
                        configured.update(self._provider_defaults(target, selector_name))
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

    def _provider_defaults(self, provider: str, selector_name: str) -> set[str]:
        selector = ModelRouter.parse_route_selector(selector_name)
        keys: set[str] = set()
        for route, capability in _DEFAULT_ROUTES:
            if not selector.serves(capability, None):
                continue
            for effort in Effort:
                if not selector.matches(effort):
                    continue
                try:
                    config = self._implementations.default_config(
                        provider, capability, route, {"effort": effort.value}
                    )
                except ModelConfigurationError:
                    continue  # Unknown custom adapter/defaults must not be guessed.
                if isinstance(config.model, str):
                    keys.add(config.model)
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
