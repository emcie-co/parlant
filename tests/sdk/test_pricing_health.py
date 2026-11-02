"""Pricing discovery must be live, informational and restricted to admins."""

import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import parlant.sdk as p

from parlant.api.app import create_api_app
from parlant.api.authorization import AuthorizationPolicy, HeaderApiKeyAuthorizationPolicy
from parlant.core.agents import Effort
from parlant.core.application import Application
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.emissions import EventEmitter
from parlant.core.engines.types import Context as EngineInfo
from parlant.core.health import HealthReporter
from parlant.core.loggers import CompositeLogger, FileLogger, Logger
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.model_routing import (
    ModelCapability,
    ModelConfigurationService,
    ModelImplementationRegistry,
    ModelScope,
)
from parlant.core.pricing import ModelPricing, PricingRegistry
from parlant.core.tracer import LocalTracer
from parlant.core.tunnel_handlers import create_tunnel_handler_registry
from tests.sdk.utils import Context, SDKTest


class _WithoutWarmup(CompassEngine):
    async def initialize(self, context: EngineInfo, event_emitter: EventEmitter) -> None:
        pass


class _PricingSDKTest(SDKTest):
    async def configure_container(self, container: p.Container) -> p.Container:
        container[CompassEngine] = _WithoutWarmup
        return container


async def health(ctx: Context) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ctx.base_url}/healthz")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


class Test_that_configured_models_are_missing_before_usage_without_affecting_health(
    _PricingSDKTest
):
    async def setup(self, server: p.Server) -> None:
        await server.create_agent(
            id="pricing-agent",
            name="Pricing",
            instructions="",
            models={
                "react": p.ModelConfig(
                    "gemini:health-unpriced", args={"api_key": "must-not-appear"}
                ),
                "matching": p.ModelConfig("gemini:health-unpriced"),
            },
        )

    async def run(self, ctx: Context) -> None:
        body = await health(ctx)
        pricing = body["pricing"]
        assert pricing["price_table"] is False
        assert pricing["currency"] is None
        selected = [m for m in pricing["models"] if m["key"] == "gemini:health-unpriced"]
        assert selected == [
            {"key": "gemini:health-unpriced", "price": "missing", "source": "configured"}
        ]
        assert "must-not-appear" not in json.dumps(body)
        assert body["status"] == "healthy"
        assert "pricing" not in body["checks"]


class Test_that_exact_bare_and_zero_prices_and_route_changes_are_visible_immediately(
    _PricingSDKTest
):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="pricing-agent",
            name="Pricing",
            instructions="",
            models={
                "react": p.ModelConfig("gemini:health-exact"),
                "matching": p.ModelConfig("gemini:health-bare"),
            },
        )
        self.free = ModelPricing(
            input_tokens=Decimal(0), cached_input_tokens=Decimal(0), output_tokens=Decimal(0)
        )
        await server.set_pricing(
            currency="EUR", models={"gemini:health-exact": self.free, "health-bare": self.free}
        )

    async def run(self, ctx: Context) -> None:
        initial = await health(ctx)
        entries = {m["key"]: m for m in initial["pricing"]["models"]}
        assert initial["pricing"]["price_table"] is True
        assert initial["pricing"]["currency"] == "EUR"
        assert entries["gemini:health-exact"]["price"] == "table"
        assert entries["gemini:health-bare"]["price"] == "table"
        await self.agent.update(models={"react": p.ModelConfig("gemini:health-new")})
        changed = await health(ctx)
        entries = {m["key"]: m for m in changed["pricing"]["models"]}
        assert entries["gemini:health-new"]["price"] == "missing"
        assert "gemini:health-exact" not in entries
        await ctx.server.set_pricing(currency="EUR", models={"gemini:health-new": self.free})
        fixed = await health(ctx)
        assert {m["key"]: m["price"] for m in fixed["pricing"]["models"]}[
            "gemini:health-new"
        ] == "table"


class Test_that_effort_specific_models_have_independent_pricing(_PricingSDKTest):
    async def setup(self, server: p.Server) -> None:
        await server.create_agent(
            id="effort-pricing-agent",
            name="Effort pricing",
            instructions="",
            models={
                "react[effort<high]": p.ModelConfig("gemini:priced-low-effort"),
                "react[effort>=high]": p.ModelConfig("gemini:unpriced-high-effort"),
            },
        )
        await server.set_pricing(
            models={
                "gemini:priced-low-effort": ModelPricing(
                    input_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("0"),
                    output_tokens=Decimal("2"),
                )
            }
        )

    async def run(self, ctx: Context) -> None:
        body = await health(ctx)
        coverage = {row["key"]: row["price"] for row in body["pricing"]["models"]}
        assert coverage["gemini:priced-low-effort"] == "table"
        assert coverage["gemini:unpriced-high-effort"] == "missing"
        assert body["status"] == "healthy"

        registry = ctx.container[PricingRegistry]
        registry.report_usage(
            "low-effort", "gemini:priced-low-effort", UsageInfo(input_tokens=10, output_tokens=0)
        )
        low_cost, low_error = registry.snapshot_trace("low-effort")
        assert low_cost is not None
        assert low_cost.total == Decimal("0.00001")
        assert low_error is None

        registry.report_usage(
            "high-effort",
            "gemini:unpriced-high-effort",
            UsageInfo(input_tokens=10, output_tokens=0),
        )
        high_cost, high_error = registry.snapshot_trace("high-effort")
        assert high_cost is None
        assert high_error is not None
        assert "gemini:unpriced-high-effort" in high_error


class Test_that_provider_default_pricing_respects_capability_and_effort_qualifiers(_PricingSDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="qualified-pricing-agent", name="Qualified pricing", instructions=""
        )

    async def run(self, ctx: Context) -> None:
        implementations = ctx.container[ModelImplementationRegistry]
        configuration = ctx.container[ModelConfigurationService]

        def default_key(capability: ModelCapability, route: str, effort: Effort) -> str:
            config = implementations.default_config(
                "gemini", capability, route, {"effort": effort.value}
            )
            assert isinstance(config.model, str)
            return config.model

        # Unrestricted provider defaults remain visible across capabilities.
        baseline = await health(ctx)
        baseline_keys = {row["key"] for row in baseline["pricing"]["models"]}
        assert default_key(ModelCapability.REACT, "react", Effort.LOW) in baseline_keys
        assert default_key(ModelCapability.REACT, "react", Effort.HIGH) in baseline_keys
        assert (
            default_key(ModelCapability.EMBEDDING, "embedding.rules", Effort.LOW) in baseline_keys
        )
        await configuration.configure(ModelScope.server(), {"*": None})

        cases: tuple[tuple[str, ModelCapability, str, tuple[Effort, ...]], ...] = (
            ("*[capability=react]", ModelCapability.REACT, "react", tuple(Effort)),
            (
                "*[capability=react,effort>=high]",
                ModelCapability.REACT,
                "react",
                (Effort.HIGH, Effort.MAX),
            ),
            (
                "*[capability=embedding,effort=low]",
                ModelCapability.EMBEDDING,
                "embedding.rules",
                (Effort.LOW,),
            ),
        )
        actual: dict[tuple[str, str], set[str]] = {}
        expected: dict[tuple[str, str], set[str]] = {}
        for scope in (ModelScope.server(), ModelScope.agent(self.agent.id)):
            for selector, capability, route, efforts in cases:
                await configuration.configure(scope, {selector: "gemini"})
                body = await health(ctx)
                key = (scope.kind.value, selector)
                actual[key] = {row["key"] for row in body["pricing"]["models"]}
                expected[key] = {default_key(capability, route, effort) for effort in efforts}
                assert body["status"] == "healthy"
                await configuration.configure(scope, {selector: None})

        # Compare all six contexts so failures show both server and agent scope.
        assert actual == expected


class Test_that_observed_keys_and_provider_reported_costs_need_no_price_table(_PricingSDKTest):
    async def run(self, ctx: Context) -> None:
        registry = ctx.container[PricingRegistry]
        registry.report_usage(
            "observed", "gemini:health-embedding", UsageInfo(input_tokens=10, output_tokens=0)
        )
        registry.report_usage(
            "router",
            "openrouter:typesafe/jev",
            UsageInfo(input_tokens=10, output_tokens=2, extra={"reported_cost": "0.002"}),
        )
        registry.report_usage(
            "litellm",
            "litellm:custom",
            UsageInfo(input_tokens=10, output_tokens=2, extra={"reported_cost": "0.002"}),
        )
        # Clearing a trace must not erase which model was observed in this process.
        registry.clear_trace("observed")
        body = await health(ctx)
        entries = {m["key"]: m for m in body["pricing"]["models"]}
        assert entries["gemini:health-embedding"] == {
            "key": "gemini:health-embedding",
            "price": "missing",
            "source": "observed",
        }
        assert entries["openrouter:typesafe/jev"]["price"] == "provider_reported"
        assert entries["litellm:custom"]["price"] == "provider_reported"
        assert len(entries) == len(body["pricing"]["models"])


class Test_that_only_admin_health_and_the_authenticated_tunnel_expose_pricing(_PricingSDKTest):
    async def run(self, ctx: Context) -> None:
        strict = ctx.container.clone()
        strict[AuthorizationPolicy] = HeaderApiKeyAuthorizationPolicy(
            api_key="pricing-admin", header_name="X-Admin-Key"
        )
        app = await create_api_app(strict)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            public = await client.get("/healthz")
            admin = await client.get("/healthz", headers={"X-Admin-Key": "pricing-admin"})
        assert set(public.json()) == {"status"}
        assert "pricing" in admin.json()
        handlers = create_tunnel_handler_registry(
            ctx.container[Application], ctx.container[HealthReporter]
        )
        tunneled = await handlers.unary["system.health"]({})
        assert tunneled["pricing"] == admin.json()["pricing"]


class Test_that_startup_warns_about_missing_configured_model_prices(_PricingSDKTest):
    async def configure_container(self, container: p.Container) -> p.Container:
        self.logs = TemporaryDirectory()
        self.path = Path(self.logs.name) / "startup.log"
        logger = container[Logger]
        assert isinstance(logger, CompositeLogger)
        logger.append(FileLogger(self.path, LocalTracer()))
        return await super().configure_container(container)

    async def setup(self, server: p.Server) -> None:
        await server.create_agent(
            id="pricing-agent",
            name="Pricing",
            instructions="",
            models={"react": p.ModelConfig("gemini:health-startup-missing")},
        )

    async def run(self, ctx: Context) -> None:
        try:
            output = self.path.read_text()
            assert "gemini:health-startup-missing" in output
            assert "set_pricing" in output
        finally:
            self.logs.cleanup()
