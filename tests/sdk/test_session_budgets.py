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
from typing import Any, cast
from decimal import Decimal

import httpx
import parlant.sdk as p
from parlant.core.emissions import EventEmitter
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.types import Context as EngineInfo
from parlant.core.nlp.common import UsageInfo
from parlant.core.pricing import PricingRegistry, SessionBudgetService
from parlant.core.sessions import SessionId, SessionStore
from parlant.core.usage_reporter import UsageReporter
from tests.sdk.utils import Context, SDKTest


class _CompassWithoutInitializationWarmup(CompassEngine):
    """Keep these cost-gate SDK tests deterministic and model-free."""

    async def initialize(
        self,
        context: EngineInfo,
        event_emitter: EventEmitter,
    ) -> None:
        pass


class _CompassBudgetSDKTest(SDKTest):
    async def configure_container(self, container: p.Container) -> p.Container:
        container[CompassEngine] = _CompassWithoutInitializationWarmup
        return container


async def _seed_cost(ctx: Context, budget: p.SessionBudget, session_id: str) -> None:
    service = ctx.container[SessionBudgetService]
    sid = SessionId(session_id)
    # First observation establishes the activation baseline at zero.
    assert await service.measure(budget._definition, sid) == Decimal(0)

    usage = UsageInfo(input_tokens=2_000_000, output_tokens=0)
    registry = ctx.container[PricingRegistry]
    registry.report_usage("seed-trace", "test-model", usage)
    await service.record_trace(
        sid,
        "seed-trace",
        {"test-model": usage},
    )


async def _wait_for_ready(ctx: Context, session_id: str, min_offset: int) -> dict[str, Any]:
    for _ in range(100):
        statuses = await ctx.client.sessions.list_events(
            session_id=session_id,
            min_offset=min_offset,
            source="ai_agent",
            kinds="status",
            wait_for_data=1,
        )
        for event in statuses:
            data = event.model_dump()["data"]
            if data["status"] == "ready":
                return cast(dict[str, Any], data)
            if data["status"] == "error":
                raise AssertionError(f"Compass processing failed: {data}")
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for terminal ready status")


class Test_that_a_priced_budget_observation_can_handoff_a_compass_session(_CompassBudgetSDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="session-budgets-agent-1",
            name="Budget handoff agent",
            instructions="A test agent",
        )
        pricing = await server.set_pricing(
            {
                "test-model": p.ModelPricing(
                    input_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("1"),
                    output_tokens=Decimal("1"),
                )
            },
            currency="credits",
        )
        self.budget = await pricing.create_session_budget(
            name="handoff",
            value=Decimal("1"),
        )
        self.measured: Decimal | None = None

        async def handoff(context: p.EngineContext, match: p.RuleEvaluation) -> None:
            self.measured = await self.budget.measure()
            await self.budget.reset()
            await p.Session.current.update(mode="manual")

        await self.agent.create_observation(
            condition="The session budget is exceeded",
            title="Session budget exceeded",
            evaluation_func=self.budget.exceeded,
            on_matched=handoff,
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        await _seed_cost(ctx, self.budget, session.id)

        customer_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Continue",
        )
        ready = await _wait_for_ready(ctx, session.id, customer_event.offset)

        stored = await ctx.container[SessionStore].read_session(SessionId(session.id))
        assert self.measured == Decimal("2")
        assert stored.mode == "manual"
        assert ready["data"]["session_mode"] == "manual"
        sdk_session = await ctx.server.get_session(id=session.id)
        assert await sdk_session.usage.measure(self.budget) == Decimal(0)


class Test_that_compass_usage_is_exposed_by_the_session_retrieve_api(_CompassBudgetSDKTest):
    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def report_usage_and_finish(
            _context: p.EngineContext,
            _payload: Any,
            _exc: Exception | None,
        ) -> p.CompassHookResult:
            self.usage_reporter.report_usage(
                "test-model",
                UsageInfo(
                    requests=2,
                    input_tokens=2_000_000,
                    cached_input_tokens=1_000_000,
                    output_tokens=500_000,
                ),
                provider_name="test-provider",
            )
            self.hook_called.set()
            return p.CompassHookResult.BAIL

        hooks.on_acknowledged.append(report_usage_and_finish)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.hook_called = asyncio.Event()
        self.usage_reporter = server.container[UsageReporter]
        self.agent = await server.create_agent(
            id="session-budgets-agent-3",
            name="Usage reporting agent",
            instructions="A test agent",
        )
        await server.set_pricing(
            {
                "test-model": p.ModelPricing(
                    input_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("0.5"),
                    output_tokens=Decimal("2"),
                )
            },
            currency="USD",
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        customer_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Report this turn",
        )
        await asyncio.wait_for(self.hook_called.wait(), timeout=10)

        async with httpx.AsyncClient(base_url=f"http://localhost:{ctx.server.port}") as http:
            for _ in range(100):
                response = await http.get(f"/sessions/{session.id}")
                if customer_event.trace_id in response.json()["usage"]:
                    break
                await asyncio.sleep(0.05)
        response.raise_for_status()
        retrieved = response.json()

        assert retrieved["metadata"] == {}
        assert retrieved["usage"] == {
            customer_event.trace_id: {
                "timestamp_utc": retrieved["usage"][customer_event.trace_id]["timestamp_utc"],
                "requests": {
                    "test-provider:test-model": {
                        "request_count": 2,
                        "input_tokens": 2_000_000,
                        "cached_input_tokens": 1_000_000,
                        "output_tokens": 500_000,
                        "extra": {},
                    }
                },
                "costs": {
                    "currency": "USD",
                    "total": "2.5",
                    # A bare "test-model" price covers the model under any provider.
                    "per_model": {"test-provider:test-model": "2.5"},
                    "per_provider": {"test-provider": "2.5"},
                },
            }
        }


class Test_that_one_model_served_by_two_providers_is_priced_at_each_rate(_CompassBudgetSDKTest):
    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def report_usage_and_finish(
            _context: p.EngineContext,
            _payload: Any,
            _exc: Exception | None,
        ) -> p.CompassHookResult:
            usage = UsageInfo(requests=1, input_tokens=1_000_000, output_tokens=0)
            self.usage_reporter.report_usage("gpt-5.5", usage, provider_name="openai")
            self.usage_reporter.report_usage("gpt-5.5", usage, provider_name="azure-eu")
            self.hook_called.set()
            return p.CompassHookResult.BAIL

        hooks.on_acknowledged.append(report_usage_and_finish)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.hook_called = asyncio.Event()
        self.usage_reporter = server.container[UsageReporter]
        self.agent = await server.create_agent(
            id="session-budgets-agent-5",
            name="Two providers agent",
            instructions="A test agent",
        )
        pricing = await server.set_pricing(
            {
                # The bare key is the model's rate under any provider; the
                # exact key overrides it for that one provider.
                "gpt-5.5": p.ModelPricing(
                    input_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("1"),
                    output_tokens=Decimal("1"),
                ),
                "azure-eu:gpt-5.5": p.ModelPricing(
                    input_tokens=Decimal("3"),
                    cached_input_tokens=Decimal("3"),
                    output_tokens=Decimal("3"),
                ),
            },
            currency="USD",
        )
        self.budget = await pricing.create_session_budget(
            name="two-providers", value=Decimal("100")
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        # Activate the budget so its baseline precedes the turn below.
        service = ctx.container[SessionBudgetService]
        assert await service.measure(self.budget._definition, SessionId(session.id)) == Decimal(0)

        customer_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Report this turn",
        )
        await asyncio.wait_for(self.hook_called.wait(), timeout=10)

        async with httpx.AsyncClient(base_url=f"http://localhost:{ctx.server.port}") as http:
            for _ in range(100):
                response = await http.get(f"/sessions/{session.id}")
                if customer_event.trace_id in response.json()["usage"]:
                    break
                await asyncio.sleep(0.05)
        response.raise_for_status()
        costs = response.json()["usage"][customer_event.trace_id]["costs"]

        assert costs == {
            "currency": "USD",
            "total": "4",
            "per_model": {"openai:gpt-5.5": "1", "azure-eu:gpt-5.5": "3"},
            "per_provider": {"openai": "1", "azure-eu": "3"},
        }
        assert await service.measure(self.budget._definition, SessionId(session.id)) == Decimal(4)


class Test_that_a_session_budget_works_without_set_pricing(_CompassBudgetSDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="session-budgets-agent-zero-config",
            name="Zero Config Budget Agent",
            instructions="Agent for zero-config budgets",
        )
        # No server.set_pricing(): providers that report their own USD charge
        # (OpenRouter, LiteLLM) need no price table, so neither does a budget.
        self.budget = await server.create_session_budget(
            name="zero-config",
            value=Decimal("1"),
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)

        service = ctx.container[SessionBudgetService]
        sid = SessionId(session.id)
        assert await service.measure(self.budget._definition, sid) == Decimal(0)

        usage = UsageInfo(
            input_tokens=1000,
            output_tokens=100,
            extra={"reported_cost": "0.25"},
        )
        ctx.container[PricingRegistry].report_usage("zero-config-trace", "openai/gpt-4o", usage)
        await service.record_trace(sid, "zero-config-trace", {"openai/gpt-4o": usage})

        assert await service.measure(self.budget._definition, sid) == Decimal("0.25")


class Test_that_a_server_budget_adopts_the_configured_pricing_currency(_CompassBudgetSDKTest):
    async def setup(self, server: p.Server) -> None:
        await server.set_pricing(
            {
                "test-model": p.ModelPricing(
                    input_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("1"),
                    output_tokens=Decimal("1"),
                )
            },
            currency="credits",
        )
        self.budget = await server.create_session_budget(
            name="currency-adoption",
            value=Decimal("5"),
        )

    async def run(self, ctx: Context) -> None:
        assert self.budget.currency == "credits"


class Test_that_creating_a_session_records_its_cache_warm_up_usage_before_any_message(SDKTest):
    """Session creation warms the matcher's prompt cache. That request is paid
    for, so it must be accounted on the session even though no turn exists yet.
    Runs the real engine (no initialization stub) against the live model."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="session-budgets-agent-warm-up",
            name="Warm-up accounting agent",
            instructions="A test agent",
        )
        # The ranker only warms its prefix once enough rules would fan out,
        # one request each; high-weight rules are ranked at the default effort.
        for index in range(5):
            await self.agent.create_observation(
                condition=f"The customer mentions topic number {index}",
                weight=p.Weight.HIGH,
            )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )

        retrieved = await ctx.client.sessions.retrieve(session_id=session.id)
        usage = cast(dict[str, dict[str, Any]], retrieved.usage)

        assert len(usage) == 1, usage
        (record,) = usage.values()
        assert record["requests"], record
        assert sum(r["input_tokens"] for r in record["requests"].values()) > 0

        # The warm-up has no turn: its trace id matches no event on the session.
        events = await ctx.client.sessions.list_events(session_id=session.id, min_offset=0)
        assert {e.trace_id for e in events}.isdisjoint(usage)
