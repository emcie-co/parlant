from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import asyncio
import os
from typing import Any, Callable, cast
from parlant.core.tracer import Tracer
from parlant.core.nlp.common import UsageInfo
from parlant.core.usage_reporter import UsageReporter

import pytest
import httpx
import parlant.sdk as p
from parlant.core.pricing import SessionBudgetService
from parlant.core.sessions import SessionId, SessionStore
from parlant.core.sessions import SessionDocumentStore
from parlant.core.sessions import CompositeSessionStore
from parlant.core.store_provider import StoreProvider, StoreProviderHints, StoreT
from parlant.adapters.db.sqlite import SQLiteDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.loggers import StdoutLogger
from parlant.core.tracer import LocalTracer
from parlant.core.application import Application
from parlant.core.tunnel_handlers.sessions import SessionTunnelHandlers
from tests.sdk.test_session_budgets import _CompassBudgetSDKTest
from tests.sdk.utils import Context


@p.tool
async def lookup_for_cost_test(query: str) -> p.ToolResult:
    """Look up a query."""
    return p.ToolResult(data=query, costs={"Search": Decimal("0.02")})


@pytest.mark.parametrize("amount", [Decimal("-1"), Decimal("NaN"), Decimal("Infinity")])
def test_that_invalid_tool_costs_are_rejected(amount: Decimal) -> None:
    with pytest.raises(ValueError):
        p.ToolResult(data={}, costs={"Search": amount})


def test_that_empty_cost_categories_are_rejected() -> None:
    with pytest.raises(ValueError):
        p.ToolResult(data={}, costs={" ": Decimal("1")})


class Test_that_late_tool_results_are_persisted_without_an_utter(_CompassBudgetSDKTest):
    async def setup(self, server: p.Server) -> None:
        self.server = server
        self.agent = await server.create_agent(
            id="tool-cost-agent", name="Tool costs", instructions="A test agent"
        )
        pricing = await server.set_pricing({}, currency="USD")
        self.budget = await pricing.create_session_budget(name="Tools", value=Decimal("1"))

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        sid = SessionId(created.id)
        budget_service = ctx.container[SessionBudgetService]
        assert await budget_service.measure(self.budget._definition, sid) == Decimal("0")
        session = await self.server.get_session(id=created.id)
        before = await ctx.container[SessionStore].read_session(sid)
        for amount in ("0.02", "0.03"):
            await session.emit_tool_result(
                lookup_for_cost_test,
                p.ToolResult(data={"answer": "Ready"}, costs={"Search": Decimal(amount)}),
                arguments={"query": "question"},
            )
        after = await ctx.container[SessionStore].read_session(sid)
        assert after.modified_utc > before.modified_utc
        events = await ctx.client.sessions.list_events(
            session_id=created.id, kinds="tool", min_offset=0
        )
        assert len(events) == 2
        assert events[0].trace_id != events[1].trace_id
        assert all(
            event.metadata["__parlant__"]["turn_trace_id"] == event.trace_id for event in events
        )
        assert events[0].model_dump()["data"]["tool_calls"][0]["result"]["costs"] == {
            "Search": "0.02"
        }
        assert await budget_service.measure(self.budget._definition, sid) == Decimal("0.05")
        async with httpx.AsyncClient(base_url=ctx.base_url) as http:
            response = await http.get(f"/sessions/{created.id}")
        response.raise_for_status()
        assert response.json()["costs"]["per_tool"] == {"Search": "0.05"}


class Test_that_a_background_result_outlives_its_originating_turn(_CompassBudgetSDKTest):
    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def start_background(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            session = p.Session.current
            self.origin_trace = self.server.container[Tracer].trace_id
            self.server.container[UsageReporter].report_usage(
                "test", UsageInfo(input_tokens=1_000_000, output_tokens=0), provider_name="test"
            )

            async def complete() -> None:
                await self.release.wait()
                await session.emit_tool_result(
                    lookup_for_cost_test,
                    p.ToolResult(data="Background answer", costs={"Search": Decimal("0.2")}),
                )

            self.task = asyncio.create_task(complete())
            self.started.set()
            return p.CompassHookResult.BAIL

        hooks.on_acknowledged.append(start_background)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.server = server
        self.release, self.started = asyncio.Event(), asyncio.Event()
        self.agent = await server.create_agent(
            id="late-tool-cost-agent", name="Late results", instructions="Test agent"
        )
        await server.set_pricing(
            {
                "test": p.ModelPricing(
                    input_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("0"),
                    output_tokens=Decimal("0"),
                )
            },
            currency="USD",
        )

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        await ctx.client.sessions.create_event(
            session_id=created.id, kind="message", source="customer", message="Start"
        )
        await asyncio.wait_for(self.started.wait(), timeout=10)
        async with httpx.AsyncClient(base_url=ctx.base_url) as http:
            for _ in range(100):
                original = (await http.get(f"/sessions/{created.id}")).json()
                if self.origin_trace in original["usage"]:
                    break
                await asyncio.sleep(0.02)
            assert self.origin_trace in original["usage"]
            self.release.set()
            await asyncio.wait_for(self.task, timeout=10)
            final = (await http.get(f"/sessions/{created.id}")).json()
        assert final["usage"][self.origin_trace] == original["usage"][self.origin_trace]
        assert len(final["usage"]) == 2
        assert final["costs"]["total"] == "1.2"
        assert final["costs"]["per_tool"] == {"Search": "0.2"}
        store = ctx.container[StoreProvider].get_store(
            SessionStore, StoreProviderHints(call_site="engine")
        )
        saved = await store.read_session(SessionId(created.id))
        internal = cast(dict[str, Any], saved.metadata)["__parlant__"]
        records = internal["usage"]["records"]
        assert len(records) == 2
        assert all(record["turn_trace_id"] == self.origin_trace for record in records.values())
        tools = await ctx.client.sessions.list_events(session_id=created.id, kinds="tool")
        assert tools[0].trace_id != self.origin_trace
        assert tools[0].metadata["__parlant__"]["turn_trace_id"] == self.origin_trace
        # Exercise the exact serializers consumed by session ingestion. Public
        # session metadata intentionally hides the reserved internal ledger.
        handlers = SessionTunnelHandlers(ctx.container[Application].sessions)
        wire_session = await handlers.read_session({"session_id": created.id})
        assert wire_session["metadata"]["__parlant__"]["usage"]["records"] == records
        wire_events = await handlers.list_events(
            {"session_id": created.id, "min_offset": 0, "wait_for_data": 0}
        )
        wire_tool = next(event for event in wire_events["events"] if event["kind"] == "tool")
        assert wire_tool["metadata"]["__parlant__"]["turn_trace_id"] == self.origin_trace


class Test_that_background_costs_use_the_current_persistent_store(
    Test_that_a_background_result_outlives_its_originating_turn
):
    async def test_run(self) -> None:
        with TemporaryDirectory() as directory:
            async with SQLiteDocumentDatabase(
                StdoutLogger(LocalTracer()), Path(directory) / "sessions.db"
            ) as database:
                self.database = database
                async with (
                    SessionDocumentStore(database) as store,
                    SessionDocumentStore(TransientDocumentDatabase()) as previous_store,
                ):
                    self.store = store
                    self.previous_store = previous_store
                    await super().test_run()

    async def configure_container(self, container: p.Container) -> p.Container:
        container = await super().configure_container(container)
        container[SessionStore] = self.previous_store
        return container

    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        # A long-lived service can retain an earlier store binding. Emission
        # and accounting must both follow the provider's current selection.
        server.container[SessionBudgetService]
        server.container[SessionStore] = self.store


class _RuntimeSessionStoreProvider(StoreProvider):
    """Keep declarations separate from persisted app/engine sessions, as managed hosting does."""

    def __init__(self, base: StoreProvider, runtime_store: SessionStore) -> None:
        self.base = base
        self.runtime_store = runtime_store

    def get_store(
        self, store_type: Callable[..., StoreT], hints: StoreProviderHints = {}
    ) -> StoreT:
        if store_type is SessionStore and hints.get("call_site") in {"app", "engine"}:
            return cast(StoreT, self.runtime_store)
        return self.base.get_store(store_type, hints)


class Test_that_background_costs_follow_the_persisted_runtime_session_route(
    Test_that_background_costs_use_the_current_persistent_store
):
    async def configure_container(self, container: p.Container) -> p.Container:
        container = await super().configure_container(container)
        container[StoreProvider] = _RuntimeSessionStoreProvider(
            container[StoreProvider],
            CompositeSessionStore(self.store, [self.previous_store]),
        )
        return container

    async def setup(self, server: p.Server) -> None:
        # Unlike the rebinding case, the SDK declaration store remains distinct.
        await Test_that_a_background_result_outlives_its_originating_turn.setup(self, server)

    async def run(self, ctx: Context) -> None:
        await super().run(ctx)
        sessions = await self.store.list_sessions()
        assert len(sessions.items) == 1
        sid = sessions.items[0].id
        labels = p.SessionLabels(sid, set(), self.server)
        await labels.add("docs-delivery-failed")
        assert "docs-delivery-failed" in (await self.store.read_session(sid)).labels
        await labels.remove("docs-delivery-failed")
        assert "docs-delivery-failed" not in (await self.store.read_session(sid)).labels


class Test_that_a_delayed_result_keeps_its_origin_after_a_new_customer_turn(
    Test_that_a_background_result_outlives_its_originating_turn
):
    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        hooks = await super().configure_hooks(hooks)
        original = hooks.on_acknowledged.pop()

        async def acknowledge(
            context: p.EngineContext,
            payload: Any,
            error: Exception | None,
        ) -> p.CompassHookResult:
            if self.started.is_set():
                self.newer_trace = self.server.container[Tracer].trace_id
                self.server.container[UsageReporter].report_usage(
                    "test",
                    UsageInfo(input_tokens=1_000_000, output_tokens=0),
                    provider_name="test",
                )
                self.newer_started.set()
                return p.CompassHookResult.BAIL
            return await original(context, payload, error)

        hooks.on_acknowledged.append(acknowledge)
        return hooks

    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.newer_started = asyncio.Event()

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        for text, ready in [("Start", self.started), ("Newer question", self.newer_started)]:
            await ctx.client.sessions.create_event(
                session_id=created.id, kind="message", source="customer", message=text
            )
            await asyncio.wait_for(ready.wait(), timeout=10)
            expected = 1 if ready is self.started else 2
            for _ in range(100):
                session = await ctx.client.sessions.retrieve(session_id=created.id)
                if len(session.usage) == expected:
                    break
                await asyncio.sleep(0.02)
            assert len(session.usage) == expected
        assert self.newer_trace != self.origin_trace
        self.release.set()
        await asyncio.wait_for(self.task, timeout=10)
        events = await ctx.client.sessions.list_events(session_id=created.id, kinds="tool")
        assert events[0].metadata["__parlant__"]["turn_trace_id"] == self.origin_trace
        assert events[0].trace_id not in (self.origin_trace, self.newer_trace)
        final = await ctx.client.sessions.retrieve(session_id=created.id)
        assert len(final.usage) == 3
        assert final.costs["total"] == "2.2"


class Test_that_failed_utterances_keep_costs_and_restore_the_callers_session(
    Test_that_a_background_result_outlives_its_originating_turn
):
    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def record_usage() -> None:
            self.server.container[UsageReporter].report_usage(
                "test",
                UsageInfo(input_tokens=1_000_000, output_tokens=0),
                provider_name="test",
            )

        async def start(
            context: p.EngineContext,
            payload: Any,
            error: Exception | None,
        ) -> p.CompassHookResult:
            self.origin_trace = self.server.container[Tracer].trace_id
            await record_usage()

            async def finish() -> None:
                session_id = p.Session.current.id
                await self.release.wait()
                for _ in range(2):
                    # The supported generation hook deliberately bails out, so
                    # no model is called. The real engine must still persist
                    # incurred usage during its exception finalization.
                    with pytest.raises(KeyError):
                        await p.Agent.current.utter(rules=[{"content": "Report the result."}])
                    assert p.Session.current.id == session_id

            self.task = asyncio.create_task(finish())
            self.started.set()
            return p.CompassHookResult.BAIL

        async def bail(
            context: p.EngineContext,
            payload: Any,
            error: Exception | None,
        ) -> p.CompassHookResult:
            await record_usage()
            return p.CompassHookResult.BAIL

        hooks.on_acknowledged.append(start)
        hooks.on_generating_messages.append(bail)
        return hooks

    async def setup(self, server: p.Server) -> None:
        await super().setup(server)

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        await ctx.client.sessions.create_event(
            session_id=created.id, kind="message", source="customer", message="Start"
        )
        await asyncio.wait_for(self.started.wait(), timeout=10)
        for _ in range(100):
            original = await ctx.client.sessions.retrieve(session_id=created.id)
            if self.origin_trace in original.usage:
                break
            await asyncio.sleep(0.02)
        self.release.set()
        await asyncio.wait_for(self.task, timeout=10)
        final = await ctx.client.sessions.retrieve(session_id=created.id)
        assert len(final.usage) == 3
        assert final.usage[self.origin_trace] == original.usage[self.origin_trace]
        assert final.costs["total"] == "3"


@pytest.mark.skipif(
    os.environ.get("PARLANT_RUN_LIVE_FOLLOWUP_TESTS") != "1",
    reason="explicit opt-in required for real Gemini follow-up conversations",
)
class Test_that_follow_up_utterances_record_new_linked_usage(
    Test_that_background_costs_follow_the_persisted_runtime_session_route
):
    """Real models, HTTP and managed-style persistent stores; no SDK mocks."""

    send_newer_turn = False

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def start_background(
            context: p.EngineContext,
            payload: Any,
            error: Exception | None,
        ) -> p.CompassHookResult:
            if self.started.is_set():
                self.newer_trace = self.server.container[Tracer].trace_id
                self.newer_started.set()
                return p.CompassHookResult.BAIL
            self.origin_trace = self.server.container[Tracer].trace_id
            self.server.container[UsageReporter].report_usage(
                "test",
                UsageInfo(input_tokens=1_000_000, output_tokens=0),
                provider_name="test",
            )

            async def complete() -> None:
                session = p.Session.current
                agent = p.Agent.current
                await self.release.wait()
                await session.emit_tool_result(
                    lookup_for_cost_test,
                    p.ToolResult(data="Background answer", costs={"Search": Decimal("0.2")}),
                )
                for _ in range(2):
                    answer = await agent.utter(
                        rules=[{"content": "Say that the background answer is complete."}],
                    )
                    assert answer.strip()

            self.task = asyncio.create_task(complete())
            self.started.set()
            return p.CompassHookResult.BAIL

        hooks.on_acknowledged.append(start_background)
        return hooks

    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.newer_started = asyncio.Event()
        await self.agent.update(models={"react": p.ModelConfig("gemini:gemini-4-argon")})
        await server.set_pricing(
            {
                "test": p.ModelPricing(
                    input_tokens=Decimal("1"),
                    output_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("0"),
                ),
                "gemini:gemini-4-argon": p.ModelPricing(
                    input_tokens=Decimal("1"),
                    output_tokens=Decimal("2"),
                    cached_input_tokens=Decimal("0.1"),
                ),
            },
            currency="USD",
        )

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        sid = SessionId(created.id)
        await ctx.client.sessions.create_event(
            session_id=created.id, kind="message", source="customer", message="Start"
        )
        await asyncio.wait_for(self.started.wait(), timeout=10)
        for _ in range(100):
            saved = await self.store.read_session(sid)
            internal = cast(dict[str, Any], saved.metadata).get("__parlant__", {})
            if self.origin_trace in internal.get("usage", {}).get("records", {}):
                break
            await asyncio.sleep(0.02)
        original = cast(dict[str, Any], saved.metadata)["__parlant__"]["usage"]["records"][
            self.origin_trace
        ]
        if self.send_newer_turn:
            await ctx.client.sessions.create_event(
                session_id=created.id, kind="message", source="customer", message="Another question"
            )
            await asyncio.wait_for(self.newer_started.wait(), timeout=10)
            for _ in range(100):
                statuses = await ctx.client.sessions.list_events(
                    session_id=created.id, kinds="status", trace_id=self.newer_trace
                )
                if any(event.data.get("status") == "ready" for event in statuses):
                    break
                await asyncio.sleep(0.02)
            assert self.newer_trace != self.origin_trace
        self.release.set()
        await asyncio.wait_for(self.task, timeout=120)
        saved = await self.store.read_session(sid)
        records = cast(dict[str, Any], saved.metadata)["__parlant__"]["usage"]["records"]
        assert len(records) == 4
        assert records[self.origin_trace] == original
        assert all(record["turn_trace_id"] == self.origin_trace for record in records.values())
        messages = await ctx.client.sessions.list_events(
            session_id=created.id, kinds="message", source="ai_agent"
        )
        assert len(messages) == 2
        traces = {event.trace_id for event in messages}
        assert len(traces) == 2 and self.origin_trace not in traces
        assert all(
            event.metadata["__parlant__"]["turn_trace_id"] == self.origin_trace
            for event in messages
        )
        for trace in traces:
            usage = records[trace]["models"]["gemini:gemini-4-argon"]
            assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0
            assert Decimal(records[trace]["costs"]["total"]) > 0
        final = await ctx.client.sessions.retrieve(session_id=created.id)
        assert Decimal(final.costs["total"]) == sum(
            Decimal(record["costs"]["total"]) for record in records.values()
        )
        # Reopen the same physical database through a new store instance.
        async with SessionDocumentStore(self.database) as reopened:
            assert (await reopened.read_session(sid)).metadata == saved.metadata


class Test_that_background_results_keep_their_origin_after_a_new_turn(
    Test_that_follow_up_utterances_record_new_linked_usage
):
    send_newer_turn = True
