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

"""Real SDK/LLM coverage for affinity-based, in-process queued responses.

Hooks and events control concurrency, not model output. Background lookup records
are fixed external-service data. No SDK, engine or model is replaced. Final
answers are graded semantically; response counts and persisted costs are exact.
"""

import asyncio
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import parlant.sdk as p
from parlant.core.application import Application
from parlant.core.common import ItemNotFoundError
from parlant.core.sessions import Event, EventKind, EventSource, SessionId, SessionStore
from parlant.core.tracer import Tracer
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test


@p.tool
async def queued_release_lookup(product: str) -> p.ToolResult:
    """Look up a release record."""
    raise AssertionError("Queued follow-ups must use persisted results, not run another lookup")


RECORDS: dict[str, dict[str, str]] = {
    "Atlas": {"product": "Atlas", "launch_date": "2031-06-14", "region": "Europe"},
    "Boreal": {"product": "Boreal", "launch_date": "2032-09-22", "region": "Europe"},
}


class QueuedSDKTest(SDKTest):
    STARTUP_TIMEOUT = 120

    async def setup(self, server: p.Server) -> None:
        self.server = server
        suffix = hashlib.sha256(type(self).__name__.encode()).hexdigest()[:10]
        self.agent = await server.create_agent(
            id=f"queued-{suffix}",
            name="Release support",
            instructions="You are a customer support agent for a software company.",
        )

    async def session(self, ctx: Context, question: str | None = None) -> p.Session:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        if question:
            await ctx.container[SessionStore].create_event(
                SessionId(created.id),
                source=EventSource.CUSTOMER,
                kind=EventKind.MESSAGE,
                trace_id="seeded-question",
                data={"message": question, "participant": {"display_name": "Morgan"}},
            )
        return await self.server.get_session(id=created.id)

    async def persist(self, session: p.Session, product: str, cost: str = "0.02") -> None:
        await session.emit_tool_result(
            queued_release_lookup,
            p.ToolResult(data=RECORDS[product], costs={"Release lookup": Decimal(cost)}),
            arguments={"product": product},
        )

    async def idle(self, ctx: Context, session: p.Session) -> None:
        await asyncio.wait_for(
            ctx.container[Application].sessions.response_coordinator.wait_idle(
                SessionId(session.id)
            ),
            timeout=150,
        )

    async def messages(self, ctx: Context, session: p.Session) -> list[str]:
        events = await ctx.container[SessionStore].list_events(
            SessionId(session.id), source=EventSource.AI_AGENT, kinds=[EventKind.MESSAGE]
        )
        replies = [
            str(event.data["message"])
            for event in events
            if isinstance(event.data, dict)
            and isinstance(event.data.get("message"), str)
            and event.data["message"].strip()
        ]
        if evidence_dir := os.environ.get("PARLANT_QUEUED_RESPONSE_EVIDENCE_DIR"):
            path = Path(evidence_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{type(self).__name__}.json").write_text(
                json.dumps({"session_id": session.id, "replies": replies}, indent=2)
            )
        return replies

    async def assert_both_records(self, text: str) -> None:
        assert await nlp_test(
            text,
            "The reply gives Atlas's launch date as 14 June 2031 and Boreal's as 22 September 2032. Any natural date format is acceptable.",
        )


class Test_that_concurrent_ready_requests_deliver_one_queued_reply(QueuedSDKTest):
    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "When will Atlas and Boreal launch?")
        await self.persist(session, "Atlas")
        await self.persist(session, "Boreal")
        await asyncio.gather(*(self.agent.queue_response(session) for _ in range(3)))
        await self.idle(ctx, session)
        replies = await self.messages(ctx, session)
        assert len(replies) == 1
        await self.assert_both_records(replies[0])


class Test_that_two_customer_messages_share_one_registered_background_batch(QueuedSDKTest):
    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.registered = {name: asyncio.Event() for name in RECORDS}
        self.release = {name: asyncio.Event() for name in RECORDS}
        self.saved = {name: asyncio.Event() for name in RECORDS}

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def register(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            session = p.Session.current
            latest = next(
                event
                for event in reversed(context.interaction.events)
                if event.kind == EventKind.MESSAGE and event.source == EventSource.CUSTOMER
            )
            assert isinstance(latest.data, dict)
            product = "Boreal" if "Boreal" in str(latest.data["message"]) else "Atlas"

            async def complete() -> None:
                await self.release[product].wait()
                await self.persist(session, product)
                self.saved[product].set()

            await p.Agent.current.queue_response(after=complete())
            self.registered[product].set()
            if product == "Atlas":
                # The second HTTP customer message cancels this still-active
                # turn. Its registered job must survive that cancellation.
                await asyncio.Future[None]()
            return p.CompassHookResult.BAIL

        hooks.on_acknowledged.append(register)
        return hooks

    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx)
        for product in RECORDS:
            await ctx.client.sessions.create_event(
                session_id=session.id,
                kind="message",
                source="customer",
                message=f"Please check when {product} launches.",
            )
            await asyncio.wait_for(self.registered[product].wait(), timeout=30)
        self.release["Atlas"].set()
        await asyncio.wait_for(self.saved["Atlas"].wait(), timeout=30)
        assert await self.messages(ctx, session) == []
        self.release["Boreal"].set()
        await self.idle(ctx, session)
        replies = await self.messages(ctx, session)
        assert len(replies) == 1
        await self.assert_both_records(replies[0])


class Test_that_a_new_customer_turn_can_fulfill_a_ready_background_notification(QueuedSDKTest):
    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.registered = asyncio.Event()
        self.saved = asyncio.Event()
        self.finish_work = asyncio.Event()
        self.generating = asyncio.Event()
        self.finish_foreground = asyncio.Event()

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def acknowledge(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            if not self.registered.is_set():
                session = p.Session.current

                async def complete() -> None:
                    await self.persist(session, "Atlas")
                    self.saved.set()
                    await self.finish_work.wait()

                await p.Agent.current.queue_response(after=complete())
                self.registered.set()
                return p.CompassHookResult.BAIL
            return p.CompassHookResult.CALL_NEXT

        async def generating(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            self.generating.set()
            await self.finish_foreground.wait()
            return p.CompassHookResult.CALL_NEXT

        hooks.on_acknowledged.append(acknowledge)
        hooks.on_generating_messages.append(generating)
        return hooks

    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx)
        await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Please check Atlas's launch date.",
        )
        await asyncio.wait_for(self.registered.wait(), timeout=30)
        await asyncio.wait_for(self.saved.wait(), timeout=30)
        await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="What date is Atlas launching?",
        )
        await asyncio.wait_for(self.generating.wait(), timeout=60)
        self.finish_work.set()
        self.finish_foreground.set()
        await self.idle(ctx, session)
        replies = await self.messages(ctx, session)
        assert len(replies) == 1
        assert await nlp_test(replies[0], "The reply gives Atlas's launch date as 14 June 2031.")


class Test_that_a_result_arriving_during_generation_is_not_lost(QueuedSDKTest):
    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.generating = asyncio.Event()
        self.release = asyncio.Event()
        self.generations = 0

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def generating(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            self.generations += 1
            if self.generations == 1:
                self.generating.set()
                await self.release.wait()
            return p.CompassHookResult.CALL_NEXT

        hooks.on_generating_messages.append(generating)
        return hooks

    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "When will Atlas and Boreal launch?")
        await self.persist(session, "Atlas")
        await self.agent.queue_response(session)
        await asyncio.wait_for(self.generating.wait(), timeout=60)
        await self.persist(session, "Boreal")
        await self.agent.queue_response(session)
        self.release.set()
        await self.idle(ctx, session)
        replies = await self.messages(ctx, session)
        assert len(replies) == 1
        assert self.generations >= 2
        await self.assert_both_records(replies[0])


class Test_that_registered_work_keeps_its_session_context_and_tool_costs(QueuedSDKTest):
    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        # Synthetic rates keep the real model's usage priced so API aggregation
        # can include the exact tool costs. An unpriced trace makes costs null.
        await self.agent.update(models={"react": p.ModelConfig("gemini:gemini-3.8-flash")})
        await server.set_pricing(
            {
                "gemini:gemini-3.8-flash": p.ModelPricing(
                    input_tokens=Decimal("1"),
                    output_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("0"),
                )
            },
            currency="USD",
        )
        self.registered = asyncio.Event()
        self.release = asyncio.Event()
        self.seen: list[tuple[str, str]] = []
        self.origin_trace = ""

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def acknowledge(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            session = p.Session.current
            self.origin_trace = self.server.container[Tracer].trace_id

            async def complete() -> None:
                await self.release.wait()
                self.seen.append((p.Session.current.id, p.Agent.current.id))
                await self.persist(session, "Atlas", "0.02")
                await self.persist(session, "Boreal", "0.03")

            await p.Agent.current.queue_response(after=complete())
            self.registered.set()
            return p.CompassHookResult.BAIL

        hooks.on_acknowledged.append(acknowledge)
        return hooks

    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx)
        await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Check the launch dates for Atlas and Boreal.",
        )
        await asyncio.wait_for(self.registered.wait(), timeout=30)
        self.release.set()
        await self.idle(ctx, session)
        assert self.seen == [(session.id, self.agent.id)]
        final = await ctx.client.sessions.retrieve(session_id=session.id)
        assert Decimal(final.costs["per_tool"]["Release lookup"]) == Decimal("0.05")
        events: list[Event] = list(
            await ctx.container[SessionStore].list_events(
                SessionId(session.id), kinds=[EventKind.TOOL]
            )
        )
        assert len(events) == 2
        assert len({event.trace_id for event in events}) == 2
        saved = await ctx.container[SessionStore].read_session(SessionId(session.id))
        records = cast(dict[str, Any], saved.metadata)["__parlant__"]["usage"]["records"]
        reply_events = await ctx.container[SessionStore].list_events(
            SessionId(session.id), kinds=[EventKind.MESSAGE], source=EventSource.AI_AGENT
        )
        assert len(reply_events) == 1
        reply_trace = reply_events[0].trace_id
        assert reply_trace not in {self.origin_trace, *(event.trace_id for event in events)}
        for trace in [reply_trace, *(event.trace_id for event in events)]:
            assert records[trace]["turn_trace_id"] == self.origin_trace
        usage = records[reply_trace]["models"]["gemini:gemini-3.8-flash"]
        assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0
        replies = await self.messages(ctx, session)
        assert len(replies) == 1
        await self.assert_both_records(replies[0])


class Test_that_handoff_during_queued_generation_suppresses_automatic_text(
    Test_that_a_result_arriving_during_generation_is_not_lost
):
    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "When will Atlas launch?")
        await self.persist(session, "Atlas")
        await self.agent.queue_response(session)
        await asyncio.wait_for(self.generating.wait(), timeout=60)
        await session.update(mode="manual")
        self.release.set()
        await self.idle(ctx, session)
        assert await self.messages(ctx, session) == []


class Test_that_a_customer_interrupts_unpublished_queued_text_and_answers_once(
    Test_that_a_result_arriving_during_generation_is_not_lost
):
    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "When will Atlas launch?")
        await self.persist(session, "Atlas")
        await self.agent.queue_response(session)
        await asyncio.wait_for(self.generating.wait(), timeout=60)
        await self.persist(session, "Boreal")
        await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Please give me both Atlas and Boreal's launch dates together.",
        )
        self.release.set()
        await self.idle(ctx, session)
        replies = await self.messages(ctx, session)
        assert len(replies) == 1
        await self.assert_both_records(replies[0])


class Test_that_results_after_visible_output_form_one_automatic_continuation(QueuedSDKTest):
    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.visible = asyncio.Event()
        self.release = asyncio.Event()
        self.deliveries = 0

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def emitted(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            self.deliveries += 1
            if self.deliveries == 1:
                self.visible.set()
                await self.release.wait()
            return p.CompassHookResult.CALL_NEXT

        hooks.on_messages_emitted.append(emitted)
        return hooks

    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "Please check the launch dates of Atlas and Boreal.")
        await self.persist(session, "Atlas")
        await self.agent.queue_response(session)
        await asyncio.wait_for(self.visible.wait(), timeout=60)
        first = await self.messages(ctx, session)
        assert len(first) == 1
        await self.persist(session, "Boreal")
        await asyncio.gather(*(self.agent.queue_response(session) for _ in range(3)))
        self.release.set()
        await self.idle(ctx, session)
        replies = await self.messages(ctx, session)
        assert len(replies) == 2
        assert replies[0] == first[0]
        assert await nlp_test(
            replies[1], "The reply gives Boreal's launch date as 22 September 2032."
        )


class Test_that_api_cancellation_stops_a_queued_reply(
    Test_that_a_result_arriving_during_generation_is_not_lost
):
    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "When will Atlas launch?")
        await self.persist(session, "Atlas")
        await self.agent.queue_response(session)
        await asyncio.wait_for(self.generating.wait(), timeout=60)
        async with httpx.AsyncClient(base_url=ctx.base_url) as http:
            cancellation = await http.post(
                f"/sessions/{session.id}/events",
                json={"kind": "status", "source": "customer", "status": "cancelled"},
            )
        cancellation.raise_for_status()
        assert cancellation.json()["data"]["status"] == "cancelled"
        await self.idle(ctx, session)
        assert await self.messages(ctx, session) == []
        assert ctx.container[Application].sessions.response_coordinator.session_count == 0


class Test_that_background_only_followups_do_not_run_fresh_matching_or_tools(QueuedSDKTest):
    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.normal_turns = 0
        await self.agent.create_observation(
            condition="The customer asks for release information",
            note="Use the release lookup to check release records.",
            tools=[queued_release_lookup],
        )

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def acknowledged(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            self.normal_turns += 1
            return p.CompassHookResult.CALL_NEXT

        hooks.on_acknowledged.append(acknowledged)
        return hooks

    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "What is Atlas's launch date?")
        await self.persist(session, "Atlas")
        await self.agent.queue_response(session)
        await self.idle(ctx, session)
        assert self.normal_turns == 0
        tools = await ctx.container[SessionStore].list_events(
            SessionId(session.id), kinds=[EventKind.TOOL]
        )
        assert len(tools) == 1
        replies = await self.messages(ctx, session)
        assert len(replies) == 1
        assert await nlp_test(replies[0], "The reply gives Atlas's launch date as 14 June 2031.")


class Test_that_manual_mode_does_not_emit_a_queued_response(QueuedSDKTest):
    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "What is Atlas's launch date?")
        await session.update(mode="manual")
        await self.persist(session, "Atlas")
        await self.agent.queue_response(session)
        await self.idle(ctx, session)
        assert await self.messages(ctx, session) == []
        tools = await ctx.container[SessionStore].list_events(
            SessionId(session.id), kinds=[EventKind.TOOL]
        )
        assert len(tools) == 1


class PendingHistorySDKTest(QueuedSDKTest):
    replacement_customer_turn = False

    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.release_work = asyncio.Event()
        self.reply_emitted = asyncio.Event()
        self.generations = 0
        self.engine_errors: list[str] = []

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def generating(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            self.generations += 1
            if self.generations >= 2:
                # A safety fuse for the known infinite retry. No second model
                # request is made; the test still fails on the attempt count.
                return p.CompassHookResult.BAIL
            return p.CompassHookResult.CALL_NEXT

        async def emitted(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            self.reply_emitted.set()
            return p.CompassHookResult.CALL_NEXT

        async def failed(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            # Surface the engine failure itself instead of reporting a timeout
            # while waiting for a message it can no longer deliver.
            assert error is not None
            self.engine_errors.append(f"{error!r}; caused by {error.__cause__!r}")
            self.reply_emitted.set()
            return p.CompassHookResult.CALL_NEXT

        hooks.on_generating_messages.append(generating)
        hooks.on_messages_emitted.append(emitted)
        hooks.on_error.append(failed)
        return hooks

    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "When will Atlas launch?")
        await self.persist(session, "Atlas")

        async def work() -> None:
            await self.release_work.wait()

        await self.agent.queue_response(session, after=work())
        await self.agent.queue_response(session)
        async with httpx.AsyncClient(base_url=ctx.base_url) as http:
            deleted = await http.delete(f"/sessions/{session.id}/events", params={"min_offset": 0})
        deleted.raise_for_status()
        assert await ctx.container[SessionStore].list_events(SessionId(session.id)) == []
        if self.replacement_customer_turn:
            await self.persist(session, "Boreal")
            await ctx.client.sessions.create_event(
                session_id=session.id,
                kind="message",
                source="customer",
                message="What date is Boreal launching?",
            )
            await asyncio.wait_for(self.reply_emitted.wait(), timeout=60)
        self.release_work.set()
        await self.idle(ctx, session)
        replies = await self.messages(ctx, session)
        assert not self.engine_errors, self.engine_errors
        assert self.generations <= 1, (
            "History changes must not cause repeated generation against the same snapshot"
        )
        assert all("Atlas" not in text for text in replies), (
            "Deleted history must not be reintroduced"
        )
        if self.replacement_customer_turn:
            assert len(replies) == 1
            assert await nlp_test(
                replies[0], "The reply gives Boreal's launch date as 22 September 2032."
            )


class Test_that_deleting_history_during_pending_work_does_not_loop(PendingHistorySDKTest):
    pass


class Test_that_regenerating_after_history_deletion_does_not_replay_old_requests(
    PendingHistorySDKTest
):
    replacement_customer_turn = True


class Test_that_deleting_a_session_during_work_does_not_recreate_it(QueuedSDKTest):
    async def run(self, ctx: Context) -> None:
        session = await self.session(ctx, "When will Atlas launch?")
        release = asyncio.Event()
        completed = asyncio.Event()

        async def work() -> None:
            await release.wait()
            try:
                await self.persist(session, "Atlas")
            finally:
                completed.set()

        await self.agent.queue_response(session, after=work())
        await self.agent.queue_response(session)
        async with httpx.AsyncClient(base_url=ctx.base_url) as http:
            deleted = await http.delete(f"/sessions/{session.id}")
            deleted.raise_for_status()
            release.set()
            await asyncio.wait_for(completed.wait(), timeout=30)
            await self.idle(ctx, session)
            retrieved = await http.get(f"/sessions/{session.id}")
        assert retrieved.status_code == 404
        with pytest.raises(ItemNotFoundError):
            await ctx.container[SessionStore].list_events(SessionId(session.id))
        assert ctx.container[Application].sessions.response_coordinator.session_count == 0


class MidstreamQueuedSDKTest(QueuedSDKTest):
    async def setup(self, server: p.Server) -> None:
        await super().setup(server)
        self.partial_visible = asyncio.Event()
        self.resume_stream = asyncio.Event()
        self.first_partial: Event | None = None
        self.first_trace = ""
        self.update_count = 0

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def updated(
            context: p.EngineContext, payload: Any, error: Exception | None
        ) -> p.CompassHookResult:
            self.update_count += 1
            if self.first_partial is None:
                events = await self.server.container[SessionStore].list_events(
                    context.session.id,
                    trace_id=context.tracer.trace_id,
                    source=EventSource.AI_AGENT,
                    kinds=[EventKind.MESSAGE],
                )
                for event in events:
                    if not isinstance(event.data, dict):
                        continue
                    chunks = event.data.get("chunks")
                    if (
                        event.data.get("message")
                        and isinstance(chunks, list)
                        and chunks
                        and chunks[-1] is not None
                    ):
                        self.first_partial = event
                        self.first_trace = event.trace_id
                        self.partial_visible.set()
                        await self.resume_stream.wait()
                        break
            return p.CompassHookResult.CALL_NEXT

        hooks.on_message_updated.append(updated)
        return hooks

    async def begin_partial_reply(self, ctx: Context) -> p.Session:
        session = await self.session(ctx, "Please explain Atlas's launch date and region.")
        await self.persist(session, "Atlas")
        await self.agent.queue_response(session)
        await asyncio.wait_for(self.partial_visible.wait(), timeout=60)
        assert self.first_partial is not None
        assert isinstance(self.first_partial.data, dict)
        chunks = self.first_partial.data["chunks"]
        assert isinstance(chunks, list) and chunks[-1] is not None
        assert self.update_count == 1
        return session

    async def assert_finished_prefix(self, ctx: Context, session: p.Session) -> None:
        assert self.first_partial is not None
        current = await ctx.container[SessionStore].read_event(session.id, self.first_partial.id)
        assert isinstance(current.data, dict) and isinstance(self.first_partial.data, dict)
        prefix = str(self.first_partial.data["message"])
        assert str(current.data["message"]).startswith(prefix), "Already visible text must remain"
        chunks = current.data.get("chunks")
        assert isinstance(chunks, list) and chunks[-1] is None, (
            "The old stream must terminate, not remain open forever"
        )


class Test_that_a_customer_interrupting_actual_streaming_keeps_a_finished_prefix_and_one_new_reply(
    MidstreamQueuedSDKTest
):
    async def run(self, ctx: Context) -> None:
        session = await self.begin_partial_reply(ctx)
        await self.persist(session, "Boreal")
        try:
            await ctx.client.sessions.create_event(
                session_id=session.id,
                kind="message",
                source="customer",
                message="What date is Boreal launching?",
            )
        finally:
            self.resume_stream.set()
        await self.idle(ctx, session)
        replies = await ctx.container[SessionStore].list_events(
            session.id,
            kinds=[EventKind.MESSAGE],
            source=EventSource.AI_AGENT,
        )
        new_replies = [event for event in replies if event.trace_id != self.first_trace]
        assert len(new_replies) == 1
        assert isinstance(new_replies[0].data, dict)
        assert await nlp_test(
            str(new_replies[0].data["message"]),
            "The reply gives Boreal's launch date as 22 September 2032.",
        )
        await self.assert_finished_prefix(ctx, session)


class Test_that_api_cancellation_during_actual_streaming_keeps_a_finished_prefix(
    MidstreamQueuedSDKTest
):
    async def run(self, ctx: Context) -> None:
        from parlant.core.background_tasks import BackgroundTaskService

        session = await self.begin_partial_reply(ctx)
        coordinator = ctx.container[Application].sessions.response_coordinator
        worker = ctx.container[BackgroundTaskService]._tasks[coordinator.task_tag(session.id)]
        async with httpx.AsyncClient(base_url=ctx.base_url) as http:
            cancellation = asyncio.create_task(
                http.post(
                    f"/sessions/{session.id}/events",
                    json={"kind": "status", "source": "customer", "status": "cancelled"},
                )
            )
            try:
                async with asyncio.timeout(10):
                    # Observe the real service task solely to establish that
                    # HTTP cancellation arrived before releasing the live stream.
                    while not worker.cancelling() and not cancellation.done():
                        await asyncio.sleep(0)
            finally:
                self.resume_stream.set()
            response = await asyncio.wait_for(cancellation, timeout=30)
        response.raise_for_status()
        assert response.json()["data"]["status"] == "cancelled"
        await self.idle(ctx, session)
        await self.assert_finished_prefix(ctx, session)
        replies = await ctx.container[SessionStore].list_events(
            session.id,
            kinds=[EventKind.MESSAGE],
            source=EventSource.AI_AGENT,
        )
        assert len(replies) == 1
        assert coordinator.session_count == 0
