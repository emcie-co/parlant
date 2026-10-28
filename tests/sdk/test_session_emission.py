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

"""Tests for emitting events into a session through the SDK Session class."""

import asyncio
from typing import Any

import parlant.sdk as p
from parlant.core.engines.compass.context import EngineContext

from tests.sdk.utils import Context, SDKTest


class Test_that_on_matched_can_emit_a_hardcoded_message_and_go_manual(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="emission-agent-1",
            name="Handoff Agent",
            instructions="Agent for session emission tests",
        )

        async def handoff(context: EngineContext, evaluation: Any) -> None:
            await p.Session.current.emit_message(
                "A human agent will take over this conversation shortly."
            )

            await p.Session.current.update(mode="manual")

        await self.agent.create_observation(
            condition="Always",
            evaluation_func=p.MATCH_ALWAYS,
            on_matched=handoff,
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Hello there",
        )

        events = await ctx.client.sessions.list_events(
            session_id=session.id,
            source="ai_agent",
            kinds="message",
            min_offset=0,
            wait_for_data=30,
        )
        messages = [e.model_dump().get("data", {}).get("message") for e in events]
        assert "A human agent will take over this conversation shortly." in messages

        refreshed = await ctx.client.sessions.retrieve(session_id=session.id)
        assert refreshed.mode == "manual"


class Test_that_an_admin_can_emit_events_into_a_fetched_session(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.server = server
        self.agent = await server.create_agent(
            id="emission-agent-2",
            name="Emission Agent",
            instructions="Agent for admin emission tests",
        )

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)

        session = await self.server.get_session(id=created.id)
        await session.emit_message(
            "A human operator has joined the conversation.",
            source=p.EventSource.HUMAN_AGENT,
        )
        await session.emit_status("processing")
        await session.emit_custom({"widget": "handoff-banner", "urgency": "high"})

        events = await ctx.client.sessions.list_events(session_id=created.id, min_offset=0)
        dumped = [e.model_dump() for e in events]

        message_events = [e for e in dumped if e["kind"] == "message"]
        assert any(
            e["data"].get("message") == "A human operator has joined the conversation."
            and e["source"] == "human_agent"
            for e in message_events
        )

        status_events = [e for e in dumped if e["kind"] == "status"]
        assert any(e["data"].get("status") == "processing" for e in status_events)

        custom_events = [e for e in dumped if e["kind"] == "custom"]
        assert any(
            e["data"] == {"widget": "handoff-banner", "urgency": "high"} for e in custom_events
        )


class Test_that_a_message_a_tool_emits_carries_the_turn_trace(SDKTest):
    async def setup(self, server: p.Server) -> None:
        @p.tool
        async def check_order_status(order_number: str) -> p.ToolResult:
            """Check the status of an order."""
            await p.Session.current.emit_message("One moment, checking that order.")
            return p.ToolResult(data=f"Order {order_number} shipped yesterday.")

        self.agent = await server.create_agent(
            id="emission-agent-3",
            name="Store Agent",
            instructions="A store's support agent.",
        )
        await self.agent.create_policy(
            title="Order status",
            content="When the customer asks about an order, check its status with the tool.",
            tools=[check_order_status],
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        customer_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Where is my order 4471?",
        )

        emitted = await _wait_for_message(ctx, session.id, "One moment, checking that order.")
        turn_trace_id = (
            await ctx.client.sessions.list_events(
                session_id=session.id, kinds="tool", min_offset=0, wait_for_data=60
            )
        )[0].trace_id

        assert customer_event.trace_id == turn_trace_id
        assert emitted.trace_id == turn_trace_id


class Test_that_a_message_a_tool_emits_from_a_background_task_carries_the_turn_trace(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.background: set[asyncio.Task[None]] = set()

        @p.tool
        async def start_order_lookup(order_number: str) -> p.ToolResult:
            """Start looking up an order; the result is sent later."""
            session = p.Session.current

            async def later() -> None:
                await asyncio.sleep(1)
                await session.emit_message("Still checking that order.")

            task = asyncio.ensure_future(later())
            self.background.add(task)
            task.add_done_callback(self.background.discard)
            return p.ToolResult(data="The lookup has started.")

        self.agent = await server.create_agent(
            id="emission-agent-4",
            name="Store Agent",
            instructions="A store's support agent.",
        )
        await self.agent.create_policy(
            title="Order lookup",
            content="When the customer asks about an order, start a lookup with the tool.",
            tools=[start_order_lookup],
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        customer_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Where is my order 4471?",
        )

        emitted = await _wait_for_message(ctx, session.id, "Still checking that order.")

        assert emitted.trace_id == customer_event.trace_id


class Test_that_a_message_emitted_outside_any_turn_gets_a_trace_of_its_own(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.server = server
        self.agent = await server.create_agent(
            id="emission-agent-5",
            name="Emission Agent",
            instructions="Agent for admin emission tests",
        )

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)

        session = await self.server.get_session(id=created.id)
        await session.emit_message("We'll be back online at 9am.")

        emitted = await _wait_for_message(ctx, created.id, "We'll be back online at 9am.")

        assert emitted.trace_id
        assert emitted.trace_id != "<main>"


async def _wait_for_message(ctx: Context, session_id: str, text: str) -> Any:
    for _ in range(90):
        events = await ctx.client.sessions.list_events(
            session_id=session_id, source="ai_agent", kinds="message", min_offset=0
        )
        for event in events:
            if event.model_dump().get("data", {}).get("message") == text:
                return event
        await asyncio.sleep(1)
    raise AssertionError(f"no message {text!r} was emitted")
