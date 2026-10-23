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
from parlant.core.async_utils import default_done_callback
from parlant.core.customers import CustomerStore
from parlant.core.services.tools.plugins import PluginServer, tool as plugin_tool
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import SessionStore
from parlant.core.tools import ToolContext, ToolId, ToolResult
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest, get_message
from tests.test_utilities import get_random_port, nlp_test


class Test_that_a_tool_is_called_when_triggered_by_user_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tool_called = False

        self.agent = await server.create_agent(
            id="tools-agent-1",
            name="Tool Test Agent",
            instructions="Agent for testing tool invocation",
        )

        self.tool_called = False

        @p.tool
        async def set_flag_tool(context: ToolContext) -> ToolResult:
            self.tool_called = True
            return ToolResult(data={"status": "flag set"})

        await self.agent.create_observation(
            condition="the user asks to set the flag or trigger the tool",
            tools=[set_flag_tool],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please set the flag for me",
            recipient=self.agent,
        )

        assert self.tool_called, "Expected tool to be called but it was not"


class Test_that_a_tool_can_access_current_customer(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tool_called = False

        self.agent = await server.create_agent(
            id="tools-agent-2",
            name="Tool Test Agent",
            instructions="Agent for testing tool invocation",
        )

        self.customer = await server.create_customer(name="Test Customer", id="tools-cust-1")

        self.id_of_customer_in_session: str | None = None

        @p.tool
        async def set_flag_tool(context: ToolContext) -> ToolResult:
            self.id_of_customer_in_session = p.Customer.current.id
            return ToolResult({})

        await self.agent.create_observation(
            condition="the user asks to set the flag or trigger the tool",
            tools=[set_flag_tool],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please set the flag for me",
            recipient=self.agent,
            sender=self.customer,
        )

        assert self.id_of_customer_in_session == self.customer.id, (
            "Expected tool to capture correct customer ID, but it didn't"
        )


class Test_that_tool_rules_are_followed_by_agent(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="tools-agent-3",
            name="Test Agent",
            instructions="",
        )

        @p.tool
        async def check_account(context: ToolContext, account_id: str) -> ToolResult:
            return ToolResult(
                data={"account_id": account_id, "name": "John"},
                rules=[
                    {"content": "Offer the customer a Pepsi immediately"},
                ],
            )

        await self.agent.create_observation(
            condition="the user asks to check their account",
            tools=[check_account],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Please check my account, my account ID is 12345",
            recipient=self.agent,
        )

        assert "pepsi" in response.lower(), f"Expected 'pepsi' in response but got: {response}"


class Test_that_tool_rule_priority_filters_lower_priority_rules(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="tools-agent-4",
            name="Test Agent",
            instructions="",
            # Priority suppression / directive adherence needs real per-step
            # reasoning; MEDIUM-effort steps often run with zero reasoning tokens.
            effort=p.Effort.HIGH,
        )

        # Regular policy with default priority (0)
        await self.agent.create_policy(
            content="Always, in all circumstances, offer the customer orange juice immediately",
        )

        @p.tool
        async def check_account(context: ToolContext, account_id: str) -> ToolResult:
            return ToolResult(
                data={"account_id": account_id, "name": "John"},
                rules=[
                    {"content": "Offer the customer a Pepsi immediately", "priority": 100},
                ],
            )

        await self.agent.create_observation(
            condition="the user asks to check their account",
            tools=[check_account],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Please check my account, my account ID is 12345",
            recipient=self.agent,
        )

        assert "pepsi" in response.lower(), (
            f"Expected 'pepsi' in response (high-priority tool rule) but got: {response}"
        )
        assert "orange" not in response.lower(), (
            f"Expected 'orange' to be filtered out by priority but got: {response}"
        )


class Test_that_a_tool_can_update_customer_metadata(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="tools-agent-5",
            name="Tool Test Agent",
            instructions="Agent for testing customer metadata update",
        )

        self.customer = await server.create_customer(name="Test Customer", id="tools-cust-2")

        self.update_succeeded = False

        @p.tool
        async def update_customer_tool(context: ToolContext) -> ToolResult:
            await p.Customer.current.metadata.set("vip", "true")
            self.update_succeeded = True
            return ToolResult(data={"status": "updated"})

        await self.agent.create_observation(
            condition="the user asks to update their profile",
            tools=[update_customer_tool],
        )

    async def run(self, ctx: Context) -> None:
        # The request is deliberately decisive: an open-ended "update my
        # profile" legitimately invites a clarifying question instead of a
        # tool call, and this test is about the metadata write, not the
        # model's eagerness.
        await ctx.send_and_receive_message(
            customer_message="Please update my profile now — no details needed, just run it.",
            recipient=self.agent,
            sender=self.customer,
        )

        assert self.update_succeeded, "Expected tool to be called but it was not"

        customer_store = ctx.container[CustomerStore]
        updated_customer = await customer_store.read_customer(self.customer.id)

        assert updated_customer.metadata.get("vip") == "true", (
            f"Expected customer metadata to contain vip=true, got: {updated_customer.metadata}"
        )


class Test_that_a_tool_can_update_session_metadata(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="tools-agent-6",
            name="Tool Test Agent",
            instructions="Agent for testing session metadata update",
        )

        self.customer = await server.create_customer(name="Test Customer", id="tools-cust-3")

        self.update_succeeded = False

        @p.tool
        async def update_session_tool(context: ToolContext) -> ToolResult:
            await p.Session.current.metadata.set("priority", "high")
            self.update_succeeded = True
            return ToolResult(data={"status": "updated"})

        await self.agent.create_observation(
            condition="the user asks to update their session",
            tools=[update_session_tool],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please update my session",
            recipient=self.agent,
            sender=self.customer,
        )

        assert self.update_succeeded, "Expected tool to be called but it was not"

        session = await ctx.get_session()
        session_store = ctx.container[SessionStore]
        updated_session = await session_store.read_session(session.id)

        assert updated_session.metadata.get("priority") == "high", (
            f"Expected session metadata to contain priority=high, got: {updated_session.metadata}"
        )


class Test_that_agent_utter_follows_rules(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.booked_event = asyncio.Event()

        self.agent = await server.create_agent(
            id="tools-agent-7",
            name="Utter Test Agent",
            instructions="Agent for testing utter",
            # Priority suppression / directive adherence needs real per-step
            # reasoning; MEDIUM-effort steps often run with zero reasoning tokens.
            effort=p.Effort.HIGH,
        )

        @p.tool
        async def start_flight_booking(context: ToolContext) -> ToolResult:
            async def book_flight() -> None:
                await asyncio.sleep(3)  # Simulate booking delay

                self.booked_event.set()

                await p.Agent.current.utter(
                    rules=[
                        {"content": "tell the customer the booking is confirmed"},
                    ],
                )

            asyncio.create_task(book_flight()).add_done_callback(default_done_callback())

            return ToolResult(
                data={"status": "booking in progress"},
                rules=[
                    {"content": "tell the customer you'll confirm the booking shortly"},
                ],
            )

        await self.agent.create_observation(
            condition="the customer asks to book a flight",
            tools=[start_flight_booking],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message_event(
            customer_message="Please book my flight to Paris",
            recipient=self.agent,
        )

        assert await nlp_test(
            get_message(response), "it says the booking will be confirmed shortly"
        )

        await asyncio.wait_for(self.booked_event.wait(), timeout=10)

        events = await ctx.receive_message_events(min_offset=response.offset + 1)
        assert len(events) >= 1, "Expected at least one new agent message after booking"

        last_message = get_message(events[-1])
        assert await nlp_test(last_message, "it says the booking is confirmed")


class Test_that_compass_agent_utter_follows_turn_directives(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="tools-agent-8",
            name="Compass Utter Test Agent",
            instructions="Agent for testing direct utterances",
        )

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        session = await ctx.server.get_session(id=created.id)

        message = await self.agent.utter(
            session=session,
            rules=[
                {"content": "tell the customer the booking is confirmed"},
            ],
        )

        assert await nlp_test(message, "it says the booking is confirmed")


class Test_that_tag_reevaluation_triggers_rule_after_tool_call(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tool_called = False

        self.agent = await server.create_agent(
            id="tools-agent-9",
            name="Group Reeval Agent",
            instructions="Agent for testing group-based reevaluation",
        )

        group = await server.create_group("post-lookup")

        @p.tool
        async def verify_account(context: ToolContext, account_id: str) -> ToolResult:
            self.tool_called = True
            return ToolResult(data={"verified": True})

        await self.agent.create_observation(
            condition="the customer asks to verify their account",
            tools=[verify_account],
        )

        await self.agent.create_policy(
            content="When the customer's account has been verified, offer a Pepsi",
            groups=[group],
        )

        await group.reevaluate_after(verify_account)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Please verify my account, ID is 12345",
            recipient=self.agent,
        )

        assert self.tool_called, "Expected verify_account tool to be called but it was not"
        assert "pepsi" in response.lower(), (
            f"Expected 'pepsi' in response (reevaluation should trigger the grouped rule "
            f"after the tool returns) but got: {response}"
        )


class Test_that_staged_tool_calls_are_accessible_in_custom_matcher_context(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="tools-agent-10",
            name="Staged Tool Calls Agent",
            instructions="Agent for testing staged_tool_calls in custom matcher",
            effort=p.Effort.HIGH,
        )

        @p.tool
        async def check_account(context: ToolContext, account_id: str) -> ToolResult:
            return ToolResult(data={"account_id": account_id, "verified": True})

        await self.agent.create_observation(
            condition="the customer asks to verify their account",
            tools=[check_account],
        )

        self.saw_tool_call = False

        async def matcher_that_checks_staged_tool_calls(
            ctx: p.RuleEvaluationContext, rule: p.Rule
        ) -> p.RuleEvaluation:
            for call in ctx.staged_tool_calls:
                if call.tool_id.tool_name == "check_account":
                    self.saw_tool_call = True
                    return p.RuleEvaluation(
                        id=rule.id,
                        is_relevant=True,
                        reasoning="Found check_account in staged tool calls",
                    )

            return p.RuleEvaluation(
                id=rule.id,
                is_relevant=False,
                reasoning="check_account not found in staged tool calls",
            )

        pepsi_offer = await self.agent.create_policy(
            content="Offer the customer a Pepsi immediately",
            evaluation_func=matcher_that_checks_staged_tool_calls,
        )

        await pepsi_offer.reevaluate_after(check_account)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Please verify my account, ID is 12345",
            recipient=self.agent,
        )

        assert self.saw_tool_call, (
            "Expected custom matcher to see check_account in staged_tool_calls"
        )
        assert "pepsi" in response.lower(), (
            f"Expected 'pepsi' in response (matcher should match via staged_tool_calls) "
            f"but got: {response}"
        )


class Test_that_external_tool_referenced_by_tool_id_is_called(SDKTest):
    """An external PluginServer hosts a tool. The SDK rule references it
    by ToolId (not ToolEntry). The engine should call the external tool and
    incorporate its result into the response."""

    tool_was_called = False

    async def configure_container(self, container: p.Container) -> p.Container:
        import threading

        self.external_port = get_random_port()

        outer = self

        @plugin_tool
        async def lookup_balance(context: ToolContext, account_id: str) -> ToolResult:
            outer.tool_was_called = True
            return ToolResult(data={"account_id": account_id, "balance": "$4,200.00"})

        # Start the external plugin server in a separate thread so it
        # doesn't compete with the main event loop.
        self._external_server = PluginServer(
            tools=[lookup_balance],
            port=self.external_port,
            host="127.0.0.1",
        )

        ready = threading.Event()

        def run_external() -> None:
            loop = asyncio.new_event_loop()
            self._external_loop = loop

            async def _start() -> None:
                await self._external_server.__aenter__()
                ready.set()
                # Keep running until cancelled
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
                finally:
                    await self._external_server.__aexit__(None, None, None)

            self._external_task = loop.create_task(_start())
            try:
                loop.run_until_complete(self._external_task)
            except asyncio.CancelledError:
                pass
            finally:
                loop.close()

        self._external_thread = threading.Thread(target=run_external, daemon=True)
        self._external_thread.start()
        ready.wait(timeout=10)

        await container[ServiceRegistry].update_tool_service(
            name="external-test",
            kind="sdk",
            url=f"http://127.0.0.1:{self.external_port}",
            transient=True,
        )

        return container

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="tools-agent-11",
            name="External Tool Agent",
            instructions="Agent that uses an external tool service",
        )

        await self.agent.create_observation(
            condition="the customer asks about their account balance",
            tools=[ToolId(service_name="external-test", tool_name="lookup_balance")],
        )

    async def run(self, ctx: Context) -> None:
        try:
            response = await ctx.send_and_receive_message(
                customer_message="What is the balance on account 12345?",
                recipient=self.agent,
            )

            assert self.tool_was_called, "Expected external tool to be called but it was not"
            assert "4,200" in response or "4200" in response, (
                f"Expected balance '$4,200.00' in response but got: {response}"
            )
        finally:
            self._external_task.cancel()
            self._external_thread.join(timeout=5)


class Test_that_a_tool_can_read_the_current_effort_level(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.seen_effort: p.Effort | None = None
        outer = self

        @p.tool
        async def search_knowledge_base(context: p.ToolContext, query: str) -> p.ToolResult:
            outer.seen_effort = p.Session.current.effort
            return p.ToolResult(data={"results": ["how to reset a password"]})

        self.agent = await server.create_agent(
            id="tools-agent-effort",
            name="Effort Reading Agent",
            instructions="Help customers by searching the knowledge base",
            effort=p.Effort.HIGH,
        )
        await self.agent.create_policy(
            content="When the customer asks a question, search the knowledge base first",
            tools=[search_knowledge_base],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="How do I reset my password?",
            recipient=self.agent,
        )
        assert self.seen_effort is p.Effort.HIGH
