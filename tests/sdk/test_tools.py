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
from parlant.core.sessions import SessionStore
from parlant.core.tools import ToolContext, ToolResult
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest, get_message
from tests.test_utilities import nlp_test


class Test_that_a_tool_is_called_when_triggered_by_user_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tool_called = False

        self.agent = await server.create_agent(
            name="Tool Test Agent",
            description="Agent for testing tool invocation",
        )

        self.tool_called = False

        @p.tool
        async def set_flag_tool(context: ToolContext) -> ToolResult:
            self.tool_called = True
            return ToolResult(data={"status": "flag set"})

        await self.agent.attach_tool(
            tool=set_flag_tool,
            condition="the user asks to set the flag or trigger the tool",
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
            name="Tool Test Agent",
            description="Agent for testing tool invocation",
        )

        self.customer = await server.create_customer(name="Test Customer")

        self.id_of_customer_in_session: str | None = None

        @p.tool
        async def set_flag_tool(context: ToolContext) -> ToolResult:
            self.id_of_customer_in_session = p.Customer.current.id
            return ToolResult({})

        await self.agent.attach_tool(
            tool=set_flag_tool,
            condition="the user asks to set the flag or trigger the tool",
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


class Test_that_tool_guidelines_are_followed_by_agent(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="",
        )

        @p.tool
        async def check_account(context: ToolContext, account_id: str) -> ToolResult:
            return ToolResult(
                data={"account_id": account_id, "name": "John"},
                guidelines=[
                    {"action": "Offer the customer a Pepsi immediately"},
                ],
            )

        await self.agent.attach_tool(
            tool=check_account,
            condition="the user asks to check their account",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Please check my account, my account ID is 12345",
            recipient=self.agent,
        )

        assert "pepsi" in response.lower(), f"Expected 'pepsi' in response but got: {response}"


class Test_that_tool_guideline_priority_filters_lower_priority_guidelines(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="",
        )

        # Regular guideline with default priority (0)
        await self.agent.create_guideline(
            condition="a]ways, in all circumstances",
            action="Offer the customer orange juice immediately",
        )

        @p.tool
        async def check_account(context: ToolContext, account_id: str) -> ToolResult:
            return ToolResult(
                data={"account_id": account_id, "name": "John"},
                guidelines=[
                    {"action": "Offer the customer a Pepsi immediately", "priority": 100},
                ],
            )

        await self.agent.attach_tool(
            tool=check_account,
            condition="the user asks to check their account",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Please check my account, my account ID is 12345",
            recipient=self.agent,
        )

        assert "pepsi" in response.lower(), (
            f"Expected 'pepsi' in response (high-priority tool guideline) but got: {response}"
        )
        assert "orange" not in response.lower(), (
            f"Expected 'orange' to be filtered out by priority but got: {response}"
        )


class Test_that_a_tool_can_update_customer_metadata(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Tool Test Agent",
            description="Agent for testing customer metadata update",
        )

        self.customer = await server.create_customer(name="Test Customer")

        self.update_succeeded = False

        @p.tool
        async def update_customer_tool(context: ToolContext) -> ToolResult:
            await p.Customer.current.metadata.set("vip", "true")
            self.update_succeeded = True
            return ToolResult(data={"status": "updated"})

        await self.agent.attach_tool(
            tool=update_customer_tool,
            condition="the user asks to update their profile",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please update my profile",
            recipient=self.agent,
            sender=self.customer,
        )

        assert self.update_succeeded, "Expected tool to be called but it was not"

        customer_store = ctx.container[CustomerStore]
        updated_customer = await customer_store.read_customer(self.customer.id)

        assert updated_customer.extra.get("vip") == "true", (
            f"Expected customer metadata to contain vip=true, got: {updated_customer.extra}"
        )


class Test_that_a_tool_can_update_session_metadata(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Tool Test Agent",
            description="Agent for testing session metadata update",
        )

        self.customer = await server.create_customer(name="Test Customer")

        self.update_succeeded = False

        @p.tool
        async def update_session_tool(context: ToolContext) -> ToolResult:
            await p.Session.current.metadata.set("priority", "high")
            self.update_succeeded = True
            return ToolResult(data={"status": "updated"})

        await self.agent.attach_tool(
            tool=update_session_tool,
            condition="the user asks to update their session",
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


class Test_that_agent_utter_follows_guidelines(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.booked_event = asyncio.Event()

        self.agent = await server.create_agent(
            name="Utter Test Agent",
            description="Agent for testing utter",
        )

        @p.tool
        async def start_flight_booking(context: ToolContext) -> ToolResult:
            session = p.Session.current

            async def book_flight() -> None:
                await asyncio.sleep(3)  # Simulate booking delay

                self.booked_event.set()

                await self.agent.utter(
                    session=session,
                    guidelines=[
                        {"action": "tell the customer the booking is confirmed"},
                    ],
                )

            asyncio.create_task(book_flight()).add_done_callback(default_done_callback())

            return ToolResult(
                data={"status": "booking in progress"},
                guidelines=[
                    {"action": "tell the customer you'll confirm the booking shortly"},
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


class Test_that_tag_reevaluation_triggers_guideline_after_tool_call(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tool_called = False

        self.agent = await server.create_agent(
            name="Tag Reeval Agent",
            description="Agent for testing tag-based reevaluation",
        )

        tag = await server.create_tag("post-lookup")

        @p.tool
        async def verify_account(context: ToolContext, account_id: str) -> ToolResult:
            self.tool_called = True
            return ToolResult(data={"verified": True})

        await self.agent.create_observation(
            condition="the customer asks to verify their account",
            tools=[verify_account],
        )

        await self.agent.create_guideline(
            condition="the customer's account has been verified",
            action="Offer a Pepsi",
            tags=[tag],
        )

        await tag.reevaluate_after(verify_account)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Please verify my account, ID is 12345",
            recipient=self.agent,
        )

        assert self.tool_called, "Expected verify_account tool to be called but it was not"
        assert "pepsi" in response.lower(), (
            f"Expected 'pepsi' in response (reevaluation should trigger the tagged guideline "
            f"after the tool returns) but got: {response}"
        )


class Test_that_staged_tool_calls_are_accessible_in_custom_matcher_context(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Staged Tool Calls Agent",
            description="Agent for testing staged_tool_calls in custom matcher",
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
            ctx: p.GuidelineMatchingContext, guideline: p.Guideline
        ) -> p.GuidelineMatch:
            for call in ctx.staged_tool_calls:
                if call.tool_id.tool_name == "check_account":
                    self.saw_tool_call = True
                    return p.GuidelineMatch(
                        id=guideline.id,
                        matched=True,
                        rationale="Found check_account in staged tool calls",
                    )

            return p.GuidelineMatch(
                id=guideline.id,
                matched=False,
                rationale="check_account not found in staged tool calls",
            )

        pepsi_offer = await self.agent.create_guideline(
            action="Offer the customer a Pepsi immediately",
            matcher=matcher_that_checks_staged_tool_calls,
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
