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

from parlant.core.tools import ToolContext, ToolResult
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest


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
