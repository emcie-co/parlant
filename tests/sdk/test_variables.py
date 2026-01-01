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

from parlant.core.context_variables import ContextVariableStore
from parlant.core.tools import ToolId
import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest


class Test_that_a_static_value_variable_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.variable = await self.agent.create_variable(
            name="subscription_plan",
            description="The current subscription plan of the user.",
        )

    async def run(self, ctx: Context) -> None:
        variable_store = ctx.container[ContextVariableStore]

        variable = await variable_store.read_variable(self.variable.id)

        assert variable.name == "subscription_plan"
        assert variable.description == "The current subscription plan of the user."
        assert variable.id == self.variable.id


class Test_that_a_tool_enabled_variable_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        @p.tool
        async def get_value(context: p.ToolContext) -> p.ToolResult:
            return p.ToolResult("premium")

        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.variable = await self.agent.create_variable(
            name="subscription_plan",
            description="The current subscription plan of the user.",
            tool=get_value,
        )

    async def run(self, ctx: Context) -> None:
        variable_store = ctx.container[ContextVariableStore]

        variable = await variable_store.read_variable(self.variable.id)

        assert variable.name == "subscription_plan"
        assert variable.description == "The current subscription plan of the user."
        assert variable.id == self.variable.id
        assert variable.tool_id == ToolId(p.INTEGRATED_TOOL_SERVICE_NAME, "get_value")


class Test_that_a_variable_value_can_be_set_for_a_customer(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.customer = await server.create_customer("John Doe")

        self.variable = await self.agent.create_variable(
            name="subscription_plan",
            description="The current subscription plan of the user.",
        )

        await self.variable.set_value_for_customer(self.customer, "premium")

    async def run(self, ctx: Context) -> None:
        assert "premium" == await self.variable.get_value_for_customer(self.customer)


class Test_that_a_variable_value_can_be_set_for_a_tag(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.tag = await server.create_tag("premium_users")

        self.variable = await self.agent.create_variable(
            name="subscription_plan",
            description="The current subscription plan of the user.",
        )

        await self.variable.set_value_for_tag(self.tag.id, "premium")

    async def run(self, ctx: Context) -> None:
        assert "premium" == await self.variable.get_value_for_tag(self.tag.id)


class Test_that_a_variable_value_can_be_set_globally(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.variable = await self.agent.create_variable(
            name="subscription_plan",
            description="The current subscription plan of the user.",
        )

        await self.variable.set_global_value("premium")

    async def run(self, ctx: Context) -> None:
        assert "premium" == await self.variable.get_global_value()


class Test_that_variables_can_be_listed(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.variable = await self.agent.create_variable(
            name="subscription_plan",
            description="The current subscription plan of the user.",
        )

    async def run(self, ctx: Context) -> None:
        variables = await self.agent.list_variables()

        assert self.variable in variables


class Test_that_a_variable_can_be_found_by_name(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.variable = await self.agent.create_variable(
            name="subscription_plan",
            description="The current subscription plan of the user.",
        )

    async def run(self, ctx: Context) -> None:
        assert await self.agent.find_variable(name="subscription_plan") == self.variable
        assert await self.agent.find_variable(name="nonexistent") is None


class Test_that_a_variable_can_be_found_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.variable = await self.agent.create_variable(
            name="subscription_plan",
            description="The current subscription plan of the user.",
        )

    async def run(self, ctx: Context) -> None:
        assert await self.agent.find_variable(id=self.variable.id) == self.variable
