# Copyright 2025 Emcie Co Ltd.
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


class Test_that_a_variable_value_can_be_set(SDKTest):
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
        await self.variable.set_value(key="value", value="premium")
