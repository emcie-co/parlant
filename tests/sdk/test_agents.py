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

from parlant.core.capabilities import CapabilityStore
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.services.tools.plugins import tool
from parlant.core.tags import Tag
from parlant.core.tools import ToolContext, ToolResult
from parlant.core.utterances import UtteranceStore
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest


class Test_that_an_agent_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        await server.create_agent(
            name="Test Agent",
            description="This is a test agent",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

    async def run(self, ctx: Context) -> None:
        agents = await ctx.client.agents.list()
        assert agents[0].name == "Test Agent"


class Test_that_a_capability_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="This is a test agent",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.capability = await self.agent.create_capability(
            title="Test Capability",
            description="Some Description",
            queries=["First Query", "Second Query"],
        )

    async def run(self, ctx: Context) -> None:
        capabilities = await ctx.container[CapabilityStore].list_capabilities()

        assert len(capabilities) == 1
        capability = capabilities[0]

        assert capability.id == self.capability.id
        assert capability.title == self.capability.title
        assert capability.description == self.capability.description
        assert capability.queries == self.capability.queries


class Test_that_an_agent_can_be_read_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="ReadById Agent",
            description="Agent to be read by ID",
            composition_mode=p.CompositionMode.FLUID,
        )

    async def run(self, ctx: Context) -> None:
        agent = await ctx.client.agents.retrieve(self.agent.id)
        assert agent.name == "ReadById Agent"
        assert agent.description == "Agent to be read by ID"


class Test_that_an_agent_can_create_guideline(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Guideline Agent",
            description="Agent for guideline test",
            composition_mode=p.CompositionMode.FLUID,
        )
        self.guideline = await self.agent.create_guideline(
            condition="Always say hello", action="Say hello to the user"
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]

        guideline = await guideline_store.read_guideline(guideline_id=self.guideline.id)

        assert guideline.content.condition == "Always say hello"
        assert guideline.content.action == "Say hello to the user"
        assert guideline.tags == [Tag.for_agent_id(self.agent.id)]


class Test_that_an_agent_can_attach_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Tool Agent",
            description="Agent for tool test",
            composition_mode=p.CompositionMode.FLUID,
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.guideline_id = await self.agent.attach_tool(
            tool=test_tool, condition="If user asks for dummy tool"
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]
        guideline_tooL_store = ctx.container[GuidelineToolAssociationStore]

        guideline = await guideline_store.read_guideline(guideline_id=self.guideline_id)

        assert guideline.content.condition == "If user asks for dummy tool"

        associations = await guideline_tooL_store.list_associations()
        assert associations
        assert len(associations) == 1

        association = associations[0]
        assert association.guideline_id == guideline.id


class Test_that_an_agent_can_create_utterance(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Utterance Agent",
            description="Agent for utterance test",
            composition_mode=p.CompositionMode.FLUID,
        )
        self.utterance_id = await self.agent.create_utterance(
            template="Hello, {user}!", tags=[Tag.for_agent_id(self.agent.id)]
        )

    async def run(self, ctx: Context) -> None:
        utterance_store = ctx.container[UtteranceStore]

        utterance = await utterance_store.read_utterance(utterance_id=self.utterance_id)

        assert utterance.value == "Hello, {user}!"
        assert Tag.for_agent_id(self.agent.id) in utterance.tags
