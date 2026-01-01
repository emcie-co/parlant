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
import time
from typing import Any
from parlant.core.capabilities import CapabilityStore
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.services.tools.plugins import tool
from parlant.core.tags import Tag
from parlant.core.tools import ToolContext, ToolResult
from parlant.core.canned_responses import CannedResponseStore
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test


class Test_that_an_agent_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        await server.create_agent(
            name="Test Agent",
            description="This is a test agent",
            composition_mode=p.CompositionMode.COMPOSITED,
        )

    async def run(self, ctx: Context) -> None:
        agents = await ctx.client.agents.list()
        assert agents[0].name == "Test Agent"


class Test_that_a_capability_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="This is a test agent",
        )

        self.capability = await self.agent.experimental_features.create_capability(
            title="Test Capability",
            description="Some Description",
            signals=["First Query", "Second Query"],
        )

    async def run(self, ctx: Context) -> None:
        capabilities = await ctx.container[CapabilityStore].list_capabilities()

        assert len(capabilities) == 1
        capability = capabilities[0]

        assert capability.id == self.capability.id
        assert capability.title == self.capability.title
        assert capability.description == self.capability.description
        assert capability.signals == self.capability.signals


class Test_that_an_agent_can_be_read_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="ReadById Agent",
            description="Agent to be read by ID",
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


class Test_that_an_agent_can_create_canned_response(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Canned Response Agent",
            description="Agent for canned response test",
        )
        self.canrep_id = await self.agent.create_canned_response(
            template="Hello, {user}!", tags=[Tag.for_agent_id(self.agent.id)]
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]

        canrep = await canrep_store.read_canned_response(canned_response_id=self.canrep_id)

        assert canrep.value == "Hello, {user}!"
        assert Tag.for_agent_id(self.agent.id) in canrep.tags


class Test_that_agents_can_be_listed(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.a1 = await server.create_agent(
            name="List Agent 1",
            description="First agent for listing",
        )

        self.a2 = await server.create_agent(
            name="List Agent 2",
            description="Second agent for listing",
        )

    async def run(self, ctx: Context) -> None:
        agents = await ctx.server.list_agents()

        assert self.a1 in agents
        assert self.a2 in agents


class Test_that_an_agent_can_be_found_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.a1 = await server.create_agent(
            name="List Agent 1",
            description="First agent for listing",
        )

    async def run(self, ctx: Context) -> None:
        assert await ctx.server.find_agent(id=self.a1.id) == self.a1
        assert await ctx.server.find_agent(id="nonexistent") is None


class Test_that_an_agent_can_be_found_using_tool_context(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Tool Context Agent",
            description="Agent for tool context test",
        )

        @p.tool
        async def check_what_is_spatio(context: ToolContext) -> ToolResult:
            agent = await p.ToolContextAccessor(context).server.find_agent(id=context.agent_id)

            if agent is None:
                return ToolResult("A spatio is a special type of spaghetti spoon.")
            else:
                return ToolResult("Spatio is the name of a famous fictional mouse.")

        await self.agent.attach_tool(check_what_is_spatio, condition="the user asks about spatio")

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="What is spatio?",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It says that spatio is the name of a mouse.")


class Test_that_the_output_of_an_agent_can_be_intercepted(SDKTest):
    # This test shows that you can intercept the agent's generated message before
    # it reaches the customer. This can be extremely important for last-minute validations.

    async def configure_hooks(self, hooks: p.EngineHooks) -> p.EngineHooks:
        async def intercept_message(
            ctx: p.EngineContext, payload: Any, exc: Exception | None
        ) -> p.EngineHookResult:
            _ = payload  # Here is where validations would run (payload is the generated message)

            await ctx.session_event_emitter.emit_message_event(
                trace_id=ctx.tracer.trace_id,
                data="Bananas! More bananas!",
            )

            # Reject the generated message
            return p.EngineHookResult.BAIL

        hooks.on_message_generated.append(intercept_message)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(name="Dummy Agent", description="")

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(customer_message="Hello", recipient=self.agent)
        assert answer == "Bananas! More bananas!"


class Test_that_an_agent_can_be_created_with_custom_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="my-custom-agent-id",
            name="Custom ID Agent",
            description="This agent has a custom ID",
        )

    async def run(self, ctx: Context) -> None:
        assert self.agent.id == "my-custom-agent-id"

        # Verify the agent can be retrieved with the custom ID
        retrieved_agent = await ctx.server.find_agent(id="my-custom-agent-id")
        assert retrieved_agent is not None
        assert retrieved_agent.id == "my-custom-agent-id"
        assert retrieved_agent.name == "Custom ID Agent"


class Test_that_an_agent_with_basic_policy_sends_preamble_and_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        from parlant.core.engines.alpha.perceived_performance_policy import (
            BasicPerceivedPerformancePolicy,
        )

        self.agent = await server.create_agent(
            name="Basic Policy Agent",
            description="Agent with basic perceived performance policy",
            perceived_performance_policy=BasicPerceivedPerformancePolicy(),
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
            message="Hello",
        )

        # Poll for messages until we get 2 messages (or timeout after 30 seconds)
        start_time = time.time()
        agent_messages: list[Any] = []
        while len(agent_messages) < 2:
            if time.time() - start_time > 30:
                raise TimeoutError(
                    f"Timeout waiting for 2 messages. Got {len(agent_messages)} messages."
                )

            agent_messages = await ctx.client.sessions.list_events(
                session_id=session.id,
                min_offset=customer_event.offset,
                source="ai_agent",
                kinds="message",
                wait_for_data=5,
            )

            if len(agent_messages) < 2:
                await asyncio.sleep(0.5)

        # With BasicPerceivedPerformancePolicy, we expect 2 messages:
        # 1. A preamble message (tagged with preamble tag)
        # 2. The actual response message
        assert len(agent_messages) == 2

        # Check that the first message is a preamble
        first_message_data = agent_messages[0].model_dump().get("data", {})
        first_message_tags = first_message_data.get("tags", [])
        assert any("preamble" in str(tag) for tag in first_message_tags)

        # Check that the second message is the actual response
        second_message_data = agent_messages[1].model_dump().get("data", {})
        assert second_message_data.get("message") is not None


class Test_that_an_agent_with_null_policy_sends_only_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        from parlant.core.engines.alpha.perceived_performance_policy import (
            NullPerceivedPerformancePolicy,
        )

        self.agent = await server.create_agent(
            name="Null Policy Agent",
            description="Agent with null perceived performance policy",
            perceived_performance_policy=NullPerceivedPerformancePolicy(),
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
            message="Hello",
        )

        agent_messages = await ctx.client.sessions.list_events(
            session_id=session.id,
            min_offset=customer_event.offset,
            source="ai_agent",
            kinds="message",
            wait_for_data=30,
        )

        # With NullPerceivedPerformancePolicy, we expect only 1 message:
        # The actual response (no preamble)
        assert len(agent_messages) == 1

        # Check that the message is the actual response (not a preamble)
        message_data = agent_messages[0].model_dump().get("data", {})
        message_tags = message_data.get("tags", [])
        assert not any("preamble" in str(tag) for tag in message_tags)
