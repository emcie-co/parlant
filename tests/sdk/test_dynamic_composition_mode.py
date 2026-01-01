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

import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest


class Test_that_guideline_composition_mode_overrides_agent_default(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for testing dynamic composition mode",
        )

        # Create canned responses
        canrep_id = await self.agent.create_canned_response(
            template="I can help you with that specific request.",
        )

        # Create guideline with STRICT composition mode
        self.guideline = await self.agent.create_guideline(
            condition="Customer asks for help",
            action="Help the customer",
            composition_mode=p.CompositionMode.STRICT,
            canned_responses=[canrep_id],
        )

    async def run(self, ctx: Context) -> None:
        # Send message that matches the guideline
        response = await ctx.send_and_receive_message(
            customer_message="I need help with something",
            recipient=self.agent,
        )

        assert response == "I can help you with that specific request."


class Test_that_journey_level_composition_mode_affects_all_states(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Test agent",
        )

        # Create journey with COMPOSITED composition mode at journey level
        self.journey = await self.agent.create_journey(
            title="Support Journey",
            description="A journey for customer support",
            conditions=["Customer seeks support"],
            composition_mode=p.CompositionMode.COMPOSITED,
        )

        # Create canned response
        await self.journey.create_canned_response(
            template="Willkommen!!! How might I serve thee today good sir!?!?!?",
        )

        # Create initial state with canned response
        self.initial_state = await self.journey.initial_state.transition_to(
            chat_state="Greet the customer",
        )

    async def run(self, ctx: Context) -> None:
        # Activate journey
        response = await ctx.send_and_receive_message(
            customer_message="Hi, I want support",
            recipient=self.agent,
        )

        assert "!!!" in response.lower() or "!?" in response.lower()


class Test_that_journey_node_composition_mode_overrides_journey_level(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Test agent",
        )

        # Create journey with FLUID composition mode
        self.journey = await self.agent.create_journey(
            title="Food order",
            description="Journey for ordering food",
            conditions=["Customer wants to order food"],
            composition_mode=p.CompositionMode.FLUID,
        )

        # Create chat state with STRICT composition mode override
        self.strict_state = await self.journey.initial_state.transition_to(
            chat_state="Ask what kind of food the customer wants",
            composition_mode=p.CompositionMode.STRICT,
            canned_responses=[
                await self.agent.create_canned_response(
                    template="What delicacy would you like to order today?",
                )
            ],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="I want to order food",
            recipient=self.agent,
            reuse_session=False,
        )

        assert response == "What delicacy would you like to order today?"


class Test_that_most_restrictive_composition_mode_wins(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Test agent",
        )

        await self.agent.create_canned_response(
            template="Would you like a banana?",
        )

        # Create guideline with FLUID composition mode
        self.guideline_fluid = await self.agent.create_guideline(
            condition="Customer needs assistance",
            action="Offer both a banana and an apple",
            composition_mode=p.CompositionMode.COMPOSITED,
        )

        # Create guideline with STRICT composition mode (more restrictive)
        self.guideline_strict = await self.agent.create_guideline(
            condition="Customer is hungry",
            action="Offer some food",
            composition_mode=p.CompositionMode.STRICT,
        )

    async def run(self, ctx: Context) -> None:
        # Send message that matches both guidelines
        response = await ctx.send_and_receive_message(
            customer_message="I'm hungry - can you assist me?",
            recipient=self.agent,
        )

        assert response == "Would you like a banana?"


class Test_that_composition_mode_does_not_persist_across_turns(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Test agent",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer wants a fruit and has not yet agreed to receive one",
            action="Offer both a banana and an apple, until the customer chooses one",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer wants a fruit",
            action="Offer a banana (just offer it once)",
            composition_mode=p.CompositionMode.STRICT,
            canned_responses=[
                await self.agent.create_canned_response(
                    template="Would you like a banana?",
                )
            ],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="You know what I'd really want right now? A good piece of fruit.",
            recipient=self.agent,
            reuse_session=True,
        )

        assert response == "Would you like a banana?"

        response = await ctx.send_and_receive_message(
            customer_message="I'm allergic to bananas, actually.",
            recipient=self.agent,
            reuse_session=True,
        )

        assert "apple" in response.lower()
