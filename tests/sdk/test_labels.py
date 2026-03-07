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

"""Tests for automatic session label propagation from matched entities."""

import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest


class Test_that_matched_guideline_labels_are_added_to_session(SDKTest):
    """Test that when a guideline with labels matches, its labels are added to the session."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Label Test Agent",
            description="Agent for testing label propagation",
        )

        await self.agent.create_guideline(
            condition="Customer asks about pricing",
            action="Provide pricing information",
            labels=["pricing", "sales"],
        )

        await self.agent.create_observation(
            condition="Customer wants to buy a car",
            labels=["cars"],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="What are your prices? I want to get a new Volvo.",
            recipient=self.agent,
        )

        session = await ctx.get_session()

        assert "pricing" in session.labels
        assert "sales" in session.labels
        assert "cars" in session.labels


class Test_that_matched_journey_labels_are_added_to_session(SDKTest):
    """Test that when a journey with labels matches, its labels are added to the session."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Label Agent",
            description="Agent for testing journey label propagation",
        )

        await self.agent.create_journey(
            title="Support Journey",
            description="A support journey with labels",
            conditions=["Customer needs support"],
            labels=["support", "help"],
        )

        await self.agent.create_journey(
            title="Greeting Journey",
            description="A greeting journey with labels",
            conditions=["Customer says hello"],
            labels=["greeting"],
        )

    async def run(self, ctx: Context) -> None:
        # Send a message that should trigger the journey
        await ctx.send_and_receive_message(
            customer_message="Good morning! I need some support today.",
            recipient=self.agent,
        )

        # Check that the session now has the labels from the matched journey
        session = await ctx.get_session()

        assert "support" in session.labels
        assert "help" in session.labels
        assert "greeting" in session.labels


class Test_that_matched_journey_state_labels_are_added_to_session(SDKTest):
    """Test that labels from the initial journey state are added to the session."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="State Label Agent",
            description="Agent for testing journey state label propagation",
        )

        # Create a journey with labels that will be propagated when the journey matches
        self.journey = await self.agent.create_journey(
            title="Checkout Journey",
            description="A checkout journey",
            conditions=["Customer wants to checkout"],
        )

        step_1 = await self.journey.initial_state.transition_to(
            chat_state="Ask for payment method",
            labels=["collect_payment_info"],
        )

        await step_1.target.transition_to(
            chat_state="Ask for shipping address",
            labels=["collect_shipping_info"],
        )

    async def run(self, ctx: Context) -> None:
        customer_messages = {
            "I want to checkout now": [],
            "I will pay with my credit card": ["collect_payment_info"],
            "My shipping address is 123 Main St": ["collect_payment_info", "collect_shipping_info"],
        }

        turn = 0

        for message, expected_labels in customer_messages.items():
            turn += 1

            await ctx.send_and_receive_message(
                customer_message=message,
                recipient=self.agent,
                reuse_session=True,
            )

            session = await ctx.get_session()

            for label in expected_labels:
                assert label in session.labels, (
                    f"Expected label '{label}' not found in session after turn {turn}."
                )
