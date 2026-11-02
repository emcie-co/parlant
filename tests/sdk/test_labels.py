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


class Test_that_matched_rule_labels_are_added_to_session(SDKTest):
    """Test that when a rule with labels matches, its labels are added to the session."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="labels-agent-1",
            name="Label Test Agent",
            instructions="Agent for testing label propagation",
        )

        await self.agent.create_observation(
            condition="Customer asks about pricing",
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
