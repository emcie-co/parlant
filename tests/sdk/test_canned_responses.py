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

from parlant.core.canned_responses import CannedResponseStore
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest


class Test_that_canned_response_can_be_created_with_field_dependencies(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Field Dependencies Agent",
            description="Agent for testing field dependencies",
        )
        self.canrep_id = await self.agent.create_canned_response(
            template="Your order status is: ready.",
            field_dependencies=["order"],
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]
        canrep = await canrep_store.read_canned_response(canned_response_id=self.canrep_id)

        assert canrep.value == "Your order status is: ready."
        assert "order" in canrep.field_dependencies


class Test_that_canned_response_with_field_dependency_is_excluded_when_field_unavailable(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="",
        )

        # Create a canned response that depends on "order" field
        canrep_with_dependency = await self.agent.create_canned_response(
            template="Your order is ready for pickup.",
            field_dependencies=["order"],  # Should disqualify this response, since it's unavailable
        )

        # Guideline that uses both canned responses - the one with dependency should be excluded
        # because no tool provides the "order" field
        await self.agent.create_guideline(
            condition="Customer asks about their order",
            action="Tell them that their order is ready for pickup",
            composition_mode=p.CompositionMode.STRICT,
            canned_responses=[canrep_with_dependency],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What about my order?",
            recipient=self.agent,
        )

        # The response with field dependency should be excluded.
        # Instead, the response is expected to be a fallback "no-match" one.
        assert "order" not in response.lower()
