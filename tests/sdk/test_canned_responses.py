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
            id="canned-responses-agent-1",
            name="Field Dependencies Agent",
            instructions="Agent for testing field dependencies",
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


class Test_that_canned_response_can_be_created_with_notes(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="canned-responses-agent-2",
            name="Note Agent",
            instructions="Agent for testing canned response note",
        )
        self.canrep_id = await self.agent.create_canned_response(
            template="We're happy to help!",
            note="Warm, friendly tone. Use for greetings.",
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]
        canrep = await canrep_store.read_canned_response(canned_response_id=self.canrep_id)

        assert canrep.value == "We're happy to help!"
        assert canrep.note == "Warm, friendly tone. Use for greetings."


class Test_that_canned_response_notes_default_to_empty(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="canned-responses-agent-3",
            name="Default Note Agent",
            instructions="Agent for testing default canned response note",
        )
        self.canrep_id = await self.agent.create_canned_response(
            template="Hello there.",
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]
        canrep = await canrep_store.read_canned_response(canned_response_id=self.canrep_id)

        assert canrep.note == ""


class Test_that_canned_response_with_field_dependency_is_excluded_when_field_unavailable(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="canned-responses-agent-4",
            name="Test Agent",
            instructions="",
        )

        # A canned response whose declared field dependency ("order") nothing
        # provides — it must be disqualified from selection entirely.
        canrep_with_dependency = await self.agent.create_canned_response(
            template="Your order is ready for pickup.",
            field_dependencies=["order"],
        )

        # A satisfiable alternative the agent should fall back to.
        canrep_fallback = await self.agent.create_canned_response(
            template="I can't check your order status right now.",
        )

        # The policy is deliberately neutral about the outcome: the response
        # content must come from the canned responses, so the exclusion is
        # observable.
        await self.agent.create_policy(
            content=(
                "When the customer asks about their order, "
                "update them on their order status using an approved response"
            ),
            canned_responses=[canrep_with_dependency, canrep_fallback],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What about my order?",
            recipient=self.agent,
        )

        # The dependency-unmet response must not drive the reply; the
        # satisfiable one should.
        assert "ready for pickup" not in response.lower(), (
            f"Expected the dependency-unmet canned response to be excluded, got: {response}"
        )
        assert "right now" in response.lower(), (
            f"Expected the satisfiable canned response to drive the reply, got: {response}"
        )
