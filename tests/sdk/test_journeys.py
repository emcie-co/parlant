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

import pytest

from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.tags import Tag
from parlant.core.tools import ToolContext, ToolId, ToolResult
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test

from parlant import sdk as p


class Test_that_journey_can_be_created_without_conditions(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Store agent",
            description="You work at a store and help customers",
        )

        self.journey = await self.agent.create_journey(
            title="Greeting the customer",
            conditions=[],
            description="1. Offer the customer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        journey = await journey_store.read_journey(journey_id=self.journey.id)

        assert journey.id == self.journey.id
        assert journey.title == "Greeting the customer"
        assert journey.description == "1. Offer the customer a Pepsi"


class Test_that_condition_guidelines_are_tagged_for_created_journey(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Store agent",
            description="You work at a store and help customers",
        )

        self.journey = await self.agent.create_journey(
            title="Greeting the customer",
            conditions=["the customer greets you", "the customer says 'Howdy'"],
            description="1. Offer the customer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]
        guideline_store = ctx.container[GuidelineStore]

        journey = await journey_store.read_journey(journey_id=self.journey.id)
        condition_guidelines = [
            await guideline_store.read_guideline(guideline_id=g_id) for g_id in journey.conditions
        ]

        assert all(g.tags == [Tag.for_journey_id(self.journey.id)] for g in condition_guidelines)


class Test_that_condition_guidelines_are_evaluated_in_journey_creation(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Store agent",
            description="You work at a store and help customers",
        )

        self.journey = await self.agent.create_journey(
            title="Greeting the customer",
            conditions=["the customer greets you", "the customer says 'Howdy'"],
            description="1. Offer the customer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]
        guideline_store = ctx.container[GuidelineStore]

        journey = await journey_store.read_journey(journey_id=self.journey.id)

        condition_guidelines = [
            await guideline_store.read_guideline(guideline_id=g_id) for g_id in journey.conditions
        ]

        assert all("continuous" in g.metadata for g in condition_guidelines)
        assert all("customer_dependent_action_data" in g.metadata for g in condition_guidelines)


class Test_that_guideline_creation_from_journey_creates_dependency_relationship(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Store agent",
            description="You work at a store and help customers",
        )

        self.journey = await self.agent.create_journey(
            title="Greeting the customer",
            conditions=["the customer greets you", "the customer says 'Howdy'"],
            description="1. Offer the customer a Pepsi",
        )

        self.guideline = await self.journey.create_guideline(
            condition="you greet the customer",
            action="check the price of Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationships = await relationship_store.list_relationships(
            kind=RelationshipKind.DEPENDENCY,
            source_id=self.guideline.id,
        )

        assert relationships
        assert len(relationships) == 1
        assert relationships[0].target.id == Tag.for_journey_id(self.journey.id)


class Test_that_journey_can_be_created_with_guideline_object_as_condition(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Store agent",
            description="You work at a store and help customers",
        )

        self.condition_guideline = await self.agent.create_guideline(
            condition="the customer greets you"
        )

        self.journey = await self.agent.create_journey(
            title="Greeting the customer",
            conditions=[self.condition_guideline],
            description="1. Offer the customer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]
        guideline_store = ctx.container[GuidelineStore]

        journey = await journey_store.read_journey(journey_id=self.journey.id)
        guideline = await guideline_store.read_guideline(guideline_id=self.condition_guideline.id)

        assert journey.conditions == [guideline.id]
        assert guideline.id == self.condition_guideline.id


class Test_that_a_created_journey_is_followed(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Store agent",
            description="You work at a store and help customers",
        )

        self.journey = await self.agent.create_journey(
            title="Greeting the customer",
            conditions=["the customer greets you"],
            description="Offer the customer a Pepsi",
        )

        await self.journey.initial_state.transition_to(
            chat_state="offer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive("Hello there", recipient=self.agent)

        assert await nlp_test(
            context=response,
            condition="There is an offering of a Pepsi",
        )


class Test_that_journey_transition_and_state_can_be_created_with_transition(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey state creation tests",
        )

        self.journey = await self.agent.create_journey(
            title="State Journey",
            conditions=[],
            description="A journey with multiple states",
        )

        self.transition_w = await self.journey.initial_state.transition_to(
            chat_state="check room availability"
        )
        self.transition_x = await self.transition_w.target.transition_to(
            chat_state="provide hotel amenities"
        )

    async def run(self, ctx: Context) -> None:
        assert self.transition_w in self.journey.transitions
        assert self.transition_x in self.journey.transitions

        assert self.transition_w.source.id == self.journey.initial_state.id
        assert self.transition_w.target.action == "check room availability"
        assert self.transition_w.target in self.journey.states

        assert self.transition_x.source.id == self.transition_w.target.id
        assert self.transition_x.target.action == "provide hotel amenities"
        assert self.transition_x.target in self.journey.states


class Test_that_journey_state_can_transition_to_a_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey state creation tests",
        )

        self.journey = await self.agent.create_journey(
            title="State Journey",
            conditions=[],
            description="A journey with multiple states",
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.transition = await self.journey.initial_state.transition_to(
            tool_instruction="check available upgrades",
            tool_state=test_tool,
        )

    async def run(self, ctx: Context) -> None:
        state = self.transition.target

        assert state.tools

        assert len(state.tools) == 1
        assert state.tools[0].tool.name == "test_tool"


class Test_that_journey_state_can_be_transitioned_with_condition(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey conditioned states Agent",
            description="Agent for journey state with condition creation tests",
        )

        self.journey = await self.agent.create_journey(
            title="Conditioned-states Journey",
            conditions=[],
            description="A journey with states depending on customer decisions",
        )

        self.transition_x = await self.journey.initial_state.transition_to(
            chat_state="ask if the customer wants breakfast"
        )
        self.transition_y = await self.transition_x.target.transition_to(
            condition="if the customer says yes",
            chat_state="add breakfast to booking",
        )
        self.transition_z = await self.transition_x.target.transition_to(
            condition="if the customer says no",
            chat_state="proceed without breakfast",
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        transitions = self.journey.transitions
        states = self.journey.states

        assert {e.id for e in transitions}.issuperset(
            {self.transition_x.id, self.transition_y.id, self.transition_z.id}
        )

        assert {n.id for n in states}.issuperset(
            {
                self.transition_x.source.id,
                self.transition_x.target.id,
                self.transition_y.target.id,
                self.transition_z.target.id,
            }
        )

        store_edges = await journey_store.list_edges(journey_id=self.journey.id)
        store_nodes = await journey_store.list_nodes(journey_id=self.journey.id)

        assert {e.id for e in store_edges}.issuperset(
            {self.transition_x.id, self.transition_y.id, self.transition_z.id}
        )
        assert {n.id for n in store_nodes}.issuperset(
            {
                self.transition_x.source.id,
                self.transition_x.target.id,
                self.transition_y.target.id,
                self.transition_z.target.id,
            }
        )

        assert self.transition_y.condition == "if the customer says yes"
        assert self.transition_z.condition == "if the customer says no"


class Test_that_if_state_has_more_than_one_transition_they_all_need_to_have_conditions(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey conditioned states Agent",
            description="Agent for journey state with condition creation tests",
        )

        self.journey = await self.agent.create_journey(
            title="Conditioned-states Journey",
            conditions=[],
            description="A journey with states depending on customer decisions",
        )

        self.transition_ask_breakfast = await self.journey.initial_state.transition_to(
            chat_state="ask if the customer wants breakfast"
        )

        self.transition_add_breakfast = await self.transition_ask_breakfast.target.transition_to(
            condition="if the customer says yes",
            chat_state="add breakfast to booking",
        )

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await self.transition_ask_breakfast.target.transition_to(
                chat_state="proceed without breakfast"
            )


class Test_that_journey_is_reevaluated_after_tool_call(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey step creation tests",
        )

        self.journey = await self.agent.create_journey(
            title="Step Journey",
            conditions=[],
            description="A journey with tool-driven decision steps",
        )

        @tool
        def check_balance(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.transition_check_balance = await self.journey.initial_state.transition_to(
            tool_instruction="check customer account balance",
            tool_state=[check_balance],
        )

        self.transition_offer_discount = await self.transition_check_balance.target.transition_to(
            condition="balance is low",
            chat_state="offer discount if balance is low",
        )

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationships = await relationship_store.list_relationships(
            kind=RelationshipKind.REEVALUATION,
            target_id=Tag.for_journey_node_id(
                self.transition_check_balance.target.id,
            ),
        )

        assert relationships
        assert len(relationships) == 1
        assert relationships[0].kind == RelationshipKind.REEVALUATION
        assert relationships[0].source.id == ToolId(
            service_name=p.INTEGRATED_TOOL_SERVICE_NAME, tool_name="check_balance"
        )
        assert relationships[0].target.id == Tag.for_journey_node_id(
            self.transition_check_balance.target.id,
        )


class Test_that_journey_state_can_transition_to_end_state(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="EndState Agent",
            description="Agent for end state transition test",
        )

        self.journey = await self.agent.create_journey(
            title="End State Journey",
            conditions=[],
            description="A journey that ends",
        )

        self.transition_to_end = await self.journey.initial_state.transition_to(state=p.END_JOURNEY)

    async def run(self, ctx: Context) -> None:
        assert self.transition_to_end in self.journey.transitions
        assert self.transition_to_end.target.id == JourneyStore.END_NODE_ID


class Test_that_journey_state_can_be_created_with_internal_action(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Calzone Seller Agent",
            description="Agent for selling calzones",
        )

        self.journey = await self.agent.create_journey(
            title="Deliver Calzone Journey",
            conditions=["the customer wants to order a calzone"],
            description="A journey to deliver calzones",
        )

        self.transition_1 = await self.journey.initial_state.transition_to(
            chat_state="Welcome the customer to the Low Cal Calzone Zone",
        )

        self.transition_2 = await self.transition_1.target.transition_to(
            chat_state="Ask them how many they want",
        )

    async def run(self, ctx: Context) -> None:
        assert self.transition_1 in self.journey.transitions
        assert self.transition_2 in self.journey.transitions

        assert self.transition_1.target.action == "Welcome the customer to the Low Cal Calzone Zone"
        assert self.transition_2.target.action == "Ask them how many they want"

        second_target = await ctx.container[JourneyStore].read_node(
            node_id=self.transition_2.target.id,
        )

        assert second_target.action == "Ask them how many they want"
        assert (
            "internal_action" in second_target.metadata
            and second_target.metadata["internal_action"]
            and second_target.action != second_target.metadata["internal_action"]
        )
