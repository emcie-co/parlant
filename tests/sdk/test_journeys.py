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
from parlant.core.canned_responses import CannedResponseStore
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
            source_id=Tag.for_journey_node_id(
                self.transition_check_balance.target.id,
            ),
        )

        assert relationships
        assert len(relationships) == 1
        assert relationships[0].kind == RelationshipKind.REEVALUATION
        assert relationships[0].source.id == Tag.for_journey_node_id(
            self.transition_check_balance.target.id,
        )

        assert relationships[0].target.id == ToolId(
            service_name=p.INTEGRATED_TOOL_SERVICE_NAME, tool_name="check_balance"
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


class Test_that_journey_can_prioritize_another_journey(SDKTest):
    STARTUP_TIMEOUT = 120

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Rel Agent",
            description="Agent testing journey-to-journey relationship",
        )

        self.journey_a = await self.agent.create_journey(
            title="Process Return",
            conditions=["customer wants to return a product"],
            description="Handle product returns",
        )

        self.journey_b = await self.agent.create_journey(
            title="Offer Exchange",
            conditions=["customer is unsure about return"],
            description="Suggest product exchanges",
        )

        self.relationship = await self.journey_a.prioritize_over(self.journey_b)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )

        assert relationship.kind == RelationshipKind.PRIORITY
        assert relationship.source.id == Tag.for_journey_id(self.journey_a.id)
        assert relationship.target.id == Tag.for_journey_id(self.journey_b.id)


class Test_that_journey_can_depend_on_a_guideline(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Rel Agent",
            description="Agent testing journey-to-guideline dependency",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer must confirm identity",
            action="Ask for last four digits of phone",
        )

        self.journey = await self.agent.create_journey(
            title="Sensitive Account Help",
            conditions=["customer requests password reset"],
            description="Assist customer securely",
        )

        self.relationship = await self.journey.depend_on(self.guideline)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )

        assert relationship.kind == RelationshipKind.DEPENDENCY
        assert relationship.source.id == Tag.for_journey_id(self.journey.id)
        assert relationship.target.id == self.guideline.id


class Test_that_journey_guideline_can_be_created_with_canned_responses(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Canned Response Agent",
            description="Agent for testing journey guideline canned response associations",
        )

        self.journey = await self.agent.create_journey(
            title="Customer Support Journey",
            conditions=["Customer needs assistance"],
            description="Handle customer support requests",
        )

        self.canrep1 = await self.journey.create_canned_response(
            template="I understand your concern about {issue}."
        )
        self.canrep2 = await self.journey.create_canned_response(
            template="Let me help you resolve {problem}."
        )

        self.guideline = await self.journey.create_guideline(
            condition="Customer describes an issue",
            action="Acknowledge and offer help",
            canned_responses=[self.canrep1, self.canrep2],
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]

        updated_canrep1 = await canrep_store.read_canned_response(self.canrep1)
        updated_canrep2 = await canrep_store.read_canned_response(self.canrep2)

        assert Tag.for_guideline_id(self.guideline.id) in updated_canrep1.tags
        assert Tag.for_guideline_id(self.guideline.id) in updated_canrep2.tags


class Test_that_journey_guideline_with_tools_can_have_canned_responses(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Tool Agent",
            description="Agent for testing journey guideline with tools and canned responses",
        )

        self.journey = await self.agent.create_journey(
            title="Tool-assisted Journey",
            conditions=["Customer needs technical help"],
            description="Provide technical assistance with tools",
        )

        @tool
        def diagnostic_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={"status": "running"})

        self.canrep = await self.journey.create_canned_response(
            template="I've run a diagnostic and found {result}."
        )

        self.guideline = await self.journey.create_guideline(
            condition="Customer reports system issue",
            action="Run diagnostic and report findings",
            tools=[diagnostic_tool],
            canned_responses=[self.canrep],
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]

        updated_canrep = await canrep_store.read_canned_response(self.canrep)

        assert Tag.for_guideline_id(self.guideline.id) in updated_canrep.tags


class Test_that_journey_state_can_have_its_own_canned_responses(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            description="Just a dummy test agent",
            composition_mode=p.CompositionMode.STRICT,
        )

        self.journey = await self.agent.create_journey(
            title="Customer Greeting Journey",
            conditions=["Customer arrives"],
            description="Greet customers with personalized responses",
        )

        self.canrep1 = await server.create_canned_response(template="How can I assist you?")
        self.canrep2 = await server.create_canned_response(template="Welcome to our store!")

        self.initial_transition = await self.journey.initial_state.transition_to(
            chat_state="Greet the customer to our store (Welcome to our store!)",
            canned_responses=[self.canrep1],
        )

        self.second_transition = await self.initial_transition.target.transition_to(
            chat_state="Ask how they can be helped",
            canned_responses=[self.canrep2],
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]

        stored_canrep1 = await canrep_store.read_canned_response(self.canrep1)
        stored_canrep2 = await canrep_store.read_canned_response(self.canrep2)

        assert Tag.for_journey_node_id(self.initial_transition.target.id) in stored_canrep1.tags
        assert Tag.for_journey_node_id(self.second_transition.target.id) in stored_canrep2.tags

        response = await ctx.send_and_receive("Hello", recipient=self.agent)

        assert response == "How can I assist you?"


class Test_that_a_journey_is_reevaluated_after_a_skipped_tool_call(SDKTest):
    async def setup(self, server: p.Server) -> None:
        @tool
        def get_customer_date_of_birth(context: ToolContext) -> ToolResult:
            return ToolResult(data={"date_of_birth": "January 1, 2000"})

        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent for testing journeys",
        )

        # We're first gonna run this guideline so as to get the tool event
        # into the context.
        await self.agent.create_guideline(
            condition="The customer greets you",
            action="Tell them their date of birth",
            tools=[get_customer_date_of_birth],
        )

        self.journey = await self.agent.create_journey(
            title="Handle Thirsty Customer",
            conditions=["Customer is thirsty"],
            description="Help a thirsty customer with a refreshing drink",
        )

        # Then we'll want to see that the journey reaches the chat state even though
        # the tool call is skipped (its previous result was already in context).
        self.t1 = await self.journey.initial_state.transition_to(
            tool_state=get_customer_date_of_birth,
        )
        self.t2 = await self.t1.target.transition_to(
            chat_state="Offer the customer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        first_response = await ctx.send_and_receive(
            "Hello", recipient=self.agent, reuse_session=True
        )

        print(first_response)

        assert await nlp_test(first_response, "It mentions the date January 1st, 2000")

        second_response = await ctx.send_and_receive(
            "I'm really thirsty", recipient=self.agent, reuse_session=True
        )

        assert await nlp_test(second_response, "It offers a Pepsi")
        # Make sure the guideline was not re-applied
        assert await nlp_test(second_response, "It does not mention one million dollars")
        print(second_response)


class Test_that_metadata_can_be_set_to_a_journey_state(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Metadata Agent",
            description="Agent for testing metadata on journey states",
        )

        self.journey = await self.agent.create_journey(
            title="Metadata Journey",
            conditions=["Customer requests information"],
            description="Provide information with metadata tracking",
        )

        self.transition = await self.journey.initial_state.transition_to(
            chat_state="Provide details",
            metadata={
                "continuous": False,
                "internal_action": "Provide detailed information about our services",
            },
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        state = await journey_store.read_node(node_id=self.transition.target.id)

        assert state.metadata.get("continuous") is False
        assert (
            state.metadata.get("internal_action")
            == "Provide detailed information about our services"
        )
