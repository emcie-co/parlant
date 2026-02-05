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

import pytest

from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.tags import Tag
from parlant.core.tools import ToolContext, ToolId, ToolResult
from parlant.core.canned_responses import CannedResponseStore
from tests.sdk.utils import Context, SDKTest, get_message
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


class Test_that_scoped_guideline_of_matched_journey_without_states_influence_response(SDKTest):
    """Test that providing a custom ID to transition_to uses that ID."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Test agent for custom state ID",
        )

        self.journey = await self.agent.create_journey(
            title="Test Journey",
            conditions=["Customer greets you"],
            description="Test journey",
        )

        await self.journey.create_guideline(
            matcher=p.Guideline.MATCH_ALWAYS,
            condition="The customer greets you",
            action="Immediately offer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message("Hello!", recipient=self.agent)
        assert "pepsi" in response.lower()


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
        response = await ctx.send_and_receive_message("Hello there", recipient=self.agent)

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


class Test_that_journey_can_take_priority_over_another_journey(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="",
        )

        # Both journeys match when customer asks about drinks
        self.high_priority = await self.agent.create_journey(
            title="Journey 1",
            conditions=["Customer asks about drinks"],
            description="Recommend Pepsi to the customer",
        )

        await self.high_priority.create_guideline(
            matcher=p.Guideline.MATCH_ALWAYS,
            action="Recommend Pepsi",
        )

        self.low_priority = await self.agent.create_journey(
            title="Journey 2",
            conditions=["Customer asks about drinks"],
            description="Recommend Coca-Cola to the customer",
        )

        await self.low_priority.create_guideline(
            matcher=p.Guideline.MATCH_ALWAYS,
            action="Recommend Coca-Cola",
        )

        await self.high_priority.prioritize_over(self.low_priority)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What drinks do you have?",
            recipient=self.agent,
        )

        # High priority journey's recommendation should apply
        assert "pepsi" in response.lower(), f"Expected Pepsi in response: {response}"
        # Low priority journey's recommendation should NOT apply
        assert "cola" not in response.lower() and "coke" not in response.lower(), (
            f"Did not expect Coca-Cola in response: {response}"
        )


class Test_that_journey_can_take_priority_over_a_guideline(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="",
        )

        # Guideline that matches when customer asks about drinks
        self.guideline = await self.agent.create_guideline(
            condition="Customer asks about drinks",
            action="Recommend Coca-Cola",
        )

        # Journey that also matches when customer asks about drinks
        self.journey = await self.agent.create_journey(
            title="Drink Recommendation Journey",
            conditions=["Customer asks about drinks"],
            description="Recommend Pepsi to the customer",
        )

        await self.journey.create_guideline(
            matcher=p.Guideline.MATCH_ALWAYS,
            action="Recommend Pepsi",
        )

        await self.journey.prioritize_over(self.guideline)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What drinks do you have?",
            recipient=self.agent,
        )

        # Journey's recommendation should apply
        assert "pepsi" in response.lower(), f"Expected Pepsi in response: {response}"
        # Guideline's recommendation should NOT apply
        assert "cola" not in response.lower() and "coke" not in response.lower(), (
            f"Did not expect Coca-Cola in response: {response}"
        )


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

        self.relationships = await self.journey.depend_on(self.guideline)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationships[0].id
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

        self.canrep1 = await server.create_canned_response(
            template="How can I assist you?",
            metadata={"mood": "friendly"},
        )
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

        response = await ctx.send_and_receive_message_event("Hello", recipient=self.agent)

        assert get_message(response) == "How can I assist you?"
        assert response.metadata == {"mood": "friendly"}


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
        first_response = await ctx.send_and_receive_message(
            "Hello", recipient=self.agent, reuse_session=True
        )

        assert await nlp_test(first_response, "It mentions the date January 1st, 2000")

        second_response = await ctx.send_and_receive_message(
            "I'm really thirsty", recipient=self.agent, reuse_session=True
        )

        assert await nlp_test(second_response, "It offers a Pepsi")


class Test_that_a_missing_data_is_shown_after_journey_is_reevaluated(SDKTest):
    async def setup(self, server: p.Server) -> None:
        @tool
        def get_customer_last_time_drank(context: ToolContext, customer_name: str) -> ToolResult:
            return ToolResult(data={"last_time_drank": "January 1, 2000"})

        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent for testing journeys",
        )

        self.journey = await self.agent.create_journey(
            title="Handle Thirsty Customer",
            conditions=["Customer is thirsty"],
            description="Help a thirsty customer with a refreshing drink",
        )

        # Then we want to verify that the journey reaches the chat state
        # even though the tool call received missing data.
        self.t1 = await self.journey.initial_state.transition_to(
            tool_instruction="Check when the customer last drank",
            tool_state=get_customer_last_time_drank,
        )
        self.t2 = await self.t1.target.transition_to(
            chat_state="Offer the customer a suitable amount of Pepsi based on when they last drank",
        )

    async def run(self, ctx: Context) -> None:
        first_response = await ctx.send_and_receive_message(
            "I'm really thirsty", recipient=self.agent, reuse_session=True
        )

        assert await nlp_test(first_response, "It asks for the customer's name")


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


class Test_that_journey_can_have_a_scoped_guideline(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            description="Dummy agent",
        )

        self.journey = await self.agent.create_journey(
            title="Order Something",
            conditions=["The customer wants to order something"],
            description="Help the customer place an order",
        )

        await self.journey.initial_state.transition_to(
            chat_state="greet the customer",
        )

        self.guideline = await self.journey.create_guideline(
            condition="The customer wants to order a banana",
            action="Ask them if they'd like green or yellow bananas",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            "Can I order a banana?",
            recipient=self.agent,
        )

        assert "green" in response.lower()


class Test_that_journey_can_be_created_with_custom_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        from parlant.core.journeys import JourneyId

        self.agent = await server.create_agent(
            name="Custom ID Agent",
            description="Agent for testing custom journey IDs",
        )

        self.custom_id = JourneyId("custom-journey-123")

        self.journey = await self.agent.create_journey(
            title="Custom ID Journey",
            conditions=["Customer needs help"],
            description="Journey with custom ID",
            id=self.custom_id,
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        journey = await journey_store.read_journey(journey_id=self.custom_id)

        assert journey.id == self.custom_id
        assert journey.title == "Custom ID Journey"
        assert journey.description == "Journey with custom ID"


class Test_that_journey_creation_fails_with_duplicate_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        from parlant.core.journeys import JourneyId

        self.agent = await server.create_agent(
            name="Duplicate ID Agent",
            description="Agent for testing duplicate journey IDs",
        )

        self.duplicate_id = JourneyId("duplicate-journey-456")

        # Create the first journey
        self.first_journey = await self.agent.create_journey(
            title="First Journey",
            conditions=["First condition"],
            description="First journey with duplicate ID",
            id=self.duplicate_id,
        )

    async def run(self, ctx: Context) -> None:
        # Attempt to create a second journey with the same ID should fail
        with pytest.raises(
            ValueError, match="Journey with id 'duplicate-journey-456' already exists"
        ):
            await self.agent.create_journey(
                title="Second Journey",
                conditions=["Second condition"],
                description="Second journey with duplicate ID",
                id=self.duplicate_id,
            )


class Test_that_end_journey_match_handlers_are_called(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Exit Handler Agent",
            description="Tests specific END_JOURNEY transition handlers",
        )

        self.journey = await self.agent.create_journey(
            title="Order Process",
            description="Order processing journey",
            conditions=["Customer wants to place an order"],
        )

        # Track which exit handler was called
        self.success_exit_called = False
        self.cancel_exit_called = False

        async def success_exit_handler(ctx: p.EngineContext, match: p.JourneyStateMatch) -> None:
            assert match.state_id == "end", "Should be exiting to END_JOURNEY"
            self.success_exit_called = True

        async def cancel_exit_handler(ctx: p.EngineContext, match: p.JourneyStateMatch) -> None:
            assert match.state_id == "end", "Should be exiting to END_JOURNEY"
            self.cancel_exit_called = True

        # Create a chat state for order confirmation
        confirmation_state = await self.journey.initial_state.transition_to(
            chat_state="Please confirm your order or cancel",
        )

        # Exit path 1: Customer confirms order (success path)
        await confirmation_state.target.transition_to(
            condition="Customer confirms the order",
            state=p.END_JOURNEY,
            on_match=success_exit_handler,
        )

        # Exit path 2: Customer cancels order (cancel path)
        await confirmation_state.target.transition_to(
            condition="Customer wants to cancel",
            state=p.END_JOURNEY,
            on_match=cancel_exit_handler,
        )

    async def run(self, ctx: Context) -> None:
        # Start the journey
        await ctx.send_and_receive_message(
            customer_message="I want to place an order",
            recipient=self.agent,
        )

        # Trigger the success exit path
        await ctx.send_and_receive_message(
            customer_message="Yes, please confirm my order",
            recipient=self.agent,
            reuse_session=True,
        )

        # Verify only the success exit handler was called
        assert self.success_exit_called, "Success exit handler should have been called"
        assert not self.cancel_exit_called, "Cancel exit handler should NOT have been called"


class Test_that_journey_state_match_handler_is_called(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.handler_called = False
        self.captured_state_id = None

        async def state_match_handler(ctx: p.EngineContext, match: p.JourneyStateMatch) -> None:
            self.handler_called = True
            self.captured_state_id = match.state_id

        self.agent = await server.create_agent(
            name="Order Agent",
            description="Agent for testing journey state match handlers",
        )

        self.journey = await self.agent.create_journey(
            title="Order Something",
            description="Journey to handle orders",
            conditions=["Customer wants to order something"],
        )

        self.state = await self.journey.initial_state.transition_to(
            condition="Customer confirmed order",
            chat_state="Great! Your order is confirmed.",
            on_match=state_match_handler,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I want to order something. Yes, confirmed!",
            recipient=self.agent,
        )

        assert self.handler_called, "State match handler should have been called"
        assert self.captured_state_id == self.state.target.id, (
            f"Expected state ID {self.state.target.id}, got {self.captured_state_id}"
        )


class Test_that_journey_state_can_be_created_with_description(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Pizza Agent",
            description="Agent for testing journey state descriptions",
        )

        self.journey = await self.agent.create_journey(
            title="Pizza Ordering",
            description="Handle pizza orders",
            conditions=["Customer wants to order pizza"],
        )

        self.transition = await self.journey.initial_state.transition_to(
            condition="Customer confirms toppings",
            chat_state="Process the order",
            description="At this point we've confirmed the pizza toppings and are ready to finalize",
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        # Read the created state/node from the store
        node = await journey_store.read_node(node_id=self.transition.target.id)

        assert (
            node.description
            == "At this point we've confirmed the pizza toppings and are ready to finalize"
        )


class Test_that_journey_state_description_affects_agent_behavior(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Spaceship Agent",
            description="Agent for testing journey state description behavior",
        )

        self.journey = await self.agent.create_journey(
            title="Spaceship Maintenance",
            description="Handle spaceship maintenance requests",
            conditions=["Customer asks about spaceship maintenance"],
        )

        await self.journey.initial_state.transition_to(
            condition="Customer needs thruster calibration",
            chat_state="Explain the calibration process",
            description="First you peel the banana, then you stick it in the thruster",
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="I need help with spaceship maintenance. Specifically thruster calibration.",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It mentions a banana")


class Test_that_different_state_types_support_description(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Multi-State Agent",
            description="Agent for testing descriptions across state types",
        )

        @tool
        def check_inventory(context: ToolContext) -> ToolResult:
            return ToolResult(data={"status": "available"})

        self.journey = await self.agent.create_journey(
            title="Order Processing",
            description="Process customer orders",
            conditions=["Customer wants to place an order"],
        )

        # ChatJourneyState with description
        self.chat_transition = await self.journey.initial_state.transition_to(
            condition="Customer provides item name",
            chat_state="Confirm the item selection",
            description="This is where we confirm what item the customer wants to order",
        )

        # ToolJourneyState with description
        self.tool_transition = await self.chat_transition.target.transition_to(
            condition="Need to check inventory",
            tool_state=check_inventory,
            description="Check if the item is in stock using our inventory system",
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        # Verify ChatJourneyState has description
        chat_node = await journey_store.read_node(node_id=self.chat_transition.target.id)
        assert (
            chat_node.description
            == "This is where we confirm what item the customer wants to order"
        )

        # Verify ToolJourneyState has description
        tool_node = await journey_store.read_node(node_id=self.tool_transition.target.id)
        assert tool_node.description == "Check if the item is in stock using our inventory system"


class Test_that_on_message_handler_is_called_for_journey_state_when_message_generated(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.handler_called = False
        self.captured_state_id = None
        self.captured_message_count = 0

        async def message_handler(ctx: p.EngineContext, match: p.JourneyStateMatch) -> None:
            self.handler_called = True
            self.captured_state_id = match.state_id
            # Verify we can access messages from context
            self.captured_message_count = len(ctx.state.message_events)

        self.agent = await server.create_agent(
            name="Booking Agent",
            description="Agent for testing journey state on_message handler",
        )

        self.journey = await self.agent.create_journey(
            title="Book Appointment",
            description="Journey to book appointments",
            conditions=["Customer wants to book an appointment"],
        )

        self.state = await self.journey.initial_state.transition_to(
            condition="Customer provides appointment details",
            chat_state="Perfect! Your appointment is scheduled.",
            on_message=message_handler,  # type: ignore[call-overload]
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I want to book an appointment for tomorrow at 3pm",
            recipient=self.agent,
        )

        # Wait for handlers to complete
        import asyncio

        await asyncio.sleep(5)

        assert self.handler_called, "on_message handler should be called"
        assert self.captured_message_count > 0, "Handler should see messages in context"
        assert self.captured_state_id == self.state.target.id, (
            f"Handler should receive correct state ID. Expected {self.state.target.id}, got {self.captured_state_id}"
        )


class Test_that_journey_state_field_provider_contributes_fields_to_canned_response(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Field Provider Agent",
            description="Agent for testing journey state field providers",
        )

        self.journey = await self.agent.create_journey(
            title="Order Journey",
            description="Handle customer orders",
            conditions=["Customer wants to order"],
        )

        canrep_id = await self.agent.create_canned_response(
            template="Your order number is {{order_number}}.",
        )

        async def provide_order_fields(ctx: p.EngineContext) -> dict[str, int]:
            return {"order_number": 12345}

        self.state = await self.journey.initial_state.transition_to(
            chat_state="Confirm the order",
            composition_mode=p.CompositionMode.STRICT,
            canned_responses=[canrep_id],
            canned_response_field_provider=provide_order_fields,
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="I want to place an order",
            recipient=self.agent,
        )

        assert response == "Your order number is 12345."


class Test_that_journey_can_link_to_another_journey_with_validation(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Hotel Booking Agent",
            description="Agent for handling hotel bookings with user validation",
            composition_mode=p.CompositionMode.STRICT,
        )

        # Create canned responses for deterministic testing
        self.room_choice_response = await server.create_canned_response(
            template="Would you like the red room or the blue room?"
        )
        self.name_request_response = await server.create_canned_response(
            template="Please provide your name for verification."
        )
        self.booking_confirmed_response = await server.create_canned_response(
            template="Great! Your hotel booking has been confirmed."
        )
        self.not_confirmed_response = await server.create_canned_response(
            template="I'm sorry, but we cannot proceed with the booking without proper validation."
        )

        # Create validation tool that always returns True
        @tool
        def validate_by_name(context: ToolContext, customer_name: str) -> ToolResult:
            return ToolResult(data={"is_valid": True})

        # Create the user validation journey
        self.validate_user_journey = await self.agent.create_journey(
            title="Validate User",
            conditions=[],
            description="Validate the user by asking for their name and verifying it",
        )

        # First state: ask for name
        self.ask_name_transition = await self.validate_user_journey.initial_state.transition_to(
            chat_state="Ask the customer for their name to verify their identity",
            canned_responses=[self.name_request_response],
        )

        # Second state: validate using the tool
        self.validate_transition = await self.ask_name_transition.target.transition_to(
            tool_instruction="Validate customer",
            tool_state=validate_by_name,
        )

        # Create the hotel booking journey
        self.book_hotel_journey = await self.agent.create_journey(
            title="Book Hotel",
            conditions=["Customer wants to book a hotel"],
            description="Booking a hotel room for the customer",
        )

        # Second state: transition to validation journey
        self.room_type = await self.book_hotel_journey.initial_state.transition_to(
            chat_state="Ask the customer if he wants the red or blue room",
            canned_responses=[self.room_choice_response],
        )

        # Third state: transition to validation journey
        self.validation_transition = await self.room_type.target.transition_to(
            journey=self.validate_user_journey,
        )

        # Fourth state: conditional booking based on validation
        self.book_success_transition = await self.validation_transition.target.transition_to(
            condition="if validation is successful",
            chat_state="Let him know we confirm the hotel booking",
            canned_responses=[self.booking_confirmed_response],
        )

        # Alternative state: apologize if validation fails
        self.apologize_transition = await self.validation_transition.target.transition_to(
            condition="if validation fails",
            chat_state="Apologize and explain that booking cannot proceed without validation",
            canned_responses=[self.not_confirmed_response],
        )

    async def run(self, ctx: Context) -> None:
        # Test the complete flow
        response1 = await ctx.send_and_receive_message(
            "I want to book a hotel room",
            recipient=self.agent,
            reuse_session=True,
        )
        assert response1 == "Would you like the red room or the blue room?"

        response2 = await ctx.send_and_receive_message(
            "I want the red room",
            recipient=self.agent,
            reuse_session=True,
        )
        assert response2 == "Please provide your name for verification."

        response3 = await ctx.send_and_receive_message(
            "My name is John Smith", recipient=self.agent, reuse_session=True
        )
        assert response3 == "Great! Your hotel booking has been confirmed."


class Test_that_journey_can_conditionally_link_to_different_sub_journeys(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Multi-Journey Agent",
            description="Agent that can link to different sub-journeys based on conditions",
            composition_mode=p.CompositionMode.STRICT,
        )

        # Create canned responses for deterministic testing
        self.service_type_response = await server.create_canned_response(
            template="What type of service do you need: technical support or billing help?"
        )
        self.tech_greeting_response = await server.create_canned_response(
            template="Welcome to technical support! Please describe your issue."
        )
        self.billing_greeting_response = await server.create_canned_response(
            template="Welcome to billing support! How can I help with your account?"
        )
        self.tech_resolved_response = await server.create_canned_response(
            template="Your technical issue has been resolved. Is there anything else?"
        )
        self.billing_resolved_response = await server.create_canned_response(
            template="Your billing inquiry has been handled. Anything else I can help with?"
        )
        self.final_response = await server.create_canned_response(
            template="Thank you for contacting support. Have a great day!"
        )

        # Create tools for both support types
        @tool
        def resolve_tech_issue(context: ToolContext, issue_description: str) -> ToolResult:
            return ToolResult(data={"status": "resolved", "solution": "Issue fixed"})

        @tool
        def resolve_billing_issue(context: ToolContext, billing_question: str) -> ToolResult:
            return ToolResult(data={"status": "resolved", "account_updated": True})

        # Create technical support sub-journey
        self.tech_support_journey = await self.agent.create_journey(
            title="Technical Support",
            conditions=[],
            description="Handle technical support requests",
        )

        self.tech_greeting = await self.tech_support_journey.initial_state.transition_to(
            chat_state="Greet customer and ask for technical issue details",
            canned_responses=[self.tech_greeting_response],
        )

        self.tech_resolution = await self.tech_greeting.target.transition_to(
            tool_instruction="Resolve the technical issue",
            tool_state=resolve_tech_issue,
        )

        self.tech_completion = await self.tech_resolution.target.transition_to(
            chat_state="Confirm technical issue resolution",
            canned_responses=[self.tech_resolved_response],
        )

        # Create billing support sub-journey
        self.billing_support_journey = await self.agent.create_journey(
            title="Billing Support",
            conditions=[],
            description="Handle billing and account inquiries",
        )

        self.billing_greeting = await self.billing_support_journey.initial_state.transition_to(
            chat_state="Greet customer and ask for billing question",
            canned_responses=[self.billing_greeting_response],
        )

        self.billing_resolution = await self.billing_greeting.target.transition_to(
            tool_instruction="Resolve the billing issue",
            tool_state=resolve_billing_issue,
        )

        self.billing_completion = await self.billing_resolution.target.transition_to(
            chat_state="Confirm billing issue resolution",
            canned_responses=[self.billing_resolved_response],
        )

        # Create main customer service journey
        self.main_journey = await self.agent.create_journey(
            title="Customer Service",
            conditions=["Customer needs support"],
            description="Route customers to appropriate support channels",
        )

        # Initial state: ask what type of service they need
        self.service_inquiry = await self.main_journey.initial_state.transition_to(
            chat_state="Ask customer what type of service they need",
            canned_responses=[self.service_type_response],
        )

        # Conditional transitions to different sub-journeys
        self.tech_transition = await self.service_inquiry.target.transition_to(
            condition="if customer needs technical support",
            journey=self.tech_support_journey,
        )

        self.billing_transition = await self.service_inquiry.target.transition_to(
            condition="if customer needs billing help",
            journey=self.billing_support_journey,
        )

    async def run(self, ctx: Context) -> None:
        # Test technical support path
        response1 = await ctx.send_and_receive_message(
            "I need some help",
            recipient=self.agent,
            reuse_session=True,
        )
        assert response1 == "What type of service do you need: technical support or billing help?"

        response2 = await ctx.send_and_receive_message(
            "I need technical support",
            recipient=self.agent,
            reuse_session=True,
        )
        assert response2 == "Welcome to technical support! Please describe your issue."

        # Test billing support path with new session
        response3 = await ctx.send_and_receive_message(
            "I need some help",
            recipient=self.agent,
            reuse_session=False,  # Start new session
        )
        assert response3 == "What type of service do you need: technical support or billing help?"

        response4 = await ctx.send_and_receive_message(
            "I have a billing question",
            recipient=self.agent,
            reuse_session=True,
        )
        assert response4 == "Welcome to billing support! How can I help with your account?"


class Test_that_three_journeys_can_be_concatenated(SDKTest):
    STARTUP_TIMEOUT = 120

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Three Journey Agent",
            description="Agent that links three journeys in sequence",
            composition_mode=p.CompositionMode.STRICT,
        )

        # Create canned responses
        self.step1_response = await server.create_canned_response(
            template="Please tell me your name."
        )
        self.step2_response = await server.create_canned_response(
            template="What's your favorite color?"
        )
        self.step3_response = await server.create_canned_response(
            template="All done! Thank you for completing all steps."
        )

        # Journey 1: Collect name
        self.journey1 = await self.agent.create_journey(
            title="Journey 1 - Name Collection",
            conditions=[],
            description="First journey to collect name",
        )

        self.name_transition = await self.journey1.initial_state.transition_to(
            chat_state="Ask for name",
            canned_responses=[self.step1_response],
        )

        # Journey 2: Collect favorite color
        self.journey2 = await self.agent.create_journey(
            title="Journey 2 - Color Collection",
            conditions=[],
            description="Second journey to collect favorite color",
        )

        self.color_transition = await self.journey2.initial_state.transition_to(
            chat_state="Ask for favorite color",
            canned_responses=[self.step2_response],
        )

        # Journey 3: Final completion
        self.journey3 = await self.agent.create_journey(
            title="Journey 3 - Completion",
            conditions=[],
            description="Third journey to complete process",
        )

        self.completion_transition = await self.journey3.initial_state.transition_to(
            chat_state="Complete the process",
            canned_responses=[self.step3_response],
        )

        # Main journey that chains all three journeys
        self.main_journey = await self.agent.create_journey(
            title="Main Journey",
            conditions=["Customer wants to start process"],
            description="Main journey that connects the three sub-journeys",
        )

        # Chain the journeys at the main level: Main -> Journey1 -> Journey2 -> Journey3
        # First transition: Main -> Journey 1 (name collection)
        self.link1 = await self.main_journey.initial_state.transition_to(
            journey=self.journey1,
        )

        # Second transition: After name collected -> Journey 2 (color collection)
        self.link2 = await self.link1.target.transition_to(
            journey=self.journey2,
        )

        # Third transition: After color collected -> Journey 3 (completion)
        self.link3 = await self.link2.target.transition_to(
            journey=self.journey3,
        )

    async def run(self, ctx: Context) -> None:
        # Test the complete flow through all three journeys
        response1 = await ctx.send_and_receive_message(
            "I want to start the process",
            recipient=self.agent,
            reuse_session=True,
        )
        assert response1 == "Please tell me your name."

        response2 = await ctx.send_and_receive_message(
            "My name is Alice",
            recipient=self.agent,
            reuse_session=True,
        )
        assert response2 == "What's your favorite color?"

        response3 = await ctx.send_and_receive_message(
            "Blue",
            recipient=self.agent,
            reuse_session=True,
        )
        assert response3 == "All done! Thank you for completing all steps."


@pytest.mark.engine
class Test_that_journey_is_not_reevaluated_when_no_associated_tool_is_called(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Bank Agent",
            description="Just a bank test agent",
        )

        @tool
        def check_balance(
            context: ToolContext,
        ) -> ToolResult:
            return ToolResult(data={"balance": 500})

        await self.agent.create_guideline(
            condition="Customer asks for account balance",
            action="Tell him his account balance",
            tools=[check_balance],
        )

        self.journey = await self.agent.create_journey(
            title="Customer Greeting Journey",
            conditions=["Customer greets you"],
            description="Greet customers with personalized responses",
        )

        self.initial_transition = await self.journey.initial_state.transition_to(
            chat_state="Greet him with 'Howdy!'",
            condition="Customer greets you",
        )

        self.second_transition = await self.initial_transition.target.transition_to(
            chat_state="Greet the customer to our bank with 'Hahoy!'",
            condition="The customer's account balance is known",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            "Good morning! What's my balance, please?", recipient=self.agent
        )

        assert "500" in response
        assert "Howdy" in response
        assert "Hahoy" not in response


class Test_that_ready_event_contains_matched_guidelines_journeys_and_states(SDKTest):
    """Test that the ready event with stage=completed contains match data."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Test agent for match data verification",
        )

        # Create a guideline
        self.guideline = await self.agent.create_guideline(
            condition="Customer greets you",
            action="Greet them back warmly",
        )

        # Create a journey with a custom state ID
        self.journey = await self.agent.create_journey(
            title="Greeting Journey",
            conditions=["Customer greets you"],
            description="Handle customer greetings",
        )

        # Create a state with a custom ID
        self.transition = await self.journey.initial_state.transition_to(
            chat_state="Say hello back warmly",
            condition="Customer greets you",
            id=p.JourneyStateId("test-greeting-state"),
        )

    async def run(self, ctx: Context) -> None:
        # Create a session and send a message
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )

        customer_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Hello there!",
        )

        # Wait for the agent to respond
        await ctx.client.sessions.list_events(
            session_id=session.id,
            min_offset=customer_event.offset,
            source="ai_agent",
            kinds="message",
            wait_for_data=30,
        )

        # Get all status events
        status_events = await ctx.client.sessions.list_events(
            session_id=session.id,
            source="ai_agent",
            kinds="status",
        )

        # Find the ready event with stage=completed
        ready_completed_events = []
        for e in status_events:
            if e.data is None:
                continue
            inner_data = e.data.get("data", {})
            if not isinstance(inner_data, dict):
                continue
            if e.data.get("status") == "ready" and inner_data.get("stage") == "completed":
                ready_completed_events.append(e)

        assert len(ready_completed_events) >= 1, "Expected at least one ready/completed event"

        ready_event = ready_completed_events[-1]  # Get the last one
        assert ready_event.data is not None
        event_data = ready_event.data.get("data", {})
        assert isinstance(event_data, dict)

        # Verify matched_guidelines is present and contains guideline IDs
        assert "matched_guidelines" in event_data, "matched_guidelines not found in ready event"
        matched_guidelines = event_data["matched_guidelines"]
        assert isinstance(matched_guidelines, list), "matched_guidelines should be a list"

        # Verify matched_journeys is present
        assert "matched_journeys" in event_data, "matched_journeys not found in ready event"
        matched_journeys = event_data["matched_journeys"]
        assert isinstance(matched_journeys, list), "matched_journeys should be a list"

        # Verify matched_journey_states is present
        assert "matched_journey_states" in event_data, (
            "matched_journey_states not found in ready event"
        )
        matched_journey_states = event_data["matched_journey_states"]
        assert isinstance(matched_journey_states, list), "matched_journey_states should be a list"

        # Verify structure - each should have an "id" key
        for g in matched_guidelines:
            assert "id" in g, "Each matched guideline should have an 'id' key"

        for j in matched_journeys:
            assert "id" in j, "Each matched journey should have an 'id' key"

        for s in matched_journey_states:
            assert "id" in s, "Each matched journey state should have an 'id' key"


class Test_that_custom_state_id_is_used_when_provided(SDKTest):
    """Test that providing a custom ID to transition_to uses that ID."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Test agent for custom state ID",
        )

        self.journey = await self.agent.create_journey(
            title="Test Journey",
            conditions=["Customer greets you"],
            description="Test journey",
        )

        # Create a state with a custom ID
        self.custom_state_id = p.JourneyStateId("my-custom-state-id")
        self.transition = await self.journey.initial_state.transition_to(
            chat_state="Test action",
            condition="Test condition",
            id=self.custom_state_id,
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        # Read the node and verify the ID
        node = await journey_store.read_node(self.custom_state_id)

        assert node.id == self.custom_state_id, f"Expected ID {self.custom_state_id}, got {node.id}"


class Test_that_journey_retriever_runs_when_journey_is_active(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Retriever Agent",
            description="Agent for testing journey retrievers",
        )

        journey = await self.agent.create_journey(
            title="Secret Journey",
            description="A journey about secrets",
            conditions=["the user wants to learn secrets"],
        )

        async def my_retriever(ctx: p.RetrieverContext) -> p.RetrieverResult:
            return p.RetrieverResult(data="The journey secret is 99")

        await journey.attach_retriever(my_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="I want to learn secrets",
            recipient=self.agent,
        )
        assert "99" in response


class Test_that_journey_retriever_does_not_run_when_journey_is_inactive(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Retriever Agent",
            description="Agent for testing journey retrievers",
        )

        self.retriever_called = False

        journey = await self.agent.create_journey(
            title="Secret Journey",
            description="A journey about secrets",
            conditions=["the user wants to learn secrets"],
        )

        async def my_retriever(ctx: p.RetrieverContext) -> p.RetrieverResult:
            self.retriever_called = True
            return p.RetrieverResult(data="The journey secret is 99")

        await journey.attach_retriever(my_retriever)

    async def run(self, ctx: Context) -> None:
        # Ask about something unrelated, journey should not be active
        await ctx.send_and_receive_message(
            customer_message="What is the weather like today?",
            recipient=self.agent,
        )
        assert not self.retriever_called, "Retriever should not be called when journey is inactive"


class Test_that_journey_on_match_is_called_when_journey_without_states_is_activated(SDKTest):
    """Test that journey on_match handler is called when a journey without states is activated."""

    async def setup(self, server: p.Server) -> None:
        self.on_match_called = False
        self.captured_journey_id = None

        async def on_match_handler(_ctx: p.EngineContext, match: p.JourneyMatch) -> None:
            self.on_match_called = True
            self.captured_journey_id = match.journey_id

        self.agent = await server.create_agent(
            name="Journey Handler Agent",
            description="Agent for testing journey on_match handler",
        )

        self.journey = await self.agent.create_journey(
            title="Simple Journey",
            description="A journey without any states",
            conditions=["Customer asks about ordering"],
            on_match=on_match_handler,
        )

        # Add a scoped guideline so the journey has some effect
        await self.journey.create_guideline(
            matcher=p.Guideline.MATCH_ALWAYS,
            action="Offer the customer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I'd like to order something",
            recipient=self.agent,
        )

        assert self.on_match_called, "Journey on_match handler should have been called"
        assert self.captured_journey_id == self.journey.id, (
            f"Expected journey ID {self.journey.id}, got {self.captured_journey_id}"
        )


class Test_that_journey_on_match_is_called_when_journey_with_states_is_activated(SDKTest):
    """Test that journey on_match handler is called when a journey with states is activated."""

    async def setup(self, server: p.Server) -> None:
        self.on_match_called = False
        self.captured_journey_id = None

        async def on_match_handler(_ctx: p.EngineContext, match: p.JourneyMatch) -> None:
            self.on_match_called = True
            self.captured_journey_id = match.journey_id

        self.agent = await server.create_agent(
            name="Journey Handler Agent",
            description="Agent for testing journey on_match handler with states",
        )

        self.journey = await self.agent.create_journey(
            title="Stateful Journey",
            description="A journey with states",
            conditions=["Customer wants to order a pizza"],
            on_match=on_match_handler,
        )

        # Add states to the journey
        self.transition = await self.journey.initial_state.transition_to(
            chat_state="Ask the customer what toppings they want",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I want to order a pizza",
            recipient=self.agent,
        )

        assert self.on_match_called, "Journey on_match handler should have been called"
        assert self.captured_journey_id == self.journey.id, (
            f"Expected journey ID {self.journey.id}, got {self.captured_journey_id}"
        )


class Test_that_journey_on_match_is_called_when_linked_journey_is_activated(SDKTest):
    """Test that journey on_match handler is called when a linked journey is activated."""

    async def setup(self, server: p.Server) -> None:
        self.parent_on_match_called = False
        self.linked_on_match_called = False
        self.parent_journey_id = None
        self.linked_journey_id = None

        async def parent_on_match_handler(_ctx: p.EngineContext, match: p.JourneyMatch) -> None:
            self.parent_on_match_called = True
            self.parent_journey_id = match.journey_id

        async def linked_on_match_handler(_ctx: p.EngineContext, match: p.JourneyMatch) -> None:
            self.linked_on_match_called = True
            self.linked_journey_id = match.journey_id

        self.agent = await server.create_agent(
            name="Linked Journey Agent",
            description="Agent for testing linked journey on_match handlers",
            composition_mode=p.CompositionMode.STRICT,
        )

        # Create canned responses for deterministic testing
        self.ask_room_response = await server.create_canned_response(
            template="Would you like the red room or the blue room?"
        )
        self.ask_name_response = await server.create_canned_response(
            template="Please provide your name for verification."
        )

        # Create a linked journey (will be activated via link, not via conditions)
        self.linked_journey = await self.agent.create_journey(
            title="User Validation",
            description="Validate the user",
            conditions=[],  # No conditions - activated only via link
            on_match=linked_on_match_handler,
        )

        # Add a state to the linked journey
        await self.linked_journey.initial_state.transition_to(
            chat_state="Ask the customer for their name",
            canned_responses=[self.ask_name_response],
        )

        # Create the parent journey that links to the validation journey
        self.parent_journey = await self.agent.create_journey(
            title="Hotel Booking",
            description="Book a hotel room",
            conditions=["Customer wants to book a hotel"],
            on_match=parent_on_match_handler,
        )

        # First state: ask for room type
        self.room_transition = await self.parent_journey.initial_state.transition_to(
            chat_state="Ask the customer which room they want",
            canned_responses=[self.ask_room_response],
        )

        # Link to the validation journey
        await self.room_transition.target.transition_to(
            journey=self.linked_journey,
        )

    async def run(self, ctx: Context) -> None:
        # First message activates parent journey
        await ctx.send_and_receive_message(
            customer_message="I want to book a hotel room",
            recipient=self.agent,
            reuse_session=True,
        )

        assert self.parent_on_match_called, (
            "Parent journey on_match handler should have been called"
        )
        assert self.parent_journey_id == self.parent_journey.id, (
            f"Expected parent journey ID {self.parent_journey.id}, got {self.parent_journey_id}"
        )

        # Second message triggers transition to linked journey
        await ctx.send_and_receive_message(
            customer_message="I want the blue room",
            recipient=self.agent,
            reuse_session=True,
        )

        assert self.linked_on_match_called, (
            "Linked journey on_match handler should have been called"
        )
        assert self.linked_journey_id == self.linked_journey.id, (
            f"Expected linked journey ID {self.linked_journey.id}, got {self.linked_journey_id}"
        )
