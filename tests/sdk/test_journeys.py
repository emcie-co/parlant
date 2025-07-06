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

from typing import cast

import pytest

from parlant.core.common import JSONSerializable
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.guidelines import GuidelineId, GuidelineStore
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
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
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


class Test_that_condition_guidelines_are_tagged_for_created_journey(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Store agent",
            description="You work at a store and help customers",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
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
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
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
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
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
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
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
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Greeting the customer",
            conditions=["the customer greets you"],
            description="1. Offer the customer a Pepsi",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive("Hello there", recipient=self.agent)

        assert nlp_test(
            context=response,
            condition="There is an offering of a Pepsi",
        )


class Test_that_journey_nodes_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey node creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Node Journey",
            conditions=[],
            description="A journey with multiple nodes",
        )

        self.node_w = await self.journey.root.link(action="check room availability")
        self.node_x = await self.journey.root.link(action="provide hotel amenities")

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]
        assert self.journey.root.forward_links[0] == self.node_w
        assert self.journey.root.forward_links[1] == self.node_x

        guidelines = await guideline_store.list_guidelines()
        assert any(cast(GuidelineId, self.node_w.id) == g.id for g in guidelines)
        assert any(cast(GuidelineId, self.node_x.id) == g.id for g in guidelines)


class Test_that_journey_node_can_connect_to_a_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey node creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Node Journey",
            conditions=[],
            description="A journey with multiple nodes",
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.node = await self.journey.root.link(
            action="check available upgrades", tools=[test_tool]
        )

    async def run(self, ctx: Context) -> None:
        guideline_tool_store = ctx.container[GuidelineToolAssociationStore]

        assert len(self.journey.root.forward_links) == 1
        assert self.journey.root.forward_links[0] == self.node

        associations = await guideline_tool_store.list_associations()
        assert associations
        assert len(associations) == 1

        association = associations[0]
        assert association.guideline_id == cast(GuidelineId, self.node.id)
        assert association.tool_id == ToolId(
            service_name=p.INTEGRATED_TOOL_SERVICE_NAME, tool_name="test_tool"
        )


class Test_that_journey_node_can_be_linked_with_condition(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey conditioned nodes Agent",
            description="Agent for journey node with condition creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Conditioned-nodes Journey",
            conditions=[],
            description="A journey with nodes depending on customer decisions",
        )

        self.node_x = await self.journey.root.link(action="ask if the customer wants breakfast")
        self.sub_node_y = await self.node_x.link(
            condition="if the customer says yes", action="add breakfast to booking"
        )
        self.sub_node_z = await self.node_x.link(
            condition="if the customer says no", action="proceed without breakfast"
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]

        assert self.node_x
        assert self.node_x.forward_links[0] == self.sub_node_y
        assert self.node_x.forward_links[1] == self.sub_node_z

        guidelines = await guideline_store.list_guidelines()
        assert any(cast(GuidelineId, self.sub_node_y.id) == g.id for g in guidelines)
        assert any(cast(GuidelineId, self.sub_node_z.id) == g.id for g in guidelines)

        guideline_y = await guideline_store.read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node_y.id)
        )
        guideline_z = await guideline_store.read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node_z.id)
        )

        assert guideline_y.content.condition == "if the customer says yes"
        assert guideline_y.content.action == "add breakfast to booking"

        assert guideline_z.content.condition == "if the customer says no"
        assert guideline_z.content.action == "proceed without breakfast"


class Test_that_if_node_has_more_than_one_link_they_all_need_to_have_conditions(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey conditioned nodes Agent",
            description="Agent for journey node with condition creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Conditioned-nodes Journey",
            conditions=[],
            description="A journey with nodes depending on customer decisions",
        )

        self.node_x = await self.journey.root.link(action="ask if the customer wants breakfast")
        self.sub_node_y = await self.node_x.link(
            condition="if the customer says yes", action="add breakfast to booking"
        )

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await self.node_x.link(action="proceed without breakfast")


class Test_that_journey_sub_step_guideline_metadata_includes_sub_steps(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for step metadata test",
        )

        self.journey = await self.agent.create_journey(
            title="Booking Flow Journey",
            description="Test journey for sub-step metadata",
            conditions=[],
        )

        self.node = await self.journey.root.link(action="ask if the customer wants to book a room")
        self.sub_node = await self.node.link(
            condition="if the customer says yes", action="ask for check-in date"
        )
        self.sub_sub_node = await self.sub_node.link(
            condition="if a date is provided", action="confirm the reservation"
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]

        parent_guideline = await guideline_store.read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node.id)
        )
        journey_steps_metadata = parent_guideline.metadata.get("journey_step", {})
        if isinstance(journey_steps_metadata, dict):
            sub_steps = journey_steps_metadata.get("sub_steps", [])
        else:
            sub_steps = []
        assert self.sub_sub_node.id in sub_steps


class Test_that_journey_steps_and_sub_nodes_are_ordered(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for step metadata test",
        )

        self.journey = await self.agent.create_journey(
            title="Room Booking Journey",
            description="Ensure journey steps and sub-steps are correctly ordered",
            conditions=[],
        )

        self.node1 = await self.journey.root.link(action="ask if the customer wants to book a room")
        self.node2 = await self.node1.link(action="ask for preferred check-in date")
        self.node3 = await self.node2.link(action="ask for preferred check-out date")

        self.sub_node3_1 = await self.node3.link(
            condition="if the customer agrees", action="proceed to room selection"
        )
        self.sub_node3_2 = await self.node3.link(
            condition="if the customer declines", action="offer alternative suggestions"
        )
        self.sub_node3_3 = await self.node3.link(
            condition="if the customer hesitates", action="highlight booking benefits"
        )

        self.sub_node3_3_1 = await self.sub_node3_3.link(
            condition="if the customer provides a date", action="validate check-in availability"
        )
        self.sub_node3_3_2 = await self.sub_node3_3.link(
            condition="if the customer provides a flexible range", action="suggest date options"
        )

        self.sub_node_3_3_2_1 = await self.sub_node3_3_2.link(
            condition="if the customer confirms a check-out date",
            action="show final booking summary",
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        assert self.journey.root.forward_links[0] == self.node1
        assert self.node1.forward_links[0] == self.node2
        assert self.node2.forward_links[0] == self.node3

        assert self.node3.forward_links[0] == self.sub_node3_1
        assert self.node3.forward_links[1] == self.sub_node3_2
        assert self.node3.forward_links[2] == self.sub_node3_3

        assert self.sub_node3_3.forward_links[0] == self.sub_node3_3_1
        assert self.sub_node3_3.forward_links[1] == self.sub_node3_3_2

        assert self.sub_node3_3_2.forward_links[0] == self.sub_node_3_3_2_1

        journey = await journey_store.read_journey(journey_id=self.journey.id)

        assert len(journey.steps) == 9
        assert journey.steps[0] == self.node1.id
        assert journey.steps[1] == self.node2.id
        assert journey.steps[2] == self.node3.id
        assert journey.steps[3] == self.sub_node3_1.id
        assert journey.steps[4] == self.sub_node3_2.id
        assert journey.steps[5] == self.sub_node3_3.id
        assert journey.steps[6] == self.sub_node3_3_1.id
        assert journey.steps[7] == self.sub_node3_3_2.id
        assert journey.steps[8] == self.sub_node_3_3_2_1.id

        node1_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.node1.id)
        )
        node2_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.node2.id)
        )
        node3_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.node3.id)
        )
        sub_node3_1_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node3_1.id)
        )
        sub_node3_2_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node3_2.id)
        )
        sub_node3_3_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node3_3.id)
        )
        sub_node3_3_1_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node3_3_1.id)
        )
        sub_node3_3_2_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node3_3_2.id)
        )
        sub_node_3_3_2_1_guideline = await ctx.container[GuidelineStore].read_guideline(
            guideline_id=cast(GuidelineId, self.sub_node_3_3_2_1.id)
        )

        assert (
            cast(dict[str, JSONSerializable], node1_guideline.metadata["journey_step"])["id"] == 1
        )
        assert (
            cast(dict[str, JSONSerializable], node2_guideline.metadata["journey_step"])["id"] == 2
        )
        assert (
            cast(dict[str, JSONSerializable], node3_guideline.metadata["journey_step"])["id"] == 3
        )
        assert (
            cast(dict[str, JSONSerializable], sub_node3_1_guideline.metadata["journey_step"])["id"]
            == 4
        )
        assert (
            cast(dict[str, JSONSerializable], sub_node3_2_guideline.metadata["journey_step"])["id"]
            == 5
        )
        assert (
            cast(dict[str, JSONSerializable], sub_node3_3_guideline.metadata["journey_step"])["id"]
            == 6
        )
        assert (
            cast(dict[str, JSONSerializable], sub_node3_3_1_guideline.metadata["journey_step"])[
                "id"
            ]
            == 7
        )
        assert (
            cast(dict[str, JSONSerializable], sub_node3_3_2_guideline.metadata["journey_step"])[
                "id"
            ]
            == 8
        )
        assert (
            cast(dict[str, JSONSerializable], sub_node_3_3_2_1_guideline.metadata["journey_step"])[
                "id"
            ]
            == 9
        )


class Test_that_journey_sub_node_reevaluate_after_journey_step_that_only_running_tools(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey step creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Step Journey",
            conditions=[],
            description="A journey with tool-driven decision steps",
        )

        @tool
        def check_balance(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.node = await self.journey.root.link(
            action="check customer account balance", tools=[check_balance]
        )

        self.sub_node = await self.node.link(
            condition="if balance is low",
            action="offer discount if balance is low",
        )

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationships = await relationship_store.list_relationships(
            kind=RelationshipKind.REEVALUATION,
            target_id=cast(GuidelineId, self.sub_node.id),
        )

        assert relationships
        assert len(relationships) == 1
        assert relationships[0].kind == RelationshipKind.REEVALUATION
        assert relationships[0].source.id == ToolId(
            service_name=p.INTEGRATED_TOOL_SERVICE_NAME, tool_name="check_balance"
        )
        assert relationships[0].target.id == cast(GuidelineId, self.sub_node.id)


class Test_that_journey_sub_sub_node_reevaluate_after_journey_sub_node_that_only_running_tools(
    SDKTest
):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey step creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Travel Discount Journey",
            conditions=[],
            description="Journey to evaluate seasonal discounts",
        )

        @tool
        def check_season(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.travel_dates_node = await self.journey.root.link(
            action="ask for customer travel dates"
        )
        self.season_sub_node = await self.travel_dates_node.link(
            action="check discount for specified dates",
            tools=[check_season],
        )
        self.discount_response_node = await self.season_sub_node.link(
            condition="if discount applies",
            action="inform the customer with a light joke",
        )
        self.no_discount_node = await self.season_sub_node.link(
            condition="if discount does not applies",
            action="apologize and explain there is no discount",
        )

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationships = await relationship_store.list_relationships(
            kind=RelationshipKind.REEVALUATION,
        )

        assert relationships
        assert len(relationships) == 2
        assert all(
            r.source.id
            == ToolId(service_name=p.INTEGRATED_TOOL_SERVICE_NAME, tool_name="check_season")
            for r in relationships
        )
        assert any(
            r.target.id == cast(GuidelineId, self.discount_response_node.id) for r in relationships
        )
        assert any(
            r.target.id == cast(GuidelineId, self.no_discount_node.id) for r in relationships
        )
