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
        assert journey.root is not None
        assert journey.title == "Greeting the customer"
        assert journey.description == "1. Offer the customer a Pepsi"


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
            description="Offer the customer a Pepsi",
        )

        await self.journey.root.connect(action="offer a Pepsi")

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive("Hello there", recipient=self.agent)

        assert await nlp_test(
            context=response,
            condition="There is an offering of a Pepsi",
        )


class Test_that_journey_edge_and_node_can_be_created_with_connection(SDKTest):
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

        self.edge_w = await self.journey.root.connect(action="check room availability")
        self.edge_x = await self.edge_w.target.connect(action="provide hotel amenities")

    async def run(self, ctx: Context) -> None:
        assert self.edge_w in self.journey.edges
        assert self.edge_x in self.journey.edges

        assert self.edge_w.source.id == self.journey.root.id
        assert self.edge_w.target.action == "check room availability"
        assert self.edge_w.target in self.journey.nodes

        assert self.edge_x.source.id == self.edge_w.target.id
        assert self.edge_x.target.action == "provide hotel amenities"
        assert self.edge_x.target in self.journey.nodes


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

        self.edge = await self.journey.root.connect(
            action="check available upgrades", tools=[test_tool]
        )

    async def run(self, ctx: Context) -> None:
        node = self.edge.target

        assert node.tools

        assert len(node.tools) == 1
        assert node.tools[0].tool.name == "test_tool"


class Test_that_journey_node_can_be_connected_with_condition(SDKTest):
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

        self.edge_x = await self.journey.root.connect(action="ask if the customer wants breakfast")
        self.edge_y = await self.edge_x.target.connect(
            condition="if the customer says yes", action="add breakfast to booking"
        )
        self.edge_z = await self.edge_x.target.connect(
            condition="if the customer says no", action="proceed without breakfast"
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        edges = self.journey.edges
        nodes = self.journey.nodes

        assert {e.id for e in edges}.issuperset({self.edge_x.id, self.edge_y.id, self.edge_z.id})

        assert {n.id for n in nodes}.issuperset(
            {
                self.edge_x.source.id,
                self.edge_x.target.id,
                self.edge_y.target.id,
                self.edge_z.target.id,
            }
        )

        store_edges = await journey_store.list_edges(journey_id=self.journey.id)
        store_nodes = await journey_store.list_nodes(journey_id=self.journey.id)

        assert {e.id for e in store_edges}.issuperset(
            {self.edge_x.id, self.edge_y.id, self.edge_z.id}
        )
        assert {n.id for n in store_nodes}.issuperset(
            {
                self.edge_x.source.id,
                self.edge_x.target.id,
                self.edge_y.target.id,
                self.edge_z.target.id,
            }
        )

        assert self.edge_y.condition == "if the customer says yes"
        assert self.edge_z.condition == "if the customer says no"


class Test_that_if_node_has_more_than_one_connect_they_all_need_to_have_conditions(SDKTest):
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

        self.edge_ask_breakfast = await self.journey.root.connect(
            action="ask if the customer wants breakfast"
        )

        self.edge_add_breakfast = await self.edge_ask_breakfast.target.connect(
            condition="if the customer says yes",
            action="add breakfast to booking",
        )

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await self.edge_ask_breakfast.target.connect(action="proceed without breakfast")


class Test_that_journey_is_reevaluated_after_tool_call(SDKTest):
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

        self.edge_check_balance = await self.journey.root.connect(
            action="check customer account balance", tools=[check_balance]
        )

        self.edge_offer_discount = await self.edge_check_balance.target.connect(
            condition="if balance is low",
            action="offer discount if balance is low",
        )

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationships = await relationship_store.list_relationships(
            kind=RelationshipKind.REEVALUATION,
            target_id=Tag.for_journey_node_id(
                self.edge_check_balance.target.id,
            ),
        )

        assert relationships
        assert len(relationships) == 1
        assert relationships[0].kind == RelationshipKind.REEVALUATION
        assert relationships[0].source.id == ToolId(
            service_name=p.INTEGRATED_TOOL_SERVICE_NAME, tool_name="check_balance"
        )
        assert relationships[0].target.id == Tag.for_journey_node_id(
            self.edge_check_balance.target.id,
        )
