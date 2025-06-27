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

from textwrap import dedent

from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.tags import Tag
from parlant.core.tools import ToolContext, ToolId, ToolResult
from tests.sdk.utils import Context, SDKTest, get_message
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
            description=dedent("""\
                1. Offer the customer a Pepsi
            """),
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
            conditions=["the customer greets you", "customer say Howdy"],
            description=dedent("""\
                1. Offer the customer a Pepsi
            """),
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]
        guideline_store = ctx.container[GuidelineStore]

        journey = await journey_store.read_journey(journey_id=self.journey.id)

        assert journey

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
            conditions=["the customer greets you", "customer say Howdy"],
            description=dedent("""\
                1. Offer the customer a Pepsi
            """),
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]
        guideline_store = ctx.container[GuidelineStore]

        journey = await journey_store.read_journey(journey_id=self.journey.id)

        assert journey

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
            conditions=["the customer greets you", "customer say Howdy"],
            description=dedent("""\
                1. Offer the customer a Pepsi
            """),
        )

        self.guideline = await self.journey.create_guideline(
            condition="Greeting the user",
            action="Get Pepsi price",
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

        self.guideline = await self.agent.create_guideline(condition="the customer greets you")

        self.journey = await self.agent.create_journey(
            title="Greeting the customer",
            conditions=[self.guideline],
            description=dedent("""\
                1. Offer the customer a Pepsi
            """),
        )

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]
        guideline_store = ctx.container[GuidelineStore]

        journey = await journey_store.read_journey(journey_id=self.journey.id)
        guideline = await guideline_store.read_guideline(guideline_id=self.guideline.id)

        assert journey
        assert journey.conditions == [guideline.id]

        assert guideline
        assert guideline.id == self.guideline.id


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
            description=dedent("""\
                1. Offer the customer a Pepsi
            """),
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )

        event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Hello there",
        )

        agent_messages = await ctx.client.sessions.list_events(
            session_id=session.id,
            min_offset=event.offset,
            source="ai_agent",
            kinds="message",
            wait_for_data=10,
        )

        assert len(agent_messages) == 1

        assert nlp_test(
            context=get_message(agent_messages[0]),
            condition="There is an offering of a Pepsi",
        )


class Test_that_journey_steps_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey step creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Step Journey",
            conditions=[],
            description="A journey with multiple steps",
        )

        self.step_w = await self.journey.create_step(description="Do W")
        self.step_x = await self.journey.create_step(description="Do X")

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]

        assert self.journey.steps[0] == self.step_w
        assert self.journey.steps[1] == self.step_x

        guidelines = await guideline_store.list_guidelines()

        assert any(self.step_w.guideline.id == g.id for g in guidelines)
        assert any(self.step_x.guideline.id == g.id for g in guidelines)


class Test_that_journey_step_can_connect_to_a_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey step creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Step Journey",
            conditions=[],
            description="A journey with multiple steps",
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.step = await self.journey.create_step(description="Do Something", tools=[test_tool])

    async def run(self, ctx: Context) -> None:
        guideline_tooL_store = ctx.container[GuidelineToolAssociationStore]

        assert len(self.journey.steps) == 1

        assert self.journey.steps[0] == self.step

        associations = await guideline_tooL_store.list_associations()
        assert associations
        assert len(associations) == 1

        association = associations[0]
        assert association.guideline_id == self.step.guideline.id
        assert association.tool_id == ToolId(
            service_name=p.INTEGRATED_TOOL_SERVICE_NAME, tool_name="test_tool"
        )


class Test_that_journey_sub_steps_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Sub-steps Agent",
            description="Agent for journey sub-step creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Sub-step Journey",
            conditions=[],
            description="A journey with a step and sub-step",
        )

        self.step_x = await self.journey.create_step(description="Do X")
        self.sub_step_y = await self.step_x.create_sub_step(description="Do Y")
        self.sub_step_z = await self.step_x.create_sub_step(description="Do Z")

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]

        assert self.step_x
        assert len(self.step_x.sub_steps) == 2
        assert self.step_x.sub_steps[0] == self.sub_step_y
        assert self.step_x.sub_steps[1] == self.sub_step_z

        guidelines = await guideline_store.list_guidelines()

        assert any(self.sub_step_y.guideline.id == g.id for g in guidelines)
        assert any(self.sub_step_z.guideline.id == g.id for g in guidelines)


class Test_that_journey_step_guideline_metadata_includes_sub_steps(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for step metadata test",
        )

        self.journey = await self.agent.create_journey(
            title="Meta Journey",
            description="Test journey for sub-step metadata",
            conditions=[],
        )

        self.step = await self.journey.create_step(description="Parent Step")
        self.sub_step = await self.step.create_sub_step(description="Child Sub-step")

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]

        parent_guideline = await guideline_store.read_guideline(guideline_id=self.step.guideline.id)

        journey_steps_metadata = parent_guideline.metadata.get("journey_step", {})
        if isinstance(journey_steps_metadata, dict):
            sub_steps = journey_steps_metadata.get("sub_steps", [])
        else:
            sub_steps = []
        assert self.sub_step.guideline.id in sub_steps


class Test_that_journey_steps_and_sub_steps_are_ordered(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for step metadata test",
        )

        self.journey = await self.agent.create_journey(
            title="Meta Journey",
            description="Test journey for sub-step metadata",
            conditions=[],
        )

        self.step1 = await self.journey.create_step(description="First Step")
        self.step2 = await self.journey.create_step(description="Second Step")
        self.step3 = await self.journey.create_step(description="Third Step")

        self.sub_step11 = await self.step1.create_sub_step(description="First Sub-Step for step 1")
        self.sub_step12 = await self.step1.create_sub_step(description="Second Sub-Step for step 1")
        self.sub_step13 = await self.step1.create_sub_step(description="Third Sub-Step for step 1")

        self.sub_step21 = await self.step2.create_sub_step(description="First Sub-Step for step 2")
        self.sub_step22 = await self.step2.create_sub_step(description="Second Sub-Step for step 2")

        self.sub_step31 = await self.step3.create_sub_step(description="First Sub-Step for step 3")

    async def run(self, ctx: Context) -> None:
        journey_store = ctx.container[JourneyStore]

        assert len(self.journey.steps) == 3
        assert len(self.journey.steps[0].sub_steps) == 3
        assert len(self.journey.steps[1].sub_steps) == 2
        assert len(self.journey.steps[2].sub_steps) == 1

        journey = await journey_store.read_journey(journey_id=self.journey.id)

        assert len(journey.steps) == 9
        assert journey.steps[0] == self.step1.guideline.id
        assert journey.steps[1] == self.sub_step11.guideline.id
        assert journey.steps[2] == self.sub_step12.guideline.id
        assert journey.steps[3] == self.sub_step13.guideline.id
        assert journey.steps[4] == self.step2.guideline.id
        assert journey.steps[5] == self.sub_step21.guideline.id
        assert journey.steps[6] == self.sub_step22.guideline.id
        assert journey.steps[7] == self.step3.guideline.id
        assert journey.steps[8] == self.sub_step31.guideline.id


class Test_that_journey_sub_step_reevaluate_after_journey_step_that_only_running_tools(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for journey step creation tests",
            composition_mode=p.CompositionMode.COMPOSITED_UTTERANCE,
        )

        self.journey = await self.agent.create_journey(
            title="Step Journey",
            conditions=[],
            description="A journey with multiple steps",
        )

        @tool
        def check_balance(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.step = await self.journey.create_step(
            description="Check customer balance", tools=[check_balance]
        )

        self.sub_step = await self.step.create_sub_step(
            description="If balance is low, offer a discount",
        )

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationships = await relationship_store.list_relationships(
            kind=RelationshipKind.REEVALUATION,
            target_id=self.sub_step.guideline.id,
        )

        assert relationships
        assert len(relationships) == 1
        assert relationships[0].kind == RelationshipKind.REEVALUATION
        assert relationships[0].source.id == ToolId(
            service_name=p.INTEGRATED_TOOL_SERVICE_NAME, tool_name="check_balance"
        )
        assert relationships[0].target.id == self.sub_step.guideline.id
