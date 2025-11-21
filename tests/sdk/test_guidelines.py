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
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.tags import Tag
from parlant.core.tools import ToolContext, ToolResult
from parlant.core.canned_responses import CannedResponseStore
import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test


class Test_that_guideline_priority_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.g1 = await self.agent.create_guideline(
            condition="Customer requests a refund",
            action="process the refund if the transaction is not frozen",
        )
        self.g2 = await self.agent.create_guideline(
            condition="An error is detected on an account",
            action="freeze all account transactions",
        )

        self.relationship = await self.g1.prioritize_over(self.g2)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )
        assert relationship.kind == RelationshipKind.PRIORITY


class Test_that_guideline_entailment_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.g1 = await self.agent.create_guideline(
            condition="A customer is visibly upset about the wait",
            action="Transfer the customer to the manager immediately",
        )
        self.g2 = await self.agent.create_guideline(
            condition="A new customer arrives", action="offer to sell pizza"
        )

        self.relationship = await self.g1.entail(self.g2)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )
        assert relationship.kind == RelationshipKind.ENTAILMENT


class Test_that_guideline_dependency_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.g1 = await self.agent.create_guideline(
            condition="A customer asks for the price of tables",
            action="state that a table costs $100",
        )
        self.g2 = await self.agent.create_guideline(
            condition="A customer expresses frustration",
            action="end your response with the word sorry",
        )

        self.relationship = await self.g2.depend_on(self.g2)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )
        assert relationship.kind == RelationshipKind.DEPENDENCY


class Test_that_guideline_disambiguation_creates_relationships(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Disambiguation Agent",
            description="Agent for disambiguation",
        )

        self.g1 = await self.agent.create_guideline(condition="A customer says they are thirsty")
        self.g2 = await self.agent.create_guideline(condition="A customer says hello")
        self.g3 = await self.agent.create_guideline(
            condition="A customer asks about pizza toppings"
        )

        self.relationships = await self.g1.disambiguate([self.g2, self.g3])

    async def run(self, ctx: Context) -> None:
        assert len(self.relationships) == 2

        for rel in self.relationships:
            assert rel.kind == RelationshipKind.DISAMBIGUATION
            assert rel.source == self.g1.id
            assert rel.target in [self.g2.id, self.g3.id]


class Test_that_attempting_to_disambiguate_a_single_target_raises_an_error(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Error Agent",
            description="Agent for error test",
        )

        self.g1 = await self.agent.create_guideline(condition="Customer asks for a recommendation")
        self.g2 = await self.agent.create_guideline(condition="Customer asks about available soups")

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await self.g1.disambiguate([self.g2])


class Test_that_a_reevaluation_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Tool Agent",
            description="Agent for tool test",
            composition_mode=p.CompositionMode.FLUID,
        )

        self.g1 = await self.agent.create_guideline(
            condition="Customer requests to update their contact information"
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.relationship = await self.g1.reevaluate_after(tool=test_tool)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )
        assert relationship.kind == RelationshipKind.REEVALUATION


class Test_that_guideline_can_prioritize_over_journey(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Guideline to Journey Agent",
            description="Agent for guideline to journey priority",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer asks about shipping",
            action="Explain standard shipping policy",
        )

        self.journey = await self.agent.create_journey(
            title="Handle Complaints",
            conditions=["Customer is upset"],
            description="Resolve the complaint flow",
        )

        self.relationship = await self.guideline.prioritize_over(self.journey)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )

        assert relationship.kind == RelationshipKind.PRIORITY
        assert relationship.source.id == self.guideline.id
        assert relationship.target.id == Tag.for_journey_id(self.journey.id)


class Test_that_guideline_can_depend_on_journey(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Guideline to Journey Agent",
            description="Agent for guideline to journey dependency",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer asks about VIP service",
            action="Explain the VIP terms",
        )

        self.journey = await self.agent.create_journey(
            title="VIP Journey",
            conditions=["Customer is a VIP"],
            description="Assist the customer in a premium flow",
        )

        self.relationship = await self.guideline.depend_on(self.journey)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )

        assert relationship.kind == RelationshipKind.DEPENDENCY
        assert relationship.source.id == self.guideline.id
        assert relationship.target.id == Tag.for_journey_id(self.journey.id)


class Test_that_agent_guideline_can_be_created_with_canned_responses(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Canned Response Agent",
            description="Agent for testing canned response associations",
        )

        self.canrep1 = await self.agent.create_canned_response(
            template="Thank you for your inquiry about {topic}."
        )
        self.canrep2 = await self.agent.create_canned_response(
            template="I'll be happy to help you with {request}."
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer asks for help",
            action="Provide assistance",
            canned_responses=[self.canrep1, self.canrep2],
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]

        guideline_tag = Tag.for_guideline_id(self.guideline.id)

        updated_canrep1 = await canrep_store.read_canned_response(self.canrep1)
        updated_canrep2 = await canrep_store.read_canned_response(self.canrep2)

        assert guideline_tag in updated_canrep1.tags
        assert guideline_tag in updated_canrep2.tags


class Test_that_agent_observation_can_be_created_with_canned_responses(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Observation Agent",
            description="Agent for testing observation with canned responses",
        )

        self.canrep = await self.agent.create_canned_response(
            template="I notice you seem {emotion}."
        )

        self.observation = await self.agent.create_observation(
            condition="Customer appears frustrated",
            canned_responses=[self.canrep],
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]

        updated_canrep = await canrep_store.read_canned_response(self.canrep)

        assert Tag.for_guideline_id(self.observation.id) in updated_canrep.tags


class Test_that_agent_guideline_can_be_created_with_metadata(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="Agent for testing guideline metadata",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer requests a callback",
            action="Schedule a callback within 24 hours",
            metadata={"continuous": True, "agent_intention_condition": "Test another property"},
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]

        guideline = await guideline_store.read_guideline(self.guideline.id)

        assert guideline.metadata["continuous"] is True
        assert guideline.metadata["agent_intention_condition"] == "Test another property"


class Test_that_guideline_can_use_custom_matcher(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            description="Dummy agent",
        )

        self.guideline = await self.agent.create_guideline(
            condition="",
            action="Offer a banana",
            matcher=p.Guideline.MATCH_ALWAYS,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive(
            customer_message="Hello, sir.",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It offers a banana")


class Test_that_custom_matcher_can_return_no_match(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            description="Dummy agent",
        )

        async def never_match(
            ctx: p.GuidelineMatchingContext, guideline: p.Guideline
        ) -> p.GuidelineMatch:
            return p.GuidelineMatch(
                matched=False,
                rationale="Custom matcher never matches",
            )

        self.guideline = await self.agent.create_guideline(
            condition="Customer greets you",
            action="Offer a banana",
            matcher=never_match,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive(
            customer_message="Hello there!",
            recipient=self.agent,
        )

        assert not await nlp_test(answer, "It mentions a banana")


class Test_that_guideline_description_affects_agent_behavior(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            description="Dummy agent",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer asks about Cachookas",
            action="Explain what Cachookas are",
            description="Cachookas are a type of ancient boomerang used to repel flies",
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive(
            customer_message="What are Cachookas?",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It mentions the concept of a boomerang")
