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
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.tools import ToolContext, ToolResult
import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest


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

        relationship = await relationship_store.read_relationship(id=self.relationship.id)
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

        relationship = await relationship_store.read_relationship(id=self.relationship.id)
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

        relationship = await relationship_store.read_relationship(id=self.relationship.id)
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

        relationship = await relationship_store.read_relationship(id=self.relationship.id)
        assert relationship.kind == RelationshipKind.REEVALUATION
