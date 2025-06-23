import pytest
from parlant.core.relationships import RelationshipKind, RelationshipStore
import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest


class Test_that_guideline_priority_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.g1 = await self.agent.create_guideline(condition="Condition 1", action="Action 1")
        self.g2 = await self.agent.create_guideline(condition="Condition 2", action="Action 2")

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

        self.g1 = await self.agent.create_guideline(condition="Condition 1", action="Action 1")
        self.g2 = await self.agent.create_guideline(condition="Condition 2", action="Action 3")

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

        self.g1 = await self.agent.create_guideline(condition="Condition 1", action="Action 1")
        self.g2 = await self.agent.create_guideline(condition="Condition 2", action="Action 2")

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

        self.g1 = await self.agent.create_guideline(condition="Ambiguous 1")
        self.g2 = await self.agent.create_guideline(condition="Ambiguous 2")
        self.g3 = await self.agent.create_guideline(condition="Ambiguous 3")

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

        self.g1 = await self.agent.create_guideline(condition="Only one")
        self.g2 = await self.agent.create_guideline(condition="Target")

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await self.g1.disambiguate([self.g2])
