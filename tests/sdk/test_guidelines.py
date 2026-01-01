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

import asyncio
import pytest
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch as _GuidelineMatch,
)
from parlant.core.common import Criticality
from parlant.core.guidelines import GuidelineStore
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.sessions import EventSource
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
        answer = await ctx.send_and_receive_message(
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
                id=guideline.id,
                matched=False,
                rationale="Custom matcher never matches",
            )

        self.guideline = await self.agent.create_guideline(
            condition="Customer greets you",
            action="Offer a banana",
            matcher=never_match,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
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
        answer = await ctx.send_and_receive_message(
            customer_message="What are Cachookas?",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It mentions the concept of a boomerang")


class Test_that_guideline_match_handler_is_called_when_guideline_matches(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Match Handler Agent",
            description="Agent for testing match handlers",
        )

        self.captured_guideline_id = None

        async def match_handler(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.captured_guideline_id = match.id

        self.guideline = await self.agent.create_guideline(
            condition="Customer says hello",
            action="Greet the customer warmly",
            on_match=match_handler,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Hello there!",
            recipient=self.agent,
        )

        assert self.captured_guideline_id == self.guideline.id, (
            "Should capture correct guideline ID"
        )


class Test_that_multiple_match_handlers_can_be_registered_for_same_guideline(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Multiple Handlers Agent",
            description="Agent for testing multiple handlers",
        )

        self.handler1_count = 0
        self.handler2_count = 0

        async def handler1(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.handler1_count += 1

        async def handler2(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.handler2_count += 1

        self.guideline = await self.agent.create_guideline(
            condition="Customer asks for help",
            action="Offer assistance",
            on_match=handler1,
        )

        async def shim_handler2(
            core_ctx: p.EngineContext,
            core_match: _GuidelineMatch,
        ) -> None:
            sdk_match = p.GuidelineMatch(
                id=core_match.guideline.id,
                matched=True,
                rationale=core_match.rationale,
            )
            await handler2(core_ctx, sdk_match)

        server.container[EngineHooks].on_guideline_match_handlers[self.guideline.id].append(
            shim_handler2
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I need help please",
            recipient=self.agent,
        )

        assert self.handler1_count == 1, "Handler 1 should be called once"
        assert self.handler2_count == 1, "Handler 2 should be called once"


class Test_that_match_handlers_for_different_guidelines_are_independent(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Independent Handlers Agent",
            description="Agent for testing independent handlers",
        )

        self.guideline1_handler_called = False
        self.guideline2_handler_called = False

        async def handler1(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.guideline1_handler_called = True

        async def handler2(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.guideline2_handler_called = True

        self.guideline1 = await self.agent.create_guideline(
            condition="Customer mentions pizza",
            action="Recommend pizza toppings",
            on_match=handler1,
        )

        self.guideline2 = await self.agent.create_guideline(
            condition="Customer mentions pasta",
            action="Recommend pasta dishes",
            on_match=handler2,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I'd like to order some pizza",
            recipient=self.agent,
        )

        assert self.guideline1_handler_called, "Guideline 1 handler should be called"
        assert not self.guideline2_handler_called, "Guideline 2 handler should NOT be called"


class Test_that_match_handler_on_journey_guideline_works(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Match Handler Agent",
            description="Agent for testing journey guideline handlers",
        )

        self.journey = await self.agent.create_journey(
            title="Order Something",
            description="Journey to handle orders",
            conditions=["Customer wants to order something"],
        )

        self.handler_called = False

        async def match_handler(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.handler_called = True

        self.guideline = await self.journey.create_guideline(
            condition="Customer wants to order a banana",
            action="Tell them it's an excellent choice",
            on_match=match_handler,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I'd like to order a banana",
            recipient=self.agent,
        )

        assert self.handler_called, "Journey guideline handler should have been called"


class Test_that_guideline_can_be_created_with_custom_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Custom ID Agent",
            description="Agent for testing custom ID functionality",
        )

        self.custom_id = p.GuidelineId("custom-guideline-789")

        self.guideline = await self.agent.create_guideline(
            condition="Customer mentions custom ID requirement",
            action="Provide custom ID assistance",
            id=self.custom_id,
        )

    async def run(self, ctx: Context) -> None:
        # Verify the guideline was created with the custom ID
        assert self.guideline.id == self.custom_id

        # Verify it can be retrieved from the store
        guideline_store = ctx.container[GuidelineStore]
        stored_guideline = await guideline_store.read_guideline(self.custom_id)

        assert stored_guideline.id == self.custom_id
        assert stored_guideline.content.condition == "Customer mentions custom ID requirement"
        assert stored_guideline.content.action == "Provide custom ID assistance"


class Test_that_guideline_creation_fails_with_duplicate_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Duplicate ID Agent",
            description="Agent for testing duplicate ID handling",
        )

        self.duplicate_id = p.GuidelineId("duplicate-guideline-101")

        # Create the first guideline
        self.first_guideline = await self.agent.create_guideline(
            condition="First guideline condition",
            action="First guideline action",
            id=self.duplicate_id,
        )

    async def run(self, ctx: Context) -> None:
        # Verify the first guideline was created
        assert self.first_guideline.id == self.duplicate_id

        # Try to create a second guideline with the same ID
        with pytest.raises(
            ValueError, match=f"Guideline with id '{self.duplicate_id}' already exists"
        ):
            await self.agent.create_guideline(
                condition="Second guideline condition",
                action="Second guideline action",
                id=self.duplicate_id,
            )


class Test_that_only_prioritized_guideline_handler_is_called_when_both_match(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Priority Test Agent",
            description="Agent for testing priority with handlers",
        )

        self.general_handler_called = False
        self.specific_handler_called = False

        async def general_handler(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.general_handler_called = True

        async def specific_handler(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.specific_handler_called = True

        # Create general guideline that would match any help request
        self.general_guideline = await self.agent.create_guideline(
            condition="Customer asks for help",
            action="Provide general help information",
            on_match=general_handler,
        )

        # Create more specific guideline that should take priority
        self.specific_guideline = await self.agent.create_guideline(
            condition="Customer asks for help with billing",
            action="Provide billing-specific help",
            on_match=specific_handler,
        )

        # Make specific guideline prioritize over general guideline
        await self.specific_guideline.prioritize_over(self.general_guideline)

    async def run(self, ctx: Context) -> None:
        # Send a message that would match both guidelines
        await ctx.send_and_receive_message(
            customer_message="I need help with billing please",
            recipient=self.agent,
        )

        # Only the specific (prioritized) guideline's handler should be called
        assert self.specific_handler_called, "Specific guideline handler should have been called"
        assert not self.general_handler_called, (
            "General guideline handler should NOT have been called "
            "because it was de-prioritized during resolution"
        )


class Test_that_guideline_can_be_created_with_criticality(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Criticality Test Agent",
            description="Agent for testing guideline criticality",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer asks about high priority issue",
            action="Escalate immediately to senior support",
            criticality=Criticality.HIGH,
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]
        stored_guideline = await guideline_store.read_guideline(guideline_id=self.guideline.id)

        assert stored_guideline.criticality == Criticality.HIGH


class Test_that_guideline_defaults_to_medium_criticality_when_not_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Default Criticality Test Agent",
            description="Agent for testing default criticality",
        )

        self.guideline = await self.agent.create_guideline(
            condition="Customer asks a general question",
            action="Provide standard information",
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]
        stored_guideline = await guideline_store.read_guideline(guideline_id=self.guideline.id)

        assert stored_guideline.criticality == Criticality.MEDIUM


class Test_that_observation_can_be_created_with_criticality(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Observation Criticality Test Agent",
            description="Agent for testing observation criticality",
        )

        self.observation = await self.agent.create_observation(
            condition="Customer shows signs of extreme frustration",
            description="High priority observation requiring immediate attention",
            criticality=Criticality.HIGH,
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]
        stored_observation = await guideline_store.read_guideline(guideline_id=self.observation.id)

        assert stored_observation.criticality == Criticality.HIGH


class Test_that_observation_defaults_to_medium_criticality_when_not_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Default Observation Criticality Test Agent",
            description="Agent for testing default observation criticality",
        )

        self.observation = await self.agent.create_observation(
            condition="Customer asks about store hours",
        )

    async def run(self, ctx: Context) -> None:
        guideline_store = ctx.container[GuidelineStore]
        stored_observation = await guideline_store.read_guideline(guideline_id=self.observation.id)

        assert stored_observation.criticality == Criticality.MEDIUM


class Test_that_on_message_handler_is_called_when_guideline_generates_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Message Handler Test Agent",
            description="Agent for testing on_message handler",
        )

        self.handler_called = False
        self.captured_message_count = 0
        self.captured_guideline_id = None

        async def message_handler(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.handler_called = True
            # Verify we can access messages from context
            self.captured_message_count = len(
                [e for e in ctx.state.message_events if e.source == EventSource.AI_AGENT]
            )
            # Verify we receive the match parameter
            self.captured_guideline_id = match.id

        self.guideline = await self.agent.create_guideline(
            condition="Customer says hello",
            action="Greet the customer warmly",
            on_message=message_handler,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Hello there!",
            recipient=self.agent,
        )

        await asyncio.sleep(5)

        assert self.handler_called, "on_message handler should be called"
        assert self.captured_message_count > 0, "Handler should see messages in context"
        assert self.captured_guideline_id == self.guideline.id, (
            "Handler should receive correct guideline match"
        )


class Test_that_on_message_handler_is_not_called_when_guideline_does_not_match(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Non-matching Handler Test Agent",
            description="Agent for testing on_message handler when guideline doesn't match",
        )

        self.handler_called = False

        async def message_handler(ctx: p.EngineContext, match: p.GuidelineMatch) -> None:
            self.handler_called = True

        self.guideline = await self.agent.create_guideline(
            condition="Customer asks about pizza",
            action="Recommend pizza toppings",
            on_message=message_handler,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I want to talk about bananas",
            recipient=self.agent,
        )

        # Wait to ensure handler is not called
        import asyncio

        await asyncio.sleep(5)

        assert not self.handler_called, (
            "on_message handler should not be called when guideline doesn't match"
        )


class Test_that_guideline_field_provider_contributes_fields_to_canned_response(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Field Provider Agent",
            description="Agent for testing field providers",
        )

        # Create a canned response with a template that uses a field
        canrep_id = await self.agent.create_canned_response(
            template="Your special number is {{lucky_number}}.",
        )

        # Field provider that returns the field value
        async def provide_fields(ctx: p.EngineContext) -> dict[str, int]:
            return {"lucky_number": 42}

        # Create guideline with STRICT mode and field provider
        self.guideline = await self.agent.create_guideline(
            condition="Customer asks for their lucky number",
            action="Tell them their lucky number",
            composition_mode=p.CompositionMode.STRICT,
            canned_responses=[canrep_id],
            canned_response_field_provider=provide_fields,
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What is my lucky number?",
            recipient=self.agent,
        )

        assert response == "Your special number is 42."


class Test_that_multiple_guidelines_can_provide_fields(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Multiple Field Provider Agent",
            description="Agent for testing multiple field providers",
        )

        # Create a canned response that uses fields from multiple providers
        canrep_id = await self.agent.create_canned_response(
            template="First: {{field_a}}, Second: {{field_b}}.",
        )

        async def provide_field_a(ctx: p.EngineContext) -> dict[str, str]:
            return {"field_a": "ALPHA"}

        async def provide_field_b(ctx: p.EngineContext) -> dict[str, str]:
            return {"field_b": "BETA"}

        # Create two guidelines that both match
        self.guideline_a = await self.agent.create_guideline(
            condition="Customer asks a question",
            action="Respond with info",
            canned_response_field_provider=provide_field_a,
        )

        self.guideline_b = await self.agent.create_guideline(
            condition="Customer wants data",
            action="Provide the requested data",
            composition_mode=p.CompositionMode.STRICT,
            canned_responses=[canrep_id],
            canned_response_field_provider=provide_field_b,
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="I have a question and I want some data please",
            recipient=self.agent,
        )

        assert response == "First: ALPHA, Second: BETA."
