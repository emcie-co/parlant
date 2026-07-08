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
from types import SimpleNamespace

import pytest
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.rule_match import RuleMatch as _RuleMatch
from parlant.core.common import Weight
from parlant.core.rules import RuleContent, RuleStore
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.sessions import EventSource
from parlant.core.groups import GroupIds
from parlant.core.tools import ToolContext, ToolResult
from parlant.core.canned_responses import CannedResponseStore
import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test


class _FakeCachedEvaluationCollection:
    def __init__(self, cached_evaluation: dict[str, object]) -> None:
        self.cached_evaluation = cached_evaluation

    async def find_one(self, filters: dict[str, object]) -> dict[str, object]:
        return self.cached_evaluation


def test_that_cached_signal_evaluations_without_anti_signals_are_ignored() -> None:
    evaluator = object.__new__(p._CachedEvaluator)

    assert not evaluator._cached_rule_evaluation_is_usable(
        {
            "properties": {},
            "signals": ["old"] * 5,
        },
        signal_proposition=True,
    )
    assert evaluator._cached_rule_evaluation_is_usable(
        {
            "properties": {},
            "signals": ["signal"] * 10,
            "anti_signals": ["anti"] * 10,
        },
        signal_proposition=True,
    )
    assert evaluator._cached_rule_evaluation_is_usable(
        {
            "properties": {},
            "signals": ["old"] * 5,
        },
        signal_proposition=False,
    )


def test_that_cached_rule_evaluation_marks_progress_complete() -> None:
    evaluator = object.__new__(p._CachedEvaluator)
    evaluator._rule_collection = _FakeCachedEvaluationCollection(
        {
            "properties": {},
            "signals": ["signal"] * 10,
            "anti_signals": ["anti"] * 10,
        }
    )
    evaluator._progress = {}
    evaluator._logger = SimpleNamespace(trace=lambda _: None)

    result = asyncio.run(
        evaluator.evaluate_rule(
            entity_id="rule-id",
            g=RuleContent(condition="condition", action="action"),
        )
    )

    assert result.signals == ["signal"] * 10
    assert evaluator._progress_for("rule-id") == 100.0


def test_that_cached_journey_evaluation_marks_progress_complete() -> None:
    evaluator = object.__new__(p._CachedEvaluator)
    evaluator._journey_collection = _FakeCachedEvaluationCollection(
        {
            "node_properties": {},
            "edge_properties": {},
        }
    )
    evaluator._progress = {}
    evaluator._logger = SimpleNamespace(trace=lambda _: None)
    journey = SimpleNamespace(id="journey-id", title="Journey", states=[], transitions=[])

    asyncio.run(evaluator.evaluate_journey(journey))

    assert evaluator._progress_for("journey-id") == 100.0


class Test_that_rule_can_take_priority_over_another_rule(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Priority Agent",
            prompt="Agent for testing rule priority",
        )

        # Both rules match when customer asks about drinks
        self.high_priority = await self.agent.create_rule(
            condition="Customer asks about drinks",
            action="Recommend Pepsi",
        )

        self.low_priority = await self.agent.create_rule(
            condition="Customer asks about drinks",
            action="Recommend Coca-Cola",
        )

        await self.high_priority.prioritize_over(self.low_priority)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What drinks do you have?",
            recipient=self.agent,
        )

        # High priority rule's action should apply
        assert "pepsi" in response.lower(), f"Expected Pepsi in response: {response}"
        # Low priority rule's action should NOT apply
        assert "cola" not in response.lower() and "coke" not in response.lower(), (
            f"Did not expect Coca-Cola in response: {response}"
        )


class Test_that_rule_entailment_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            prompt="Agent for rule relationships",
        )

        self.g1 = await self.agent.create_rule(
            condition="A customer is visibly upset about the wait",
            action="Transfer the customer to the manager immediately",
        )
        self.g2 = await self.agent.create_rule(
            condition="A new customer arrives", action="offer to sell pizza"
        )

        self.relationship = await self.g1.entail(self.g2)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )
        assert relationship.kind == RelationshipKind.ENTAILMENT


class Test_that_rule_dependency_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            prompt="Agent for rule relationships",
        )

        self.g1 = await self.agent.create_rule(
            condition="A customer asks for the price of tables",
            action="state that a table costs $100",
        )
        self.g2 = await self.agent.create_rule(
            condition="A customer expresses frustration",
            action="end your response with the word sorry",
        )

        self.relationships = await self.g2.depend_on(self.g2)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationships[0].id
        )
        assert relationship.kind == RelationshipKind.DEPENDENCY


class Test_that_rule_disambiguation_creates_relationships(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Disambiguation Agent",
            prompt="Agent for disambiguation",
        )

        self.g1 = await self.agent.create_rule(condition="A customer says they are thirsty")
        self.g2 = await self.agent.create_rule(condition="A customer says hello")
        self.g3 = await self.agent.create_rule(condition="A customer asks about pizza toppings")

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
            prompt="Agent for error test",
        )

        self.g1 = await self.agent.create_rule(condition="Customer asks for a recommendation")
        self.g2 = await self.agent.create_rule(condition="Customer asks about available soups")

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await self.g1.disambiguate([self.g2])


class Test_that_a_reevaluation_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Tool Agent",
            prompt="Agent for tool test",
            composition_mode=p.CompositionMode.FLUID,
        )

        self.g1 = await self.agent.create_rule(
            condition="Customer requests to update their contact information"
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        [self.relationship] = await self.g1.reevaluate_after(test_tool)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationship.id
        )
        assert relationship.kind == RelationshipKind.REEVALUATION


class Test_that_rule_can_take_priority_over_journey(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            prompt="",
        )

        # Rule that matches when customer asks about drinks
        self.rule = await self.agent.create_rule(
            condition="Customer asks about drinks",
            action="Recommend Pepsi",
        )

        # Journey that also matches when customer asks about drinks
        self.journey = await self.agent.create_journey(
            title="Drink Recommendation Journey",
            triggers=["Customer asks about drinks"],
            description="Recommend Coca-Cola to the customer",
        )

        await self.journey.create_rule(
            matcher=p.Rule.MATCH_ALWAYS,
            action="Recommend Coca-Cola",
        )

        await self.rule.prioritize_over(self.journey)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What drinks do you have?",
            recipient=self.agent,
        )

        # Rule's action should apply
        assert "pepsi" in response.lower(), f"Expected Pepsi in response: {response}"
        # Journey's recommendation should NOT apply
        assert "cola" not in response.lower() and "coke" not in response.lower(), (
            f"Did not expect Coca-Cola in response: {response}"
        )


class Test_that_rule_can_depend_on_journey(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rule to Journey Agent",
            prompt="Agent for rule to journey dependency",
        )

        self.rule = await self.agent.create_rule(
            condition="Customer asks about VIP service",
            action="Explain the VIP terms",
        )

        self.journey = await self.agent.create_journey(
            title="VIP Journey",
            triggers=["Customer is a VIP"],
            description="Assist the customer in a premium flow",
        )

        self.relationships = await self.rule.depend_on(self.journey)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationships[0].id
        )

        assert relationship.kind == RelationshipKind.DEPENDENCY
        assert relationship.source.id == self.rule.id
        assert relationship.target.id == GroupIds.for_journey_id(self.journey.id)


class Test_that_rule_can_be_created_with_inline_dependencies(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Inline Deps Agent",
            prompt="Agent for inline dependency creation",
        )

        self.g1 = await self.agent.create_rule(
            condition="Customer greets",
            action="Greet them back",
        )

        self.g2 = await self.agent.create_rule(
            condition="Customer asks about pricing",
            action="Provide pricing info",
        )

        self.g3 = await self.agent.create_rule(
            condition="Customer wants a quote",
            action="Generate a quote based on pricing",
            dependencies=[self.g1, self.g2],
        )

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]
        relationships = await relationship_store.list_relationships(
            source_id=self.g3.id,
            kind=RelationshipKind.DEPENDENCY,
        )

        assert len(relationships) == 2
        target_ids = {r.target.id for r in relationships}
        assert self.g1.id in target_ids
        assert self.g2.id in target_ids


class Test_that_observation_can_be_created_with_inline_dependencies(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Obs Deps Agent",
            prompt="Agent for observation inline dependencies",
        )

        self.g1 = await self.agent.create_rule(
            condition="Customer mentions a product",
            action="Note the product",
        )

        self.observation = await self.agent.create_observation(
            condition="Customer seems interested in buying",
            dependencies=[self.g1],
        )

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]
        relationships = await relationship_store.list_relationships(
            source_id=self.observation.id,
            kind=RelationshipKind.DEPENDENCY,
        )

        assert len(relationships) == 1
        assert relationships[0].target.id == self.g1.id


class Test_that_agent_rule_can_be_created_with_canned_responses(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Canned Response Agent",
            prompt="Agent for testing canned response associations",
        )

        self.canrep1 = await self.agent.create_canned_response(
            template="Thank you for your inquiry about {topic}."
        )
        self.canrep2 = await self.agent.create_canned_response(
            template="I'll be happy to help you with {request}."
        )

        self.rule = await self.agent.create_rule(
            condition="Customer asks for help",
            action="Provide assistance",
            canned_responses=[self.canrep1, self.canrep2],
        )

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]

        rule_group = GroupIds.for_rule_id(self.rule.id)

        updated_canrep1 = await canrep_store.read_canned_response(self.canrep1)
        updated_canrep2 = await canrep_store.read_canned_response(self.canrep2)

        assert rule_group in updated_canrep1.groups
        assert rule_group in updated_canrep2.groups


class Test_that_agent_observation_can_be_created_with_canned_responses(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Observation Agent",
            prompt="Agent for testing observation with canned responses",
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

        assert GroupIds.for_rule_id(self.observation.id) in updated_canrep.groups


class Test_that_agent_rule_can_be_created_with_metadata(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            prompt="Agent for testing rule metadata",
        )

        self.rule = await self.agent.create_rule(
            condition="Customer requests a callback",
            action="Schedule a callback within 24 hours",
            metadata={"continuous": True, "agent_intention_condition": "Test another property"},
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]

        rule = await rule_store.read_rule(self.rule.id)

        assert rule.metadata["continuous"] is True
        assert rule.metadata["agent_intention_condition"] == "Test another property"


class Test_that_rule_can_use_custom_matcher(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            prompt="Dummy agent",
        )

        self.rule = await self.agent.create_rule(
            condition="",
            action="Offer a banana",
            matcher=p.Rule.MATCH_ALWAYS,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello, sir.",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It offers a banana")


class Test_that_multiple_rules_can_use_custom_matcher(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            prompt="Dummy agent",
        )

        self.g1 = await self.agent.create_rule(
            action="Offer a cookie",
            matcher=p.Rule.MATCH_ALWAYS,
        )

        self.g2 = await self.agent.create_rule(
            action="Greet with 'Howdy'",
            matcher=p.Rule.MATCH_ALWAYS,
        )

        self.g3 = await self.agent.create_rule(
            action="Offer milk",
            matcher=p.Rule.MATCH_ALWAYS,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello, sir.",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It offers milk")
        assert await nlp_test(answer, "It greets with 'Howdy'")
        assert await nlp_test(answer, "It offers a cookie")


class Test_that_custom_matcher_can_return_no_match(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            prompt="Dummy agent",
        )

        async def never_match(ctx: p.RuleMatchingContext, rule: p.Rule) -> p.RuleMatch:
            return p.RuleMatch(
                id=rule.id,
                matched=False,
                rationale="Custom matcher never matches",
            )

        self.rule = await self.agent.create_rule(
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


class Test_that_rule_can_use_custom_matcher_with_compass_engine(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            prompt="Dummy agent",
            engine="compass",
            output_mode=p.OutputMode.STREAM,
            composition_mode=p.CompositionMode.FLUID,
        )

        self.rule = await self.agent.create_rule(
            condition="",
            action="Offer a banana",
            matcher=p.Rule.MATCH_ALWAYS,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello, sir.",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It offers a banana")


class Test_that_custom_matcher_can_return_no_match_with_compass_engine(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            prompt="Dummy agent",
            engine="compass",
            output_mode=p.OutputMode.STREAM,
            composition_mode=p.CompositionMode.FLUID,
        )

        async def never_match(ctx: p.RuleMatchingContext, rule: p.Rule) -> p.RuleMatch:
            return p.RuleMatch(
                id=rule.id,
                matched=False,
                rationale="Custom matcher never matches",
            )

        self.rule = await self.agent.create_rule(
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


class Test_that_a_rule_is_reevaluated_after_its_tool_runs_with_compass_engine(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Reevaluation Agent",
            prompt="A test agent.",
            engine="compass",
            output_mode=p.OutputMode.STREAM,
            composition_mode=p.CompositionMode.FLUID,
            effort=p.Effort.HIGH,
        )

        @tool
        async def get_weather(context: ToolContext) -> ToolResult:
            return ToolResult(data={"weather": "sunny"})

        # Enable the tool when the customer asks about the weather, so the model
        # calls it this turn.
        await self.agent.attach_tool(
            tool=get_weather,
            condition="the customer asks about the weather",
        )

        # A code-matched rule that only matches once a tool has run (i.e. a
        # tool result is staged). It starts unmatched and is excluded from the
        # recaller, so without reevaluation it never surfaces.
        async def only_after_a_tool_ran(ctx: p.RuleMatchingContext, rule: p.Rule) -> p.RuleMatch:
            ran = len(ctx.staged_events) > 0
            return p.RuleMatch(
                id=rule.id,
                matched=ran,
                rationale="a tool has run" if ran else "no tool has run yet",
            )

        rule = await self.agent.create_rule(
            condition="a tool has produced a result",
            action="End your reply with the exact token BANANAZ",
            matcher=only_after_a_tool_ran,
        )
        # Reevaluate this rule after get_weather runs.
        await rule.reevaluate_after(get_weather)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What's the weather like today?",
            recipient=self.agent,
        )

        assert "BANANAZ" in response.upper(), f"Expected BANANAZ in response, got: {response}"


class Test_that_rule_description_affects_agent_behavior(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            prompt="Dummy agent",
        )

        self.rule = await self.agent.create_rule(
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


class Test_that_rule_match_handler_is_called_when_rule_matches(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Match Handler Agent",
            prompt="Agent for testing match handlers",
        )

        self.captured_rule_id = None

        async def match_handler(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.captured_rule_id = match.id

        self.rule = await self.agent.create_rule(
            condition="Customer says hello",
            action="Greet the customer warmly",
            on_selected=match_handler,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Hello there!",
            recipient=self.agent,
        )

        assert self.captured_rule_id == self.rule.id, "Should capture correct rule ID"


class Test_that_multiple_match_handlers_can_be_registered_for_same_rule(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Multiple Handlers Agent",
            prompt="Agent for testing multiple handlers",
        )

        self.handler1_count = 0
        self.handler2_count = 0

        async def handler1(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.handler1_count += 1

        async def handler2(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.handler2_count += 1

        self.rule = await self.agent.create_rule(
            condition="Customer asks for help",
            action="Offer assistance",
            on_selected=handler1,
        )

        async def shim_handler2(
            core_ctx: p.EngineContext,
            core_match: _RuleMatch,
        ) -> None:
            sdk_match = p.RuleMatch(
                id=core_match.rule.id,
                matched=True,
                rationale=core_match.rationale,
            )
            await handler2(core_ctx, sdk_match)

        server.container[EngineHooks].on_rule_selected_handlers[self.rule.id].append(shim_handler2)

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I need help please",
            recipient=self.agent,
        )

        assert self.handler1_count == 1, "Handler 1 should be called once"
        assert self.handler2_count == 1, "Handler 2 should be called once"


class Test_that_match_handlers_for_different_rules_are_independent(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Independent Handlers Agent",
            prompt="Agent for testing independent handlers",
        )

        self.rule1_handler_called = False
        self.rule2_handler_called = False

        async def handler1(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.rule1_handler_called = True

        async def handler2(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.rule2_handler_called = True

        self.rule1 = await self.agent.create_rule(
            condition="Customer mentions pizza",
            action="Recommend pizza toppings",
            on_selected=handler1,
        )

        self.rule2 = await self.agent.create_rule(
            condition="Customer mentions pasta",
            action="Recommend pasta dishes",
            on_selected=handler2,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I'd like to order some pizza",
            recipient=self.agent,
        )

        assert self.rule1_handler_called, "Rule 1 handler should be called"
        assert not self.rule2_handler_called, "Rule 2 handler should NOT be called"


class Test_that_journey_scoped_rule_can_use_custom_matcher(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy Agent",
            prompt="Dummy agent",
        )

        self.journey = await self.agent.create_journey(
            title="Order Something",
            description="Journey to handle orders",
            triggers=["Customer wants to order something"],
        )

        self.rule = await self.journey.create_rule(
            condition="",
            action="Offer a banana",
            matcher=p.Rule.MATCH_ALWAYS,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello, I'd like to order something.",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It offers a banana")


class Test_that_match_handler_on_journey_scoped_rule_works(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Match Handler Agent",
            prompt="Agent for testing journey rule handlers",
        )

        self.journey = await self.agent.create_journey(
            title="Order Something",
            description="Journey to handle orders",
            triggers=["Customer wants to order something"],
        )

        self.handler_called = False

        async def match_handler(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.handler_called = True

        self.rule = await self.journey.create_rule(
            condition="Customer wants to order a banana",
            action="Tell them it's an excellent choice",
            on_selected=match_handler,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I'd like to order a banana",
            recipient=self.agent,
        )

        assert self.handler_called, "Journey rule handler should have been called"


class Test_that_rule_can_be_created_with_custom_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Custom ID Agent",
            prompt="Agent for testing custom ID functionality",
        )

        self.custom_id = p.RuleId("custom-rule-789")

        self.rule = await self.agent.create_rule(
            condition="Customer mentions custom ID requirement",
            action="Provide custom ID assistance",
            id=self.custom_id,
        )

    async def run(self, ctx: Context) -> None:
        # Verify the rule was created with the custom ID
        assert self.rule.id == self.custom_id

        # Verify it can be retrieved from the store
        rule_store = ctx.container[RuleStore]
        stored_rule = await rule_store.read_rule(self.custom_id)

        assert stored_rule.id == self.custom_id
        assert stored_rule.content.condition == "Customer mentions custom ID requirement"
        assert stored_rule.content.action == "Provide custom ID assistance"


class Test_that_rule_creation_fails_with_duplicate_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Duplicate ID Agent",
            prompt="Agent for testing duplicate ID handling",
        )

        self.duplicate_id = p.RuleId("duplicate-rule-101")

        # Create the first rule
        self.first_rule = await self.agent.create_rule(
            condition="First rule condition",
            action="First rule action",
            id=self.duplicate_id,
        )

    async def run(self, ctx: Context) -> None:
        # Verify the first rule was created
        assert self.first_rule.id == self.duplicate_id

        # Try to create a second rule with the same ID
        with pytest.raises(ValueError, match=f"Rule with id '{self.duplicate_id}' already exists"):
            await self.agent.create_rule(
                condition="Second rule condition",
                action="Second rule action",
                id=self.duplicate_id,
            )


class Test_that_only_prioritized_rule_handler_is_called_when_both_match(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Priority Test Agent",
            prompt="Agent for testing priority with handlers",
        )

        self.general_handler_called = False
        self.specific_handler_called = False

        async def general_handler(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.general_handler_called = True

        async def specific_handler(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.specific_handler_called = True

        # Create general rule that would match any help request
        self.general_rule = await self.agent.create_rule(
            condition="Customer asks for help",
            action="Provide general help information",
            on_selected=general_handler,
        )

        # Create more specific rule that should take priority
        self.specific_rule = await self.agent.create_rule(
            condition="Customer asks for help with billing",
            action="Provide billing-specific help",
            on_selected=specific_handler,
        )

        # Make specific rule prioritize over general rule
        await self.specific_rule.prioritize_over(self.general_rule)

    async def run(self, ctx: Context) -> None:
        # Send a message that would match both rules
        await ctx.send_and_receive_message(
            customer_message="I need help with billing please",
            recipient=self.agent,
        )

        # Only the specific (prioritized) rule's handler should be called
        assert self.specific_handler_called, "Specific rule handler should have been called"
        assert not self.general_handler_called, (
            "General rule handler should NOT have been called "
            "because it was de-prioritized during resolution"
        )


class Test_that_rule_can_be_created_with_criticality(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Weight Test Agent",
            prompt="Agent for testing rule criticality",
        )

        self.rule = await self.agent.create_rule(
            condition="Customer asks about high priority issue",
            action="Escalate immediately to senior support",
            criticality=Weight.HIGH,
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        stored_rule = await rule_store.read_rule(rule_id=self.rule.id)

        assert stored_rule.weight == Weight.HIGH


class Test_that_rule_defaults_to_medium_criticality_when_not_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Default Weight Test Agent",
            prompt="Agent for testing default criticality",
        )

        self.rule = await self.agent.create_rule(
            condition="Customer asks a general question",
            action="Provide standard information",
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        stored_rule = await rule_store.read_rule(rule_id=self.rule.id)

        assert stored_rule.weight == Weight.MEDIUM


class Test_that_observation_can_be_created_with_criticality(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Observation Weight Test Agent",
            prompt="Agent for testing observation criticality",
        )

        self.observation = await self.agent.create_observation(
            condition="Customer shows signs of extreme frustration",
            note="High priority observation requiring immediate attention",
            criticality=Weight.HIGH,
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        stored_observation = await rule_store.read_rule(rule_id=self.observation.id)

        assert stored_observation.weight == Weight.HIGH


class Test_that_observation_defaults_to_medium_criticality_when_not_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Default Observation Weight Test Agent",
            prompt="Agent for testing default observation criticality",
        )

        self.observation = await self.agent.create_observation(
            condition="Customer asks about store hours",
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        stored_observation = await rule_store.read_rule(rule_id=self.observation.id)

        assert stored_observation.weight == Weight.MEDIUM


class Test_that_on_message_handler_is_called_when_rule_generates_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Message Handler Test Agent",
            prompt="Agent for testing on_message handler",
        )

        self.handler_called = False
        self.captured_message_count = 0
        self.captured_rule_id = None

        async def message_handler(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.handler_called = True
            # Verify we can access messages from context
            self.captured_message_count = len(
                [e for e in ctx.state.message_events if e.source == EventSource.AI_AGENT]
            )
            # Verify we receive the match parameter
            self.captured_rule_id = match.id

        self.rule = await self.agent.create_rule(
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
        assert self.captured_rule_id == self.rule.id, "Handler should receive correct rule match"


class Test_that_on_message_handler_is_not_called_when_rule_does_not_match(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Non-matching Handler Test Agent",
            prompt="Agent for testing on_message handler when rule doesn't match",
        )

        self.handler_called = False

        async def message_handler(ctx: p.EngineContext, match: p.RuleMatch) -> None:
            self.handler_called = True

        self.rule = await self.agent.create_rule(
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
            "on_message handler should not be called when rule doesn't match"
        )


class Test_that_rule_field_provider_contributes_fields_to_canned_response(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Field Provider Agent",
            prompt="Agent for testing field providers",
        )

        # Create a canned response with a template that uses a field
        canrep_id = await self.agent.create_canned_response(
            template="Your special number is {{lucky_number}}.",
        )

        # Field provider that returns the field value
        async def provide_fields(ctx: p.EngineContext) -> dict[str, int]:
            return {"lucky_number": 42}

        # Create rule with STRICT mode and field provider
        self.rule = await self.agent.create_rule(
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


class Test_that_multiple_rules_can_provide_fields(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Multiple Field Provider Agent",
            prompt="Agent for testing multiple field providers",
        )

        # Create a canned response that uses fields from multiple providers
        canrep_id = await self.agent.create_canned_response(
            template="Fruit: {{fruit}}, Vegetable: {{vegetable}}.",
        )

        async def provide_field_a(ctx: p.EngineContext) -> dict[str, str]:
            return {"fruit": "banana"}

        async def provide_field_b(ctx: p.EngineContext) -> dict[str, str]:
            return {"vegetable": "carrot"}

        # Create two rules that both match
        self.rule_a = await self.agent.create_rule(
            condition="Customer asks for a fruit recommendation",
            action="Recommend a banana",
            canned_response_field_provider=provide_field_a,
        )

        self.rule_b = await self.agent.create_rule(
            condition="Customer wants a vegetable recommendation",
            action="Suggest a carrot",
            composition_mode=p.CompositionMode.STRICT,
            canned_responses=[canrep_id],
            canned_response_field_provider=provide_field_b,
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="I'd like both a fruit and a vegetable recommendation.",
            recipient=self.agent,
        )

        assert response == "Fruit: banana, Vegetable: carrot."


class Test_that_rule_retriever_runs_when_rule_matches(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Retriever Agent",
            prompt="Agent for testing rule retrievers",
        )

        rule = await self.agent.create_rule(
            condition="the user asks about the secret code",
            action="tell them the secret code from the retrieved data",
        )

        async def my_retriever(ctx: p.RetrieverContext) -> p.RetrieverResult:
            return p.RetrieverResult(data="The secret code is 42")

        await rule.attach_retriever(my_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What is the secret code?",
            recipient=self.agent,
        )
        assert "42" in response


class Test_that_rule_retriever_does_not_run_when_rule_does_not_match(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Retriever Agent",
            prompt="Agent for testing rule retrievers",
        )

        self.retriever_called = False

        rule = await self.agent.create_rule(
            condition="the user asks about the secret code",
            action="tell them the secret code from the retrieved data",
        )

        async def my_retriever(ctx: p.RetrieverContext) -> p.RetrieverResult:
            self.retriever_called = True
            return p.RetrieverResult(data="The secret code is 42")

        await rule.attach_retriever(my_retriever)

    async def run(self, ctx: Context) -> None:
        # Ask about something unrelated, rule should not match
        await ctx.send_and_receive_message(
            customer_message="What is the weather like today?",
            recipient=self.agent,
        )
        assert not self.retriever_called, "Retriever should not be called when rule doesn't match"


class Test_that_untracked_rule_is_reapplied_in_same_session(SDKTest):
    """Test that a rule with track=False is always treated as actionable,
    even after being applied once in the same session."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            prompt="",
        )

        await self.agent.create_rule(
            condition="The customer wants something to drink",
            action="Insist that your favorite drink is Pepsi",
            track=False,
        )

    async def run(self, ctx: Context) -> None:
        # First message - customer is thirsty
        first_response = await ctx.send_and_receive_message(
            customer_message="Hi, I want a drink please...",
            recipient=self.agent,
            reuse_session=True,
        )
        assert "pepsi" in first_response.lower(), (
            f"First response should offer Pepsi, got: {first_response}"
        )

        # Second message - customer is still thirsty (ignores the offer)
        second_response = await ctx.send_and_receive_message(
            customer_message="Hmmm... What do you have?",
            recipient=self.agent,
            reuse_session=True,
        )
        assert "pepsi" in second_response.lower(), (
            f"Second response should still offer Pepsi, got: {second_response}"
        )


class Test_that_a_rule_with_custom_group_is_followed(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Group Test Agent",
            prompt="Agent for testing custom groups",
        )

        group = await server.create_group("vip")

        await self.agent.create_rule(
            condition="always, in all circumstances",
            action="Offer a Pepsi",
            groups=[group],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello there",
            recipient=self.agent,
        )

        assert "pepsi" in response.lower(), f"Expected 'pepsi' in response but got: {response}"


class Test_that_tag_prioritize_over_deprioritizes_target_rule(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Group Priority Agent",
            prompt="Agent for testing group-based prioritization",
        )

        group = await server.create_group("priority-group")

        await self.agent.create_rule(
            condition="always, in all circumstances",
            action="Offer a Pepsi",
            groups=[group],
        )

        g2 = await self.agent.create_rule(
            condition="always, in all circumstances",
            action="Offer orange juice",
        )

        await group.prioritize_over(g2)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello",
            recipient=self.agent,
        )

        assert "pepsi" in response.lower(), f"Expected 'pepsi' in response but got: {response}"
        assert "orange" not in response.lower(), (
            f"Expected 'orange' to be filtered out by group prioritization but got: {response}"
        )


class Test_that_tag_depend_on_deactivates_tagged_rule_when_dependency_not_met(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Group Dependency Agent",
            prompt="Agent for testing group-based dependency",
        )

        group = await server.create_group("dep-group")

        await self.agent.create_rule(
            condition="always, in all circumstances",
            action="Offer a Pepsi",
            groups=[group],
        )

        g2 = await self.agent.create_rule(
            condition="the customer has explicitly said the word 'banana'",
            action="Offer Coke",
        )

        await group.depend_on(g2)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello, how are you",
            recipient=self.agent,
        )

        assert "pepsi" not in response.lower(), (
            f"Expected 'pepsi' NOT in response (dependency not met) but got: {response}"
        )


class Test_that_rule_depend_on_tag_deactivates_rule_when_tagged_dependency_not_met(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rule Group Dependency Agent",
            prompt="Agent for testing rule dependency on a custom group",
        )

        t1 = await server.create_group("drink-group")

        g1 = await self.agent.create_rule(
            matcher=p.MATCH_ALWAYS,
            action="Offer a Pepsi",
        )

        await self.agent.create_rule(
            condition="the customer has explicitly said the word 'banana'",
            action="Offer Coke",
            groups=[t1],
        )

        # g1 depends on group t1 — if no grouped rule is active, g1 is deactivated
        await g1.depend_on(t1)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello",
            recipient=self.agent,
        )

        assert "pepsi" not in response.lower(), (
            f"Expected 'pepsi' NOT in response (group dependency not met) but got: {response}"
        )


class Test_that_rule_depend_on_tag_deactivates_when_not_all_tagged_members_matched(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Group ALL Dependency Agent",
            prompt="Agent for testing ALL semantics on group dependency (default)",
        )

        t1 = await server.create_group("drink-group")

        g1 = await self.agent.create_rule(
            matcher=p.MATCH_ALWAYS,
            action="Offer a Pepsi",
        )

        # Two rules grouped with t1; only one will match
        await self.agent.create_rule(
            matcher=p.MATCH_ALWAYS,
            action="Offer Coke",
            groups=[t1],
        )

        await self.agent.create_rule(
            condition="the customer has explicitly said the word 'banana'",
            action="Offer Sprite",
            groups=[t1],
        )

        # g1 depends on group t1 — bare Group maps to GROUP_ALL (all members must match)
        await g1.depend_on(t1)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello",
            recipient=self.agent,
        )

        assert "pepsi" not in response.lower(), (
            f"Expected 'pepsi' NOT in response (GROUP_ALL: not all t1 members matched) "
            f"but got: {response}"
        )


class Test_that_rule_depend_on_any_of_tag_activates_when_at_least_one_tagged_member_matched(
    SDKTest,
):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Group AnyOf Dependency Agent",
            prompt="Agent for testing AnyOf semantics on group dependency",
        )

        t1 = await server.create_group("drink-group")

        g1 = await self.agent.create_rule(
            matcher=p.MATCH_ALWAYS,
            action="Offer a Pepsi",
        )

        # Two rules grouped with t1; only one will match
        await self.agent.create_rule(
            matcher=p.MATCH_ALWAYS,
            action="Offer Coke",
            groups=[t1],
        )

        await self.agent.create_rule(
            condition="the customer has explicitly said the word 'banana'",
            action="Offer Sprite",
            groups=[t1],
        )

        # g1 depends on AnyOf(t1) — at least one grouped member matched should activate g1
        await g1.depend_on(p.AnyOf(group=t1))

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello",
            recipient=self.agent,
        )

        assert "pepsi" in response.lower(), (
            f"Expected 'pepsi' in response (AnyOf: at least one t1 member matched) "
            f"but got: {response}"
        )


class Test_that_rule_depend_on_any_activates_when_one_of_two_rules_matched(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Depend On Any Agent",
            prompt="Agent for testing depend_on_any with rules",
        )

        self.g1 = await self.agent.create_rule(
            matcher=p.MATCH_ALWAYS,
            action="Offer a Pepsi",
        )

        self.g2 = await self.agent.create_rule(
            matcher=p.MATCH_ALWAYS,
            action="Offer Coke",
        )

        self.g3 = await self.agent.create_rule(
            condition="the customer has explicitly said the word 'banana'",
            action="Offer Sprite",
        )

        # g1 activates if EITHER g2 or g3 is active (g2 will match, g3 won't)
        await self.g1.depend_on_any(self.g2, self.g3)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello",
            recipient=self.agent,
        )

        assert "pepsi" in response.lower(), (
            f"Expected 'pepsi' in response (depend_on_any: g2 matched) but got: {response}"
        )


class Test_that_observation_can_be_created_with_a_title(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Title Agent",
            prompt="Agent for testing rule titles",
        )

        self.observation = await self.agent.create_observation(
            condition="the customer asks about the weather",
            title="Weather inquiries",
        )

    async def run(self, ctx: Context) -> None:
        store = ctx.container[RuleStore]
        stored = await store.read_rule(self.observation.id)
        assert stored.title == "Weather inquiries"
