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

# mypy: ignore-errors

import asyncio
from types import SimpleNamespace

import pytest
from parlant.core.engines.compass.hooks import CompassHooks
from parlant.core.engines.rule_evaluation import RuleEvaluation as _RuleEvaluation
from parlant.core.common import Weight
from parlant.core.rules import RuleStore
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.groups import GroupIds
from parlant.core.tools import ToolContext, ToolResult
from parlant.core.canned_responses import CannedResponseStore
import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test


class _FakeCachedIndexingCollection:
    def __init__(self, cached_indexing_result: dict[str, object]) -> None:
        self.cached_indexing_result = cached_indexing_result

    async def find_one(self, filters: dict[str, object]) -> dict[str, object]:
        return self.cached_indexing_result


def test_that_cached_signal_indexing_without_anti_signals_is_ignored() -> None:
    indexer = object.__new__(p._CachedIndexer)

    assert not indexer._cached_rule_indexing_is_usable(
        {
            "references_other_policies": False,
            "signals": ["old"] * 5,
        },
        signal_proposition=True,
    )
    assert indexer._cached_rule_indexing_is_usable(
        {
            "references_other_policies": False,
            "signals": ["signal"] * 10,
            "anti_signals": ["anti"] * 10,
        },
        signal_proposition=True,
    )
    assert indexer._cached_rule_indexing_is_usable(
        {
            "references_other_policies": False,
            "signals": ["old"] * 5,
        },
        signal_proposition=False,
    )


def test_that_cached_rule_indexing_marks_progress_complete() -> None:
    indexer = object.__new__(p._CachedIndexer)
    indexer._rule_collection = _FakeCachedIndexingCollection(
        {
            "references_other_policies": False,
            "signals": ["signal"] * 10,
            "anti_signals": ["anti"] * 10,
        }
    )
    indexer._progress = {}
    indexer._logger = SimpleNamespace(trace=lambda _: None)

    result = asyncio.run(
        indexer.index_rule(
            entity_id="rule-id",
            g=p._RuleText(condition="condition", content="a rule content"),
        )
    )

    assert result.signals == ["signal"] * 10
    assert indexer._progress_for("rule-id") == 100.0


class Test_that_rule_can_take_priority_over_another_rule(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-1",
            name="Priority Agent",
            instructions="Agent for testing rule priority",
        )

        # Both policies match when customer asks about drinks
        self.high_priority = await self.agent.create_policy(
            title="Recommend Pepsi",
            content="When the customer asks about drinks, recommend Pepsi",
        )

        self.low_priority = await self.agent.create_policy(
            title="Recommend Coca-Cola",
            content="When the customer asks about drinks, recommend Coca-Cola",
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
            id="rules-agent-2",
            name="Rel Agent",
            instructions="Agent for rule relationships",
        )

        self.g1 = await self.agent.create_policy(
            content=(
                "When a customer is visibly upset about the wait, "
                "transfer the customer to the manager immediately"
            ),
        )
        self.g2 = await self.agent.create_policy(
            content="When a new customer arrives, offer to sell pizza",
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
            id="rules-agent-3",
            name="Rel Agent",
            instructions="Agent for rule relationships",
        )

        self.g1 = await self.agent.create_policy(
            content="When a customer asks for the price of tables, state that a table costs $100",
        )
        self.g2 = await self.agent.create_policy(
            content=(
                "When a customer expresses frustration, end your response with the word sorry"
            ),
        )

        self.relationships = await self.g2.depend_on(self.g2)

    async def run(self, ctx: Context) -> None:
        relationship_store = ctx.container[RelationshipStore]

        relationship = await relationship_store.read_relationship(
            relationship_id=self.relationships[0].id
        )
        assert relationship.kind == RelationshipKind.DEPENDENCY


class Test_that_a_reevaluation_relationship_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-4",
            name="Tool Agent",
            instructions="Agent for tool test",
        )

        self.g1 = await self.agent.create_observation(
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


class Test_that_rule_can_be_created_with_inline_dependencies(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-5",
            name="Inline Deps Agent",
            instructions="Agent for inline dependency creation",
        )

        self.g1 = await self.agent.create_policy(
            content="When the customer greets, greet them back",
        )

        self.g2 = await self.agent.create_policy(
            content="When the customer asks about pricing, provide pricing info",
        )

        self.g3 = await self.agent.create_policy(
            content="When the customer wants a quote, generate a quote based on pricing",
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
            id="rules-agent-6",
            name="Obs Deps Agent",
            instructions="Agent for observation inline dependencies",
        )

        self.g1 = await self.agent.create_policy(
            content="When the customer mentions a product, note the product",
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
            id="rules-agent-7",
            name="Canned Response Agent",
            instructions="Agent for testing canned response associations",
        )

        self.canrep1 = await self.agent.create_canned_response(
            template="Thank you for your inquiry about {topic}."
        )
        self.canrep2 = await self.agent.create_canned_response(
            template="I'll be happy to help you with {request}."
        )

        self.rule = await self.agent.create_policy(
            content="When the customer asks for help, provide assistance",
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
            id="rules-agent-8",
            name="Observation Agent",
            instructions="Agent for testing observation with canned responses",
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
            id="rules-agent-9",
            name="Test Agent",
            instructions="Agent for testing rule metadata",
        )

        self.rule = await self.agent.create_policy(
            content="When the customer requests a callback, schedule a callback within 24 hours",
            metadata={"owner": "support", "region": "emea"},
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]

        rule = await rule_store.read_rule(self.rule.id)

        assert rule.metadata["owner"] == "support"
        assert rule.metadata["region"] == "emea"


class Test_that_rule_can_use_custom_matcher(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-10",
            name="Dummy Agent",
            instructions="Dummy agent",
        )

        self.rule = await self.agent.create_policy(
            content="Offer a banana",
            evaluation_func=p.MATCH_ALWAYS,
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
            id="rules-agent-11",
            name="Dummy Agent",
            instructions="Dummy agent",
        )

        self.g1 = await self.agent.create_policy(
            content="Offer a cookie",
            evaluation_func=p.MATCH_ALWAYS,
        )

        self.g2 = await self.agent.create_policy(
            content="Greet with 'Howdy'",
            evaluation_func=p.MATCH_ALWAYS,
        )

        self.g3 = await self.agent.create_policy(
            content="Offer milk",
            evaluation_func=p.MATCH_ALWAYS,
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
            id="rules-agent-12",
            name="Dummy Agent",
            instructions="Dummy agent",
        )

        async def never_match(ctx: p.RuleEvaluationContext, rule: p.Rule) -> p.RuleEvaluation:
            return p.RuleEvaluation(
                id=rule.id,
                is_relevant=False,
                reasoning="Custom evaluator never finds relevance",
            )

        self.rule = await self.agent.create_policy(
            content="When the customer greets you, offer a banana",
            evaluation_func=never_match,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello there!",
            recipient=self.agent,
        )

        assert not await nlp_test(answer, "It mentions a banana")


class Test_that_rule_can_use_custom_matcher_with_streaming_output(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-13",
            name="Dummy Agent",
            instructions="Dummy agent",
            output_mode=p.OutputMode.STREAM,
        )

        self.rule = await self.agent.create_policy(
            content="Offer a banana",
            evaluation_func=p.MATCH_ALWAYS,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello, sir.",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It offers a banana")


class Test_that_custom_matcher_can_return_no_match_with_streaming_output(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-14",
            name="Dummy Agent",
            instructions="Dummy agent",
            output_mode=p.OutputMode.STREAM,
        )

        async def never_match(ctx: p.RuleEvaluationContext, rule: p.Rule) -> p.RuleEvaluation:
            return p.RuleEvaluation(
                id=rule.id,
                is_relevant=False,
                reasoning="Custom evaluator never finds relevance",
            )

        self.rule = await self.agent.create_policy(
            content="When the customer greets you, offer a banana",
            evaluation_func=never_match,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello there!",
            recipient=self.agent,
        )

        assert not await nlp_test(answer, "It mentions a banana")


class Test_that_a_rule_is_reevaluated_after_its_tool_runs(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-15",
            name="Reevaluation Agent",
            instructions="A test agent.",
            output_mode=p.OutputMode.STREAM,
            effort=p.Effort.HIGH,
        )

        @tool
        async def get_weather(context: ToolContext) -> ToolResult:
            return ToolResult(data={"weather": "sunny"})

        # Enable the tool when the customer asks about the weather, so the model
        # calls it this turn.
        await self.agent.create_observation(
            condition="the customer asks about the weather",
            tools=[get_weather],
        )

        # A code-matched rule that only matches once a tool has run (i.e. a
        # tool result is staged). It starts unmatched and is excluded from the
        # recaller, so without reevaluation it never surfaces.
        async def only_after_a_tool_ran(
            ctx: p.RuleEvaluationContext, rule: p.Rule
        ) -> p.RuleEvaluation:
            ran = len(ctx.staged_events) > 0
            return p.RuleEvaluation(
                id=rule.id,
                is_relevant=ran,
                reasoning="a tool has run" if ran else "no tool has run yet",
            )

        rule = await self.agent.create_policy(
            content="When a tool has produced a result, end your reply with the exact token BANANAZ",
            evaluation_func=only_after_a_tool_ran,
        )
        # Reevaluate this rule after get_weather runs.
        await rule.reevaluate_after(get_weather)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What's the weather like today?",
            recipient=self.agent,
        )

        assert "BANANAZ" in response.upper(), f"Expected BANANAZ in response, got: {response}"


class Test_that_rule_content_affects_agent_behavior(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-16",
            name="Dummy Agent",
            instructions="Dummy agent",
        )

        self.rule = await self.agent.create_policy(
            content=(
                "When the customer asks about Cachookas, explain what Cachookas are. "
                "Cachookas are a type of ancient boomerang used to repel flies."
            ),
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
            id="rules-agent-17",
            name="Match Handler Agent",
            instructions="Agent for testing match handlers",
        )

        self.captured_rule_id = None

        async def match_handler(ctx: p.EngineContext, match: p.RuleEvaluation) -> None:
            self.captured_rule_id = match.id

        self.rule = await self.agent.create_policy(
            content="When the customer says hello, greet the customer warmly",
            on_matched=match_handler,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Hello there!",
            recipient=self.agent,
        )

        assert self.captured_rule_id == self.rule.id, "Should capture correct rule ID"


class Test_that_rule_discovery_handler_runs_only_on_session_admission(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-18",
            name="Discovery Handler Agent",
            instructions="Agent for testing rule discovery handlers",
        )
        self.discovered = 0
        self.matched = 0

        async def on_discovered(ctx: p.EngineContext, evaluation: p.RuleEvaluation) -> None:
            self.discovered += 1

        async def on_matched(ctx: p.EngineContext, evaluation: p.RuleEvaluation) -> None:
            self.matched += 1

        await self.agent.create_policy(
            content="Acknowledge the customer briefly",
            evaluation_func=p.MATCH_ALWAYS,
            on_discovered=on_discovered,
            on_matched=on_matched,
        )

    async def run(self, ctx: Context) -> None:
        # Both turns must share one session: admission is per-session, so a
        # fresh session per message would legitimately re-fire on_discovered.
        await ctx.send_and_receive_message(
            customer_message="Hello", recipient=self.agent, reuse_session=True
        )
        await ctx.send_and_receive_message(
            customer_message="Hello again", recipient=self.agent, reuse_session=True
        )

        assert self.discovered == 1
        assert self.matched == 2


class Test_that_multiple_match_handlers_can_be_registered_for_same_rule(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-19",
            name="Multiple Handlers Agent",
            instructions="Agent for testing multiple handlers",
        )

        self.handler1_count = 0
        self.handler2_count = 0

        async def handler1(ctx: p.EngineContext, match: p.RuleEvaluation) -> None:
            self.handler1_count += 1

        async def handler2(ctx: p.EngineContext, match: p.RuleEvaluation) -> None:
            self.handler2_count += 1

        self.rule = await self.agent.create_policy(
            content="When the customer asks for help, offer assistance",
            on_matched=handler1,
        )

        async def shim_handler2(
            core_ctx: p.EngineContext,
            core_match: _RuleEvaluation,
        ) -> None:
            sdk_match = p.RuleEvaluation(
                id=core_match.rule.id,
                is_relevant=True,
                reasoning=core_match.reasoning,
            )
            await handler2(core_ctx, sdk_match)

        server.container[CompassHooks].on_rule_matched_handlers[self.rule.id].append(shim_handler2)

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
            id="rules-agent-20",
            name="Independent Handlers Agent",
            instructions="Agent for testing independent handlers",
        )

        self.rule1_handler_called = False
        self.rule2_handler_called = False

        async def handler1(ctx: p.EngineContext, match: p.RuleEvaluation) -> None:
            self.rule1_handler_called = True

        async def handler2(ctx: p.EngineContext, match: p.RuleEvaluation) -> None:
            self.rule2_handler_called = True

        self.rule1 = await self.agent.create_policy(
            content="When the customer mentions pizza, recommend pizza toppings",
            on_matched=handler1,
        )

        self.rule2 = await self.agent.create_policy(
            content="When the customer mentions pasta, recommend pasta dishes",
            on_matched=handler2,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I'd like to order some pizza",
            recipient=self.agent,
        )

        assert self.rule1_handler_called, "Rule 1 handler should be called"
        assert not self.rule2_handler_called, "Rule 2 handler should NOT be called"


class Test_that_rule_can_be_created_with_custom_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-21",
            name="Custom ID Agent",
            instructions="Agent for testing custom ID functionality",
        )

        self.custom_id = p.RuleId("custom-rule-789")

        self.rule = await self.agent.create_policy(
            content="When the customer mentions a custom ID requirement, provide custom ID assistance",
            id=self.custom_id,
        )

    async def run(self, ctx: Context) -> None:
        # Verify the rule was created with the custom ID
        assert self.rule.id == self.custom_id

        # Verify it can be retrieved from the store
        rule_store = ctx.container[RuleStore]
        stored_rule = await rule_store.read_rule(self.custom_id)

        assert stored_rule.id == self.custom_id
        assert (
            stored_rule.content
            == "When the customer mentions a custom ID requirement, provide custom ID assistance"
        )


class Test_that_rule_creation_fails_with_duplicate_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-22",
            name="Duplicate ID Agent",
            instructions="Agent for testing duplicate ID handling",
        )

        self.duplicate_id = p.RuleId("duplicate-rule-101")

        # Create the first policy
        self.first_rule = await self.agent.create_policy(
            content="When the customer asks the first question, give the first answer",
            id=self.duplicate_id,
        )

    async def run(self, ctx: Context) -> None:
        # Verify the first policy was created
        assert self.first_rule.id == self.duplicate_id

        # Try to create a second policy with the same ID
        with pytest.raises(ValueError, match=f"Rule with id '{self.duplicate_id}' already exists"):
            await self.agent.create_policy(
                content="When the customer asks the second question, give the second answer",
                id=self.duplicate_id,
            )


class Test_that_only_prioritized_rule_handler_is_called_when_both_match(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-23",
            name="Priority Test Agent",
            instructions="Agent for testing priority with handlers",
        )

        self.general_handler_called = False
        self.specific_handler_called = False

        async def general_handler(ctx: p.EngineContext, match: p.RuleEvaluation) -> None:
            self.general_handler_called = True

        async def specific_handler(ctx: p.EngineContext, match: p.RuleEvaluation) -> None:
            self.specific_handler_called = True

        # Create general policy that would match any help request
        self.general_rule = await self.agent.create_policy(
            content="When the customer asks for help, provide general help information",
            on_matched=general_handler,
        )

        # Create more specific policy that should take priority
        self.specific_rule = await self.agent.create_policy(
            content="When the customer asks for help with billing, provide billing-specific help",
            on_matched=specific_handler,
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


class Test_that_rule_can_be_created_with_weight(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-24",
            name="Weight Test Agent",
            instructions="Agent for testing rule weight",
        )

        self.rule = await self.agent.create_policy(
            content=(
                "When the customer asks about a high priority issue, "
                "escalate immediately to senior support"
            ),
            weight=Weight.HIGH,
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        stored_rule = await rule_store.read_rule(rule_id=self.rule.id)

        assert stored_rule.weight == Weight.HIGH


class Test_that_rule_defaults_to_medium_weight_when_not_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-25",
            name="Default Weight Test Agent",
            instructions="Agent for testing default weight",
        )

        self.rule = await self.agent.create_policy(
            content="When the customer asks a general question, provide standard information",
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        stored_rule = await rule_store.read_rule(rule_id=self.rule.id)

        assert stored_rule.weight == Weight.MEDIUM


class Test_that_observation_can_be_created_with_weight(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-26",
            name="Observation Weight Test Agent",
            instructions="Agent for testing observation weight",
        )

        self.observation = await self.agent.create_observation(
            condition="Customer shows signs of extreme frustration",
            note="High priority observation requiring immediate attention",
            weight=Weight.HIGH,
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        stored_observation = await rule_store.read_rule(rule_id=self.observation.id)

        assert stored_observation.weight == Weight.HIGH


class Test_that_observation_defaults_to_medium_weight_when_not_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-27",
            name="Default Observation Weight Test Agent",
            instructions="Agent for testing default observation weight",
        )

        self.observation = await self.agent.create_observation(
            condition="Customer asks about store hours",
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        stored_observation = await rule_store.read_rule(rule_id=self.observation.id)

        assert stored_observation.weight == Weight.MEDIUM


class Test_that_zero_arity_rule_field_provider_contributes_fields_to_canned_response(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-28",
            name="Field Provider Agent",
            instructions="Agent for testing field providers",
        )

        # Create a canned response with a template that uses a field
        canrep_id = await self.agent.create_canned_response("{{greeting}}")

        # Field provider that returns the field value
        async def provide_fields() -> dict[str, str]:
            return {"greeting": "Howdy partner!"}

        # Create policy with an associated canned response and field provider
        self.rule = await self.agent.create_policy(
            content="When the customer greets you, greet them back",
            canned_responses=[canrep_id],
            canned_response_field_provider=provide_fields,
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello",
            recipient=self.agent,
        )

        assert "howdy" in response.lower(), f"Expected 'Howdy' in response, got: {response}"


class Test_that_multiple_rules_can_provide_fields(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-29",
            name="Multiple Field Provider Agent",
            instructions="Agent for testing multiple field providers",
        )

        # Create a canned response that uses fields from multiple providers
        canrep_id = await self.agent.create_canned_response(
            template="Fruit: {{fruit}}, Vegetable: {{vegetable}}.",
        )

        async def provide_field_a(ctx: p.EngineContext) -> dict[str, str]:
            return {"fruit": "banana"}

        async def provide_field_b(ctx: p.EngineContext) -> dict[str, str]:
            return {"vegetable": "carrot"}

        # Create two policies that both match
        self.rule_a = await self.agent.create_policy(
            content="When the customer asks for a fruit recommendation, recommend a banana",
            canned_response_field_provider=provide_field_a,
        )

        self.rule_b = await self.agent.create_policy(
            content="When the customer wants a vegetable recommendation, suggest a carrot",
            canned_responses=[canrep_id],
            canned_response_field_provider=provide_field_b,
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="I'd like both a fruit and a vegetable recommendation.",
            recipient=self.agent,
        )

        # Both matched policies' field providers must contribute to the same
        # canned-response render ("Fruit: {{fruit}}, Vegetable: {{vegetable}}.").
        assert "banana" in response.lower(), f"Expected 'banana' in response, got: {response}"
        assert "carrot" in response.lower(), f"Expected 'carrot' in response, got: {response}"


class Test_that_rule_retriever_runs_when_rule_matches(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-30",
            name="Retriever Agent",
            instructions="Agent for testing rule retrievers",
        )

        rule = await self.agent.create_policy(
            content=(
                "When the user asks about the secret code, "
                "tell them the secret code from the retrieved data"
            ),
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
            id="rules-agent-31",
            name="Retriever Agent",
            instructions="Agent for testing rule retrievers",
        )

        self.retriever_called = False

        rule = await self.agent.create_policy(
            content=(
                "When the user asks about the secret code, "
                "tell them the secret code from the retrieved data"
            ),
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


class Test_that_a_policy_reapplies_across_turns_in_the_same_session(SDKTest):
    """A matched policy stays applicable on later turns of the same session."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-32",
            name="Test Agent",
            instructions="",
        )

        await self.agent.create_policy(
            content="When the customer wants something to drink, insist that your favorite drink is Pepsi",
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
            id="rules-agent-33",
            name="Group Test Agent",
            instructions="Agent for testing custom groups",
        )

        group = await server.create_group("vip")

        await self.agent.create_policy(
            content="Always, in all circumstances, offer a Pepsi",
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
            id="rules-agent-34",
            name="Group Priority Agent",
            instructions="Agent for testing group-based prioritization",
        )

        group = await server.create_group("priority-group")

        await self.agent.create_policy(
            content="Always, in all circumstances, offer a Pepsi",
            groups=[group],
        )

        g2 = await self.agent.create_policy(
            content="Always, in all circumstances, offer orange juice",
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
            id="rules-agent-35",
            name="Group Dependency Agent",
            instructions="Agent for testing group-based dependency",
        )

        group = await server.create_group("dep-group")

        await self.agent.create_policy(
            content="Always, in all circumstances, offer a Pepsi",
            groups=[group],
        )

        g2 = await self.agent.create_policy(
            content="When the customer has explicitly said the word 'banana', offer Coke",
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
            id="rules-agent-36",
            name="Rule Group Dependency Agent",
            instructions="Agent for testing rule dependency on a custom group",
        )

        t1 = await server.create_group("drink-group")

        g1 = await self.agent.create_policy(
            content="Offer a Pepsi",
            evaluation_func=p.MATCH_ALWAYS,
        )

        await self.agent.create_policy(
            content="When the customer has explicitly said the word 'banana', offer Coke",
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
            id="rules-agent-37",
            name="Group ALL Dependency Agent",
            instructions="Agent for testing ALL semantics on group dependency (default)",
        )

        t1 = await server.create_group("drink-group")

        g1 = await self.agent.create_policy(
            content="Offer a Pepsi",
            evaluation_func=p.MATCH_ALWAYS,
        )

        # Two policies grouped with t1; only one will match
        await self.agent.create_policy(
            content="Offer Coke",
            evaluation_func=p.MATCH_ALWAYS,
            groups=[t1],
        )

        await self.agent.create_policy(
            content="When the customer has explicitly said the word 'banana', offer Sprite",
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
            id="rules-agent-38",
            name="Group AnyOf Dependency Agent",
            instructions="Agent for testing AnyOf semantics on group dependency",
        )

        t1 = await server.create_group("drink-group")

        g1 = await self.agent.create_policy(
            content="Offer a Pepsi",
            evaluation_func=p.MATCH_ALWAYS,
        )

        # Two policies grouped with t1; only one will match
        await self.agent.create_policy(
            content="Offer Coke",
            evaluation_func=p.MATCH_ALWAYS,
            groups=[t1],
        )

        await self.agent.create_policy(
            content="When the customer has explicitly said the word 'banana', offer Sprite",
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
            id="rules-agent-39",
            name="Depend On Any Agent",
            instructions="Agent for testing depend_on_any with rules",
        )

        self.g1 = await self.agent.create_policy(
            content="Offer a Pepsi",
            evaluation_func=p.MATCH_ALWAYS,
        )

        self.g2 = await self.agent.create_policy(
            content="Offer Coke",
            evaluation_func=p.MATCH_ALWAYS,
        )

        self.g3 = await self.agent.create_policy(
            content="When the customer has explicitly said the word 'banana', offer Sprite",
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
            id="rules-agent-40",
            name="Title Agent",
            instructions="Agent for testing rule titles",
        )

        self.observation = await self.agent.create_observation(
            condition="the customer asks about the weather",
            title="Weather inquiries",
        )

    async def run(self, ctx: Context) -> None:
        store = ctx.container[RuleStore]
        stored = await store.read_rule(self.observation.id)
        assert stored.title == "Weather inquiries"


class Test_that_an_observation_carries_authored_signals_and_anti_signals(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-41",
            name="Signals Agent",
            instructions="Agent for testing observation signals",
        )

        self.observation = await self.agent.create_observation(
            condition="the customer is disputing a charge",
            signals=[
                "I don't recognize this charge",
                "Why was I billed twice this month?",
            ],
            anti_signals=["How much does the premium plan cost?"],
        )

    async def run(self, ctx: Context) -> None:
        store = ctx.container[RuleStore]
        stored = await store.read_rule(self.observation.id)

        # Proposition may merge additional generated entries; the authored ones
        # must survive, in order, at the front.
        assert list(stored.signals)[:2] == [
            "I don't recognize this charge",
            "Why was I billed twice this month?",
        ]
        assert list(stored.anti_signals)[:1] == ["How much does the premium plan cost?"]


class Test_that_includes_current_customer_activates_a_policy_for_a_group_member(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-42",
            name="Membership Agent",
            instructions="Dummy agent",
        )

        self.vip = await server.create_group(name="vip")
        self.member = await server.create_customer(
            id="member-42",
            name="Member",
            groups=[self.vip],
        )

        self.rule = await self.agent.create_policy(
            content="Offer a banana",
            evaluation_func=self.vip.includes_current_customer,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello there.",
            recipient=self.agent,
            sender=self.member,
        )

        assert await nlp_test(answer, "It offers a banana")


class Test_that_includes_current_customer_keeps_a_policy_inactive_for_a_non_member(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-43",
            name="Membership Agent",
            instructions="Dummy agent",
        )

        self.vip = await server.create_group(name="vip")
        self.outsider = await server.create_customer(
            id="outsider-43",
            name="Outsider",
            groups=[],
        )

        self.rule = await self.agent.create_policy(
            content="Offer a banana",
            evaluation_func=self.vip.includes_current_customer,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello there.",
            recipient=self.agent,
            sender=self.outsider,
        )

        assert not await nlp_test(answer, "It mentions a banana")


class Test_that_excludes_current_customer_activates_a_policy_for_a_non_member(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-44",
            name="Membership Agent",
            instructions="Dummy agent",
        )

        self.blocked = await server.create_group(name="blocked")
        self.outsider = await server.create_customer(
            id="outsider-44",
            name="Outsider",
            groups=[],
        )

        self.rule = await self.agent.create_policy(
            content="Offer a banana",
            evaluation_func=self.blocked.excludes_current_customer,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello there.",
            recipient=self.agent,
            sender=self.outsider,
        )

        assert await nlp_test(answer, "It offers a banana")


class Test_that_excludes_current_customer_keeps_a_policy_inactive_for_a_group_member(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rules-agent-45",
            name="Membership Agent",
            instructions="Dummy agent",
        )

        self.blocked = await server.create_group(name="blocked")
        self.member = await server.create_customer(
            id="member-45",
            name="Member",
            groups=[self.blocked],
        )

        self.rule = await self.agent.create_policy(
            content="Offer a banana",
            evaluation_func=self.blocked.excludes_current_customer,
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="Hello there.",
            recipient=self.agent,
            sender=self.member,
        )

        assert not await nlp_test(answer, "It mentions a banana")
