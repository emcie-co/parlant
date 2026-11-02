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
import time
from typing import Any
from parlant.core.rule_tool_associations import RuleToolAssociationStore
from parlant.core.rules import RuleStore
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.services.tools.plugins import tool
from parlant.core.groups import GroupIds
from parlant.core.tools import ToolContext, ToolResult
from parlant.core.canned_responses import CannedResponseStore
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test


class Test_that_a_compass_agent_can_configure_its_model_routes(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-1",
            name="Routed Agent",
            instructions="An agent with model routes",
            models={
                "react": p.ModelConfig("gemini:gemini-3.1-flash-lite"),
                "review.*": p.ModelConfig(
                    "openai:gpt-4o",
                    rate_limit=p.RateLimit(
                        max_concurrency=2,
                        requests_per_minute=60,
                        bucket="review-test",
                    ),
                ),
            },
        )

    async def run(self, ctx: Context) -> None:
        assert self.agent.models == {
            "react": p.ModelConfig("gemini:gemini-3.1-flash-lite"),
            "review.*": p.ModelConfig(
                "openai:gpt-4o",
                rate_limit=p.RateLimit(
                    max_concurrency=2,
                    requests_per_minute=60,
                    bucket="review-test",
                ),
            ),
        }


class Test_that_a_compass_agent_can_update_and_remove_model_routes(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-2",
            name="Live Routed Agent",
            instructions="An agent whose routes change while the server is running",
            models={"review.*": p.ModelConfig("openai:gpt-4o")},
        )

        await self.agent.update(
            models={
                "review.*": None,
                "review.tools[effort>=high]": p.ModelConfig("anthropic:claude-sonnet-4-5"),
            }
        )

    async def run(self, ctx: Context) -> None:
        assert self.agent.models == {
            "review.tools[effort>=high]": p.ModelConfig("anthropic:claude-sonnet-4-5")
        }


class Test_that_a_compass_agent_can_use_an_injected_secret_loader(SDKTest):
    SECRET_LOADER = p.MappingSecretLoader({"production/openai/api-key": "resolved-secret"})

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-3",
            name="Secret Routed Agent",
            instructions="An agent with an externally managed model credential",
            models={
                "review.*": p.ModelConfig(
                    "openai:gpt-4o",
                    args={
                        "api_key": p.SecretRef("production/openai/api-key"),
                    },
                )
            },
        )

    async def run(self, ctx: Context) -> None:
        assert ctx.container[p.SecretLoader] is self.SECRET_LOADER
        assert self.agent.models == {
            "review.*": p.ModelConfig(
                "openai:gpt-4o",
                args={"api_key": p.SecretRef("production/openai/api-key")},
            )
        }


class Test_that_an_agent_can_create_a_policy(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-4",
            name="Rule Agent",
            instructions="Agent for rule test",
        )
        self.rule = await self.agent.create_policy(
            content="Always say hello to the user",
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]

        rule = await rule_store.read_rule(rule_id=self.rule.id)

        assert rule.content == "Always say hello to the user"
        assert rule.groups == [GroupIds.for_agent_id(self.agent.id)]
        assert len(rule.signals) == 10
        assert all(signal for signal in rule.signals)
        assert len(rule.anti_signals) == 10
        assert all(anti_signal for anti_signal in rule.anti_signals)


class Test_that_sdk_trigger_effort_is_forwarded_as_core_effort(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-5",
            name="Effort Agent",
            instructions="Agent for trigger effort forwarding test",
        )
        self.rule = await self.agent.create_policy(
            content="When the customer requests a regulated action, apply additional care",
            trigger_effort=p.Effort.HIGH,
            evaluation_func=p.MATCH_ALWAYS,
        )

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]

        rule = await rule_store.read_rule(rule_id=self.rule.id)

        assert rule.effort == p.Effort.HIGH
        assert "effort_lift" not in rule.metadata


class Test_that_an_agent_gets_specific_rule_objects_when_creating_rules(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-6",
            name="Rule Agent",
            instructions="Agent for rule object tests",
        )

        self.observation = await self.agent.create_observation(
            condition="The customer is discussing billing",
            note="Billing is the active topic.",
        )
        self.policy = await self.agent.create_policy(
            title="Billing privacy",
            content="Never reveal billing information without verification.",
        )
        self.relationships = await self.policy.depend_on(self.observation)

    async def run(self, ctx: Context) -> None:
        assert isinstance(self.observation, p.Rule)
        assert isinstance(self.observation, p.Observation)
        assert not isinstance(self.observation, p.Policy)

        assert isinstance(self.policy, p.Rule)
        assert isinstance(self.policy, p.Policy)
        assert not isinstance(self.policy, p.Observation)
        assert self.policy.content == "Never reveal billing information without verification."

        rule_store = ctx.container[RuleStore]

        stored_observation = await rule_store.read_rule(self.observation.id)
        assert stored_observation.condition == "The customer is discussing billing"
        assert stored_observation.content == "Billing is the active topic."

        stored_policy = await rule_store.read_rule(self.policy.id)
        assert stored_policy.title == "Billing privacy"
        assert stored_policy.content == self.policy.content

        relationship_store = ctx.container[RelationshipStore]
        relationship = await relationship_store.read_relationship(self.relationships[0].id)
        assert relationship.kind == RelationshipKind.DEPENDENCY
        assert relationship.source.id == self.policy.id
        assert relationship.target.id == self.observation.id


class Test_that_an_observation_can_carry_a_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-7",
            name="Tool Agent",
            instructions="Agent for tool test",
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        observation = await self.agent.create_observation(
            condition="If user asks for dummy tool",
            tools=[test_tool],
        )
        self.rule_id = observation.id

    async def run(self, ctx: Context) -> None:
        rule_store = ctx.container[RuleStore]
        rule_tooL_store = ctx.container[RuleToolAssociationStore]

        rule = await rule_store.read_rule(rule_id=self.rule_id)

        assert rule.condition == "If user asks for dummy tool"

        associations = await rule_tooL_store.list_associations()
        assert associations
        assert len(associations) == 1

        association = associations[0]
        assert association.rule_id == rule.id


class Test_that_an_agent_can_create_canned_response(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-8",
            name="Canned Response Agent",
            instructions="Agent for canned response test",
        )
        self.canrep_id = await self.agent.create_canned_response(template="Hello, {user}!")

    async def run(self, ctx: Context) -> None:
        canrep_store = ctx.container[CannedResponseStore]

        canrep = await canrep_store.read_canned_response(canned_response_id=self.canrep_id)

        assert canrep.value == "Hello, {user}!"
        assert GroupIds.for_agent_id(self.agent.id) in canrep.groups


class Test_that_agents_can_be_listed(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.a1 = await server.create_agent(
            id="agents-agent-9",
            name="List Agent 1",
            instructions="First agent for listing",
        )

        self.a2 = await server.create_agent(
            id="agents-agent-10",
            name="List Agent 2",
            instructions="Second agent for listing",
        )

    async def run(self, ctx: Context) -> None:
        agents = await ctx.server.list_agents()

        assert self.a1 in agents
        assert self.a2 in agents


class Test_that_an_agent_can_be_found_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.a1 = await server.create_agent(
            id="agents-agent-11",
            name="List Agent 1",
            instructions="First agent for listing",
        )

    async def run(self, ctx: Context) -> None:
        assert await ctx.server.find_agent(id=self.a1.id) == self.a1
        assert await ctx.server.find_agent(id="nonexistent") is None


class Test_that_an_agent_can_be_found_using_tool_context(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-12",
            name="Tool Context Agent",
            instructions="Agent for tool context test",
        )

        @p.tool
        async def check_what_is_spatio(context: ToolContext) -> ToolResult:
            agent = await p.Server.current.find_agent(id=context.agent_id)

            if agent is None:
                return ToolResult("A spatio is a special type of spaghetti spoon.")
            else:
                return ToolResult("Spatio is the name of a famous fictional mouse.")

        await self.agent.create_observation(
            condition="the user asks about spatio", tools=[check_what_is_spatio]
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(
            customer_message="What is spatio?",
            recipient=self.agent,
        )

        assert await nlp_test(answer, "It says that spatio is the name of a mouse.")


class Test_that_the_output_of_an_agent_can_be_intercepted(SDKTest):
    # This test shows that you can intercept the agent's generated message before
    # it reaches the customer. This can be extremely important for last-minute validations.

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def intercept_message(
            ctx: p.EngineContext, payload: Any, exc: Exception | None
        ) -> p.CompassHookResult:
            _ = payload  # Here is where validations would run (payload is the generated message)

            await ctx.session_event_emitter.emit_message_event(
                trace_id=ctx.tracer.trace_id,
                data="Bananas! More bananas!",
            )

            # Reject the generated message
            return p.CompassHookResult.BAIL

        hooks.on_message_generated.append(intercept_message)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-13", name="Dummy Agent", instructions=""
        )

    async def run(self, ctx: Context) -> None:
        answer = await ctx.send_and_receive_message(customer_message="Hello", recipient=self.agent)
        assert answer == "Bananas! More bananas!"


class Test_that_an_agent_can_be_created_with_custom_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="my-custom-agent-id",
            name="Custom ID Agent",
            instructions="This agent has a custom ID",
        )

    async def run(self, ctx: Context) -> None:
        assert self.agent.id == "my-custom-agent-id"

        # Verify the agent can be retrieved with the custom ID
        retrieved_agent = await ctx.server.find_agent(id="my-custom-agent-id")
        assert retrieved_agent is not None
        assert retrieved_agent.id == "my-custom-agent-id"
        assert retrieved_agent.name == "Custom ID Agent"


class Test_that_an_agent_can_be_created_with_streaming_output_mode(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="agents-agent-14",
            name="Streaming Agent",
            instructions="Agent with streaming output mode",
            output_mode=p.OutputMode.STREAM,
        )

    async def run(self, ctx: Context) -> None:
        # Verify the agent was created with streaming output mode
        agent = await ctx.server.find_agent(id=self.agent.id)
        assert agent is not None
        assert agent.output_mode == p.OutputMode.STREAM

        # Send a message and verify streaming behavior
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )

        customer_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Hello",
        )

        # Wait for the agent to start responding, then check for chunks
        start_time = time.time()
        agent_message = None

        while time.time() - start_time < 30:
            agent_messages = await ctx.client.sessions.list_events(
                session_id=session.id,
                min_offset=customer_event.offset,
                source="ai_agent",
                kinds="message",
                wait_for_data=5,
            )

            if agent_messages:
                agent_message = agent_messages[0]
                message_data = agent_message.model_dump().get("data", {})
                chunks = message_data.get("chunks")

                # Streaming response should have chunks
                if chunks is not None and len(chunks) > 0:
                    # If the last chunk is None, streaming is complete
                    if chunks[-1] is None:
                        break

            await asyncio.sleep(1)

        assert agent_message is not None
        message_data = agent_message.model_dump().get("data", {})
        chunks = message_data.get("chunks")

        # Verify that chunks exist and streaming completed (last chunk is None)
        assert chunks is not None
        assert len(chunks) > 0
        assert chunks[-1] is None  # Null terminator indicates completion

        # Verify the final message contains content
        assert message_data.get("message") is not None
        assert len(message_data.get("message", "")) > 0
