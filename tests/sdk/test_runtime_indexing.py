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

import parlant.sdk as p
from parlant.core.engines.compass.matching.rule_recaller import RuleRecaller
from parlant.core.rules import RuleStore

from tests.sdk.utils import Context, SDKTest


class Test_that_a_policy_created_while_serving_is_indexed_like_one_created_during_setup(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="runtime-indexing-agent-1",
            name="Refunds",
            instructions="Handles refunds and store questions",
        )
        self.setup_policy = await self.agent.create_policy(
            content="When the customer asks about opening hours, tell them the store hours",
        )

    async def run(self, ctx: Context) -> None:
        runtime_policy = await self.agent.create_policy(
            content="When the customer wants to return a damaged item, start the returns flow",
        )

        await ctx.server.wait_for_indexing()

        store = ctx.container[RuleStore]
        from_setup = await store.read_rule(self.setup_policy.id)
        from_runtime = await store.read_rule(runtime_policy.id)

        # Startup indexing gave the setup policy generated signals, anti-signals
        # and a title; the runtime policy must have received the same treatment.
        assert from_setup.signals and from_setup.anti_signals and from_setup.title
        assert from_runtime.signals and from_runtime.anti_signals and from_runtime.title


class Test_that_a_policy_created_while_serving_is_covered_by_its_agents_trained_recall_frame(
    SDKTest
):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="runtime-indexing-agent-2",
            name="Refunds",
            instructions="Handles refunds and store questions",
        )
        await self.agent.create_policy(
            content="When the customer asks about opening hours, tell them the store hours",
        )

    async def run(self, ctx: Context) -> None:
        recaller = ctx.container[RuleRecaller]
        startup_frame = recaller._frames_by_agent[self.agent.id]

        runtime_policy = await self.agent.create_policy(
            content="When the customer wants to return a damaged item, start the returns flow",
        )

        assert runtime_policy.id not in startup_frame.by_rule

        await ctx.server.wait_for_indexing()

        frame = recaller._frames_by_agent[self.agent.id]
        assert runtime_policy.id in frame.by_rule
        assert len(frame.by_rule) >= 2


class Test_that_an_agent_created_while_serving_with_a_burst_of_policies_is_trained_once(SDKTest):
    async def run(self, ctx: Context) -> None:
        recaller = ctx.container[RuleRecaller]

        agent = await ctx.server.create_agent(
            id="runtime-indexing-agent-3",
            name="Tenant Support",
            instructions="Handles support questions for a tenant that just signed up",
        )
        policies = [
            await agent.create_policy(content=content)
            for content in [
                "When the customer wants a refund, start the refund flow",
                "When the customer asks about opening hours, tell them the store hours",
                "When the customer asks where their order is, look up the order status",
            ]
        ]

        await ctx.server.wait_for_indexing()

        frame = recaller._frames_by_agent[agent.id]
        assert {policy.id for policy in policies} <= set(frame.by_rule)

        # Every training run caches its frame under the agent and the exact rule
        # set it was trained over. Training once per policy would have left one
        # entry per intermediate rule set; training once after the burst leaves one.
        frames_trained_for_agent = [key for key in recaller._frame_cache if key[0] == agent.id]
        assert len(frames_trained_for_agent) == 1


class Test_that_a_custom_evaluated_policy_created_while_serving_gets_no_signal_proposals(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="runtime-indexing-agent-4",
            name="Refunds",
            instructions="Handles refunds and store questions",
        )

    async def run(self, ctx: Context) -> None:
        recaller = ctx.container[RuleRecaller]

        policy = await self.agent.create_policy(
            content="Always greet the customer by name",
            evaluation_func=p.MATCH_ALWAYS,
        )

        await ctx.server.wait_for_indexing()

        stored = await ctx.container[RuleStore].read_rule(policy.id)

        assert list(stored.signals) == []
        assert list(stored.anti_signals) == []
        # The request it did queue (a title) was still processed, and its agent retrained.
        assert stored.title
        assert policy.id in recaller._frames_by_agent[self.agent.id].by_rule
