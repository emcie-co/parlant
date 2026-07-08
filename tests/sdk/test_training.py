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

from tests.sdk.utils import Context, SDKTest


class Test_that_startup_trains_recall_discriminants_per_agent(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Refunds",
            prompt="Handles refunds and store questions",
        )
        await self.agent.create_rule(
            condition="the customer wants a refund",
            action="start the refund flow",
        )
        await self.agent.create_rule(
            condition="the customer asks about opening hours",
            action="tell them the store hours",
        )

    async def run(self, ctx: Context) -> None:
        recaller = ctx.container[RuleRecaller]

        # Startup trained a frame for this agent, over the agent's own rules.
        assert self.agent.id in recaller._frames_by_agent
        frame = recaller._frames_by_agent[self.agent.id]
        assert len(frame.by_rule) >= 2
