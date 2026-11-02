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

from typing import Any

import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest


class Test_that_resolve_on_preparing_skips_matching_and_still_responds(SDKTest):
    """A RESOLVE from an on_preparing hook skips the discovery/matching stage and
    goes straight to the responder — the dev-detected-greeting shortcut (Compass
    only). We prove matching was skipped by attaching an on_matched handler to a
    policy that would otherwise match and asserting it never fired, while the agent
    still replies."""

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def on_preparing(
            context: p.EngineContext, payload: Any, exception: Exception | None
        ) -> p.CompassHookResult:
            return p.CompassHookResult.RESOLVE

        hooks.on_preparing.append(on_preparing)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.policy_matched = False

        self.agent = await server.create_agent(
            id="hook-results-agent-1",
            name="Test Agent",
            instructions="A test agent",
        )

        async def on_matched(context: p.EngineContext, evaluation: p.RuleEvaluation) -> None:
            self.policy_matched = True

        await self.agent.create_policy(
            content="Greet the customer back warmly when they greet you.",
            on_matched=on_matched,
        )

    async def run(self, ctx: Context) -> None:
        reply = await ctx.send_and_receive_message("Hello", self.agent)

        # Discovery/matching was skipped: the policy never matched (its on_matched
        # handler never fired) — yet the agent still produced a reply.
        assert self.policy_matched is False
        assert reply.strip()
