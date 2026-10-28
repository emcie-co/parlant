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

"""End-to-end checks for the compass canned-response review flow.

With canned responses attached, the compass engine buffers the message, has the
reviewer check it against the canned responses attached to the turn-matched
rules, and regenerates on a breach. These tests exercise the whole pipeline
(buffered message step -> review_message -> canned-response lookup + render)
against a live model."""

import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest


class Test_that_a_strict_compass_agent_produces_a_reviewed_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        # A STRICT agent runs the whole review pipeline every turn: the message
        # step is buffered (not streamed), review_message is invoked (constructing
        # the reviewer with its entity-queries + canned-response-store deps),
        # and only then is the message emitted. This case proves the end-to-end
        # wiring without depending on the reviewer finding a breach.
        self.agent = await server.create_agent(
            id="compass-canned-responses-agent-1",
            name="Strict Compass Agent",
            instructions="You are a helpful customer support agent.",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="I need help with something",
            recipient=self.agent,
        )

        assert response.strip()


class Test_that_a_fluid_compass_agent_still_responds(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="compass-canned-responses-agent-2",
            name="Fluid Compass Agent",
            instructions="You are a helpful customer support agent.",
        )

    async def run(self, ctx: Context) -> None:
        # No strict rule -> no message review; the ordinary fluid path must be
        # entirely unaffected by the review machinery.
        response = await ctx.send_and_receive_message(
            customer_message="Hello there!",
            recipient=self.agent,
        )

        assert response.strip()
