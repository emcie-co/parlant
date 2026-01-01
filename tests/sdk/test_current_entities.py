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


class Test_that_hooks_can_access_current_sdk_entities(SDKTest):
    async def configure_hooks(self, hooks: p.EngineHooks) -> p.EngineHooks:
        async def on_acknowledged(
            context: p.EngineContext, payload: Any, exception: Exception | None
        ) -> p.EngineHookResult:
            self.captured_server = p.Server.current
            self.captured_agent = p.Agent.current
            self.captured_customer = p.Customer.current
            return p.EngineHookResult.CALL_NEXT

        hooks.on_acknowledged.append(on_acknowledged)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Test Agent",
            description="A test agent",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message("Hello", self.agent)

        assert self.captured_server == ctx.server

        assert self.captured_agent is not None
        assert self.captured_agent.id == self.agent.id
        assert self.captured_agent.name == self.agent.name

        assert self.captured_customer is not None
        assert self.captured_customer.id == p.Customer.guest.id
        assert self.captured_customer.name == p.Customer.guest.name
