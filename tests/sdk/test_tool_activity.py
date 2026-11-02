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

from collections.abc import Mapping

import parlant.sdk as p

from parlant.core.tracer import AttributeValue, LocalTracer
from tests.sdk.utils import Context, SDKTest


class Test_that_tool_activity_begins_before_review_and_execution(SDKTest):
    async def configure_container(self, container: p.Container) -> p.Container:
        self.events: list[tuple[str, object, dict[str, AttributeValue]]] = []
        events = self.events

        class RecordingTracer(LocalTracer):
            def add_event(self, name: str, attributes: Mapping[str, AttributeValue] = {}) -> None:
                if name in ("loop.tools.started", "loop.tools.finished", "tool.called"):
                    events.append((name, self.get_attribute("step_index"), dict(attributes)))
                super().add_event(name, attributes)

        container[p.Tracer] = RecordingTracer()
        return container

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def review_calls(
            context: p.EngineContext,
            payload: p.CompassHookPayload.ReviewingToolCalls,
            exception: Exception | None,
        ) -> p.CompassHookResult:
            self.events.append(("review", context.tracer.get_attribute("step_index"), {}))
            self.reviewed_arguments = [dict(call.args) for call in payload.tool_calls]
            return p.CompassHookResult.CALL_NEXT

        hooks.on_reviewing_tool_calls.append(review_calls)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.reviewed_arguments = []
        self.executed_accounts: list[str] = []
        self.agent = await server.create_agent(
            id="early-tool-activity-agent",
            name="Account assistant",
            instructions="Help customers check their account status using the account lookup tool.",
        )

        @p.tool
        async def lookup_account(context: p.ToolContext, account: str) -> p.ToolResult:
            """Look up the status of the named account."""
            self.events.append(("execution", None, {}))
            self.executed_accounts.append(account)
            return p.ToolResult(data={"status": "active", "account": account})

        await self.agent.create_observation(
            condition="the customer asks to check an account's status",
            tools=[lookup_account],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            "Please check the status of account ABC123.", recipient=self.agent
        )
        assert self.executed_accounts == ["ABC123"]
        assert self.reviewed_arguments == [{"account": "ABC123"}]
        names = [name for name, _, _ in self.events]
        assert names.count("loop.tools.started") == names.count("loop.tools.finished") == 1
        assert names.index("loop.tools.started") < names.index("review")
        assert names.index("review") < names.index("tool.called") < names.index("execution")
        assert names.index("execution") < names.index("loop.tools.finished")
        tools = [
            (name, step, attrs) for name, step, attrs in self.events if name.startswith("loop.")
        ]
        assert tools[0][1] == tools[1][1]
        assert tools[1][2]["outcome"] == "completed"
