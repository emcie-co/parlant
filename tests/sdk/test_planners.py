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

from dataclasses import dataclass, field
from typing import Sequence

from parlant.core.engines.alpha.engine_context import EngineContext
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.planners import (
    Plan,
    Planner,
)
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCall,
    ToolCallInferenceResult,
    ToolCallResult,
)
from parlant.core.tools import ToolContext, ToolResult
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest


@dataclass
class LifecycleRecord:
    guidelines_matched_count: int = 0
    guidelines_resolved_count: int = 0
    tools_inferred_count: int = 0
    tools_called_count: int = 0
    inferred_tool_calls: list[list[ToolCall]] = field(default_factory=list)


class TrackingPlan(Plan):
    def __init__(self, inner: Plan) -> None:
        super().__init__()
        self._inner = inner
        self.record = LifecycleRecord()

    @property
    def reasoning(self) -> str:
        return self._inner.reasoning

    async def on_guidelines_matched(
        self,
        context: EngineContext,
        matched_guidelines: list[GuidelineMatch],
    ) -> None:
        self.record.guidelines_matched_count += 1
        await self._inner.on_guidelines_matched(context, matched_guidelines)

    async def on_guidelines_resolved(self, context: EngineContext) -> None:
        self.record.guidelines_resolved_count += 1
        await self._inner.on_guidelines_resolved(context)

    async def on_tools_inferred(
        self,
        context: EngineContext,
        inference_result: ToolCallInferenceResult,
    ) -> Sequence[ToolCall]:
        self.record.tools_inferred_count += 1
        tool_calls = await self._inner.on_tools_inferred(context, inference_result)
        self.record.inferred_tool_calls.append(list(tool_calls))
        return tool_calls

    async def on_tools_called(
        self,
        context: EngineContext,
        tool_results: Sequence[ToolCallResult],
    ) -> None:
        self.record.tools_called_count += 1
        await self._inner.on_tools_called(context, tool_results)
        self.needs_additional_iteration = self._inner.needs_additional_iteration


@dataclass
class PlannerRecord:
    create_plan_count: int = 0
    plans: list[TrackingPlan] = field(default_factory=list)


class TrackingPlanner(Planner):
    def __init__(self, inner: Planner) -> None:
        self._inner = inner
        self.record = PlannerRecord()

    async def create_plan(self, context: EngineContext) -> Plan:
        self.record.create_plan_count += 1
        inner_plan = await self._inner.create_plan(context)
        tracking_plan = TrackingPlan(inner_plan)
        self.record.plans.append(tracking_plan)
        return tracking_plan


class Test_that_null_planner_passes_tools_through_when_present(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(p.NullPlanner())
        self.tool_called = False

        self.agent = await server.create_agent(
            name="Planner Test Agent",
            description="Agent for testing planner behavior",
            planner=self.tracking_planner,
        )

        @p.tool
        async def get_account_balance(context: ToolContext, account_id: str) -> ToolResult:
            self.tool_called = True
            return ToolResult(data={"account_id": account_id, "balance": 1500.00})

        await self.agent.attach_tool(
            tool=get_account_balance,
            condition="the user asks about their account balance",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="What is the balance of account ABC123?",
            recipient=self.agent,
        )

        assert self.tool_called, "Expected tool to be called"
        assert self.tracking_planner.record.create_plan_count == 1

        plan = self.tracking_planner.record.plans[0]
        assert plan.record.guidelines_resolved_count >= 1
        assert plan.record.tools_inferred_count >= 1
        assert len(plan.record.inferred_tool_calls) >= 1
        assert len(plan.record.inferred_tool_calls[0]) == 1


class Test_that_null_planner_works_when_no_tools_present(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(p.NullPlanner())

        self.agent = await server.create_agent(
            name="Planner Test Agent",
            description="Agent for testing planner behavior",
            planner=self.tracking_planner,
        )

        await self.agent.create_guideline(
            condition="always",
            action="greet the user politely",
        )

        await self.agent.create_guideline(
            condition="always",
            action="mention the current weather is sunny",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Hello there",
            recipient=self.agent,
        )

        assert self.tracking_planner.record.create_plan_count == 1

        plan = self.tracking_planner.record.plans[0]
        assert plan.record.guidelines_resolved_count >= 1
        assert plan.record.tools_called_count >= 1
        assert plan.needs_additional_iteration is False


class Test_that_null_planner_passes_multiple_tools_through_without_sequencing(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(p.NullPlanner())
        self.weather_called = False
        self.time_called = False

        self.agent = await server.create_agent(
            name="Planner Test Agent",
            description="Agent for testing planner behavior",
            planner=self.tracking_planner,
        )

        @p.tool
        async def get_weather(context: ToolContext, city: str) -> ToolResult:
            self.weather_called = True
            return ToolResult(data={"city": city, "weather": "sunny", "temperature": 25})

        @p.tool
        async def get_time(context: ToolContext, city: str) -> ToolResult:
            self.time_called = True
            return ToolResult(data={"city": city, "time": "14:30"})

        await self.agent.attach_tool(
            tool=get_weather,
            condition="the user asks about the weather",
        )

        await self.agent.attach_tool(
            tool=get_time,
            condition="the user asks about the time",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="What is the weather and time in London?",
            recipient=self.agent,
        )

        assert self.weather_called, "Expected weather tool to be called"
        assert self.time_called, "Expected time tool to be called"
        assert self.tracking_planner.record.create_plan_count == 1

        plan = self.tracking_planner.record.plans[0]
        assert plan.record.tools_inferred_count >= 1
        assert len(plan.record.inferred_tool_calls[0]) == 2
        assert plan.needs_additional_iteration is False
