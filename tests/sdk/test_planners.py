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
    MissingToolData,
    ToolCall,
    ToolCallInferenceResult,
    ToolCallResult,
    ToolInsights,
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
    tool_insights_after_on_tools_called: list[ToolInsights] = field(default_factory=list)
    needs_additional_iteration_history: list[bool] = field(default_factory=list)


class TrackingPlan(Plan):
    def __init__(self, inner: Plan, inject_synthetic_missing_data: bool = False) -> None:
        super().__init__()
        self._inner = inner
        self._inject_synthetic_missing_data = inject_synthetic_missing_data
        self.record = LifecycleRecord()
        self.reevaluate_all_tool_guidelines = inner.reevaluate_all_tool_guidelines

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
        if self._inject_synthetic_missing_data:
            context.state.tool_insights = ToolInsights(
                evaluations=context.state.tool_insights.evaluations,
                missing_data=[MissingToolData(parameter="synthetic_param")],
                invalid_data=context.state.tool_insights.invalid_data,
            )
        await self._inner.on_tools_called(context, tool_results)
        self.record.tool_insights_after_on_tools_called.append(context.state.tool_insights)
        self.record.needs_additional_iteration_history.append(
            self._inner.needs_additional_iteration
        )
        self.needs_additional_iteration = self._inner.needs_additional_iteration


@dataclass
class PlannerRecord:
    create_plan_count: int = 0
    plans: list[TrackingPlan] = field(default_factory=list)


class TrackingPlanner(Planner):
    def __init__(self, inner: Planner, inject_synthetic_missing_data: bool = False) -> None:
        self._inner = inner
        self._inject_synthetic_missing_data = inject_synthetic_missing_data
        self.record = PlannerRecord()

    async def create_plan(self, context: EngineContext) -> Plan:
        self.record.create_plan_count += 1
        inner_plan = await self._inner.create_plan(context)
        tracking_plan = TrackingPlan(inner_plan, self._inject_synthetic_missing_data)
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


class Test_that_tool_orchestration_planner_defers_tool_with_missing_dependency(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.find_address_called = False
        self.ship_order_call_count = 0
        self.ship_order_address: str | None = None

        self.agent = await server.create_agent(
            name="Shipping Agent",
            description="Agent that helps customers ship orders",
            planner=self.tracking_planner,
        )

        @p.tool
        async def find_address(context: ToolContext, customer_number: str) -> ToolResult:
            self.find_address_called = True
            return ToolResult(data={"address": "123 Main St"})

        @p.tool
        async def ship_order(context: ToolContext, customer_address: str) -> ToolResult:
            self.ship_order_call_count += 1
            self.ship_order_address = customer_address
            return ToolResult(data={"status": "shipped", "address": customer_address})

        await self.agent.attach_tool(
            tool=find_address,
            condition="You may need to find a customer's address based on their customer number",
        )

        await self.agent.attach_tool(
            tool=ship_order,
            condition="the user wants to ship their order",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please ship my order. My customer number is 12345.",
            recipient=self.agent,
        )

        assert self.find_address_called, "Expected find_address to be called"
        assert self.tracking_planner.record.create_plan_count == 1

        plan = self.tracking_planner.record.plans[0]
        assert plan.record.tools_inferred_count >= 1

        first_iteration_calls = plan.record.inferred_tool_calls[0]
        assert len(first_iteration_calls) == 1, (
            f"Expected orchestrator to defer ship_order on first iteration, "
            f"but {len(first_iteration_calls)} tool calls were passed through"
        )
        assert "find_address" in str(first_iteration_calls[0].tool_id), (
            f"Expected find_address on first iteration, got {first_iteration_calls[0].tool_id}"
        )

        assert self.ship_order_call_count == 1, (
            f"Expected ship_order to be called exactly once after find_address, "
            f"got {self.ship_order_call_count}"
        )
        assert self.ship_order_address == "123 Main St", (
            f"Expected ship_order to be called with '123 Main St' (the address "
            f"returned by find_address), got {self.ship_order_address!r}"
        )


class Test_that_tool_orchestration_planner_reports_missing_argument_when_address_cannot_be_resolved(
    SDKTest,
):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.ship_order_called = False

        self.agent = await server.create_agent(
            name="Shipping Agent",
            description="Agent that helps customers ship orders",
            planner=self.tracking_planner,
        )

        @p.tool
        async def ship_order(context: ToolContext, customer_address: str) -> ToolResult:
            self.ship_order_called = True
            return ToolResult(data={"status": "shipped", "address": customer_address})

        await self.agent.attach_tool(
            tool=ship_order,
            condition="the user wants to ship their order",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please ship my order. My customer number is 12345.",
            recipient=self.agent,
        )

        assert not self.ship_order_called, (
            "Expected ship_order not to be called when customer_address cannot be resolved"
        )

        plan = self.tracking_planner.record.plans[0]
        all_missing_params = [
            d.parameter
            for insights in plan.record.tool_insights_after_on_tools_called
            for d in insights.missing_data
        ]
        assert any("address" in param for param in all_missing_params), (
            f"Expected a missing_data entry for customer_address, got {all_missing_params}"
        )


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


class Test_that_tool_orchestration_plan_clears_tool_insights_when_deferring(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            ),
            inject_synthetic_missing_data=True,
        )

        self.agent = await server.create_agent(
            name="Shipping Agent",
            description="Agent that helps customers ship orders",
            planner=self.tracking_planner,
        )

        @p.tool
        async def find_address(context: ToolContext, customer_number: str) -> ToolResult:
            return ToolResult(data={"address": "123 Main St"})

        @p.tool
        async def ship_order(context: ToolContext, customer_address: str) -> ToolResult:
            return ToolResult(data={"status": "shipped", "address": customer_address})

        await self.agent.attach_tool(
            tool=find_address,
            condition="You may need to find a customer's address based on their customer number",
        )
        await self.agent.attach_tool(
            tool=ship_order,
            condition="the user wants to ship their order",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please ship my order. My customer number is 12345.",
            recipient=self.agent,
        )

        plan = self.tracking_planner.record.plans[0]
        deferring_iterations = [
            i for i, n in enumerate(plan.record.needs_additional_iteration_history) if n
        ]
        assert deferring_iterations, (
            "Expected planner to defer (needs_additional_iteration=True) on at least one iteration"
        )

        for i in deferring_iterations:
            insights = plan.record.tool_insights_after_on_tools_called[i]
            assert insights.missing_data == [], (
                f"Expected missing_data to be cleared on deferring iteration {i}, "
                f"got {insights.missing_data}"
            )
            assert insights.invalid_data == [], (
                f"Expected invalid_data to be cleared on deferring iteration {i}, "
                f"got {insights.invalid_data}"
            )


class Test_that_tool_orchestration_plan_preserves_tool_insights_when_not_deferring(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            ),
            inject_synthetic_missing_data=True,
        )

        self.agent = await server.create_agent(
            name="Balance Agent",
            description="Agent that helps customers check their account balance",
            planner=self.tracking_planner,
        )

        @p.tool
        async def get_account_balance(context: ToolContext, account_id: str) -> ToolResult:
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

        plan = self.tracking_planner.record.plans[0]
        assert plan.record.tools_called_count >= 1
        assert all(n is False for n in plan.record.needs_additional_iteration_history), (
            f"Expected no deferring iterations, got "
            f"{plan.record.needs_additional_iteration_history}"
        )

        for i, insights in enumerate(plan.record.tool_insights_after_on_tools_called):
            assert any(d.parameter == "synthetic_param" for d in insights.missing_data), (
                f"Expected synthetic missing_data to be preserved on iteration {i}, "
                f"got {insights.missing_data}"
            )


class Test_that_tool_orchestration_planner_reevaluates_guideline_condition_based_on_prior_tool_result(
    SDKTest
):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.get_subscription_status_called = False
        self.activate_enterprise_features_called = False

        self.agent = await server.create_agent(
            name="Enterprise Agent",
            description="Agent that helps customers activate enterprise features",
            planner=self.tracking_planner,
        )

        @p.tool
        async def get_subscription_status(context: ToolContext, user_id: str) -> ToolResult:
            self.get_subscription_status_called = True
            return ToolResult(data={"user_id": user_id, "plan": "enterprise"})

        @p.tool
        async def activate_enterprise_features(context: ToolContext, user_id: str) -> ToolResult:
            self.activate_enterprise_features_called = True
            return ToolResult(data={"user_id": user_id, "status": "activated"})

        await self.agent.attach_tool(
            tool=get_subscription_status,
            condition="you need to determine the user's subscription plan",
        )

        await self.agent.attach_tool(
            tool=activate_enterprise_features,
            condition="the user's subscription plan has been confirmed as enterprise",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please activate enterprise features. My user ID is USR001.",
            recipient=self.agent,
        )

        assert self.get_subscription_status_called, "Expected get_subscription_status to be called"
        assert self.activate_enterprise_features_called, (
            "Expected activate_enterprise_features to be called after subscription "
            "status was confirmed as enterprise in a prior iteration"
        )


class Test_that_tool_orchestration_planner_defers_tool_whose_description_requires_prior_tool(
    SDKTest,
):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.diagnose_system_called = False
        self.file_incident_report_called = False
        self.file_incident_report_diagnosis: str | None = None

        self.agent = await server.create_agent(
            name="IT Support Agent",
            description="Agent that handles system diagnostics and incident reporting",
            planner=self.tracking_planner,
        )

        @p.tool
        async def diagnose_system(context: ToolContext, component: str) -> ToolResult:
            """Diagnoses a system component and returns a summary of detected issues. Always run this before filing an incident report."""
            self.diagnose_system_called = True
            return ToolResult(
                data={"component": component, "issues": ["high CPU usage", "memory leak"]}
            )

        @p.tool
        async def file_incident_report(
            context: ToolContext, component: str, diagnosis: str
        ) -> ToolResult:
            """Files an incident report for a system component. Must only be called after diagnose_system has been run for the same component; use the issues summary from diagnose_system as the diagnosis argument."""
            self.file_incident_report_called = True
            self.file_incident_report_diagnosis = diagnosis
            return ToolResult(data={"ticket_id": "INC-001", "status": "filed"})

        await self.agent.attach_tool(
            tool=diagnose_system,
            condition="a system component needs to be diagnosed",
        )

        await self.agent.attach_tool(
            tool=file_incident_report,
            condition="the user wants to file an incident report",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="The payment system is having issues. Please file an incident report.",
            recipient=self.agent,
        )

        assert self.diagnose_system_called, "Expected diagnose_system to be called"
        assert self.file_incident_report_called, "Expected file_incident_report to be called"

        plan = self.tracking_planner.record.plans[0]
        first_iteration_calls = plan.record.inferred_tool_calls[0]
        assert len(first_iteration_calls) == 1, (
            f"Expected planner to defer file_incident_report on first iteration per its description, "
            f"but {len(first_iteration_calls)} tool calls were passed through"
        )
        assert "diagnose_system" in str(first_iteration_calls[0].tool_id), (
            f"Expected diagnose_system on first iteration, got {first_iteration_calls[0].tool_id}"
        )

        assert self.file_incident_report_diagnosis is not None and (
            "cpu" in self.file_incident_report_diagnosis.lower()
            or "memory" in self.file_incident_report_diagnosis.lower()
            or "usage" in self.file_incident_report_diagnosis.lower()
        ), (
            f"Expected file_incident_report to be called with diagnosis from diagnose_system, "
            f"got {self.file_incident_report_diagnosis!r}"
        )


class Test_that_tool_orchestration_planner_prefers_account_number_from_tool_over_customer_provided(
    SDKTest,
):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.find_account_number_called = False
        self.charge_order_called = False
        self.charge_order_account_number: str | None = None

        self.agent = await server.create_agent(
            name="Billing Agent",
            description="Agent that handles billing and order charges",
            planner=self.tracking_planner,
        )

        self.customer = await server.create_customer("John Doe")

        self.email_var = await self.agent.create_variable(
            name="email",
            description="The customer's email address",
        )
        await self.email_var.set_value_for_customer(self.customer, "ddd@gmail.com")

        @p.tool
        async def find_account_number_from_email(context: ToolContext, email: str) -> ToolResult:
            """Looks up the account number associated with a customer's email address."""
            self.find_account_number_called = True
            return ToolResult(data={"account_number": "654321"})

        @p.tool(consequential=True)
        async def charge_order(
            context: ToolContext, account_number: str, order_id: str
        ) -> ToolResult:
            """Charges an order to the specified account. When account_number can be obtained from find_account_number_from_email, prefer that value over any account number provided by the customer."""
            self.charge_order_called = True
            self.charge_order_account_number = account_number
            return ToolResult(data={"order_id": order_id, "account_number": account_number})

        await self.agent.attach_tool(
            tool=find_account_number_from_email,
            condition="the customer's email address is available in context",
        )

        await self.agent.attach_tool(
            tool=charge_order,
            condition="the user wants to charge an order",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please charge order ORD-999. My account number is 123456.",
            recipient=self.agent,
            sender=self.customer,
        )

        assert self.find_account_number_called, (
            "Expected find_account_number_from_email to be called"
        )
        assert self.charge_order_called, "Expected charge_order to be called"
        assert self.charge_order_account_number == "654321", (
            f"Expected charge_order to use account number 654321 from find_account_number_from_email, "
            f"got {self.charge_order_account_number!r}"
        )

        plan = self.tracking_planner.record.plans[0]
        first_iteration_calls = plan.record.inferred_tool_calls[0]
        assert len(first_iteration_calls) == 1, (
            f"Expected planner to defer charge_order on first iteration, "
            f"but {len(first_iteration_calls)} tool calls were passed through"
        )
        assert "find_account_number_from_email" in str(first_iteration_calls[0].tool_id), (
            f"Expected find_account_number_from_email on first iteration, "
            f"got {first_iteration_calls[0].tool_id}"
        )


class Test_that_tool_orchestration_planner_selects_specific_over_general_overlapping_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.vehicle_price_call_count = 0
        self.motorcycle_price_call_count = 0

        self.agent = await server.create_agent(
            name="Vehicle Sales Agent",
            description="Agent that provides vehicle pricing information",
            planner=self.tracking_planner,
        )

        @p.tool
        async def check_vehicle_price(context: ToolContext, model: str) -> ToolResult:
            """Returns the listed price for any vehicle model."""
            self.vehicle_price_call_count += 1
            return ToolResult(data={"model": model, "price": 25000})

        @p.tool
        async def check_motorcycle_price(context: ToolContext, model: str) -> ToolResult:
            """Returns the listed price for a motorcycle model. Use this instead of check_vehicle_price when the vehicle is a motorcycle."""
            self.motorcycle_price_call_count += 1
            return ToolResult(data={"model": model, "price": 18000})

        await self.agent.attach_tool(
            tool=check_vehicle_price,
            condition="the user asks about vehicle pricing",
        )

        await self.agent.attach_tool(
            tool=check_motorcycle_price,
            condition="the user asks about motorcycle pricing",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="What's the price of a Harley-Davidson Street Glide?",
            recipient=self.agent,
        )

        assert self.motorcycle_price_call_count == 1, (
            "Expected check_motorcycle_price to be called once"
        )
        assert self.vehicle_price_call_count == 0, (
            "Expected check_vehicle_price not to be called — overlapped by the more specific check_motorcycle_price"
        )

        plan = self.tracking_planner.record.plans[0]
        first_iteration_calls = plan.record.inferred_tool_calls[0]
        assert len(first_iteration_calls) == 1, (
            f"Expected planner to select only one of the two overlapping tools, "
            f"but {len(first_iteration_calls)} tool calls were passed through"
        )
        assert "motorcycle" in str(first_iteration_calls[0].tool_id), (
            f"Expected check_motorcycle_price to be selected, got {first_iteration_calls[0].tool_id}"
        )


class Test_that_tool_orchestration_planner_selects_preferred_tool_when_two_overlap(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.get_account_balance_call_count = 0
        self.get_account_summary_call_count = 0

        self.agent = await server.create_agent(
            name="Banking Agent",
            description="Agent that provides account information to customers",
            planner=self.tracking_planner,
        )

        @p.tool
        async def get_account_balance(context: ToolContext, account_id: str) -> ToolResult:
            """Returns the current balance for the given account."""
            self.get_account_balance_call_count += 1
            return ToolResult(data={"account_id": account_id, "balance": 1500.00})

        @p.tool
        async def get_account_summary(context: ToolContext, account_id: str) -> ToolResult:
            """Returns a full account summary including balance, recent transactions, and account status. Use this instead of get_account_balance when the customer wants an account overview."""
            self.get_account_summary_call_count += 1
            return ToolResult(
                data={
                    "account_id": account_id,
                    "balance": 1500.00,
                    "status": "active",
                    "recent_transactions": 3,
                }
            )

        await self.agent.attach_tool(
            tool=get_account_balance,
            condition="the user asks about their account balance or account overview",
        )

        await self.agent.attach_tool(
            tool=get_account_summary,
            condition="the user asks about their account balance or account overview",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Can you show me my account overview? My account ID is ACC-789.",
            recipient=self.agent,
        )

        assert self.get_account_summary_call_count == 1, (
            "Expected get_account_summary to be called once"
        )
        assert self.get_account_balance_call_count == 0, (
            "Expected get_account_balance not to be called — overlapped by the more complete get_account_summary"
        )

        plan = self.tracking_planner.record.plans[0]
        first_iteration_calls = plan.record.inferred_tool_calls[0]
        assert len(first_iteration_calls) == 1, (
            f"Expected planner to select only one of the two overlapping tools, "
            f"but {len(first_iteration_calls)} tool calls were passed through"
        )
        assert "summary" in str(first_iteration_calls[0].tool_id), (
            f"Expected get_account_summary to be selected, got {first_iteration_calls[0].tool_id}"
        )


class Test_that_tool_orchestration_planner_handles_three_iteration_tool_chain(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.get_patient_id_called = False
        self.get_medical_record_called = False
        self.book_appointment_record_id: str | None = None
        self.book_appointment_specialty: str | None = None

        self.agent = await server.create_agent(
            name="Healthcare Agent",
            description="Agent that helps patients book specialist appointments",
            planner=self.tracking_planner,
        )

        @p.tool
        async def get_patient_id(context: ToolContext, name: str) -> ToolResult:
            """Looks up a patient's ID by their full name."""
            self.get_patient_id_called = True
            return ToolResult(data={"patient_id": "P-001"})

        @p.tool
        async def get_medical_record(context: ToolContext, patient_id: str) -> ToolResult:
            """Retrieves a patient's medical record by patient ID. Must be called after get_patient_id to obtain the patient_id."""
            self.get_medical_record_called = True
            return ToolResult(data={"record_id": "REC-001", "patient_id": patient_id})

        @p.tool
        async def book_specialist_appointment(
            context: ToolContext, record_id: str, specialty: str
        ) -> ToolResult:
            """Books a specialist appointment for a patient. Must be called after get_medical_record to obtain the record_id."""
            self.book_appointment_record_id = record_id
            self.book_appointment_specialty = specialty
            return ToolResult(
                data={"status": "booked", "record_id": record_id, "specialty": specialty}
            )

        await self.agent.attach_tool(
            tool=get_patient_id,
            condition="the user wants to book an appointment and provides a patient name",
        )

        await self.agent.attach_tool(
            tool=get_medical_record,
            condition="the user wants to book an appointment and provides a patient name",
        )

        await self.agent.attach_tool(
            tool=book_specialist_appointment,
            condition="the user wants to book a specialist appointment",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please book a cardiology appointment for patient John Smith.",
            recipient=self.agent,
        )

        assert self.get_patient_id_called, "Expected get_patient_id to be called"
        assert self.get_medical_record_called, "Expected get_medical_record to be called"
        assert self.book_appointment_record_id == "REC-001", (
            f"Expected book_specialist_appointment to receive record_id='REC-001', "
            f"got {self.book_appointment_record_id!r}"
        )
        assert self.book_appointment_specialty is not None, (
            "Expected book_specialist_appointment to be called with a specialty"
        )

        plan = self.tracking_planner.record.plans[0]
        assert len(plan.record.inferred_tool_calls) >= 3, (
            f"Expected at least 3 iterations for the three-step chain, "
            f"got {len(plan.record.inferred_tool_calls)}"
        )
        first_iteration_calls = plan.record.inferred_tool_calls[0]
        assert len(first_iteration_calls) == 1, (
            f"Expected exactly 1 tool call on first iteration (get_patient_id only), "
            f"got {len(first_iteration_calls)}"
        )
        assert "get_patient_id" in str(first_iteration_calls[0].tool_id), (
            f"Expected get_patient_id on first iteration, got {first_iteration_calls[0].tool_id}"
        )


class Test_that_tool_orchestration_planner_handles_overlap_and_dependency_simultaneously(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tracking_planner = TrackingPlanner(
            p.ToolOrchestrationPlanner(
                logger=server.container[p.Logger],
                tracer=server.container[p.Tracer],
                nlp_service=server.container[p.NLPService],
            )
        )
        self.get_product_price_called = False
        self.get_catalog_price_called = False
        self.apply_member_discount_listed_price: float | None = None

        self.agent = await server.create_agent(
            name="E-Commerce Agent",
            description="Agent that helps customers with product pricing and member discounts",
            planner=self.tracking_planner,
        )

        @p.tool
        async def get_product_price(context: ToolContext, product_id: str) -> ToolResult:
            """Returns the current real-time price for a specific product. Preferred over get_catalog_price."""
            self.get_product_price_called = True
            return ToolResult(data={"product_id": product_id, "price": 49.99})

        @p.tool
        async def get_catalog_price(context: ToolContext, product_id: str) -> ToolResult:
            """Returns the general catalog price for a product. Use get_product_price instead when available, as it returns more accurate real-time pricing."""
            self.get_catalog_price_called = True
            return ToolResult(data={"product_id": product_id, "price": 55.00})

        @p.tool
        async def apply_member_discount(
            context: ToolContext, listed_price: float, member_id: str
        ) -> ToolResult:
            """Applies a member discount to a product price. Must be called after get_product_price to obtain the listed_price."""
            self.apply_member_discount_listed_price = listed_price
            return ToolResult(
                data={
                    "member_id": member_id,
                    "original_price": listed_price,
                    "discounted_price": listed_price * 0.9,
                }
            )

        await self.agent.attach_tool(
            tool=get_product_price,
            condition="the user asks about the price of a product",
        )

        await self.agent.attach_tool(
            tool=get_catalog_price,
            condition="the user asks about the price of a product and no product price has been retrieved from a tool yet in this conversation",
        )

        await self.agent.attach_tool(
            tool=apply_member_discount,
            condition="the user is a member and wants a discounted price for a product",
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="I'm a member with ID M-123. What's the discounted price for product PRD-42?",
            recipient=self.agent,
        )

        assert self.get_product_price_called, "Expected get_product_price to be called"
        # Verify overlap: the dependency tool received the price from get_product_price (49.99),
        # not from get_catalog_price (55.00), confirming get_product_price was preferred.
        assert self.apply_member_discount_listed_price == 49.99, (
            f"Expected apply_member_discount to receive listed_price=49.99 from get_product_price, "
            f"got {self.apply_member_discount_listed_price!r}"
        )

        plan = self.tracking_planner.record.plans[0]
        first_iteration_calls = plan.record.inferred_tool_calls[0]
        # Overlap: orchestrator selected get_product_price over get_catalog_price in iter 1.
        assert len(first_iteration_calls) == 1, (
            f"Expected exactly 1 tool call on first iteration "
            f"(get_product_price only — overlap over get_catalog_price), "
            f"got {len(first_iteration_calls)}: {[str(c.tool_id) for c in first_iteration_calls]}"
        )
        assert "get_product_price" in str(first_iteration_calls[0].tool_id), (
            f"Expected get_product_price on first iteration, got {first_iteration_calls[0].tool_id}"
        )
