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

from abc import ABC, abstractmethod
import json
from itertools import chain
from typing import Sequence

from parlant.core.agents import AgentId
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.engine_context import EngineContext
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCall,
    ToolCallInferenceResult,
    ToolCallResult,
    ToolInsights,
)
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.service import NLPService
from parlant.core.tracer import Tracer

_PLANNER_SPAN_NAME = "planner"


class Plan(ABC):
    def __init__(self) -> None:
        self.needs_additional_iteration: bool = False
        self.thoughts: list[str] = []

    @property
    @abstractmethod
    def reasoning(self) -> str: ...

    @abstractmethod
    async def on_guidelines_matched(
        self,
        context: EngineContext,
        matched_guidelines: list[GuidelineMatch],
    ) -> None:
        """Called after guideline matching but before relational resolution."""
        ...

    @abstractmethod
    async def on_guidelines_resolved(self, context: EngineContext) -> None:
        """Called after relational resolution and tool-enabled/ordinary split."""
        ...

    @abstractmethod
    async def on_tools_inferred(
        self,
        context: EngineContext,
        inference_result: ToolCallInferenceResult,
    ) -> Sequence[ToolCall]:
        """Called after tool calls have been inferred but before they're executed."""
        ...

    @abstractmethod
    async def on_tools_called(
        self,
        context: EngineContext,
        tool_results: Sequence[ToolCallResult],
    ) -> None:
        """Called after tool calls have been executed."""
        ...


class Planner(ABC):
    @abstractmethod
    async def create_plan(self, context: EngineContext) -> Plan: ...


class NullPlan(Plan):
    @property
    def reasoning(self) -> str:
        return ""

    async def on_guidelines_matched(
        self,
        context: EngineContext,
        matched_guidelines: list[GuidelineMatch],
    ) -> None:
        pass

    async def on_guidelines_resolved(self, context: EngineContext) -> None:
        pass

    async def on_tools_inferred(
        self,
        context: EngineContext,
        inference_result: ToolCallInferenceResult,
    ) -> Sequence[ToolCall]:
        return list(chain.from_iterable(inference_result.batches))

    async def on_tools_called(
        self,
        context: EngineContext,
        tool_results: Sequence[ToolCallResult],
    ) -> None:
        pass


class NullPlanner(Planner):
    async def create_plan(self, context: EngineContext) -> Plan:
        return NullPlan()


class BasicPlan(Plan):
    """Base plan with built-in tracing and logger scoping.

    Derived classes implement do_ methods instead of on_ methods.
    """

    def __init__(self, logger: Logger, tracer: Tracer) -> None:
        super().__init__()
        self._logger = logger
        self._tracer = tracer

    @abstractmethod
    async def do_on_guidelines_matched(
        self,
        context: EngineContext,
        matched_guidelines: list[GuidelineMatch],
    ) -> None: ...

    @abstractmethod
    async def do_on_guidelines_resolved(self, context: EngineContext) -> None: ...

    @abstractmethod
    async def do_on_tools_inferred(
        self,
        context: EngineContext,
        inference_result: ToolCallInferenceResult,
    ) -> Sequence[ToolCall]: ...

    @abstractmethod
    async def do_on_tools_called(
        self,
        context: EngineContext,
        tool_results: Sequence[ToolCallResult],
    ) -> None: ...

    async def on_guidelines_matched(
        self,
        context: EngineContext,
        matched_guidelines: list[GuidelineMatch],
    ) -> None:
        with self._logger.scope(type(self).__name__):
            with self._tracer.span(_PLANNER_SPAN_NAME):
                await self.do_on_guidelines_matched(context, matched_guidelines)

    async def on_guidelines_resolved(self, context: EngineContext) -> None:
        with self._logger.scope(type(self).__name__):
            with self._tracer.span(_PLANNER_SPAN_NAME):
                await self.do_on_guidelines_resolved(context)

    async def on_tools_inferred(
        self,
        context: EngineContext,
        inference_result: ToolCallInferenceResult,
    ) -> Sequence[ToolCall]:
        with self._logger.scope(type(self).__name__):
            with self._tracer.span(_PLANNER_SPAN_NAME):
                return await self.do_on_tools_inferred(context, inference_result)

    async def on_tools_called(
        self,
        context: EngineContext,
        tool_results: Sequence[ToolCallResult],
    ) -> None:
        with self._logger.scope(type(self).__name__):
            with self._tracer.span(_PLANNER_SPAN_NAME):
                await self.do_on_tools_called(context, tool_results)


class BasicPlanner(Planner):
    """Base planner with built-in tracing and logger scoping.

    Derived classes implement do_create_plan() instead of create_plan().
    """

    def __init__(self, logger: Logger, tracer: Tracer) -> None:
        self._logger = logger
        self._tracer = tracer

    @abstractmethod
    async def do_create_plan(self, context: EngineContext) -> Plan: ...

    async def create_plan(self, context: EngineContext) -> Plan:
        with self._logger.scope(type(self).__name__):
            with self._tracer.span(_PLANNER_SPAN_NAME):
                return await self.do_create_plan(context)


class ToolInferenceChoiceSchema(DefaultBaseModel):
    tool_call_ids: list[str]
    further_iteration_needed: bool


class ToolOrchestrationPlan(BasicPlan):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        schematic_generator: SchematicGenerator[ToolInferenceChoiceSchema],
    ) -> None:
        super().__init__(logger, tracer)
        self._schematic_generator = schematic_generator

    @property
    def reasoning(self) -> str:
        return ""

    async def do_on_guidelines_matched(
        self,
        context: EngineContext,
        matched_guidelines: list[GuidelineMatch],
    ) -> None:
        pass

    async def do_on_guidelines_resolved(self, context: EngineContext) -> None:
        pass

    async def do_on_tools_inferred(
        self,
        context: EngineContext,
        inference_result: ToolCallInferenceResult,
    ) -> Sequence[ToolCall]:
        all_calls = list(chain.from_iterable(inference_result.batches))
        has_missing_data = bool(inference_result.insights.missing_data)

        if len(all_calls) <= 1:
            self.needs_additional_iteration = len(all_calls) == 1 and has_missing_data
            return all_calls

        call_index = {str(i): call for i, call in enumerate(all_calls, start=1)}

        prompt = self._build_prompt(context, call_index)

        result = await self._schematic_generator.generate(prompt)

        self._logger.trace(f"Completion:\n{result.content.model_dump_json(indent=2)}")

        chosen_ids = set(result.content.tool_call_ids)
        chosen_calls = [call_index[cid] for cid in call_index if cid in chosen_ids]

        if len(chosen_calls) == len(all_calls):
            self.needs_additional_iteration = False
        else:
            self.needs_additional_iteration = result.content.further_iteration_needed

        for cid, call in call_index.items():
            if cid in chosen_ids:
                self._logger.debug(f"Running tool call {cid}: {call.tool_id}")
            else:
                self._logger.debug(f"Deferring tool call {cid}: {call.tool_id}")

        self._logger.debug(f"needs_additional_iteration={self.needs_additional_iteration}")

        return chosen_calls

    async def do_on_tools_called(
        self,
        context: EngineContext,
        tool_results: Sequence[ToolCallResult],
    ) -> None:
        if self.needs_additional_iteration:
            # Forget missing/invalid parameter data from this iteration so that any
            # deferred tool is re-inferred fresh through the standard guideline-matching
            # to tool-calling path in the next iteration.
            context.state.tool_insights = ToolInsights(
                evaluations=context.state.tool_insights.evaluations,
                missing_data=[],
                invalid_data=[],
            )

    def _build_prompt(
        self,
        context: EngineContext,
        call_index: dict[str, ToolCall],
    ) -> PromptBuilder:
        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        tool_calls_text = "\n".join(
            f"{cid}) tool: {call.tool_id}, arguments: {json.dumps(dict(call.arguments))}"
            for cid, call in call_index.items()
        )

        result_structure = json.dumps(
            {
                "tool_call_ids": ["<IDs of tool calls to execute now>"],
                "further_iteration_needed": "<true if remaining tool calls should run in a subsequent iteration, false otherwise>",
            },
            indent=2,
        )

        builder.add_section(
            name="tool-call-planner-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-------------------
You are one part of an agentic AI system whose role is to engage in multi-turn conversations with customers on behalf of a business.
Before every response to the customer, the agent may use a number of tools to receive information or perform real-world changes.
A single LLM call is used per each tool to decide whether it should be used, and if so, with which arguments.
The individual tools are evaluated concurrently, without consideration of what other tools should run.



TASK DESCRIPTION
-------------------

The following tool calls have been inferred for this conversation turn.
Your job is to decide which of these tool calls should be executed right now,
and which (if any) should be deferred to a later iteration.

Consider dependencies between tools: if one tool's output is needed as input
for another, run the dependency first and defer the dependent tool.
If tools are independent of each other, they can all run together.
In case of doubt - prefer running fewer tools at this iteration, and others for later iterations.

You may choose not to perform an inferred tool call iff:
1. The call has an argument which should be updated based on the output of another tool. For example, if one tool call requires the customer's email and there's another tool call to find_customer_email.
2. There are two overlapping tool calls - meaning two calls that perform the same function. In that case choose only the more relevant one.
3. You are given explicit conditions for aborting / postponing a tool call in one of the tool descriptions or guidelines.

Inferred Tool Calls: ###
{tool_calls_text}
###

Respond with a JSON object in this format:
```json
{result_structure}
```
""",
            props={
                "tool_calls_text": tool_calls_text,
                "result_structure": result_structure,
            },
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)
        builder.add_interaction_history(
            context.interaction.events,
            context.state.tool_events,
        )
        builder.add_context_variables(context.state.context_variables)
        builder.add_glossary(list(context.state.glossary_terms))
        builder.add_staged_tool_events(context.state.tool_events)

        all_guidelines = list(
            chain(
                context.state.ordinary_guideline_matches,
                context.state.tool_enabled_guideline_matches.keys(),
            )
        )

        if all_guidelines:
            guidelines_text = "\n".join(
                f"{i}) When {g.guideline.content.condition}, then {g.guideline.content.action}"
                for i, g in enumerate(all_guidelines, start=1)
            )
            builder.add_section(
                name="tool-orchestration-guidelines",
                template="""
Active Guidelines: ###
{guidelines_text}
###
""",
                props={"guidelines_text": guidelines_text},
            )

        return builder


class ToolOrchestrationPlanner(BasicPlanner):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        nlp_service: NLPService,
    ) -> None:
        super().__init__(logger, tracer)
        self._nlp_service = nlp_service

    async def do_create_plan(self, context: EngineContext) -> Plan:
        schematic_generator = await self._nlp_service.get_schematic_generator(
            ToolInferenceChoiceSchema,
        )
        return ToolOrchestrationPlan(self._logger, self._tracer, schematic_generator)


class PlannerProvider:
    """Provides planners on a per-agent basis."""

    def __init__(self, default_planner: Planner) -> None:
        self._default_planner = default_planner
        self._agent_planners: dict[AgentId, Planner] = {}

    def get_planner(self, agent_id: AgentId) -> Planner:
        return self._agent_planners.get(agent_id, self._default_planner)

    def set_planner(self, agent_id: AgentId, planner: Planner) -> None:
        self._agent_planners[agent_id] = planner
