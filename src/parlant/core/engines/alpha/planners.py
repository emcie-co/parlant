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
from itertools import chain
from typing import Sequence

from parlant.core.agents import AgentId
from parlant.core.engines.alpha.engine_context import EngineContext
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCall,
    ToolCallInferenceResult,
    ToolCallResult,
)
from parlant.core.loggers import Logger
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


class PlannerProvider:
    """Provides planners on a per-agent basis."""

    def __init__(self, default_planner: Planner) -> None:
        self._default_planner = default_planner
        self._agent_planners: dict[AgentId, Planner] = {}

    def get_planner(self, agent_id: AgentId) -> Planner:
        return self._agent_planners.get(agent_id, self._default_planner)

    def set_planner(self, agent_id: AgentId, planner: Planner) -> None:
        self._agent_planners[agent_id] = planner
