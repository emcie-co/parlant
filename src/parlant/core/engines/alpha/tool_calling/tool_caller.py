# Copyright 2025 Emcie Co Ltd.
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
from collections import defaultdict
from dataclasses import dataclass, asdict, field
import json
import time
import traceback
from typing import Mapping, NewType, Optional, Sequence

from parlant.core import async_utils
from parlant.core.agents import Agent
from parlant.core.common import JSONSerializable, generate_id
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.glossary import Term
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, ToolResult
from parlant.core.tools import (
    Tool,
    ToolContext,
    ToolId,
    ToolService,
    DEFAULT_PARAMETER_PRECEDENCE,
)

ToolCallId = NewType("ToolCallId", str)
ToolResultId = NewType("ToolResultId", str)


@dataclass(frozen=True)
class ToolCall:
    id: ToolCallId
    tool_id: ToolId
    arguments: Mapping[str, JSONSerializable]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ToolCall):
            return bool(self.tool_id == value.tool_id and self.arguments == value.arguments)
        return False


@dataclass(frozen=True)
class ToolCallResult:
    id: ToolResultId
    tool_call: ToolCall
    result: ToolResult


@dataclass(frozen=True, kw_only=True)
class ProblematicToolData:
    parameter: str
    significance: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)
    examples: Optional[Sequence[str]] = field(default=None)
    precedence: Optional[int] = field(default=DEFAULT_PARAMETER_PRECEDENCE)
    choices: Optional[Sequence[str]] = field(default=None)


@dataclass(frozen=True, kw_only=True)
class MissingToolData(ProblematicToolData):
    pass


@dataclass(frozen=True, kw_only=True)
class InvalidToolData(ProblematicToolData):
    invalid_value: str


@dataclass(frozen=True)
class ToolInsights:
    missing_data: Sequence[MissingToolData] = field(default_factory=list)
    invalid_data: Sequence[InvalidToolData] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCallInferenceResult:
    total_duration: float
    batch_count: int
    batch_generations: Sequence[GenerationInfo]
    batches: Sequence[Sequence[ToolCall]]
    insights: ToolInsights


@dataclass(frozen=True)
class ToolCallContext:
    agent: Agent
    services: dict[str, ToolService]
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    ordinary_guideline_matches: Sequence[GuidelineMatch]
    journeys: Sequence[Journey]
    tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]]
    staged_events: Sequence[EmittedEvent]


@dataclass(frozen=True)
class ToolCallBatchResult:
    tool_calls: Sequence[ToolCall]
    generation_info: GenerationInfo
    insights: ToolInsights


class ToolCallBatch(ABC):
    @abstractmethod
    async def process(self) -> ToolCallBatchResult: ...


class ToolCallBatcher(ABC):
    @abstractmethod
    async def create_batches(
        self,
        tools: Mapping[tuple[ToolId, Tool], Sequence[GuidelineMatch]],
        context: ToolCallContext,
    ) -> Sequence[ToolCallBatch]: ...


class ToolCaller:
    def __init__(
        self,
        logger: Logger,
        service_registry: ServiceRegistry,
        batcher: ToolCallBatcher,
    ) -> None:
        self._logger = logger
        self._service_registry = service_registry
        self.batcher = batcher

    async def infer_tool_calls(
        self,
        agent: Agent,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        journeys: Sequence[Journey],
        staged_events: Sequence[EmittedEvent],
        tool_context: ToolContext,
    ) -> ToolCallInferenceResult:
        with self._logger.scope("ToolCaller"):
            if not tool_enabled_guideline_matches:
                return ToolCallInferenceResult(
                    total_duration=0.0,
                    batch_count=0,
                    batch_generations=[],
                    batches=[],
                    insights=ToolInsights(),
                )

            t_start = time.time()

            tools: dict[tuple[ToolId, Tool], list[GuidelineMatch]] = defaultdict(list)
            services: dict[str, ToolService] = {}

            for guideline_match, tool_ids in tool_enabled_guideline_matches.items():
                for tool_id in tool_ids:
                    if tool_id.service_name not in services:
                        services[
                            tool_id.service_name
                        ] = await self._service_registry.read_tool_service(tool_id.service_name)

                    tool = await services[tool_id.service_name].resolve_tool(
                        tool_id.tool_name, tool_context
                    )

                    tools[(tool_id, tool)].append(guideline_match)

            with self._logger.operation("Creating batches"):
                batches = await self.batcher.create_batches(
                    tools=tools,
                    context=ToolCallContext(
                        agent=agent,
                        services=services,
                        context_variables=context_variables,
                        interaction_history=interaction_history,
                        terms=terms,
                        ordinary_guideline_matches=ordinary_guideline_matches,
                        journeys=journeys,
                        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
                        staged_events=staged_events,
                    ),
                )

            with self._logger.operation("Processing batches"):
                batch_tasks = [batch.process() for batch in batches]
                batch_results = await async_utils.safe_gather(*batch_tasks)

            t_end = time.time()

            # Aggregate insights from all batch results (e.g., missing data across batches)
            aggregated_missing_data: list[MissingToolData] = []
            aggregated_invalid_data: list[InvalidToolData] = []
            for result in batch_results:
                if result.insights and result.insights.missing_data:
                    aggregated_missing_data.extend(result.insights.missing_data)
                if result.insights and result.insights.invalid_data:
                    aggregated_invalid_data.extend(result.insights.invalid_data)

            return ToolCallInferenceResult(
                total_duration=t_end - t_start,
                batch_count=len(batches),
                batch_generations=[result.generation_info for result in batch_results],
                batches=[result.tool_calls for result in batch_results],
                insights=ToolInsights(
                    missing_data=aggregated_missing_data, invalid_data=aggregated_invalid_data
                ),
            )

    async def _run_tool(
        self,
        context: ToolContext,
        tool_call: ToolCall,
        tool_id: ToolId,
    ) -> ToolCallResult:
        try:
            self._logger.debug(
                f"Execution::Invocation: ({tool_call.tool_id.to_string()}/{tool_call.id})"
                + (f"\n{json.dumps(tool_call.arguments, indent=2)}" if tool_call.arguments else "")
            )

            try:
                service = await self._service_registry.read_tool_service(tool_id.service_name)

                result = await service.call_tool(
                    tool_id.tool_name,
                    context,
                    tool_call.arguments,
                )

                self._logger.debug(
                    f"Execution::Result: Tool call succeeded ({tool_call.tool_id.to_string()}/{tool_call.id})\n{json.dumps(asdict(result), indent=2, default=str)}"
                )
            except Exception as exc:
                self._logger.error(
                    f"Execution::Result: Tool call failed ({tool_id.to_string()}/{tool_call.id})\n{traceback.format_exception(exc)}"
                )
                raise

            return ToolCallResult(
                id=ToolResultId(generate_id()),
                tool_call=tool_call,
                result={
                    "data": result.data,
                    "metadata": result.metadata,
                    "control": result.control,
                    "utterances": result.utterances,
                    "utterance_fields": result.utterance_fields,
                },
            )
        except Exception as e:
            self._logger.error(
                f"Execution::Error: ToolId: {tool_call.tool_id.to_string()}', "
                f"Arguments:\n{json.dumps(tool_call.arguments, indent=2)}"
                + "\nTraceback:\n"
                + "\n".join(traceback.format_exception(e)),
            )

            return ToolCallResult(
                id=ToolResultId(generate_id()),
                tool_call=tool_call,
                result={
                    "data": "Tool call error",
                    "metadata": {"error_details": str(e)},
                    "control": {},
                    "utterances": [],
                    "utterance_fields": {},
                },
            )

    async def execute_tool_calls(
        self,
        context: ToolContext,
        tool_calls: Sequence[ToolCall],
    ) -> Sequence[ToolCallResult]:
        with self._logger.scope("ToolCaller"):
            with self._logger.operation("Execution"):
                tool_results = await async_utils.safe_gather(
                    *(
                        self._run_tool(
                            context=context,
                            tool_call=tool_call,
                            tool_id=tool_call.tool_id,
                        )
                        for tool_call in tool_calls
                    )
                )

                return tool_results
