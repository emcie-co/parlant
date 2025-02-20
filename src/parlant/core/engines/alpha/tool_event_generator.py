# Copyright 2024 Emcie Co Ltd.
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

from dataclasses import dataclass
from itertools import chain
from typing import Mapping, Optional, Sequence

from parlant.core.customers import Customer
from parlant.core.tools import ToolContext
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.loggers import Logger
from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, SessionId, ToolEventData
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.glossary import Term
from parlant.core.engines.alpha.tool_caller import ToolCallInferenceSchema, ToolCaller, ToolInsights
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.tools import ToolId


@dataclass(frozen=True)
class ToolEventGenerationResult:
    generations: Sequence[GenerationInfo]
    events: Sequence[Optional[EmittedEvent]]
    insights: ToolInsights


class ToolEventGenerator:
    def __init__(
        self,
        logger: Logger,
        correlator: ContextualCorrelator,
        service_registry: ServiceRegistry,
        schematic_generator: SchematicGenerator[ToolCallInferenceSchema],
    ) -> None:
        self._logger = logger
        self._correlator = correlator
        self._service_registry = service_registry

        self.tool_caller = ToolCaller(logger, service_registry, schematic_generator)

    async def generate_events(
        self,
        event_emitter: EventEmitter,
        session_id: SessionId,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
    ) -> ToolEventGenerationResult:
        if not tool_enabled_guideline_propositions:
            self._logger.debug("Skipping tool calling; no tools associated with guidelines found")
            return ToolEventGenerationResult(generations=[], events=[], insights=ToolInsights())

        inference_result = await self.tool_caller.infer_tool_calls(
            agent,
            context_variables,
            interaction_history,
            terms,
            ordinary_guideline_propositions,
            tool_enabled_guideline_propositions,
            staged_events,
        )

        tool_calls = list(chain.from_iterable(inference_result.batches))

        if not tool_calls:
            return ToolEventGenerationResult(
                generations=inference_result.batch_generations,
                events=[],
                insights=inference_result.insights,
            )

        tool_results = await self.tool_caller.execute_tool_calls(
            ToolContext(
                agent_id=agent.id,
                session_id=session_id,
                customer_id=customer.id,
            ),
            tool_calls,
        )

        if not tool_results:
            return ToolEventGenerationResult(
                generations=inference_result.batch_generations,
                events=[],
                insights=inference_result.insights,
            )

        event_data: ToolEventData = {
            "tool_calls": [
                {
                    "tool_id": r.tool_call.tool_id.to_string(),
                    "arguments": r.tool_call.arguments,
                    "result": r.result,
                }
                for r in tool_results
            ]
        }

        event = await event_emitter.emit_tool_event(
            correlation_id=self._correlator.correlation_id,
            data=event_data,
        )

        return ToolEventGenerationResult(
            generations=inference_result.batch_generations,
            events=[event],
            insights=inference_result.insights,
        )
