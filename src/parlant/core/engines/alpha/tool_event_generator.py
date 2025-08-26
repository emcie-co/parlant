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

from dataclasses import dataclass
from itertools import chain
from typing import Mapping, Optional, Sequence
from parlant.core.customers import Customer
from parlant.core.engines.alpha.loaded_context import LoadedContext
from parlant.core.tools import ToolContext
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.loggers import Logger
from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, SessionId, ToolEventData
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.glossary import Term
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCallContext,
    ToolCaller,
    ToolInsights,
)
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.tools import ToolId


@dataclass(frozen=True)
class ToolEventGenerationResult:
    generations: Sequence[GenerationInfo]
    events: Sequence[Optional[EmittedEvent]]
    insights: ToolInsights


@dataclass(frozen=True)
class ToolPreexecutionState:
    event_emitter: EventEmitter
    session_id: SessionId
    agent: Agent
    customer: Customer
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    ordinary_guideline_matches: Sequence[GuidelineMatch]
    tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]]
    staged_events: Sequence[EmittedEvent]


class ToolEventGenerator:
    def __init__(
        self,
        logger: Logger,
        tool_caller: ToolCaller,
        correlator: ContextualCorrelator,
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._correlator = correlator
        self._service_registry = service_registry
        self._tool_caller = tool_caller

    async def create_preexecution_state(
        self,
        event_emitter: EventEmitter,
        session_id: SessionId,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
    ) -> ToolPreexecutionState:
        return ToolPreexecutionState(
            event_emitter,
            session_id,
            agent,
            customer,
            context_variables,
            interaction_history,
            terms,
            ordinary_guideline_matches,
            tool_enabled_guideline_matches,
            staged_events,
        )

    async def generate_events(
        self,
        preexecution_state: ToolPreexecutionState,
        context: LoadedContext,
    ) -> ToolEventGenerationResult:
        _ = preexecution_state  # Not used for now, but good to have for extensibility

        if not context.state.tool_enabled_guideline_matches:
            self._logger.trace("Skipping tool calling; no tools associated with guidelines found")
            return ToolEventGenerationResult(generations=[], events=[], insights=ToolInsights())

        await context.session_event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "status": "processing",
                "data": {"stage": "Fetching data"},
            },
        )

        tool_call_context = ToolCallContext(
            agent=context.agent,
            session_id=context.session.id,
            customer_id=context.customer.id,
            context_variables=context.state.context_variables,
            interaction_history=context.interaction.history,
            terms=list(context.state.glossary_terms),
            ordinary_guideline_matches=context.state.ordinary_guideline_matches,
            tool_enabled_guideline_matches=context.state.tool_enabled_guideline_matches,
            journeys=context.state.journeys,
            staged_events=context.state.tool_events,
        )

        inference_result = await self._tool_caller.infer_tool_calls(
            context=tool_call_context,
        )

        tool_calls = list(chain.from_iterable(inference_result.batches))

        if not tool_calls:
            return ToolEventGenerationResult(
                generations=inference_result.batch_generations,
                events=[],
                insights=inference_result.insights,
            )

        tool_context = ToolContext(
            agent_id=context.agent.id,
            session_id=context.session.id,
            customer_id=context.customer.id,
        )

        tool_results = await self._tool_caller.execute_tool_calls(
            tool_context,
            tool_calls,
        )

        if not tool_results:
            return ToolEventGenerationResult(
                generations=inference_result.batch_generations,
                events=[],
                insights=inference_result.insights,
            )

        events = []
        for r in tool_results:
            event_data: ToolEventData = {
                "tool_calls": [
                    {
                        "tool_id": r.tool_call.tool_id.to_string(),
                        "arguments": r.tool_call.arguments,
                        "result": r.result,
                    }
                ]
            }
            if r.result["control"].get("lifespan", "session") == "session":
                events.append(
                    await context.session_event_emitter.emit_tool_event(
                        correlation_id=self._correlator.correlation_id,
                        data=event_data,
                    )
                )
            else:
                events.append(
                    await context.response_event_emitter.emit_tool_event(
                        correlation_id=self._correlator.correlation_id,
                        data=event_data,
                    )
                )

        return ToolEventGenerationResult(
            generations=inference_result.batch_generations,
            events=events,
            insights=inference_result.insights,
        )
