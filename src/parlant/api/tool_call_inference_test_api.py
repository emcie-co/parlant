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

from typing import Mapping, Optional, Sequence
from fastapi import APIRouter, HTTPException, status

from parlant.api.agents import AgentIdPath
from parlant.api.common import JSONSerializableDTO, ToolNameField, apigen_config
from parlant.api.customers import CustomerIdPath
from parlant.api.sessions import EventCorrelationIdField, EventIdPath, EventKindDTO, EventSourceDTO
from parlant.core.agents import AgentStore
from parlant.core.common import DefaultBaseModel
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import CustomerStore
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolCall, ToolCaller
from parlant.core.glossary import Term
from parlant.core.sessions import Event, EventKind, EventSource, SessionStore
from parlant.core.tools import Tool, ToolContext, ToolId, ToolService


API_GROUP = "engine_test"


class UsageInfoDTO(DefaultBaseModel):
    input_tokens: int
    output_tokens: int
    extra: Optional[Mapping[str, int]] = None


class GenerationInfoDTO(DefaultBaseModel):
    schema_name: str
    model: str
    duration: float
    usage: UsageInfoDTO


class ToolCallDTO(DefaultBaseModel):
    tool_name: str
    parameters: Mapping[str, JSONSerializableDTO]
    rationale: str


class ToolCallInferenceResultDTO(DefaultBaseModel):
    total_duration: float
    batch_generations: Sequence[GenerationInfoDTO]
    tool_calls: Sequence[ToolCallDTO]


class EmittedEventDTO(DefaultBaseModel):
    source: EventSourceDTO
    kind: EventKindDTO
    correlation_id: EventCorrelationIdField
    data: JSONSerializableDTO


class ToolCallInferenceParamsDTO(DefaultBaseModel):
    agent_id: AgentIdPath
    customer_id: CustomerIdPath
    events: Sequence[EventIdPath]
    staged_events: Sequence[EmittedEventDTO]
    available_tools: Sequence[ToolNameField]


def create_test_tool_call_inference_router(
    tool_caller: ToolCaller,
    agent_store: AgentStore,
    customer_store: CustomerStore,
    session_store: SessionStore,
    tool_service: ToolService,
) -> APIRouter:
    test_router = APIRouter()

    @test_router.post(
        "",
        status_code=status.HTTP_202_ACCEPTED,
        operation_id="infer_tool_calls",
        response_model=ToolCallInferenceResultDTO,
        responses={
            status.HTTP_202_ACCEPTED: {
                "description": "Tool calls successfully inferred. Returns the inference results.",
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="infer_tool_calls"),
    )
    async def infer_tool_calls(
        params: ToolCallInferenceParamsDTO,
    ) -> ToolCallInferenceResultDTO:
        agent = await agent_store.read_agent(params.agent_id)
        customer = await customer_store.read_customer(params.customer_id)

        sessions = await session_store.list_sessions(params.agent_id, params.customer_id)
        if len(sessions) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No sessions found for given agent and customer",
            )

        session_id = sessions[0].id
        events: list[Event] = []
        for event_id in params.events:
            event = await session_store.read_event(session_id, event_id)
            events.append(event)

        staged_events: list[EmittedEvent] = []
        for staged_event in params.staged_events:
            staged_events.append(
                EmittedEvent(
                    source=EventSource(staged_event.source),
                    kind=EventKind(staged_event.kind),
                    correlation_id=staged_event.correlation_id,
                    data=staged_event.data,
                )
            )

        available_tools: list[Tool] = []
        tool_ids: list[ToolId] = []
        for tool_id_str in params.available_tools:
            tool_id = ToolId.from_string(tool_id_str)
            tool_ids.append(tool_id)
            tool = await tool_service.read_tool(tool_id.tool_name)
            available_tools.append(tool)

        context_variables: list[tuple[ContextVariable, ContextVariableValue]] = []
        terms: list[Term] = []
        ordinary_guideline_matches: list[GuidelineMatch] = []
        tool_enabled_guideline_matches: dict[GuidelineMatch, list[ToolId]] = {}

        tool_context = ToolContext(
            agent_id=agent.id,
            customer_id=customer.id,
            session_id=session_id,
        )

        tool_call_inference_result = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=context_variables,
            interaction_history=events,
            terms=terms,
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            journeys=[],
            staged_events=staged_events,
            tool_context=tool_context,
        )

        batch_generations = [
            GenerationInfoDTO(
                schema_name=gen.schema_name,
                model=gen.model,
                duration=gen.duration,
                usage=UsageInfoDTO(
                    input_tokens=gen.usage.input_tokens,
                    output_tokens=gen.usage.output_tokens,
                    extra=gen.usage.extra,
                ),
            )
            for gen in tool_call_inference_result.batch_generations
        ]

        all_tool_calls: list[ToolCall] = []
        for batch in tool_call_inference_result.batches:
            all_tool_calls.extend(batch)

        tool_calls = []
        for tool_call in all_tool_calls:
            tool_calls.append(
                ToolCallDTO(
                    tool_name=tool_call.tool_id.tool_name,
                    parameters=tool_call.arguments,
                    rationale="Tool call inferred from conversation context",
                )
            )

        return ToolCallInferenceResultDTO(
            total_duration=tool_call_inference_result.total_duration,
            batch_generations=batch_generations,
            tool_calls=tool_calls,
        )

    return test_router
