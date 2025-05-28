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
from parlant.api.common import JSONSerializableDTO, apigen_config
from parlant.api.context_variables import ContextVariableIdPath
from parlant.api.customers import CustomerIdPath
from parlant.api.glossary import TermIdPath
from parlant.api.guidelines import GuidelineIdPath
from parlant.api.sessions import EventCorrelationIdField, EventIdPath, EventKindDTO, EventSourceDTO
from parlant.core.agents import AgentStore
from parlant.core.common import DefaultBaseModel
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import CustomerStore
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_match import PreviouslyAppliedType
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import GuidelineMatcher
from parlant.core.glossary import GlossaryStore, Term
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.sessions import Event, EventKind, EventSource, SessionStore


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


class GuidelineMatchDTO(DefaultBaseModel):
    guideline: GuidelineIdPath
    score: int
    rationale: str
    guideline_previously_applied: PreviouslyAppliedType = PreviouslyAppliedType.NO
    guideline_is_continuous: bool = False
    should_reapply: bool = False


class GuidelineMatchingResultDTO(DefaultBaseModel):
    total_duration: float
    batch_count: int
    batch_generations: Sequence[GenerationInfoDTO]
    batches: Sequence[Sequence[GuidelineMatchDTO]]


class EmittedEventDTO(DefaultBaseModel):
    source: EventSourceDTO
    kind: EventKindDTO
    correlation_id: EventCorrelationIdField
    data: JSONSerializableDTO


class GuidelineMatchingParamsDTO(DefaultBaseModel):
    agent_id: AgentIdPath
    customer_id: CustomerIdPath
    context_variables: Sequence[ContextVariableIdPath]
    events: Sequence[EventIdPath]
    terms: Sequence[TermIdPath]
    staged_events: Sequence[EmittedEventDTO]
    guidelines: Sequence[GuidelineIdPath]


def create_test_guideline_matching_router(
    guideline_matcher: GuidelineMatcher,
    agent_store: AgentStore,
    customer_store: CustomerStore,
    context_variable_store: ContextVariableStore,
    guideline_store: GuidelineStore,
    glossary_store: GlossaryStore,
    session_store: SessionStore,
) -> APIRouter:
    test_router = APIRouter()

    @test_router.post(
        "",
        status_code=status.HTTP_202_ACCEPTED,
        operation_id="match_guidelines",
        response_model=GuidelineMatchingResultDTO,
        responses={
            status.HTTP_202_ACCEPTED: {
                "description": "Guidelines successfully matched. Returns the matching results.",
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="match_guidelines"),
    )
    async def match_guidelines(
        params: GuidelineMatchingParamsDTO,
    ) -> GuidelineMatchingResultDTO:
        agent = await agent_store.read_agent(params.agent_id)
        customer = await customer_store.read_customer(params.customer_id)

        context_variables: list[tuple[ContextVariable, ContextVariableValue]] = []
        for context_variable_id in params.context_variables:
            context_variable = await context_variable_store.read_variable(context_variable_id)
            context_variable_values = await context_variable_store.list_values(context_variable_id)
            for _, context_variable_value in context_variable_values:
                context_variables.append((context_variable, context_variable_value))

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

        terms: list[Term] = []
        for term_id in params.terms:
            if glossary_store:
                term = await glossary_store.read_term(term_id)
                terms.append(term)

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

        guidelines: list[Guideline] = []
        for guideline_id in params.guidelines:
            guideline = await guideline_store.read_guideline(guideline_id)
            guidelines.append(guideline)

        guideline_matching_result = await guideline_matcher.match_guidelines(
            agent=agent,
            customer=customer,
            context_variables=context_variables,
            interaction_history=events,
            terms=terms,
            staged_events=staged_events,
            guidelines=guidelines,
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
            for gen in guideline_matching_result.batch_generations
        ]

        batches = [
            [
                GuidelineMatchDTO(
                    guideline=match.guideline.id,
                    score=match.score,
                    rationale=match.rationale,
                    guideline_previously_applied=match.guideline_previously_applied,
                    guideline_is_continuous=match.guideline_is_continuous,
                    should_reapply=match.should_reapply,
                )
                for match in batch
            ]
            for batch in guideline_matching_result.batches
        ]

        return GuidelineMatchingResultDTO(
            total_duration=guideline_matching_result.total_duration,
            batch_count=guideline_matching_result.batch_count,
            batch_generations=batch_generations,
            batches=batches,
        )

    return test_router
