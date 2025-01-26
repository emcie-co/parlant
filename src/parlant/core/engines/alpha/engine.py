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

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
import traceback
from typing import Mapping, Optional, Sequence, cast
from croniter import croniter
from typing_extensions import override

from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import Customer, CustomerStore
from parlant.core.engines.alpha.fluid_message_generator import FluidMessageGenerator
from parlant.core.engines.alpha.message_assembler import MessageAssembler
from parlant.core.engines.alpha.message_event_composer import MessageEventComposer
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineContent, GuidelineStore
from parlant.core.guideline_connections import GuidelineConnectionStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationStore,
)
from parlant.core.glossary import Term, GlossaryStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import (
    ContextVariable as StoredContextVariable,
    Event,
    GuidelineProposition as StoredGuidelineProposition,
    GuidelinePropositionInspection,
    MessageGenerationInspection,
    PreparationIteration,
    PreparationIterationGenerations,
    Session,
    SessionStore,
    Term as StoredTerm,
    ToolEventData,
)
from parlant.core.engines.alpha.hooks import lifecycle_hooks
from parlant.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionResult,
)
from parlant.core.engines.alpha.guideline_proposition import (
    GuidelineProposition,
)
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.alpha.utils import context_variables_to_json
from parlant.core.engines.types import Context, Engine, UtteranceReason, UtteranceRequest
from parlant.core.emissions import EventEmitter, EmittedEvent
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.logging import Logger
from parlant.core.tools import ToolContext, ToolId


@dataclass(frozen=True)
class _InteractionState:
    history: Sequence[Event]
    last_known_event_offset: int


class AlphaEngine(Engine):
    def __init__(
        self,
        logger: Logger,
        correlator: ContextualCorrelator,
        agent_store: AgentStore,
        session_store: SessionStore,
        customer_store: CustomerStore,
        context_variable_store: ContextVariableStore,
        glossary_store: GlossaryStore,
        guideline_store: GuidelineStore,
        guideline_connection_store: GuidelineConnectionStore,
        service_registry: ServiceRegistry,
        guideline_tool_association_store: GuidelineToolAssociationStore,
        guideline_proposer: GuidelineProposer,
        tool_event_generator: ToolEventGenerator,
        fluid_message_generator: FluidMessageGenerator,
        message_assembler: MessageAssembler,
    ) -> None:
        self._logger = logger
        self._correlator = correlator

        self._agent_store = agent_store
        self._session_store = session_store
        self._customer_store = customer_store
        self._context_variable_store = context_variable_store
        self._glossary_store = glossary_store
        self._guideline_store = guideline_store
        self._guideline_connection_store = guideline_connection_store
        self._service_registry = service_registry
        self._guideline_tool_association_store = guideline_tool_association_store

        self._guideline_proposer = guideline_proposer
        self._tool_event_generator = tool_event_generator
        self._fluid_message_generator = fluid_message_generator
        self._message_assembler = message_assembler

    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        interaction_state = await self._load_interaction_state(context)

        try:
            with self._logger.operation(f"Processing context for session {context.session_id}"):
                await self._do_process(context, interaction_state, event_emitter)
            return True
        except asyncio.CancelledError:
            return False
        except Exception as exc:
            formatted_exception = traceback.format_exception(exc)

            self._logger.error(f"Processing error: {formatted_exception}")

            if await lifecycle_hooks.call_on_error(context, event_emitter, exc):
                await event_emitter.emit_status_event(
                    correlation_id=self._correlator.correlation_id,
                    data={
                        "status": "error",
                        "acknowledged_offset": interaction_state.last_known_event_offset,
                        "data": {"exception": formatted_exception},
                    },
                )

            return False
        except BaseException as exc:
            self._logger.critical(f"Critical processing error: {traceback.format_exception(exc)}")
            raise

    @override
    async def utter(
        self,
        context: Context,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        interaction_state = await self._load_interaction_state(context)
        try:
            with self._logger.operation(
                f"Uttering in session {context.session_id} using actions '{[r.action for r in requests]}'"
            ):
                await self._do_utter(context, interaction_state, event_emitter, requests)
            return True

        except asyncio.CancelledError:
            self._logger.warning(f"Uttering in session {context.session_id} was cancelled.")
            return False

        except Exception as exc:
            formatted_exception = traceback.format_exception(type(exc), exc, exc.__traceback__)
            self._logger.error(
                f"Error during uttering in session {context.session_id}: {formatted_exception}"
            )

            await event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "status": "error",
                    "acknowledged_offset": interaction_state.last_known_event_offset,
                    "data": {"exception": formatted_exception},
                },
            )
            return False

        except BaseException as exc:
            self._logger.critical(
                f"Critical error during uttering in session {context.session_id}: "
                f"{traceback.format_exception(type(exc), exc, exc.__traceback__)}"
            )
            raise

    async def _load_interaction_state(self, context: Context) -> _InteractionState:
        history = list(await self._session_store.list_events(context.session_id))
        last_known_event_offset = history[-1].offset if history else -1

        return _InteractionState(
            history=history,
            last_known_event_offset=last_known_event_offset,
        )

    async def _do_process(
        self,
        context: Context,
        interaction: _InteractionState,
        event_emitter: EventEmitter,
    ) -> None:
        agent = await self._agent_store.read_agent(context.agent_id)
        session = await self._session_store.read_session(context.session_id)
        customer = await self._customer_store.read_customer(session.customer_id)

        if not await lifecycle_hooks.call_on_acknowledging(context, event_emitter):
            return

        await event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "acknowledged_offset": interaction.last_known_event_offset,
                "status": "acknowledged",
                "data": {},
            },
        )

        if not await lifecycle_hooks.call_on_acknowledged(context, event_emitter):
            return

        try:
            if not await lifecycle_hooks.call_on_preparing(context, event_emitter):
                return

            context_variables = await self._load_context_variables(
                agent_id=context.agent_id, session=session, customer=customer
            )

            terms = set(
                await self._load_relevant_terms(
                    agents=[agent],
                    context_variables=context_variables,
                    interaction_history=interaction.history,
                )
            )

            await event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "acknowledged_offset": interaction.last_known_event_offset,
                    "status": "processing",
                    "data": {},
                },
            )

            all_tool_events: list[EmittedEvent] = []
            preparation_iterations: list[PreparationIteration] = []
            prepared_to_respond = False

            while not prepared_to_respond:
                if not await lifecycle_hooks.call_on_preparation_iteration_start(
                    context, event_emitter, all_tool_events
                ):
                    break

                all_possible_guidelines = await self._guideline_store.list_guidelines(
                    guideline_set=agent.id,
                )

                guideline_proposition_result = await self._guideline_proposer.propose_guidelines(
                    agents=[agent],
                    customer=customer,
                    guidelines=list(all_possible_guidelines),
                    context_variables=context_variables,
                    interaction_history=interaction.history,
                    terms=list(terms),
                    staged_events=all_tool_events,
                )

                (
                    ordinary_guideline_propositions,
                    tool_enabled_guideline_propositions,
                ) = await self._load_guideline_propositions(
                    agents=[agent],
                    guideline_proposition_result=guideline_proposition_result,
                )

                terms.update(
                    await self._load_relevant_terms(
                        agents=[agent],
                        propositions=list(
                            chain(
                                ordinary_guideline_propositions,
                                tool_enabled_guideline_propositions.keys(),
                            ),
                        ),
                    )
                )

                tool_event_generation_result = await self._tool_event_generator.generate_events(
                    event_emitter=event_emitter,
                    session_id=context.session_id,
                    agents=[agent],
                    customer=customer,
                    context_variables=context_variables,
                    interaction_history=interaction.history,
                    terms=list(terms),
                    ordinary_guideline_propositions=ordinary_guideline_propositions,
                    tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
                    staged_events=all_tool_events,
                )

                tool_events = (
                    [e for e in tool_event_generation_result.events if e]
                    if tool_event_generation_result
                    else []
                )

                all_tool_events += tool_events

                terms.update(
                    set(
                        await self._load_relevant_terms(
                            agents=[agent],
                            staged_events=tool_events,
                        )
                    )
                )

                preparation_iterations.append(
                    PreparationIteration(
                        guideline_propositions=[
                            StoredGuidelineProposition(
                                guideline_id=proposition.guideline.id,
                                condition=proposition.guideline.content.condition,
                                action=proposition.guideline.content.action,
                                score=proposition.score,
                                rationale=proposition.rationale,
                            )
                            for proposition in chain(
                                ordinary_guideline_propositions,
                                tool_enabled_guideline_propositions.keys(),
                            )
                        ],
                        tool_calls=[
                            tool_call
                            for tool_event in tool_events
                            for tool_call in cast(ToolEventData, tool_event.data)["tool_calls"]
                        ],
                        terms=[
                            StoredTerm(
                                id=term.id,
                                name=term.name,
                                description=term.description,
                                synonyms=term.synonyms,
                            )
                            for term in terms
                        ],
                        context_variables=[
                            StoredContextVariable(
                                id=variable.id,
                                name=variable.name,
                                description=variable.description,
                                key=session.customer_id,
                                value=value.data,
                            )
                            for variable, value in context_variables
                        ],
                        generations=PreparationIterationGenerations(
                            guideline_proposition=GuidelinePropositionInspection(
                                total_duration=guideline_proposition_result.total_duration,
                                batches=guideline_proposition_result.batch_generations,
                            ),
                            tool_calls=tool_event_generation_result.generations
                            if tool_event_generation_result
                            else [],
                        ),
                    )
                )

                if not tool_events:
                    prepared_to_respond = True

                if len(preparation_iterations) == agent.max_engine_iterations:
                    self._logger.warning(
                        f"Reached max tool call iterations ({agent.max_engine_iterations})"
                    )
                    prepared_to_respond = True

                if tool_call_control_outputs := [
                    tool_call["result"]["control"]
                    for tool_event in all_tool_events
                    for tool_call in cast(ToolEventData, tool_event.data)["tool_calls"]
                ]:
                    current_session_mode = session.mode
                    new_session_mode = current_session_mode

                    for control_output in tool_call_control_outputs:
                        new_session_mode = control_output.get("mode") or current_session_mode

                    if new_session_mode != current_session_mode:
                        self._logger.info(
                            f"Changing session {session.id} mode to '{new_session_mode}'"
                        )

                        await self._session_store.update_session(
                            session_id=session.id,
                            params={
                                "mode": new_session_mode,
                            },
                        )

                if not await lifecycle_hooks.call_on_preparation_iteration_end(
                    context,
                    event_emitter,
                    all_tool_events,
                    [gp.guideline for gp in ordinary_guideline_propositions]
                    + [gp.guideline for gp in tool_enabled_guideline_propositions.keys()],
                ):
                    break

            message_generation_inspections = []

            if not await lifecycle_hooks.call_on_generating_messages(
                context,
                event_emitter,
                all_tool_events,
                [gp.guideline for gp in ordinary_guideline_propositions]
                + [gp.guideline for gp in tool_enabled_guideline_propositions.keys()],
            ):
                return

            all_emitted_events = [*all_tool_events]

            for event_generation_result in await self._get_message_composer(agent).generate_events(
                event_emitter=event_emitter,
                agents=[agent],
                customer=customer,
                context_variables=context_variables,
                interaction_history=interaction.history,
                terms=list(terms),
                ordinary_guideline_propositions=ordinary_guideline_propositions,
                tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
                staged_events=all_tool_events,
            ):
                message_generation_inspections.append(
                    MessageGenerationInspection(
                        generation=event_generation_result.generation_info,
                        messages=[
                            e.data["message"]
                            if e and e.kind == "message" and isinstance(e.data, dict)
                            else None
                            for e in event_generation_result.events
                        ],
                    )
                )

                all_emitted_events += [e for e in event_generation_result.events if e]

            await self._session_store.create_inspection(
                session_id=context.session_id,
                correlation_id=self._correlator.correlation_id,
                preparation_iterations=preparation_iterations,
                message_generations=message_generation_inspections,
            )

            await lifecycle_hooks.call_on_generated_messages(
                context,
                event_emitter,
                all_emitted_events,
                [gp.guideline for gp in ordinary_guideline_propositions]
                + [gp.guideline for gp in tool_enabled_guideline_propositions.keys()],
            )

        except asyncio.CancelledError:
            await event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "acknowledged_offset": interaction.last_known_event_offset,
                    "status": "cancelled",
                    "data": {},
                },
            )

            self._logger.warning("Processing cancelled")

            raise
        finally:
            await event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "acknowledged_offset": interaction.last_known_event_offset,
                    "status": "ready",
                    "data": {},
                },
            )

    async def _do_utter(
        self,
        context: Context,
        interaction: _InteractionState,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> None:
        agent = await self._agent_store.read_agent(context.agent_id)
        session = await self._session_store.read_session(context.session_id)
        customer = await self._customer_store.read_customer(session.customer_id)

        await event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "acknowledged_offset": interaction.last_known_event_offset,
                "status": "acknowledged",
                "data": {},
            },
        )

        try:
            context_variables = await self._load_context_variables(
                agent_id=context.agent_id, session=session, customer=customer
            )

            terms = set(
                await self._load_relevant_terms(
                    agents=[agent],
                    context_variables=context_variables,
                )
            )

            ordinary_guideline_propositions = await self._utter_requests_to_guideline_propositions(
                requests
            )

            await event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "acknowledged_offset": interaction.last_known_event_offset,
                    "status": "typing",
                    "data": {},
                },
            )

            message_generation_inspections = []

            for event_generation_result in await self._get_message_composer(agent).generate_events(
                event_emitter=event_emitter,
                agents=[agent],
                customer=customer,
                context_variables=context_variables,
                interaction_history=[],
                terms=list(terms),
                ordinary_guideline_propositions=ordinary_guideline_propositions,
                tool_enabled_guideline_propositions={},
                staged_events=[],
            ):
                message_generation_inspections.append(
                    MessageGenerationInspection(
                        generation=event_generation_result.generation_info,
                        messages=[
                            e.data["message"]
                            if e and e.kind == "message" and isinstance(e.data, dict)
                            else None
                            for e in event_generation_result.events
                        ],
                    )
                )

            await self._session_store.create_inspection(
                session_id=context.session_id,
                correlation_id=self._correlator.correlation_id,
                preparation_iterations=[],
                message_generations=message_generation_inspections,
            )

        except asyncio.CancelledError:
            self._logger.warning("Uttering cancelled")

            await event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "acknowledged_offset": interaction.last_known_event_offset,
                    "status": "cancelled",
                    "data": {},
                },
            )

            raise
        finally:
            await event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "acknowledged_offset": interaction.last_known_event_offset,
                    "status": "ready",
                    "data": {},
                },
            )

    def _get_message_composer(self, agent: Agent) -> MessageEventComposer:
        match agent.composition_mode:
            case "fluid":
                return self._fluid_message_generator
            case "strict-assembly" | "composited-assembly" | "fluid-assembly":
                return self._message_assembler

    async def _load_context_variables(
        self,
        agent_id: AgentId,
        session: Session,
        customer: Customer,
    ) -> Sequence[tuple[ContextVariable, ContextVariableValue]]:
        agent_variables = await self._context_variable_store.list_variables(
            variable_set=agent_id,
        )

        context_variables = []
        keys = (
            [session.customer_id]
            + [f"tag:{tag_id}" for tag_id in customer.tags]
            + [ContextVariableStore.GLOBAL_KEY]
        )

        for variable in agent_variables:
            value = None
            for key in keys:
                existing_value = await self._context_variable_store.read_value(
                    variable_set=agent_id,
                    key=key,
                    variable_id=variable.id,
                )
                if existing_value:
                    value = await fresh_value(
                        context_variable_store=self._context_variable_store,
                        service_registery=self._service_registry,
                        agent_id=agent_id,
                        session=session,
                        variable_id=variable.id,
                        key=key,
                        current_time=datetime.now(),
                    )
                    break

            if not value:
                generated_value = await fresh_value(
                    context_variable_store=self._context_variable_store,
                    service_registery=self._service_registry,
                    agent_id=agent_id,
                    session=session,
                    variable_id=variable.id,
                    key=key,
                    current_time=datetime.now(),
                )
                if generated_value:
                    value = generated_value

            if value is not None:
                context_variables.append((variable, value))

        return context_variables

    async def _load_guideline_propositions(
        self,
        agents: Sequence[Agent],
        guideline_proposition_result: GuidelinePropositionResult,
    ) -> tuple[
        Sequence[GuidelineProposition],
        Mapping[GuidelineProposition, Sequence[ToolId]],
    ]:
        inferred_propositions = await self._propose_connected_guidelines(
            guideline_set=agents[0].id,
            propositions=guideline_proposition_result.propositions,
        )

        all_relevant_guidelines = [
            *guideline_proposition_result.propositions,
            *inferred_propositions,
        ]

        tool_enabled_guidelines = await self._find_tool_enabled_guidelines_propositions(
            guideline_propositions=all_relevant_guidelines,
        )

        ordinary_guidelines = list(
            set(all_relevant_guidelines).difference(tool_enabled_guidelines),
        )

        return ordinary_guidelines, tool_enabled_guidelines

    async def _propose_connected_guidelines(
        self,
        guideline_set: str,
        propositions: Sequence[GuidelineProposition],
    ) -> Sequence[GuidelineProposition]:
        connected_guidelines_by_proposition = defaultdict[GuidelineProposition, list[Guideline]](
            list
        )

        for proposition in propositions:
            connected_guideline_ids = {
                c.target
                for c in await self._guideline_connection_store.list_connections(
                    indirect=True,
                    source=proposition.guideline.id,
                )
            }

            for connected_guideline_id in connected_guideline_ids:
                if any(connected_guideline_id == p.guideline.id for p in propositions):
                    # no need to add this connected one as it's already an assumed proposition
                    continue

                connected_guideline = await self._guideline_store.read_guideline(
                    guideline_set=guideline_set,
                    guideline_id=connected_guideline_id,
                )

                connected_guidelines_by_proposition[proposition].append(
                    connected_guideline,
                )

        proposition_and_inferred_guideline_guideline_pairs: list[
            tuple[GuidelineProposition, Guideline]
        ] = []

        for proposition, connected_guidelines in connected_guidelines_by_proposition.items():
            for connected_guideline in connected_guidelines:
                if existing_connections := [
                    connection
                    for connection in proposition_and_inferred_guideline_guideline_pairs
                    if connection[1] == connected_guideline
                ]:
                    assert len(existing_connections) == 1
                    existing_connection = existing_connections[0]

                    # We're basically saying, if this connected guideline is already
                    # connected to a proposition with a higher priority than the proposition
                    # at hand, then we want to keep the associated with the proposition
                    # that has the higher priority, because it will go down as the inferred
                    # priority of our connected guideline's proposition...
                    #
                    # Now try to read that out loud in one go :)
                    if existing_connection[0].score >= proposition.score:
                        continue  # Stay with existing one
                    else:
                        # This proposition's score is higher, so it's better that
                        # we associate the connected guideline with this one.
                        # we'll add it soon, but meanwhile let's remove the old one.
                        proposition_and_inferred_guideline_guideline_pairs.remove(
                            existing_connection,
                        )

                proposition_and_inferred_guideline_guideline_pairs.append(
                    (proposition, connected_guideline),
                )

        return [
            GuidelineProposition(
                guideline=connection[1],
                score=connection[0].score,
                rationale="Automatically inferred from context",
            )
            for connection in proposition_and_inferred_guideline_guideline_pairs
        ]

    async def _find_tool_enabled_guidelines_propositions(
        self,
        guideline_propositions: Sequence[GuidelineProposition],
    ) -> Mapping[GuidelineProposition, Sequence[ToolId]]:
        guideline_tool_associations = list(
            await self._guideline_tool_association_store.list_associations()
        )
        guideline_propositions_by_id = {p.guideline.id: p for p in guideline_propositions}

        relevant_associations = [
            a for a in guideline_tool_associations if a.guideline_id in guideline_propositions_by_id
        ]

        tools_for_guidelines: dict[GuidelineProposition, list[ToolId]] = defaultdict(list)

        for association in relevant_associations:
            tools_for_guidelines[guideline_propositions_by_id[association.guideline_id]].append(
                association.tool_id
            )

        return dict(tools_for_guidelines)

    async def _load_relevant_terms(
        self,
        agents: Sequence[Agent],
        context_variables: Optional[Sequence[tuple[ContextVariable, ContextVariableValue]]] = None,
        interaction_history: Optional[Sequence[Event]] = None,
        propositions: Optional[Sequence[GuidelineProposition]] = None,
        staged_events: Optional[Sequence[EmittedEvent]] = None,
    ) -> Sequence[Term]:
        assert len(agents) == 1

        agent = agents[0]

        context = ""

        if context_variables:
            context += f"\n{context_variables_to_json(context_variables=context_variables)}"

        if interaction_history:
            context += str([e.data for e in interaction_history])

        if propositions:
            context += str(
                [
                    f"When {p.guideline.content.condition}, then {p.guideline.content.action}"
                    for p in propositions
                ]
            )

        if staged_events:
            context += str([e.data for e in staged_events])

        if context:
            return await self._glossary_store.find_relevant_terms(
                term_set=agent.id,
                query=context,
            )
        return []

    async def _utter_requests_to_guideline_propositions(
        self,
        requests: Sequence[UtteranceRequest],
    ) -> Sequence[GuidelineProposition]:
        def utterance_to_proposition(i: int, utterance: UtteranceRequest) -> GuidelineProposition:
            rationales = {
                UtteranceReason.BUY_TIME: "An external module has determined that this response is necessary, and you must adhere to it.",
                UtteranceReason.FOLLOW_UP: "An external module has determined that this response is necessary, and you must adhere to it.",
            }

            conditions = {
                UtteranceReason.BUY_TIME: "-- RIGHT NOW!",
                UtteranceReason.FOLLOW_UP: "-- RIGHT NOW!",
            }

            return GuidelineProposition(
                guideline=Guideline(
                    id=GuidelineId(f"<utterance-request-{i}>"),
                    creation_utc=datetime.now(timezone.utc),
                    content=GuidelineContent(
                        condition=conditions[utterance.reason],
                        action=utterance.action,
                    ),
                ),
                rationale=rationales[utterance.reason],
                score=10,
            )

        return [utterance_to_proposition(i, request) for i, request in enumerate(requests, start=1)]


async def fresh_value(
    context_variable_store: ContextVariableStore,
    service_registery: ServiceRegistry,
    agent_id: AgentId,
    session: Session,
    variable_id: ContextVariableId,
    key: str,
    current_time: datetime,
) -> Optional[ContextVariableValue]:
    variable = await context_variable_store.read_variable(
        variable_set=agent_id,
        id=variable_id,
    )

    value = await context_variable_store.read_value(
        variable_set=agent_id,
        variable_id=variable_id,
        key=key,
    )

    if not variable.tool_id:
        return value

    if value and variable.freshness_rules:
        cron = croniter(variable.freshness_rules, value.last_modified)
        if cron.get_next(datetime) > current_time:
            return value

    tool_context = ToolContext(
        agent_id=agent_id, session_id=session.id, customer_id=session.customer_id
    )
    tool_service = await service_registery.read_tool_service(variable.tool_id.service_name)
    tool_result = await tool_service.call_tool(
        variable.tool_id.tool_name, context=tool_context, arguments={}
    )

    return await context_variable_store.update_value(
        variable_set=agent_id,
        variable_id=variable_id,
        key=key,
        data=tool_result.data,
    )
