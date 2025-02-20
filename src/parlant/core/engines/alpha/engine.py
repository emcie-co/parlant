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

from __future__ import annotations
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from pprint import pformat
import traceback
from typing import Optional, Sequence, cast
from croniter import croniter
from typing_extensions import override

from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import Customer, CustomerStore
from parlant.core.engines.alpha.fluid_message_generator import FluidMessageGenerator
from parlant.core.engines.alpha.hooks import LifecycleHooks
from parlant.core.engines.alpha.message_assembler import MessageAssembler
from parlant.core.engines.alpha.message_event_composer import (
    MessageEventComposer,
)
from parlant.core.engines.alpha.tool_caller import ToolInsights
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
    MessageEventData,
    MessageGenerationInspection,
    PreparationIteration,
    PreparationIterationGenerations,
    Session,
    SessionStore,
    Term as StoredTerm,
    ToolEventData,
)
from parlant.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionResult,
)
from parlant.core.engines.alpha.guideline_proposition import (
    GuidelineProposition,
)
from parlant.core.engines.alpha.tool_event_generator import (
    ToolEventGenerationResult,
    ToolEventGenerator,
)
from parlant.core.engines.alpha.utils import context_variables_to_json
from parlant.core.engines.types import Context, Engine, UtteranceReason, UtteranceRequest
from parlant.core.emissions import EventEmitter, EmittedEvent
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.logging import Logger
from parlant.core.tools import ToolContext, ToolId


@dataclass(frozen=True)
class _InteractionState:
    """Helper class to access a session's interaction state"""

    @staticmethod
    def empty() -> _InteractionState:
        """Returns an empty interaction state"""
        return _InteractionState([], -1)

    history: Sequence[Event]
    """An sequenced event-by-event representation of the interaction"""

    last_known_event_offset: int
    """An accessor which is often useful when emitting status events"""


@dataclass(frozen=True)
class _LoadedContext:
    """Helper class to access loaded values that are relevant for responding in a particular context"""

    info: Context
    """The raw context which is here represented in its loaded form"""

    agent: Agent
    """The agent which is currently requested to respond"""

    customer: Customer
    """The customer to which the agent is responding"""

    session: Session
    """The session being processed"""

    event_emitter: EventEmitter
    """Emits new events into the loaded session"""

    interaction: _InteractionState
    """A snapshot of the interaction history in the loaded session"""


@dataclass(frozen=False)
class _ResponsePreparationState:
    """Helper class to access and update the state needed for responding properly"""

    context_variables: list[tuple[ContextVariable, ContextVariableValue]]
    glossary_terms: set[Term]
    ordinary_guideline_propositions: list[GuidelineProposition]
    tool_enabled_guideline_propositions: dict[GuidelineProposition, list[ToolId]]
    tool_events: list[EmittedEvent]
    tool_insights: ToolInsights
    iterations_completed: int
    prepared_to_respond: bool
    message_events: list[EmittedEvent]

    @property
    def ordinary_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.ordinary_guideline_propositions]

    @property
    def tool_enabled_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.tool_enabled_guideline_propositions.keys()]

    @property
    def guidelines(self) -> list[Guideline]:
        return self.ordinary_guidelines + self.tool_enabled_guidelines

    @property
    def all_events(self) -> list[EmittedEvent]:
        return self.tool_events + self.message_events


class AlphaEngine(Engine):
    """The main AI processing engine (as of Feb 25, the latest and greatest processing engine)"""

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
        lifecycle_hooks: LifecycleHooks,
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

        self._lifecycle_hooks = lifecycle_hooks

    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        """Processes a context and emits new events as needed"""

        # Load the full relevant information from storage.
        loaded_context = await self._load_context(context, event_emitter)

        try:
            with self._logger.operation(f"Processing context for session {context.session_id}"):
                await self._do_process(loaded_context, event_emitter)
            return True
        except asyncio.CancelledError:
            return False
        except Exception as exc:
            formatted_exception = pformat(traceback.format_exception(exc))

            self._logger.error(f"Processing error: {formatted_exception}")

            if await self._lifecycle_hooks.call_on_error(context, event_emitter, exc):
                await self._emit_error_event(loaded_context, formatted_exception)

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
        """Produces a new message into a session, guided by specific utterance requests"""

        # Load the full relevant information from storage.
        loaded_context = await self._load_context(
            context,
            event_emitter,
            # Results seem to be more consistent with the requests
            # if we ignore the interaction's content.
            load_interaction=False,
        )

        try:
            with self._logger.operation(f"Uttering in session {context.session_id}"):
                await self._do_utter(loaded_context, requests)
            return True
        except asyncio.CancelledError:
            self._logger.warning(f"Uttering in session {context.session_id} was cancelled.")
            return False
        except Exception as exc:
            formatted_exception = pformat(traceback.format_exception(exc))

            self._logger.error(
                f"Error during uttering in session {context.session_id}: {formatted_exception}"
            )

            if await self._lifecycle_hooks.call_on_error(context, event_emitter, exc):
                await self._emit_error_event(loaded_context, formatted_exception)

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
        context: _LoadedContext,
        event_emitter: EventEmitter,
    ) -> None:
        if not await self._lifecycle_hooks.call_on_acknowledging(context.info, event_emitter):
            return  # Hook requested to bail out

        # Mark that this latest session state has been seen by the agent.
        await self._emit_acknowledgement_event(context)

        if not await self._lifecycle_hooks.call_on_acknowledged(context.info, event_emitter):
            return  # Hook requested to bail out

        try:
            if not await self._lifecycle_hooks.call_on_preparing(context.info, event_emitter):
                return  # Hook requested to bail out

            preparation_state = await self._initialize_preparation_state(context)
            preparation_iteration_inspections = []

            # Mark that the agent is in the process of preparing for a response.
            await self._emit_processing_event(context)

            while not preparation_state.prepared_to_respond:
                # Need more data before we're ready to respond

                if not await self._lifecycle_hooks.call_on_preparation_iteration_start(
                    context.info, event_emitter, preparation_state.tool_events
                ):
                    break  # Hook requested to finish preparing

                # Get more data (guidelines, tools, etc.,)
                # This happens in iterations in order to support a feedback loop
                # where particular tool-call results may trigger new or different
                # guidelines that we need to follow.
                iteration_inspection = await self._run_preparation_iteration(
                    context,
                    preparation_state,
                )

                # Save results for later inspection.
                preparation_iteration_inspections.append(iteration_inspection)

                # Some tools may update session mode (e.g. from automatic to manual).
                # This is particularly important to support human handoff.
                await self._update_session_mode(context, preparation_state)

                if not await self._lifecycle_hooks.call_on_preparation_iteration_end(
                    context.info,
                    event_emitter,
                    preparation_state.tool_events,
                    preparation_state.guidelines,
                ):
                    break

            if not await self._lifecycle_hooks.call_on_generating_messages(
                context.info,
                event_emitter,
                preparation_state.tool_events,
                preparation_state.guidelines,
            ):
                return

            # Money time: communicate with the customer given
            # all of the information we have prepared.
            message_generation_inspections = await self._generate_messages(
                context,
                preparation_state,
            )

            # Save results for later inspection.
            await self._session_store.create_inspection(
                session_id=context.session.id,
                correlation_id=self._correlator.correlation_id,
                preparation_iterations=preparation_iteration_inspections,
                message_generations=message_generation_inspections,
            )

            await self._lifecycle_hooks.call_on_generated_messages(
                context.info,
                event_emitter,
                preparation_state.all_events,
                preparation_state.guidelines,
            )

        except asyncio.CancelledError:
            # Task was cancelled. This usually happens for 1 of 2 reasons:
            #   1. The server is shutting down
            #   2. New information arrived and the currently loaded
            #      processing context is likely to be obsolete
            self._logger.warning("Processing cancelled")
            await self._emit_cancellation_event(context)
            raise
        finally:
            # Mark that the agent is ready to receive and respond to new events.
            await self._emit_ready_event(context)

    async def _do_utter(
        self,
        context: _LoadedContext,
        requests: Sequence[UtteranceRequest],
    ) -> None:
        try:
            preparation_state = await self._initialize_preparation_state(context)

            # Only use the specified utterance requests as guidelines here.
            preparation_state.ordinary_guideline_propositions.extend(
                # Utterance requests are reduced to guidelines, to take advantage
                # of the engine's ability to consistently adhere to guidelines.
                await self._utterance_requests_to_guideline_propositions(requests)
            )

            # Money time: communicate with the customer given the
            # specified utterance requests.
            message_generation_inspections = await self._generate_messages(
                context, preparation_state
            )

            # Save results for later inspection.
            await self._session_store.create_inspection(
                session_id=context.session.id,
                correlation_id=self._correlator.correlation_id,
                preparation_iterations=[],
                message_generations=message_generation_inspections,
            )

        except asyncio.CancelledError:
            self._logger.warning("Uttering cancelled")
            raise
        finally:
            # Mark that the agent is ready to receive and respond to new events.
            await self._emit_ready_event(context)

    async def _load_context(
        self,
        context: Context,
        event_emitter: EventEmitter,
        load_interaction: bool = True,
    ) -> _LoadedContext:
        # Load the full entities from storage.

        agent = await self._agent_store.read_agent(context.agent_id)
        session = await self._session_store.read_session(context.session_id)
        customer = await self._customer_store.read_customer(session.customer_id)

        if load_interaction:
            interaction = await self._load_interaction_state(context)
        else:
            interaction = _InteractionState([], -1)

        return _LoadedContext(
            info=context,
            agent=agent,
            customer=customer,
            session=session,
            event_emitter=event_emitter,
            interaction=interaction,
        )

    async def _initialize_preparation_state(
        self,
        context: _LoadedContext,
    ) -> _ResponsePreparationState:
        state = _ResponsePreparationState(
            context_variables=[],
            glossary_terms=set(),
            ordinary_guideline_propositions=[],
            tool_enabled_guideline_propositions={},
            tool_events=[],
            tool_insights=ToolInsights(),
            iterations_completed=0,
            prepared_to_respond=False,
            message_events=[],
        )

        # Load the relevant context variable values.
        state.context_variables = await self._load_context_variables(context)

        # Load relevant glossary terms, initially based
        # mostly on the current interaction history.
        state.glossary_terms.update(await self._load_glossary_terms(context, state))

        return state

    async def _run_preparation_iteration(
        self,
        context: _LoadedContext,
        state: _ResponsePreparationState,
    ) -> PreparationIteration:
        # Match relevant guidelines, retrieving them in a
        # structured format such that we can distinguish
        # between ordinary and tool-enabled ones.
        (
            guideline_proposition_result,
            state.ordinary_guideline_propositions,
            state.tool_enabled_guideline_propositions,
        ) = await self._load_guideline_propositions(context, state)

        # Matched guidelines may use glossasry terms, so we need to ground our
        # response by reevaluating the relevant terms given these new guidelines.
        state.glossary_terms.update(await self._load_glossary_terms(context, state))

        # Infer any needed tool calls and execute them,
        # adding the resulting tool events to the session.
        (
            tool_event_generation_result,
            new_tool_events,
            tool_insights,
        ) = await self._call_tools(context, state)

        state.tool_events += new_tool_events
        state.tool_insights = tool_insights

        # Tool calls may have returned with data that uses glossary terms,
        # so we need to ground our response again by reevaluating terms.
        state.glossary_terms.update(await self._load_glossary_terms(context, state))

        # Mark that another iteration has been completed
        # (this is important to avoid running more than K max iterations)
        state.iterations_completed += 1

        # If there's no new information to consider (which would have come from
        # the tools), then we can consider ourselves prepared to respond.
        if not new_tool_events:
            state.prepared_to_respond = True
        # Alternatively, we we've reached the max number of iterations,
        # we should just go ahead and respond anyway, despite possibly
        # needing more data for a fully accurate response.
        #
        # This is a trade-off that can be controlled by adjusting the max.
        elif state.iterations_completed == context.agent.max_engine_iterations:
            self._logger.warning(
                f"Reached max tool call iterations ({context.agent.max_engine_iterations})"
            )
            state.prepared_to_respond = True

        # Return structured inspection information, useful for later troubleshooting.
        return PreparationIteration(
            guideline_propositions=[
                StoredGuidelineProposition(
                    guideline_id=proposition.guideline.id,
                    condition=proposition.guideline.content.condition,
                    action=proposition.guideline.content.action,
                    score=proposition.score,
                    rationale=proposition.rationale,
                )
                for proposition in chain(
                    state.ordinary_guideline_propositions,
                    state.tool_enabled_guideline_propositions.keys(),
                )
            ],
            tool_calls=[
                tool_call
                for tool_event in new_tool_events
                for tool_call in cast(ToolEventData, tool_event.data)["tool_calls"]
            ],
            terms=[
                StoredTerm(
                    id=term.id,
                    name=term.name,
                    description=term.description,
                    synonyms=term.synonyms,
                )
                for term in state.glossary_terms
            ],
            context_variables=[
                StoredContextVariable(
                    id=variable.id,
                    name=variable.name,
                    description=variable.description,
                    key=context.session.customer_id,
                    value=value.data,
                )
                for variable, value in state.context_variables
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

    async def _update_session_mode(
        self,
        context: _LoadedContext,
        state: _ResponsePreparationState,
    ) -> None:
        # Do we even have control-requests coming from any called tools?
        if tool_call_control_outputs := [
            tool_call["result"]["control"]
            for tool_event in state.tool_events
            for tool_call in cast(ToolEventData, tool_event.data)["tool_calls"]
        ]:
            # Yes we do. Update session mode as needed.

            current_session_mode = context.session.mode
            new_session_mode = current_session_mode

            for control_output in tool_call_control_outputs:
                new_session_mode = control_output.get("mode") or current_session_mode

            if new_session_mode != current_session_mode:
                self._logger.info(
                    f"Changing session {context.session.id} mode to '{new_session_mode}'"
                )

                await self._session_store.update_session(
                    session_id=context.session.id,
                    params={
                        "mode": new_session_mode,
                    },
                )

    async def _generate_messages(
        self,
        context: _LoadedContext,
        state: _ResponsePreparationState,
    ) -> Sequence[MessageGenerationInspection]:
        message_generation_inspections = []

        for event_generation_result in await self._get_message_composer(
            context.agent
        ).generate_events(
            event_emitter=context.event_emitter,
            agent=context.agent,
            customer=context.customer,
            context_variables=state.context_variables,
            interaction_history=context.interaction.history,
            terms=list(state.glossary_terms),
            ordinary_guideline_propositions=state.ordinary_guideline_propositions,
            tool_enabled_guideline_propositions=state.tool_enabled_guideline_propositions,
            tool_insights=state.tool_insights,
            staged_events=state.tool_events,
        ):
            state.message_events += [e for e in event_generation_result.events if e]

            message_generation_inspections.append(
                MessageGenerationInspection(
                    generation=event_generation_result.generation_info,
                    messages=[
                        cast(MessageEventData, e.data)
                        if e and e.kind == "message" and isinstance(e.data, dict)
                        else None
                        for e in event_generation_result.events
                    ],
                )
            )

        return message_generation_inspections

    async def _emit_error_event(self, context: _LoadedContext, exception_details: str) -> None:
        await context.event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "status": "error",
                "acknowledged_offset": context.interaction.last_known_event_offset,
                "data": {"exception": exception_details},
            },
        )

    async def _emit_acknowledgement_event(self, context: _LoadedContext) -> None:
        await context.event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "acknowledged_offset": context.interaction.last_known_event_offset,
                "status": "acknowledged",
                "data": {},
            },
        )

    async def _emit_processing_event(self, context: _LoadedContext) -> None:
        await context.event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "acknowledged_offset": context.interaction.last_known_event_offset,
                "status": "processing",
                "data": {},
            },
        )

    async def _emit_cancellation_event(self, context: _LoadedContext) -> None:
        await context.event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "acknowledged_offset": context.interaction.last_known_event_offset,
                "status": "cancelled",
                "data": {},
            },
        )

    async def _emit_ready_event(self, context: _LoadedContext) -> None:
        await context.event_emitter.emit_status_event(
            correlation_id=self._correlator.correlation_id,
            data={
                "acknowledged_offset": context.interaction.last_known_event_offset,
                "status": "ready",
                "data": {},
            },
        )

    def _get_message_composer(self, agent: Agent) -> MessageEventComposer:
        # Each agent may use a different composition mode,
        # and, moreover, the same agent can change composition
        # modes every now and then. This makes sure that we are
        # composing the message using the right mechanism for this agent.
        match agent.composition_mode:
            case "fluid":
                return self._fluid_message_generator
            case "strict_assembly" | "composited_assembly" | "fluid_assembly":
                return self._message_assembler

        raise Exception("Unsupported agent composition mode")

    async def _load_context_variables(
        self,
        context: _LoadedContext,
    ) -> list[tuple[ContextVariable, ContextVariableValue]]:
        variables_supported_by_agent = await self._context_variable_store.list_variables(
            variable_set=context.agent.id,
        )

        result = []

        keys_to_check_in_order_of_importance = (
            [context.customer.id]  # Customer-specific value
            + [f"tag:{tag_id}" for tag_id in context.customer.tags]  # Tag-specific value
            + [ContextVariableStore.GLOBAL_KEY]  # Global value
        )

        for variable in variables_supported_by_agent:
            # Try keys in order of importance, stopping at and using
            # the first (and most important) set key for each variable.
            for key in keys_to_check_in_order_of_importance:
                if value := await self._load_context_variable_value(context, variable, key):
                    result.append((variable, value))
                    break

        return result

    async def _load_guideline_propositions(
        self,
        context: _LoadedContext,
        state: _ResponsePreparationState,
    ) -> tuple[
        GuidelinePropositionResult,
        list[GuidelineProposition],
        dict[GuidelineProposition, list[ToolId]],
    ]:
        # Step 1: Retrieve all of the installed guidelines for this agent.
        all_stored_guidelines = await self._guideline_store.list_guidelines(
            guideline_set=context.agent.id,
        )

        # Step 2: Filter the best matches out of those.
        proposition_result = await self._guideline_proposer.propose_guidelines(
            agent=context.agent,
            customer=context.customer,
            guidelines=all_stored_guidelines,
            context_variables=state.context_variables,
            interaction_history=context.interaction.history,
            terms=list(state.glossary_terms),
            staged_events=state.tool_events,
        )

        # Step 3: Load connected guidelines that may not have
        # been inferrable just by looking at the interaction.
        inferred_propositions = await self._propose_connected_guidelines(
            guideline_set=context.agent.id,
            propositions=proposition_result.propositions,
        )

        # Step 4: Put all propositions in one basket, looking at them as a whole.
        all_relevant_guidelines = [
            *proposition_result.propositions,
            *inferred_propositions,
        ]

        # Step 5: Distinguish between ordinary and tool-enabled guidelines.
        # We do this here as it creates a better subsequent control flow in the engine.
        tool_enabled_guidelines = await self._find_tool_enabled_guidelines_propositions(
            guideline_propositions=all_relevant_guidelines,
        )
        ordinary_guidelines = list(
            set(all_relevant_guidelines).difference(tool_enabled_guidelines),
        )

        return proposition_result, ordinary_guidelines, tool_enabled_guidelines

    async def _propose_connected_guidelines(
        self,
        guideline_set: str,
        propositions: Sequence[GuidelineProposition],
    ) -> Sequence[GuidelineProposition]:
        # Some guidelines cannot be inferred simply by evaluating an interaction.
        #
        # For example, if we matched a guideline, "When X, Then Y",
        # we also need to load and account for "When Y, Then Z".
        # Such connections are pre-indexed in a graph behind the scenes,
        # and those are the ones we are loading here.

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
    ) -> dict[GuidelineProposition, list[ToolId]]:
        # Create a convenient accessor dict for tool-enabled guidelines (and their tools).
        # This allows for optimized control and data flow in the engine.

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

    async def _load_glossary_terms(
        self,
        context: _LoadedContext,
        state: _ResponsePreparationState,
    ) -> Sequence[Term]:
        # Glossary terms are retrieved using semantic similarity.
        # The querying process is done with a text query, for which
        # the K most relevant terms are retrieved.
        #
        # We thus build an optimized query here based on our context and state.
        query = ""

        if state.context_variables:
            query += f"\n{context_variables_to_json(state.context_variables)}"

        if context.interaction.history:
            query += str([e.data for e in context.interaction.history])

        if state.guidelines:
            query += str(
                [f"When {g.content.condition}, then {g.content.action}" for g in state.guidelines]
            )

        if state.tool_events:
            query += str([e.data for e in state.tool_events])

        if query:
            return await self._glossary_store.find_relevant_terms(
                term_set=context.agent.id,
                query=query,
            )

        return []

    async def _call_tools(
        self, context: _LoadedContext, state: _ResponsePreparationState
    ) -> tuple[ToolEventGenerationResult, list[EmittedEvent], ToolInsights]:
        result = await self._tool_event_generator.generate_events(
            event_emitter=context.event_emitter,
            session_id=context.session.id,
            agent=context.agent,
            customer=context.customer,
            context_variables=state.context_variables,
            interaction_history=context.interaction.history,
            terms=list(state.glossary_terms),
            ordinary_guideline_propositions=state.ordinary_guideline_propositions,
            tool_enabled_guideline_propositions=state.tool_enabled_guideline_propositions,
            staged_events=state.tool_events,
        )

        tool_events = [e for e in result.events if e] if result else []

        return result, tool_events, result.insights

    async def _utterance_requests_to_guideline_propositions(
        self,
        requests: Sequence[UtteranceRequest],
    ) -> Sequence[GuidelineProposition]:
        # Utterance requests are reduced to guidelines, to take advantage
        # of the engine's ability to consistently adhere to guidelines.

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

    async def _load_context_variable_value(
        self,
        context: _LoadedContext,
        variable: ContextVariable,
        key: str,
    ) -> Optional[ContextVariableValue]:
        return await load_fresh_context_variable_value(
            context_variable_store=self._context_variable_store,
            service_registery=self._service_registry,
            agent_id=context.agent.id,
            session=context.session,
            variable=variable,
            key=key,
        )


# This is module-level and public for isolated testability purposes.
async def load_fresh_context_variable_value(
    context_variable_store: ContextVariableStore,
    service_registery: ServiceRegistry,
    agent_id: AgentId,
    session: Session,
    variable: ContextVariable,
    key: str,
    current_time: datetime = datetime.now(timezone.utc),
) -> Optional[ContextVariableValue]:
    # Load the existing value
    value = await context_variable_store.read_value(
        variable_set=agent_id,
        variable_id=variable.id,
        key=key,
    )

    # If there's no tool attached to this variable,
    # return the value we found for the key.
    # Note that this may be None here, which is okay.
    if not variable.tool_id:
        return value

    # So we do have a tool attached.
    # Do we already have a value, and is it sufficiently fresh?
    if value and variable.freshness_rules:
        cron_iterator = croniter(variable.freshness_rules, value.last_modified)

        if cron_iterator.get_next(datetime) > current_time:
            # We already have a fresh value in store. Return it.
            return value

    # We don't have a sufficiently fresh value.
    # Get an updated one, utilizing the associated tool.

    tool_context = ToolContext(
        agent_id=agent_id,
        session_id=session.id,
        customer_id=session.customer_id,
    )

    tool_service = await service_registery.read_tool_service(variable.tool_id.service_name)

    tool_result = await tool_service.call_tool(
        variable.tool_id.tool_name,
        context=tool_context,
        arguments={},
    )

    return await context_variable_store.update_value(
        variable_set=agent_id,
        variable_id=variable.id,
        key=key,
        data=tool_result.data,
    )
