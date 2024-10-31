import asyncio
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
import traceback
from typing import Mapping, Optional, Sequence, cast

from emcie.server.core.agents import Agent, AgentId, AgentStore
from emcie.server.core.common import generate_id
from emcie.server.core.context_variables import (
    ContextVariable,
    ContextVariableStore,
    ContextVariableValue,
)
from emcie.server.core.guidelines import Guideline, GuidelineStore
from emcie.server.core.guideline_connections import ConnectionKind, GuidelineConnectionStore
from emcie.server.core.guideline_tool_associations import (
    GuidelineToolAssociationStore,
)
from emcie.server.core.glossary import Term, GlossaryStore
from emcie.server.core.services.tools.service_registry import ServiceRegistry
from emcie.server.core.sessions import (
    ContextVariable as StoredContextVariable,
    Event,
    GuidelineProposition as StoredGuidelineProposition,
    PreparationIteration,
    SessionId,
    SessionStore,
    Term as StoredTerm,
    ToolEventData,
)
from emcie.server.core.engines.alpha.guideline_proposer import GuidelineProposer
from emcie.server.core.engines.alpha.guideline_proposition import (
    GuidelineProposition,
)
from emcie.server.core.engines.alpha.message_event_producer import MessageEventProducer
from emcie.server.core.engines.alpha.tool_event_producer import ToolEventProducer
from emcie.server.core.engines.alpha.utils import context_variables_to_json
from emcie.server.core.engines.types import Context, Engine
from emcie.server.core.emissions import EventEmitter, EmittedEvent
from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.logging import Logger
from emcie.server.core.tools import ToolId


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
        context_variable_store: ContextVariableStore,
        glossary_store: GlossaryStore,
        guideline_store: GuidelineStore,
        guideline_connection_store: GuidelineConnectionStore,
        service_registry: ServiceRegistry,
        guideline_tool_association_store: GuidelineToolAssociationStore,
        guideline_proposer: GuidelineProposer,
        tool_event_producer: ToolEventProducer,
        message_event_producer: MessageEventProducer,
    ) -> None:
        self._logger = logger
        self._correlator = correlator

        self._agent_store = agent_store
        self._session_store = session_store
        self._context_variable_store = context_variable_store
        self._glossary_store = glossary_store
        self._guideline_store = guideline_store
        self._guideline_connection_store = guideline_connection_store
        self._service_registry = service_registry
        self._guideline_tool_association_store = guideline_tool_association_store
        self._guideline_proposer = guideline_proposer
        self._tool_event_producer = tool_event_producer
        self._message_event_producer = message_event_producer

    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        interaction_state = await self._load_interaction_state(context)

        try:
            with self._correlator.correlation_scope(f"process-session({context.session_id})"):
                await self._do_process(context, interaction_state, event_emitter)
            return True
        except asyncio.CancelledError:
            return False
        except Exception as exc:
            formatted_exception = traceback.format_exception(exc)

            self._logger.error(f"Processing error: {formatted_exception}")

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
        end_user_id = (await self._session_store.read_session(context.session_id)).end_user_id

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
                agent_id=context.agent_id,
                session_id=context.session_id,
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
                (
                    ordinary_guideline_propositions,
                    tool_enabled_guideline_propositions,
                ) = await self._load_guideline_propositions(
                    agents=[agent],
                    context_variables=context_variables,
                    interaction_history=interaction.history,
                    terms=list(terms),
                    staged_events=all_tool_events,
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

                tool_events = await self._tool_event_producer.produce_events(
                    event_emitter=event_emitter,
                    session_id=context.session_id,
                    agents=[agent],
                    context_variables=context_variables,
                    interaction_history=interaction.history,
                    terms=list(terms),
                    ordinary_guideline_propositions=ordinary_guideline_propositions,
                    tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
                    staged_events=all_tool_events,
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
                                predicate=proposition.guideline.content.predicate,
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
                                key=end_user_id,
                                value=value.data,
                            )
                            for variable, value in context_variables
                        ],
                    )
                )

                if not tool_events:
                    prepared_to_respond = True

                if len(preparation_iterations) == agent.max_engine_iterations:
                    self._logger.warning(
                        f"Reached max tool call iterations ({agent.max_engine_iterations})"
                    )
                    prepared_to_respond = True

            await self._session_store.create_inspection(
                session_id=context.session_id,
                correlation_id=self._correlator.correlation_id,
                preparation_iterations=preparation_iterations,
            )

            await self._message_event_producer.produce_events(
                event_emitter=event_emitter,
                agents=[agent],
                context_variables=context_variables,
                interaction_history=interaction.history,
                terms=list(terms),
                ordinary_guideline_propositions=ordinary_guideline_propositions,
                tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
                staged_events=all_tool_events,
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

    async def _load_context_variables(
        self,
        agent_id: AgentId,
        session_id: SessionId,
    ) -> Sequence[tuple[ContextVariable, ContextVariableValue]]:
        session = await self._session_store.read_session(session_id)

        variables = await self._context_variable_store.list_variables(
            variable_set=agent_id,
        )

        return [
            (
                variable,
                await self._context_variable_store.read_value(
                    variable_set=agent_id,
                    key=session.end_user_id,  # noqa: F821
                    variable_id=variable.id,
                ),
            )
            for variable in variables
        ]

    async def _load_guideline_propositions(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
    ) -> tuple[Sequence[GuidelineProposition], Mapping[GuidelineProposition, Sequence[ToolId]]]:
        all_relevant_guidelines = await self._fetch_guideline_propositions(
            agents=agents,
            context_variables=context_variables,
            interaction_history=interaction_history,
            staged_events=staged_events,
            terms=terms,
        )

        tool_enabled_guidelines = await self._find_tool_enabled_guidelines_propositions(
            guideline_propositions=all_relevant_guidelines,
        )

        ordinary_guidelines = list(
            set(all_relevant_guidelines).difference(tool_enabled_guidelines),
        )

        return ordinary_guidelines, tool_enabled_guidelines

    async def _fetch_guideline_propositions(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[GuidelineProposition]:
        assert len(agents) == 1

        all_possible_guidelines = await self._guideline_store.list_guidelines(
            guideline_set=agents[0].id,
        )

        direct_propositions = await self._guideline_proposer.propose_guidelines(
            agents=agents,
            guidelines=list(all_possible_guidelines),
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            staged_events=staged_events,
        )

        inferred_propositions = await self._propose_connected_guidelines(
            guideline_set=agents[0].id,
            propositions=direct_propositions,
        )

        return [*direct_propositions, *inferred_propositions]

    async def _propose_connected_guidelines(
        self,
        guideline_set: str,
        propositions: Sequence[GuidelineProposition],
    ) -> Sequence[GuidelineProposition]:
        connected_guidelines_by_proposition = defaultdict[
            GuidelineProposition, list[tuple[Guideline, ConnectionKind]]
        ](list)

        for proposition in propositions:
            connected_guideline_ids = {
                (c.target, c.kind)
                for c in await self._guideline_connection_store.list_connections(
                    indirect=True,
                    source=proposition.guideline.id,
                )
            }

            for connected_guideline_id, connection_kind in connected_guideline_ids:
                if any(connected_guideline_id == p.guideline.id for p in propositions):
                    # no need to add this connected one as it's already an assumed proposition
                    continue

                connected_guideline = await self._guideline_store.read_guideline(
                    guideline_set=guideline_set,
                    guideline_id=connected_guideline_id,
                )

                connected_guidelines_by_proposition[proposition].append(
                    (connected_guideline, connection_kind),
                )

        proposition_and_inferred_guideline_guideline_pairs: list[
            tuple[GuidelineProposition, Guideline, ConnectionKind]
        ] = []

        for proposition, connected_guidelines in connected_guidelines_by_proposition.items():
            for connected_guideline, connection_kind in connected_guidelines:
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
                    if (
                        existing_connection[2] == ConnectionKind.ENTAILS
                        and connection_kind == ConnectionKind.SUGGESTS
                    ):
                        continue  # Stay with existing one
                    elif existing_connection[0].score >= proposition.score:
                        continue  # Stay with existing one
                    else:
                        # This proposition's score is higher, so it's better that
                        # we associate the connected guideline with this one.
                        # we'll add it soon, but meanwhile let's remove the old one.
                        proposition_and_inferred_guideline_guideline_pairs.remove(
                            existing_connection,
                        )

                proposition_and_inferred_guideline_guideline_pairs.append(
                    (proposition, connected_guideline, connection_kind),
                )

        return [
            GuidelineProposition(
                guideline=connection[1],
                score={
                    ConnectionKind.SUGGESTS.name: connection[0].score // 2,
                    ConnectionKind.ENTAILS.name: connection[0].score,
                }[connection[2].name],
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
                    f"When {p.guideline.content.predicate}, then {p.guideline.content.action}"
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
