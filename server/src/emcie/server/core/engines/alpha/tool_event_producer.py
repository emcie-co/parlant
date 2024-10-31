from typing import Mapping, Sequence

from emcie.common.tools import ToolContext
from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.nlp.generation import SchematicGenerator
from emcie.server.core.logging import Logger
from emcie.server.core.agents import Agent
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.services.tools.service_registry import ServiceRegistry
from emcie.server.core.sessions import Event, SessionId, ToolEventData
from emcie.server.core.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.core.glossary import Term
from emcie.server.core.engines.alpha.tool_caller import ToolCallInferenceSchema, ToolCaller
from emcie.server.core.emissions import EmittedEvent, EventEmitter
from emcie.server.core.tools import ToolId


class ToolEventProducer:
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

        self._tool_caller = ToolCaller(logger, service_registry, schematic_generator)

    async def produce_events(
        self,
        event_emitter: EventEmitter,
        session_id: SessionId,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[EmittedEvent]:
        assert len(agents) == 1

        if not tool_enabled_guideline_propositions:
            self._logger.debug("Skipping tool calling; no tools associated with guidelines found")
            return []

        tool_calls = list(
            await self._tool_caller.infer_tool_calls(
                agents,
                context_variables,
                interaction_history,
                terms,
                ordinary_guideline_propositions,
                tool_enabled_guideline_propositions,
                staged_events,
            )
        )

        if not tool_calls:
            return []
        tool_results = await self._tool_caller.execute_tool_calls(
            ToolContext(agent_id=agents[0].id, session_id=session_id),
            tool_calls,
        )

        if not tool_results:
            return []

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

        return [event]
