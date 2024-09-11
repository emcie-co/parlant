from itertools import chain
from typing import Mapping, Sequence, cast

from emcie.common.tools import Tool, ToolContext
from emcie.server.contextual_correlator import ContextualCorrelator
from emcie.server.engines.alpha.tool_call_evaluation import ToolCallEvaluationsSchema
from emcie.server.llm.json_generators import JSONGenerator
from emcie.server.logger import Logger
from emcie.server.core.agents import Agent
from emcie.server.core.common import JSONSerializable
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.sessions import Event, SessionId, ToolEventData
from emcie.server.core.tools import ToolService
from emcie.server.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.core.terminology import Term
from emcie.server.engines.alpha.tool_caller import ToolCaller
from emcie.server.engines.event_emitter import EmittedEvent


class ToolEventProducer:
    def __init__(
        self,
        logger: Logger,
        correlator: ContextualCorrelator,
        tool_service: ToolService,
        tool_caller_evaluation_generator: JSONGenerator[ToolCallEvaluationsSchema],
    ) -> None:
        self.logger = logger
        self.correlator = correlator

        self.tool_caller = ToolCaller(logger, tool_service, tool_caller_evaluation_generator)

    async def produce_events(
        self,
        session_id: SessionId,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[EmittedEvent]:
        assert len(agents) == 1

        emitted_events = []

        if not tool_enabled_guideline_propositions:
            self.logger.debug("Skipping tool calling; no tools associated with guidelines found")
            return []

        tool_calls = list(
            await self.tool_caller.infer_tool_calls(
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

        tools = list(chain(*tool_enabled_guideline_propositions.values()))

        tool_results = await self.tool_caller.execute_tool_calls(
            ToolContext(session_id=session_id),
            tool_calls,
            tools,
        )

        if not tool_results:
            return []

        data: ToolEventData = {
            "tool_calls": [
                {
                    "tool_name": r.tool_call.name,
                    "arguments": r.tool_call.arguments,
                    "result": r.result,
                }
                for r in tool_results
            ]
        }

        emitted_events.append(
            EmittedEvent(
                source="server",
                kind="tool",
                correlation_id=self.correlator.correlation_id,
                data=cast(JSONSerializable, data),
            )
        )

        return emitted_events
