from itertools import chain
from typing import Mapping, Sequence, cast
from loguru import logger

from emcie.common.tools import Tool
from emcie.server.core.agents import Agent
from emcie.server.core.common import JSONSerializable
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.sessions import Event, ToolEventData
from emcie.server.core.tools import ToolService
from emcie.server.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.core.terminology import Term
from emcie.server.engines.alpha.tool_caller import ToolCaller
from emcie.server.engines.alpha.utils import make_llm_client
from emcie.server.engines.common import ProducedEvent


class ToolEventProducer:
    def __init__(
        self,
        tool_service: ToolService,
    ) -> None:
        self._llm_client = make_llm_client("openai")
        self.tool_caller = ToolCaller(tool_service)

    async def produce_events(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[ProducedEvent],
    ) -> Sequence[ProducedEvent]:
        assert len(agents) == 1

        produced_events = []

        if not tool_enabled_guideline_propositions:
            logger.debug("Skipping tool calling; no tools associated with guidelines found")
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
            tool_calls,
            tools,
        )

        if not tool_results:
            return []

        data: ToolEventData = {
            "tool_results": [
                {
                    "tool_name": r.tool_call.name,
                    "arguments": r.tool_call.arguments,
                    "result": r.result,
                }
                for r in tool_results
            ]
        }

        produced_events.append(
            ProducedEvent(
                source="server",
                kind=Event.TOOL_KIND,
                data=cast(JSONSerializable, data),
            )
        )

        return produced_events
