from abc import ABC, abstractmethod
from collections.abc import Sequence
from io import StringIO
from typing import cast

from parlant.core.agents import CompositionMode, MessageOutputMode
from parlant.core.emissions import MessageEventHandle, StatusEventHandle
from parlant.core.engines.alpha.message_event_composer import MessageEventComposition
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import EngineContext, IterationState
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.react import (
    Message,
    ParameterSpec,
    ReactGenerator,
    ReasoningConfig,
    ReasoningDelta,
    Role,
    StepCompleted,
    TextDelta,
    TextPart,
    ToolCallStarted,
    ToolResultPart,
    ToolSpec,
)
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    Participant,
    StatusEventData,
    ToolEventData,
)
from parlant.core.tracer import Tracer


class BaseResponder(ABC):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        optimization_policy: OptimizationPolicy,
        react: ReactGenerator,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._optimization_policy = optimization_policy
        self._react = react

    @abstractmethod
    async def do_respond(self, context: EngineContext) -> Sequence[MessageEventComposition]: ...

    def _build_system_prompt(self, context: EngineContext) -> str:
        builder = PromptBuilder()

        builder.add_section(
            name="responder-general-instructions",
            template="""\
Please generate a response to the customer message based on the following information. Follow the guidelines and use the tools as needed.

Whenever you need to apply reasoning, please show your thoughts as your preliminary reasoning.

When you might need to comply with policy, please use the associated tool first to understand the policy requirements.

Always reason step by step before *and after* running *any* tool, including the policy tool.
""",
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)
        builder.add_context_variables(context.state.context_variables)
        builder.add_glossary(list(context.state.glossary_terms))

        return builder.build()

    def _build_history(self, context: EngineContext) -> list[Message]:
        cache_key = context.session.id

        system_message = Message(
            role=Role.SYSTEM,
            cache_key=cache_key,
            parts=[TextPart(text=self._build_system_prompt(context))],
        )

        history = [system_message]

        for event in context.interaction.events:
            if event.kind == EventKind.MESSAGE and event.source == EventSource.CUSTOMER:
                history.append(
                    Message(
                        role=Role.USER,
                        cache_key=cache_key,
                        parts=[TextPart(text=cast(MessageEventData, event.data)["message"])],
                    )
                )
            elif event.source == EventSource.CUSTOMER_UI:
                history.append(
                    Message(
                        role=Role.USER,
                        cache_key=cache_key,
                        parts=[TextPart(text=f"[Customer UI Event]: {event.data}")],
                    )
                )
            elif event.kind == EventKind.MESSAGE and event.source in (
                EventSource.AI_AGENT,
                EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
            ):
                history.append(
                    Message(
                        role=Role.ASSISTANT,
                        cache_key=cache_key,
                        parts=[TextPart(text=cast(MessageEventData, event.data)["message"])],
                    )
                )
            elif event.source == EventSource.HUMAN_AGENT:
                message_data = cast(MessageEventData, event.data)

                history.append(
                    Message(
                        role=Role.ASSISTANT,
                        cache_key=cache_key,
                        parts=[
                            TextPart(
                                text=f"[Intervention by human agent. Name: {message_data['participant']['display_name']}]: {message_data['message']}"
                            )
                        ],
                    )
                )
            elif event.kind == EventKind.TOOL and event.source == EventSource.SYSTEM:
                call_id = 0

                for call in cast(ToolEventData, event.data)["tool_calls"]:
                    call_id += 1
                    is_error = "error_details" in call.get("result", {}).get("metadata", {})

                    history.append(
                        Message(
                            role=Role.TOOL,
                            cache_key=cache_key,
                            parts=[
                                ToolResultPart(
                                    call_id=str(call_id),
                                    name=call["tool_id"],
                                    content=call["result"].get("data", {}),
                                    is_error=is_error,
                                )
                            ],
                        )
                    )

        return history
