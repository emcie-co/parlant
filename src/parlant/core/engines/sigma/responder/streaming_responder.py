import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import StringIO
from typing import cast

from parlant.core.agents import CompositionMode
from parlant.core.emissions import EmittedEvent, MessageEventHandle, StatusEventHandle
from parlant.core.engines.alpha.message_event_composer import MessageEventComposition
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import EngineContext, IterationState
from parlant.core.engines.sigma.responder.base_responder import BaseResponder
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.react import (
    Message,
    ParameterSpec,
    ReasoningConfig,
    ReasoningDelta,
    Role,
    StepCompleted,
    StreamEvent,
    TextDelta,
    TextPart,
    ToolCallPart,
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


@dataclass
class _ResponseState:
    current_event: StreamEvent | None = None

    history: list[Message] = field(default_factory=list)

    reasoning_handle: StatusEventHandle | None = None
    reasoning_buffer: StringIO | None = None
    reasoning_chunks: list[str | None] = field(default_factory=list)

    message_handle: MessageEventHandle | None = None
    message_buffer: StringIO | None = None
    message_chunks: list[str | None] = field(default_factory=list)

    message_events: list[EmittedEvent] = field(default_factory=list)
    generation_info: Mapping[str, GenerationInfo] = field(default_factory=dict)

    running_tools: bool = False


class StreamingResponder(BaseResponder):
    async def do_respond(self, context: EngineContext) -> Sequence[MessageEventComposition]:
        state = _ResponseState(history=self._build_history(context))

        while not context.state.prepared_to_respond:
            async for event in self._react.stream_step(
                history=state.history,
                tools=await self._get_tools(context),
                tool_choice="auto",
                reasoning=self._get_reasoning_config(context),
            ):
                await self._on_new_event(state, event)

                await self._update_reasoning(context, state)
                await self._update_tool_calls(context, state)
                await self._update_message(context, state)

            context.state.iterations.append(
                IterationState(
                    matched_guidelines=[],
                    resolved_guidelines=[],
                    tool_insights=ToolInsights(evaluations=[], missing_data=[]),
                    executed_tools=[],
                )
            )

            if len(context.state.iterations) == context.agent.max_engine_iterations:
                # TODO: We need to force a message here in some way...
                # Maybe we can control max turns in the generator itself?
                context.state.prepared_to_respond = True

        await context.session_event_emitter.emit_status_event(
            trace_id=context.tracer.trace_id,
            data=StatusEventData(status="ready", data={"stage": "completed"}),
        )

        # TODO: we need to return generation info from react generator
        return [
            MessageEventComposition(
                generation_info=state.generation_info,
                events=state.message_events,
            )
        ]

    async def _get_tools(self, context: EngineContext) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="check_response_policy",
                description="Use this tool to plan any response so that it aligns with policy.",
                parameters=[
                    ParameterSpec(
                        name="thoughts",
                        type="string",
                        description="Your current thoughts about the response you are planning to generate.",
                    ),
                ],
            )
        ]

    def _get_reasoning_config(self, context: EngineContext) -> ReasoningConfig:
        return ReasoningConfig(
            enabled=True,
            budget_tokens=8192,
            effort="medium",
            visibility="summary",
        )

    async def _on_new_event(self, state: _ResponseState, event: StreamEvent) -> None:
        state.current_event = event

        if isinstance(event, StepCompleted):
            state.history.append(event.result.message)

    async def _update_reasoning(self, context: EngineContext, state: _ResponseState) -> None:
        match state.current_event:
            case ReasoningDelta():
                if state.reasoning_handle is None:  # First reasoning chunk
                    state.reasoning_buffer = StringIO()
                    state.reasoning_buffer.write(state.current_event.text)
                    state.reasoning_chunks = [state.current_event.text]

                    state.reasoning_handle = await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(
                            status="processing",
                            message=state.reasoning_buffer.getvalue(),
                            chunks=state.reasoning_chunks,
                        ),
                    )
                else:  # Subsequent reasoning chunk
                    assert state.reasoning_buffer is not None

                    state.reasoning_buffer.write(state.current_event.text)
                    state.reasoning_chunks.append(state.current_event.text)

                    state.reasoning_handle = await state.reasoning_handle.update(
                        StatusEventData(
                            status="processing",
                            message=state.reasoning_buffer.getvalue(),
                            chunks=state.reasoning_chunks,
                        )
                    )
            case StepCompleted(result=result) if state.reasoning_handle is not None:
                await state.reasoning_handle.update(
                    StatusEventData(
                        status="processing",
                        message=result.message.reasoning,
                        chunks=[*state.reasoning_chunks, None],
                    )
                )

                state.reasoning_buffer = None
                state.reasoning_chunks = []
                state.reasoning_handle = None
            case _ if (
                state.reasoning_handle is not None
            ):  # In case reasoning is followed by events other than StepCompletion
                assert state.reasoning_buffer is not None

                await state.reasoning_handle.update(
                    StatusEventData(
                        status="processing",
                        message=state.reasoning_buffer.getvalue(),
                        chunks=[*state.reasoning_chunks, None],
                    )
                )

                state.reasoning_buffer = None
                state.reasoning_chunks = []
                state.reasoning_handle = None

    async def _update_tool_calls(self, context: EngineContext, state: _ResponseState) -> None:
        match state.current_event:
            case ToolCallStarted():
                if not state.running_tools:
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="processing", message="Evaluating tools"),
                    )

                state.running_tools = True
            case StepCompleted(result=result) if result.needs_tools:
                if len(result.tool_calls) > 2:
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="processing", message="Running tools"),
                    )
                else:
                    assert len(result.tool_calls) == 1

                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(
                            status="processing",
                            message=f"Running tool: {result.tool_calls[0].name}",
                        ),
                    )

                await self._simulate_tool_calls(context, state, result.tool_calls)
            case _:
                state.running_tools = False

    async def _simulate_tool_calls(
        self, context: EngineContext, state: _ResponseState, tool_calls: Sequence[ToolCallPart]
    ) -> None:
        parts: list[ToolResultPart] = []

        for tool_call in tool_calls:
            await asyncio.sleep(1)

            parts.append(
                ToolResultPart(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    content="No special policy to consider at this point!",
                    is_error=False,
                )
            )

        state.history.append(
            Message(
                role=Role.TOOL,
                cache_key=context.session.id,
                parts=parts,
            )
        )

    async def _update_message(self, context: EngineContext, state: _ResponseState) -> None:
        match state.current_event:
            case TextDelta():
                if state.message_handle is None:  # First message chunk
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="typing"),
                    )

                    state.message_buffer = StringIO()
                    state.message_buffer.write(state.current_event.text)
                    state.message_chunks = [state.current_event.text]

                    state.message_handle = await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=MessageEventData(
                            message=state.message_buffer.getvalue(),
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=state.message_chunks,
                        ),
                    )
                else:  # Subsequent message chunk
                    assert state.message_buffer is not None

                    state.message_buffer.write(state.current_event.text)
                    state.message_chunks.append(state.current_event.text)

                    state.message_handle = await state.message_handle.update(
                        MessageEventData(
                            message=state.message_buffer.getvalue(),
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=state.message_chunks,
                        ),
                    )
            case StepCompleted(result=result) if state.message_handle is not None:
                state.message_handle = await state.message_handle.update(
                    MessageEventData(
                        message=result.message.text,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                        chunks=[*state.message_chunks, None],
                    )
                )

                # FIXME: This should come from the generator itself
                generation_info = GenerationInfo(
                    schema_name="react",
                    model="",
                    duration=0.0,
                    usage=UsageInfo(input_tokens=0, output_tokens=0),
                )

                tbd = "react"

                state.generation_info = {tbd: generation_info}
                state.message_events = [state.message_handle.event]

                if not result.needs_tools:
                    context.state.prepared_to_respond = True

                state.message_buffer = None
                state.message_chunks = []
                state.message_handle = None
