# Copyright 2026 Emcie Co Ltd.
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
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import StringIO
from typing import cast

from parlant.core.emissions import MessageEventHandle, StatusEventHandle
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import EngineContext, IterationState
from parlant.core.engines.sigma.loop.loop import Loop, LoopJob, LoopResult
from parlant.core.nlp.common import ModelSize
from parlant.core.nlp.react import (
    Message,
    ParameterSpec,
    ReasoningDelta,
    Role,
    StepCompleted,
    StepResult,
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
class _LoopState:
    start_time: float = field(default_factory=asyncio.get_event_loop().time)

    current_event: StreamEvent | None = None

    history: list[Message] = field(default_factory=list)
    in_the_middle_of_running_tools: bool = False

    reasoning_handle: StatusEventHandle | None = None
    reasoning_buffer: StringIO | None = None
    reasoning_chunks: list[str | None] = field(default_factory=list)

    message_handle: MessageEventHandle | None = None
    message_buffer: StringIO | None = None
    message_chunks: list[str | None] = field(default_factory=list)

    steps: list[StepResult] = field(default_factory=list)


class StreamingLoop(Loop):
    async def run(self, job: LoopJob) -> LoopResult:
        context, prompt = job.context, job.prompt

        state = _LoopState(history=self._build_history(context, job))

        while not context.state.prepared_to_respond:
            async for event in self._react.stream_step(
                history=state.history,
                tools=await self._get_tools(context),
                tool_choice="auto",
                reasoning=job.reasoning_config,
                hints={"model_size": job.model_size, "service_tier": "priority"},
            ):
                await self._on_new_event(state, event)
                await self._update_reasoning(context, state)
                await self._update_tool_calls(context, state)
                await self._update_message(context, state)

            context.state.iterations.append(
                IterationState(
                    matched_guidelines=[],
                    resolved_guidelines=[],
                    tool_insights=ToolInsights(evaluations={}, missing_data={}),
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

        return LoopResult(prompt=prompt, steps=state.steps)

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

    async def _on_new_event(self, state: _LoopState, event: StreamEvent) -> None:
        state.current_event = event

        if isinstance(event, StepCompleted):
            state.history.append(event.result.message)
            state.steps.append(event.result)
            self._logger.info(f"{self.__class__.__name__} step usage:\n {event.result.usage}")

    async def _update_reasoning(self, context: EngineContext, state: _LoopState) -> None:
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

    async def _update_tool_calls(self, context: EngineContext, state: _LoopState) -> None:
        match state.current_event:
            case ToolCallStarted():
                if not state.in_the_middle_of_running_tools:
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="processing", message="Evaluating tools"),
                    )

                state.in_the_middle_of_running_tools = True
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
                state.in_the_middle_of_running_tools = False

    async def _simulate_tool_calls(
        self, context: EngineContext, state: _LoopState, tool_calls: Sequence[ToolCallPart]
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

    async def _update_message(self, context: EngineContext, state: _LoopState) -> None:
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

                if not result.needs_tools:
                    context.state.prepared_to_respond = True

                state.message_buffer = None
                state.message_chunks = []
                state.message_handle = None

    def _get_model_size(self, context: EngineContext, state: _LoopState) -> ModelSize:
        return ModelSize.MEDIUM

    def _build_history(self, context: EngineContext, job: LoopJob) -> list[Message]:
        cache_key = context.session.id

        system_message = Message(
            role=Role.SYSTEM,
            cache_key=cache_key,
            parts=[TextPart(text=job.prompt)],
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

        # Providers (e.g. Gemini, Anthropic) require at least one non-system
        # turn. When the agent speaks first — a greeting before any customer
        # message — the interaction is empty and the history holds only the
        # system message; give the model a user turn to respond to.
        if len(history) == 1:
            history.append(
                Message(
                    role=Role.USER,
                    cache_key=cache_key,
                    parts=[TextPart(text="[The conversation has not started yet.]")],
                )
            )

        if job.reminder:
            history.append(
                Message(
                    role=Role.SYSTEM,
                    cache_key=cache_key,
                    parts=[
                        TextPart(
                            text=f"""\
[Note to self as a reminder while interacting with the user]:
### Start of note-to-self
{job.reminder(context)}
### End of note-to-self"""
                        )
                    ],
                )
            )

        return history
