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

from parlant.core.async_utils import safe_gather
from parlant.core.emissions import MessageEventHandle, StatusEventHandle
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.sigma.response_state import EngineContext, IterationState
from parlant.core.engines.sigma.loop.loop import Loop, LoopJob, LoopResult
from parlant.core.nlp.common import ModelSize
from parlant.core.nlp.react import (
    Message,
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
    Usage,
    tool_specs_from_tools,
)
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    Participant,
    StatusEventData,
    ToolCall,
    ToolEventData,
)
from parlant.core.tools import ToolResult


@dataclass
class _LoopState:
    start_time: float = field(default_factory=asyncio.get_event_loop().time)

    current_event: StreamEvent | None = None

    history: list[Message] = field(default_factory=list)
    # Index of the turn-instructions message in `history`, so it can be replaced
    # in place when guidelines are reevaluated between steps. Stable because the
    # instructions sit before the last customer message and all later events are
    # appended after them.
    instructions_index: int | None = None
    in_the_middle_of_running_tools: bool = False

    reasoning_handle: StatusEventHandle | None = None
    reasoning_buffer: StringIO | None = None
    reasoning_chunks: list[str | None] = field(default_factory=list)

    message_handle: MessageEventHandle | None = None
    message_buffer: StringIO | None = None
    message_chunks: list[str | None] = field(default_factory=list)

    steps: list[StepResult] = field(default_factory=list)


class StreamingLoop(Loop):
    async def prefill(self, job: LoopJob) -> Usage:
        self._logger.debug(f"Prefilling job for session {job.context.session.id}")

        # Warm the cache for the stable prefix only — the system instructions and
        # the conversation so far. The per-turn instructions are dynamic and sit
        # past the cache breakpoint, so we leave them out here.
        history, _ = await self._build_history(job, include_turn_instructions=False)
        usage = await self._react.prefill(
            history=history,
            tools=await self._get_tools(job.context),
            tool_choice="auto",
            reasoning=job.reasoning_config,
            hints={"model_size": job.model_size},
        )

        if usage.input_tokens > 0:
            self._logger.info(f"{self.__class__.__name__} prefill usage:\n {usage}")

        return usage

    async def run(self, job: LoopJob) -> LoopResult:
        # Give hooks (e.g. retrievers) a chance to stage tool events before we
        # build the history — _build_history folds context.state.tool_events in.
        if not await self._hooks.call_on_generating_messages(job.context):
            # A hook requested that we not proceed with generating a response.
            return LoopResult(job=job, steps=[])

        history, instructions_index = await self._build_history(job)
        state = _LoopState(history=history, instructions_index=instructions_index)

        while not job.context.state.prepared_to_respond:
            # After the first step, refresh the turn instructions in place: this
            # re-invokes the rematch callback (reevaluating guidelines gated on the
            # tools that ran) and swaps just the instructions message — the rest of
            # the step history is preserved.
            if (
                state.instructions_index is not None
                and job.turn_instructions is not None
                and job.context.state.iterations
            ):
                refreshed = await job.turn_instructions(job.context)
                state.history[state.instructions_index] = self._instructions_message(
                    refreshed, job.context.session.id
                )

            async for event in self._react.stream_step(
                history=state.history,
                tools=await self._get_tools(job.context),
                tool_choice="auto",
                reasoning=job.reasoning_config,
                hints={"model_size": job.model_size},
            ):
                await self._on_new_event(state, event)
                await self._update_reasoning(job.context, state)
                await self._update_tool_calls(job.context, state)
                await self._update_message(job.context, state)

            job.context.state.iterations.append(
                IterationState(
                    matched_guidelines=[],
                    ruled_out=[],
                    resolved_guidelines=[],
                    tool_insights=ToolInsights(evaluations={}, missing_data={}),
                    executed_tools=[],
                )
            )

            if len(job.context.state.iterations) == job.context.agent.max_engine_iterations:
                # TODO: We need to force a message here in some way...
                # Maybe we can control max turns in the generator itself?
                # Maybe we should just add to the prompt that we've failed to
                # converge to a desired outcome and are now stopping.
                job.context.state.prepared_to_respond = True

        await job.context.session_event_emitter.emit_status_event(
            trace_id=job.context.tracer.trace_id,
            data=StatusEventData(status="ready", data={"stage": "completed"}),
        )

        await self._hooks.call_on_messages_emitted(job.context)

        return LoopResult(job=job, steps=state.steps)

    async def _get_tools(self, context: EngineContext) -> list[ToolSpec]:
        return [*tool_specs_from_tools(context.state.available_tools)]

    async def _on_new_event(self, state: _LoopState, event: StreamEvent) -> None:
        state.current_event = event

        if isinstance(event, StepCompleted):
            state.history.append(event.result.message)
            state.steps.append(event.result)

            if event.result.message.reasoning:
                self._logger.info(
                    f"{self.__class__.__name__} step reasoning:\n {event.result.message.reasoning}"
                )
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
                if len(result.tool_calls) == 1:
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(
                            status="processing",
                            message=f"Running tool: {result.tool_calls[0].name}",
                        ),
                    )
                else:
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="processing", message="Running tools"),
                    )

                await self._run_tool_calls(context, state, result.tool_calls)
            case _:
                state.in_the_middle_of_running_tools = False

    async def _run_tool_calls(
        self,
        context: EngineContext,
        state: _LoopState,
        tool_calls: Sequence[ToolCallPart],
    ) -> None:
        # Run all of the step's tool calls concurrently; results keep call order.
        results: tuple[ToolResult | None] = await safe_gather(
            *(self._run_tool_call(context, tool_call) for tool_call in tool_calls)
        )

        calls_and_results = list(zip(tool_calls, results))

        step_parts: list[ToolResultPart] = []
        transient_call_ids: list[str] = []
        persisted_call_ids: list[str] = []

        for tool_call, result in calls_and_results:
            if result is not None:
                if result.control.get("lifespan", "session") == "session":
                    persisted_call_ids.append(tool_call.id)
                else:
                    transient_call_ids.append(tool_call.id)

                step_parts.append(
                    ToolResultPart(
                        call_id=tool_call.id,
                        name=tool_call.name,
                        content=result.data,
                        is_error="error_details" in result.metadata,
                    )
                )
            else:
                step_parts.append(
                    ToolResultPart(
                        call_id=tool_call.id,
                        name=tool_call.name,
                        content=f"Unknown tool: {tool_call.name}",
                        is_error=True,
                    )
                )

        # Emit transient results into the transient response context
        transient_tool_event = await context.response_event_emitter.emit_tool_event(
            trace_id=context.tracer.trace_id,
            data=ToolEventData(
                tool_calls=[
                    ToolCall(
                        tool_id=tool_call.name,
                        arguments=tool_call.args,
                        result={
                            "data": result.data,
                            "metadata": result.metadata,
                            "control": result.control,
                            "guidelines": result.guidelines,
                            "canned_responses": result.canned_responses,
                            "canned_response_fields": result.canned_response_fields,
                        },
                        rationale=state.reasoning_buffer.getvalue()
                        if state.reasoning_buffer
                        else "Not provided.",
                    )
                    for tool_call, result in calls_and_results
                    if result and tool_call.id in transient_call_ids
                ]
            ),
        )

        # Emit persisted results into the session response context
        persisted_tool_event = await context.session_event_emitter.emit_tool_event(
            trace_id=context.tracer.trace_id,
            data=ToolEventData(
                tool_calls=[
                    ToolCall(
                        tool_id=tool_call.name,
                        arguments=tool_call.args,
                        result={
                            "data": result.data,
                            "metadata": result.metadata,
                            "control": result.control,
                            "guidelines": result.guidelines,
                            "canned_responses": result.canned_responses,
                            "canned_response_fields": result.canned_response_fields,
                        },
                        rationale=state.reasoning_buffer.getvalue()
                        if state.reasoning_buffer
                        else "Not provided.",
                    )
                    for tool_call, result in calls_and_results
                    if result and tool_call.id in persisted_call_ids
                ]
            ),
        )

        context.state.tool_events.append(transient_tool_event)
        context.state.tool_events.append(persisted_tool_event)

        # Finally, append all the react step parts
        state.history.append(
            Message(
                role=Role.TOOL,
                cache_key=context.session.id,
                parts=list(step_parts),
            )
        )

    async def _run_tool_call(
        self, context: EngineContext, tool_call: ToolCallPart
    ) -> ToolResult | None:
        tool_id = context.state.tool_ids_by_name.get(tool_call.name)

        if tool_id is None:
            self._logger.warning(f"Model requested an unknown tool: {tool_call.name}")
            return None

        return await self._tool_runner.run_tool(context, tool_id, tool_call.args)

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

                # A message was produced on this step's completion (it was
                # already streamed/emitted above).
                await self._hooks.call_on_message_generated(context, result.message.text)

    def _get_model_size(self, context: EngineContext, state: _LoopState) -> ModelSize:
        return ModelSize.MEDIUM

    def _tool_event_messages(self, data: ToolEventData, cache_key: str) -> list[Message]:
        messages: list[Message] = []

        call_id = 0
        for call in data["tool_calls"]:
            call_id += 1
            is_error = "error_details" in call.get("result", {}).get("metadata", {})

            messages.append(
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

        return messages

    async def _build_history(
        self,
        job: LoopJob,
        *,
        include_turn_instructions: bool = True,
    ) -> tuple[list[Message], int | None]:
        cache_key = job.context.session.id

        system_message = Message(
            role=Role.SYSTEM,
            cache_key=cache_key,
            parts=[TextPart(text=job.system_instructions)],
        )

        history = [system_message]

        for event in job.context.interaction.events:
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
                history.extend(
                    self._tool_event_messages(cast(ToolEventData, event.data), cache_key)
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

        # Tool events staged for this turn (e.g. retriever results emitted during
        # the on_generating_messages hook) aren't in the interaction history yet,
        # so fold them in here so the model sees the retrieved context.
        for tool_event in job.context.state.tool_events:
            history.extend(
                self._tool_event_messages(cast(ToolEventData, tool_event.data), cache_key)
            )

        instructions_index: int | None = None

        if include_turn_instructions and job.turn_instructions:
            turn_instructions = await job.turn_instructions(job.context)

            # Place the instructions immediately BEFORE the last customer message
            # rather than at the very end. Ending the prompt on an imperative note
            # makes the model treat it as the turn to answer — it paraphrases or
            # echoes the instructions back instead of replying. Keeping the
            # customer's message last keeps the model answering the customer.
            instructions_index = next(
                (i for i in range(len(history) - 1, -1, -1) if history[i].role == Role.USER),
                len(history),
            )
            history.insert(
                instructions_index, self._instructions_message(turn_instructions, cache_key)
            )

        return history, instructions_index

    def _instructions_message(self, turn_instructions: str, cache_key: str) -> Message:
        return Message(
            role=Role.SYSTEM,
            cache_key=cache_key,
            parts=[
                TextPart(
                    text=f"""\
[The following is context about the current state of the conversation — the guidelines, glossary, and tools relevant to it. Treat it as background that informs your next reply; it is NOT itself a message addressed to you, so never respond to it, acknowledge it, or refer to it.]:
{turn_instructions}"""
                )
            ],
        )
