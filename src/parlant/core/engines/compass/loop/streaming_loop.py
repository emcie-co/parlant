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
from io import StringIO

from parlant.core.engines.compass.loop.base_loop import BaseLoop, _LoopState
from parlant.core.engines.compass.context import EngineContext
from parlant.core.nlp.react import StepCompleted, StreamEvent, TextDelta, ToolCallStarted
from parlant.core.sessions import MessageEventData, Participant, StatusEventData


class StreamingLoop(BaseLoop):
    _LOOP_MESSAGE_MODE = "stream"
    """Emits the assistant's message incrementally: each text delta extends the
    message event's growing buffer + `chunks`, so consumers can render it as it's
    produced. The final step-completion update null-terminates `chunks`."""

    async def _surface_message_event(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        match event:
            case TextDelta(text=text):
                if state.message.handle is None:  # First message chunk
                    # Buffer without emitting when either a tool preamble isn't
                    # allowed, or STRICT composition requires reviewing the whole
                    # message before any of it is shown.
                    if not self._tool_preamble_is_allowed(
                        state
                    ) or self._should_buffer_message_for_review(context):
                        if state.message.buffer is None:
                            state.message.buffer = StringIO()
                            state.message.chunks = []

                        state.message.buffer.write(text)
                        state.message.chunks.append(text)
                        await context.dependencies.hooks.call_on_message_updated(
                            context, state.message.buffer.getvalue()
                        )
                        return

                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="typing"),
                    )

                    if state.cancellation_pending():
                        # A newer customer message superseded this turn while
                        # the step shield deferred its cancellation: abandon
                        # before the first visible chunk goes out. (Once
                        # streaming has begun, the message is already on the
                        # wire and completes normally.)
                        raise asyncio.CancelledError()

                    state.message.buffer = StringIO()
                    state.message.buffer.write(text)
                    state.message.chunks = [text]

                    state.message.handle = await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=MessageEventData(
                            message=state.message.buffer.getvalue(),
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=state.message.chunks,
                        ),
                        metadata=state.message.output_metadata(),
                    )
                    self._mark_user_visible_message_emitted(context, state)
                    await context.dependencies.hooks.call_on_message_updated(
                        context, state.message.buffer.getvalue()
                    )
                else:  # Subsequent message chunk
                    assert state.message.buffer is not None

                    state.message.buffer.write(text)
                    state.message.chunks.append(text)

                    state.message.handle = await state.message.handle.update(
                        MessageEventData(
                            message=state.message.buffer.getvalue(),
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=state.message.chunks,
                        ),
                    )
                    await context.dependencies.hooks.call_on_message_updated(
                        context, state.message.buffer.getvalue()
                    )
            case ToolCallStarted() if state.message.handle is not None:
                # Text -> tool transition within a step: finalize the in-flight message
                # as its own bubble so post-tool text starts a fresh one (and the tool
                # status follows the message). The buffer is what was actually shown.
                buffered = state.message.buffer.getvalue() if state.message.buffer else ""
                preamble = (
                    self._tool_preamble_text(buffered)
                    if self._should_trim_tool_preamble_text(state)
                    else buffered
                )

                preamble_data = MessageEventData(
                    message=preamble,
                    participant=Participant(id=context.agent.id, display_name=context.agent.name),
                    chunks=[preamble, None],
                )
                await state.message.handle.update(preamble_data)
                self._compass_tracer.loop_message(
                    preamble_data,
                    self._LOOP_MESSAGE_MODE,
                    state.message.consume_output_ref(),
                )
                self._record_emitted_message(context, preamble_data)

                state.message.emitted_len += len(preamble)
                state.message.keep_tool_text_prefix(state.message.emitted_len)
                state.message.clear_transient_output()
            case ToolCallStarted() if state.message.buffer is not None:
                state.message.suppress_tool_text()
                state.message.clear_transient_output()
            case StepCompleted(result=result):
                # Under STRICT composition the message was buffered (not streamed);
                # review it before emitting. A breach raises _SemanticFailure and
                # regenerates, so nothing was shown to the user.
                await self._maybe_review_message(context, result)

                # Emit the authoritative remainder: everything this step's message holds
                # beyond what interrupt-splits already emitted. Anchoring on
                # `result.message.text` (not the raw buffer) keeps the final segment
                # correct even if a provider delivered tail text outside the deltas.
                remaining = result.message.text[state.message.emitted_len :]

                if state.message.handle is not None:
                    final_data = MessageEventData(
                        message=remaining,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                        chunks=[*state.message.chunks, None],
                    )
                    await state.message.handle.update(final_data)
                    self._compass_tracer.loop_message(
                        final_data,
                        self._LOOP_MESSAGE_MODE,
                        state.message.consume_output_ref(),
                    )
                    self._record_emitted_message(context, final_data)

                    state.message.clear_transient_output()
                    if result.needs_tools:
                        state.message.keep_tool_text_prefix(len(result.message.text))
                    state.message.emitted_len = 0

                    await self._complete_message_step(context, result)
                elif result.needs_tools and state.message.has_custom_commit_policy():
                    state.message.clear_transient_output()
                    state.message.emitted_len = 0
                elif result.needs_tools and not self._tool_preamble_is_allowed(state):
                    state.message.suppress_tool_text()
                    state.message.clear_transient_output()
                    state.message.emitted_len = 0
                elif remaining:
                    if state.cancellation_pending():
                        # Supersede check: nothing streamed yet, so the whole
                        # reply can still be abandoned.
                        raise asyncio.CancelledError()

                    # Text arrived only in the final message (no deltas streamed) — emit
                    # it once as a complete, terminated message.
                    preamble = (
                        self._tool_preamble_text(remaining)
                        if result.needs_tools and self._should_trim_tool_preamble_text(state)
                        else remaining
                    )
                    whole_message_data = MessageEventData(
                        message=preamble,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                        chunks=[preamble, None],
                    )
                    await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=whole_message_data,
                        metadata=state.message.output_metadata(),
                    )
                    self._compass_tracer.loop_message(
                        whole_message_data,
                        self._LOOP_MESSAGE_MODE,
                        state.message.consume_output_ref(),
                    )
                    self._record_emitted_message(context, whole_message_data)
                    self._mark_user_visible_message_emitted(context, state)
                    if result.needs_tools:
                        state.message.keep_tool_text_prefix(len(preamble))

                    state.message.emitted_len = 0

                    await self._complete_message_step(context, result)
                else:
                    state.message.emitted_len = 0
