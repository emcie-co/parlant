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

from io import StringIO

from parlant.core.engines.compass.loop.base_loop import (
    BaseLoop,
    _LoopState,
)
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.nlp.react import StepCompleted, StreamEvent, TextDelta, ToolCallStarted
from parlant.core.sessions import MessageEventData, Participant, StatusEventData


class StreamingLoop(BaseLoop):
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
                if state.mark_phase_started("message"):
                    CompassTracer(context.tracer).loop_message_started("streaming")

                if state.message.handle is None:  # First message chunk
                    if not self._tool_preamble_is_allowed(state):
                        if state.message.buffer is None:
                            state.message.buffer = StringIO()
                            state.message.chunks = []

                        state.message.buffer.write(text)
                        state.message.chunks.append(text)
                        return

                    status_data = StatusEventData(status="typing")
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=status_data,
                    )

                    state.message.buffer = StringIO()
                    state.message.buffer.write(text)
                    state.message.chunks = [text]

                    message_data = MessageEventData(
                        message=state.message.buffer.getvalue(),
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                        chunks=state.message.chunks,
                    )
                    self._record_message_event(context, state, message_data, "streaming")
                    state.message.handle = await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=message_data,
                    )
                    self._mark_user_visible_message_emitted(state)
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

                await state.message.handle.update(
                    MessageEventData(
                        message=preamble,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                        chunks=[preamble, None],
                    )
                )

                state.message.emitted_len += len(preamble)
                state.message.keep_tool_text_prefix(state.message.emitted_len)
                state.message.clear_transient_output()
                if state.mark_phase_finished("message"):
                    CompassTracer(context.tracer).loop_message_finished(
                        "streaming",
                        emitted=bool(preamble),
                    )
            case ToolCallStarted() if state.message.buffer is not None:
                state.message.suppress_tool_text()
                state.message.clear_transient_output()
                if state.mark_phase_finished("message"):
                    CompassTracer(context.tracer).loop_message_finished(
                        "streaming",
                        emitted=False,
                    )
            case StepCompleted(result=result):
                # Emit the authoritative remainder: everything this step's message holds
                # beyond what interrupt-splits already emitted. Anchoring on
                # `result.message.text` (not the raw buffer) keeps the final segment
                # correct even if a provider delivered tail text outside the deltas.
                remaining = result.message.text[state.message.emitted_len :]

                if state.message.handle is not None:
                    await state.message.handle.update(
                        MessageEventData(
                            message=remaining,
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=[*state.message.chunks, None],
                        )
                    )

                    state.message.clear_transient_output()
                    if result.needs_tools:
                        state.message.keep_tool_text_prefix(len(result.message.text))
                    state.message.emitted_len = 0

                    await self._complete_message_step(context, result)
                    if state.mark_phase_finished("message"):
                        CompassTracer(context.tracer).loop_message_finished(
                            "streaming",
                            emitted=True,
                        )
                elif result.needs_tools and state.message.has_custom_commit_policy():
                    state.message.clear_transient_output()
                    state.message.emitted_len = 0
                    if state.mark_phase_finished("message"):
                        CompassTracer(context.tracer).loop_message_finished(
                            "streaming",
                            emitted=False,
                        )
                elif result.needs_tools and not self._tool_preamble_is_allowed(state):
                    state.message.suppress_tool_text()
                    state.message.clear_transient_output()
                    state.message.emitted_len = 0
                    if state.mark_phase_finished("message"):
                        CompassTracer(context.tracer).loop_message_finished(
                            "streaming",
                            emitted=False,
                        )
                elif remaining:
                    # Text arrived only in the final message (no deltas streamed) — emit
                    # it once as a complete, terminated message.
                    if state.mark_phase_started("message"):
                        CompassTracer(context.tracer).loop_message_started("streaming")

                    preamble = (
                        self._tool_preamble_text(remaining)
                        if result.needs_tools and self._should_trim_tool_preamble_text(state)
                        else remaining
                    )
                    message_data = MessageEventData(
                        message=preamble,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                        chunks=[preamble, None],
                    )
                    self._record_message_event(context, state, message_data, "streaming")
                    await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=message_data,
                    )
                    self._mark_user_visible_message_emitted(state)
                    if result.needs_tools:
                        state.message.keep_tool_text_prefix(len(preamble))

                    state.message.emitted_len = 0

                    await self._complete_message_step(context, result)
                    if state.mark_phase_finished("message"):
                        CompassTracer(context.tracer).loop_message_finished(
                            "streaming",
                            emitted=True,
                        )
                else:
                    state.message.emitted_len = 0
                    if state.mark_phase_finished("message"):
                        CompassTracer(context.tracer).loop_message_finished(
                            "streaming",
                            emitted=False,
                        )
