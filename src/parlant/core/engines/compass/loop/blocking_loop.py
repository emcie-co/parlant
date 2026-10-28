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


class BlockingLoop(BaseLoop):
    _LOOP_MESSAGE_MODE = "block"
    """Emits the assistant's message once, complete, when the step finishes — no
    incremental updates and no `chunks`, so consumers render it as a single,
    non-streamed message. Text deltas still arrive from the provider; they're just
    not surfaced until the message is whole."""

    async def _surface_message_event(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        match event:
            case TextDelta(text=text):
                # Don't emit deltas in block mode, but DO accumulate the text: it's
                # needed to split the message on a text -> tool transition and to know
                # what was already shown. Show a typing indicator on the first text of
                # each segment.
                if state.message.buffer is None:  # First text since the last emit
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="typing"),
                    )
                    state.message.buffer = StringIO()

                state.message.buffer.write(text)
            case ToolCallStarted() if state.message.buffer is not None:
                # Text -> tool transition within a step: emit the text so far as its own
                # complete message before the tool status, so post-tool text isn't glued
                # onto it.
                buffered = state.message.buffer.getvalue()

                if not self._tool_preamble_is_allowed(state):
                    state.message.suppress_tool_text()
                    state.message.buffer = None
                    return

                preamble = (
                    self._tool_preamble_text(buffered)
                    if self._should_trim_tool_preamble_text(state)
                    else buffered
                )

                if preamble:
                    if state.cancellation_pending():
                        # A newer customer message superseded this turn while
                        # the step shield deferred its cancellation: abandon
                        # before anything user-visible goes out.
                        raise asyncio.CancelledError()

                    preamble_data = MessageEventData(
                        message=preamble,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                    )
                    await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=preamble_data,
                        metadata=state.message.output_metadata(),
                    )
                    self._compass_tracer.loop_message(
                        preamble_data,
                        self._LOOP_MESSAGE_MODE,
                        state.message.consume_output_ref(),
                    )
                    self._record_emitted_message(context, preamble_data)
                    self._mark_user_visible_message_emitted(context, state)
                    state.message.emitted_len += len(preamble)
                    state.message.keep_tool_text_prefix(state.message.emitted_len)

                state.message.clear_transient_output()
            case StepCompleted(result=result):
                # Under STRICT composition, review the whole message before emitting
                # it; a breach raises _SemanticFailure and regenerates (nothing shown).
                await self._maybe_review_message(context, result)

                # Agent-side moderation, at the same pre-emission point: a flag
                # regenerates (raises _SemanticFailure); on the terminal give-up step it
                # can't retry, so it returns the fallback text to emit instead.
                moderation_override = await self._maybe_moderate_message(context, result)

                # Emit the authoritative remainder once, complete and without `chunks`:
                # everything the step's message holds beyond what interrupt-splits
                # already emitted. Anchoring on `result.message.text` (not the buffer)
                # keeps it correct even if a provider delivered text outside the deltas.
                remaining = result.message.text[state.message.emitted_len :]

                if result.needs_tools and state.message.has_custom_commit_policy():
                    state.message.buffer = None
                    state.message.emitted_len = 0
                    return

                if result.needs_tools and not self._tool_preamble_is_allowed(state):
                    state.message.suppress_tool_text()
                    state.message.buffer = None
                    state.message.emitted_len = 0
                    return

                # The final (non-tool-preamble) message consults the
                # message-generated hooks BEFORE emission: block mode is the
                # one place a BAIL can still suppress the message, since
                # nothing has reached the customer yet (the handler may emit
                # its own replacement). Tool preambles keep the observational
                # post-emission call below, like streamed output.
                proceed = True
                if result.message.text and not result.needs_tools:
                    proceed = await self._complete_message_step(context, result)

                preamble = (
                    self._tool_preamble_text(remaining)
                    if result.needs_tools and self._should_trim_tool_preamble_text(state)
                    else remaining
                )

                if preamble and proceed:
                    if state.cancellation_pending():
                        # Same supersede check for the final message.
                        raise asyncio.CancelledError()

                    moderation_metadata = None
                    if moderation_override is not None:
                        # Only set on the terminal give-up step whose forced message was
                        # itself flagged: emit the configured fallback in its place.
                        preamble, moderation_metadata = moderation_override

                    final_data = MessageEventData(
                        message=preamble,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                    )
                    await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=final_data,
                        metadata=state.message.output_metadata(moderation_metadata),
                    )
                    self._compass_tracer.loop_message(
                        final_data,
                        self._LOOP_MESSAGE_MODE,
                        state.message.consume_output_ref(),
                    )
                    self._record_emitted_message(context, final_data)
                    self._mark_user_visible_message_emitted(context, state)
                    if result.needs_tools:
                        state.message.keep_tool_text_prefix(len(preamble))

                state.message.buffer = None
                state.message.emitted_len = 0

                if result.message.text and result.needs_tools:
                    await self._complete_message_step(context, result)
