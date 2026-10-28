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

"""The model's streamed reasoning is the only status text that is internal.
The loop marks it with ``__parlant__.internal_reasoning`` so the API can tell
it apart from tool narrations and stage labels, which every caller may read."""

from typing import cast

from parlant.core import metadata as metadata_utils
from parlant.core.engines.compass.loop.base_loop import _LoopState, _ReasoningEventProcessor
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.nlp.react import ReasoningDelta
from parlant.core.sessions import EventKind, StatusEventData

from tests.core.stable.engines.compass.matching.utils import create_engine_context


async def test_that_streamed_model_reasoning_is_emitted_as_a_status_event_flagged_internal_reasoning() -> (
    None
):
    context = create_engine_context(conversation=[])
    state = _LoopState()

    await _ReasoningEventProcessor().process(
        context, state, ReasoningDelta(text="The customer is asking")
    )
    await _ReasoningEventProcessor().process(
        context, state, ReasoningDelta(text=" about order 123.")
    )

    buffer = cast(EventBuffer, context.session_event_emitter)
    assert len(buffer.events) == 1

    event = buffer.events[0]
    assert event.kind == EventKind.STATUS

    data = cast(StatusEventData, event.data)
    assert data["message"] == "The customer is asking about order 123."
    assert data["chunks"] == ["The customer is asking", " about order 123."]

    # The flag is set on the first chunk and survives every later update.
    assert metadata_utils.internal_value(event.metadata, ("internal_reasoning",), None) is True
