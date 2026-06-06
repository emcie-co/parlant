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

from typing import Any, cast

from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.compass.loop.loop import LoopJob
from parlant.core.engines.compass.loop.streaming_loop import StreamingLoop
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.react import Role
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.guideline_matching.utils import create_engine_context


def _make_streaming_loop() -> StreamingLoop:
    # _build_history only reads the LoopJob, so the heavier collaborators
    # (meter/optimization_policy/react/tool_runner) aren't exercised here.
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    return StreamingLoop(
        logger=logger,
        tracer=tracer,
        meter=cast(Any, None),
        optimization_policy=cast(Any, None),
        react=cast(Any, None),
        tool_runner=cast(Any, None),
        hooks=EngineHooks(),
    )


async def test_that_turn_instructions_are_placed_before_the_last_customer_message() -> None:
    context = create_engine_context(
        conversation=[
            (EventSource.CUSTOMER, "hi"),
            (EventSource.AI_AGENT, "hello, how can I help?"),
            (EventSource.CUSTOMER, "buying a house for the first time, what do I need to know?"),
        ],
    )
    context.state = ResponseState()

    marker = "TURN_INSTRUCTIONS_MARKER_12345"

    async def turn_instructions(_: EngineContext) -> str:
        return marker

    job = LoopJob(
        context=context,
        system_instructions="SYSTEM_INSTRUCTIONS",
        turn_instructions=turn_instructions,
    )

    history, instructions_index = await _make_streaming_loop()._build_history(job)

    # The model's most recent turn must be the customer's message — not the
    # imperative instructions note, which it otherwise tends to answer / echo.
    assert history[-1].role == Role.USER
    assert "buying a house" in history[-1].text

    # The turn instructions appear exactly once, immediately before that last
    # customer message, and _build_history reports their index (so the loop can
    # replace them in place when reevaluating).
    instruction_indices = [i for i, m in enumerate(history) if marker in m.text]
    assert instruction_indices == [len(history) - 2]
    assert instructions_index == len(history) - 2
