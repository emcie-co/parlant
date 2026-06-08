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

from collections.abc import Sequence
import traceback
from typing_extensions import override

from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EventEmitter
from parlant.core.engines.entity_context import EntityContext
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.engine_context import Interaction
from parlant.core.engines.compass.matcher import Matcher
from parlant.core.engines.compass.responder import Responder
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.engines.compass.task_runner import TaskRunner
from parlant.core.engines.types import Context, Engine, UtteranceRequest
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.sessions import StatusEventData
from parlant.core.tracer import Tracer


class CompassEngine(Engine):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        matcher: Matcher,
        responder: Responder,
        task_runner: TaskRunner,
        entity_queries: EntityQueries,
        hooks: EngineHooks,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._matcher = matcher
        self._responder = responder
        self._task_runner = task_runner

        self._entity_queries = entity_queries
        self._hooks = hooks

    @override
    async def initialize(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> None:
        # Warm the provider cache for the system prompt as soon as the session
        # exists, before any message arrives. The interaction is empty here, so
        # only the stable system prefix is warmed; the real turn later reads it.
        engine_context = await self._load_context(
            context,
            event_emitter,
            load_interaction=False,
        )

        # Warm the response state (mostly the tool pool — the interaction is empty,
        # so guideline matching has little to chew on) for the prefill below.
        await self._load_usable_guidelines(engine_context)
        await self._matcher.fill(engine_context)

        # TODO: This should prepare EITHER the responder OR the task runner,
        # depending on the effort level and context
        await self._responder.prepare(engine_context)

    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        # Load the context up front so the error hook (and the lifecycle hooks
        # below) always have it, mirroring the alpha engine.
        engine_context = await self._load_context(context, event_emitter)

        try:
            if not await self._hooks.call_on_acknowledging(engine_context):
                return False  # Hook requested to bail out

            await event_emitter.emit_status_event(
                trace_id=self._tracer.trace_id,
                data=StatusEventData(status="acknowledged"),
            )

            if not await self._hooks.call_on_acknowledged(engine_context):
                return False  # Hook requested to bail out

            await event_emitter.emit_status_event(
                trace_id=self._tracer.trace_id,
                data=StatusEventData(status="processing", message="Thinking"),
            )

            # Fire on_preparing before the (latency-heavy) guideline/tool loading
            # so preparation-time hooks — e.g. global retrievers — start fetching
            # in parallel and have their results ready by message generation.
            if not await self._hooks.call_on_preparing(engine_context):
                return False  # Hook requested to bail out

            await self._load_usable_guidelines(engine_context)

            # Initial match before responding: both the composition-mode decision
            # and the guideline-gated retriever hook (fired at the start of the
            # response loop) need the matches already in place.
            await self._refresh_state(engine_context)

            # The responder re-invokes _refresh_state when (re)building the turn
            # instructions after each step, to reevaluate guidelines gated on the
            # tools that just ran.
            await self._responder.respond(engine_context, self._refresh_state)
        except Exception as e:
            self._logger.error(
                f"Error processing context: {e}\n\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
            )

            if await self._hooks.call_on_error(engine_context, e):
                await event_emitter.emit_status_event(
                    trace_id=self._tracer.trace_id,
                    data=StatusEventData(status="error"),
                )

            return False

        return True

    @override
    async def utter(
        self,
        context: Context,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        return False

    async def _load_context(
        self,
        context: Context,
        event_emitter: EventEmitter,
        load_interaction: bool = True,
    ) -> EngineContext:
        # Load the full entities from storage.

        agent = await self._entity_queries.read_agent(context.agent_id)
        session = await self._entity_queries.read_session(context.session_id)
        customer = await self._entity_queries.read_customer(session.customer_id)

        if load_interaction:
            interaction = await self._load_interaction_state(context)
        else:
            interaction = Interaction([])

        result = EngineContext(
            info=context,
            logger=self._logger,
            tracer=self._tracer,
            agent=agent,
            customer=customer,
            session=session,
            session_event_emitter=event_emitter,
            response_event_emitter=EventBuffer(agent),
            interaction=interaction,
            state=ResponseState(),
        )

        # Set in context for access by hooks and other components
        EntityContext.set(result)

        return result

    async def _load_interaction_state(self, context: Context) -> Interaction:
        history = await self._entity_queries.find_events(context.session_id)

        return Interaction(
            events=history,
        )

    async def _load_usable_guidelines(self, context: EngineContext) -> None:
        # The agent's full set of guidelines (used by both guideline matching and
        # tool-relevance scoping); loaded once before the two run in parallel.
        context.state.usable_guidelines = list(
            await self._entity_queries.find_guidelines_for_context(context.agent.id, [])
        )

    async def _refresh_state(self, engine_context: EngineContext) -> None:
        # Called by the responder when (re)building the turn instructions: the
        # initial fill, then an update (reevaluation) after each subsequent step.
        if not engine_context.state.iterations:
            await self._matcher.fill(engine_context)
        else:
            await self._matcher.update(engine_context)
