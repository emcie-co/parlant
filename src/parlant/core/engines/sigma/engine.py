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
from parlant.core.engines.alpha.entity_context import EntityContext
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import EngineContext, Interaction, ResponseState
from parlant.core.engines.sigma.responder import Responder
from parlant.core.engines.sigma.task_runner import TaskRunner
from parlant.core.engines.types import Context, Engine, UtteranceRequest
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.sessions import StatusEventData
from parlant.core.tracer import Tracer


class SigmaEngine(Engine):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        responder: Responder,
        task_runner: TaskRunner,
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._responder = responder
        self._task_runner = task_runner
        self._entity_queries = entity_queries

    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        try:
            engine_context = await self._load_context(context, event_emitter)

            # await self._task_runner.run(Task(engine_context))
            await self._responder.respond(engine_context)
        except Exception as e:
            self._logger.error(
                f"Error processing context: {e}\n\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
            )

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
            state=ResponseState(
                context_variables=[],
                glossary_terms=set(),
                capabilities=[],
                iterations=[],
                ordinary_guideline_matches=[],
                tool_enabled_guideline_matches={},
                journeys=[],
                journey_paths={
                    k: list(v) for k, v in session.agent_states[-1].journey_paths.items()
                }
                if session.agent_states
                else {},
                tool_events=[],
                tool_insights=ToolInsights(),
                prepared_to_respond=False,
                message_events=[],
            ),
        )

        # Set in context for access by hooks and other components
        # FIXME: remove type ignore
        EntityContext.set(result)  # type: ignore

        return result

    async def _load_interaction_state(self, context: Context) -> Interaction:
        history = await self._entity_queries.find_events(context.session_id)

        return Interaction(
            events=history,
        )
