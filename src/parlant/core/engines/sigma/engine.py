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

from collections import defaultdict
from collections.abc import Iterable, Sequence
import traceback
from typing_extensions import override

from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EventEmitter
from parlant.core.engines.alpha.entity_context import EntityContext
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.engine_context import EngineContext, Interaction
from parlant.core.engines.sigma.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.engines.sigma.responder import Responder
from parlant.core.engines.sigma.response_state import ResponseState
from parlant.core.engines.sigma.task_runner import TaskRunner
from parlant.core.engines.types import Context, Engine, UtteranceRequest
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.sessions import StatusEventData
from parlant.core.tools import Tool, ToolId
from parlant.core.tracer import Tracer


class SigmaEngine(Engine):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        guideline_recaller: GuidelineRecaller,
        responder: Responder,
        task_runner: TaskRunner,
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._guideline_recaller = guideline_recaller
        self._responder = responder
        self._task_runner = task_runner

        self._entity_queries = entity_queries

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

        # TODO: This should prepare EITHER the responder OR the task runner,
        # depending on the effort level and context
        await self._responder.prepare(engine_context)

    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        try:
            engine_context = await self._load_context(context, event_emitter)

            await event_emitter.emit_status_event(
                trace_id=self._tracer.trace_id,
                data=StatusEventData(status="processing", message="Checking policies"),
            )

            await self._load_guidelines(engine_context)

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
            state=ResponseState(),
        )

        # Resolve the agent's stable tool catalog here (shared by the prefill in
        # initialize() and the response in process()), so the tools block at the
        # front of the cached prefix is identical across both.
        await self._load_available_tools(result)

        # Set in context for access by hooks and other components
        # FIXME: remove type ignore
        EntityContext.set(result)  # type: ignore

        return result

    async def _load_interaction_state(self, context: Context) -> Interaction:
        history = await self._entity_queries.find_events(context.session_id)

        return Interaction(
            events=history,
        )

    async def _load_available_tools(self, context: EngineContext) -> None:
        # The agent's full tool catalog: every tool reachable from its usable
        # guidelines, sorted deterministically. It is independent of what matched
        # this turn, so the cached tools prefix stays stable across turns.
        context.state.usable_guidelines = list(
            await self._entity_queries.find_guidelines_for_context(context.agent.id, [])
        )
        guideline_ids = {g.id for g in context.state.usable_guidelines}

        tool_ids = sorted(
            {
                association.tool_id
                for association in await self._entity_queries.find_guideline_tool_associations()
                if association.guideline_id in guideline_ids
            }
        )

        context.state.available_tools = await self._resolve_tool_ids(tool_ids)

    async def _load_guidelines(self, context: EngineContext) -> None:
        # Reuse the usable guidelines already loaded by _load_available_tools.
        guidelines = await self._guideline_recaller.recall(context, context.state.usable_guidelines)

        matches = [
            GuidelineMatch(
                guideline=rc.guideline,
                rationale="This may or may not be relevant right now - use your judgment.",
            )
            for rc in guidelines.recalled_guidelines
        ]

        # Distinguish between ordinary and tool-enabled guidelines (as the alpha
        # engine does) — tool-enabled ones carry the tools the response may run.
        context.state.tool_enabled_guideline_matches = (
            await self._find_tool_enabled_guideline_matches(matches)
        )
        context.state.ordinary_guideline_matches = list(
            set(matches).difference(set(context.state.tool_enabled_guideline_matches.keys()))
        )

        # The per-turn matched subset (for the prompt's tool descriptions). The
        # full catalog the model may actually call lives in state.available_tools.
        matched_tool_ids = list(
            dict.fromkeys(
                tool_id
                for tool_ids in context.state.tool_enabled_guideline_matches.values()
                for tool_id in tool_ids
            )
        )
        context.state.tools = await self._resolve_tool_ids(matched_tool_ids)

    async def _find_tool_enabled_guideline_matches(
        self,
        guideline_matches: Sequence[GuidelineMatch],
    ) -> dict[GuidelineMatch, list[ToolId]]:
        matches_by_id = {m.guideline.id: m for m in guideline_matches}

        tools_for_guidelines: dict[GuidelineMatch, list[ToolId]] = defaultdict(list)
        for association in await self._entity_queries.find_guideline_tool_associations():
            if association.guideline_id in matches_by_id:
                tools_for_guidelines[matches_by_id[association.guideline_id]].append(
                    association.tool_id
                )

        return dict(tools_for_guidelines)

    async def _resolve_tool_ids(self, tool_ids: Iterable[ToolId]) -> list[Tool]:
        """Resolve ToolIds into their full Tool definitions, skipping any that
        fail to resolve."""
        tools: list[Tool] = []
        for tool_id in tool_ids:
            try:
                service = await self._entity_queries.read_tool_service(tool_id.service_name)
                tools.append(await service.read_tool(tool_id.tool_name))
            except Exception as e:
                self._logger.warning(f"Failed to resolve tool {tool_id.to_string()}: {e}")

        return tools
