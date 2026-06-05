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

from parlant.core.async_utils import safe_gather
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EventEmitter
from parlant.core.engines.alpha.entity_context import EntityContext
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.engine_context import Interaction
from parlant.core.engines.guideline_matcher_registry import GuidelineMatcherRegistry
from parlant.core.engines.sigma.guideline_matching.guideline_function_matcher import (
    GuidelineFunctionMatcher,
)
from parlant.core.engines.sigma.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.engines.sigma.responder import Responder
from parlant.core.engines.sigma.response_state import EngineContext, ResponseState
from parlant.core.engines.sigma.task_runner import TaskRunner
from parlant.core.engines.types import Context, Engine, UtteranceRequest
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.sessions import StatusEventData
from parlant.core.tools import Tool, ToolId, ToolRelevanceResult
from parlant.core.tracer import Tracer


class SigmaEngine(Engine):
    _MAX_AVAILABLE_TOOLS = 10

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        guideline_recaller: GuidelineRecaller,
        guideline_function_matcher: GuidelineFunctionMatcher,
        matcher_registry: GuidelineMatcherRegistry,
        responder: Responder,
        task_runner: TaskRunner,
        entity_queries: EntityQueries,
        hooks: EngineHooks,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._guideline_recaller = guideline_recaller
        self._guideline_function_matcher = guideline_function_matcher
        self._matcher_registry = matcher_registry
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

        # No conversation yet, so there are no matched tools — the available set
        # is just the tools most relevant to the agent description.
        await self._load_usable_guidelines(engine_context)
        await self._load_agent_tool_pool(engine_context)
        self._select_available_tools(engine_context)

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

            await self._rematch(engine_context)

            await self._responder.respond(engine_context)
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

    async def _rematch(self, engine_context: EngineContext) -> None:
        # Guideline matching and tool relevance are both embedding-bound and
        # independent, so run them in parallel to hide the added latency.
        await safe_gather(
            self._load_guidelines(engine_context),
            self._load_agent_tool_pool(engine_context),
        )

        self._select_available_tools(engine_context)

    def _build_tool_query(self, context: EngineContext) -> str:
        messages = [f"{m.source}: {m.content}" for m in context.interaction.messages]
        return f"{context.agent.description or ''}\n\n{messages}"

    async def _agent_candidate_tool_ids(self, context: EngineContext) -> set[ToolId]:
        guideline_ids = {g.id for g in context.state.usable_guidelines}
        return {
            association.tool_id
            for association in await self._entity_queries.find_guideline_tool_associations()
            if association.guideline_id in guideline_ids
        }

    async def _load_agent_tool_pool(self, context: EngineContext) -> None:
        # Rank the agent's candidate tools against the agent description + the
        # conversation, scoped per service. Each service ranks only its own tools;
        # we merge the scored results across services.
        candidate_ids = await self._agent_candidate_tool_ids(context)
        # Map names back to ToolIds so a tool call (which carries only a name) can
        # be routed to its service when run.
        context.state.tool_ids_by_name = {tid.tool_name: tid for tid in candidate_ids}

        if not candidate_ids:
            context.state.agent_tool_pool = []
            return

        query = self._build_tool_query(context)

        names_by_service: dict[str, list[str]] = defaultdict(list)
        for tool_id in candidate_ids:
            names_by_service[tool_id.service_name].append(tool_id.tool_name)

        results: list[ToolRelevanceResult] = []
        for service_name, names in names_by_service.items():
            try:
                service = await self._entity_queries.read_tool_service(service_name)
                results.extend(
                    await service.find_relevant_tools(query, names, self._MAX_AVAILABLE_TOOLS)
                )
            except Exception as e:
                self._logger.warning(f"Failed to rank tools for service {service_name}: {e}")

        results.sort(key=lambda r: r.score, reverse=True)
        context.state.agent_tool_pool = [r.tool for r in results]

    def _select_available_tools(self, context: EngineContext) -> None:
        # Matched-turn tools are always included; fill up to _MAX_AVAILABLE_TOOLS
        # with the most relevant general tools.
        chosen: list[Tool] = list(context.state.matched_tools)
        seen = {tool.name for tool in chosen}
        for tool in context.state.agent_tool_pool:
            if len(chosen) >= self._MAX_AVAILABLE_TOOLS:
                break
            if tool.name not in seen:
                seen.add(tool.name)
                chosen.append(tool)

        # Emit by name so an unchanged selection is byte-identical turn to turn,
        # keeping the cached tools prefix warm (selection uses scores; emission
        # order is stable).
        context.state.available_tools = sorted(chosen, key=lambda tool: tool.name)

    async def _load_guidelines(self, context: EngineContext) -> None:
        # Reuse the usable guidelines already loaded by _load_usable_guidelines.
        usable_guidelines = context.state.usable_guidelines

        # Partition by whether a guideline has a code (Python) matcher. Those go
        # to the function matcher; the rest to the (LLM) recaller. An explicit
        # matcher is authoritative, so the recaller never even evaluates them.
        function_attached = [
            g for g in usable_guidelines if self._matcher_registry.get(g.id) is not None
        ]
        recall_candidates = [
            g for g in usable_guidelines if self._matcher_registry.get(g.id) is None
        ]

        # Both are independent, so run them concurrently.
        function_matches, recalled = await safe_gather(
            self._guideline_function_matcher.match(context, function_attached),
            self._guideline_recaller.recall(context, recall_candidates),
        )

        matches = list(function_matches) + [
            GuidelineMatch(
                guideline=rc.guideline,
                rationale="This may or may not be relevant right now - use your judgment.",
            )
            for rc in recalled.recalled_guidelines
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
        context.state.matched_tools = await self._resolve_tool_ids(matched_tool_ids)

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
