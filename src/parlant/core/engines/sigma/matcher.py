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
from itertools import chain
from typing import cast

from parlant.core.async_utils import safe_gather
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.guideline_matcher_registry import GuidelineMatcherRegistry
from parlant.core.engines.sigma.guideline_matching.guideline_function_matcher import (
    GuidelineFunctionMatcher,
)
from parlant.core.engines.sigma.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.engines.sigma.response_state import EngineContext
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.sessions import ToolEventData
from parlant.core.tools import Tool, ToolId, ToolRelevanceResult


class Matcher:
    """Prepares the turn's response state: which guidelines apply and which tools
    are offered to the model.

    ``fill`` does the initial preparation; ``update`` refreshes it after a step
    (reevaluating guidelines gated on the tools that just ran). Both leave
    ``context.state`` carrying the matched guidelines, the matched guidelines'
    tools, and the offered tool catalog.
    """

    _MAX_AVAILABLE_TOOLS = 10

    def __init__(
        self,
        logger: Logger,
        guideline_recaller: GuidelineRecaller,
        guideline_function_matcher: GuidelineFunctionMatcher,
        matcher_registry: GuidelineMatcherRegistry,
        relationship_store: RelationshipStore,
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._guideline_recaller = guideline_recaller
        self._guideline_function_matcher = guideline_function_matcher
        self._matcher_registry = matcher_registry
        self._relationship_store = relationship_store
        self._entity_queries = entity_queries

    async def fill(self, context: EngineContext) -> None:
        """Initial preparation: match all usable guidelines and rank the agent's
        tool pool (independent, so in parallel), then select the offered tools."""
        await safe_gather(self._match(context), self._rank_tool_pool(context))
        await self._select_tools(context)

    async def update(self, context: EngineContext) -> None:
        """Refresh after a step: reevaluate guidelines gated on the tools that
        just ran, then re-select tools. The tool pool ranking depends on the
        (unchanged) conversation, so it isn't re-ranked."""
        await self._reevaluate(context)
        await self._select_tools(context)

    # --- guideline matching ---

    async def _match(self, context: EngineContext) -> None:
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

        await self._record(context, matches, append=False)

    async def _reevaluate(self, context: EngineContext) -> None:
        executed_tool_ids = self._executed_tool_ids(context)
        if not executed_tool_ids:
            return

        gated = await self._find_guidelines_gated_on_tools(context, executed_tool_ids)
        already_matched = self._matched_guideline_ids(context)
        candidates = [g for g in gated if g.id not in already_matched]
        if not candidates:
            return

        # Code-attached gated guidelines are gated by their matcher (which can see
        # the new tool results via the context); the rest are armed — surfaced for
        # the react model to apply now that the tool's results are present.
        code = [g for g in candidates if self._matcher_registry.get(g.id) is not None]
        armed = [g for g in candidates if self._matcher_registry.get(g.id) is None]

        code_matches = await self._guideline_function_matcher.match(context, code)
        armed_matches = [
            GuidelineMatch(
                guideline=g,
                rationale="Reevaluated as relevant after a tool it depends on was executed.",
            )
            for g in armed
        ]
        new_matches = list(code_matches) + armed_matches
        if not new_matches:
            return

        await self._record(context, new_matches, append=True)

    async def _record(
        self,
        context: EngineContext,
        matches: Sequence[GuidelineMatch],
        *,
        append: bool,
    ) -> None:
        # Classify into ordinary vs tool-enabled (the latter carries the tool ids
        # the matched guidelines enable).
        tool_enabled = await self._find_tool_enabled_guideline_matches(matches)
        ordinary = [m for m in matches if m not in tool_enabled]

        if append:
            # Never reorder, so the already-rendered guidelines stay a
            # byte-identical prefix.
            context.state.ordinary_guideline_matches.extend(ordinary)
            context.state.tool_enabled_guideline_matches.update(tool_enabled)
        else:
            context.state.tool_enabled_guideline_matches = tool_enabled
            context.state.ordinary_guideline_matches = list(
                set(matches).difference(set(tool_enabled.keys()))
            )

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

    def _matched_guideline_ids(self, context: EngineContext) -> set[GuidelineId]:
        return {
            m.guideline.id
            for m in chain(
                context.state.ordinary_guideline_matches,
                context.state.tool_enabled_guideline_matches,
            )
        }

    def _executed_tool_ids(self, context: EngineContext) -> set[ToolId]:
        # The react loop records executed tools (by name) in state.tool_events;
        # map them back to ToolIds via the per-turn name->id table.
        executed: set[ToolId] = set()
        for event in context.state.tool_events:
            for call in cast(ToolEventData, event.data)["tool_calls"]:
                if tool_id := context.state.tool_ids_by_name.get(call["tool_id"]):
                    executed.add(tool_id)
        return executed

    async def _find_guidelines_gated_on_tools(
        self,
        context: EngineContext,
        tool_ids: set[ToolId],
    ) -> list[Guideline]:
        usable_by_id = {g.id: g for g in context.state.usable_guidelines}
        gated: dict[GuidelineId, Guideline] = {}

        for tool_id in tool_ids:
            relationships = await self._relationship_store.list_relationships(
                kind=RelationshipKind.REEVALUATION,
                indirect=False,
                target_id=tool_id,
            )

            for relationship in relationships:
                # Source is a guideline (match by id prefix, as elsewhere) or a tag
                # (match by tag membership).
                matched = [
                    g for gid, g in usable_by_id.items() if gid.startswith(relationship.source.id)
                ]

                if not matched and relationship.source.kind.is_tag:
                    matched = [g for g in usable_by_id.values() if relationship.source.id in g.tags]

                for guideline in matched:
                    gated[guideline.id] = guideline

        return list(gated.values())

    # --- tool selection ---

    async def _select_tools(self, context: EngineContext) -> None:
        # Resolve the matched guidelines' tools, then fold them with the ranked
        # pool into the offered catalog. Idempotent when nothing changed (so the
        # rendered prompt stays byte-identical), and picks up any tools a
        # reevaluated tool-enabled guideline brought in.
        await self._resolve_matched_tools(context)
        self._select_available_tools(context)

    async def _resolve_matched_tools(self, context: EngineContext) -> None:
        tool_ids = list(
            dict.fromkeys(
                tool_id
                for tool_ids in context.state.tool_enabled_guideline_matches.values()
                for tool_id in tool_ids
            )
        )
        context.state.matched_tools = await self._resolve_tool_ids(tool_ids)

    async def _rank_tool_pool(self, context: EngineContext) -> None:
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
