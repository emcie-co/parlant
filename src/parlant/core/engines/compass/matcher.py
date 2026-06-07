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
from enum import IntEnum, auto
from itertools import chain
import traceback
from typing import cast

from parlant.core.agents import Effort
from parlant.core.async_utils import safe_gather
from parlant.core.common import Criticality
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.guideline_matcher_registry import GuidelineMatcherRegistry
from parlant.core.engines.compass.guideline_matching.guideline_function_matcher import (
    GuidelineFunctionMatcher,
)
from parlant.core.engines.compass.guideline_matching.guideline_ranker import GuidelineRanker
from parlant.core.engines.compass.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.sessions import ToolEventData
from parlant.core.tags import TagId
from parlant.core.tools import Tool, ToolId, ToolRelevanceResult


class MatcherStrategy(IntEnum):
    """How much effort to spend deciding whether a guideline applies, cheapest to
    most thorough."""

    RECALL = auto()
    RANK = auto()
    DISTILL = auto()


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
        guideline_ranker: GuidelineRanker,
        guideline_function_matcher: GuidelineFunctionMatcher,
        matcher_registry: GuidelineMatcherRegistry,
        relationship_store: RelationshipStore,
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._guideline_recaller = guideline_recaller
        self._guideline_ranker = guideline_ranker
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
        matches = await self._run_batches(context, context.state.usable_guidelines)
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

        matches = await self._run_batches(context, candidates)

        if not matches:
            return

        await self._record(context, matches, append=True)

    def _get_strategy(self, context: EngineContext, guideline: Guideline) -> MatcherStrategy:
        strategy = MatcherStrategy.RECALL

        match context.agent.effort:
            case Effort.MIN:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = MatcherStrategy.RECALL
                    case Criticality.MEDIUM:
                        strategy = MatcherStrategy.RECALL
                    case Criticality.HIGH:
                        strategy = MatcherStrategy.RANK
            case Effort.LOW:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = MatcherStrategy.RECALL
                    case Criticality.MEDIUM:
                        strategy = MatcherStrategy.RECALL
                    case Criticality.HIGH:
                        strategy = MatcherStrategy.DISTILL
            case Effort.MEDIUM:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = MatcherStrategy.RECALL
                    case Criticality.MEDIUM:
                        strategy = MatcherStrategy.RANK
                    case Criticality.HIGH:
                        strategy = MatcherStrategy.DISTILL
            case Effort.HIGH:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = MatcherStrategy.RANK
                    case Criticality.MEDIUM:
                        strategy = MatcherStrategy.RANK
                    case Criticality.HIGH:
                        strategy = MatcherStrategy.DISTILL
            case Effort.MAX:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = MatcherStrategy.RANK
                    case Criticality.MEDIUM:
                        strategy = MatcherStrategy.DISTILL
                    case Criticality.HIGH:
                        strategy = MatcherStrategy.DISTILL

        if strategy < MatcherStrategy.RANK:
            # There are some special conditions under which we want
            # to ensure a baseline of matching effort, since their
            # matching carries important implications.

            if guideline.labels:
                # If the guideline has labels, we want to ensure high-quality analytics,
                # so we should at least rank - not use embeddings.
                return MatcherStrategy.RANK

            if self._check_if_has_dependencies(context, guideline):
                # If the guideline has dependencies, it may be gating important follow-up
                # guidelines, so we should at least rank - not use embeddings.
                return MatcherStrategy.RANK

            if self._check_if_has_tools(context, guideline):
                # If the guideline has tools, it may be gating important interactions
                # with data and/or actions, so we should at least rank - not use embeddings.
                return MatcherStrategy.RANK

        return strategy

    async def _load_strategy_signals(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> None:
        associations = await self._entity_queries.find_guideline_tool_associations()
        context.state.guideline_ids_with_tools = {a.guideline_id for a in associations}

        # A guideline "has dependencies" if it takes part in a dependency
        # relationship — either as the dependent (source) or as the guideline being
        # depended on (target, which gates its dependents). Either side means
        # getting its match right carries downstream consequences.
        #
        # Endpoints may be guidelines directly, or tags (TAG_ALL/TAG_ANY) standing
        # in for every guideline that carries the tag; we resolve both against the
        # candidate guidelines so the per-guideline check can stay a plain lookup.
        dependency_relationships = chain(
            await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY, indirect=False
            ),
            await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY_ANY, indirect=False
            ),
        )

        dependency_guideline_ids: set[GuidelineId] = set()
        dependency_tags: set[TagId] = set()
        for relationship in dependency_relationships:
            for endpoint in (relationship.source, relationship.target):
                if endpoint.kind == RelationshipEntityKind.GUIDELINE:
                    dependency_guideline_ids.add(cast(GuidelineId, endpoint.id))
                elif endpoint.kind.is_tag:
                    dependency_tags.add(cast(TagId, endpoint.id))

        context.state.guideline_ids_with_dependencies = {
            guideline.id
            for guideline in guidelines
            if guideline.id in dependency_guideline_ids
            or not dependency_tags.isdisjoint(guideline.tags)
        }

    def _check_if_has_tools(self, context: EngineContext, guideline: Guideline) -> bool:
        return guideline.id in context.state.guideline_ids_with_tools

    def _check_if_has_dependencies(self, context: EngineContext, guideline: Guideline) -> bool:
        return guideline.id in context.state.guideline_ids_with_dependencies

    async def _run_batches(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> Sequence[GuidelineMatch]:
        """Decide which of `guidelines` apply by routing each to a strategy and
        running the resulting batches in parallel.

        A guideline with a code (Python) matcher always goes to the function
        matcher, regardless of strategy — an explicit matcher is authoritative. The
        rest are bucketed by `_get_strategy`.
        """
        if not guidelines:
            return []

        # _get_strategy runs per guideline and is synchronous, so precompute the
        # store-backed signals it consults (which guidelines have tools / are in a
        # dependency relationship) once, here.
        await self._load_strategy_signals(context, guidelines)

        code_batch: list[Guideline] = []
        recall_batch: list[Guideline] = []
        rank_batch: list[Guideline] = []
        distill_batch: list[Guideline] = []

        for guideline in guidelines:
            if self._matcher_registry.get(guideline.id) is not None:
                code_batch.append(guideline)
                continue

            match self._get_strategy(context, guideline):
                case MatcherStrategy.RECALL:
                    recall_batch.append(guideline)
                case MatcherStrategy.RANK:
                    rank_batch.append(guideline)
                case MatcherStrategy.DISTILL:
                    distill_batch.append(guideline)

        if distill_batch:
            # Distillation isn't wired yet; don't silently drop these guidelines.
            self._logger.warning(
                f"Distillation is not wired yet; skipping {len(distill_batch)} guideline(s)."
            )

        code_matches, recalled, ranked = await safe_gather(
            self._guideline_function_matcher.match(context, code_batch),
            self._guideline_recaller.recall(context, recall_batch),
            self._guideline_ranker.rank(context, rank_batch),
        )

        self._logger.info(
            f"{self.__class__.__name__} guideline ranking usage:\n {ranked.generation_info}"
        )

        matches = list(code_matches)
        matches += [
            GuidelineMatch(
                guideline=rc.guideline,
                rationale="This may or may not be relevant right now - use your judgment.",
            )
            for rc in recalled.recalled_guidelines
            if rc.is_relevant
        ]
        matches += [
            GuidelineMatch(
                guideline=rk.guideline,
                rationale=rk.reasoning
                or "This may or may not be relevant right now - use your judgment.",
            )
            for rk in ranked.ranked_guidelines
            if rk.is_relevant
        ]
        return matches

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
                self._logger.warning(
                    f"Failed to rank tools for service {service_name}: {e!r}\n"
                    f"{traceback.format_exc()}"
                )

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
