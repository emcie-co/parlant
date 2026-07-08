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
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from parlant.core.async_utils import safe_gather
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.entity_cq import EntityQueries
from parlant.core.rules import RuleId
from parlant.core.nlp.embedding import Embedder, EmbeddingCache
from parlant.core.nlp.service import NLPService
from parlant.core.sessions import EventSource
from parlant.core.tools import Tool, ToolId


_EPSILON = 1e-12


class ToolRecaller:
    DEFAULT_MAX_TOOLS = 16

    def __init__(
        self,
        entity_queries: EntityQueries,
        nlp_service: NLPService,
        embedding_cache: EmbeddingCache,
        max_tools: int = DEFAULT_MAX_TOOLS,
    ) -> None:
        self._entity_queries = entity_queries
        self._nlp_service = nlp_service
        self._embedding_cache = embedding_cache
        self._max_tools = max_tools

    async def prepare(self, context: EngineContext) -> None:
        with context.tracer.span("match.tool.recall"):
            tools_by_rule = await self._load_tools_by_rule(context)
            tools_by_id = self._tools_by_id(tools_by_rule)

            context.state.tools_by_rule = tools_by_rule
            context.state.tool_relevance_scores = dict(
                await self._score_tools(context, tools_by_id)
            )
            context.tracer.set_attribute("candidate_count", len(tools_by_id))
            context.tracer.set_attribute(
                "scored_count",
                len(context.state.tool_relevance_scores),
            )

    async def select(self, context: EngineContext) -> None:
        tools_by_id = self._tools_by_id(context.state.tools_by_rule)

        turn_tool_ids = list(
            dict.fromkeys(
                tool_id
                for tool_ids in context.state.tool_enabled_rule_matches.values()
                for tool_id in tool_ids
                if tool_id in tools_by_id
            )
        )

        session_rule_ids = {rule.id for rule in context.state.session_rules}
        session_tool_ids = [
            tool_id
            for rule_id in sorted(session_rule_ids, key=str)
            for tool_id, _ in sorted(
                context.state.tools_by_rule.get(rule_id, set()),
                key=lambda item: (item[1].name, item[0].service_name),
            )
            if tool_id in tools_by_id
        ]

        all_tool_ids = list(tools_by_id.keys())
        selected_ids = self._select_tool_ids(
            turn_tool_ids=turn_tool_ids,
            session_tool_ids=session_tool_ids,
            all_tool_ids=all_tool_ids,
            tools_by_id=tools_by_id,
            scores=context.state.tool_relevance_scores,
        )
        selected_id_set = set(selected_ids)

        context.state.matched_tools = sorted(
            [tools_by_id[tool_id] for tool_id in turn_tool_ids if tool_id in selected_id_set],
            key=lambda tool: tool.name,
        )
        context.state.available_tools = sorted(
            [tools_by_id[tool_id] for tool_id in selected_ids],
            key=lambda tool: tool.name,
        )
        context.state.tool_ids_by_name = {
            tools_by_id[tool_id].name: tool_id for tool_id in selected_ids
        }

    async def _load_tools_by_rule(
        self,
        context: EngineContext,
    ) -> dict[RuleId, set[tuple[ToolId, Tool]]]:
        usable_rule_ids = {rule.id for rule in context.state.usable_rules}
        tools_by_rule: dict[RuleId, set[tuple[ToolId, Tool]]] = defaultdict(set)

        associations = await self._entity_queries.find_rule_tool_associations()

        for association in associations:
            if association.rule_id not in usable_rule_ids:
                continue

            if tool := await self._resolve_tool_by_id(association.tool_id):
                tools_by_rule[association.rule_id].add((association.tool_id, tool))

        return tools_by_rule

    async def _resolve_tool_by_id(self, tool_id: ToolId) -> Tool | None:
        try:
            service = await self._entity_queries.read_tool_service(tool_id.service_name)
            return await service.read_tool(tool_id.tool_name)
        except Exception:
            return None

    def _select_tool_ids(
        self,
        *,
        turn_tool_ids: Sequence[ToolId],
        session_tool_ids: Sequence[ToolId],
        all_tool_ids: Sequence[ToolId],
        tools_by_id: Mapping[ToolId, Tool],
        scores: Mapping[ToolId, float],
    ) -> list[ToolId]:
        selected: list[ToolId] = []
        seen_ids: set[ToolId] = set()
        seen_names: set[str] = set()

        def add_bucket(tool_ids: Iterable[ToolId]) -> None:
            for tool_id in self._rank_tool_ids(tool_ids, tools_by_id, scores):
                if len(selected) >= self._max_tools:
                    return

                if tool_id in seen_ids:
                    continue

                tool = tools_by_id[tool_id]
                if tool.name in seen_names:
                    continue

                selected.append(tool_id)
                seen_ids.add(tool_id)
                seen_names.add(tool.name)

        turn_set = set(turn_tool_ids)
        session_set = set(session_tool_ids) - turn_set
        remaining_tool_ids = [
            tool_id
            for tool_id in all_tool_ids
            if tool_id not in turn_set and tool_id not in session_set
        ]

        add_bucket(turn_tool_ids)
        add_bucket(session_tool_ids)
        add_bucket(remaining_tool_ids)

        return selected

    def _rank_tool_ids(
        self,
        tool_ids: Iterable[ToolId],
        tools_by_id: Mapping[ToolId, Tool],
        scores: Mapping[ToolId, float],
    ) -> list[ToolId]:
        return sorted(
            dict.fromkeys(tool_id for tool_id in tool_ids if tool_id in tools_by_id),
            key=lambda tool_id: (
                -scores.get(tool_id, 0.0),
                tools_by_id[tool_id].name,
                tool_id.service_name,
            ),
        )

    async def _score_tools(
        self,
        context: EngineContext,
        tools_by_id: Mapping[ToolId, Tool],
    ) -> Mapping[ToolId, float]:
        if not tools_by_id:
            return {}

        queries = self._build_queries(context)
        if not queries or self._nlp_service is None:
            return {tool_id: 0.0 for tool_id in tools_by_id}

        embedder = await self._nlp_service.get_embedder()
        tool_ids = list(tools_by_id)
        tool_texts = [
            self._tool_embedding_content(tool_id, tools_by_id[tool_id]) for tool_id in tool_ids
        ]
        query_vectors, tool_vectors = await safe_gather(
            self._embed_many(embedder, queries),
            self._embed_many(embedder, tool_texts),
        )

        tools_array = np.asarray(tool_vectors, dtype=np.float64)
        queries_array = np.asarray(query_vectors, dtype=np.float64)
        centroid = tools_array.mean(axis=0)

        tool_units, tool_valid = self._row_units(tools_array - centroid)
        query_units, query_valid = self._row_units(queries_array - centroid)
        query_units = query_units[query_valid]

        if query_units.shape[0] == 0:
            return {tool_id: 0.0 for tool_id in tools_by_id}

        scores: dict[ToolId, float] = {}
        for index, tool_id in enumerate(tool_ids):
            if not tool_valid[index]:
                scores[tool_id] = 0.0
                continue

            scores[tool_id] = float((query_units @ tool_units[index]).max())

        return scores

    async def _embed_many(
        self,
        embedder: Embedder,
        texts: Sequence[str],
    ) -> list[tuple[float, ...]]:
        if self._embedding_cache is None:
            result = await embedder.embed(texts)
            return [self._as_tuple(vector) for vector in result.vectors]

        cached_vectors: list[tuple[float, ...] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            if cached_result := await self._embedding_cache.get(
                embedder_type=type(embedder),
                texts=[text],
            ):
                cached_vectors[index] = self._as_tuple(cached_result.vectors[0])
            else:
                missing_indices.append(index)
                missing_texts.append(text)

        if missing_texts:
            result = await embedder.embed(missing_texts)
            for index, text, vector in zip(missing_indices, missing_texts, result.vectors):
                await self._embedding_cache.set(
                    embedder_type=type(embedder),
                    texts=[text],
                    vectors=[vector],
                )
                cached_vectors[index] = self._as_tuple(vector)

        assert all(vector is not None for vector in cached_vectors)
        return [vector for vector in cached_vectors if vector is not None]

    def _build_queries(self, context: EngineContext) -> list[str]:
        queries = [self._build_cumulative_query(context)]
        if latest_user_query := self._build_latest_user_turn_query(context):
            queries.append(latest_user_query)

        return list(dict.fromkeys(query for query in queries if query))

    def _build_cumulative_query(self, context: EngineContext) -> str:
        if not context.interaction.messages and not context.state.session_summary:
            return ""

        lines: list[str] = []
        if context.state.session_summary:
            lines.append(f"Session summary: {context.state.session_summary}")

        lines.extend(
            f"{message.source}: {message.content}" for message in context.interaction.messages
        )
        return "\n".join(lines)

    def _build_latest_user_turn_query(self, context: EngineContext) -> str:
        for message in reversed(context.interaction.messages):
            if message.source == EventSource.CUSTOMER:
                return f"{message.source}: {message.content}"

        return ""

    def _tool_embedding_content(self, tool_id: ToolId, tool: Tool) -> str:
        sections = [f"# {tool.name}", f"Service: {tool_id.service_name}"]

        if tool.description:
            sections.append(tool.description)

        if tool.parameters:
            parameter_lines = []
            for name, (descriptor, _) in sorted(tool.parameters.items()):
                description = descriptor.get("description")
                if description:
                    parameter_lines.append(f"- {name}: {description}")
                else:
                    parameter_lines.append(f"- {name}")

            sections.append("Parameters:\n" + "\n".join(parameter_lines))

        if tool.required:
            sections.append("Required parameters: " + ", ".join(sorted(tool.required)))

        return "\n\n".join(sections)

    def _tools_by_id(
        self,
        tools_by_rule: Mapping[RuleId, set[tuple[ToolId, Tool]]],
    ) -> dict[ToolId, Tool]:
        tools_by_id: dict[ToolId, Tool] = {}
        for tool_id, tool in sorted(
            (item for tools in tools_by_rule.values() for item in tools),
            key=lambda item: (item[1].name, item[0].service_name),
        ):
            tools_by_id.setdefault(tool_id, tool)

        return tools_by_id

    def _row_units(
        self,
        rows: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        valid = norms.reshape(-1) > _EPSILON
        safe = np.where(norms > _EPSILON, norms, 1.0)
        return rows / safe, valid

    def _as_tuple(self, vector: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in vector)
