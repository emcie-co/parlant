# Copyright 2025 Emcie Co Ltd.
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

from collections import deque
from itertools import chain
from typing import Mapping, Sequence, cast

from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.tool_calling.overlapping_tools_batch import (
    OverlappingToolsBatch,
    OverlappingToolsBatchSchema,
)
from parlant.core.engines.alpha.tool_calling.single_tool_batch import (
    SingleToolBatch,
    SingleToolBatchSchema,
)
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCallBatch,
    ToolCallBatcher,
    ToolCallContext,
)
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.relationships import RelationshipStore, RelationshipKind
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import Tool, ToolId, ToolOverlap


class DefaultToolCallBatcher(ToolCallBatcher):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        service_registry: ServiceRegistry,
        single_tool_schematic_generator: SchematicGenerator[SingleToolBatchSchema],
        overlapping_tools_schematic_generator: SchematicGenerator[OverlappingToolsBatchSchema],
        relationship_store: RelationshipStore,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy
        self._service_registry = service_registry
        self._single_tool_schematic_generator = single_tool_schematic_generator
        self._overlapping_tools_schematic_generator = overlapping_tools_schematic_generator
        self._relationship_store = relationship_store

    async def create_batches(
        self,
        tools: Mapping[tuple[ToolId, Tool], Sequence[GuidelineMatch]],
        context: ToolCallContext,
    ) -> Sequence[ToolCallBatch]:
        result: list[ToolCallBatch] = []
        independent_tools = {}
        dependent_tools = {}
        overlapping_tools_batches = []
        visited = set()

        tool_id_to_tool = {k[0]: (k[1], v) for k, v in tools.items()}

        async def collect_overlapping_tools(
            root_id: ToolId,
        ) -> list[tuple[ToolId, Tool, Sequence[GuidelineMatch]]]:
            overlapped_tools: list[tuple[ToolId, Tool, Sequence[GuidelineMatch]]] = []
            queue = deque([root_id])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                all_relationships = list(
                    chain(
                        await self._relationship_store.list_relationships(
                            source_id=current, indirect=False, kind=RelationshipKind.OVERLAP
                        ),
                        await self._relationship_store.list_relationships(
                            target_id=current, indirect=False, kind=RelationshipKind.OVERLAP
                        ),
                    )
                )
                for r in all_relationships:
                    neighbor = (
                        cast(ToolId, r.target.id)
                        if cast(ToolId, r.target.id) != current
                        else cast(ToolId, r.source.id)
                    )
                    if neighbor in tool_id_to_tool and neighbor not in visited:
                        if tool_id_to_tool[neighbor][0].overlap == ToolOverlap.NONE:
                            self._logger.warning(
                                f"Overlap relationship ignored because: {tool_id_to_tool[neighbor][0].name} has ToolOverlap.NONE"
                            )
                            continue
                        overlapped_tools.append(
                            (
                                neighbor,
                                tool_id_to_tool[neighbor][0],
                                tool_id_to_tool[neighbor][1],
                            )
                        )
                        queue.append(neighbor)
            return overlapped_tools

        for (tool_id, _tool), guidelines in tools.items():
            if _tool.overlap == ToolOverlap.NONE:
                independent_tools[tool_id] = (_tool, guidelines)
            elif _tool.overlap == ToolOverlap.ALWAYS:
                dependent_tools[tool_id] = (_tool, guidelines)
            elif _tool.overlap == ToolOverlap.AUTO and tool_id not in visited:
                overlapped = await collect_overlapping_tools(tool_id)
                if overlapped:
                    overlapped.append((tool_id, _tool, guidelines))
                    overlapping_tools_batches.append(overlapped)
                else:
                    independent_tools[tool_id] = (_tool, guidelines)

        if independent_tools:
            context_without_reference_tools = ToolCallContext(
                agent=context.agent,
                session_id=context.session_id,
                customer_id=context.customer_id,
                context_variables=context.context_variables,
                interaction_history=context.interaction_history,
                terms=context.terms,
                ordinary_guideline_matches=list(
                    chain(
                        context.ordinary_guideline_matches,
                        context.tool_enabled_guideline_matches.keys(),
                    )
                ),
                journeys=context.journeys,
                tool_enabled_guideline_matches={},
                staged_events=context.staged_events,
            )
            result.extend(
                self._create_single_tool_batch(
                    candidate_tool=(k, v[0], v[1]), context=context_without_reference_tools
                )
                for k, v in independent_tools.items()
            )
        if dependent_tools:
            result.extend(
                self._create_single_tool_batch(candidate_tool=(k, v[0], v[1]), context=context)
                for k, v in dependent_tools.items()
            )

        if overlapping_tools_batches:
            result.extend(
                self._create_overlapping_tools_batch(overlapping_tools_batch=b, context=context)
                for b in overlapping_tools_batches
            )
        return result

    def _create_single_tool_batch(
        self,
        candidate_tool: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
        context: ToolCallContext,
    ) -> ToolCallBatch:
        return SingleToolBatch(
            logger=self._logger,
            optimization_policy=self._optimization_policy,
            service_registry=self._service_registry,
            schematic_generator=self._single_tool_schematic_generator,
            candidate_tool=candidate_tool,
            context=context,
        )

    def _create_overlapping_tools_batch(
        self,
        overlapping_tools_batch: Sequence[tuple[ToolId, Tool, Sequence[GuidelineMatch]]],
        context: ToolCallContext,
    ) -> ToolCallBatch:
        return OverlappingToolsBatch(
            logger=self._logger,
            optimization_policy=self._optimization_policy,
            service_registry=self._service_registry,
            schematic_generator=self._overlapping_tools_schematic_generator,
            overlapping_tools_batch=overlapping_tools_batch,
            context=context,
        )
