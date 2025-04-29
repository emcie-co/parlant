from itertools import chain
from typing import Mapping, Sequence
from parlant.core.engines.alpha.guideline_match import GuidelineMatch
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
from parlant.core.relationships import RelationshipStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import Tool, ToolId, ToolOverlap


class DefaultToolCallBatcher(ToolCallBatcher):
    def __init__(
        self,
        logger: Logger,
        service_registry: ServiceRegistry,
        schematic_generator: SchematicGenerator[SingleToolBatchSchema],
        relationship_store: RelationshipStore,
    ):
        self._logger = logger
        self._service_registry = service_registry
        self._schematic_generator = schematic_generator
        self._relationship_store = relationship_store

    async def create_batches(
        self,
        tools: Mapping[tuple[ToolId, Tool], Sequence[GuidelineMatch]],
        context: ToolCallContext,
    ) -> Sequence[ToolCallBatch]:
        result: list[ToolCallBatch] = []

        independent_tools = {}
        dependent_tools = {}

        for tool_id, _tool in tools:
            if _tool.overlap == ToolOverlap.NONE:
                independent_tools[(tool_id, _tool)] = tools[(tool_id, _tool)]
            else:
                dependent_tools[(tool_id, _tool)] = tools[(tool_id, _tool)]

        if independent_tools:
            context_without_reference_tools = ToolCallContext(
                agent=context.agent,
                services=context.services,
                context_variables=context.context_variables,
                interaction_history=context.interaction_history,
                terms=context.terms,
                ordinary_guideline_matches=list(
                    chain(
                        context.ordinary_guideline_matches,
                        context.tool_enabled_guideline_matches.keys(),
                    )
                ),
                tool_enabled_guideline_matches={},
                staged_events=context.staged_events,
            )
            result.extend(
                self._create_single_tool_batch(
                    candidate_tool=(k[0], k[1], v), context=context_without_reference_tools
                )
                for k, v in independent_tools.items()
            )
        if dependent_tools:
            result.extend(
                self._create_single_tool_batch(candidate_tool=(k[0], k[1], v), context=context)
                for k, v in dependent_tools.items()
            )

        return result

    def _create_single_tool_batch(
        self,
        candidate_tool: tuple[ToolId, Tool, Sequence[GuidelineMatch]],
        context: ToolCallContext,
    ) -> ToolCallBatch:
        return SingleToolBatch(
            logger=self._logger,
            service_registry=self._service_registry,
            schematic_generator=self._schematic_generator,
            candidate_tool=candidate_tool,
            context=context,
        )
