import asyncio
from collections.abc import Sequence
from enum import Enum
from typing import Any, cast
from typing_extensions import override
from parlant.core import async_utils

from parlant.core.common import JSONSerializable
from parlant.core.engines.alpha.guideline_matching.common import measure_guideline_matching_batch

from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_check import (
    JourneyBacktrackCheck,
    JourneyBacktrackCheckSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_node_selection import (
    JourneyBacktrackNodeSelection,
    JourneyBacktrackNodeSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_next_step_selection import (
    JourneyNextStepSelection,
    JourneyNextStepSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchResult,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matching_context import (
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo


class JourneyNodeKind(Enum):
    FORK = "fork"
    CHAT = "chat"
    TOOL = "tool"
    NA = "NA"


ROOT_INDEX = "1"


class GenericJourneyNodeSelectionBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        meter: Meter,
        guideline_store: GuidelineStore,
        optimization_policy: OptimizationPolicy,
        schematic_generator_journey_node_selection: SchematicGenerator[
            JourneyBacktrackNodeSelectionSchema
        ],
        schematic_generator_next_step_selection: SchematicGenerator[JourneyNextStepSelectionSchema],
        schematic_generator_journey_backtrack_check: SchematicGenerator[
            JourneyBacktrackCheckSchema
        ],
        examined_journey: Journey,
        context: GuidelineMatchingContext,
        node_guidelines: Sequence[Guideline] = [],
        journey_path: Sequence[str | None] = [],
    ) -> None:
        self._logger = logger
        self._meter = meter

        self._guideline_store = guideline_store

        self._optimization_policy = optimization_policy
        self._schematic_generator_journey_node_selection = (
            schematic_generator_journey_node_selection
        )
        self._schematic_generator_next_step_selection = schematic_generator_next_step_selection
        self._schematic_generator_journey_backtrack_check = (
            schematic_generator_journey_backtrack_check
        )
        self._context = context
        self._examined_journey = examined_journey
        self._node_guidelines = node_guidelines
        self._previous_path: Sequence[str | None] = journey_path

    @property
    @override
    def size(self) -> int:
        return 1

    def auto_return_match(self) -> GuidelineMatchingBatchResult | None:
        def _get_guideline_node_index(guideline: Guideline) -> str:
            return str(
                cast(dict[str, JSONSerializable], guideline.metadata["journey_node"]).get(
                    "index", "-1"
                ),
            )

        def _get_follow_ups(guideline: Guideline) -> Sequence[GuidelineId]:
            return cast(
                dict[str, Sequence[GuidelineId]],
                guideline.metadata.get("journey_node", {}),
            ).get("follow_ups", [])

        def _get_kind(guideline: Guideline) -> JourneyNodeKind:
            return JourneyNodeKind(
                cast(dict[str, Any], guideline.metadata.get("journey_node", {})).get("kind", "NA")
            )

        node_index_to_guideline: dict[str, Guideline] = {
            _get_guideline_node_index(g): g for g in self._node_guidelines
        }
        guideline_id_to_node_index: dict[GuidelineId, str] = {
            g.id: _get_guideline_node_index(g) for g in self._node_guidelines
        }
        guideline_id_to_guideline: dict[GuidelineId, Guideline] = {
            g.id: g for g in self._node_guidelines
        }
        root_guideline = next(
            g for g in self._node_guidelines if _get_guideline_node_index(g) == ROOT_INDEX
        )

        if self._previous_path and self._previous_path[-1]:
            last_visited_node_index = self._previous_path[-1]
            last_visited_guideline = node_index_to_guideline[last_visited_node_index]
            kind = _get_kind(last_visited_guideline)
            outgoing_edges = _get_follow_ups(last_visited_guideline)

            if kind == JourneyNodeKind.TOOL and len(outgoing_edges) == 1:
                generation_info = GenerationInfo(
                    schema_name="No inference performed",
                    model="No inference performed",
                    duration=0.0,
                    usage=UsageInfo(
                        input_tokens=0,
                        output_tokens=0,
                        extra={},
                    ),
                )
                current_node: GuidelineId = outgoing_edges[0]
                journey_path = list(self._previous_path) + [
                    _get_guideline_node_index(guideline_id_to_guideline[current_node])
                ]
                while (
                    current_node
                    and _get_kind(guideline_id_to_guideline[current_node]) == JourneyNodeKind.FORK
                ):
                    if len(_get_follow_ups(guideline_id_to_guideline[current_node])) != 1:
                        return None
                    current_node = GuidelineId(
                        _get_follow_ups(guideline_id_to_guideline[current_node])[0]
                    )
                    journey_path.append(guideline_id_to_node_index[current_node])

                if guideline_id_to_guideline[current_node]:
                    return GuidelineMatchingBatchResult(
                        matches=[
                            GuidelineMatch(
                                guideline=guideline_id_to_guideline[current_node],
                                score=10,
                                rationale="This guideline was selected as part of a 'journey' - a sequence of actions that are performed in order. It was automatically selected as the only viable follow up for the last step that was executed",
                                metadata={
                                    "journey_path": journey_path,
                                    "step_selection_journey_id": self._examined_journey.id,
                                },
                            )
                        ],
                        generation_info=generation_info,
                    )
                else:
                    return GuidelineMatchingBatchResult(
                        matches=[
                            GuidelineMatch(
                                guideline=root_guideline,
                                score=10,
                                rationale="Root guideline returned to indicate exit journey",
                                metadata={
                                    "journey_path": journey_path,
                                    "step_selection_journey_id": self._examined_journey.id,
                                },
                            )
                        ],
                        generation_info=generation_info,
                    )
        return None

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        automatic_match = self.auto_return_match()
        if automatic_match:
            return automatic_match  # TODO fix this

        journey_conditions = list(
            await async_utils.safe_gather(
                *[
                    self._guideline_store.read_guideline(c)
                    for c in self._examined_journey.conditions
                ]
            )
        )

        async with measure_guideline_matching_batch(self._meter, self):
            if not self._previous_path or all(p is None for p in self._previous_path):
                next_step_selector = JourneyNextStepSelection(
                    logger=self._logger,
                    guideline_store=self._guideline_store,
                    optimization_policy=self._optimization_policy,
                    schematic_generator=self._schematic_generator_next_step_selection,
                    examined_journey=self._examined_journey,
                    context=self._context,
                    node_guidelines=self._node_guidelines,
                    journey_path=[],
                    journey_conditions=journey_conditions,
                )
                return await next_step_selector.process()
            elif (
                self._previous_path
                and not all(p is None for p in self._previous_path)
                and self._previous_path[-1]
            ):
                backtrack_checker = JourneyBacktrackCheck(
                    logger=self._logger,
                    guideline_store=self._guideline_store,
                    optimization_policy=self._optimization_policy,
                    schematic_generator=self._schematic_generator_journey_backtrack_check,
                    examined_journey=self._examined_journey,
                    context=self._context,
                    node_guidelines=self._node_guidelines,
                    journey_path=self._previous_path,
                    journey_conditions=journey_conditions,
                )
                next_step_selector = JourneyNextStepSelection(
                    logger=self._logger,
                    guideline_store=self._guideline_store,
                    optimization_policy=self._optimization_policy,
                    schematic_generator=self._schematic_generator_next_step_selection,
                    examined_journey=self._examined_journey,
                    context=self._context,
                    node_guidelines=self._node_guidelines,
                    journey_path=self._previous_path,
                    journey_conditions=journey_conditions,
                )

                backtrack_task = asyncio.create_task(backtrack_checker.process())
                next_step_task = asyncio.create_task(next_step_selector.process())

                backtrack_result = await backtrack_task

                if backtrack_result.requires_backtracking:
                    next_step_task.cancel()
                    try:
                        await next_step_task
                    except asyncio.CancelledError:
                        pass

                    node_selector = JourneyBacktrackNodeSelection(
                        logger=self._logger,
                        guideline_store=self._guideline_store,
                        optimization_policy=self._optimization_policy,
                        schematic_generator=self._schematic_generator_journey_node_selection,
                        examined_journey=self._examined_journey,
                        context=self._context,
                        node_guidelines=self._node_guidelines,
                        journey_path=self._previous_path,
                        journey_conditions=journey_conditions,
                    )
                    return await node_selector.process()
                else:
                    return await next_step_task
            else:
                # run backtrack check unless need to backtrack
                backtrack_checker = JourneyBacktrackCheck(
                    logger=self._logger,
                    guideline_store=self._guideline_store,
                    optimization_policy=self._optimization_policy,
                    schematic_generator=self._schematic_generator_journey_backtrack_check,
                    examined_journey=self._examined_journey,
                    context=self._context,
                    node_guidelines=self._node_guidelines,
                    journey_path=self._previous_path,
                    journey_conditions=journey_conditions,
                )

                backtrack_result = await backtrack_checker.process()

                if not backtrack_result.requires_backtracking:
                    return GuidelineMatchingBatchResult(
                        matches=[], generation_info=backtrack_result.generation_info
                    )
                else:
                    if backtrack_result.backtrack_to_same_journey_process:
                        node_selector = JourneyBacktrackNodeSelection(
                            logger=self._logger,
                            guideline_store=self._guideline_store,
                            optimization_policy=self._optimization_policy,
                            schematic_generator=self._schematic_generator_journey_node_selection,
                            examined_journey=self._examined_journey,
                            context=self._context,
                            node_guidelines=self._node_guidelines,
                            journey_path=self._previous_path,
                            journey_conditions=journey_conditions,
                        )
                        return await node_selector.process()
                    else:
                        next_step_selector = JourneyNextStepSelection(
                            logger=self._logger,
                            guideline_store=self._guideline_store,
                            optimization_policy=self._optimization_policy,
                            schematic_generator=self._schematic_generator_next_step_selection,
                            examined_journey=self._examined_journey,
                            context=self._context,
                            node_guidelines=self._node_guidelines,
                            journey_path=self._previous_path,
                            journey_conditions=journey_conditions,
                        )
                        return await next_step_selector.process()
