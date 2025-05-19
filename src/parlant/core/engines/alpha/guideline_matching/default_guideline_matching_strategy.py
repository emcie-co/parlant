import math
from typing import Sequence
from typing_extensions import override


from parlant.core.engines.alpha.guideline_matching.generic_guideline_not_previously_applied_batch import (
    GenericNotPreviouslyAppliedGuidelineMatchesSchema,
    GenericNotPreviouslyAppliedGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic_guideline_previously_applied_batch import (
    GenericPreviouslyAppliedGuidelineMatchesSchema,
    GenericPreviouslyAppliedGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic_observational_batch import (
    GenericObservationalGuidelineMatchesSchema,
    GenericObservationalGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingContext,
    GuidelineMatchingStrategy,
    GuidelineMatchingStrategyResolver,
)
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.tags import TagId


class GenericGuidelineMatchingStrategy(GuidelineMatchingStrategy):
    def __init__(
        self,
        logger: Logger,
        observational_guideline_schematic_generator: SchematicGenerator[
            GenericObservationalGuidelineMatchesSchema
        ],
        previously_applied_guideline_schematic_generator: SchematicGenerator[
            GenericPreviouslyAppliedGuidelineMatchesSchema
        ],
        not_previously_applied_guideline_schematic_generator: SchematicGenerator[
            GenericNotPreviouslyAppliedGuidelineMatchesSchema
        ],
    ) -> None:
        self._logger = logger
        self._observational_guideline_schematic_generator = (
            observational_guideline_schematic_generator
        )
        self._not_previously_applied_guideline_schematic_generator = (
            not_previously_applied_guideline_schematic_generator
        )
        self._previously_applied_guideline_schematic_generator = (
            previously_applied_guideline_schematic_generator
        )

    @override
    async def create_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        observational_batch: list[Guideline] = []
        previously_applied_batch: list[Guideline] = []
        not_previously_applied: list[Guideline] = []
        for g in guidelines:
            if not g.content.action:
                observational_batch.append(g)
            else:
                if g.metadata.get("continuous", False):
                    not_previously_applied.append(g)
                else:
                    if g.id in context.session.agent_state.applied_guideline_ids:
                        previously_applied_batch.append(g)
                    else:
                        not_previously_applied.append(g)

        guideline_batches: list[GuidelineMatchingBatch] = []
        if observational_batch:
            guideline_batches.extend(
                self._create_sub_batches_observational_guideline(observational_batch, context)
            )
        if previously_applied_batch:
            guideline_batches.extend(
                self._create_sub_batches_previously_applied_guideline(observational_batch, context)
            )
        if not_previously_applied:
            guideline_batches.extend(
                self._create_sub_batches_not_previously_applied_guideline(
                    observational_batch, context
                )
            )
        return guideline_batches

    async def _create_sub_batches_observational_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_sub_batch_observational_guideline(
                    guidelines=list(batch.values()),
                    context=context,
                )
            )

        return batches

    def _create_sub_batch_observational_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> GenericObservationalGuidelineMatchingBatch:
        return GenericObservationalGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._observational_guideline_schematic_generator,
            guidelines=guidelines,
            context=context,
        )

    async def _create_sub_batches_previously_applied_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_sub_batch_previously_applied_guideline(
                    guidelines=list(batch.values()),
                    context=context,
                )
            )

        return batches

    def _create_sub_batch_previously_applied_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> GenericPreviouslyAppliedGuidelineMatchingBatch:
        return GenericPreviouslyAppliedGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._previously_applied_guideline_schematic_generator,
            guidelines=guidelines,
            context=context,
        )

    async def _create_sub_batches_not_previously_applied_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_sub_batch_previously_applied_guideline(
                    guidelines=list(batch.values()),
                    context=context,
                )
            )

        return batches

    def _create_sub_batch_not_previously_applied_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> GenericNotPreviouslyAppliedGuidelineMatchingBatch:
        return GenericNotPreviouslyAppliedGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._not_previously_applied_guideline_schematic_generator,
            guidelines=guidelines,
            context=context,
        )

    def _get_optimal_batch_size(self, guidelines: dict[GuidelineId, Guideline]) -> int:
        guideline_n = len(guidelines)

        if guideline_n <= 10:
            return 1
        elif guideline_n <= 20:
            return 2
        elif guideline_n <= 30:
            return 3
        else:
            return 5


class DefaultGuidelineMatchingStrategyResolver(GuidelineMatchingStrategyResolver):
    def __init__(
        self,
        generic_strategy: GenericGuidelineMatchingStrategy,
        logger: Logger,
    ) -> None:
        self._generic_strategy = generic_strategy
        self._logger = logger

        self.guideline_overrides: dict[GuidelineId, GuidelineMatchingStrategy] = {}
        self.tag_overrides: dict[TagId, GuidelineMatchingStrategy] = {}

    @override
    async def resolve(self, guideline: Guideline) -> GuidelineMatchingStrategy:
        if override_strategy := self.guideline_overrides.get(guideline.id):
            return override_strategy

        tag_strategies = [s for tag_id, s in self.tag_overrides.items() if tag_id in guideline.tags]

        if first_tag_strategy := next(iter(tag_strategies), None):
            if len(tag_strategies) > 1:
                self._logger.warning(
                    f"More than one tag-based strategy override found for guideline (id='{guideline.id}'). Choosing first strategy ({first_tag_strategy.__class__.__name__})"
                )
            return first_tag_strategy

        return self._generic_strategy
