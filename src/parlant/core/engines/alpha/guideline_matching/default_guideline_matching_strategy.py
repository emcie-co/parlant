from typing_extensions import override

from parlant.core.engines.alpha.guideline_matching.generic_actionable_batch import (
    GenericActionableGuidelineMatching,
)
from parlant.core.engines.alpha.guideline_matching.generic_observational_batch import (
    GenericObservationalGuidelineMatching,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingStrategy,
    GuidelineMatchingStrategyResolver,
)
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.tags import TagId


class DefaultGuidelineMatchingStrategyResolver(GuidelineMatchingStrategyResolver):
    def __init__(
        self,
        generic_actionable_strategy: GenericActionableGuidelineMatching,
        generic_observational_strategy: GenericObservationalGuidelineMatching,
        logger: Logger,
    ) -> None:
        self._generic_actionable_strategy = generic_actionable_strategy
        self._generic_observational_strategy = generic_observational_strategy
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

        if guideline.content.action:
            return self._generic_actionable_strategy
        else:
            return self._generic_observational_strategy
