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

from typing_extensions import override


from parlant.core.engines.alpha.guideline_matching.generic.generic_guideline_matching_strategy import (
    GenericGuidelineMatchingStrategy,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingStrategy,
    GuidelineMatchingStrategyResolver,
)
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.tags import TagId


class GenericGuidelineMatchingStrategyResolver(GuidelineMatchingStrategyResolver):
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
