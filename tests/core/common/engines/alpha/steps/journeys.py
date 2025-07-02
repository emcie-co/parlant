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

from pytest_bdd import given, parsers

from parlant.core.journeys import JourneyStore
from parlant.core.guidelines import Guideline, GuidelineStore

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(
    given,
    parsers.parse(
        'a journey titled "{journey_title}" to {journey_description} when {a_condition_holds}'
    ),
)
def given_a_journey_to_when(
    context: ContextOfTest,
    journey_title: str,
    journey_description: str,
    a_condition_holds: str,
) -> None:
    guideline_store = context.container[GuidelineStore]
    journey_store = context.container[JourneyStore]

    conditioning_guideline: Guideline = context.sync_await(
        guideline_store.create_guideline(condition=a_condition_holds, action=None)
    )

    journey = context.sync_await(
        journey_store.create_journey(
            conditions=[conditioning_guideline.id],
            title=journey_title,
            description=journey_description,
        )
    )

    context.journeys[journey.title] = journey
