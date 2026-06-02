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

from lagom import Container
from pytest import fixture

from parlant.core.engines.sigma.guideline_matching.guideline_ranker import GuidelineRanker
from parlant.core.sessions import EventSource

from tests.core.stable.engines.sigma.guideline_matching.utils import (
    base_test_that_guidelines_are_ranked_correctly,
)


@fixture
def ranker(container: Container) -> GuidelineRanker:
    return container[GuidelineRanker]


GUIDELINES_DICT: dict[str, dict[str, str]] = {
    "ask_toppings": {
        "condition": "the customer asks about toppings",
        "action": "list the available toppings",
    },
}


def test_that_a_guideline_ranker_can_be_created(ranker: GuidelineRanker) -> None:
    assert ranker is not None


async def test_that_a_relevant_guideline_is_ranked_as_relevant(ranker: GuidelineRanker) -> None:
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")],
        conversation_guideline_names=["ask_toppings"],
        relevant_guideline_names=["ask_toppings"],
        irrelevant_guideline_names=[],
    )
