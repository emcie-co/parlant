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

from parlant.core.engines.compass.guideline_matching.guideline_ranker import GuidelineRanker
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.guideline_matching.utils import (
    base_test_that_guidelines_are_ranked_correctly,
)


@fixture
def ranker(container: Container) -> GuidelineRanker:
    return container[GuidelineRanker]


GUIDELINES_DICT: dict[str, dict[str, str]] = {
    "reset_password": {
        "condition": "When a customer wants to reset their password",
        "action": "ask for their email address to send them a password",
    },
}


# Taken from tests/core/unstable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_partially_fulfilled_action_with_missing_behavioral_part_is_matched_again
async def test_that_partially_fulfilled_action_with_missing_behavioral_part_is_matched_again(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, can you reset my password?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, for that I will need your email please so I will send you the password. What's your email address?",
        ),
        (
            EventSource.CUSTOMER,
            "I forgot what I was going to say, can you continue from the same point?",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["reset_password"],
        relevant_guideline_names=["reset_password"],
        irrelevant_guideline_names=[],
    )
