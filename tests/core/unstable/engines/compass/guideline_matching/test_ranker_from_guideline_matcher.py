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

from typing import cast

from lagom import Container
from pytest import fixture

from parlant.core.common import JSONSerializable
from parlant.core.engines.compass.guideline_matching.guideline_ranker import GuidelineRanker
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.guideline_matching.utils import (
    base_test_that_guidelines_are_ranked_correctly,
    create_context_variable,
    create_staged_tool_event,
    create_term,
)


@fixture
def ranker(container: Container) -> GuidelineRanker:
    return container[GuidelineRanker]


GUIDELINES_DICT: dict[str, dict[str, str]] = {
    "vegetarian_customer": {
        "condition": "the customer is vegetarian or vegan",
    },
    "lock_card_request_1": {
        "condition": "the customer indicated that they wish to lock their credit card",
    },
    "lock_card_request_2": {
        "condition": "the customer lost their credit card",
    },
    "season_is_winter": {
        "condition": "it is the season of winter",
    },
    "frustrated_customer_observational": {
        "condition": "the customer is frustrated",
    },
    "unclear_request": {
        "condition": "the customer indicates that the agent does not understand their request",
    },
    "credit_limits_discussion": {
        "condition": "credit limits are discussed",
    },
    "unknown_service": {
        "condition": "The customer is asking for a service you have no information about within this prompt",
    },
    "delivery_order": {
        "condition": "the customer is in the process of ordering delivery",
    },
    "unanswered_questions": {
        "condition": "the customer repeatedly ignores the agent's question, and they remain unanswered",
    },
}


# Taken from tests/core/unstable/engines/alpha/test_guideline_matcher.py::test_that_observational_guidelines_are_detected_correctly_when_lots_of_data_is_available
async def test_that_observational_guidelines_are_detected_correctly_when_lots_of_data_is_available(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I think I lost my credit card somewhere yesterday.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear about your lost card. I can help you with that right away. For security purposes, could you please verify the last four digits of your card?",
        ),
        (
            EventSource.CUSTOMER,
            "It's the Platinum Rewards one.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for that information. To proceed with locking your Platinum Rewards card, I'll need to verify the last four digits of the card number. This helps ensure we're securing the correct card on your account. Could you please provide those digits?",
        ),
        (
            EventSource.CUSTOMER,
            "blorgnet, I just need it locked right now. I have multiple cards and I don't memorize all the numbers.",
        ),
        (
            EventSource.AI_AGENT,
            "I understand your concern and the urgency of the situation. Let me try another approach. I can see your account has two Platinum Rewards cards. Would it be helpful if I locked both cards temporarily until we identify which one is missing? Or would you prefer to check your online banking app where you can see the last four digits of your cards?",
        ),
        (
            EventSource.CUSTOMER,
            "Just lock all of them. I need to go to a meeting in 5 minutes.",
        ),
        (
            EventSource.AI_AGENT,
            "I've placed temporary locks on all your Platinum Rewards cards. This will prevent any unauthorized transactions while keeping your account active. The locks are effective immediately. Would you like me to order replacement cards for you now, or would you prefer to wait in case you find the missing card?",
        ),
        (
            EventSource.CUSTOMER,
            "When will your branch on Main Street be open?",
        ),
        (
            EventSource.AI_AGENT,
            "Our Main Street branch is open Monday through Friday from 9:00 AM to 5:00 PM, and Saturday from 9:00 AM to 1:00 PM. It's closed on Sundays. Regarding your cards, would you like me to order replacements for any of them at this time?",
        ),
        (
            EventSource.CUSTOMER,
            "And do they handle international wire transfers there?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes, our Main Street branch does handle international wire transfers. You would need to bring a valid ID and the recipient's banking details. The branch can also help with your card situation if you prefer to handle it in person. In the meantime, your cards remain locked for your protection. Is there anything else you need assistance with today?",
        ),
        (
            EventSource.CUSTOMER,
            "What's the maximum amount for international transfers?",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=[
            "vegetarian_customer",
            "lock_card_request_1",
            "lock_card_request_2",
            "season_is_winter",
            "frustrated_customer_observational",
            "unclear_request",
            "credit_limits_discussion",
            "unknown_service",
            "delivery_order",
            "unanswered_questions",
        ],
        relevant_guideline_names=[
            "lock_card_request_1",
            "lock_card_request_2",
            "season_is_winter",
            "frustrated_customer_observational",
            "unanswered_questions",
        ],
        irrelevant_guideline_names=[
            "vegetarian_customer",
            "unclear_request",
            "credit_limits_discussion",
            "unknown_service",
            "delivery_order",
        ],
        terms=[
            create_term(
                name="blorgnet",
                description="a figure of speech, meaning being annoyed by whoever you're interacting with",
            ),
        ],
        context_variables=[
            create_context_variable(
                name="customer_location",
                data={"location": "Australia"},
            ),
            create_context_variable(
                name="date",
                data={"date": "August 15th, 2024"},
            ),
        ],
        staged_events=[
            create_staged_tool_event(
                cast(
                    JSONSerializable,
                    {
                        "tool_calls": [
                            {
                                "tool_id": "local:get_season",
                                "arguments": {},
                                "result": {
                                    "data": "winter",
                                    "metadata": {},
                                    "control": {},
                                },
                            }
                        ]
                    },
                )
            ),
        ],
    )
