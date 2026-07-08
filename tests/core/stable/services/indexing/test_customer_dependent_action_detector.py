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
from parlant.core.rules import RuleContent
from parlant.core.services.indexing.customer_dependent_action_detector import (
    CustomerDependentActionDetector,
)


async def check_rule(container: Container, rule: RuleContent, is_customer_dependent: bool) -> None:
    customer_dependent_action_detector = container[CustomerDependentActionDetector]
    result = await customer_dependent_action_detector.detect_if_customer_dependent(
        rule=rule,
    )
    assert (
        is_customer_dependent == result.is_customer_dependent
    ), f"""Rule incorrectly marked as {"not " if is_customer_dependent else ""}customer dependent:
Condition: {rule.condition}
Action: {rule.action}"""


async def test_that_actions_which_are_not_customer_dependent_are_classified_correctly(
    container: Container,
) -> None:
    rules = [
        RuleContent(
            condition="The customer asks about vegetarian options",
            action="list all vegetarian pizza options",
        ),
        RuleContent(
            condition="A user reports an error during account setup.",
            action="Apologize for the inconvenience and confirm the report receipt.",
        ),
        RuleContent(
            condition="The user is anxious",
            action="Finish your response with our slogan - 'are you ready for some fun???'",
        ),
        RuleContent(
            condition="the customer asks about job openings.",
            action="emphasize that we have plenty of positions relevant to the customer, and over 10,000 openings overall",
        ),
        RuleContent(
            condition="The customer asks you to ease the mood",
            action="inform the customer that this is a serious conversation",
        ),
        RuleContent(
            condition="The customer complains about slow service",
            action="apologize sincerely and explain that we are working to improve response times",
        ),
        RuleContent(
            condition="The customer asks about store hours",
            action="inform them that we are open Monday through Friday 9 AM to 6 PM",
        ),
        RuleContent(
            condition="The customer seems confused about our return policy",
            action="clearly explain our 30-day return policy and provide examples of eligible items",
        ),
    ]

    for g in rules:
        await check_rule(container=container, rule=g, is_customer_dependent=False)


async def test_that_actions_which_are_customer_dependent_are_classified_correctly(
    container: Container,
) -> None:
    rules = [
        RuleContent(
            condition="The customer orders alcohol",
            action="Get the customer's age",
        ),
        RuleContent(
            condition="the customer wants to book an appointment",
            action="Ask for the name of the person they want to meet and the time they want to meet them",
        ),
        RuleContent(
            condition="The customer speaks a language other than English",
            action="Ask the customer for their location",
        ),
        RuleContent(
            condition="The customer is navigating through a troubleshooting guide for a product malfunction.",
            action="Provide step-by-step assistance without rushing, ensuring understanding at each step.",
        ),
        RuleContent(
            condition="The customer asks you to ease the mood",
            action="Play tic-tac-toe with the customer, ensuring to play the optimal strategy, until you either win or the game draws",
        ),
        RuleContent(
            condition="The customer asks you to ease the mood",
            action="inform the customer that this is a serious conversation and ask them to tell a joke",
        ),
        RuleContent(
            condition="The customer wants to cancel their subscription",
            action="ask for their account email and the reason for cancellation",
        ),
        RuleContent(
            condition="The customer reports a billing issue",
            action="request their account number and ask them to describe the specific issue they're experiencing",
        ),
        RuleContent(
            condition="The customer wants to schedule a callback",
            action="ask for their preferred time and phone number, then confirm the appointment details",
        ),
    ]

    for g in rules:
        await check_rule(container=container, rule=g, is_customer_dependent=True)
