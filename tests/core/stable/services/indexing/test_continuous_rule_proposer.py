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

import asyncio
from lagom import Container

from parlant.core.rules import RuleContent
from parlant.core.services.indexing.rule_continuous_proposer import RuleContinuousProposer


async def test_that_non_continuous_rules_mark_as_non_continuous(
    container: Container,
) -> None:
    continuous_proposer = container[RuleContinuousProposer]

    rules = [
        RuleContent(
            condition="The customer asks about vegetarian options",
            action="list all vegetarian pizza options",
        ),
        RuleContent(
            condition="The customer requests a custom pizza",
            action="Guide the customer through choosing base, sauce, toppings, and cheese",
        ),
        RuleContent(
            condition="The customer wants to repeat a previous order",
            action="The customer wants to repeat a previous order",
        ),
        RuleContent(
            condition="The customer wants to modify an order",
            action="Assist in making the desired changes and confirm the new order details",
        ),
        # RuleContent(
        #     condition="The user mentions a hobby",
        #     action="Show interest and encourage them to share more about it",
        # ),
        RuleContent(
            condition="A user reports an error during account setup.",
            action="Apologize for the inconvenience and confirm the report receipt.",
        ),
        RuleContent(
            condition="The customer is navigating through a troubleshooting guide for a product malfunction.",
            action="Provide step-by-step assistance without rushing, ensuring understanding at each step.",
        ),
    ]

    tasks = [continuous_proposer.propose_continuous(rule=g) for g in rules]

    results = await asyncio.gather(*tasks)

    for g, result in zip(rules, results):
        assert not result.is_continuous, (
            f"Rule failed to be marked as non continuous:\n"
            f"Condition: {g.condition}\n"
            f"Action: {g.action}"
        )


async def test_that_continuous_rules_mark_as_continuous(
    container: Container,
) -> None:
    continuous_proposer = container[RuleContinuousProposer]

    rules = [
        RuleContent(
            condition="The customer is above 60",
            action="Use language that is simple and not overly technical",
        ),
        RuleContent(
            condition="The user is showing signs of frustration",
            action="Tell them it's going to be ok and respond with empathy and provide supportive assistance",
        ),
        RuleContent(
            condition="The user mentions they have dietary restrictions.",
            action="Ensure all food recommendations consider the user's dietary needs throughout the conversation.",
        ),
        RuleContent(
            condition="The user starts discussing a complex technical issue.",
            action="Use simple and clear language to explain solutions",
        ),
        RuleContent(
            condition="The user is browsing items on a multilingual website.",
            action="Communicate in the user's preferred language.",
        ),
        RuleContent(
            condition="The customer expresses urgency in their requests.",
            action="Prioritize their needs and respond promptly.",
        ),
        RuleContent(
            condition="The user indicates they have dietary restrictions while discussing meal options.",
            action="Ensure that all suggested meal options respect their dietary restrictions.",
        ),
    ]

    tasks = [continuous_proposer.propose_continuous(rule=g) for g in rules]

    results = await asyncio.gather(*tasks)

    for g, result in zip(rules, results):
        assert result.is_continuous, (
            f"Rule failed to be marked as continuous:\nCondition: {g.condition}\nAction: {g.action}"
        )
