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

from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.guideline_continuous_proposer import GuidelineContinuousProposer


async def test_that_non_continuous_guidelines_mark_as_non_continuous(
    container: Container,
) -> None:
    continuous_proposer = container[GuidelineContinuousProposer]

    guidelines = [
        GuidelineContent(
            condition="The customer asks about vegetarian options",
            action="list all vegetarian pizza options",
        ),
        GuidelineContent(
            condition="The customer requests a custom pizza",
            action="Guide the customer through choosing base, sauce, toppings, and cheese",
        ),
        GuidelineContent(
            condition="The customer wants to repeat a previous order",
            action="The customer wants to repeat a previous order",
        ),
        GuidelineContent(
            condition="The customer wants to modify an order",
            action="Assist in making the desired changes and confirm the new order details",
        ),
        # GuidelineContent(
        #     condition="The user mentions a hobby",
        #     action="Show interest and encourage them to share more about it",
        # ),
        GuidelineContent(
            condition="A user reports an error during account setup.",
            action="Apologize for the inconvenience and confirm the report receipt.",
        ),
        GuidelineContent(
            condition="The customer is navigating through a troubleshooting guide for a product malfunction.",
            action="Provide step-by-step assistance without rushing, ensuring understanding at each step.",
        ),
    ]

    tasks = [continuous_proposer.propose_continuous(guideline=g) for g in guidelines]

    results = await asyncio.gather(*tasks)

    for g, result in zip(guidelines, results):
        assert not result.is_continuous, (
            f"Guideline failed to be marked as non continuous:\n"
            f"Condition: {g.condition}\n"
            f"Action: {g.action}"
        )


async def test_that_continuous_guidelines_mark_as_continuous(
    container: Container,
) -> None:
    continuous_proposer = container[GuidelineContinuousProposer]

    guidelines = [
        GuidelineContent(
            condition="The customer is above 60",
            action="Use language that is simple and not overly technical",
        ),
        GuidelineContent(
            condition="The user is showing signs of frustration",
            action="Tell them it's going to be ok and respond with empathy and provide supportive assistance",
        ),
        GuidelineContent(
            condition="The user mentions they have dietary restrictions.",
            action="Ensure all food recommendations consider the user's dietary needs throughout the conversation.",
        ),
        GuidelineContent(
            condition="The user starts discussing a complex technical issue.",
            action="Use simple and clear language to explain solutions",
        ),
        GuidelineContent(
            condition="The user is browsing items on a multilingual website.",
            action="Communicate in the user's preferred language.",
        ),
        GuidelineContent(
            condition="The customer expresses urgency in their requests.",
            action="Prioritize their needs and respond promptly.",
        ),
        GuidelineContent(
            condition="The user indicates they have dietary restrictions while discussing meal options.",
            action="Ensure that all suggested meal options respect their dietary restrictions.",
        ),
        GuidelineContent(
            condition="The user wants to replace their current meal with a healthier option.",
            action="Suggest healthier alternatives and then assist the user in replacing their meal choice until they are satisfied",
        ),
    ]

    tasks = [continuous_proposer.propose_continuous(guideline=g) for g in guidelines]

    results = await asyncio.gather(*tasks)

    for g, result in zip(guidelines, results):
        assert result.is_continuous, (
            f"Guideline failed to be marked as continuous:\n"
            f"Condition: {g.condition}\n"
            f"Action: {g.action}"
        )
