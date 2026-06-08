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
    create_capability,
)


@fixture
def ranker(container: Container) -> GuidelineRanker:
    return container[GuidelineRanker]


GUIDELINES_DICT: dict[str, dict[str, str]] = {
    "transfer_to_manager": {
        "condition": "When customer ask to talk with a manager",
        "action": "Hand them over to a manager immediately.",
    },
    "don't_transfer_to_manager": {
        "condition": "When customer ask to talk with a manager",
        "action": "Explain that it's not possible to talk with a manager and that you are here to help",
    },
    "first_order_and_order_more_than_2": {
        "condition": "When this is the customer first time ordering in the restaurant and the order they made includes more than 2 pizzas",
        "action": "offer 2 for 1 sale",
    },
    "first_order_and_order_exactly_2": {
        "condition": "When this is the customer first time ordering in the restaurant and the order they made includes exactly 2 pizzas",
        "action": "offer 2 for 1 sale",
    },
    "identify_problem": {
        "condition": "When customer say that they got an error or that something is not working",
        "action": "help them identify the source of the problem",
    },
    "frustrated_customer": {
        "condition": "the customer appears frustrated or upset",
        "action": "Acknowledge the customer's concerns, apologize for any inconvenience, and offer a solution or escalate the issue to a supervisor if necessary.",
    },
    "do_payment": {
        "condition": "the customer wants to pay for a product",
        "action": "Use the do_payment tool to process their payment.",
    },
    "problem_with_order": {
        "condition": "The customer is reporting a problem with their order.",
        "action": "Apologize and ask for more details about the issue.",
    },
    "cancel_subscription": {
        "condition": "When the user asks for help canceling a subscription.",
        "action": "Help them cancel it",
    },
    "ordering_sandwich": {
        "condition": "the customer wants to order a sandwich",
        "action": "only discuss options which are in stock",
    },
    "unsupported_capability": {
        "condition": "When a customer asks about a capability that is not supported",
        "action": "ask the customer for their age before proceeding",
    },
    "multiple_capabilities": {
        "condition": "When there are multiple capabilities that are relevant for the customer's request",
        "action": "ask the customer which of the capabilities they want to use",
    },
}


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_a_guideline_whose_condition_is_partially_satisfied_not_matched
async def test_that_a_guideline_whose_condition_is_partially_satisfied_not_matched(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, it's my first time here!",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to our pizza store! what would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "I want 2 pizzas please",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["first_order_and_order_more_than_2"],
        relevant_guideline_names=[],
        irrelevant_guideline_names=["first_order_and_order_more_than_2"],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_whose_condition_was_partially_fulfilled_now_matches
async def test_that_guideline_whose_condition_was_partially_fulfilled_now_matches(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, it's my first time here!",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to our pizza store! what would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "I want 2 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Cool so I will process your order right away. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I want another pizza please.",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["first_order_and_order_more_than_2"],
        relevant_guideline_names=["first_order_and_order_more_than_2"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches
async def test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, it's my first time here!",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to our pizza store! what would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "I want 3 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Cool so I will process your order right away. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I want 2 pizzas please.",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["first_order_and_order_exactly_2"],
        relevant_guideline_names=["first_order_and_order_exactly_2"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches_with_subtopic
async def test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches_with_subtopic(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, it's my first time here!",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to our pizza store! what would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "I want 3 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Cool so I will process your order right away. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "I went to this other pizza place and they had some great pizza/",
        ),
        (
            EventSource.AI_AGENT,
            "Happy to hear that! We also have some great pizzas here. Would you like anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I want 2 pizzas please.",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["first_order_and_order_exactly_2"],
        relevant_guideline_names=["first_order_and_order_exactly_2"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches_after_long_conversation
async def test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches_after_long_conversation(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, it's my first time here!",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to our pizza store! what would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "Can you tell me about your menu?",
        ),
        (
            EventSource.AI_AGENT,
            "Our menu includes a variety of pizzas, sandwiches, and drinks. What are you in the mood for?",
        ),
        (
            EventSource.CUSTOMER,
            "When was this place opened?",
        ),
        (
            EventSource.AI_AGENT,
            "We opened in 2020. Would you like to order something?",
        ),
        (EventSource.CUSTOMER, "Are you guys open on weekends?"),
        (EventSource.AI_AGENT, "Yes, we are open on weekends. What would you like to order?"),
        (
            EventSource.CUSTOMER,
            "I want 2 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Cool so I will process your order right away. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I want another pizza please.",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["first_order_and_order_more_than_2"],
        relevant_guideline_names=["first_order_and_order_more_than_2"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_conflicting_actions_with_similar_conditions_are_both_matched
async def test_that_conflicting_actions_with_similar_conditions_are_both_matched(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Look it's been over an hour and my problem was not solved. You are not helping and "
            "I want to talk with a manager immediately!",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["transfer_to_manager", "don't_transfer_to_manager"],
        relevant_guideline_names=["transfer_to_manager", "don't_transfer_to_manager"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_with_already_applied_condition_but_unaddressed_action_is_not_matched_when_conversation_was_drifted
async def test_that_guideline_with_already_applied_condition_but_unaddressed_action_is_not_matched_when_conversation_was_drifted(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            " Hi, can you help me cancel my subscription?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, I can walk you through the process. Are you using the mobile app or the website?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually, before that — how do I change my billing address?",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["cancel_subscription"],
        relevant_guideline_names=[],
        irrelevant_guideline_names=["cancel_subscription"],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_with_already_applied_condition_but_unaddressed_action_is_not_matched_when_conversation_was_drifted_2
async def test_that_guideline_with_already_applied_condition_but_unaddressed_action_is_not_matched_when_conversation_was_drifted_2(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, the app keeps crashing on my phone.",
        ),
        (
            EventSource.AI_AGENT,
            "Sorry to hear that! Can you tell me a bit more about what you were doing when it crashed?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure, but can you help me back up my data first?",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["identify_problem"],
        relevant_guideline_names=[],
        irrelevant_guideline_names=["identify_problem"],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_with_already_matched_condition_but_unaddressed_action_is_matched
async def test_that_guideline_with_already_matched_condition_but_unaddressed_action_is_matched(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey there, can I get one cheese pizza?"),
        (
            EventSource.AI_AGENT,
            "No, we don't have those",
        ),
        (
            EventSource.CUSTOMER,
            "I thought you're a pizza shop, this is very frustrating",
        ),
        (
            EventSource.AI_AGENT,
            "I don't know what to tell you, we're out ingredients at this time",
        ),
        (
            EventSource.CUSTOMER,
            "What the heck! I'm never ordering from you guys again",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["frustrated_customer"],
        relevant_guideline_names=["frustrated_customer"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_is_still_matched_when_conversation_still_on_the_same_topic_that_made_condition_hold
async def test_that_guideline_is_still_matched_when_conversation_still_on_the_same_topic_that_made_condition_hold(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey can I order 2 cheese pizzas please?"),
        (
            EventSource.AI_AGENT,
            "Sure! would you like a drink with that?",
        ),
        (
            EventSource.CUSTOMER,
            "No, thanks. How can I pay?",
        ),
        (
            EventSource.AI_AGENT,
            "It will cost $20.9. Could you please provide your credit card number?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure, it's 1111 2222 3333 4444.",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["do_payment"],
        relevant_guideline_names=["do_payment"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold
async def test_that_guideline_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hi, I just received my order, and the pizza is cold."),
        (
            EventSource.AI_AGENT,
            "I'm so sorry to hear that. Could you tell me more about the issue?",
        ),
        (EventSource.CUSTOMER, "Yeah, it's not just cold — the box was crushed too."),
        (EventSource.AI_AGENT, "That's really unacceptable. Let me make this right."),
        (EventSource.CUSTOMER, "And this isn’t the first time, honestly."),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["problem_with_order"],
        relevant_guideline_names=["problem_with_order"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_guideline_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold_2
async def test_that_guideline_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold_2(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I wanted to order a sandwich",
        ),
        (
            EventSource.AI_AGENT,
            "Hello there! We currently have either PB&J or cream cheese, which one would you like",
        ),
        (EventSource.CUSTOMER, "What's lower on calories, PB&J or cream cheese?"),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["ordering_sandwich"],
        relevant_guideline_names=["ordering_sandwich"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_previously_applied_guidelines_are_matched_based_on_capabilities
async def test_that_previously_applied_guidelines_are_matched_based_on_capabilities(
    ranker: GuidelineRanker,
) -> None:
    capabilities = [
        create_capability(
            "Reset Password",
            "The ability to send the customer an email with a link to reset their password. The password can only be reset via this link",
            id="cap_123",
            signals=["reset password", "password"],
            tags=[],
        )
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Set my password to 1234",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["unsupported_capability"],
        relevant_guideline_names=["unsupported_capability"],
        irrelevant_guideline_names=[],
        capabilities=capabilities,
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_actionable_batch.py::test_that_previously_applied_guidelines_are_not_matched_based_on_irrelevant_capabilities
async def test_that_previously_applied_guidelines_are_not_matched_based_on_irrelevant_capabilities(
    ranker: GuidelineRanker,
) -> None:
    capabilities = [
        create_capability(
            "Reset Password",
            "The ability to send the customer an email with a link to reset their password. The password can only be reset via this link",
            id="cap_123",
            signals=["reset password", "password"],
            tags=[],
        )
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I want to reset my password",
        ),
    ]
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=conversation,
        conversation_guideline_names=["unsupported_capability", "multiple_capabilities"],
        relevant_guideline_names=[],
        irrelevant_guideline_names=["unsupported_capability", "multiple_capabilities"],
        capabilities=capabilities,
    )
