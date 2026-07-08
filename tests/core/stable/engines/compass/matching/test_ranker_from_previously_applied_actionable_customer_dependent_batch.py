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

from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.matching.utils import (
    base_test_that_rules_are_ranked_correctly,
    create_capability,
)


@fixture
def ranker(container: Container) -> RuleRanker:
    return container[RuleRanker]


RULES_DICT: dict[str, dict[str, str]] = {
    "reservation_location": {
        "condition": "customer wants to make a reservation",
        "action": "check if they prefer inside or outside",
    },
    "issue_reporting": {
        "condition": "The customer is reporting a technical issue",
        "action": "Ask for the exact error message or steps to reproduce the issue",
    },
    "order_lookup": {
        "condition": "The customer wants to check their order status",
        "action": "Ask for their order number",
    },
    "order_alcohol": {
        "condition": "The customer wants to order alcohol",
        "action": "Check their age",
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


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_matched_when_customer_hasnt_completed_their_side
async def test_that_customer_dependent_rule_is_matched_when_customer_hasnt_completed_their_side(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I’d like to book a table for tomorrow night.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Would you prefer to sit inside or outside?",
        ),
        (
            EventSource.CUSTOMER,
            "7 PM would be great.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["reservation_location"],
        relevant_rule_names=["reservation_location"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_not_matched_when_customer_has_completed_their_side
async def test_that_customer_dependent_rule_is_not_matched_when_customer_has_completed_their_side(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I’d like to book a table for tomorrow night.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Would you prefer to sit inside or outside?",
        ),
        (
            EventSource.CUSTOMER,
            "I prefer it outside, thanks",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["reservation_location"],
        relevant_rule_names=[],
        irrelevant_rule_names=["reservation_location"],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_matched_when_customer_hasnt_completed_their_side_over_several_messages
async def test_that_customer_dependent_rule_is_matched_when_customer_hasnt_completed_their_side_over_several_messages(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I’d like to book a table for tomorrow night.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Would you prefer to sit inside or outside?",
        ),
        (
            EventSource.CUSTOMER,
            "Tomorrow at 7 PM would be great.",
        ),
        (
            EventSource.AI_AGENT,
            "Great, I’ve noted 7 PM. Do you have a seating preference?",
        ),
        (
            EventSource.CUSTOMER,
            "And can it be a quiet table if possible?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["reservation_location"],
        relevant_rule_names=["reservation_location"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_not_matched_when_customer_hasnt_completed_their_side_but_change_subject
async def test_that_customer_dependent_rule_is_not_matched_when_customer_hasnt_completed_their_side_but_change_subject(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Your app keeps crashing when I try to open it.",
        ),
        (
            EventSource.AI_AGENT,
            "I’m sorry to hear that! Could you tell me the exact error message you’re seeing?",
        ),
        (
            EventSource.CUSTOMER,
            "Anyway, I was also wondering if you have any discounts available right now?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["issue_reporting"],
        relevant_rule_names=[],
        irrelevant_rule_names=["issue_reporting"],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_matched_when_customer_hasnt_completed_their_side_on_the_second_time
async def test_that_customer_dependent_rule_is_matched_when_customer_hasnt_completed_their_side_on_the_second_time(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you check the status of my phone order?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Could you share the order number?",
        ),
        (
            EventSource.CUSTOMER,
            "It’s 12345. Thanks.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. It's on the way and should arrive by Thursday.",
        ),
        (
            EventSource.CUSTOMER,
            "Great. What about the headphones I ordered last week?",
        ),
        (
            EventSource.AI_AGENT,
            "I'll check right now. Whats the order number for them?",
        ),
        (
            EventSource.CUSTOMER,
            "I need to check just a second",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["order_lookup"],
        relevant_rule_names=["order_lookup"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_matched_when_condition_arises_for_the_second_time
async def test_that_customer_dependent_rule_is_matched_when_condition_arises_for_the_second_time(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you check the status of my phone order?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Could you share the order number?",
        ),
        (
            EventSource.CUSTOMER,
            "It’s 12345. Thanks.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. It's on the way and should arrive by Thursday.",
        ),
        (
            EventSource.CUSTOMER,
            "Great. What about the headphones I ordered last week?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["order_lookup"],
        relevant_rule_names=["order_lookup"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_not_matched_when_condition_arises_for_the_second_time_but_completed
async def test_that_customer_dependent_rule_is_not_matched_when_condition_arises_for_the_second_time_but_completed(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you check the status of my phone order?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Could you share the order number?",
        ),
        (
            EventSource.CUSTOMER,
            "It’s 12345. Thanks.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. It's on the way and should arrive by Thursday.",
        ),
        (
            EventSource.CUSTOMER,
            "Great. What about the headphones I ordered last week?",
        ),
        (
            EventSource.AI_AGENT,
            "I'll check right now. Whats the order number for them?",
        ),
        (
            EventSource.CUSTOMER,
            "It’s 11122.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["order_lookup"],
        relevant_rule_names=[],
        irrelevant_rule_names=["order_lookup"],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_not_matched_when_condition_arises_for_the_second_time_but_dont_need_to_take_the_action_again
async def test_that_customer_dependent_rule_is_not_matched_when_condition_arises_for_the_second_time_but_dont_need_to_take_the_action_again(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi can I get 2 beers?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, but first, may I ask your age?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm 25 thank God!",
        ),
        (
            EventSource.AI_AGENT,
            "Perfect — I’ve added 2 beers to your order. Would you like anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, I'd also like some wine, please.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["order_alcohol"],
        relevant_rule_names=[],
        irrelevant_rule_names=["order_alcohol"],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_matched_based_on_capabilities_1
async def test_that_customer_dependent_rule_is_matched_based_on_capabilities_1(
    ranker: RuleRanker,
) -> None:
    capabilities = [
        create_capability(
            "Reset Password",
            "The ability to send the customer an email with a link to reset their password. The password can only be reset via this link",
            id="cap_123",
            signals=["reset password", "password"],
            groups=[],
        )
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Teach me how to tame dinosaurs",
        ),
        (
            EventSource.AI_AGENT,
            "Before proceeding, may I ask for your age?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure! But can you help me get ice cream first?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["unsupported_capability", "multiple_capabilities"],
        relevant_rule_names=["unsupported_capability"],
        irrelevant_rule_names=["multiple_capabilities"],
        capabilities=capabilities,
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_customer_dependent_batch.py::test_that_customer_dependent_rule_is_matched_based_on_capabilities_2
async def test_that_customer_dependent_rule_is_matched_based_on_capabilities_2(
    ranker: RuleRanker,
) -> None:
    capabilities = [
        create_capability(
            "Increase Credit Limit",
            "The ability to increase the customer's credit limit",
            id="cap_123",
            signals=["increase credit limit", "credit limit"],
            groups=[],
        ),
        create_capability(
            "Decrease Credit Limit",
            "The ability to decrease the customer's credit limit",
            id="cap_123",
            signals=["decrease credit limit", "credit limit"],
            groups=[],
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you help me change my credit limits",
        ),
        (
            EventSource.AI_AGENT,
            "I can help you either increase or decrease your credit limit. Which option are you interested in?",
        ),
        (
            EventSource.CUSTOMER,
            "I just want to change them...",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["unsupported_capability", "multiple_capabilities"],
        relevant_rule_names=["multiple_capabilities"],
        irrelevant_rule_names=["unsupported_capability"],
        capabilities=capabilities,
    )
