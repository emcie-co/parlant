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

from parlant.core.container import Container
from pytest import fixture

from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.matching.utils import (
    base_test_that_rules_are_ranked_correctly,
)


@fixture
def ranker(container: Container) -> RuleRanker:
    return container[RuleRanker]


RULES_DICT: dict[str, dict[str, str]] = {
    "problem_so_restart": {
        "condition": "The customer has a problem with the app and hasn't tried anything yet",
        "action": "Suggest to do restart",
    },
    "reset_password": {
        "condition": "When a customer wants to reset their password",
        "action": "ask for their email address to send them a password",
    },
    "calm_and_reset_password": {
        "condition": "When a customer wants to reset their password",
        "action": "tell them that it's ok and it happens to everyone and ask for their email address to send them a password",
    },
    "frustrated_so_discount": {
        "condition": "The customer expresses frustration, impatience, or dissatisfaction",
        "action": "apologize and offer a discount",
    },
    "confirm_reservation": {
        "condition": "Whenever the customer has placed a reservation, submitted an order, or added items to an order.",
        "action": "ask whether the customer would like to add anything else before finalizing the reservation or order",
    },
    "order_status": {
        "condition": "The customer is asking about a status of an order.",
        "action": "retrieve it's status and inform the customer",
    },
    "return_conditions": {
        "condition": "The customer is asking about return terms.",
        "action": "refer them to the company's website",
    },
    "unsupported_capability": {
        "condition": "When a customer asks about a capability that is not supported",
        "action": "inform the customer that the capability is not supported and make a joke",
    },
    "problem_with_order": {
        "condition": "The customer is reporting a problem with their order.",
        "action": "Apologize and ask for more details about the issue.",
    },
}


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_previously_matched_rule_are_not_matched_when_there_is_no_new_reason
async def test_that_previously_matched_rule_are_not_matched_when_there_is_no_new_reason(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, the app keeps crashing on my phone.",
        ),
        (
            EventSource.AI_AGENT,
            "Sorry to hear that! Let’s try restarting the app and clearing the cache.",
        ),
        (
            EventSource.CUSTOMER,
            "I did that but it's crashing!",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["problem_so_restart"],
        relevant_rule_names=[],
        irrelevant_rule_names=["problem_so_restart"],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_partially_fulfilled_action_with_missing_behavioral_part_is_not_matched_again
async def test_that_partially_fulfilled_action_with_missing_behavioral_part_is_not_matched_again(
    ranker: RuleRanker,
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
            "123@emcie.co",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["calm_and_reset_password"],
        relevant_rule_names=[],
        irrelevant_rule_names=["calm_and_reset_password"],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_rule_that_was_reapplied_earlier_and_should_not_reapply_based_on_the_most_recent_interaction_is_not_matched_1
async def test_that_rule_that_was_reapplied_earlier_and_should_not_reapply_based_on_the_most_recent_interaction_is_not_matched_1(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Ugh, why is this taking so long? I placed my order 40 minutes ago.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm really sorry for the delay, and I completely understand how frustrating that must be. I’ll look into it right away, and I can also offer you a discount for the inconvenience.",
        ),
        (
            EventSource.CUSTOMER,
            "OK, thanks. I will be waiting",
        ),
        (
            EventSource.AI_AGENT,
            "Of course. I'm here to help, and I’ll keep you updated as soon as I know more",
        ),
        (
            EventSource.CUSTOMER,
            "I got the delivery now and it's totally broken! Are you serious, you guys? This is ridiculous.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm so sorry—that should absolutely not have happened. I’ll report this right away, and I can offer you a discount for the trouble.",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you that's nice of you.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["frustrated_so_discount"],
        relevant_rule_names=[],
        irrelevant_rule_names=["frustrated_so_discount"],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_rule_that_was_reapplied_earlier_and_should_not_reapply_based_on_the_most_recent_interaction_is_not_matched_2
async def test_that_rule_that_was_reapplied_earlier_and_should_not_reapply_based_on_the_most_recent_interaction_is_not_matched_2(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey I haven’t receive my order, I placed it 2 weeks ago.",
        ),
        (
            EventSource.AI_AGENT,
            "Let me check on that for you. Can you provide the order number?",
        ),
        (
            EventSource.CUSTOMER,
            "12233",
        ),
        (
            EventSource.AI_AGENT,
            "Thanks! I see it’s on the way and should arrive this weekend.",
        ),
        (
            EventSource.CUSTOMER,
            "Okay, thanks. I also have another order from a different store, what’s the status of that one?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, let me take a look. Could you share the order number for that one too?",
        ),
        (
            EventSource.CUSTOMER,
            "I think 111222.",
        ),
        (
            EventSource.AI_AGENT,
            "Hmm, that number doesn’t seem right. Could you double-check it?",
        ),
        (
            EventSource.CUSTOMER,
            "How can I change the address of an order?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["order_status"],
        relevant_rule_names=[],
        irrelevant_rule_names=["order_status"],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_rule_that_was_reapplied_earlier_and_should_reapply_again_based_on_the_most_recent_interaction_is_matched
async def test_that_rule_that_was_reapplied_earlier_and_should_reapply_again_based_on_the_most_recent_interaction_is_matched(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I’d like to book a table for 2 at 7 PM tonight.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it — a table for 2 at 7 PM. Would you like to add anything else before I confirm the reservation?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, actually — it’s for a birthday. Can we get a small cake?",
        ),
        (
            EventSource.AI_AGENT,
            "Absolutely! I’ve added a birthday cake to your reservation. Would you like anything else before I send it through?",
        ),
        (
            EventSource.CUSTOMER,
            "Oh, and can we have a table near the window if possible?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["confirm_reservation"],
        relevant_rule_names=["confirm_reservation"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_rule_that_should_reapply_is_matched_when_condition_holds_in_the_last_several_messages
async def test_that_rule_that_should_reapply_is_matched_when_condition_holds_in_the_last_several_messages(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I’d like to book a table for 2 at 7 PM tonight.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it — a table for 2 at 7 PM. Would you like to add anything else before I confirm the reservation?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, actually — it’s for a birthday. Can we get a small cake? Do you have chocolate cakes?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes we have chocolate and cheese cakes. What would you want?",
        ),
        (
            EventSource.CUSTOMER,
            "Great so add one chocolate cake please.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["confirm_reservation"],
        relevant_rule_names=["confirm_reservation"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_reapplied_rule_is_still_applied_when_handling_conditions_sub_issue
async def test_that_reapplied_rule_is_still_applied_when_handling_conditions_sub_issue(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I’d like to book a table for 2 at 7 PM tonight.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it — a table for 2 at 7 PM. Would you like to add anything else before I confirm the reservation?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, actually — it’s for a birthday. Can we get a small cake? Do you have chocolate cakes?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes we have chocolate and cheese cakes. What would you want?",
        ),
        (
            EventSource.CUSTOMER,
            "Great so add one chocolate cake please.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["confirm_reservation"],
        relevant_rule_names=["confirm_reservation"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_rule_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold
async def test_that_rule_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hi, I just received my order, and the pizza is cold."),
        (
            EventSource.AI_AGENT,
            "I'm so sorry to hear that. Could you tell me more about the issue?",
        ),
        (EventSource.CUSTOMER, "Yeah, it's not just cold — the box was crushed too."),
        (EventSource.AI_AGENT, "That's really unacceptable. Let me make this right."),
        (EventSource.CUSTOMER, "And I got a parking ticket before coming."),
        (EventSource.AI_AGENT, "I'm sorry to hear that. "),
        (EventSource.CUSTOMER, "And this isn’t the first time you've ruined my order, honestly."),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["problem_with_order"],
        relevant_rule_names=["problem_with_order"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_previously_applied_rules_are_matched_based_on_capabilities


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_previously_applied_rules_are_matched_based_on_capabilities_emotional_reasoning


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_previously_applied_rules_are_matched_based_on_capabilities_with_context_change


# Taken from tests/core/stable/engines/alpha/test_previously_applied_actionable_batch.py::test_that_previously_applied_rules_are_not_matched_based_on_irrelevant_capabilities
