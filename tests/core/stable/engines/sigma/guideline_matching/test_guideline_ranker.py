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
from parlant.core.guidelines import Guideline
from parlant.core.sessions import EventSource

from tests.core.stable.engines.sigma.guideline_matching.utils import (
    create_engine_context,
    create_guideline,
)


@fixture
def ranker(container: Container) -> GuidelineRanker:
    return container[GuidelineRanker]


# Named guideline definitions referenced by the tests below. Each test passes the
# names it wants the ranker to evaluate, and which of those it expects to be
# (ir)relevant. Extend this dictionary as new test cases are added.
GUIDELINES_DICT: dict[str, dict[str, str]] = {
    "ask_toppings": {
        "condition": "the customer asks about toppings",
        "action": "list the available toppings",
    },
    "wants_refund": {
        "condition": "the customer wants a refund",
        "action": "start the refund flow",
    },
    "opening_hours": {
        "condition": "the customer asks about opening hours",
        "action": "tell them the store hours",
    },
    "address_location": {
        "condition": "the customer needs to know our address",
        "action": "Inform the customer that our address is at Sapir 2, Herzliya.",
    },
    "issue_resolved": {
        "condition": "the customer previously expressed stress or dissatisfaction, but the issue has been alleviated",
        "action": "confirm the issue is fully resolved",
    },
    "class_booking": {
        "condition": "the customer asks about booking a class or an appointment",
        "action": "Provide available times and facilitate the booking process, "
        "ensuring to clarify any necessary details such as class type.",
    },
}


def create_guideline_by_name(name: str) -> Guideline:
    spec = GUIDELINES_DICT[name]
    return create_guideline(condition=spec["condition"], action=spec.get("action"))


async def base_test_that_guidelines_are_ranked_correctly(
    ranker: GuidelineRanker,
    conversation: list[tuple[EventSource, str]],
    conversation_guideline_names: list[str],
    relevant_guideline_names: list[str],
    irrelevant_guideline_names: list[str],
) -> None:
    """Rank ``conversation_guideline_names`` against ``conversation`` and assert that:

    - every guideline in ``relevant_guideline_names`` was ranked as relevant, and
    - every guideline in ``irrelevant_guideline_names`` was ranked as not relevant.

    A guideline that appears in neither list is a "don't care": any decision the
    ranker makes about it is accepted.
    """
    # Sanity: the expected lists must reference guidelines that are actually ranked,
    # and a guideline can't be both relevant and irrelevant.
    assert set(relevant_guideline_names) <= set(conversation_guideline_names)
    assert set(irrelevant_guideline_names) <= set(conversation_guideline_names)
    assert not (set(relevant_guideline_names) & set(irrelevant_guideline_names))

    guidelines_by_name = {
        name: create_guideline_by_name(name) for name in conversation_guideline_names
    }

    context = create_engine_context(conversation=conversation)

    result = await ranker.rank(context, list(guidelines_by_name.values()))

    relevance_by_id = {
        ranked.guideline.id: ranked.is_relevant for ranked in result.ranked_guidelines
    }

    for name in relevant_guideline_names:
        guideline = guidelines_by_name[name]
        assert relevance_by_id.get(guideline.id) is True, (
            f"expected guideline {name!r} to be ranked as relevant, but it wasn't"
        )

    for name in irrelevant_guideline_names:
        guideline = guidelines_by_name[name]
        assert relevance_by_id.get(guideline.id) is False, (
            f"expected guideline {name!r} to be ranked as not relevant, but it was"
        )


def test_that_a_guideline_ranker_can_be_created(ranker: GuidelineRanker) -> None:
    assert ranker is not None


async def test_that_a_relevant_guideline_is_ranked_as_relevant(ranker: GuidelineRanker) -> None:
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")],
        conversation_guideline_names=["ask_toppings"],
        relevant_guideline_names=["ask_toppings"],
        irrelevant_guideline_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_guideline_matcher.py
# ::test_that_relevant_guidelines_are_matched_parametrized_2
async def test_that_relevant_guidelines_are_matched_parametrized_2(
    ranker: GuidelineRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm feeling a bit stressed about coming in. Can I cancel my class for today?",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear that. While cancellation is not possible now, "
            "how about a lighter session? Maybe it helps to relax.",
        ),
        (
            EventSource.CUSTOMER,
            "I suppose that could work. What do you suggest?",
        ),
        (
            EventSource.AI_AGENT,
            "How about our guided meditation session every Tuesday evening at 20:00? "
            "It's very calming and might be just what you need right now.",
        ),
        (
            EventSource.CUSTOMER,
            "Alright, please book me into that. Thank you for understanding.",
        ),
        (
            EventSource.AI_AGENT,
            "You're welcome! I've switched your booking to the meditation session. "
            "Remember, it's okay to feel stressed. We're here to support you.",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks, I really appreciate it.",
        ),
        (
            EventSource.AI_AGENT,
            "Anytime! Is there anything else I can assist you with today?",
        ),
        (
            EventSource.CUSTOMER,
            "No, that's all for now.",
        ),
        (
            EventSource.AI_AGENT,
            "Take care and see you soon at the meditation class. "
            "Our gym is at the mall on the 2nd floor.",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you!",
        ),
    ]

    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        conversation=conversation,
        conversation_guideline_names=["class_booking", "issue_resolved"],
        relevant_guideline_names=["issue_resolved"],
        irrelevant_guideline_names=["class_booking"],
    )
