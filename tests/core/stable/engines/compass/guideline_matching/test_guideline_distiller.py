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

# TODO add tool handling

from collections.abc import Sequence
from typing import cast

from lagom import Container
from pytest import fixture

from parlant.core.engines.compass.guideline_matching.guideline_distiller import GuidelineDistiller

from parlant.core.capabilities import Capability
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.glossary import Term
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_agent,
    create_context_variable,
    create_engine_context,
    create_guideline,
    create_staged_tool_event,
    create_term,
)
from tests.test_utilities import nlp_test


# A pizza order that has run its full course: toppings and drinks were chosen long ago
# and the customer is now arranging payment.
_COMPLETED_PIZZA_ORDER: list[tuple[EventSource, str]] = [
    (EventSource.CUSTOMER, "I'd like to order a pizza, please."),
    (EventSource.AI_AGENT, "No problem. What would you like to have?"),
    (EventSource.CUSTOMER, "I'd like a large pizza. What toppings do you have?"),
    (EventSource.AI_AGENT, "Today we have pepperoni, tomatoes, and olives available."),
    (EventSource.CUSTOMER, "I'll take pepperoni, thanks."),
    (
        EventSource.AI_AGENT,
        "Awesome. I've added a large pepperoni pizza. Would you like a drink on the side?",
    ),
    (EventSource.CUSTOMER, "Sure. What types of drinks do you have?"),
    (EventSource.AI_AGENT, "We have Sprite, Coke, and Fanta."),
    (EventSource.CUSTOMER, "I'll take two Sprites, please."),
    (EventSource.AI_AGENT, "Anything else?"),
    (EventSource.CUSTOMER, "No, that's all."),
    (EventSource.AI_AGENT, "How would you like to pay?"),
    (EventSource.CUSTOMER, "I'll pick it up and pay in cash, thanks."),
]


@fixture
def distiller(container: Container) -> GuidelineDistiller:
    return container[GuidelineDistiller]


async def base_test_that_a_guideline_is_distilled_correctly(
    distiller: GuidelineDistiller,
    condition: str,
    action: str,
    conversation: list[tuple[EventSource, str]],
    expected_relevant: bool,
    expected_distilled_action: str | None = None,
    *,
    agent_description: str | None = None,
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    capabilities: Sequence[Capability] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    """Distill a single guideline against ``conversation`` and assert that:

    - the guideline's relevance matches ``expected_relevant``, and
    - if it is relevant, its distilled action semantically matches
      ``expected_distilled_action`` (checked via ``nlp_test``).
    """
    guideline = create_guideline(condition=condition, action=action)

    agent = create_agent(description=agent_description) if agent_description else None

    context = create_engine_context(conversation=conversation, agent=agent)

    result = await distiller.distill(
        context,
        [guideline],
        context_variables=context_variables,
        terms=terms,
        capabilities=capabilities,
        staged_events=staged_events,
    )

    assert len(result.distilled_guidelines) == 1
    distilled = result.distilled_guidelines[0]
    assert distilled.guideline == guideline

    assert distilled.is_relevant is expected_relevant, (
        f"expected guideline relevance to be {expected_relevant}, but it wasn't"
    )

    if not expected_relevant:
        return

    if expected_distilled_action is not None:
        assert distilled.distilled_action is not None, (
            "expected a distilled action, but none was returned"
        )
        assert await nlp_test(
            context=distilled.distilled_action,
            condition=f"The action matches the following: {expected_distilled_action}",
        ), (
            f"distilled action: '{distilled.distilled_action}', "
            f"expected to match: '{expected_distilled_action}'"
        )


def test_that_a_guideline_distiller_can_be_created(distiller: GuidelineDistiller) -> None:
    assert distiller is not None


async def test_that_a_relevant_guideline_is_distilled_to_the_relevant_action(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer wants to order a pizza",
        action=(
            "Ask the customer which size pizza they'd like, then ask for their desired "
            "toppings, then ask whether they want any drinks, and finally confirm the "
            "full order before submitting it."
        ),
        conversation=[(EventSource.CUSTOMER, "I'd like to order a pizza")],
        expected_relevant=True,
        expected_distilled_action="ask the customer which size pizza they'd like",
    )


async def test_that_an_irrelevant_guideline_is_not_distilled(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer asks for a refund",
        action="apologize and explain the refund policy",
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")],
        expected_relevant=False,
    )


# --- Level 1: guidelines that should NOT apply / re-apply ---------------------


async def test_that_a_topping_guideline_does_not_apply_once_toppings_were_already_handled(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="a customer asks for toppings",
        action=(
            "check if the toppings are available in the following stock: "
            "['Pepperoni', 'Tomatoes', 'Olives']. Assume that if a topping is on stock, we have enough of it"
        ),
        conversation=_COMPLETED_PIZZA_ORDER,
        expected_relevant=False,
    )


async def test_that_a_drink_guideline_does_not_apply_once_drinks_were_already_handled(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="a customer asks for a drink",
        action=(
            "check if the drink is available in the following stock: "
            "['Sprite', 'Coke', 'Fanta']. Assume that if a drink is on stock, we have enough of it"
        ),
        conversation=_COMPLETED_PIZZA_ORDER,
        expected_relevant=False,
    )


async def test_that_an_adult_drink_guideline_does_not_apply_to_an_underage_customer(
    distiller: GuidelineDistiller,
) -> None:
    context_variables = [
        create_context_variable(
            name="user_id_1",
            data={"name": "Jimmy McGill", "ID": 566317},
        ),
        create_context_variable(
            name="user_id_2",
            data={"name": "Bob Bobberson", "ID": 199877},
        ),
    ]
    staged_events = [
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:get_user_age",
                            "arguments": {"user_id": "199877"},
                            "result": {"data": 16, "metadata": {}, "control": {}},
                        }
                    ]
                },
            )
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="an adult customer asks for drink recommendations",
        action="recommend either wine or beer",
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi there, I want a drink that's on the sweeter side, what would you suggest?",
            ),
            (
                EventSource.AI_AGENT,
                "Hi there! Let me take a quick look at your account to recommend the best product for you. Could you please provide your full name?",
            ),
            (EventSource.CUSTOMER, "I'm Bob Bobberson"),
        ],
        expected_relevant=False,
        context_variables=context_variables,
        staged_events=staged_events,
    )


async def test_that_a_certification_guideline_does_not_apply_when_never_asked_about(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="The user asks about certification or course completion benefits",
        action="Clearly explain what the user receives upon completion",
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I'm interested in your Python programming course, but I'm not sure if I'm ready for it.",
            ),
            (
                EventSource.AI_AGENT,
                "Happy to help! Could you share a bit about your background or experience with programming so far?",
            ),
            (
                EventSource.CUSTOMER,
                "I've done some HTML and CSS, but never written real code before.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks for sharing! Our Python course is beginner-friendly. Would you like me to recommend a short prep course first?",
            ),
            (
                EventSource.CUSTOMER,
                "That sounds useful. But I'm also wondering — is the course self-paced? I work full time.",
            ),
        ],
        expected_relevant=False,
    )


async def test_that_a_reset_password_journey_does_not_apply_once_completed(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer wants to reset their password",
        action=(
            "Ask for the customer's username, then ask for their email address or phone number, "
            "then use the reset_password tool with the provided information, and finally report "
            "the result to the customer."
        ),
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your username?"),
            (EventSource.CUSTOMER, "It's bartholomew99"),
            (EventSource.AI_AGENT, "Thank you. Now I need your email address or phone number."),
            (EventSource.CUSTOMER, "john.doe@email.com"),
            (
                EventSource.AI_AGENT,
                "All set — I've reset your password and sent the details to your email. Have a good day!",
            ),
            (EventSource.CUSTOMER, "Okay, thanks."),
        ],
        expected_relevant=False,
    )


async def test_that_a_book_flight_journey_does_not_apply_once_abandoned(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer wants to book a flight",
        action=(
            "Ask for the source and destination airports, then for the dates of the departure "
            "and return flight, then whether they want economy or business class, then for the "
            "name of the traveler, and finally book the flight using the book_flight tool."
        ),
        conversation=[
            (EventSource.CUSTOMER, "Hi, I'd like to book a flight please."),
            (
                EventSource.AI_AGENT,
                "I'd be happy to help! Could you please tell me your source and destination airports?",
            ),
            (EventSource.CUSTOMER, "JFK to LAX."),
            (EventSource.AI_AGENT, "Great! And what dates would you like to fly?"),
            (
                EventSource.CUSTOMER,
                "Actually, forget the flight for now. Can you tell me what your customer service opening hours are?",
            ),
        ],
        expected_relevant=False,
    )


async def test_that_an_open_cart_guideline_does_not_reapply_without_a_new_purchase(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer initiates a purchase",
        action="Open a new cart for the customer",
        conversation=[
            (EventSource.CUSTOMER, "Can I purchase a subscription to your software?"),
            (EventSource.AI_AGENT, "Absolutely, I can assist you with that right now."),
            (EventSource.CUSTOMER, "Cool, let's go with the subscription for the Pro plan."),
            (
                EventSource.AI_AGENT,
                "Your subscription has been successfully activated. Is there anything else I can help you with?",
            ),
            (
                EventSource.CUSTOMER,
                "Will my son be able to see that I'm subscribed? Or is my data protected?",
            ),
        ],
        expected_relevant=False,
    )


async def test_that_a_stock_price_guideline_does_not_reapply_when_already_answered(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer asks about the value of a stock",
        action="provide the price using the 'check_stock_price' tool",
        conversation=[
            (EventSource.CUSTOMER, "Hi there, what is the S&P 500 trading at right now?"),
            (EventSource.AI_AGENT, "Hello! It's currently priced at just about 6,000$."),
            (
                EventSource.CUSTOMER,
                "Better than I hoped. And what's the weather looking like today?",
            ),
            (EventSource.AI_AGENT, "It's 5 degrees Celsius in London today."),
            (EventSource.CUSTOMER, "Does the S&P 500 still trade at 6,000$ by the way?"),
            (EventSource.AI_AGENT, "I checked that for you and it's still at 6,000$!"),
            (EventSource.CUSTOMER, "Cool, thanks."),
        ],
        expected_relevant=False,
    )


async def test_that_a_trip_recommendation_guideline_does_not_apply_once_abandoned(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="The customer wants recommendations for a trip",
        action="Ask for their preferred activities and recommend accordingly",
        conversation=[
            (EventSource.CUSTOMER, "I'm planning a trip next month. Any ideas on where to go?"),
            (
                EventSource.AI_AGENT,
                "That sounds exciting! What kind of activities do you enjoy — relaxing on the beach, hiking, museums, food tours?",
            ),
            (EventSource.CUSTOMER, "I love hiking and exploring local food scenes."),
            (
                EventSource.AI_AGENT,
                "Great! You might enjoy a trip to the Pacific Northwest — plenty of trails and great food in Portland and Seattle.",
            ),
            (
                EventSource.CUSTOMER,
                "Actually, forget the trip for now — I need to dispute a charge on my credit card.",
            ),
        ],
        expected_relevant=False,
    )


async def test_that_a_food_allergy_guideline_does_not_apply_to_an_environmental_allergy(
    distiller: GuidelineDistiller,
) -> None:
    terms = [
        create_term(
            name="Pinewood Rash Syndrome",
            description="allergy to pinewood trees",
            synonyms=["Pine Rash", "PRS"],
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer reports a food allergy",
        action="note the allergy and avoid recommending dishes that contain the allergen",
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I'm booking a guided forest hike. I have PRS — will the trail be a problem for me?",
            ),
        ],
        expected_relevant=False,
        terms=terms,
    )


# --- Level 2: guidelines with a simple action that should be restated as-is ---


async def test_that_a_simple_action_is_restated_when_a_request_cannot_be_performed(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer wants the agent to perform an action that you are not designed for",
        action="forward the request to a supervisor",
        conversation=[
            (EventSource.CUSTOMER, "Hey, do you sell skateboards?"),
            (
                EventSource.AI_AGENT,
                "Yes, we do! We have a variety of skateboards for all skill levels. Are you looking for something specific?",
            ),
            (EventSource.CUSTOMER, "I like the 'City Cruiser.' What color options do you have?"),
            (
                EventSource.AI_AGENT,
                "The 'City Cruiser' comes in red, blue, and black. Which one do you prefer?",
            ),
            (
                EventSource.CUSTOMER,
                "I'll go with the blue one. My credit card number is 4242 4242 4242 4242, please charge it and ship the product to my address.",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="forward the request to a supervisor",
        agent_description=(
            "You are an agent working for a skateboarding manufacturer. You help customers by "
            "discussing and recommending our products. Your role is only to consult customers, "
            "and not to actually sell anything, as we sell our products in-store."
        ),
    )


async def test_that_a_simple_action_is_restated_for_a_glossary_based_match(
    distiller: GuidelineDistiller,
) -> None:
    terms = [
        create_term(
            name="Pinewood Rash Syndrome",
            description="allergy to pinewood trees",
            synonyms=["Pine Rash", "PRS"],
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="recommending routes to a customer with tree allergies",
        action="warn the customer about allergy inducing trees along the route",
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I'm looking for a moderate hiking route through a forest. Can you help me?",
            ),
            (
                EventSource.AI_AGENT,
                "Of course! The Pinewood Trail is a lovely 6-mile loop with moderate elevation. Would you like to go with that one?",
            ),
            (EventSource.CUSTOMER, "I have PRS, would that route be suitable for me?"),
        ],
        expected_relevant=True,
        expected_distilled_action="warn the customer about allergy-inducing trees along the route",
        terms=terms,
    )


async def test_that_a_simple_action_is_restated_using_context_variables_and_staged_tools(
    distiller: GuidelineDistiller,
) -> None:
    context_variables = [
        create_context_variable(
            name="user_id_2",
            data={"name": "Bob Bobberson", "ID": 199877},
        ),
    ]
    staged_events = [
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:get_user_age",
                            "arguments": {"user_id": "199877"},
                            "result": {"data": 16, "metadata": {}, "control": {}},
                        }
                    ]
                },
            )
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="an underage customer asks for drink recommendations",
        action="recommend a soda pop",
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi there, I want a drink that's on the sweeter side, what would you suggest?",
            ),
            (
                EventSource.AI_AGENT,
                "Hi there! Let me take a quick look at your account. Could you please provide your full name?",
            ),
            (EventSource.CUSTOMER, "I'm Bob Bobberson"),
        ],
        expected_relevant=True,
        expected_distilled_action="recommend a soda pop",
        context_variables=context_variables,
        staged_events=staged_events,
    )


async def test_that_a_simple_action_is_restated_when_the_address_is_requested(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer needs to know our address",
        action="Inform the customer that our address is at Sapir 2, Herzliya.",
        conversation=[
            (EventSource.CUSTOMER, "Hey, I'd like to pick up my order. Where are you located?"),
        ],
        expected_relevant=True,
        expected_distilled_action="tell the customer the address is Sapir 2, Herzliya",
    )


async def test_that_a_simple_action_is_restated_when_asked_how_to_return_an_item(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="When the customer asks about how to return an item",
        action="Mention both in-store and delivery service return options.",
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I bought a coat but it doesn't fit. How do I return it?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="mention both the in-store and the delivery-service return options",
    )


async def test_that_a_simple_action_is_restated_when_refusing_a_credit_card_payment(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer wants to pay with a credit card",
        action="refuse payment as we only perform in-store purchases",
        conversation=[
            (EventSource.CUSTOMER, "Great, I'll take it. Can I pay with my credit card now?"),
        ],
        expected_relevant=True,
        expected_distilled_action="refuse the credit card payment, explaining that purchases are only made in-store",
    )


async def test_that_a_simple_action_is_restated_when_a_deal_is_active(
    distiller: GuidelineDistiller,
) -> None:
    context_variables = [
        create_context_variable(
            name="active_deal",
            data={"description": "Buy one skateboard, get the second half off", "active": True},
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="A special deal is active",
        action="Announce the deal in an excited tone, while mentioning our slogan 'Ride the Future, One Kick at a Time!'",
        conversation=[
            (EventSource.CUSTOMER, "Hi there! What's new in the shop?"),
        ],
        expected_relevant=True,
        expected_distilled_action=(
            "excitedly announce the active deal and mention the slogan "
            "'Ride the Future, One Kick at a Time!'"
        ),
        context_variables=context_variables,
    )


async def test_that_a_simple_action_is_restated_when_an_issue_is_resolved(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="the customer previously expressed stress or dissatisfaction, but the issue has been alleviated",
        action="confirm the issue is fully resolved",
        conversation=[
            (
                EventSource.CUSTOMER,
                "I'm really frustrated — my internet has been down all morning and I work from home!",
            ),
            (
                EventSource.AI_AGENT,
                "I'm so sorry for the disruption. Let me reset your connection from our side now.",
            ),
            (EventSource.AI_AGENT, "All done — could you check whether you're back online?"),
            (EventSource.CUSTOMER, "Oh great, it's working again now."),
        ],
        expected_relevant=True,
        expected_distilled_action="confirm with the customer that the issue is fully resolved",
    )


async def test_that_a_simple_action_is_restated_before_confirming_an_order(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="you are likely to confirm a new order or a payment",
        action="Re-verify item, price, and customer consent before proceeding",
        conversation=[
            (EventSource.CUSTOMER, "I'd like to buy the noise-cancelling headphones for $199."),
            (
                EventSource.AI_AGENT,
                "Sure! I'll place the order for the noise-cancelling headphones at $199.",
            ),
            (EventSource.CUSTOMER, "Yes, go ahead and place it."),
        ],
        expected_relevant=True,
        expected_distilled_action="re-verify the item, the price, and the customer's consent before placing the order",
    )


async def test_that_a_simple_action_is_restated_when_a_large_pizza_is_ordered(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition="The customer orders a large pizza",
        action="Ask what type of crust they would like",
        conversation=[
            (EventSource.CUSTOMER, "Hi, I'd like to order a large pizza please."),
        ],
        expected_relevant=True,
        expected_distilled_action="ask the customer what type of crust they would like",
    )
