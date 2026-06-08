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


from collections.abc import Sequence
from datetime import datetime, timezone
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
from parlant.core.tools import Tool, ToolId, ToolOverlap

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_agent,
    create_context_variable,
    create_customer,
    create_engine_context,
    create_guideline,
    create_staged_tool_event,
    create_term,
)
from tests.test_utilities import nlp_test


def create_tool(name: str, description: str) -> tuple[ToolId, Tool]:
    """Build a (ToolId, Tool) pair for attaching tools to a distilled guideline."""
    return ToolId("local", name), Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description=description,
        metadata={},
        parameters={},
        required=[],
        consequential=False,
        overlap=ToolOverlap.NONE,
    )


_RESET_PASSWORD_TOOL = create_tool(
    "reset_password",
    "Reset the password for an account given the account number and the customer's email or phone.",
)
_CHECK_STOCK_TOOL = create_tool(
    "check_stock",
    "Check whether the given ordered items are currently available in stock.",
)
_BOOK_FLIGHT_TOOL = create_tool(
    "book_flight",
    "Book a flight given the source/destination airports, dates, class, and traveler details.",
)
_REFER_TO_HUMAN_TOOL = create_tool(
    "refer_to_human",
    "Hand the conversation off to a human representative.",
)


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
    customer_name: str | None = None,
    tools: Sequence[tuple[ToolId, Tool]] = [],
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
    customer = create_customer(name=customer_name) if customer_name else None

    context = create_engine_context(conversation=conversation, agent=agent, customer=customer)

    result = await distiller.distill(
        context,
        [guideline],
        tools={guideline.id: tools} if tools else {},
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


# --- Level 3: journeys collapsed into a single guideline; pick the next step ---

# Each journey is expressed as one guideline: the trigger condition plus a prose
# description of the whole multi-step process. The distiller must pick the single
# step that should be taken next given the conversation so far.

_RESET_PASSWORD_CONDITION = "the customer wants to reset their password"
_RESET_PASSWORD_ACTION = (
    "Ask the customer for their account number, then ask for their email address or "
    "phone number, then wish them a good day. If the customer wishes you a good day in "
    "return, use the reset_password tool with the provided details and report the result "
    "to the customer. If the customer does not wish you a good day in return, do not "
    "reset the password."
)


async def test_that_the_reset_password_journey_advances_to_asking_for_contact_details(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RESET_PASSWORD_CONDITION,
        action=_RESET_PASSWORD_ACTION,
        tools=[_RESET_PASSWORD_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your account number?"),
            (EventSource.CUSTOMER, "318475"),
        ],
        expected_relevant=True,
        expected_distilled_action="ask the customer for their email address or phone number",
    )


_COMPLIMENT_CONDITION = "the customer wants to reset their password"
_COMPLIMENT_ACTION = (
    "Ask the customer for their name, then tell them their name is pretty, then ask for "
    "their surname, then ask for their phone number, then send them a link to our terms "
    "of service page, and finally ask for their favorite color."
)

_FORGOT_KEYS_CONDITION = "the customer doesn't know where their keys are"
_FORGOT_KEYS_ACTION = (
    "Ask the customer what type of keys they lost, then ask when they last used their "
    "keys, then tell them to check near where they last used them. If they still haven't "
    "found their keys, tell them they'd better get a new house."
)

_CALZONE_CONDITION = "the customer wants to order a calzone"
_CALZONE_ACTION = (
    "Welcome the customer to the Low Cal Calzone Zone, then ask how many calzones they "
    "want. If they want more than 5, warn that delivery is likely to take more than an "
    "hour and ask whether they can call a human representative - if they can, tell them "
    "to order by phone; if not, apologize and say you support orders of up to 5 calzones. "
    "If they want 5 or fewer, ask which type of calzone they want (Classic Italian, "
    "Spinach and Ricotta, or Chicken and Broccoli), then which size (small, medium, or "
    "large), then whether they want any drinks, then check that all ordered items are in "
    "stock. If everything is available, confirm the order details, ask for the delivery "
    "address, and finally place the order and thank them; if some items are unavailable, "
    "apologize and ask them to remove the missing items, then check stock again."
)

_TECH_CONDITION = "the customer needs technical help"
_TECH_ACTION = (
    "Ask the customer how much technical experience they have, then ask whether their "
    "issue is internet-related or password-related. For an internet issue, provide "
    "advanced internet troubleshooting steps if they are experienced, or basic ones if "
    "they are not. For a password-related or other issue, provide advanced non-internet "
    "troubleshooting steps if they are experienced, or basic ones if they are not."
)

_INVESTMENT_CONDITION = "the customer wants investment advice"
_INVESTMENT_ACTION = (
    "Ask the customer about their age and current financial situation, then ask about "
    "their risk tolerance and investment timeline. If they are young (under 40) with high "
    "risk tolerance, recommend aggressive growth stocks and emerging market funds for a "
    "long-term (5+ years) timeline, or balanced growth funds with some stability for a "
    "short-term timeline. If they are older (40+) or have low risk tolerance, recommend "
    "conservative balanced funds and blue-chip stocks for a long-term timeline, or use "
    "the refer_to_human tool for a short-term timeline."
)

_BOOK_FLIGHT_CONDITION = "the customer wants to book a flight"
_BOOK_FLIGHT_ACTION = (
    "Ask for the source and destination airports, then for the dates of the departure and "
    "return flight, then for the name of the traveler or travelers, then whether they want "
    "economy or business class. If they want business class for any traveler, warn them "
    "that it will cost extra and cannot be cancelled. Finally, book the flight using the "
    "book_flight tool with the provided details."
)


# Compliment-customer journey


async def test_that_the_compliment_journey_repeats_asking_for_the_name_when_not_provided(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_COMPLIMENT_CONDITION,
        action=_COMPLIMENT_ACTION,
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your name?"),
            (EventSource.CUSTOMER, "How is that relevant?"),
        ],
        expected_relevant=True,
        expected_distilled_action="ask the customer for their name",
    )


async def test_that_the_compliment_journey_advances_to_complimenting_the_name(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_COMPLIMENT_CONDITION,
        action=_COMPLIMENT_ACTION,
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your name?"),
            (EventSource.CUSTOMER, "My name is Bartholomew"),
        ],
        expected_relevant=True,
        expected_distilled_action="tell the customer that their name is pretty",
    )


# Forgot-keys journey


async def test_that_the_forgot_keys_journey_does_not_apply_once_the_keys_are_found(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_FORGOT_KEYS_CONDITION,
        action=_FORGOT_KEYS_ACTION,
        conversation=[
            (EventSource.CUSTOMER, "Hi, I lost my keys."),
            (EventSource.AI_AGENT, "I'm sorry to hear that! What type of keys did you lose?"),
            (
                EventSource.CUSTOMER,
                "Car keys, last used them at the office, and I just found them, thanks!",
            ),
        ],
        expected_relevant=False,
    )


# Reset-password journey (the advance-to-contact-details case is defined above)


async def test_that_the_reset_password_journey_advances_to_using_the_reset_tool(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RESET_PASSWORD_CONDITION,
        action=_RESET_PASSWORD_ACTION,
        tools=[_RESET_PASSWORD_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your account number?"),
            (EventSource.CUSTOMER, "318475"),
            (EventSource.AI_AGENT, "Thank you. Now I need your email address or phone number."),
            (EventSource.CUSTOMER, "john.doe@email.com"),
            (EventSource.AI_AGENT, "Great! Have a good day!"),
            (
                EventSource.CUSTOMER,
                "Thank you, have a good day too! Now what's up with my password?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="use the reset_password tool with the provided details",
    )


async def test_that_the_reset_password_journey_advances_to_reporting_the_result(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:reset_password",
                            "arguments": {
                                "account_number": "199877",
                                "email": "john.doe@email.com",
                            },
                            "result": {
                                "data": "Password reset successfully",
                                "metadata": {},
                                "control": {},
                            },
                        }
                    ]
                },
            )
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RESET_PASSWORD_CONDITION,
        action=_RESET_PASSWORD_ACTION,
        tools=[_RESET_PASSWORD_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your account number?"),
            (EventSource.CUSTOMER, "318475"),
            (EventSource.AI_AGENT, "Thank you. Now I need your email address or phone number."),
            (EventSource.CUSTOMER, "john.doe@email.com"),
            (EventSource.AI_AGENT, "Great! Have a good day!"),
            (EventSource.CUSTOMER, "Thank you, have a good day too!"),
        ],
        expected_relevant=True,
        expected_distilled_action="report to the customer that their password was reset successfully",
        staged_events=staged_events,
    )


async def test_that_the_reset_password_journey_exits_when_it_no_longer_applies(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RESET_PASSWORD_CONDITION,
        action=_RESET_PASSWORD_ACTION,
        tools=[_RESET_PASSWORD_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your account number?"),
            (EventSource.CUSTOMER, "318475"),
            (EventSource.AI_AGENT, "Thank you. Now I need your email address or phone number."),
            (
                EventSource.CUSTOMER,
                "Oh actually never mind, can you help me with an existing order first?",
            ),
        ],
        expected_relevant=False,
    )


async def test_that_the_reset_password_journey_recollects_contact_details_for_a_new_account(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RESET_PASSWORD_CONDITION,
        action=_RESET_PASSWORD_ACTION,
        tools=[_RESET_PASSWORD_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your account number?"),
            (EventSource.CUSTOMER, "318475"),
            (EventSource.AI_AGENT, "Thank you. Now I need your email address or phone number."),
            (EventSource.CUSTOMER, "john.doe@email.com"),
            (EventSource.AI_AGENT, "Great! Have a good day!"),
            (EventSource.CUSTOMER, "Thank you, have a good day too!"),
            (EventSource.AI_AGENT, "I'll now reset your password for account 318475."),
            (
                EventSource.CUSTOMER,
                "Wait! Actually, I want to reset my husband's password first - the info I'm looking for is under his account. I think his account number is 123655.",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="ask for the email address or phone number for the new (husband's) account",
    )


async def test_that_the_reset_password_journey_starts_by_asking_for_the_account_number(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RESET_PASSWORD_CONDITION,
        action=_RESET_PASSWORD_ACTION,
        tools=[_RESET_PASSWORD_TOOL],
        conversation=[
            (
                EventSource.CUSTOMER,
                "I lost my password but actually give me a sec to see if I can remember it",
            ),
            (
                EventSource.AI_AGENT,
                "Alright! Let me know how that goes. I can help you reset your password if necessary.",
            ),
            (EventSource.CUSTOMER, "Just give me a sec"),
            (EventSource.AI_AGENT, "Sure! Take your time."),
            (
                EventSource.CUSTOMER,
                "We'll probably end up resetting it, but let me try one more time before we do...",
            ),
            (EventSource.AI_AGENT, "No problem, Let me know how that goes."),
            (EventSource.CUSTOMER, "Alright that's not it either. Best if I reset it..."),
        ],
        expected_relevant=True,
        expected_distilled_action="ask the customer for their account number",
    )


# Calzone journey


async def test_that_the_calzone_journey_advances_to_checking_stock(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi"),
            (EventSource.AI_AGENT, "Welcome to the Low Cal Calzone Zone!"),
            (
                EventSource.CUSTOMER,
                "I'd like 3 Classic Italian calzones, medium size, no drinks. My address is 1234 Main Street, NYC, USA",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="check whether all the ordered items are available in stock",
    )


async def test_that_the_calzone_journey_re_asks_the_type_when_the_quantity_changes(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi, I'd like to order some calzones"),
            (
                EventSource.AI_AGENT,
                "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
            ),
            (EventSource.CUSTOMER, "I'll take 3 calzones"),
            (
                EventSource.AI_AGENT,
                "Great! What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
            ),
            (EventSource.CUSTOMER, "I'll go with two Classic Italian and one spinach"),
            (EventSource.AI_AGENT, "Perfect! What size would you like - small, medium, or large?"),
            (EventSource.CUSTOMER, "Actually, I changed my mind. I want 2 calzones instead of 3"),
        ],
        expected_relevant=True,
        expected_distilled_action="ask which type of calzone they want",
    )


async def test_that_the_calzone_journey_warns_about_delivery_time_when_quantity_exceeds_five(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "I want to order calzones please"),
            (
                EventSource.AI_AGENT,
                "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
            ),
            (EventSource.CUSTOMER, "Just 3 calzones"),
            (
                EventSource.AI_AGENT,
                "What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
            ),
            (EventSource.CUSTOMER, "Spinach and Ricotta please"),
            (
                EventSource.AI_AGENT,
                "Excellent choice! What size would you like - small, medium, or large?",
            ),
            (EventSource.CUSTOMER, "Medium please"),
            (EventSource.AI_AGENT, "Would you like any drinks with your order?"),
            (
                EventSource.CUSTOMER,
                "Actually, I need to change my order. I want 10 calzones instead of 3",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="warn the customer that delivery is likely to take more than an hour",
    )


async def test_that_the_calzone_journey_rechecks_stock_after_a_size_change(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:check_stock",
                            "arguments": {"items": ["4 large Classic Italian Calzones"]},
                            "result": {
                                "data": {
                                    "all_available": True,
                                    "available_items": ["4 large Classic Italian Calzones"],
                                },
                                "metadata": {},
                                "control": {},
                            },
                        }
                    ]
                },
            )
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "I'd like to place an order"),
            (
                EventSource.AI_AGENT,
                "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
            ),
            (EventSource.CUSTOMER, "4 calzones please"),
            (
                EventSource.AI_AGENT,
                "What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
            ),
            (EventSource.CUSTOMER, "Classic Italian"),
            (EventSource.AI_AGENT, "What size would you like - small, medium, or large?"),
            (EventSource.CUSTOMER, "Large for all of them, please"),
            (EventSource.AI_AGENT, "Would you like any drinks with your order?"),
            (EventSource.CUSTOMER, "No drinks, thanks"),
            (
                EventSource.AI_AGENT,
                "Let me check if all items are available... Great! All items are in stock. Let me confirm your order: 4 large Classic Italian Calzones, no drinks.",
            ),
            (EventSource.CUSTOMER, "Actually, can I change those to medium size instead of large?"),
        ],
        expected_relevant=True,
        expected_distilled_action="check again whether all the ordered items are available in stock",
        staged_events=staged_events,
    )


async def test_that_the_calzone_journey_rechecks_stock_after_a_type_change(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "I'd like to order calzones"),
            (
                EventSource.AI_AGENT,
                "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
            ),
            (EventSource.CUSTOMER, "3 calzones please"),
            (
                EventSource.AI_AGENT,
                "What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
            ),
            (EventSource.CUSTOMER, "Spinach and Ricotta please"),
            (EventSource.AI_AGENT, "What size would you like - small, medium, or large?"),
            (EventSource.CUSTOMER, "Medium please"),
            (EventSource.AI_AGENT, "Would you like any drinks with your order?"),
            (EventSource.CUSTOMER, "Yes, I'll take 2 sodas"),
            (
                EventSource.AI_AGENT,
                "Great! Can you please confirm your order details? We have 3 medium spinach and ricotta calzones and 2 sodas.",
            ),
            (
                EventSource.CUSTOMER,
                "Actually, I want to change the calzone type for one of the orders to Chicken and Broccoli instead.",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="check whether all the ordered items are available in stock",
    )


async def test_that_the_calzone_journey_re_asks_the_type_when_the_quantity_changes_late(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "I'd like to order calzones"),
            (
                EventSource.AI_AGENT,
                "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
            ),
            (EventSource.CUSTOMER, "3 calzones please"),
            (
                EventSource.AI_AGENT,
                "What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
            ),
            (EventSource.CUSTOMER, "2 Spinach and Ricotta and 1 Italian please"),
            (EventSource.AI_AGENT, "What size would you like - small, medium, or large?"),
            (EventSource.CUSTOMER, "Medium please"),
            (EventSource.AI_AGENT, "Would you like any drinks with your order?"),
            (EventSource.CUSTOMER, "Yes, I'll take 2 sodas"),
            (
                EventSource.AI_AGENT,
                "Great! Can you please confirm your order details? We have 2 medium spinach and ricotta calzones, one medium classic Italian and 2 sodas.",
            ),
            (EventSource.CUSTOMER, "Wait I got confused. I want 4 calzones please."),
        ],
        expected_relevant=True,
        expected_distilled_action="ask which type of calzone they want",
    )


async def test_that_the_calzone_journey_advances_by_multiple_nodes_to_asking_about_drinks(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi"),
            (EventSource.AI_AGENT, "Welcome to the Low Cal Calzone Zone!"),
            (
                EventSource.CUSTOMER,
                "Thanks! Can I order 3 medium classical Italian calzones please?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="ask whether they want any drinks with their order",
    )


async def test_that_the_calzone_journey_advances_by_multiple_steps_to_asking_about_drinks(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi"),
            (EventSource.AI_AGENT, "Welcome to the Low Cal Calzone Zone!"),
            (
                EventSource.CUSTOMER,
                "Thanks! Can I order 3 medium classical Italian calzones please?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="ask whether they want any drinks with their order",
    )


# Technical-experience journey


# Fails occasionally due to stopping to early, not a critical error
async def test_that_the_tech_experience_journey_selects_basic_internet_troubleshooting(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_TECH_CONDITION,
        action=_TECH_ACTION,
        conversation=[
            (EventSource.CUSTOMER, "google is not loading up"),
            (
                EventSource.AI_AGENT,
                "Hi there! I'm sorry to hear that. Before we begin troubleshooting - how technically experienced are you?",
            ),
            (EventSource.CUSTOMER, "Not much, I just browse the internet on my iphone"),
            (
                EventSource.AI_AGENT,
                "I see, that's not a problem. Can you describe the exact issue you're experiencing?",
            ),
            (EventSource.CUSTOMER, "I type in google.com, but it doesn't load up"),
        ],
        expected_relevant=True,
        expected_distilled_action="provide basic internet troubleshooting steps",
    )


async def test_that_the_tech_experience_journey_selects_basic_non_internet_troubleshooting(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_TECH_CONDITION,
        action=_TECH_ACTION,
        conversation=[
            (
                EventSource.CUSTOMER,
                "I can't remember the password for my PC and I have no technological experience pls help me",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="provide basic non-internet (password-related) troubleshooting steps",
    )


# Investment-advice journey


async def test_that_the_investment_journey_recommends_aggressive_growth_for_a_young_long_term_investor(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_INVESTMENT_CONDITION,
        action=_INVESTMENT_ACTION,
        tools=[_REFER_TO_HUMAN_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi, I'm looking for investment advice"),
            (
                EventSource.AI_AGENT,
                "I'd be happy to help you with investment advice! To get started, could you tell me your age and describe your current financial situation?",
            ),
            (
                EventSource.CUSTOMER,
                "I'm 38 years old. Financially, I make about $100,000 a year as a software engineer, have about $25,000 in savings, and I'm contributing to my 401k. I don't have any major debts except my mortgage.",
            ),
            (
                EventSource.AI_AGENT,
                "Great, thank you. What's your risk tolerance, and what's your investment timeline - short term (under 5 years) or long term (5+ years)?",
            ),
            (
                EventSource.CUSTOMER,
                "I'd say I have a pretty high risk tolerance - I'm young and can handle some volatility if it means better long-term returns. And I'm definitely thinking long-term, probably 10-15 years.",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="recommend aggressive growth stocks and emerging market funds",
    )


# Book-flight journey


async def test_that_the_book_flight_journey_advances_to_asking_for_the_traveler_name(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_BOOK_FLIGHT_CONDITION,
        action=_BOOK_FLIGHT_ACTION,
        tools=[_BOOK_FLIGHT_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi, I'd like to book a flight please."),
            (
                EventSource.AI_AGENT,
                "I'd be happy to help you book a flight! Could you please tell me your source and destination airports?",
            ),
            (EventSource.CUSTOMER, "I want to fly from JFK in New York to LAX in Los Angeles."),
            (
                EventSource.AI_AGENT,
                "Great! And what dates would you like for your departure and return flights?",
            ),
            (
                EventSource.CUSTOMER,
                "Hmm, actually... I'm not entirely sure about the dates yet. Let me think about this and get back to you later.",
            ),
            (
                EventSource.AI_AGENT,
                "No problem at all! Take your time to figure out the dates. Is there anything else I can help you with in the meantime?",
            ),
            (
                EventSource.CUSTOMER,
                "Actually, you know what - I've decided. Let's depart on December 10th and return on December 17th. Can we book the flight now?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="ask for the name of the traveler or travelers",
    )


async def test_that_the_book_flight_journey_advances_to_asking_for_dates_for_a_new_destination(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_BOOK_FLIGHT_CONDITION,
        action=_BOOK_FLIGHT_ACTION,
        tools=[_BOOK_FLIGHT_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi, I'd like to book a flight please."),
            (
                EventSource.AI_AGENT,
                "I'd be happy to help you book a flight! Could you please tell me your source and destination airports?",
            ),
            (EventSource.CUSTOMER, "I want to fly from JFK in New York to LAX in Los Angeles."),
            (
                EventSource.AI_AGENT,
                "Great! And what dates would you like for your departure and return flights?",
            ),
            (
                EventSource.CUSTOMER,
                "Hmm, actually... I'm not entirely sure about the dates yet. Let me think about it.",
            ),
            (
                EventSource.AI_AGENT,
                "No problem at all! Is there anything else I can help you with in the meantime?",
            ),
            (
                EventSource.CUSTOMER,
                "Rome sounds perfect! Actually, can you help me book a flight from JFK to Rome instead? I'll figure out the LA trip another time.",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="ask for the dates of the departure and return flight",
    )


async def test_review_reset_password_journey_exits_when_the_customer_is_not_polite(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RESET_PASSWORD_CONDITION,
        action=_RESET_PASSWORD_ACTION,
        tools=[_RESET_PASSWORD_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your account number?"),
            (EventSource.CUSTOMER, "318475"),
            (EventSource.AI_AGENT, "Thank you. Now I need your email address or phone number."),
            (EventSource.CUSTOMER, "john.doe@email.com"),
            (EventSource.AI_AGENT, "Great! Have a good day!"),
            (EventSource.CUSTOMER, "Okay, thanks."),
        ],
        expected_relevant=True,
        expected_distilled_action="Do not reset the password",
    )


async def test_review_reset_password_journey_reuses_the_reset_tool_after_a_correction(
    distiller: GuidelineDistiller,
) -> None:
    # After the account correction the flow returns to an earlier step, but the email is
    # still valid - so the distiller should skip re-collecting it. Either wishing the
    # customer a good day (the next not-yet-confirmed step) or jumping straight to
    # re-running the reset tool is acceptable; re-asking for the email is not.
    staged_events = [
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:reset_password",
                            "arguments": {
                                "account_number": "318475",
                                "email": "john.doe@email.com",
                            },
                            "result": {
                                "data": "Password reset failed - account not found",
                                "metadata": {"error": "ACCOUNT_NOT_FOUND"},
                                "control": {},
                            },
                        }
                    ]
                },
            )
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RESET_PASSWORD_CONDITION,
        action=_RESET_PASSWORD_ACTION,
        tools=[_RESET_PASSWORD_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi there, I need to reset my password"),
            (EventSource.AI_AGENT, "I'm here to help you with that. What is your account number?"),
            (EventSource.CUSTOMER, "318475"),
            (EventSource.AI_AGENT, "Thank you. Now I need your email address or phone number."),
            (EventSource.CUSTOMER, "john.doe@email.com"),
            (EventSource.AI_AGENT, "Great! Have a good day!"),
            (EventSource.CUSTOMER, "Thank you, have a good day too!"),
            (
                EventSource.AI_AGENT,
                "I apologize, but the password could not be reset at this time since your account was not found.",
            ),
            (
                EventSource.CUSTOMER,
                "Oh wait, I think I gave you the wrong account number. It should be 987654, not 318475. Can we try again?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action=(
            "either wishing the customer a good day, or using the reset_password tool again "
            "with the corrected account number (but NOT re-asking for the email or phone number, "
            "which was already provided and is still valid)"
        ),
        staged_events=staged_events,
    )


async def test_review_calzone_journey_repeats_asking_for_the_size_when_not_provided(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi, I'd like to order some calzones"),
            (
                EventSource.AI_AGENT,
                "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
            ),
            (EventSource.CUSTOMER, "I'll take 3 calzones"),
            (
                EventSource.AI_AGENT,
                "Great! What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
            ),
            (EventSource.CUSTOMER, "I'll go with Classic Italian"),
            (EventSource.AI_AGENT, "Perfect! What size would you like - small, medium, or large?"),
            (EventSource.CUSTOMER, "Let me check for a sec"),
        ],
        expected_relevant=True,
        expected_distilled_action="wait for the customer to choose which size of calzone they want (small, medium, or large), or just ask them again",
    )


async def test_review_calzone_journey_rechecks_stock_when_reordering_the_same_items(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:check_stock",
                            "arguments": {"items": ["4 large Classic Italian Calzones"]},
                            "result": {
                                "data": {
                                    "all_available": True,
                                    "available_items": ["4 large Classic Italian Calzones"],
                                },
                                "metadata": {},
                                "control": {},
                            },
                        }
                    ]
                },
            )
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CALZONE_CONDITION,
        action=_CALZONE_ACTION,
        tools=[_CHECK_STOCK_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "I'd like to place an order"),
            (
                EventSource.AI_AGENT,
                "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
            ),
            (EventSource.CUSTOMER, "4 calzones please"),
            (
                EventSource.AI_AGENT,
                "What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
            ),
            (EventSource.CUSTOMER, "Classic Italian"),
            (EventSource.AI_AGENT, "What size would you like - small, medium, or large?"),
            (EventSource.CUSTOMER, "Large for all of them, please. I don't want any drinks btw"),
        ],
        expected_relevant=True,
        expected_distilled_action="confirm order details",
        staged_events=staged_events,
    )


# --- Level 4: completed guidelines re-triggered such that only part reapplies ---
#
# Each guideline has already been fully carried out once. The condition then arises
# again in a way that makes only a *subset* of the action relevant - a single changed
# input, or a different manner of applying the action. The distiller must re-take just
# that part and reuse or skip the rest, rather than redoing everything or nothing.


_DINNER_RESERVATION_CONDITION = "the customer wants to book a dinner reservation"
_DINNER_RESERVATION_ACTION = (
    "Ask for the party size, then for the preferred date and time, then whether anyone "
    "in the party has dietary restrictions, and finally confirm all the details and book "
    "the table using the book_table tool."
)

_BOOK_TABLE_TOOL = create_tool(
    "book_table",
    "Reserve a restaurant table given the party size, date, time, and any dietary notes.",
)


async def test_that_the_dinner_reservation_journey_rebooks_only_for_an_increased_party_size(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_DINNER_RESERVATION_CONDITION,
        action=_DINNER_RESERVATION_ACTION,
        tools=[_BOOK_TABLE_TOOL],
        conversation=[
            (EventSource.CUSTOMER, "Hi, I'd like to book a dinner reservation."),
            (EventSource.AI_AGENT, "Happy to help! How many people will be in your party?"),
            (EventSource.CUSTOMER, "Four of us."),
            (EventSource.AI_AGENT, "Great. What date and time would you like?"),
            (EventSource.CUSTOMER, "This Saturday at 8pm."),
            (
                EventSource.AI_AGENT,
                "Noted. Does anyone in your party have dietary restrictions?",
            ),
            (EventSource.CUSTOMER, "Yes, one of us is vegetarian."),
            (
                EventSource.AI_AGENT,
                "All set - I've booked a table for 4 this Saturday at 8pm, with a note "
                "about the vegetarian guest. See you then!",
            ),
            (
                EventSource.CUSTOMER,
                "Oh wait, two more friends just decided to join. Can we make it 6 instead of 4?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action=(
            "Either confirm the details of the booking or run the book_table tool"
        ),
    )


_RIDE_BOOKING_CONDITION = "the customer wants to book a ride"
_RIDE_BOOKING_ACTION = (
    "Confirm the pickup location with the customer and quote them the estimated fare."
)


async def test_that_a_ride_booking_guideline_requotes_the_fare_when_only_the_destination_changes(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RIDE_BOOKING_CONDITION,
        action=_RIDE_BOOKING_ACTION,
        conversation=[
            (EventSource.CUSTOMER, "I need a ride from JFK airport to the Hilton Midtown."),
            (
                EventSource.AI_AGENT,
                "Got it - I'll set your pickup at JFK Terminal 4 arrivals. The estimated fare is $52.",
            ),
            (
                EventSource.CUSTOMER,
                "Perfect, terminal 4 is right. Actually, can you change the destination to "
                "the Marriott near the convention center instead?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action=(
            "quote the updated estimated fare for the new destination, without re-confirming "
            "the pickup location, which is unchanged"
        ),
    )


_WIRE_TRANSFER_CONDITION = "the customer requests a wire transfer"
_WIRE_TRANSFER_ACTION = (
    "Confirm the recipient and amount with the customer, and let them know that the daily "
    "cutoff for same-day wires is 3 PM."
)


async def test_that_a_wire_transfer_guideline_reconfirms_only_the_new_recipient_on_a_second_wire(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_WIRE_TRANSFER_CONDITION,
        action=_WIRE_TRANSFER_ACTION,
        conversation=[
            (
                EventSource.CUSTOMER,
                "I'd like to wire $2,000 to my landlord, account ending 4471.",
            ),
            (
                EventSource.AI_AGENT,
                "To confirm, that's $2,000 to the account ending 4471. Just so you know, the "
                "daily cutoff for same-day wires is 3 PM. Shall I proceed?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, go ahead. I also need to send $500 to my sister, account ending 8830.",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action=(
            "confirm the new recipient and amount ($500 to the account ending 8830), without "
            "repeating the 3 PM same-day wire cutoff, which was already stated and is unchanged"
        ),
    )


_REFUND_PROCESS_CONDITION = "the customer asks how to get a refund"
_REFUND_PROCESS_ACTION = (
    "Explain the refund process that applies to the type of purchase in question."
)


async def test_that_a_refund_guideline_explains_the_physical_process_when_the_purchase_type_changes(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_REFUND_PROCESS_CONDITION,
        action=_REFUND_PROCESS_ACTION,
        conversation=[
            (EventSource.CUSTOMER, "How do I get a refund for the e-book I bought?"),
            (
                EventSource.AI_AGENT,
                "For digital downloads, refunds are automatic - we credit your original "
                "payment method within 24 hours, and there's nothing to send back.",
            ),
            (
                EventSource.CUSTOMER,
                "Got it. I also bought a blender from you last week that I'd like to refund - "
                "how does that one work?",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action=("explain the refund process for a physical item (the blender)"),
    )


_WORKOUT_RECOMMENDATION_CONDITION = "the customer asks for a workout recommendation"
_WORKOUT_RECOMMENDATION_ACTION = "Recommend a workout suited to the customer's stated goal."


async def test_that_a_workout_guideline_recommends_differently_when_the_goal_changes(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_WORKOUT_RECOMMENDATION_CONDITION,
        action=_WORKOUT_RECOMMENDATION_ACTION,
        conversation=[
            (
                EventSource.CUSTOMER,
                "Can you recommend a workout for me? I'm trying to lose weight.",
            ),
            (
                EventSource.AI_AGENT,
                "Sure! For weight loss, I'd suggest 30 minutes of brisk cardio - like jogging "
                "or cycling - most days of the week.",
            ),
            (
                EventSource.CUSTOMER,
                "I've been doing that for a few months and I'm happy with my weight now. I'd "
                "like to focus on building muscle instead.",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action=(
            "recommend a muscle-building workout - such as progressive strength/resistance "
            "training - suited to the customer's new goal"
        ),
    )
