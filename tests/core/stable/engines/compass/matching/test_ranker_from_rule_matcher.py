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
from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.matching.utils import (
    base_test_that_rules_are_ranked_correctly,
    create_capability,
    create_context_variable,
    create_staged_tool_event,
    create_term,
)


@fixture
def ranker(container: Container) -> RuleRanker:
    return container[RuleRanker]


RULES_DICT: dict[str, dict[str, str]] = {
    "check_drinks_in_stock": {
        "condition": "a customer asks for a drink",
        "action": "check if the drink is available in the following stock: "
        "['Sprite', 'Coke', 'Fanta']. Assume that if a drink is on stock, we have enough of it",
    },
    "check_toppings_in_stock": {
        "condition": "a customer asks for toppings",
        "action": "check if the toppings are available in the following stock: "
        "['Pepperoni', 'Tomatoes', 'Olives']. Assume that if a topping is on stock, we have enough of it",
    },
    "address_location": {
        "condition": "the customer needs to know our address",
        "action": "Inform the customer that our address is at Sapir 2, Herzliya.",
    },
    "payment_process": {
        "condition": "a customer is in the payment process",
        "action": "Follow the payment instructions, "
        "which are: 1. Pay in cash only, 2. Pay only at the location.",
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
    "frustrated_customer": {
        "condition": "the customer appears frustrated or upset",
        "action": "Acknowledge the customer's concerns, apologize for any inconvenience, and offer a solution or escalate the issue to a supervisor if necessary.",
    },
    "holiday_season": {
        "condition": "the interaction takes place during the holiday season",
        "action": "Mention any holiday-related offers, adjusted schedules, or greetings to make the interaction festive and accommodating.",
    },
    "request_for_feedback": {
        "condition": "the customer is asked for feedback about the service or product",
        "action": "Politely request their feedback, emphasizing its value for improvement, and provide simple instructions for submitting their response.",
    },
    "first_time_customer": {
        "condition": "the customer mentions it is their first time using the service",
        "action": "Welcome them warmly, provide a brief overview of how the service works, and offer any resources to help them get started.",
    },
    "suggest_drink_underage": {
        "condition": "an underage customer asks for drink recommendations",
        "action": "recommend a soda pop",
    },
    "suggest_drink_adult": {
        "condition": "an adult customer asks for drink recommendations",
        "action": "recommend either wine or beer",
    },
    "tree_allergies": {
        "condition": "recommending routes to a customer with tree allergies",
        "action": "warn the customer about allergy inducing trees along the route",
    },
    "credit_payment1": {
        "condition": "the customer requests a credit card payment",
        "action": "guide the customer through the payment process",
    },
    "credit_payment2": {
        "condition": "the customer wants to pay with a credit card",
        "action": "refuse payment as we only perform in-store purchases",
    },
    "cant_perform_request": {
        "condition": "the customer wants to agent to perform an action that you are not designed for",
        "action": "forward the request to a supervisor",
    },
    "announce_deals": {
        "condition": "A special deal is active",
        "action": "Announce the deal in an excited tone, while mentioning our slogan 'Ride the Future, One Kick at a Time!'",
    },
    "cheese_pizza": {
        "condition": "The customer is in the process of ordering a cheese pizza",
        "action": "Ask which toppings they would like",
    },
    "cheese_pizza_process": {
        "condition": "The customer is in the process of ordering a cheese pizza",
        "action": "Refer to the pizza as a 'pie'",
    },
    "summer_sale": {
        "condition": "In the season of summer",
        "action": "Mention we offer two large pizzas for the price of one",
    },
    "large_pizza_crust": {
        "condition": "The customer orders a large pizza",
        "action": "Ask what type of crust they would like",
    },
    "add_to_count": {
        "condition": "the customer asks you to add 1 to the count",
        "action": "Search the interaction history for the most recent count, add 1 to it and respond with the new count",
    },
    "cow_response": {"condition": "The customer says hello", "action": "respond like a cow would"},
    "many_actions": {
        "condition": "the customer asked a question about birds",
        "action": "answer their question enthusiastically, while not using punctuation. Also say that the kingfisher is your favorite bird",
    },
    "medical_record": {
        "condition": "you are likely to discuss a patient's medical record",
        "action": "Do not send any personal information",
    },
    "provide_diagnosis": {
        "condition": "you are likely to provide a diagnosis or medical advice.",
        "action": "Ensure the message includes a disclaimer that it is not a substitute for professional medical advice.",
    },
    "confirm_order": {
        "condition": "you are likely to confirm a new order or a payment",
        "action": "Re-verify item, price, and customer consent before proceeding",
    },
    "discuss_money": {
        "condition": "you are likely to discuss account balances or transactions.",
        "action": "Require customer authentication confirmation before responding.",
    },
    "human_resources": {
        "condition": "you are likely going to share a candidate’s application status",
        "action": "Avoid disclosing internal evaluation notes or third-party feedback",
    },
    "replace_card": {
        "condition": "The user wants to replace their card",
        "action": "List the cards and then assist the user to replace their card until matter is resolved",
    },
    "special_character_condition": {
        "condition": """The customer wishes to speak to either:
    1. a human agent
    2. A doctor / nurse / other medical professional
    3. a customer service representative
        """,
        "action": """Instruct them to call our office at this number:
        123-453-1212 and then choose "/" to speak with a human agent""",
    },
    "vegetarian_customer": {
        "condition": "the customer is vegetarian or vegan",
    },
    "ever_requested_lock_card": {
        "condition": "the customer ever indicated that they wish to lock their credit card",
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
        "condition": "The customer is asking for a service you don't recognize according to this prompt information",
    },
    "delivery_order": {
        "condition": "the customer is in the process of ordering delivery",
    },
    "unanswered_questions": {
        "condition": "the customer repeatedly ignores the agent's question, and they remain unanswered",
    },
    "unsupported_capability": {
        "condition": "When a customer asks about a capability that is not supported",
    },
    "reset_password": {
        "condition": "The customer currently wants to reset their password",
    },
    "lost_card": {
        "condition": "The customer says that they lost their card",
    },
    "book_flight": {
        "condition": "The customer wants to book a flight",
    },
    "book_flight_2": {
        "condition": "The conversation is about flight booking",
    },
}


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_relevant_rules_are_matched_parametrized_2
async def test_that_relevant_rules_are_matched_parametrized_2(ranker: RuleRanker) -> None:
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
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "class_booking",
            "issue_resolved",
            "address_location",
        ],
        relevant_rule_names=["issue_resolved"],
        irrelevant_rule_names=["class_booking", "address_location"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_irrelevant_rules_are_not_matched_parametrized_1
async def test_that_irrelevant_rules_are_not_matched_parametrized_1(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "I'd like to order a pizza, please."),
        (EventSource.AI_AGENT, "No problem. What would you like to have?"),
        (EventSource.CUSTOMER, "I'd like a large pizza. What toppings do you have?"),
        (EventSource.AI_AGENT, "Today we have pepperoni, tomatoes, and olives available."),
        (EventSource.CUSTOMER, "I'll take pepperoni, thanks."),
        (
            EventSource.AI_AGENT,
            "Awesome. I've added a large pepperoni pizza. Would you like a drink on the side?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure. What types of drinks do you have?",
        ),
        (
            EventSource.AI_AGENT,
            "We have Sprite, Coke, and Fanta.",
        ),
        (EventSource.CUSTOMER, "I'll take two Sprites, please."),
        (EventSource.AI_AGENT, "Anything else?"),
        (EventSource.CUSTOMER, "No, that's all."),
        (EventSource.AI_AGENT, "How would you like to pay?"),
        (EventSource.CUSTOMER, "I'll pick it up and pay in cash, thanks."),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["check_toppings_in_stock", "check_drinks_in_stock"],
        relevant_rule_names=[],
        irrelevant_rule_names=["check_toppings_in_stock", "check_drinks_in_stock"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rules_are_matched_based_on_agent_description
async def test_that_rules_are_matched_based_on_agent_description(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey, do you sell skateboards?"),
        (
            EventSource.AI_AGENT,
            "Yes, we do! We have a variety of skateboards for all skill levels. Are you looking for something specific?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking for a skateboard for a beginner. What do you recommend?",
        ),
        (
            EventSource.AI_AGENT,
            "For beginners, I recommend our complete skateboards with a sturdy deck and softer wheels for easier control. Would you like to see some options?",
        ),
        (EventSource.CUSTOMER, "That sounds perfect. Can you show me a few?"),
        (
            EventSource.AI_AGENT,
            "Sure! We have a few options: the 'Smooth Ride' model, the 'City Cruiser,' and the 'Basic Starter.' Which one would you like to know more about?",
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
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["cant_perform_request"],
        relevant_rule_names=["cant_perform_request"],
        irrelevant_rule_names=[],
        agent_description="You are an agent working for a skateboarding manufacturer. You help customers by discussing and recommending our products."
        "Your role is only to consult customers, and not to actually sell anything, as we sell our products in-store.",
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rules_are_matched_based_on_glossary
async def test_that_rules_are_matched_based_on_glossary(ranker: RuleRanker) -> None:
    terms = [
        create_term(
            name="skateboard",
            description="a time-traveling device",
            groups=[],
        ),
        create_term(
            name="Pinewood Rash Syndrome",
            description="allergy to pinewood trees",
            synonyms=["Pine Rash", "PRS"],
            groups=[],
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I'm looking for a hiking route through a forest. Can you help me?",
        ),
        (
            EventSource.AI_AGENT,
            "Of course! I can help you find a trail. Are you looking for an easy, moderate, or challenging hike?",
        ),
        (
            EventSource.CUSTOMER,
            "I'd prefer something moderate, not too easy but also not too tough.",
        ),
        (
            EventSource.AI_AGENT,
            "Great choice! We have a few moderate trails in the Redwood Forest and the Pinewood Trail. Would you like details on these?",
        ),
        (EventSource.CUSTOMER, "Yes, tell me more about the Pinewood Trail."),
        (
            EventSource.AI_AGENT,
            "The Pinewood Trail is a 6-mile loop with moderate elevation changes. It takes about 3-4 hours to complete. The scenery is beautiful, with plenty of shade and a stream crossing halfway through. Would you like to go with that one?",
        ),
        (EventSource.CUSTOMER, "I have PRS, would that route be suitable for me?"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["tree_allergies"],
        relevant_rule_names=["tree_allergies"],
        irrelevant_rule_names=[],
        terms=terms,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_conflicting_actions_with_similar_conditions_are_both_detected
async def test_that_conflicting_actions_with_similar_conditions_are_both_detected(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey, do you sell skateboards?"),
        (
            EventSource.AI_AGENT,
            "Yes, we do! We have a variety of skateboards for all skill levels. Are you looking for something specific?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking for a skateboard for a beginner. What do you recommend?",
        ),
        (
            EventSource.AI_AGENT,
            "For beginners, I recommend our complete skateboards with a sturdy deck and softer wheels for easier control. Would you like to see some options?",
        ),
        (
            EventSource.CUSTOMER,
            "That sounds perfect. Can you show me a few?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! We have a few options: the 'Smooth Ride' model, the 'City Cruiser,' and the 'Basic Starter.' Which one would you like to know more about?",
        ),
        (
            EventSource.CUSTOMER,
            "I like the 'City Cruiser.' What color options do you have?",
        ),
        (
            EventSource.AI_AGENT,
            "The 'City Cruiser' comes in red, blue, and black. Which one do you prefer?",
        ),
        (
            EventSource.CUSTOMER,
            "I'll go with the blue one.",
        ),
        (
            EventSource.AI_AGENT,
            "Great choice! I'll add the blue 'City Cruiser' to your cart. Would you like to add any accessories like a helmet or grip tape?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, I'll take a helmet. What do you have in stock?",
        ),
        (
            EventSource.AI_AGENT,
            "We have helmets in small, medium, and large sizes, all available in black and gray. What size do you need?",
        ),
        (
            EventSource.CUSTOMER,
            "I need a medium. I'll take one in black.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it! Your blue 'City Cruiser' skateboard and black medium helmet are ready for checkout. How would you like to pay?",
        ),
        (
            EventSource.CUSTOMER,
            "I'll pay with a credit card, thanks.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["credit_payment1", "credit_payment2"],
        relevant_rule_names=["credit_payment1", "credit_payment2"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rules_are_matched_based_on_staged_tool_calls_and_context_variables
async def test_that_rules_are_matched_based_on_staged_tool_calls_and_context_variables(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there, I want a drink that's on the sweeter side, what would you suggest?",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! Let me take a quick look at your account to recommend the best product for you. Could you please provide your full name?",
        ),
        (EventSource.CUSTOMER, "I'm Bob Bobberson"),
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
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:get_user_age",
                            "arguments": {"user_id": "816779"},
                            "result": {"data": 30, "metadata": {}, "control": {}},
                        }
                    ]
                },
            )
        ),
    ]
    context_variables = [
        create_context_variable(
            name="user_id_1",
            data={"name": "Jimmy McGill", "ID": 566317},
            groups=[],
        ),
        create_context_variable(
            name="user_id_2",
            data={"name": "Bob Bobberson", "ID": 199877},
            groups=[],
        ),
        create_context_variable(
            name="user_id_3",
            data={"name": "Dorothy Dortmund", "ID": 816779},
            groups=[],
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["suggest_drink_underage", "suggest_drink_adult"],
        relevant_rule_names=["suggest_drink_underage"],
        irrelevant_rule_names=["suggest_drink_adult"],
        context_variables=context_variables,
        staged_events=staged_events,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rules_are_matched_based_on_staged_tool_calls_without_context_variables
async def test_that_rules_are_matched_based_on_staged_tool_calls_without_context_variables(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there, I want a drink that's on the sweeter side, what would you suggest?",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! Let me take a quick look at your account to recommend the best product for you. Could you please provide your ID number?",
        ),
        (EventSource.CUSTOMER, "It's 199877"),
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
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:get_user_age",
                            "arguments": {"user_id": "816779"},
                            "result": {"data": 30, "metadata": {}, "control": {}},
                        }
                    ]
                },
            )
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["suggest_drink_underage", "suggest_drink_adult"],
        relevant_rule_names=["suggest_drink_underage"],
        irrelevant_rule_names=["suggest_drink_adult"],
        staged_events=staged_events,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_already_addressed_rules_are_not_matched
async def test_that_already_addressed_rules_are_not_matched(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey there, can I get one cheese pizza?"),
        (EventSource.AI_AGENT, "Of course! What toppings would you like?"),
        (EventSource.CUSTOMER, "Mushrooms if they're fresh"),
        (
            EventSource.AI_AGENT,
            "All of our toppings are fresh! Are you collecting it from our shop or should we ship it to your address?",
        ),
        (EventSource.CUSTOMER, "Ship it to my address please"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["cheese_pizza"],
        relevant_rule_names=[],
        irrelevant_rule_names=["cheese_pizza"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rules_referring_to_continuous_processes_are_detected_even_if_already_fulfilled
async def test_that_rules_referring_to_continuous_processes_are_detected_even_if_already_fulfilled(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey there, can I get one cheese pizza?"),
        (
            EventSource.AI_AGENT,
            "Of course! What toppings would you like on your pie?",
        ),
        (EventSource.CUSTOMER, "Mushrooms if they're fresh"),
        (
            EventSource.AI_AGENT,
            "All of our toppings are fresh! Are you collecting the pie from our shop or should we ship it to your address?",
        ),
        (EventSource.CUSTOMER, "Ship it to my address please"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["cheese_pizza_process"],
        relevant_rule_names=["cheese_pizza_process"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_already_addressed_condition_but_unaddressed_action_is_matched
async def test_that_rule_with_already_addressed_condition_but_unaddressed_action_is_matched(
    ranker: RuleRanker,
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
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["frustrated_customer"],
        relevant_rule_names=["frustrated_customer"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_is_not_detected_based_on_its_action
async def test_that_rule_is_not_detected_based_on_its_action(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "There's currently a 20 percent discount on all items! Ride the Future, One Kick at a Time!",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["announce_deals"],
        relevant_rule_names=["announce_deals"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_fulfilled_action_regardless_of_condition_can_be_reapplied
async def test_that_rule_with_fulfilled_action_regardless_of_condition_can_be_reapplied(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "The count is on 0! Your turn",
        ),
        (
            EventSource.AI_AGENT,
            "I choose to add to the count. The count is now 2.",
        ),
        (
            EventSource.CUSTOMER,
            "add one to the count please",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["add_to_count"],
        relevant_rule_names=["add_to_count"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_initial_response_is_matched
async def test_that_rule_with_initial_response_is_matched(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hello!",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["cow_response"],
        relevant_rule_names=["cow_response"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_multiple_actions_is_partially_fulfilled_when_a_few_actions_occurred
async def test_that_rule_with_multiple_actions_is_partially_fulfilled_when_a_few_actions_occurred(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there! I was wondering - what's the life expectancy of owls?",
        ),
        (
            EventSource.AI_AGENT,
            "Owls are amazing depending on the species owls can live 5 to 30 years in the wild and even longer in captivity wow owls are incredible",
        ),
        (
            EventSource.CUSTOMER,
            "That's shorter than I expected, thank you!",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["many_actions"],
        relevant_rule_names=[],
        irrelevant_rule_names=["many_actions"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_detected_1
async def test_that_observational_rules_are_detected_1(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I want to order a pizza. Which toppings do you have?",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! We have pepperoni, tomatoes, mushrooms and olives",
        ),
        (
            EventSource.CUSTOMER,
            "Oh, I'm on a plant-based diet. Do you have pizzas that I could eat?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["vegetarian_customer"],
        relevant_rule_names=["vegetarian_customer"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_irrelevant_observational_rules_are_not_detected_1
async def test_that_irrelevant_observational_rules_are_not_detected_1(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I want to order a pizza. Which toppings do you have?",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! We have pepperoni, tomatoes, mushrooms and olives",
        ),
        (
            EventSource.CUSTOMER,
            "I don't like pepperoni, so I guess I'll go with mushrooms",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["vegetarian_customer"],
        relevant_rule_names=[],
        irrelevant_rule_names=["vegetarian_customer"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_detected_2
async def test_that_observational_rules_are_detected_2(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I didn't get any help from the previous representative. If this continues I'll switch to the competitors. Don't thread on me!",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I apologize for what happened on your previous interaction with us - what is it that you're trying to do exactly?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking to modify an order I made through the online store",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["frustrated_customer_observational"],
        relevant_rule_names=["frustrated_customer_observational"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_irrelevant_observational_rules_are_not_detected_2
async def test_that_irrelevant_observational_rules_are_not_detected_2(
    ranker: RuleRanker,
) -> None:
    context_variables = [
        create_context_variable(
            name="customer_location",
            data={"location": "Australia"},
            groups=[],
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hello, I need some banking help today",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I'd be happy to help with your banking needs. What specific assistance are you looking for today?",
        ),
        (
            EventSource.CUSTOMER,
            "I want a new account",
        ),
        (
            EventSource.AI_AGENT,
            "Sure thing! Do you know what kind of account you're looking for? Is it personal or for business?",
        ),
        (
            EventSource.CUSTOMER,
            "hi",
        ),
        (
            EventSource.AI_AGENT,
            "Hello! I see you were interested in opening a new account. I'd be happy to help with that. We offer several account types:\n\n1. Personal checking accounts\n2. Personal savings accounts\n3. Business accounts\n4. Investment accounts\n\nWhich one would you like to learn more about?",
        ),
        (
            EventSource.CUSTOMER,
            "the personal saving one",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "unanswered_questions",
            "unknown_service",
            "delivery_order",
        ],
        relevant_rule_names=[],
        irrelevant_rule_names=[
            "unanswered_questions",
            "unknown_service",
            "delivery_order",
        ],
        context_variables=context_variables,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_detected_3
async def test_that_observational_rules_are_detected_3(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I want one family size mushroom pie",
        ),
        (
            EventSource.AI_AGENT,
            "Got it! Would you like to pick it up from our store? It will take at least 15 minutes to prepare.",
        ),
        (
            EventSource.CUSTOMER,
            "Do you guys deliver to the city?",
        ),
        (
            EventSource.AI_AGENT,
            "We do! Would you like to pick up your order or to have it delivered?",
        ),
        (
            EventSource.CUSTOMER,
            "Let's just say that I'd like to play the old tambourine, can you help me do that?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "delivery_order",
            "unknown_service",
        ],
        relevant_rule_names=["unknown_service"],
        irrelevant_rule_names=["delivery_order"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_detected_5
async def test_that_observational_rules_are_detected_5(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you sign me up to a saving plan based on my current balance?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "unknown_service",
        ],
        relevant_rule_names=["unknown_service"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_detected_4
async def test_that_observational_rules_are_detected_4(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I'm having trouble with my credit card statement.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear you're having trouble with your credit card statement. I'd be happy to help. Could you please let me know what specific issue you're experiencing?",
        ),
        (
            EventSource.CUSTOMER,
            "I made a payment two weeks ago but it's not showing up.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for providing that information. I understand how concerning it can be when a payment doesn't appear on your statement. To help you better, could you tell me when exactly you made the payment, how much it was for, and which payment method you used?",
        ),
        (
            EventSource.CUSTOMER,
            "It was $500 on March 15th using the online banking system.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for those details. I'll look into this for you right away. Sometimes online payments can take 1-3 business days to process, but it definitely should have appeared by now. Let me check our records to see if there's any trace of your $500 payment from March 15th. May I have the last four digits of your credit card to verify your account?",
        ),
        (
            EventSource.CUSTOMER,
            "The last four digits are 4321.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for providing that information. I can see your account ending in 4321, and I'm checking for the $500 payment from March 15th. I do see a payment was initiated on that date, but it appears there might have been a processing delay. The good news is that the payment has been applied to your account as of March 18th. It should be reflected in your next statement. Is there anything else about your credit card account I can help you with today?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, I'd like to add interest on my credit limits.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["unknown_service", "credit_limits_discussion"],
        relevant_rule_names=["unknown_service", "credit_limits_discussion"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_detected_based_on_context_variables
async def test_that_observational_rules_are_detected_based_on_context_variables(
    ranker: RuleRanker,
) -> None:
    context_variables = [
        create_context_variable(
            name="user_id_1",
            data={"name": "Jimmy McGill", "ID": 566317},
            groups=[],
        ),
        create_context_variable(
            name="season",
            data={"season": "Winter"},
            groups=[],
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I didn't get any help from the previous representative. If this continues I'll switch to the competitors. Don't thread on me!",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I apologize for what happened on your previous interaction with us - what is it that you're trying to do exactly?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking to modify an order I made through the online store",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["season_is_winter"],
        relevant_rule_names=["season_is_winter"],
        irrelevant_rule_names=[],
        context_variables=context_variables,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_detected_based_on_tool_results
async def test_that_observational_rules_are_detected_based_on_tool_results(
    ranker: RuleRanker,
) -> None:
    staged_events = [
        create_staged_tool_event(
            cast(
                JSONSerializable,
                {
                    "tool_calls": [
                        {
                            "tool_id": "local:get_season",
                            "arguments": {},
                            "result": {"data": "winter", "metadata": {}, "control": {}},
                        }
                    ]
                },
            )
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I didn't get any help from the previous representative. If this continues I'll switch to the competitors. Don't thread on me!",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I apologize for what happened on your previous interaction with us - what is it that you're trying to do exactly?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking to modify an order I made through the online store",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "season_is_winter",
            "ever_requested_lock_card",
            "lost_card",
        ],
        relevant_rule_names=["season_is_winter"],
        irrelevant_rule_names=["ever_requested_lock_card", "lost_card"],
        staged_events=staged_events,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_matched_based_on_glossary
async def test_that_observational_rules_are_matched_based_on_glossary(
    ranker: RuleRanker,
) -> None:
    terms = [
        create_term(
            name="play the old tambourine",
            description="local slang for getting your order delivered to your home",
            groups=[],
        ),
    ]
    capabilities = [
        create_capability(
            title="Delivery",
            description="The ability to deliver orders of pizza",
            id="cap_123",
            signals=["delivery"],
            groups=[],
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I want one family size mushroom pie",
        ),
        (
            EventSource.AI_AGENT,
            "Got it! Would you like to pick it up from our store? It will take at least 15 minutes to prepare.",
        ),
        (
            EventSource.CUSTOMER,
            "Do you guys deliver to the city?",
        ),
        (
            EventSource.AI_AGENT,
            "We do! Would you like to pick up your order or to have it delivered?",
        ),
        (
            EventSource.CUSTOMER,
            "Let's just say that I'd like to play the old tambourine, can you help me with that?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "delivery_order",
            "unknown_service",
            "unanswered_questions",
        ],
        relevant_rule_names=["delivery_order"],
        irrelevant_rule_names=["unknown_service", "unanswered_questions"],
        terms=terms,
        capabilities=capabilities,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_matched_based_on_vague_customer_message
async def test_that_observational_rules_are_matched_based_on_vague_customer_message(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "credit limits",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["credit_limits_discussion"],
        relevant_rule_names=["credit_limits_discussion"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_matched_based_on_old_messages
async def test_that_observational_rules_are_matched_based_on_old_messages(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hello, I need to lock my credit card immediately. I think I lost it.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear about your lost card. I can help you lock it right away. Could you please verify the last four digits of the card you need to lock?",
        ),
        (
            EventSource.CUSTOMER,
            "It's 7890.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for verifying. I've placed a temporary lock on your credit card ending in 7890. This will prevent any new purchases, cash advances, or balance transfers. The lock is effective immediately. Would you like me to order a replacement card for you as well?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes please, send it to my home address.",
        ),
        (
            EventSource.AI_AGENT,
            "I've ordered a replacement card to be sent to your home address on file. You should receive it within 5-7 business days. Is there anything else I can assist you with regarding your card?",
        ),
        (
            EventSource.CUSTOMER,
            "That's all for the card. Actually, I've been thinking about investing. What stocks do you think I should buy?",
        ),
        (
            EventSource.AI_AGENT,
            "I understand you're interested in exploring investment opportunities. While I can't recommend specific stocks, I'd be happy to discuss our investment services. We offer managed portfolios, self-directed trading accounts, and retirement investment options. What are your investment goals? Are you looking for long-term growth, income generation, or something else?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm mostly interested in tech stocks for long-term growth.",
        ),
        (
            EventSource.AI_AGENT,
            "Long-term growth in the technology sector is a popular investment strategy. Our bank offers several ways to invest in technology. You could consider our Technology Sector Fund which provides diversified exposure to various tech companies, or our self-directed trading platform if you prefer selecting individual stocks. Would you like me to connect you with one of our investment advisors who can provide more detailed information based on your specific financial situation?",
        ),
        (
            EventSource.CUSTOMER,
            "How much would I need to start with the Technology Sector Fund?",
        ),
        (
            EventSource.AI_AGENT,
            "Our Technology Sector Fund has a minimum initial investment of $1,000. After that, you can make additional investments of $100 or more at any time. The fund has an expense ratio of 0.85%, which is competitive for actively managed sector funds. Would you like me to send you our fund prospectus with more detailed information about historical performance and holdings?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, please email me the prospectus. And what about cryptocurrency investments?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["ever_requested_lock_card", "lost_card"],
        relevant_rule_names=["ever_requested_lock_card"],
        irrelevant_rule_names=["lost_card"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_not_matched_based_when_topic_was_shifted
async def test_that_observational_rules_are_not_matched_based_when_topic_was_shifted(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I forgot my password. Can you help me reset it?",
        ),
        (
            EventSource.AI_AGENT,
            "Of course, I'd be happy to help. Can you please provide your account name?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, it's jenny_the_cat89",
        ),
        (
            EventSource.AI_AGENT,
            "Thanks! Now, could you share the email address or phone number associated with your account?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure, it's jenny@example.com",
        ),
        (
            EventSource.AI_AGENT,
            "Great. I hope you're having a lovely day!",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks, you too!",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you! Resetting your password now...",
        ),
        (
            EventSource.AI_AGENT,
            "Your password has been successfully reset. Please check your email for further instructions.",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks! Also, I'd like to change my credit limit.",
        ),
        (
            EventSource.AI_AGENT,
            "I'd be glad to help with that. Could you tell me what you'd like your new credit limit to be?",
        ),
        (
            EventSource.CUSTOMER,
            "I'd like to increase it to $5,000.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["reset_password", "credit_limits_discussion"],
        relevant_rule_names=["credit_limits_discussion"],
        irrelevant_rule_names=["reset_password"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_matched_when_conversation_is_on_sub_topic
async def test_that_observational_rules_are_matched_when_conversation_is_on_sub_topic(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, I need to book a flight.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Can you please tell me your departure and destination airports?",
        ),
        (
            EventSource.CUSTOMER,
            "Flying from JFK to LAX.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. What date would you like to travel?",
        ),
        (
            EventSource.CUSTOMER,
            "July 18th.",
        ),
        (
            EventSource.AI_AGENT,
            "And would you prefer economy or business class?",
        ),
        (
            EventSource.CUSTOMER,
            "Business class, please.",
        ),
        (
            EventSource.AI_AGENT,
            "Perfect. Lastly, can I have the name of the traveler?",
        ),
        (
            EventSource.CUSTOMER,
            "Jennifer Morales.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["book_flight", "book_flight_2"],
        relevant_rule_names=["book_flight", "book_flight_2"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_both_observational_and_actionable_rules_are_matched_together
async def test_that_both_observational_and_actionable_rules_are_matched_together(
    ranker: RuleRanker,
) -> None:
    context_variables = [
        create_context_variable(
            name="season",
            data={"season": "Spring"},
            groups=[],
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there, I'm looking for a class to help me relax. It's been a stressful winter.",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome! I understand that winter can be stressful. We have several relaxation classes available. Would you like to hear about our meditation or yoga options?",
        ),
        (
            EventSource.CUSTOMER,
            "I'd be interested in booking a meditation class, but I'm not sure which one is right for me.",
        ),
        (
            EventSource.AI_AGENT,
            "We have beginner meditation every Monday at 6 PM, and advanced sessions on Thursdays at 7 PM. Both are excellent for stress relief. Which would work better for your schedule?",
        ),
        (
            EventSource.CUSTOMER,
            "Monday at 6 PM sounds perfect. How do I book it?",
        ),
        (
            EventSource.AI_AGENT,
            "Great choice! I can book you for the Monday 6 PM meditation class. Could you please provide your name and contact information?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm Taylor Smith, phone is 555-123-4567. By the way, do you have any vegan food options in your café?",
        ),
        (
            EventSource.AI_AGENT,
            "Thanks, Taylor! I've booked your Monday 6 PM meditation class. And yes, our café offers several vegan options including smoothies, salads, and plant-based protein bowls. Would you like to order something to enjoy after your class?",
        ),
        (
            EventSource.CUSTOMER,
            "Not right now, thank you. Oh, I just realized - I might be running late. Where exactly is your location?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "vegetarian_customer",
            "season_is_winter",
            "frustrated_customer_observational",
            "unclear_request",
            "credit_limits_discussion",
            "unknown_service",
            "delivery_order",
            "unanswered_questions",
            "ever_requested_lock_card",
            "lost_card",
            "address_location",
            "class_booking",
            "holiday_season",
            "first_time_customer",
            "request_for_feedback",
            "large_pizza_crust",
            "announce_deals",
            "summer_sale",
            "frustrated_customer",
        ],
        relevant_rule_names=[
            "vegetarian_customer",
            "address_location",
        ],
        irrelevant_rule_names=[
            "season_is_winter",
            "frustrated_customer_observational",
            "unclear_request",
            "credit_limits_discussion",
            "unknown_service",
            "delivery_order",
            "unanswered_questions",
            "ever_requested_lock_card",
            "lost_card",
            "class_booking",
            "holiday_season",
            "first_time_customer",
            "request_for_feedback",
            "large_pizza_crust",
            "announce_deals",
            "summer_sale",
            "frustrated_customer",
        ],
        context_variables=context_variables,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_irrelevant_rules_are_not_matched_parametrized_2
async def test_that_irrelevant_rules_are_not_matched_parametrized_2(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Could you add some pretzels to my order?"),
        (EventSource.AI_AGENT, "Pretzels have been added to your order. Anything else?"),
        (EventSource.CUSTOMER, "Do you have Coke? I'd like one, please."),
        (EventSource.AI_AGENT, "Coke has been added to your order."),
        (EventSource.CUSTOMER, "Great, where are you located at?"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["check_drinks_in_stock"],
        relevant_rule_names=[],
        irrelevant_rule_names=["check_drinks_in_stock"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_is_matched
async def test_that_rule_with_agent_intention_is_matched(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, can you let me know what my recent lab results say?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["medical_record"],
        relevant_rule_names=["medical_record"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_is_matched_2
async def test_that_rule_with_agent_intention_is_matched_2(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I've had a sore throat and a fever for three days. Do you think it’s strep?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "provide_diagnosis",
            "confirm_order",
            "discuss_money",
        ],
        relevant_rule_names=["provide_diagnosis"],
        irrelevant_rule_names=["confirm_order", "discuss_money"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_is_matched_3
async def test_that_rule_with_agent_intention_is_matched_3(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey do you sell iPhone 15?",
        ),
        (
            EventSource.AI_AGENT,
            "Absolutely! would you like to buy one?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, go ahead and place the order for the iPhone 15.",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["confirm_order"],
        relevant_rule_names=["confirm_order"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_is_matched_4
async def test_that_rule_with_agent_intention_is_matched_4(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you tell me my last 5 transactions?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["discuss_money"],
        relevant_rule_names=["discuss_money"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_is_matched_5
async def test_that_rule_with_agent_intention_is_matched_5(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Just checking in - any update on my interview from last week?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["human_resources"],
        relevant_rule_names=["human_resources"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_is_not_matched
async def test_that_rule_with_agent_intention_is_not_matched(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey do you sell iPhone 15?",
        ),
        (
            EventSource.AI_AGENT,
            "Absolutely! would you like to buy one?",
        ),
        (
            EventSource.CUSTOMER,
            "How much does it cost?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["confirm_order"],
        relevant_rule_names=[],
        irrelevant_rule_names=["confirm_order"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_is_not_matched_2
async def test_that_rule_with_agent_intention_is_not_matched_2(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, do you sell iPhone 15? I want to buy one",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["confirm_order"],
        relevant_rule_names=[],
        irrelevant_rule_names=["confirm_order"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_and_customer_dependent_action_that_was_previously_applied_is_matched
async def test_that_rule_with_agent_intention_and_customer_dependent_action_that_was_previously_applied_is_matched(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey do you sell iPhone 15?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes, we do! Would you like to place an order?",
        ),
        (
            EventSource.CUSTOMER,
            "How much does it cost?",
        ),
        (
            EventSource.AI_AGENT,
            "It’s currently on sale for $5,000",
        ),
        (
            EventSource.CUSTOMER,
            "Sounds good so I want to order one",
        ),
        (
            EventSource.AI_AGENT,
            "Great, so before proceeding I want to confirm - you like to order one iPhone 15 for 5000$",
        ),
        (
            EventSource.CUSTOMER,
            "Hmm let me check",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["confirm_order"],
        relevant_rule_names=["confirm_order"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_that_was_previously_applied_is_matched
async def test_that_rule_with_agent_intention_that_was_previously_applied_is_matched(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I've had a sore throat and a fever for three days. Do you think it’s strep?",
        ),
        (
            EventSource.AI_AGENT,
            "I'm not a medical professional, so I can't provide a diagnosis. However, a sore "
            "throat and fever can be symptoms of several conditions, including strep throat",
        ),
        (
            EventSource.CUSTOMER,
            "Okay, but if it is strep, can I just take antibiotics I have left over from last time?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["provide_diagnosis"],
        relevant_rule_names=["provide_diagnosis"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_that_was_previously_applied_but_should_not_reapply_is_not_matched
async def test_that_rule_with_agent_intention_that_was_previously_applied_but_should_not_reapply_is_not_matched(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I've had a sore throat and a fever for three days. Do you think it’s strep?",
        ),
        (
            EventSource.AI_AGENT,
            "I'm not a medical professional, so I can't provide a diagnosis. However, a sore "
            "throat and fever can be symptoms of several conditions, including strep throat",
        ),
        (
            EventSource.CUSTOMER,
            "Alright, I’ll try to see a doctor soon. Also, can you remind me how to update my insurance information on the website?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["provide_diagnosis"],
        relevant_rule_names=[],
        irrelevant_rule_names=["provide_diagnosis"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_rule_with_agent_intention_that_was_matched_but_action_wasnt_taken_is_not_matched_again
async def test_that_rule_with_agent_intention_that_was_matched_but_action_wasnt_taken_is_not_matched_again(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey do you sell iPhone 15?",
        ),
        (
            EventSource.AI_AGENT,
            "Absolutely! would you like to buy one?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, go ahead and place the order for the iPhone 15",
        ),
        (
            EventSource.AI_AGENT,
            "Great so I ordered you one iPhone 15. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "How much did it cost by the way?",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["confirm_order"],
        relevant_rule_names=[],
        irrelevant_rule_names=["confirm_order"],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_matched_based_on_capabilities_1
async def test_that_observational_rules_are_matched_based_on_capabilities_1(
    ranker: RuleRanker,
) -> None:
    capabilities = [
        create_capability(
            title="Reset Password",
            description="The ability to send the customer an email with a link to reset their password. The password can only be reset via this link",
            id="cap_123",
            signals=["reset password", "password"],
            groups=[],
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "can you set my password to 4321?"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["unsupported_capability"],
        relevant_rule_names=["unsupported_capability"],
        irrelevant_rule_names=[],
        capabilities=capabilities,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_are_not_falsely_matched_based_on_capabilities
async def test_that_observational_rules_are_not_falsely_matched_based_on_capabilities(
    ranker: RuleRanker,
) -> None:
    capabilities = [
        create_capability(
            title="Reset Password",
            description="The ability to send the customer an email with a link to reset their password. The password can only be reset via this link",
            id="cap_123",
            signals=["reset password", "password"],
            groups=[],
        ),
    ]
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "can you reset my password?"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["unsupported_capability"],
        relevant_rule_names=[],
        irrelevant_rule_names=["unsupported_capability"],
        capabilities=capabilities,
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_a_rule_that_has_several_steps_is_still_matched
async def test_that_a_rule_that_has_several_steps_is_still_matched(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "replace",
        ),
        (
            EventSource.AI_AGENT,
            "Can you select out of the following cards which card do you want to replace?",
        ),
        (
            EventSource.AI_AGENT,
            "1. C11223344 2.D1212121",
        ),
        (
            EventSource.CUSTOMER,
            "First one",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "replace_card",
        ],
        relevant_rule_names=[
            "replace_card",
        ],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/stable/engines/alpha/test_rule_matcher.py::test_that_condition_with_special_characters_causes_no_errors
async def test_that_condition_with_special_characters_causes_no_errors(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "I want to talk to a nurse!!!"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["special_character_condition"],
        relevant_rule_names=["special_character_condition"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/unstable/engines/alpha/test_rule_matcher.py::test_that_relevant_rules_are_matched_parametrized_1
async def test_that_relevant_rules_are_matched_parametrized_1(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "I'd like to order a pizza, please."),
        (EventSource.AI_AGENT, "No problem. What would you like to have?"),
        (EventSource.CUSTOMER, "I'd like a large pizza. What toppings do you have?"),
        (EventSource.AI_AGENT, "Today, we have pepperoni, tomatoes, and olives available."),
        (EventSource.CUSTOMER, "I'll take pepperoni, thanks."),
        (
            EventSource.AI_AGENT,
            "Awesome. I've added a large pepperoni pizza. Would you like a drink on the side?",
        ),
        (EventSource.CUSTOMER, "Sure. What types of drinks do you have?"),
        (EventSource.AI_AGENT, "We have Sprite, Coke, and Fanta."),
        (EventSource.CUSTOMER, "I'll take two Sprites, please."),
        (EventSource.AI_AGENT, "Anything else?"),
        (EventSource.CUSTOMER, "No, that's all. I want to pay."),
        (EventSource.AI_AGENT, "No problem! We accept only cash."),
        (EventSource.CUSTOMER, "Sure, I'll pay the delivery guy."),
        (EventSource.AI_AGENT, "Unfortunately, we accept payments only at our location."),
        (EventSource.CUSTOMER, "So what should I do now?"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=[
            "check_toppings_in_stock",
            "check_drinks_in_stock",
            "payment_process",
            "address_location",
        ],
        relevant_rule_names=[
            "address_location",
        ],
        irrelevant_rule_names=[
            "check_toppings_in_stock",
            "check_drinks_in_stock",
            "payment_process",
        ],
    )


# Taken from tests/core/unstable/engines/alpha/test_rule_matcher.py::test_that_rule_that_needs_to_be_reapplied_is_matched
async def test_that_rule_that_needs_to_be_reapplied_is_matched(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.AI_AGENT,
            "Hi! Welcome to PizzaBot, your virtual pizza assistant. How can I help you today?",
        ),
        (EventSource.CUSTOMER, "Hi, I’d like to order a pizza."),
        (EventSource.AI_AGENT, "Great choice! What size would you like—small, medium, or large?"),
        (EventSource.CUSTOMER, "I’ll take a large pizza."),
        (
            EventSource.AI_AGENT,
            "Got it! And what type of crust would you like? We have classic, thin, stuffed, or gluten-free.",
        ),
        (EventSource.CUSTOMER, "Let’s go with stuffed crust."),
        (
            EventSource.AI_AGENT,
            "Yum! Now for the toppings. Would you like a classic like pepperoni or something custom?",
        ),
        (EventSource.CUSTOMER, "Can I do half pepperoni and half veggie?"),
        (
            EventSource.AI_AGENT,
            "Absolutely! Your pizza will have half pepperoni and half veggie. For the veggie side, would you like a mix of mushrooms, onions, bell peppers, and olives?",
        ),
        (EventSource.CUSTOMER, "Yes, that sounds perfect."),
        (EventSource.AI_AGENT, "Great! Would you like any extra cheese or dipping sauces?"),
        (EventSource.CUSTOMER, "Extra cheese for sure, and ranch dipping sauce, please."),
        (
            EventSource.AI_AGENT,
            "Got it. A large, stuffed crust pizza with half pepperoni, half veggie, extra cheese, and ranch dipping sauce. Anything else?",
        ),
        (EventSource.CUSTOMER, "Nope, that’s all. How long will it take?"),
        (
            EventSource.AI_AGENT,
            "Your pizza will be ready in about 25 minutes. Would you like delivery or pickup?",
        ),
        (EventSource.CUSTOMER, "Delivery please?"),
        (
            EventSource.AI_AGENT,
            "Great, the total would be 10$, would you like to pay by credit or cash?",
        ),
        (EventSource.CUSTOMER, "Actually hold up, could you add another large pizza to the order?"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["large_pizza_crust"],
        relevant_rule_names=["large_pizza_crust"],
        irrelevant_rule_names=[],
    )


# Taken from tests/core/unstable/engines/alpha/test_rule_matcher.py::test_that_rules_based_on_context_variables_arent_matched_repetitively
async def test_that_rules_based_on_context_variables_arent_matched_repetitively(
    ranker: RuleRanker,
) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.AI_AGENT,
            "Hi! Welcome to PizzaBot, your virtual pizza assistant. We have a special summer deal - two large pizzas for the price of one! How can I help you today?",
        ),
        (EventSource.CUSTOMER, "Hi, I’d like to order a pizza."),
        (EventSource.AI_AGENT, "Great choice! What size would you like—small, medium, or large?"),
        (EventSource.CUSTOMER, "I’ll take a large pizza."),
        (
            EventSource.AI_AGENT,
            "Got it! And what type of crust would you like? We have classic, thin, stuffed, or gluten-free.",
        ),
        (EventSource.CUSTOMER, "Let’s go with stuffed crust."),
        (
            EventSource.AI_AGENT,
            "Yum! Now for the toppings. Would you like a classic like pepperoni or something custom?",
        ),
        (EventSource.CUSTOMER, "Can I do half pepperoni and half veggie?"),
        (
            EventSource.AI_AGENT,
            "Absolutely! Your pizza will have half pepperoni and half veggie. For the veggie side, would you like a mix of mushrooms, onions, bell peppers, and olives?",
        ),
        (EventSource.CUSTOMER, "Yes, that sounds perfect."),
        (EventSource.AI_AGENT, "Great! Would you like any extra cheese or dipping sauces?"),
        (EventSource.CUSTOMER, "Extra cheese for sure, and ranch dipping sauce, please."),
        (
            EventSource.AI_AGENT,
            "Got it. A large, stuffed crust pizza with half pepperoni, half veggie, extra cheese, and ranch dipping sauce. Anything else?",
        ),
        (EventSource.CUSTOMER, "Nope, that’s all. How long will it take?"),
        (
            EventSource.AI_AGENT,
            "Your pizza will be ready in about 25 minutes. Would you like delivery or pickup?",
        ),
        (EventSource.CUSTOMER, "Delivery please?"),
        (
            EventSource.AI_AGENT,
            "Great, the total would be 10$, would you like to pay by credit or cash?",
        ),
        (EventSource.CUSTOMER, "Actually hold up, could you add another large pizza to the order?"),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["summer_sale"],
        relevant_rule_names=[],
        irrelevant_rule_names=["summer_sale"],
        context_variables=[
            create_context_variable(
                name="season",
                data={"season": "Summer"},
            )
        ],
    )


# Taken from tests/core/unstable/engines/alpha/test_rule_matcher.py::test_that_observational_rules_arent_wrongly_implied
async def test_that_observational_rules_arent_wrongly_implied(ranker: RuleRanker) -> None:
    conversation: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I didn't get any help from the previous representative. If this continues I'll switch to the competitors. Don't thread on me!",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I apologize for what happened on your previous interaction with us - what is it that you're trying to do exactly?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking to modify an order I made through the online store",
        ),
    ]
    await base_test_that_rules_are_ranked_correctly(
        ranker,
        RULES_DICT,
        conversation=conversation,
        conversation_rule_names=["season_is_winter"],
        relevant_rule_names=[],
        irrelevant_rule_names=["season_is_winter"],
        context_variables=[
            create_context_variable(
                name="Date",
                data={"Year": "2025", "Month": "January", "Day": 24},
            ),
        ],
        staged_events=[
            create_staged_tool_event(
                cast(
                    JSONSerializable,
                    {
                        "tool_calls": [
                            {
                                "tool_id": "local:get_weather",
                                "arguments": {},
                                "result": {
                                    "data": "The weather is rainy",
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
