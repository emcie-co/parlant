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

from pytest_bdd import given, parsers

from parlant.core.agents import AgentId
from parlant.core.common import Criticality, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.entity_cq import EntityCommands
from parlant.core.evaluations import GuidelinePayload, PayloadOperation
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineStore

from parlant.core.services.indexing.behavioral_change_evaluation import GuidelineEvaluator
from parlant.core.sessions import AgentState, SessionId, SessionStore, SessionUpdateParams
from parlant.core.tags import Tag
from parlant.core.tools import ToolId
from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


def get_guideline_properties(
    context: ContextOfTest,
    condition: str,
    action: str | None,
) -> dict[str, JSONSerializable]:
    guideline_evaluator = context.container[GuidelineEvaluator]
    guideline_evaluation_data = context.sync_await(
        guideline_evaluator.evaluate(
            payloads=[
                GuidelinePayload(
                    content=GuidelineContent(
                        condition=condition,
                        action=action,
                    ),
                    tool_ids=[],
                    operation=PayloadOperation.ADD,
                    action_proposition=True,
                    properties_proposition=True,
                    journey_node_proposition=False,
                )
            ],
        )
    )
    metadata = guideline_evaluation_data[0].properties_proposition or {}
    return metadata


@step(given, parsers.parse("a guideline to {do_something} when {a_condition_holds}"))
def given_a_guideline_to_when(
    context: ContextOfTest,
    do_something: str,
    a_condition_holds: str,
) -> None:
    guideline_store = context.container[GuidelineStore]

    metadata = get_guideline_properties(context, a_condition_holds, do_something)

    context.sync_await(
        guideline_store.create_guideline(
            condition=a_condition_holds,
            action=do_something,
            metadata=metadata,
        )
    )


@step(
    given,
    parsers.parse(
        "a guideline to {do_something} when {a_condition_holds} with criticality {criticality}"
    ),
)
def given_a_guideline_to_when_with_criticality(
    context: ContextOfTest,
    do_something: str,
    a_condition_holds: str,
    criticality: str,
) -> None:
    guideline_store = context.container[GuidelineStore]

    metadata = get_guideline_properties(context, a_condition_holds, do_something)
    guideline_criticality = {
        "high": Criticality.HIGH,
        "medium": Criticality.MEDIUM,
        "low": Criticality.LOW,
    }[criticality]
    context.sync_await(
        guideline_store.create_guideline(
            condition=a_condition_holds,
            action=do_something,
            metadata=metadata,
            criticality=guideline_criticality,
        )
    )


@step(
    given, parsers.parse('an observational guideline "{guideline_name}" when {a_condition_holds}')
)
def given_an_observational_guideline_to(
    context: ContextOfTest,
    guideline_name: str,
    a_condition_holds: str,
) -> None:
    guideline_store = context.container[GuidelineStore]

    metadata = get_guideline_properties(context, a_condition_holds, None)

    guideline = context.sync_await(
        guideline_store.create_guideline(
            condition=a_condition_holds,
            action=None,
            metadata=metadata,
        )
    )

    context.guidelines[guideline_name] = guideline


@step(
    given,
    parsers.parse('a previously applied guideline "{guideline_name}"'),
)
def given_a_previously_applied_guideline(
    context: ContextOfTest,
    guideline_name: str,
    session_id: SessionId,
) -> None:
    session = context.sync_await(context.container[SessionStore].read_session(session_id))

    applied_guideline_ids = [context.guidelines[guideline_name].id]
    applied_guideline_ids.extend(
        session.agent_states[-1].applied_guideline_ids if session.agent_states else []
    )

    context.sync_await(
        context.container[EntityCommands].update_session(
            session_id=session_id,
            params=SessionUpdateParams(
                agent_states=list(session.agent_states)
                + [
                    AgentState(
                        trace_id="<main>",
                        applied_guideline_ids=applied_guideline_ids,
                        journey_paths={},
                    )
                ]
            ),
        )
    )


@step(
    given,
    parsers.parse('a guideline "{guideline_name}" to {do_something} when {a_condition_holds}'),
)
def given_a_guideline_name_to_when(
    context: ContextOfTest,
    guideline_name: str,
    do_something: str,
    a_condition_holds: str,
    agent_id: AgentId,
) -> None:
    guideline_store = context.container[GuidelineStore]

    metadata = get_guideline_properties(context, a_condition_holds, do_something)

    context.guidelines[guideline_name] = context.sync_await(
        guideline_store.create_guideline(
            condition=a_condition_holds,
            action=do_something,
            metadata=metadata,
        )
    )

    _ = context.sync_await(
        guideline_store.upsert_tag(
            context.guidelines[guideline_name].id,
            Tag.for_agent_id(agent_id),
        )
    )


@step(
    given,
    parsers.parse(
        'a disambiguation group head "{disambiguation_name}" to activate when {a_condition_holds}'
    ),
)
def given_an_observation_name_of(
    context: ContextOfTest,
    disambiguation_name: str,
    a_condition_holds: str,
) -> None:
    guideline_store = context.container[GuidelineStore]

    metadata = get_guideline_properties(context, a_condition_holds, None)

    context.guidelines[disambiguation_name] = context.sync_await(
        guideline_store.create_guideline(
            condition=a_condition_holds,
            action=None,
            metadata=metadata,
        )
    )


@step(given, "50 other random guidelines")
def given_50_other_random_guidelines(
    context: ContextOfTest,
    agent_id: AgentId,
) -> list[Guideline]:
    guideline_store = context.container[GuidelineStore]

    def create_guideline(condition: str, action: str) -> Guideline:
        metadata = get_guideline_properties(context, condition, action)

        guideline = context.sync_await(
            guideline_store.create_guideline(
                condition=condition,
                action=action,
                metadata=metadata,
            )
        )

        _ = context.sync_await(
            guideline_store.upsert_tag(
                guideline.id,
                Tag.for_agent_id(agent_id),
            )
        )

        return guideline

    guidelines: list[Guideline] = []

    for guideline_params in [
        {
            "condition": "The customer mentions being hungry",
            "action": "Suggest our pizza specials to the customer",
        },
        {
            "condition": "The customer asks about vegetarian options",
            "action": "list all vegetarian pizza options",
        },
        {
            "condition": "The customer inquires about delivery times",
            "action": "Provide the estimated delivery time based on their location",
        },
        {
            "condition": "The customer seems undecided",
            "action": "Recommend our top three most popular pizzas",
        },
        {
            "condition": "The customer asks for discount or promotions",
            "action": "Inform the customer about current deals or coupons",
        },
        {
            "condition": "The conversation starts",
            "action": "Greet the customer and ask if they'd like to order a pizza",
        },
        {
            "condition": "The customer mentions a food allergy",
            "action": "Ask for specific allergies and recommend safe menu options",
        },
        {
            "condition": "The customer requests a custom pizza",
            "action": "Guide the customer through choosing base, sauce, toppings, and cheese",
        },
        {
            "condition": "The customer wants to repeat a previous order",
            "action": "Retrieve the customer’s last order details and confirm if they want the same",
        },
        {
            "condition": "The customer asks about portion sizes",
            "action": "Describe the different pizza sizes and how many they typically serve",
        },
        {
            "condition": "The customer requests a drink",
            "action": "list available beverages and suggest popular pairings with "
            "their pizza choice",
        },
        {
            "condition": "The customer asks for the price",
            "action": "Provide the price of the selected items and any additional costs",
        },
        {
            "condition": "The customer expresses concern about calories",
            "action": "Offer information on calorie content and suggest lighter options if desired",
        },
        {
            "condition": "The customer mentions a special occasion",
            "action": "Suggest our party meal deals and ask if they would like to include desserts",
        },
        {
            "condition": "The customer wants to know the waiting area",
            "action": "Inform about the waiting facilities at our location or "
            "suggest comfortable seating arrangements",
        },
        {
            "condition": "The customer is comparing pizza options",
            "action": "Highlight the unique features of different pizzas we offer",
        },
        {
            "condition": "The customer asks for recommendations",
            "action": "Suggest pizzas based on their previous orders or popular trends",
        },
        {
            "condition": "The customer is interested in combo deals",
            "action": "Explain the different combo offers and their benefits",
        },
        {
            "condition": "The customer asks if ingredients are fresh",
            "action": "Assure them of the freshness and quality of our ingredients",
        },
        {
            "condition": "The customer wants to modify an order",
            "action": "Assist in making the desired changes and confirm the new order details",
        },
        {
            "condition": "The customer has connectivity issues during ordering",
            "action": "Suggest completing the order via a different method (phone, app)",
        },
        {
            "condition": "The customer expresses dissatisfaction with a previous order",
            "action": "Apologize and offer a resolution (discount, replacement)",
        },
        {
            "condition": "The customer inquires about loyalty programs",
            "action": "Describe our loyalty program benefits and enrollment process",
        },
        {
            "condition": "The customer is about to end the conversation without ordering",
            "action": "Offer a quick summary of unique selling points or a one-time "
            "discount to encourage purchase",
        },
        {
            "condition": "The customer asks for gluten-free options",
            "action": "list our gluten-free pizza bases and toppings",
        },
        {
            "condition": "The customer is looking for side orders",
            "action": "Recommend complementary side dishes like garlic bread or salads",
        },
        {
            "condition": "The customer mentions children",
            "action": "Suggest our kids' menu or family-friendly options",
        },
        {
            "condition": "The customer is having trouble with the online payment",
            "action": "Offer assistance with the payment process or propose an "
            "alternative payment method",
        },
        {
            "condition": "The customer wants to know the origin of ingredients",
            "action": "Provide information about the source and quality assurance "
            "of our ingredients",
        },
        {
            "condition": "The customer asks for a faster delivery option",
            "action": "Explain express delivery options and any associated costs",
        },
        {
            "condition": "The customer seems interested in healthy eating",
            "action": "Highlight our health-conscious options like salads or "
            "pizzas with whole wheat bases",
        },
        {
            "condition": "The customer wants a contactless delivery",
            "action": "Confirm the address and explain the process for contactless delivery",
        },
        {
            "condition": "The customer is a returning customer",
            "action": "Welcome them back and ask if they would like to order their "
            "usual or try something new",
        },
        {
            "condition": "The customer inquires about our environmental impact",
            "action": "Share information about our sustainability practices and "
            "eco-friendly packaging",
        },
        {
            "condition": "The customer is planning a large event",
            "action": "Offer catering services and discuss bulk order discounts",
        },
        {
            "condition": "The customer seems in a rush",
            "action": "Suggest our quickest delivery option and process the order promptly",
        },
        {
            "condition": "The customer wants to pick up the order",
            "action": "Provide the pickup location and expected time until the order is ready",
        },
        {
            "condition": "The customer expresses interest in a specific topping",
            "action": "Offer additional information about that topping and suggest "
            "other complementary toppings",
        },
        {
            "condition": "The customer is making a business order",
            "action": "Propose our corporate deals and ask about potential regular "
            "orders for business meetings",
        },
        {
            "condition": "The customer asks for cooking instructions",
            "action": "Provide details on how our pizzas are made or instructions "
            "for reheating if applicable",
        },
        {
            "condition": "The customer inquires about the chefs",
            "action": "Share background information on our chefs’ expertise and experience",
        },
        {
            "condition": "The customer asks about non-dairy options",
            "action": "list our vegan cheese alternatives and other non-dairy products",
        },
        {
            "condition": "The customer expresses excitement about a new menu item",
            "action": "Provide more details about the item and suggest adding it to their order",
        },
        {
            "condition": "The customer wants a quiet place to eat",
            "action": "Describe the ambiance of our quieter dining areas or "
            "recommend off-peak times",
        },
        {
            "condition": "The customer asks about our app",
            "action": "Explain the features of our app and benefits of ordering through it",
        },
        {
            "condition": "The customer has difficulty deciding",
            "action": "Offer to make a selection based on their preferences or "
            "our chef’s recommendations",
        },
        {
            "condition": "The customer mentions they are in a specific location",
            "action": "Check if we deliver to that location and inform them about "
            "the nearest outlet",
        },
        {
            "condition": "The customer is concerned about food safety",
            "action": "Reassure them about our health and safety certifications and practices",
        },
        {
            "condition": "The customer is looking for a quiet place to eat",
            "action": "Describe the ambiance of our quieter dining areas or "
            "recommend off-peak times",
        },
        {
            "condition": "The customer shows interest in repeat orders",
            "action": "Introduce features like scheduled deliveries or subscription "
            "services to simplify their future orders",
        },
    ]:
        guidelines.append(create_guideline(**guideline_params))

    return guidelines


@step(given, parsers.parse('the guideline called "{guideline_id}"'))
def given_the_guideline_called(
    context: ContextOfTest,
    agent_id: AgentId,
    guideline_id: str,
) -> Guideline:
    guideline_store = context.container[GuidelineStore]

    def create_guideline(condition: str, action: str) -> Guideline:
        metadata = get_guideline_properties(context, condition, action)

        guideline = context.sync_await(
            guideline_store.create_guideline(
                condition=condition,
                action=action,
                metadata=metadata,
            )
        )

        _ = context.sync_await(
            guideline_store.upsert_tag(
                guideline.id,
                Tag.for_agent_id(agent_id),
            )
        )

        return guideline

    guidelines = {
        "check_drinks_in_stock": {
            "condition": "a client asks for a drink",
            "action": "check if the drink is available in stock",
        },
        "check_toppings_in_stock": {
            "condition": "a client asks about toppings or order pizza with toppings",
            "action": "check what toppings are available in stock",
        },
        "ask_expert_about_Spot": {
            "condition": "a client asks for information about Spot",
            "action": "ask and get the answer from the expert",
        },
        "check_toppings_or_drinks_in_stock": {
            "condition": "a client asks for toppings or drinks",
            "action": "check if they are available in stock",
        },
        "calculate_sum": {
            "condition": "an equation involves adding numbers",
            "action": "calculate the sum",
        },
        "check_drinks_or_toppings_in_stock": {
            "condition": "a client asks for a drink or toppings",
            "action": "check what drinks or toppings are available in stock",
        },
        "calculate_addition_or_multiplication": {
            "condition": "an equation contains addition or multiplication",
            "action": "calculate it",
        },
        "retrieve_account_information": {
            "condition": "asked for information about an account",
            "action": "answer by retrieving the information from the database",
        },
        "calculate_addition": {
            "condition": "an equation contains an add function",
            "action": "get the result from the add tool",
        },
        "calculate_multiplication": {
            "condition": "an equation contains a multiply function",
            "action": "get the result from the multiply tool",
        },
        "transfer_money_between_accounts": {
            "condition": "asked to transfer money from one account to another",
            "action": "check if the account has enough balance to make the transfer"
            "and then proceed with the transfer",
        },
        "retrieve_Spot_information": {
            "condition": "asked for information about Spot",
            "action": "answer by retrieving the information from the database",
        },
        "retrieve_account_balance": {
            "condition": "asked for information about an account",
            "action": "answer by retrieving the information from the database",
        },
    }

    guideline = create_guideline(**guidelines[guideline_id])

    context.guidelines[guideline_id] = guideline

    return guideline


@step(
    given,
    parsers.parse('that the "{guideline_name}" guideline was matched in the previous iteration'),
)
def given_was_matched_in_previous_iteration(
    context: ContextOfTest,
    guideline_name: str,
) -> None:
    guideline = context.guidelines[guideline_name]

    context.guideline_matches[guideline_name] = GuidelineMatch(
        guideline=guideline,
        score=10,
        rationale="",
    )


@step(
    given,
    parsers.parse(
        'that the "{guideline_name}" guideline is matched with a priority of {score} because {rationale}'  # noqb
    ),
)
def given_a_guideline_match(
    context: ContextOfTest,
    guideline_name: str,
    score: int,
    rationale: str,
) -> None:
    guideline = context.guidelines[guideline_name]

    context.guideline_matches[guideline_name] = GuidelineMatch(
        guideline=guideline,
        score=score,
        rationale=rationale,
    )


@step(
    given,
    parsers.parse('a guideline relationship whereby "{guideline_a}" entails "{guideline_b}"'),
)
def given_an_entailment_guideline_relationship(
    context: ContextOfTest,
    guideline_a: str,
    guideline_b: str,
) -> None:
    store = context.container[RelationshipStore]

    context.sync_await(
        store.create_relationship(
            source=RelationshipEntity(
                id=context.guidelines[guideline_a].id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=context.guidelines[guideline_b].id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            kind=RelationshipKind.ENTAILMENT,
        )
    )


@step(
    given,
    parsers.parse('a guideline "{guideline}" is grouped under "{disambiguation_head}"'),
)
def given_an_guideline_grouped_under(
    context: ContextOfTest,
    guideline: str,
    disambiguation_head: str,
) -> None:
    store = context.container[RelationshipStore]

    context.sync_await(
        store.create_relationship(
            source=RelationshipEntity(
                id=context.guidelines[disambiguation_head].id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=context.guidelines[guideline].id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            kind=RelationshipKind.DISAMBIGUATION,
        )
    )


@step(
    given,
    parsers.parse(
        'a dependency relationship between the guideline "{guideline_name}" and the "{journey_title}" journey'
    ),
)
def given_an_dependency_between_guideline_and_a_journey(
    context: ContextOfTest,
    guideline_name: str,
    journey_title: str,
) -> None:
    store = context.container[RelationshipStore]
    journey = context.journeys[journey_title]

    context.sync_await(
        store.create_relationship(
            source=RelationshipEntity(
                id=context.guidelines[guideline_name].id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=Tag.for_journey_id(journey.id),
                kind=RelationshipEntityKind.TAG,
            ),
            kind=RelationshipKind.DEPENDENCY,
        )
    )


@step(
    given,
    parsers.parse(
        'a reevaluation relationship between the guideline "{guideline_name}" and the "{tool_name}" tool'
    ),
)
def given_an_reevaluation_between_guideline_and_a_tool(
    context: ContextOfTest,
    guideline_name: str,
    tool_name: str,
) -> None:
    store = context.container[RelationshipStore]

    context.sync_await(
        store.create_relationship(
            source=RelationshipEntity(
                id=context.guidelines[guideline_name].id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=ToolId(service_name="local", tool_name=tool_name),
                kind=RelationshipEntityKind.TOOL,
            ),
            kind=RelationshipKind.DEPENDENCY,
        )
    )
