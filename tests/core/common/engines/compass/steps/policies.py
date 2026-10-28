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

"""Policy steps for the compass BDD suites.

The alpha suites' condition/action guidelines are ported as compass policies:
one description-based rule whose content reads "When {condition}, then
{action}". Observations stay condition-based, mirroring the SDK's
create_observation. Production metadata (the cross-reference flag and recall
signals) is proposed through the real indexing processor — except for bulk
filler policies, where content-embedding recall suffices and per-policy LLM
propositions would only add cost.
"""

from pytest_bdd import given, parsers

from parlant.core.agents import AgentId
from parlant.core.common import Weight, JSONSerializable
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core import metadata as metadata_utils
from parlant.core.indexing import PayloadOperation, RulePayload
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.rules import Rule, RuleStore

from parlant.core.services.indexing.indexing_service import RuleIndexingProcessor
from parlant.core.sessions import SessionId, SessionStore, SessionUpdateParams
from parlant.core.groups import GroupIds
from parlant.core.tools import ToolId
from tests.core.common.engines.compass.utils import step
from tests.core.common.utils import ContextOfTest

_DISCOVERED_RULE_IDS_PATH = ("compass", "discovered_rule_ids")
_RULES_EVER_MATCHED_IN_SESSION_PATH = ("compass", "rules_ever_matched_in_session")


def create_policy_for_test(
    context: ContextOfTest,
    condition: str,
    action: str,
    *,
    agent_id: AgentId | None = None,
    weight: Weight | None = None,
    propose_metadata: bool = True,
) -> Rule:
    """Create a policy rule the way production would: content-based, with the
    indexing propositions (cross-reference flag + recall signals) applied."""
    content = f"When {condition}, then {action}"

    metadata: dict[str, JSONSerializable] = {}
    signals: list[str] = []
    anti_signals: list[str] = []

    if propose_metadata:
        invoice = context.sync_await(
            context.container[RuleIndexingProcessor].process(
                payloads=[
                    RulePayload(
                        content=content,
                        operation=PayloadOperation.ADD,
                        relativity_proposition=True,
                        signal_proposition=True,
                        agent_id=agent_id,
                    )
                ],
            )
        )[0]

        if invoice.references_other_policies_proposition is not None:
            metadata = metadata_utils.set_internal_value(
                metadata,
                ("references_other_policies",),
                invoice.references_other_policies_proposition,
            )
        signals = list(invoice.signals_proposition or [])
        anti_signals = list(invoice.anti_signals_proposition or [])

    rule = context.sync_await(
        context.container[RuleStore].create_rule(
            content=content,
            weight=weight,
            metadata=metadata,
            signals=signals,
            anti_signals=anti_signals,
        )
    )

    if agent_id is not None:
        context.sync_await(
            context.container[RuleStore].upsert_group(rule.id, GroupIds.for_agent_id(agent_id))
        )
        rule = context.sync_await(context.container[RuleStore].read_rule(rule.id))

    return rule


def create_observation_for_test(
    context: ContextOfTest,
    condition: str,
    *,
    agent_id: AgentId | None = None,
) -> Rule:
    invoice = context.sync_await(
        context.container[RuleIndexingProcessor].process(
            payloads=[
                RulePayload(
                    condition=condition,
                    operation=PayloadOperation.ADD,
                    relativity_proposition=True,
                    signal_proposition=True,
                    agent_id=agent_id,
                )
            ],
        )
    )[0]

    metadata: dict[str, JSONSerializable] = {}
    if invoice.references_other_policies_proposition is not None:
        metadata = metadata_utils.set_internal_value(
            metadata,
            ("references_other_policies",),
            invoice.references_other_policies_proposition,
        )

    rule = context.sync_await(
        context.container[RuleStore].create_rule(
            condition=condition,
            metadata=metadata,
            signals=list(invoice.signals_proposition or []),
            anti_signals=list(invoice.anti_signals_proposition or []),
        )
    )

    if agent_id is not None:
        context.sync_await(
            context.container[RuleStore].upsert_group(rule.id, GroupIds.for_agent_id(agent_id))
        )
        rule = context.sync_await(context.container[RuleStore].read_rule(rule.id))

    return rule


@step(given, parsers.parse("a policy to {do_something} when {a_condition_holds}"))
def given_a_policy_to_when(
    context: ContextOfTest,
    do_something: str,
    a_condition_holds: str,
) -> None:
    create_policy_for_test(context, a_condition_holds, do_something)


@step(
    given,
    parsers.parse("a policy to {do_something} when {a_condition_holds} with weight {weight}"),
)
def given_a_policy_to_when_with_weight(
    context: ContextOfTest,
    do_something: str,
    a_condition_holds: str,
    weight: str,
) -> None:
    create_policy_for_test(
        context,
        a_condition_holds,
        do_something,
        weight={
            "high": Weight.HIGH,
            "medium": Weight.MEDIUM,
            "low": Weight.LOW,
        }[weight],
    )


@step(given, parsers.parse('an observation "{policy_name}" when {a_condition_holds}'))
def given_an_observation_when(
    context: ContextOfTest,
    policy_name: str,
    a_condition_holds: str,
) -> None:
    context.rules[policy_name] = create_observation_for_test(context, a_condition_holds)


@step(
    given,
    parsers.parse('a previously applied policy "{policy_name}"'),
)
def given_a_previously_applied_policy(
    context: ContextOfTest,
    policy_name: str,
    session_id: SessionId,
) -> None:
    """Mark the policy as already applied in this session, the compass way:
    it enters the discovered working set AND the ever-matched ledger (which is
    what satisfies dependency relationships and suppresses re-tracking)."""
    session = context.sync_await(context.container[SessionStore].read_session(session_id))
    rule_id = str(context.rules[policy_name].id)

    metadata = dict(session.metadata)
    for path in (_DISCOVERED_RULE_IDS_PATH, _RULES_EVER_MATCHED_IN_SESSION_PATH):
        existing = list(
            metadata_utils.internal_value(metadata, path, []) or []  # type: ignore[arg-type]
        )
        if rule_id not in existing:
            metadata = metadata_utils.set_internal_value(metadata, path, [*existing, rule_id])

    context.sync_await(
        context.container[SessionStore].update_session(
            session_id,
            SessionUpdateParams(metadata=metadata),
        )
    )


@step(
    given,
    parsers.parse('a policy "{policy_name}" to {do_something} when {a_condition_holds}'),
)
def given_a_policy_name_to_when(
    context: ContextOfTest,
    policy_name: str,
    do_something: str,
    a_condition_holds: str,
    agent_id: AgentId,
) -> None:
    context.rules[policy_name] = create_policy_for_test(
        context, a_condition_holds, do_something, agent_id=agent_id
    )


@step(
    given,
    parsers.parse(
        'a policy "{policy_name}" to {do_something} when {a_condition_holds} with weight {weight}'
    ),
)
def given_a_policy_name_to_when_with_weight(
    context: ContextOfTest,
    policy_name: str,
    do_something: str,
    a_condition_holds: str,
    weight: str,
    agent_id: AgentId,
) -> None:
    context.rules[policy_name] = create_policy_for_test(
        context,
        a_condition_holds,
        do_something,
        agent_id=agent_id,
        weight={
            "high": Weight.HIGH,
            "medium": Weight.MEDIUM,
            "low": Weight.LOW,
        }[weight],
    )


@step(given, "50 other random policies")
def given_50_other_random_policies(
    context: ContextOfTest,
    agent_id: AgentId,
) -> list[Rule]:
    # Bulk filler: matters for crowding/scale, not for individual verdicts, so
    # skip the per-policy LLM propositions — content-embedding recall suffices.
    policies: list[Rule] = []

    for params in [
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
        policies.append(
            create_policy_for_test(
                context,
                params["condition"],
                params["action"],
                agent_id=agent_id,
                propose_metadata=False,
            )
        )

    return policies


@step(given, parsers.parse('the policy called "{policy_id}"'))
def given_the_policy_called(
    context: ContextOfTest,
    agent_id: AgentId,
    policy_id: str,
) -> Rule:
    policies = {
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

    rule = create_policy_for_test(
        context,
        policies[policy_id]["condition"],
        policies[policy_id]["action"],
        agent_id=agent_id,
    )

    context.rules[policy_id] = rule

    return rule


@step(
    given,
    parsers.parse('that the "{policy_name}" policy was matched in the previous iteration'),
)
def given_was_matched_in_previous_iteration(
    context: ContextOfTest,
    policy_name: str,
) -> None:
    context.rule_matches[policy_name] = RuleEvaluation(
        rule=context.rules[policy_name],
        reasoning="",
    )


@step(
    given,
    parsers.parse('that the "{policy_name}" policy is matched because {rationale}'),
)
def given_a_policy_match(
    context: ContextOfTest,
    policy_name: str,
    rationale: str,
) -> None:
    context.rule_matches[policy_name] = RuleEvaluation(
        rule=context.rules[policy_name],
        reasoning=rationale,
    )


@step(
    given,
    parsers.parse('a policy relationship whereby "{policy_a}" entails "{policy_b}"'),
)
def given_an_entailment_policy_relationship(
    context: ContextOfTest,
    policy_a: str,
    policy_b: str,
) -> None:
    context.sync_await(
        context.container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=context.rules[policy_a].id,
                kind=RelationshipEntityKind.RULE,
            ),
            target=RelationshipEntity(
                id=context.rules[policy_b].id,
                kind=RelationshipEntityKind.RULE,
            ),
            kind=RelationshipKind.ENTAILMENT,
        )
    )


@step(
    given,
    parsers.parse(
        'a reevaluation relationship between the policy "{policy_name}" and the "{tool_name}" tool'
    ),
)
def given_a_reevaluation_between_policy_and_a_tool(
    context: ContextOfTest,
    policy_name: str,
    tool_name: str,
) -> None:
    context.sync_await(
        context.container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=context.rules[policy_name].id,
                kind=RelationshipEntityKind.RULE,
            ),
            target=RelationshipEntity(
                id=ToolId(service_name="local", tool_name=tool_name),
                kind=RelationshipEntityKind.TOOL,
            ),
            kind=RelationshipKind.REEVALUATION,
        )
    )
