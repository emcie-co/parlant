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
from typing import Any, Mapping, Optional, cast

import pytest

from parlant.core.container import Container
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.loggers import Logger
from parlant.core.services.indexing.common import IndexingError

from parlant.core.services.indexing.rule_relativity_proposer import RuleRelativityProposer


class _FakeModelRouter:
    def __init__(self, generator: Any) -> None:
        self._generator = generator

    async def get_schematic_generator(self, **kwargs: Any) -> Any:
        return self._generator

    async def resolve_schematic_reasoning_effort(self, **kwargs: Any) -> str | None:
        return None


class _RateLimitedGenerator:
    async def generate(self, prompt: Any, hints: Mapping[str, Any] = {}) -> Any:
        raise RuntimeError("429 RESOURCE_EXHAUSTED: free-tier requests per minute")


async def test_that_a_failed_relativity_proposal_reports_the_providers_error(
    container: Container,
) -> None:
    # Indexing runs before the server is ready, so this error is often the
    # only thing a developer sees. "Indexing failed" alone sends them looking
    # in the wrong place; the provider's own error says what to fix.
    proposer = RuleRelativityProposer(
        logger=container[Logger],
        optimization_policy=container[CompassOptimizationPolicy],
        model_router=cast(Any, _FakeModelRouter(_RateLimitedGenerator())),
    )

    with pytest.raises(IndexingError) as exc_info:
        await proposer.propose_relativity(content="Offer store credit instead", condition=None)

    assert "429 RESOURCE_EXHAUSTED" in str(exc_info.value)


async def test_that_rules_referencing_other_policies_are_classified_as_cross_referential(
    container: Container,
) -> None:
    proposer = container[RuleRelativityProposer]

    rules: list[tuple[Optional[str], Optional[str]]] = [
        # Implicit complement: "not supported" negates a coverage set defined
        # entirely by the agent's other policies.
        (
            "Tell them that you cannot help them with this matter",
            "The customer asks about a capability that is not supported",
        ),
        # Quantified meta-reference: quantifies over the whole policy set.
        (
            "Politely say this is beyond what you can help with",
            "The customer's request doesn't fall under any of our policies",
        ),
        # Definite reference: names a specific sibling policy.
        (
            "Offer store credit instead",
            "The customer wants to return an item and the return policy does not apply",
        ),
        # Comparative deference to an unnamed but specific other policy.
        (
            "Offer the standard refund flow unless a more specific return policy applies to the item",
            "The customer requests a refund",
        ),
        # Conflict observation: quantifies over the agent's active policies.
        (
            "Follow the more restrictive policy and note the conflict",
            "Two or more of the agent's active policies give conflicting guidance for the customer's request",
        ),
    ]

    results = await asyncio.gather(
        *(
            proposer.propose_relativity(content=description, condition=condition)
            for description, condition in rules
        )
    )

    for (description, condition), result in zip(rules, results):
        assert result.references_other_policies, (
            f"Rule failed to be classified as referencing other policies:\n"
            f"Condition: {condition}\n"
            f"Description: {description}"
        )


async def test_that_self_contained_rules_are_not_classified_as_cross_referential(
    container: Container,
) -> None:
    proposer = container[RuleRelativityProposer]

    rules: list[tuple[Optional[str], Optional[str]]] = [
        (
            "Collect the order number and start the refund flow",
            "The customer wants a refund for a delivered order",
        ),
        (
            "Use language that is simple and not overly technical",
            "The customer is above 60",
        ),
        (
            "List all vegetarian pizza options",
            "The customer asks about vegetarian options",
        ),
        (
            "Apologize for the inconvenience and confirm the report receipt",
            "A user reports an error during account setup",
        ),
    ]

    results = await asyncio.gather(
        *(
            proposer.propose_relativity(content=description, condition=condition)
            for description, condition in rules
        )
    )

    for (description, condition), result in zip(rules, results):
        assert not result.references_other_policies, (
            f"Rule wrongly classified as referencing other policies:\n"
            f"Condition: {condition}\n"
            f"Description: {description}"
        )


# --- Real-world corpus: the tau-bench retail and airline agents -------------------
#
# Faithful shapes: SDK policies store their whole text in `description`
# (condition=None) with an optional title; observations store a
# bare condition. The proposer must classify these as reliably as the
# hand-written archetypes above. The negatives here are
# deliberately hard: in-rule complements ("but nothing else", "any other
# reason than the ones listed above"), self-referential "the rules" mentions,
# and insurance-style "covered" vocabulary — none of which reference OTHER
# policies.

_RuleParts = tuple[Optional[str], Optional[str], Optional[str]]  # (title, description, condition)


def _policy(description: str, title: str | None = None) -> _RuleParts:
    return title, description, None


def _observation(condition: str) -> _RuleParts:
    return None, None, condition


async def _classify(
    proposer: RuleRelativityProposer,
    rules: list[_RuleParts],
) -> list[bool]:
    results = await asyncio.gather(
        *(
            proposer.propose_relativity(content=description, condition=condition, title=title)
            for title, description, condition in rules
        )
    )
    return [result.references_other_policies for result in results]


_TAUBENCH_RETAIL_CROSS_REFERENTIAL = [
    # "cannot be handled within the scope of your actions" is the complement
    # of the coverage set defined by the agent's other policies.
    _policy(
        """\
You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.
To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
        title="Human Handoff",
    ),
]

_TAUBENCH_RETAIL_SELF_CONTAINED = [
    _policy(
        """\
Exchange or modify order tools can only be called once per order.
Be sure that all items to be changed are collected into a list before making the tool call!!!
""",
        title="Generic action rules",
    ),
    _policy(
        """\
An order can only be cancelled if its status is 'pending', and you should check its status before taking the action.

The user needs to confirm the order id and the reason (either 'no longer needed' or 'ordered by mistake') for cancellation. Other reasons are not acceptable.

After user confirmation, the order status will be changed to 'cancelled', and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.
""",
        title="Cancel pending order",
    ),
    # "but nothing else" negates against the rule's OWN enumeration, not the
    # policy set — a hard negative for the implicit-complement flavor.
    _policy(
        """\
An order can only be modified if its status is 'pending', and you should check its status before taking the action.

For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.
""",
        title="Modify pending order",
    ),
    _policy(
        """\
The user can only choose a single payment method different from the original payment method.

If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.

After user confirmation, the order status will be kept as 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise it will be refunded within 5 to 7 business days.
""",
        title="Modify payment",
    ),
    _policy(
        """\
This action can only be called once, and will change the order status to 'pending (items modifed)'. The agent will not be able to modify or cancel the order anymore. So you must confirm all the details are correct and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all the items they want to modify.

For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.
""",
        title="Modify items",
    ),
    _policy(
        """\
An order can only be returned if its status is 'delivered', and you should check its status before taking the action.

The user needs to confirm the order id and the list of items to be returned.

The user needs to provide a payment method to receive the refund.

The refund must either go to the original payment method, or an existing gift card.

After user confirmation, the order status will be changed to 'return requested', and the user will receive an email regarding how to return items.
""",
        title="Return delivered order",
    ),
    _policy(
        """\
An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.

For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

After user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order.
""",
        title="Exchange delivered order",
    ),
    _observation("you need to calculate anything"),
    _observation("you need to find the user's ID"),
    _observation("you need to change the user's address"),
    _observation(
        "you need to get the user's details (like addresses, payment methods, and orders)"
    ),
    _observation("you need to get an order's details"),
    _observation("you need to get a product's details and/or list its variants"),
    _observation("you need to get an item's details"),
    _observation("you need to find a product or list all available product types"),
    _observation("you need to cancel a pending order"),
    _observation("you need to modify a pending order's address"),
    _observation("you need to modify a pending order's payment method"),
    _observation("you need to modify a pending order's items"),
    _observation("you need to exchange a delivered order's items"),
    _observation("you need to return a delivered order's items"),
]

_TAUBENCH_AIRLINE_CROSS_REFERENTIAL = [
    # "out of your scope" / "cannot be handled within the scope of your
    # actions": implicit complement of the agent's coverage.
    _policy(
        """\
When the user's request is out of your scope *and the user explicitly asks to be transferred to a human agent*,
you should offer to transfer the user to a human agent if and only if the request cannot be handled within
the scope of your actions. If the user asks to be transferred, first make a tool call to transfer_to_human_agents,
and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.

Never suggest or initiate human handoff yourself. It should always come from a user request.
""",
        title="Human Handoff",
    ),
    # Definite reference to a sibling policy by name ("the refunds and
    # compensation policy").
    _policy(
        """\
When you may need to send a certificate to the user, carefully review the refunds and compensation
policy and send the certificate only if and to the extent that the user is eligible
""",
    ),
]

_TAUBENCH_AIRLINE_SELF_CONTAINED = [
    _policy(
        """\
You must first obtain the user id from the user.

You should then ask for the trip type, origin, destination.

Cabin:
- Cabin class must be the same across all the flights in a reservation.

Passengers:
- Each reservation can have at most five passengers.
- You need to collect the first name, last name, and date of birth for each passenger.
- All passengers must fly the same flights in the same cabin.

Payment:
- Each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards.
- The remaining amount of a travel certificate is not refundable.
- All payment methods must already be in user profile for safety reasons.

Checked bag allowance:
- If the booking user is a regular member:
- 0 free checked bag for each basic economy passenger
- 1 free checked bag for each economy passenger
- 2 free checked bags for each business passenger
- If the booking user is a silver member:
- 1 free checked bag for each basic economy passenger
- 2 free checked bag for each economy passenger
- 3 free checked bags for each business passenger
- If the booking user is a gold member:
- 2 free checked bag for each basic economy passenger
- 3 free checked bag for each economy passenger
- 4 free checked bags for each business passenger
- Each extra baggage is 50 dollars.

Do not add checked bags that the user does not need.

Travel insurance:
- You should ask if the user wants to buy the travel insurance.
- The travel insurance is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.
""",
        title="Book flight",
    ),
    # "you must make sure the rules apply" refers to the rules stated WITHIN
    # this policy — a hard self-reference negative.
    _policy(
        """\
First, you must obtain the user id and reservation id.
- The user must provide their user id.
- If the user doesn't know their reservation id, you should help locate it using available tools.

Change flights:
- Basic economy flights cannot be modified.
- Other reservations can be modified without changing the origin, destination, and trip type.
- Some flight segments can be kept, but their prices will not be updated based on the current price.
- The API does not check these for you, so you must make sure the rules apply before calling the API!

Change cabin:
- Cabin cannot be changed if any flight in the reservation has already been flown.
- In other cases, all reservations, including basic economy, can change cabin without changing the flights.
- Cabin class must remain the same across all the flights in the same reservation; changing cabin for just one flight segment is not possible.
- If the price after cabin change is higher than the original price, the user is required to pay for the difference.
- If the price after cabin change is lower than the original price, the user is should be refunded the difference.

Change baggage and insurance:
- The user can add but not remove checked bags.
- The user cannot add insurance after initial booking.

Change passengers:
- The user can modify passengers but cannot modify the number of passengers.
- Even a human agent cannot modify the number of passengers.

Payment:
- If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method. The payment method must already be in user profile for safety reasons.
""",
        title="Modify flight",
    ),
    # "covered by insurance" is product coverage (business data), not the
    # agent's policy set — the insurance-vocabulary hard negative.
    _policy(
        """\
First, you must obtain the user id and reservation id.
- The user must provide their user id.
- If the user doesn't know their reservation id, you should help locate it using available tools.

You must also obtain the reason for cancellation (change of plan, airline cancelled flight, or other reasons)

If any portion of the flight has already been flown, you cannot help and transfer is needed.

Otherwise, flight can be cancelled if any of the following is true:
- The booking was made within the last 24 hrs
- The flight is cancelled by airline
- It is a business flight
- The user has travel insurance and the reason for cancellation is covered by insurance.

The API does not check that cancellation rules are met, so you must make sure the rules apply before calling the API!

Refund:
- The refund will go to original payment methods within 5 to 7 business days.
""",
        title="Cancel flight",
    ),
    # "any other reason than the ones listed above" complements the rule's
    # OWN list — another in-rule complement negative.
    _policy(
        """\
Do not proactively offer a compensation unless the user explicitly asks for one.

Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy.

Always confirms the facts before offering compensation.

Only compensate if the user is a silver/gold member or has travel insurance or flies business.

- If the user complains about cancelled flights in a reservation, the agent can offer a certificate as a gesture after confirming the facts, with the amount being $100 times the number of passengers.

- If the user complains about delayed flights in a reservation and wants to change or cancel the reservation, the agent can offer a certificate as a gesture after confirming the facts and changing or cancelling the reservation, with the amount being $50 times the number of passengers.

Do not offer compensation for any other reason than the ones listed above.
""",
        title="Refunds and compensation",
    ),
    _policy(
        """\
Reason carefully every time before using the calculate tool, on how to use it properly.
Make sure to keep in mind the *needs and goals* for which you're calculating, not just interim calculations.
Some calculations have to do with multiple different entities or values.
Feel free to structure your calculation over multiple calls to the calculate tool, to be as sure as safe as possible that you get it right.
When appropriate, make sure to communicate and break down your calculations to the user, connecting the results to their needs and goals where applicable.
""",
        title="Safety and Reliability of Calculations",
    ),
    _observation("you need to update the reservation's flights"),
    _observation("you need to update the reservation's passengers"),
    _observation("you need to book a reservation"),
    _observation("you need to cancel a reservation"),
    _observation("you need to update the baggage details"),
    _observation("you need to get the user's details (like reservations and payment methods)"),
    _observation("you need to get a reservation's details"),
    _observation("you need to find an airport or list all available airports"),
    _observation("you need to search for flights"),
    _observation("you need to get the status of a flight"),
]


async def test_that_taubench_retail_cross_referential_policies_are_classified(
    container: Container,
) -> None:
    proposer = container[RuleRelativityProposer]

    verdicts = await _classify(proposer, _TAUBENCH_RETAIL_CROSS_REFERENTIAL)

    for (title, description, _condition), verdict in zip(
        _TAUBENCH_RETAIL_CROSS_REFERENTIAL, verdicts
    ):
        assert verdict, (
            f"Rule failed to be classified as referencing other policies:\n"
            f"Title: {title}\nDescription: {description}"
        )


async def test_that_taubench_retail_self_contained_rules_are_not_classified(
    container: Container,
) -> None:
    proposer = container[RuleRelativityProposer]

    verdicts = await _classify(proposer, _TAUBENCH_RETAIL_SELF_CONTAINED)

    for (title, description, condition), verdict in zip(_TAUBENCH_RETAIL_SELF_CONTAINED, verdicts):
        assert not verdict, (
            f"Rule wrongly classified as referencing other policies:\n"
            f"Title: {title}\nCondition: {condition}\n"
            f"Description: {description}"
        )


async def test_that_taubench_airline_cross_referential_policies_are_classified(
    container: Container,
) -> None:
    proposer = container[RuleRelativityProposer]

    verdicts = await _classify(proposer, _TAUBENCH_AIRLINE_CROSS_REFERENTIAL)

    for (title, description, _condition), verdict in zip(
        _TAUBENCH_AIRLINE_CROSS_REFERENTIAL, verdicts
    ):
        assert verdict, (
            f"Rule failed to be classified as referencing other policies:\n"
            f"Title: {title}\nDescription: {description}"
        )


async def test_that_taubench_airline_self_contained_rules_are_not_classified(
    container: Container,
) -> None:
    proposer = container[RuleRelativityProposer]

    verdicts = await _classify(proposer, _TAUBENCH_AIRLINE_SELF_CONTAINED)

    for (title, description, condition), verdict in zip(_TAUBENCH_AIRLINE_SELF_CONTAINED, verdicts):
        assert not verdict, (
            f"Rule wrongly classified as referencing other policies:\n"
            f"Title: {title}\nCondition: {condition}\n"
            f"Description: {description}"
        )
