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

# ============================================================================
# τ-bench (tau2-bench) retail distiller tests.
#
# These tests are derived from real failures of the old ParlanTau retail agent
# on the tau2-bench retail benchmark. In that agent each policy was implemented
# as a journey; here each policy is instead expressed as a SINGLE guideline whose
# action is the corresponding section of the retail policy.md (plus the relevant
# generic confirmation rule), with the journey's tools attached and the agent
# description + glossary carried over from the old agent.
#
# Each test reconstructs the interaction (and the tool results that had already
# been gathered, as staged tool events) up to the message on which the benchmark
# agent failed, and asserts the distiller would have produced the correct next
# step the agent missed.
# ============================================================================

from datetime import datetime, timezone
from typing import cast

from lagom import Container
from pytest import fixture

from parlant.core.common import JSONSerializable
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.compass.guideline_matching.guideline_distiller import GuidelineDistiller
from parlant.core.glossary import Term
from parlant.core.sessions import EventSource
from parlant.core.tools import (
    Tool,
    ToolId,
    ToolOverlap,
    ToolParameterDescriptor,
    ToolParameterOptions,
)

from tests.core.stable.engines.compass.guideline_matching.test_guideline_distiller import (
    base_test_that_a_guideline_is_distilled_correctly,
)
from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_staged_tool_event,
    create_term,
)


@fixture
def distiller(container: Container) -> GuidelineDistiller:
    return container[GuidelineDistiller]


# --- Retail agent description and glossary (carried over from the old agent) ---

_RETAIL_AGENT_DESCRIPTION = (
    "You are a customer-service agent for an online retail store. "
    "You help authenticated customers with their orders and account. "
    "You can cancel or modify pending orders, return or exchange delivered orders, "
    "update the customer's default address, and look up profile, order, and product "
    "information. All order modifications are done through tool calling. Never fabricate "
    "that an order was modified if you do not see such data in the tool results. You serve "
    "one customer per conversation and never fabricate information. You are polite, concise, "
    "and factual."
)


def _retail_glossary() -> list[Term]:
    return [
        create_term(
            name="product",
            description=(
                "A type of item sold in our store (e.g. 'T-Shirt'). Each product has a "
                "unique product ID and contains multiple variants."
            ),
        ),
        create_term(
            name="variant",
            description=(
                "A specific version of a product with particular options (e.g. color, size, "
                "material). Each variant has a unique item ID, a price, and an availability "
                "status. Item ID and product ID are different and must not be confused."
            ),
            synonyms=["item", "option", "specification"],
        ),
        create_term(
            name="order status",
            description=(
                "The lifecycle state of an order. Possible values: 'pending' (not yet shipped, "
                "can be cancelled or modified), 'processed' (being prepared, no changes allowed), "
                "'delivered' (received by customer, can be returned or exchanged), 'cancelled'."
            ),
        ),
        create_term(
            name="payment method",
            description=(
                "How a customer pays. Three types exist: gift card, PayPal account, and credit "
                "card. Gift card refunds are immediate; other refunds take 5-7 business days."
            ),
        ),
        create_term(
            name="exchange",
            description=(
                "Swapping a variant in a delivered order for a different variant of the same "
                "product. Cannot change product type. Changes order status to 'exchange "
                "requested'. Customer receives an email with return instructions."
            ),
        ),
    ]


# --- τ-bench retail tool definitions (name + description + parameters) ---


def _str_param(description: str) -> tuple[ToolParameterDescriptor, ToolParameterOptions]:
    return {"type": "string", "description": description}, ToolParameterOptions()


def _str_array_param(description: str) -> tuple[ToolParameterDescriptor, ToolParameterOptions]:
    return {
        "type": "array",
        "item_type": "string",
        "description": description,
    }, ToolParameterOptions()


def _retail_tool(
    name: str,
    description: str,
    parameters: dict[str, tuple[ToolParameterDescriptor, ToolParameterOptions]],
) -> tuple[ToolId, Tool]:
    return ToolId("retail", name), Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description=description,
        metadata={},
        parameters=parameters,
        required=list(parameters.keys()),
        consequential=False,
        overlap=ToolOverlap.AUTO,
    )


_GET_ORDER_DETAILS = _retail_tool(
    "get_order_details",
    "Get the status and details of an order.",
    {"order_id": _str_param("The order id, e.g. '#W0000000' (note the leading '#').")},
)
_GET_USER_DETAILS = _retail_tool(
    "get_user_details",
    "Get the details of a user, including their profile, payment methods, and order ids.",
    {"user_id": _str_param("The user id, e.g. 'sara_doe_496'.")},
)
_GET_PRODUCT_DETAILS = _retail_tool(
    "get_product_details",
    "Get the inventory details of a product, including its variants.",
    {"product_id": _str_param("The product id, e.g. '6086499569' (different from the item id).")},
)
_GET_ITEM_DETAILS = _retail_tool(
    "get_item_details",
    "Get the inventory details of an item (variant).",
    {"item_id": _str_param("The item id (variant), e.g. '1008292230'.")},
)
_LIST_ALL_PRODUCT_TYPES = _retail_tool(
    "list_all_product_types",
    "List the name and product id of all product types.",
    {},
)
_EXCHANGE_DELIVERED_ORDER_ITEMS = _retail_tool(
    "exchange_delivered_order_items",
    "Exchange items in a delivered order for new items of the same product type.",
    {
        "order_id": _str_param("The order id, e.g. '#W0000000' (note the leading '#')."),
        "item_ids": _str_array_param("The item ids to be exchanged."),
        "new_item_ids": _str_array_param(
            "The item ids to exchange for; each matches the item id in the same position and "
            "must be of the same product."
        ),
        "payment_method_id": _str_param(
            "The payment method id used to pay or receive a refund for the price difference, "
            "e.g. 'gift_card_0000000' or 'credit_card_0000000'."
        ),
    },
)

_EXCHANGE_TOOLS = [
    _GET_ORDER_DETAILS,
    _GET_USER_DETAILS,
    _GET_PRODUCT_DETAILS,
    _GET_ITEM_DETAILS,
    _LIST_ALL_PRODUCT_TYPES,
    _EXCHANGE_DELIVERED_ORDER_ITEMS,
]


# --- Policy sections (from tau2-bench retail policy.md), used as guideline actions ---

_EXCHANGE_CONDITION = (
    "The customer's identity has been verified via a successful account lookup, and they want "
    "to exchange items from a delivered order"
)
_EXCHANGE_ACTION = (
    "An order can only be exchanged if its status is 'delivered', and you should check its "
    "status before taking the action. In particular, remember to remind the customer to confirm "
    "they have provided all items to be exchanged. "
    "For a delivered order, each item can be exchanged to an available new item of the same "
    "product but of a different product option. There cannot be any change of product types, "
    "e.g. modify a shirt to a shoe. "
    "The user must provide a payment method to pay or receive a refund of the price difference. "
    "If the user provides a gift card, it must have enough balance to cover the price difference. "
    "Before taking the exchange action (which updates the database), you must list the action "
    "details and obtain explicit user confirmation ('yes') to proceed. "
    "After user confirmation, call exchange_delivered_order_items; the order status will be "
    "changed to 'exchange requested', and the user will receive an email regarding how to return "
    "items. There is no need to place a new order."
)


def _staged(
    tool_name: str, arguments: dict[str, JSONSerializable], result: JSONSerializable
) -> EmittedEvent:
    return create_staged_tool_event(
        cast(
            JSONSerializable,
            {
                "tool_calls": [
                    {
                        "tool_id": f"retail:{tool_name}",
                        "arguments": arguments,
                        "result": {"data": result, "metadata": {}, "control": {}},
                    }
                ]
            },
        )
    )


async def test_that_a_confirmed_exchange_is_distilled_to_calling_the_exchange_tool(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _staged(
            "find_user_id_by_name_zip",
            {"first_name": "Mei", "last_name": "Kovacs", "zip": "28236"},
            "mei_kovacs_8020",
        ),
        _staged(
            "get_user_details",
            {"user_id": "mei_kovacs_8020"},
            {
                "user_id": "mei_kovacs_8020",
                "name": {"first_name": "Mei", "last_name": "Kovacs"},
                "email": "mei.kovacs@example.com",
                "payment_methods": {
                    "credit_card_3124723": {
                        "source": "credit_card",
                        "brand": "visa",
                        "last_four": "7803",
                        "id": "credit_card_3124723",
                    }
                },
                "orders": ["#W6390527", "#W7800651", "#W8065207"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W6390527"},
            {
                "order_id": "#W6390527",
                "user_id": "mei_kovacs_8020",
                "status": "delivered",
                "items": [
                    {
                        "name": "Desk Lamp",
                        "product_id": "6817146515",
                        "item_id": "1569765161",
                        "price": 164.28,
                        "options": {
                            "color": "white",
                            "brightness": "medium",
                            "power source": "USB",
                        },
                    },
                    {
                        "name": "Water Bottle",
                        "product_id": "8310926033",
                        "item_id": "1434748144",
                        "price": 49.72,
                        "options": {"capacity": "750ml", "material": "stainless steel"},
                    },
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 214.0,
                        "payment_method_id": "credit_card_3124723",
                    }
                ],
            },
        ),
        _staged(
            "get_product_details",
            {"product_id": "6817146515"},
            {
                "name": "Desk Lamp",
                "product_id": "6817146515",
                "variants": {
                    "9083642334": {
                        "item_id": "9083642334",
                        "options": {
                            "color": "silver",
                            "brightness": "low",
                            "power source": "AC adapter",
                        },
                        "available": True,
                        "price": 153.23,
                    },
                },
            },
        ),
    ]

    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_EXCHANGE_CONDITION,
        action=_EXCHANGE_ACTION,
        tools=_EXCHANGE_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Mei Kovacs",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi! I'd like to exchange a couple of items I bought recently - a water bottle and a desk lamp. Can you help me with that?",
            ),
            (
                EventSource.AI_AGENT,
                "Hello! I can certainly help you with that exchange. To get started, could you please provide your email address, or your name and zip code?",
            ),
            (
                EventSource.CUSTOMER,
                "I'm not sure what email I used, but my name is Mei Kovacs and my zip code is 28236.",
            ),
            (
                EventSource.AI_AGENT,
                "Hi Mei, I've found your account and order #W6390527, which contains the Desk Lamp and the Water Bottle. Which items would you like to exchange, and to what options?",
            ),
            (
                EventSource.CUSTOMER,
                "For the desk lamp, I'd like to exchange it for a low brightness version that uses an AC adapter. I'm fine with any color.",
            ),
            (
                EventSource.AI_AGENT,
                "We have that available: the Desk Lamp in Silver, low brightness, AC adapter. That variant is $153.23 versus your current $164.28, so you'd receive a $11.05 refund.",
            ),
            (
                EventSource.CUSTOMER,
                "Actually, I'd just like to exchange the desk lamp for the silver, low brightness, AC adapter version. Let's skip the water bottle.",
            ),
            (
                EventSource.AI_AGENT,
                "Here are the details for your proposed exchange: Desk Lamp (White, Medium, USB) -> Desk Lamp (Silver, Low, AC adapter), with an $11.05 refund to your Visa credit card. Shall I proceed?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, please go ahead and process the exchange for the desk lamp.",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="call the exchange_delivered_order_items tool",
    )


# --- Additional τ-bench retail tool definitions ---


def _addr_params() -> dict[str, tuple[ToolParameterDescriptor, ToolParameterOptions]]:
    return {
        "address1": _str_param("Primary address line."),
        "address2": _str_param("Secondary address line (e.g. suite)."),
        "city": _str_param("City."),
        "state": _str_param("State/province code."),
        "country": _str_param("Country."),
        "zip": _str_param("Zip/postal code."),
    }


_FIND_USER_ID_BY_EMAIL = _retail_tool(
    "find_user_id_by_email",
    "Find a user id by email.",
    {"email": _str_param("The email of the user, e.g. 'something@example.com'.")},
)
_FIND_USER_ID_BY_NAME_ZIP = _retail_tool(
    "find_user_id_by_name_zip",
    "Find a user id by first name, last name, and zip code.",
    {
        "first_name": _str_param("The customer's first name, e.g. 'John'."),
        "last_name": _str_param("The customer's last name, e.g. 'Doe'."),
        "zip": _str_param("The customer's zip code, e.g. '12345'."),
    },
)
_CANCEL_PENDING_ORDER = _retail_tool(
    "cancel_pending_order",
    "Cancel a pending order and refund it to the original payment method.",
    {
        "order_id": _str_param("The order id, e.g. '#W0000000' (note the leading '#')."),
        "reason": _str_param("The reason: either 'no longer needed' or 'ordered by mistake'."),
    },
)
_RETURN_DELIVERED_ORDER_ITEMS = _retail_tool(
    "return_delivered_order_items",
    "Return items from a delivered order; the refund goes to the original method or a gift card.",
    {
        "order_id": _str_param("The order id, e.g. '#W0000000' (note the leading '#')."),
        "item_ids": _str_array_param("The item ids to return."),
        "payment_method_id": _str_param(
            "The payment method id to receive the refund (the original method or an existing "
            "gift card), e.g. 'gift_card_0000000'."
        ),
    },
)
_MODIFY_PENDING_ORDER_ADDRESS = _retail_tool(
    "modify_pending_order_address",
    "Modify the shipping address of a pending order.",
    {
        "order_id": _str_param("The order id, e.g. '#W0000000' (note the leading '#')."),
        **_addr_params(),
    },
)
_MODIFY_PENDING_ORDER_PAYMENT = _retail_tool(
    "modify_pending_order_payment",
    "Modify the payment method of a pending order to a single different method.",
    {
        "order_id": _str_param("The order id, e.g. '#W0000000' (note the leading '#')."),
        "payment_method_id": _str_param(
            "The new payment method id, e.g. 'gift_card_0000000' or 'credit_card_0000000'."
        ),
    },
)
_MODIFY_PENDING_ORDER_ITEMS = _retail_tool(
    "modify_pending_order_items",
    "Modify items in a pending order to new variants of the same product (callable once per order).",
    {
        "order_id": _str_param("The order id, e.g. '#W0000000' (note the leading '#')."),
        "item_ids": _str_array_param("The item ids to be modified."),
        "new_item_ids": _str_array_param(
            "The new item ids, each matching the item id in the same position and of the same product."
        ),
        "payment_method_id": _str_param(
            "The payment method id for the price difference, e.g. 'gift_card_0000000'."
        ),
    },
)
_MODIFY_USER_ADDRESS = _retail_tool(
    "modify_user_address",
    "Modify the default address of a user.",
    {"user_id": _str_param("The user id, e.g. 'sara_doe_496'."), **_addr_params()},
)

_AUTH_TOOLS = [_FIND_USER_ID_BY_EMAIL, _FIND_USER_ID_BY_NAME_ZIP]
_CANCEL_TOOLS = [_GET_ORDER_DETAILS, _GET_USER_DETAILS, _CANCEL_PENDING_ORDER]
_RETURN_TOOLS = [_GET_ORDER_DETAILS, _GET_USER_DETAILS, _RETURN_DELIVERED_ORDER_ITEMS]
_MODIFY_ITEMS_TOOLS = [
    _GET_ORDER_DETAILS,
    _GET_USER_DETAILS,
    _GET_PRODUCT_DETAILS,
    _GET_ITEM_DETAILS,
    _LIST_ALL_PRODUCT_TYPES,
    _MODIFY_PENDING_ORDER_ITEMS,
]
_MODIFY_PAYMENT_TOOLS = [_GET_ORDER_DETAILS, _GET_USER_DETAILS, _MODIFY_PENDING_ORDER_PAYMENT]
_MODIFY_ADDRESS_TOOLS = [_GET_ORDER_DETAILS, _GET_USER_DETAILS, _MODIFY_PENDING_ORDER_ADDRESS]
_MODIFY_USER_ADDRESS_TOOLS = [_MODIFY_USER_ADDRESS]


# --- Policy sections (from retail policy.md) used as guideline actions ---

_AUTH_CONDITION = "The customer has not yet been identified (authenticated) in this conversation"
_AUTH_ACTION = (
    "At the beginning of the conversation you must authenticate the user's identity by locating "
    "their user id via email, or via first name + last name + zip code. This must be done even if "
    "the user already provides a user id. Prefer email lookup first; if the user can't provide an "
    "email, look them up by name and zip code. If the necessary details haven't been provided yet, "
    "ask the customer for their email, or their first name, last name, and zip code."
)

_CANCEL_CONDITION = (
    "The customer's identity has been verified via a successful account lookup, and they want to "
    "cancel an entire order"
)
_CANCEL_ACTION = (
    "An order can only be cancelled if its status is 'pending', and you should check its status "
    "before taking the action. The user needs to confirm the order id and the reason for the "
    "cancellation - either 'no longer needed' or 'ordered by mistake'; other reasons are not "
    "acceptable. Before cancelling (which updates the database), list the action details and obtain "
    "explicit user confirmation ('yes'). After user confirmation, call cancel_pending_order; the "
    "order status will be changed to 'cancelled', and the total will be refunded to the original "
    "payment method (immediately for a gift card, otherwise in 5-7 business days)."
)

_RETURN_CONDITION = (
    "The customer's identity has been verified via a successful account lookup, and they want to "
    "return items from an order"
)
_RETURN_ACTION = (
    "An order can only be returned if its status is 'delivered', and you should check its status "
    "before taking the action. The user needs to confirm the order id and the list of items to "
    "return, and provide a payment method to receive the refund (the original payment method, or "
    "an existing gift card). Before processing the return (which updates the database), list the "
    "details and obtain explicit user confirmation ('yes'). After confirmation, call "
    "return_delivered_order_items; the order status will be changed to 'return requested' and the "
    "user will receive an email regarding how to return items."
)

_MODIFY_ITEMS_CONDITION = (
    "The customer's identity has been verified via a successful account lookup, and they want to "
    "modify the items in a pending order"
)
_MODIFY_ITEMS_ACTION = (
    "An order can only be modified if its status is 'pending'. This action can only be called once "
    "and will change the order status to 'pending (items modified)'; afterwards you can no longer "
    "modify or cancel the order, so confirm all details and be cautious. Remind the customer to "
    "confirm they have provided all the items they want to modify. Each item can be modified to an "
    "available new item of the same product but a different option; there cannot be any change of "
    "product type. The user must provide a payment method to pay or receive a refund of the price "
    "difference. Before modifying (which updates the database), list the details and obtain explicit "
    "user confirmation ('yes'). After confirmation, call modify_pending_order_items."
)

_MODIFY_PAYMENT_CONDITION = (
    "The customer's identity has been verified via a successful account lookup, and they want to "
    "change the payment method of a pending order"
)
_MODIFY_PAYMENT_ACTION = (
    "An order can only be modified if its status is 'pending'. The user can only choose a single "
    "payment method different from the original one. If they choose a gift card, it must have enough "
    "balance to cover the total amount. Before modifying (which updates the database), list the "
    "details and obtain explicit user confirmation ('yes'). After confirmation, call "
    "modify_pending_order_payment; the order status stays 'pending' and the original payment method "
    "is refunded (immediately for a gift card, otherwise in 5-7 business days)."
)

_MODIFY_ADDRESS_CONDITION = (
    "The customer's identity has been verified via a successful account lookup, and they want to "
    "change the shipping address of a pending order"
)
_MODIFY_ADDRESS_ACTION = (
    "An order can only be modified if its status is 'pending', and you should check its status "
    "before taking the action. For a pending order you can modify its shipping address. Before "
    "modifying (which updates the database), summarize the old vs new address and obtain explicit "
    "user confirmation ('yes'). After confirmation, call modify_pending_order_address."
)

_MODIFY_USER_ADDRESS_CONDITION = (
    "The customer's identity has been verified via a successful account lookup, and they want to "
    "update their default address"
)
_MODIFY_USER_ADDRESS_ACTION = (
    "Update the customer's default (profile) address. Before updating (which updates the database), "
    "summarize the new address and obtain explicit user confirmation ('yes'). After confirmation, "
    "call modify_user_address with the customer's user id and the new address."
)


# --- Reusable staged tool events ---


def _find_by_name_zip(first: str, last: str, zip_code: str, user_id: str) -> EmittedEvent:
    return _staged(
        "find_user_id_by_name_zip",
        {"first_name": first, "last_name": last, "zip": zip_code},
        user_id,
    )


async def test_that_a_confirmed_return_is_distilled_to_calling_the_return_tool(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Daiki", "Johnson", "80273", "daiki_johnson_9523"),
        _staged(
            "get_user_details",
            {"user_id": "daiki_johnson_9523"},
            {
                "user_id": "daiki_johnson_9523",
                "name": {"first_name": "Daiki", "last_name": "Johnson"},
                "payment_methods": {"paypal_5364164": {"source": "paypal", "id": "paypal_5364164"}},
                "orders": ["#W9502127"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W9502127"},
            {
                "order_id": "#W9502127",
                "user_id": "daiki_johnson_9523",
                "status": "delivered",
                "items": [
                    {
                        "name": "Air Purifier",
                        "product_id": "3821016478",
                        "item_id": "8302289002",
                        "price": 547.55,
                        "options": {"room size": "large"},
                    },
                    {
                        "name": "Robotic Vacuum Cleaner",
                        "product_id": "4762493753",
                        "item_id": "4965355367",
                        "price": 578.49,
                        "options": {"capacity": "1L"},
                    },
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 2623.69,
                        "payment_method_id": "paypal_5364164",
                    }
                ],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RETURN_CONDITION,
        action=_RETURN_ACTION,
        tools=_RETURN_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Daiki Johnson",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I need to return an air purifier and a vacuum cleaner from my recent order.",
            ),
            (
                EventSource.AI_AGENT,
                "I can help with that. Could you provide your email, or your name and zip code, and the order ID?",
            ),
            (
                EventSource.CUSTOMER,
                "I don't have an email, but my name is Daiki Johnson and my zip code is 80273. The order is #W9502127.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks - I've located order #W9502127. You'd like to return the Air Purifier and the Robotic Vacuum Cleaner, for a refund of $1,126.04. Where should the refund go?",
            ),
            (
                EventSource.CUSTOMER,
                "Send the refund to my original payment method, PayPal.",
            ),
            (
                EventSource.AI_AGENT,
                "To confirm: return the Air Purifier and Robotic Vacuum Cleaner from #W9502127, $1,126.04 refunded to your PayPal. Shall I proceed?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, please go ahead and process the return right now.",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="call the return_delivered_order_items tool",
    )


async def test_that_a_confirmed_item_modification_is_distilled_to_calling_the_modify_items_tool(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _staged(
            "get_order_details",
            {"order_id": "#W9300146"},
            {
                "order_id": "#W9300146",
                "user_id": "aarav_anderson_8794",
                "status": "pending",
                "items": [
                    {
                        "name": "Desk Lamp",
                        "product_id": "6817146515",
                        "item_id": "8384507844",
                        "price": 153.23,
                        "options": {
                            "color": "white",
                            "brightness": "high",
                            "power source": "USB",
                        },
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 153.23,
                        "payment_method_id": "gift_card_7245228",
                    }
                ],
            },
        ),
        _find_by_name_zip("Aarav", "Anderson", "76137", "aarav_anderson_8794"),
        _staged(
            "get_user_details",
            {"user_id": "aarav_anderson_8794"},
            {
                "user_id": "aarav_anderson_8794",
                "name": {"first_name": "Aarav", "last_name": "Anderson"},
                "payment_methods": {
                    "gift_card_7245228": {
                        "source": "gift_card",
                        "balance": 53.6,
                        "id": "gift_card_7245228",
                    }
                },
                "orders": ["#W9300146"],
            },
        ),
        _staged(
            "get_product_details",
            {"product_id": "6817146515"},
            {
                "name": "Desk Lamp",
                "product_id": "6817146515",
                "variants": {
                    "5320792178": {
                        "item_id": "5320792178",
                        "options": {
                            "color": "black",
                            "brightness": "medium",
                            "power source": "AC adapter",
                        },
                        "available": True,
                        "price": 135.24,
                    }
                },
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_MODIFY_ITEMS_CONDITION,
        action=_MODIFY_ITEMS_ACTION,
        tools=_MODIFY_ITEMS_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Aarav Anderson",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I placed order #W9300146 and I'd like to swap the Desk Lamp in it for the cheapest Desk Lamp variant you have.",
            ),
            (
                EventSource.AI_AGENT,
                "The cheapest available Desk Lamp is the black, medium-brightness, AC-adapter variant for $135.24, an $17.99 refund. First, could you confirm your identity with your name and zip code?",
            ),
            (
                EventSource.CUSTOMER,
                "My name is Aarav Anderson and my zip code is 76137.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Aarav - I've verified your account. To confirm: swap the Desk Lamp in #W9300146 to the black medium AC-adapter variant, with $17.99 refunded. How would you like the difference handled?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, please go ahead and make the swap. I'm fine with the $17.99 being refunded to my gift card. Thank you!",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="call the modify_pending_order_items tool",
    )


async def test_that_a_confirmed_default_address_update_is_distilled_to_calling_modify_user_address(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Lucas", "Santos", "80239", "lucas_santos_6600"),
        _staged(
            "get_user_details",
            {"user_id": "lucas_santos_6600"},
            {
                "user_id": "lucas_santos_6600",
                "name": {"first_name": "Lucas", "last_name": "Santos"},
                "address": {
                    "address1": "986 Lakeview Drive",
                    "city": "Denver",
                    "state": "CO",
                    "country": "USA",
                    "zip": "80239",
                },
                "orders": ["#W1588712", "#W7895761"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W7895761"},
            {
                "order_id": "#W7895761",
                "user_id": "lucas_santos_6600",
                "status": "delivered",
                "address": {
                    "address1": "943 Maple Drive",
                    "address2": "Suite 356",
                    "city": "Chicago",
                    "state": "IL",
                    "country": "USA",
                    "zip": "60621",
                },
                "items": [],
                "payment_history": [],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_MODIFY_USER_ADDRESS_CONDITION,
        action=_MODIFY_USER_ADDRESS_ACTION,
        tools=_MODIFY_USER_ADDRESS_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Lucas Santos",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I'd like to update my default shipping address.",
            ),
            (
                EventSource.AI_AGENT,
                "Happy to help. Could you confirm your identity with your email, or name and zip code?",
            ),
            (
                EventSource.CUSTOMER,
                "My name is Lucas Santos and my zip code is 80239. I'd like my default address changed to my daughter's place: 943 Maple Drive, Suite 356, Chicago, IL 60621.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Lucas - I've verified your account. To confirm, I'll set your default address to 943 Maple Drive, Suite 356, Chicago, IL 60621. Shall I proceed?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, please go ahead and update my default address to my daughter's place in Chicago.",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="call the modify_user_address tool",
    )


async def test_that_a_confirmed_payment_change_is_distilled_to_calling_modify_payment(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Lei", "Li", "85033", "lei_li_6575"),
        _staged(
            "get_user_details",
            {"user_id": "lei_li_6575"},
            {
                "user_id": "lei_li_6575",
                "name": {"first_name": "Lei", "last_name": "Li"},
                "payment_methods": {
                    "paypal_8225578": {"source": "paypal", "id": "paypal_8225578"},
                    "credit_card_4466831": {
                        "source": "credit_card",
                        "brand": "mastercard",
                        "last_four": "2697",
                        "id": "credit_card_4466831",
                    },
                },
                "orders": ["#W5166363"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W5166363"},
            {
                "order_id": "#W5166363",
                "user_id": "lei_li_6575",
                "status": "pending",
                "items": [
                    {
                        "name": "Laptop",
                        "product_id": "4760268021",
                        "item_id": "8997785118",
                        "price": 2249.83,
                        "options": {"screen size": "15-inch"},
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 2249.83,
                        "payment_method_id": "paypal_8225578",
                    }
                ],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_MODIFY_PAYMENT_CONDITION,
        action=_MODIFY_PAYMENT_ACTION,
        tools=_MODIFY_PAYMENT_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Lei Li",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, my name is Lei Li, zip code 85033. For my pending order #W5166363, can I change the payment method from PayPal to my credit card?",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Lei - I've verified your account. Your order #W5166363 is pending. To confirm, I'll switch its payment method from PayPal to your Mastercard ending 2697. Shall I proceed?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, please go ahead and change it to my credit card. Thank you.",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="call the modify_pending_order_payment tool",
    )


async def test_that_a_confirmed_order_address_change_is_distilled_to_calling_modify_address(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Mei", "Patel", "76165", "mei_patel_7272"),
        _staged(
            "get_user_details",
            {"user_id": "mei_patel_7272"},
            {
                "user_id": "mei_patel_7272",
                "name": {"first_name": "Mei", "last_name": "Patel"},
                "orders": ["#W4082615"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W4082615"},
            {
                "order_id": "#W4082615",
                "user_id": "mei_patel_7272",
                "status": "pending",
                "address": {
                    "address1": "443 Maple Drive",
                    "address2": "Suite 394",
                    "city": "Fort Worth",
                    "state": "TX",
                    "country": "USA",
                    "zip": "76165",
                },
                "items": [],
                "payment_history": [],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_MODIFY_ADDRESS_CONDITION,
        action=_MODIFY_ADDRESS_ACTION,
        tools=_MODIFY_ADDRESS_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Mei Patel",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I think I typed my shipping address wrong on a pending order. My name is Mei Patel, zip code 76165.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Mei - I've verified your account. Your pending order #W4082615 ships to 443 Maple Drive, Suite 394, Fort Worth, TX 76165. What should it be?",
            ),
            (
                EventSource.CUSTOMER,
                "It should be 445 Maple Drive, Suite 394, Fort Worth, TX 76165.",
            ),
            (
                EventSource.AI_AGENT,
                "Got it. To confirm, I'll update the shipping address on #W4082615 to 445 Maple Drive, Suite 394, Fort Worth, TX 76165. Shall I proceed?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, please go ahead and update it to 445 Maple Drive.",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="call the modify_pending_order_address tool",
    )


async def test_that_a_confirmed_cancellation_is_distilled_to_calling_the_cancel_tool(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Daiki", "Sanchez", "46236", "daiki_sanchez_3253"),
        _staged(
            "get_user_details",
            {"user_id": "daiki_sanchez_3253"},
            {
                "user_id": "daiki_sanchez_3253",
                "name": {"first_name": "Daiki", "last_name": "Sanchez"},
                "payment_methods": {
                    "credit_card_1058222": {
                        "source": "credit_card",
                        "brand": "visa",
                        "last_four": "6593",
                        "id": "credit_card_1058222",
                    }
                },
                "orders": ["#W9348897"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W9348897"},
            {
                "order_id": "#W9348897",
                "user_id": "daiki_sanchez_3253",
                "status": "pending",
                "items": [
                    {
                        "name": "Action Camera",
                        "product_id": "3377618313",
                        "item_id": "1586641416",
                        "price": 481.5,
                        "options": {"resolution": "4K"},
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 1166.98,
                        "payment_method_id": "credit_card_1058222",
                    }
                ],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CANCEL_CONDITION,
        action=_CANCEL_ACTION,
        tools=_CANCEL_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Daiki Sanchez",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, my name is Daiki Sanchez, zip 46236. I'd like to cancel my whole order #W9348897 so I can reorder.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Daiki - I've verified your account and your order #W9348897 is pending. Could you tell me the reason for the cancellation?",
            ),
            (
                EventSource.CUSTOMER,
                "Ordered by mistake.",
            ),
            (
                EventSource.AI_AGENT,
                "To confirm: I'll cancel order #W9348897 (reason: ordered by mistake) and refund $1,166.98 to your Visa ending 6593. Nothing has been cancelled yet - shall I proceed?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, please go ahead and cancel it.",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="call the cancel_pending_order tool",
    )


async def test_that_an_exchange_with_everything_gathered_is_distilled_to_asking_for_confirmation(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Yusuf", "Rossi", "19122", "yusuf_rossi_9620"),
        _staged(
            "get_user_details",
            {"user_id": "yusuf_rossi_9620"},
            {
                "user_id": "yusuf_rossi_9620",
                "name": {"first_name": "Yusuf", "last_name": "Rossi"},
                "payment_methods": {
                    "credit_card_9513926": {
                        "source": "credit_card",
                        "brand": "mastercard",
                        "last_four": "2478",
                        "id": "credit_card_9513926",
                    }
                },
                "orders": ["#W2378156"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W2378156"},
            {
                "order_id": "#W2378156",
                "user_id": "yusuf_rossi_9620",
                "status": "delivered",
                "items": [
                    {
                        "name": "Smart Thermostat",
                        "product_id": "4896585277",
                        "item_id": "4983901480",
                        "price": 262.47,
                        "options": {"compatibility": "Apple HomeKit"},
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 262.47,
                        "payment_method_id": "credit_card_9513926",
                    }
                ],
            },
        ),
        _staged(
            "get_product_details",
            {"product_id": "4896585277"},
            {
                "name": "Smart Thermostat",
                "product_id": "4896585277",
                "variants": {
                    "8722653925": {
                        "item_id": "8722653925",
                        "options": {"compatibility": "Google Assistant"},
                        "available": True,
                        "price": 245.84,
                    }
                },
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_EXCHANGE_CONDITION,
        action=_EXCHANGE_ACTION,
        tools=_EXCHANGE_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Yusuf Rossi",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I'd like to exchange the Smart Thermostat in my delivered order. My name is Yusuf Rossi, zip 19122, order #W2378156.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Yusuf - I've located order #W2378156. For the Smart Thermostat we have the Google Assistant compatible variant available (a $16.63 refund versus your current one).",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, that's exactly the item and option I want to exchange - the thermostat for the Google Assistant compatible version.",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action=(
            "list the exchange details and ask the customer for explicit confirmation before "
            "processing the exchange"
        ),
    )


async def test_that_an_unauthenticated_customer_is_distilled_to_looking_up_their_account(
    distiller: GuidelineDistiller,
) -> None:
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_AUTH_CONDITION,
        action=_AUTH_ACTION,
        tools=_AUTH_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Sofia Li",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I'd like to return a digital camera I received.",
            ),
            (
                EventSource.AI_AGENT,
                "I can help with that. First I need to verify your account - could you provide your email, or your first name, last name, and zip code?",
            ),
            (
                EventSource.CUSTOMER,
                "I don't remember which email I used, but my name is Sofia Li and my zip code is 78260.",
            ),
        ],
        expected_relevant=True,
        expected_distilled_action="call the find_user_id_by_name_zip tool",
    )


async def test_that_an_already_verified_exchange_is_distilled_to_calling_the_exchange_tool(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Aarav", "Anderson", "19031", "aarav_anderson_8794"),
        _staged(
            "get_order_details",
            {"order_id": "#W3470184"},
            {
                "order_id": "#W3470184",
                "user_id": "aarav_anderson_8794",
                "status": "delivered",
                "items": [
                    {
                        "name": "Wireless Earbuds",
                        "product_id": "9924732112",
                        "item_id": "2757705742",
                        "price": 232.49,
                        "options": {"water resistance": "IPX7", "color": "blue"},
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 232.49,
                        "payment_method_id": "credit_card_6182206",
                    }
                ],
            },
        ),
        _staged(
            "get_user_details",
            {"user_id": "aarav_anderson_8794"},
            {
                "user_id": "aarav_anderson_8794",
                "name": {"first_name": "Aarav", "last_name": "Anderson"},
                "payment_methods": {
                    "credit_card_6182206": {
                        "source": "credit_card",
                        "brand": "visa",
                        "last_four": "1862",
                        "id": "credit_card_6182206",
                    }
                },
                "orders": ["#W3470184"],
            },
        ),
        _staged(
            "get_item_details",
            {"item_id": "1646531091"},
            {
                "item_id": "1646531091",
                "options": {"water resistance": "IPX4", "color": "blue", "battery life": "6 hours"},
                "available": True,
                "price": 258.97,
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_EXCHANGE_CONDITION,
        action=_EXCHANGE_ACTION,
        tools=_EXCHANGE_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Aarav Anderson",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I ordered Wireless Earbuds with IPX7 water resistance by mistake. My name is Aarav Anderson, zip 19031, order #W3470184. I'd like to exchange them for the IPX4 version.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Aarav - I've verified your account and located order #W3470184. The IPX4 Wireless Earbuds are available. To confirm: exchange the IPX7 earbuds for the IPX4 variant, with the price difference on your Visa ending 1862. Shall I proceed?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, I confirm - please exchange the IPX7 earbuds for the IPX4 version.",
            ),
            (
                EventSource.AI_AGENT,
                "To proceed with processing this exchange, I need your user ID or email address to access your account details.",
            ),
            (
                EventSource.CUSTOMER,
                "I'm sorry, but I don't remember my email address or user ID. Is there another way you can verify my account?",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="call the exchange_delivered_order_items tool",
    )


# ============================================================================
# Cases where the correct next step is NOT a tool call (asking the customer,
# rejecting an out-of-policy request, or the pre-mutation confirmation gate).
# These are constructed retail scenarios, not necessarily old-agent failures.
# ============================================================================


async def test_that_a_cancellation_without_a_reason_is_distilled_to_asking_for_the_reason(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Emma", "Kim", "10001", "emma_kim_2992"),
        _staged(
            "get_user_details",
            {"user_id": "emma_kim_2992"},
            {
                "user_id": "emma_kim_2992",
                "name": {"first_name": "Emma", "last_name": "Kim"},
                "orders": ["#W2189999"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W2189999"},
            {
                "order_id": "#W2189999",
                "user_id": "emma_kim_2992",
                "status": "pending",
                "items": [
                    {
                        "name": "Backpack",
                        "product_id": "2524789262",
                        "item_id": "5726859009",
                        "price": 198.0,
                        "options": {"color": "black"},
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 198.0,
                        "payment_method_id": "gift_card_3300242",
                    }
                ],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CANCEL_CONDITION,
        action=_CANCEL_ACTION,
        tools=_CANCEL_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Emma Kim",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I'd like to cancel an order. My name is Emma Kim, zip code 10001.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Emma - I've verified your account. Which order would you like to cancel?",
            ),
            (EventSource.CUSTOMER, "Order #W2189999, please."),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="ask the customer for the reason for the cancellation",
    )


async def test_that_cancelling_a_non_pending_order_is_distilled_to_informing_it_cannot_be_cancelled(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Liam", "Garcia", "30303", "liam_garcia_4517"),
        _staged(
            "get_user_details",
            {"user_id": "liam_garcia_4517"},
            {
                "user_id": "liam_garcia_4517",
                "name": {"first_name": "Liam", "last_name": "Garcia"},
                "orders": ["#W7654321"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W7654321"},
            {
                "order_id": "#W7654321",
                "user_id": "liam_garcia_4517",
                "status": "delivered",
                "items": [
                    {
                        "name": "Bluetooth Speaker",
                        "product_id": "4768869376",
                        "item_id": "7597543861",
                        "price": 302.67,
                        "options": {"color": "blue"},
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 302.67,
                        "payment_method_id": "credit_card_7901829",
                    }
                ],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_CANCEL_CONDITION,
        action=_CANCEL_ACTION,
        tools=_CANCEL_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Liam Garcia",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, my name is Liam Garcia, zip 30303. I'd like to cancel my order #W7654321.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Liam - I've verified your account and pulled up order #W7654321.",
            ),
            (EventSource.CUSTOMER, "Great, please cancel the whole thing."),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action=(
            "tell the customer that the order cannot be cancelled because only pending orders "
            "can be cancelled (do not cancel it)"
        ),
    )


async def test_that_returning_a_non_delivered_order_is_distilled_to_informing_it_cannot_be_returned(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Sofia", "Li", "78260", "sofia_li_9219"),
        _staged(
            "get_user_details",
            {"user_id": "sofia_li_9219"},
            {
                "user_id": "sofia_li_9219",
                "name": {"first_name": "Sofia", "last_name": "Li"},
                "orders": ["#W8855135"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W8855135"},
            {
                "order_id": "#W8855135",
                "user_id": "sofia_li_9219",
                "status": "pending",
                "items": [
                    {
                        "name": "Digital Camera",
                        "product_id": "8940227892",
                        "item_id": "9228757377",
                        "price": 2895.55,
                        "options": {"resolution": "24MP"},
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 2895.55,
                        "payment_method_id": "credit_card_3563913",
                    }
                ],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_RETURN_CONDITION,
        action=_RETURN_ACTION,
        tools=_RETURN_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Sofia Li",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, I'd like to return the digital camera from order #W8855135. My name is Sofia Li, zip 78260.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Sofia - I've verified your account and pulled up order #W8855135.",
            ),
            (EventSource.CUSTOMER, "Please go ahead and start the return for the camera."),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action=(
            "tell the customer that the order cannot be returned because only delivered orders "
            "can be returned (do not process a return)"
        ),
    )


async def test_that_an_address_change_without_the_new_address_is_distilled_to_asking_for_it(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Noah", "Brown", "94016", "noah_brown_6789"),
        _staged(
            "get_user_details",
            {"user_id": "noah_brown_6789"},
            {
                "user_id": "noah_brown_6789",
                "name": {"first_name": "Noah", "last_name": "Brown"},
                "orders": ["#W3344556"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W3344556"},
            {
                "order_id": "#W3344556",
                "user_id": "noah_brown_6789",
                "status": "pending",
                "address": {
                    "address1": "12 Pine Street",
                    "address2": "Apt 4",
                    "city": "San Francisco",
                    "state": "CA",
                    "country": "USA",
                    "zip": "94016",
                },
                "items": [],
                "payment_history": [],
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_MODIFY_ADDRESS_CONDITION,
        action=_MODIFY_ADDRESS_ACTION,
        tools=_MODIFY_ADDRESS_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Noah Brown",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, my name is Noah Brown, zip 94016. I need to change the shipping address on my pending order #W3344556.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Noah - I've verified your account and your order #W3344556 is pending, so its shipping address can still be changed.",
            ),
            (EventSource.CUSTOMER, "Great, let's update it."),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action="ask the customer for the new shipping address",
    )


async def test_that_a_fully_specified_item_change_is_distilled_to_asking_for_confirmation(
    distiller: GuidelineDistiller,
) -> None:
    staged_events = [
        _find_by_name_zip("Ava", "Martin", "60614", "ava_martin_1123"),
        _staged(
            "get_user_details",
            {"user_id": "ava_martin_1123"},
            {
                "user_id": "ava_martin_1123",
                "name": {"first_name": "Ava", "last_name": "Martin"},
                "payment_methods": {
                    "gift_card_5519111": {
                        "source": "gift_card",
                        "balance": 120.0,
                        "id": "gift_card_5519111",
                    }
                },
                "orders": ["#W5512098"],
            },
        ),
        _staged(
            "get_order_details",
            {"order_id": "#W5512098"},
            {
                "order_id": "#W5512098",
                "user_id": "ava_martin_1123",
                "status": "pending",
                "items": [
                    {
                        "name": "T-Shirt",
                        "product_id": "9523456873",
                        "item_id": "9612497925",
                        "price": 50.88,
                        "options": {"color": "blue", "size": "M"},
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 50.88,
                        "payment_method_id": "gift_card_5519111",
                    }
                ],
            },
        ),
        _staged(
            "get_product_details",
            {"product_id": "9523456873"},
            {
                "name": "T-Shirt",
                "product_id": "9523456873",
                "variants": {
                    "8124970213": {
                        "item_id": "8124970213",
                        "options": {"color": "red", "size": "M"},
                        "available": True,
                        "price": 50.88,
                    }
                },
            },
        ),
    ]
    await base_test_that_a_guideline_is_distilled_correctly(
        distiller,
        condition=_MODIFY_ITEMS_CONDITION,
        action=_MODIFY_ITEMS_ACTION,
        tools=_MODIFY_ITEMS_TOOLS,
        agent_description=_RETAIL_AGENT_DESCRIPTION,
        customer_name="Ava Martin",
        terms=_retail_glossary(),
        conversation=[
            (
                EventSource.CUSTOMER,
                "Hi, my name is Ava Martin, zip 60614. On my pending order #W5512098 I'd like to change the blue T-Shirt to the red one, same size.",
            ),
            (
                EventSource.AI_AGENT,
                "Thanks, Ava - I've verified your account. The red T-Shirt (size M) is available at the same price, so there's no price difference. Is that the only change you'd like to make?",
            ),
            (
                EventSource.CUSTOMER,
                "Yes, that's the only change. Any difference can go to my gift card.",
            ),
        ],
        staged_events=staged_events,
        expected_relevant=True,
        expected_distilled_action=(
            "summarize the proposed item change and ask the customer to confirm with 'yes' before "
            "making the modification"
        ),
    )
