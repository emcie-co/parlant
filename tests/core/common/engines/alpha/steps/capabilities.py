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

from typing import Any
from pytest_bdd import given, parsers
from parlant.core.capabilities import CapabilityStore
from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


CAPABILITIES: dict[str, dict[str, Any]] = {
    "offer_loan": {
        "title": "offer_loan",
        "description": "You can offer a loan of up to 10,000$ to the customer. The lone is immediately added to the customer's balance.",
        "signals": ["offering loan", "low balance", "increase balance", "need more money"],
    },
    "replace_card": {
        "title": "replace_card",
        "description": "Issue and send a replacement for the customer's credit or debit card if it is lost, stolen, or damaged.",
        "signals": [
            "my card was stolen",
            "I lost my card",
            "need a new card",
            "replace my credit card",
        ],
    },
    "lock_card": {
        "title": "lock_card",
        "description": "Temporarily freeze a customer's credit or debit card to prevent any transactions. This is a reversible action often used when a card is misplaced.",
        "signals": [
            "freeze my card",
            "I misplaced my card",
            "lock my account",
            "stop payments on my card",
        ],
    },
    "reset_password": {
        "title": "reset_password",
        "description": "Assist the customer in resetting the password for their online account if they have forgotten it or are locked out.",
        "signals": [
            "forgot my password",
            "can't log in",
            "need to reset my password",
            "change my login details",
        ],
    },
    "increase_limit": {
        "title": "increase_limit",
        "description": "Offer to increase the customer's credit limit on their credit card account.",
        "signals": [],
    },
    "decrease_limit": {
        "title": "decrease_limit",
        "description": "Offer to decrease the customer's credit limit on their credit card account, which can be a tool for managing spending.",
        "signals": [
            "save money",
            "reduce my spending ability",
            "I want a lower limit",
        ],
    },
    "cancel_subscription": {
        "title": "cancel_subscription",
        "description": "Assist the customer in identifying and canceling recurring subscriptions to online services that are charged to their account. Can help reduce the customer's spending.",
        "signals": [
            "stop a recurring payment",
            "reduce spending",
            "manage my subscriptions",
        ],
    },
    "switch_delivery_method": {
        "title": "switch_delivery_method",
        "description": "Allow the customer to change the shipping or delivery method for an existing order that has not yet been shipped. Possible options are UPS, FEDEX, or private courier.",
        "signals": [
            "change my shipping method",
            "switch delivery service",
            "can I get faster shipping",
            "choose a different delivery option",
        ],
    },
    "check_order_status": {
        "title": "check_order_status",
        "description": "Provide the customer with the current status of their order, such as 'processing', 'awaiting shipment', or 'shipped'.",
        "signals": [
            "has my order shipped yet",
            "where is my order",
        ],
    },
    "check_balance": {
        "title": "check_balance",
        "description": "Provide the customer with the current balance of their bank account or credit card.",
        "signals": [
            "what is my account balance",
            "how much money do I have",
        ],
    },
    "check_order_location": {
        "title": "check_order_location",
        "description": "Provide the current physical location or detailed tracking information for a customer's order that has already been shipped.",
        "signals": [
            "track my package",
            "where is my package right now",
            "find my order's location",
            "delivery tracking",
        ],
    },
    "offer_loan_no_minors_in_description": {
        "title": "offer_loan_no_minors_in_description",
        "description": "You can offer a loan of up to 10,000$ to the customer. Do not offer this to customers under the age of 21.",
        "signals": [
            "need a loan",
            "I need some money",
            "can I borrow money",
        ],
    },
    "reset_router": {
        "title": "reset_router",
        "description": "Perform a remote reset of the customer's internet router. This action, also known as PDMM, does not require the customer to have physical access to the device. Use simple, non-technical language when explaining this to the customer.",
        "signals": [
            "my internet is not working",
            "the router is broken",
            "no connection",
        ],
    },
}


@step(given, parsers.parse('the capability "{capability_name}"'))
def given_a_capability(
    context: ContextOfTest,
    capability_name: str,
) -> None:
    capability_store = context.container[CapabilityStore]
    context.sync_await(capability_store.create_capability(**CAPABILITIES[capability_name]))
