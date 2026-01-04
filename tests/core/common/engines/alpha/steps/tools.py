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

from typing import Any, cast
from pytest_bdd import given, parsers

from parlant.core.tools import ToolParameterOptions
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipEntity,
    RelationshipStore,
    RelationshipKind,
)
from parlant.core.agents import AgentId, AgentStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationStore,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import LocalToolService, ToolId

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


TOOLS: dict[str, dict[str, Any]] = {
    "get_terrys_offering": {
        "name": "get_terrys_offering",
        "description": "Explain Terry's offering",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "get_available_drinks": {
        "name": "get_available_drinks",
        "description": "Get the drinks available in stock",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "get_available_toppings": {
        "name": "get_available_toppings",
        "description": "Get the toppings available in stock",
        "module_path": "tests.tool_utilities",
        "consequential": True,
        "parameters": {},
        "required": [],
    },
    "expert_answer": {
        "name": "expert_answer",
        "description": "Get answers to questions by consulting documentation",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "user_query": {
                "type": "string",
                "description": "The query from the customer",
            }
        },
        "required": ["user_query"],
    },
    "get_available_product_by_type": {
        "name": "get_available_product_by_type",
        "description": "Get the products available in stock by type",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "product_type": {
                "type": "string",
                "description": "The type of product (either 'drinks' or 'toppings')",
                "enum": ["drinks", "toppings"],
            }
        },
        "required": ["product_type"],
    },
    "add": {
        "name": "add",
        "description": "Getting the addition calculation between two numbers",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "first_number": {
                "type": "number",
                "description": "The first number",
            },
            "second_number": {
                "type": "number",
                "description": "The second number",
            },
        },
        "required": ["first_number", "second_number"],
    },
    "multiply": {
        "name": "multiply",
        "description": "Getting the multiplication calculation between two numbers",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "first_number": {
                "type": "number",
                "description": "The first number",
            },
            "second_number": {
                "type": "number",
                "description": "The second number",
            },
        },
        "required": ["first_number", "second_number"],
    },
    "get_account_balance": {
        "name": "get_account_balance",
        "description": "Get the account balance by given name",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "account_name": {
                "type": "string",
                "description": "The name of the account",
            }
        },
        "required": ["account_name"],
    },
    "get_account_loans": {
        "name": "get_account_loans",
        "description": "Get the account loans by given name",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "account_name": {
                "type": "string",
                "description": "The name of the account",
            }
        },
        "required": ["account_name"],
    },
    "transfer_money": {
        "name": "transfer_money",
        "description": "Transfer money from one account to another",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "amount": {"type": "integer", "description": "The number of coins to transfer"},
            "from_account": {
                "type": "string",
                "description": "The name of the account from which money will be transferred",
            },
            "to_account": {
                "type": "string",
                "description": "The name of the account to which money will be transferred",
            },
        },
        "required": ["amount", "from_account", "to_account"],
    },
    "check_fruit_price": {
        "name": "check_fruit_price",
        "description": "Reports the price of 1 kg of a certain fruit",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "fruit": {
                "type": "string",
                "description": "Fruit to check for",
            },
        },
        "required": ["fruit"],
    },
    "check_vegetable_price": {
        "name": "check_vegetable_price",
        "description": "Reports the price of 1 kg of a certain vegetable",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "vegetable": {
                "type": "string",
                "description": "Vegetable to check for",
            },
        },
        "required": ["vegetable"],
    },
    "recommend_drink": {
        "name": "recommend_drink",
        "description": "Recommends a drink based on the user's age",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "user_is_adult": {
                "type": "boolean",
            },
        },
        "required": ["user_is_adult"],
    },
    "check_username_validity": {
        "name": "check_username_validity",
        "description": "Checks if the user's name is valid for our service",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "name": {
                "type": "string",
            },
        },
        "required": ["name"],
    },
    "get_available_soups": {
        "name": "get_available_soups",
        "description": "Checks which soups are currently in stock",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "get_keyleth_stamina": {
        "name": "get_keyleth_stamina",
        "description": "",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "consult_policy": {
        "name": "consult_policy",
        "description": "",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "other_inquiries": {
        "name": "other_inquiries",
        "description": "This tool needs to be run when looking for answers that are not covered by other resources",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "try_unlock_card": {
        "name": "try_unlock_card",
        "description": "This tool unlocks a credit card",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "last_6_digits": {
                "type": "string",
            },
        },
        "required": [],
    },
    "find_answer": {
        "name": "find_answer",
        "description": "Get an answer to a question",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "inquiry": {
                "type": "string",
            },
        },
        "required": ["inquiry"],
    },
    "pay_cc_bill": {
        "name": "pay_cc_bill",
        "description": "Pay credit bard bill. Payment date is given in format DD-MM-YYYY",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "payment_date": {
                "type": "string",
            },
        },
        "required": ["payment_date"],
    },
    "register_for_sweepstake": {
        "name": "register_for_sweepstake",
        "description": "Register for a sweepstake given multiple required details",
        "module_path": "tests.tool_utilities",
        "consequential": True,
        "parameters": {
            "first_name": (
                {
                    "type": "string",
                    "enum": ["Sushi", "Mushi", "Tushi"],
                },
                ToolParameterOptions(precedence=1),
            ),
            "last_name": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=1),
            ),
            "father_name": (
                {
                    "type": "string",
                }
            ),
            "mother_name": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=2),
            ),
            "entry_type": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=3),
            ),
            "n_entries": (
                {
                    "type": "int",
                },
                ToolParameterOptions(precedence=3),
            ),
            "donation_target": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=3),
            ),
            "donation_percent": (
                {
                    "type": "int",
                },
                ToolParameterOptions(precedence=3),
            ),
        },
        "required": [
            "first_name",
            "last_name",
            "father_name",
            "mother_name",
            "entry_type",
            "n_entries",
            "donation_target",
            "donation_percent",
        ],
    },
    "register_for_confusing_sweepstake": {
        "name": "register_for_confusing_sweepstake",
        "description": "Register for a sweepstake with more confusing parameter options",
        "module_path": "tests.tool_utilities",
        "consequential": True,
        "parameters": {
            "first_name": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=11),
            ),
            "last_name": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=11),
            ),
            "father_name": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=-1),
            ),
            "mother_name": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=-1),
            ),
            "entry_type": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=30),
            ),
            "n_entries": (
                {
                    "type": "int",
                },
                ToolParameterOptions(precedence=30),
            ),
            "donation_target": (
                {
                    "type": "string",
                },
                ToolParameterOptions(precedence=-3),
            ),
            "donation_percent": (
                {
                    "type": "int",
                },
                ToolParameterOptions(precedence=-3),
            ),
        },
        "required": [
            "first_name",
            "last_name",
            "father_name",
            "mother_name",
            "entry_type",
            "n_entries",
        ],
    },
    "calculate_salary": {
        "name": "calculate_salary",
        "description": "Calculate the salary of an employee according to other employees",
        "module_path": "tests.tool_utilities",
        "consequential": True,
        "parameters": {
            "name": {
                "type": "string",
                "enum": ["John n Coke", "Mike Andike", "Bruno Twix", "Chris Pikrim"],
            },
            "manager": (
                {
                    "type": "string",
                    "enum": ["Mike Andike", "Bruno Twix", "Jay Libelly"],
                },
                ToolParameterOptions(hidden=True),
            ),
            "director": (
                {
                    "type": "string",
                    "enum": ["Bruno Twix", "Jay Libelly", "John n Coke"],
                },
                ToolParameterOptions(hidden=True),
            ),
            "friend": (
                {
                    "type": "string",
                    "enum": ["Chris Pikrim", "Jay Libelly", "Mike Andike"],
                },
                ToolParameterOptions(display_name="homie"),
            ),
            "mistress": {
                "type": "string",
                "enum": ["Jay Libelly", "Chris Pikrim", "Mike Andike"],
            },
            "cleaner": (
                {
                    "type": "string",
                    "enum": ["Mike Andike", "Bruno Twix", "Chris Pikrim"],
                },
                ToolParameterOptions(display_name="The robot"),
            ),
        },
        "required": ["name", "manager", "director", "cleaner"],
    },
    "calculate_expected_salary": {
        "name": "calculate_expected_salary",
        "description": "Calculate the expected salary of an employee according to their features",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "name": {
                "type": "string",
                "enum": ["John", "Shone", "David"],
            },
            "Residence": (
                {
                    "type": "string",
                    "enum": [" City Center", "Suburban", "Kibbutz"],
                },
                ToolParameterOptions(hidden=True),
            ),
            "Car": (
                {
                    "type": "string",
                    "enum": ["Toyota", "Tesla", "Ford"],
                },
                ToolParameterOptions(hidden=True),
            ),
        },
        "required": ["name", "manager", "director", "cleaner"],
    },
    "get_products_by_type": {
        "name": "get_products_by_type",
        "description": "Get all products that match the specified product type ",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "product_type": {
                "type": "string",
                "enum": ["Monitor", "Keyboard", "Mouse", "Headset", "Audio", "Laptop", "Other"],
            }
        },
        "required": ["product_type"],
    },
    "get_bookings": {
        "name": "get_bookings",
        "description": "Gets all flight bookings for a customer",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "customer_id": {
                "type": "string",
            }
        },
        "required": ["customer_id"],
    },
    "get_qualification_info": {
        "name": "get_qualification_info",
        "description": "Get the qualification information for the customer",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "transfer_coins": {
        "name": "transfer_coins",
        "description": "Transfer coins from one account to another",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "amount": {"type": "integer", "description": "the number of coins to transfer"},
            "from_account": {
                "type": "string",
                "description": "The name of the person whose account the coins will be transferred from",
            },
            "to_account": {
                "type": "string",
                "description": "The name of the person whose account the coins will be transferred to",
            },
            "pincode": {
                "type": "string",
                "description": "the pincode for the account the coins are transferred from",
            },
        },
        "required": ["amount", "from_account", "to_account", "pincode"],
    },
    "search_electronic_products": {
        "name": "search_electronic_products",
        "description": "Search for electronic products in the inventory based on various criteria",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "keyword": {
                "type": "string",
                "description": "Search term to match against product names and descriptions",
            },
            "product_type": {
                "type": "string",
                "description": "Filter by product category",
                "enum": ["Monitor", "Keyboard", "Mouse", "Headset", "Audio", "Laptop", "Other"],
            },
            "min_price": {"type": "integer", "description": "Minimum price filter"},
            "max_price": {"type": "integer", "description": "Maximum price filter"},
            "in_stock_only": {
                "type": "boolean",
                "description": "Only show products that are currently in stock",
            },
            "brand": {
                "type": "string",
                "description": "Brand name",
            },
        },
        "required": ["keyword"],
    },
    "search_products": {
        "name": "search_products",
        "description": "Search for products in the inventory based on various criteria",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "keyword": {
                "type": "string",
                "description": "Search term to match against product names and descriptions",
            },
            "product_type": {
                "type": "string",
                "description": "Filter by product category",
                "enum": [
                    "Electronics",
                    "Clothing",
                    "Home",
                    "Beauty",
                    "Toys",
                    "Sports",
                    "Automotive",
                    "Other",
                ],
            },
            "min_price": {"type": "integer", "description": "Minimum price filter"},
            "max_price": {"type": "integer", "description": "Maximum price filter"},
            "in_stock_only": {
                "type": "boolean",
                "description": "Only show products that are currently in stock",
            },
            "brand": {"type": "string", "description": "Brand or manufacturer name"},
        },
        "required": ["keyword"],
    },
    "book_flight": {
        "name": "book_flight",
        "description": "Books a flight with the provided details",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "departure_city": {
                "type": "string",
                "description": "The name of the city the user flighting from",
            },
            "destination_city": {
                "type": "string",
                "description": "The name of the city the user is flighting to",
            },
        },
        "required": ["departure_city", "destination_city", "departure_date"],
    },
    "send_email": {
        "name": "send_email",
        "description": "Sends an email to the specified recipient.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "to": {
                "type": "string",
                "description": "the name of the person to send the email to",
            },
            "subject": {
                "type": "string",
                "description": "The subject of the mail",
            },
            "body": {
                "type": "string",
                "description": "The body of the mail",
            },
        },
        "required": ["to", "subject"],
    },
    "schedule_meeting": {
        "name": "schedule_meeting",
        "description": "Schedules a meeting with the given participant.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "participant": {
                "type": "string",
                "description": "The participant name to include in the meeting",
            },
            "date": {
                "type": "string",
                "description": "The meeting date given in format DD-MM-YYYY",
            },
            "time": {
                "type": "string",
                "description": "The meeting hour given in 24-hour format HH:MM",
            },
            "agenda": {
                "type": "string",
                "description": "The meeting agenda",
            },
        },
        "required": ["participant", "date", "time"],
    },
    "schedule_appointment": {
        "name": "schedule_appointment",
        "description": "Schedules a new appointment for a patient with a specific doctor.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "patient": {
                "type": "string",
                "description": "The name of the patient for whom the appointment is being scheduled. Will be the user name if specified and the appointment is for the user.",
            },
            "doctor_name": {
                "type": "string",
                "description": "The name of the doctor the appointment is with.",
            },
            "date": {
                "type": "string",
                "description": "The appointment date in format DD-MM-YYYY.",
            },
            "time": {
                "type": "string",
                "description": "The appointment time in 24-hour format HH:MM.",
            },
            "reason": {
                "type": "string",
                "description": "The reason for the appointment (optional).",
            },
        },
        "required": ["patient", "doctor_name", "date", "time"],
    },
    "reschedule_appointment": {
        "name": "reschedule_appointment",
        "description": "Reschedules an existing appointment for a patient with a specific doctor.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "patient": {
                "type": "string",
                "description": "The name of the patient whose appointment is being rescheduled. Will be the user name if specified and the appointment is for the user.",
            },
            "doctor_name": {
                "type": "string",
                "description": "The name of the doctor the appointment is with.",
            },
            "new_date": {
                "type": "string",
                "description": "The new date for the appointment in format DD-MM-YYYY.",
            },
            "new_time": {
                "type": "string",
                "description": "The new time for the appointment in 24-hour format HH:MM.",
            },
        },
        "required": ["patient", "doctor_name", "new_date", "new_time"],
    },
    "transfer_shekels": {
        "name": "transfer_shekels",
        "description": "Transfers a specified amount in shekels from one account to another.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "amount": {"type": "integer", "description": "The amount of shekels to transfer"},
            "from_account": {
                "type": "string",
                "description": "The name of the account sending the shekels",
            },
            "to_account": {
                "type": "string",
                "description": "The name of the account receiving the shekels",
            },
        },
        "required": ["amount", "from_account", "to_account"],
    },
    "transfer_dollars": {
        "name": "transfer_dollars",
        "description": "Transfers a specified amount in shekels from one account to another.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "amount": {"type": "integer", "description": "The amount of dollars to transfer"},
            "from_account": {
                "type": "string",
                "description": "The name of the account sending the dollars",
            },
            "to_account": {
                "type": "string",
                "description": "The name of the account receiving the dollars",
            },
        },
        "required": ["amount", "from_account", "to_account"],
    },
    "reset_password": {
        "name": "reset_password",
        "description": "Reset's a password for an account based on its username. Must provide either phone number or email address for verification.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "username": {
                "type": "string",
                "description": "The account's username",
            },
            "phone_number": {
                "type": "string",
                "description": "The account's associated phone number",
            },
            "email": {
                "type": "string",
                "description": "The account's associated email address",
            },
        },
        "required": ["username"],
    },
    "set_a_bbq_appointment": {
        "name": "set_a_bbq_appointment",
        "description": "Set a BBQ appointment",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "start_time": {
                "type": "datetime",
            },
            "description": {
                "type": "string",
            },
            "participants": {"type": "array", "item_type": "string"},
            "participants_rating": {"type": "array", "item_type": "number"},
            "end_time": {
                "type": "datetime",
            },
            "location": {
                "type": "string",
                "enum": ["meeting room", "phone booth", "kitchen"],
            },
            "alternative_locations": {
                "type": "array",
                "item_type": "string",
                "enum": ["meeting room", "phone booth", "kitchen"],
            },
            "meat_to_buy_in_kg": {"type": "number"},
            "vegetarians": {"type": "integer"},
        },
        "required": ["start_time", "description", "participants"],
    },
    "find_bbq_appointments": {
        "name": "find_bbq_appointments",
        "description": "Find a BBQ appointment in calendar",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "day": {
                "type": "date",
            },
            "participants": {"type": "array", "item_type": "string"},
            "location": {
                "type": "string",
                "enum": ["meeting room", "phone booth", "kitchen"],
            },
        },
        "required": [],
    },
    "give_boolean_types": {
        "name": "give_boolean_types",
        "description": "Get the boolean types",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "boolean": {
                "type": "array",
                "item_type": "boolean",
            },
            "optional_boolean": {
                "type": "boolean",
            },
        },
        "required": ["boolean"],
    },
    "check_current_time": {
        "name": "check_current_time",
        "description": "Check the current time",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "class_access_validator": {
        "name": "class_access_validator",
        "description": "Checks if the traveler is eligible for business class (21+), else restricts to economy.",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "check_current_time_emit": {
        "name": "check_current_time_emit",
        "description": "Check the current time",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "class_eligibility_checker": {
        "name": "class_eligibility_checker",
        "description": "Checks if the traveler is eligible for business class (21+), else restricts to economy.",
        "module_path": "tests.tool_utilities",
        "parameters": {"age": {"type": "integer", "description": "The age of the traveler"}},
        "required": ["age"],
    },
    "availability_check": {
        "name": "availability_check",
        "description": "Check if the luxury suite is available for booking",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "check_customer_location": {
        "name": "check_customer_location",
        "description": "Check the customer's location",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "check_eligibility": {
        "name": "check_eligibility",
        "description": "Check the customer's eligibility for a loan",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "account_id": {
                "type": "int",
            },
            "amount": {
                "type": "int",
            },
        },
        "required": ["account_id", "amount"],
    },
    "change_credit_limit": {
        "name": "change_credit_limit",
        "description": "Changes the credit limit for an account. Can increase or decrease by $10,000 without supervisor approval. Larger changes require supervisor authorization.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "username": {
                "type": "string",
                "description": "The account's username",
            },
            "new_limit": {
                "type": "int",
                "description": "The new requested credit limit in USD",
            },
            "current_limit": {
                "type": "int",
                "description": "The current credit limit in USD",
            },
        },
        "required": ["username", "new_limit", "current_limit"],
    },
    "get_credit_limit": {
        "name": "get_credit_limit",
        "description": "Retrieves the current credit limit for a given account.",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "username": {
                "type": "string",
                "description": "The account's username",
            }
        },
        "required": ["username"],
    },
    "list_cards": {
        "name": "list_cards",
        "description": "List all cards associated with the customer's account",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
    "lock_card": {
        "name": "lock_card",
        "description": "Lock a specific card for security reasons",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "card_number": {
                "type": "string",
                "description": "The card number (last 4 digits) to lock",
            },
            "reason": {
                "type": "string",
                "description": "The reason for locking the card",
            },
        },
        "required": ["card_number", "reason"],
    },
    "schedule_appointment_2": {
        "name": "schedule_appointment_2",
        "description": "Schedule an appointment",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "date": {
                "type": "datetime",
                "description": "The date of the appointment",
            },
        },
        "required": ["date"],
    },
    "check_lab_results": {
        "name": "check_lab_results",
        "description": "Check the lab results for a patient",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "name": {
                "type": "string",
                "description": "The name of the patient",
            }
        },
        "required": ["name"],
    },
}


@step(given, parsers.parse('an association between "{guideline_name}" and "{tool_name}"'))
def given_a_guideline_tool_association(
    context: ContextOfTest,
    tool_name: str,
    guideline_name: str,
) -> GuidelineToolAssociation:
    guideline_tool_association_store = context.container[GuidelineToolAssociationStore]

    return context.sync_await(
        guideline_tool_association_store.create_association(
            guideline_id=context.guidelines[guideline_name].id,
            tool_id=ToolId("local", tool_name),
        )
    )


@step(
    given,
    parsers.parse(
        'an association between "{guideline_name}" and "{tool_name}" from "{service_name}"'
    ),
)
def given_a_guideline_association_with_tool_from_a_service(
    context: ContextOfTest,
    service_name: str,
    tool_name: str,
    guideline_name: str,
) -> GuidelineToolAssociation:
    guideline_tool_association_store = context.container[GuidelineToolAssociationStore]

    return context.sync_await(
        guideline_tool_association_store.create_association(
            guideline_id=context.guidelines[guideline_name].id,
            tool_id=ToolId(service_name, tool_name),
        )
    )


@step(given, parsers.parse('the tool "{tool_name}" from "{service_name}"'))
def given_the_tool_from_service(
    context: ContextOfTest,
    tool_name: str,
    service_name: str,
) -> None:
    service_registry = context.container[ServiceRegistry]

    local_tool_service = cast(
        LocalToolService,
        context.sync_await(
            service_registry.update_tool_service(name=service_name, kind="local", url="")
        ),
    )

    service_tools: dict[str, dict[str, Any]] = {
        "first_service": {
            "schedule": {
                "name": "schedule",
                "description": "This tool is used to book a meeting with Larry David as host",
                "module_path": "tests.tool_utilities",
                "consequential": True,
                "parameters": {},
                "required": [],
            }
        },
        "second_service": {
            "schedule": {
                "name": "schedule",
                "description": "This tool is used to book a meeting with Larry David as guest",
                "module_path": "tests.tool_utilities",
                "consequential": True,
                "parameters": {},
                "required": [],
            }
        },
        "ksp": {
            "available_products_by_category": {
                "name": "available_products_by_category",
                "description": "",
                "module_path": "tests.tool_utilities",
                "parameters": {
                    "category": {
                        "type": "string",
                        "enum": ["laptops", "peripherals"],
                    },
                },
                "required": ["category"],
            },
            "available_products_by_categories": {
                "name": "available_products_by_categories",
                "description": "",
                "module_path": "tests.tool_utilities",
                "parameters": {
                    "categories": {
                        "type": "array",
                        "item_type": "string",
                        "enum": ["laptops", "peripherals"],
                    },
                },
                "required": ["categories"],
            },
        },
    }

    tool = context.sync_await(
        local_tool_service.create_tool(**service_tools[service_name][tool_name])
    )

    context.tools[tool_name] = tool


@step(given, parsers.parse('the tool "{tool_name}"'))
def given_a_tool(
    context: ContextOfTest,
    tool_name: str,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool = context.sync_await(local_tool_service.create_tool(**TOOLS[tool_name]))

    context.tools[tool_name] = tool


@step(given, parsers.parse("an agent with a maximum of {max_engine_iterations} engine iterations"))
def given_max_engine_iteration(
    context: ContextOfTest,
    agent_id: AgentId,
    max_engine_iterations: str,
) -> None:
    agent_store = context.container[AgentStore]

    context.sync_await(
        agent_store.update_agent(
            agent_id=agent_id,
            params={"max_engine_iterations": int(max_engine_iterations)},
        )
    )


@step(
    given,
    parsers.parse(
        'a cross-service tool relationship whereby "{tool_a}" from "{service_a}" overlaps with "{tool_b}" from "{service_b}"'
    ),
)
def given_an_overlapping_tools_relationship_from_service(
    context: ContextOfTest,
    tool_a: str,
    tool_b: str,
    service_a: str,
    service_b: str,
) -> None:
    store = context.container[RelationshipStore]
    tool_a_id = ToolId(service_name=service_a, tool_name=tool_a)
    tool_b_id = ToolId(service_name=service_b, tool_name=tool_b)

    context.sync_await(
        store.create_relationship(
            source=RelationshipEntity(
                id=tool_a_id,
                kind=RelationshipEntityKind.TOOL,
            ),
            target=RelationshipEntity(
                id=tool_b_id,
                kind=RelationshipEntityKind.TOOL,
            ),
            kind=RelationshipKind.OVERLAP,
        )
    )


@step(
    given,
    parsers.parse('a tool relationship whereby "{tool_a}" overlaps with "{tool_b}"'),
)
def given_an_overlapping_tools_relationship(
    context: ContextOfTest,
    tool_a: str,
    tool_b: str,
) -> None:
    store = context.container[RelationshipStore]
    tool_a_id = ToolId(service_name="local", tool_name=tool_a)
    tool_b_id = ToolId(service_name="local", tool_name=tool_b)

    context.sync_await(
        store.create_relationship(
            source=RelationshipEntity(
                id=tool_a_id,
                kind=RelationshipEntityKind.TOOL,
            ),
            target=RelationshipEntity(
                id=tool_b_id,
                kind=RelationshipEntityKind.TOOL,
            ),
            kind=RelationshipKind.OVERLAP,
        )
    )
