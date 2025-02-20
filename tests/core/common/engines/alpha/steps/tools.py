# Copyright 2024 Emcie Co Ltd.
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

from parlant.core.tools import Tool
from parlant.core.agents import AgentId, AgentStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationStore,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import LocalToolService, ToolId

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


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

    async def create_tool(
        name: str,
        module_path: str,
        description: str,
        parameters: dict[str, Any],
        required: list[str],
    ) -> Tool:
        return await local_tool_service.create_tool(
            name=name,
            module_path=module_path,
            description=description,
            parameters=parameters,
            required=required,
        )

    service_tools: dict[str, dict[str, Any]] = {
        "first_service": {
            "schedule": {
                "name": "schedule",
                "description": "",
                "module_path": "tests.tool_utilities",
                "parameters": {},
                "required": [],
            }
        },
        "second_service": {
            "schedule": {
                "name": "schedule",
                "description": "",
                "module_path": "tests.tool_utilities",
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
            }
        },
    }

    tool = context.sync_await(create_tool(**service_tools[service_name][tool_name]))

    context.tools[tool_name] = tool


@step(given, parsers.parse('the tool "{tool_name}"'))
def given_a_tool(
    context: ContextOfTest,
    tool_name: str,
) -> None:
    local_tool_service = context.container[LocalToolService]

    async def create_tool(
        name: str,
        module_path: str,
        description: str,
        parameters: dict[str, Any],
        required: list[str],
    ) -> Tool:
        return await local_tool_service.create_tool(
            name=name,
            module_path=module_path,
            description=description,
            parameters=parameters,
            required=required,
        )

    tools: dict[str, dict[str, Any]] = {
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
            "parameters": {},
            "required": [],
        },
        "expert_answer": {
            "name": "expert_answer",
            "description": "Get answers to questions by consulting documentation",
            "module_path": "tests.tool_utilities",
            "parameters": {
                "user_query": {"type": "string", "description": "The query from the customer"}
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
                "first_number": {"type": "number", "description": "The first number"},
                "second_number": {"type": "number", "description": "The second number"},
            },
            "required": ["first_number", "second_number"],
        },
        "multiply": {
            "name": "multiply",
            "description": "Getting the multiplication calculation between two numbers",
            "module_path": "tests.tool_utilities",
            "parameters": {
                "first_number": {"type": "number", "description": "The first number"},
                "second_number": {"type": "number", "description": "The second number"},
            },
            "required": ["first_number", "second_number"],
        },
        "get_account_balance": {
            "name": "get_account_balance",
            "description": "Get the account balance by given name",
            "module_path": "tests.tool_utilities",
            "parameters": {
                "account_name": {"type": "string", "description": "The name of the account"}
            },
            "required": ["account_name"],
        },
        "get_account_loans": {
            "name": "get_account_loans",
            "description": "Get the account loans by given name",
            "module_path": "tests.tool_utilities",
            "parameters": {
                "account_name": {"type": "string", "description": "The name of the account"}
            },
            "required": ["account_name"],
        },
        "transfer_money": {
            "name": "transfer_money",
            "description": "Transfer money from one account to another",
            "module_path": "tests.tool_utilities",
            "parameters": {
                "from_account": {
                    "type": "string",
                    "description": "The account from which money will be transferred",
                },
                "to_account": {
                    "type": "string",
                    "description": "The account to which money will be transferred",
                },
            },
            "required": ["from_account", "to_account"],
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
    }

    tool = context.sync_await(create_tool(**tools[tool_name]))

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
