from dataclasses import Field, dataclass  # noqa
from datetime import datetime, timezone
from itertools import chain  # noqa
from typing import Any, Optional, Sequence, cast  # noqa
from pytest import mark

from lagom import Container  # noqa
from parlant.core.agents import Agent, AgentId, AgentStore  # noqa
from parlant.core.common import DefaultBaseModel, generate_id, JSONSerializable  # noqa
from parlant.core.context_variables import (
    ContextVariable,  # noqa
    ContextVariableId,  # noqa
    ContextVariableValue,
    ContextVariableValueId,  # noqa
)
from parlant.core.customers import Customer, CustomerStore  # noqa
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.engine import AlphaEngine  # noqa
from parlant.core.glossary import Term
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.nlp.generation import SchematicGenerator  # noqa
from parlant.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,  # noqa
    GuidelinePropositionsSchema,  # noqa
)
from parlant.core.engines.alpha.guideline_proposition import (
    GuidelineProposition,  # noqa
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId, GuidelineStore  # noqa
from parlant.core.sessions import EventSource, SessionStore  # noqa
from parlant.core.loggers import Logger  # noqa
from parlant.core.glossary import TermId  # noqa

from parlant.core.tools import LocalToolService, Tool, ToolId, ToolParameterOptions
from tests.core.common.utils import ContextOfTest, create_event_message  # noqa
from tests.test_utilities import SyncAwaiter  # noqa

# TODO remove noqas

tools: dict[str, dict[str, Any]] = {
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
    "get_products_by_ingredient": {
        "name": "get_products_by_ingredient",
        "description": "Get all pizza types that contain a specific ingredient",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "ingredient": {
                "type": "string",
                "enum": [
                    "tomato sauce",
                    "mozzarella cheese",
                    "fresh basil",
                    "olive oil",
                    "pepperoni slices",
                    "bbq sauce",
                    "grilled chicken",
                    "red onions",
                    "cilantro",
                    "bell peppers",
                    "mushrooms",
                    "black olives",
                    "spinach",
                    "ham",
                    "pineapple",
                    "pepperoni",
                    "sausage",
                    "bacon",
                    "ground beef",
                    "buffalo sauce",
                    "chicken",
                    "blue cheese crumbles",
                    "parmesan cheese",
                    "gorgonzola cheese",
                    "ricotta cheese",
                    "feta cheese",
                    "sun-dried tomatoes",
                    "kalamata olives",
                    "artichoke hearts",
                    "onions",
                ],
            }
        },
        "required": ["ingredient"],
    },
    "get_availability_by_type_and_amount": {
        "name": "get_availability_by_type_and_amount",
        "description": "Check if a specific pizza type is available in the requested amount",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "product_type": {
                "type": "string",
                "enum": [
                    "Margherita",
                    "Pepperoni",
                    "BBQ Chicken",
                    "Veggie Supreme",
                    "Hawaiian",
                    "Meat Lovers",
                    "Buffalo Chicken",
                    "Four Cheese",
                    "Mediterranean",
                    "Supreme",
                ],
            },
            "amount": {
                "type": "number",
                "description": "The amount of pizzas requested",
            },
        },
        "required": ["product_type", "amount"],
    },
    "get_availability_by_type": {
        "name": "get_availability_by_type",
        "description": "Check if a specific pizza type is available",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "product_type": {
                "type": "string",
                "enum": [
                    "Margherita",
                    "Pepperoni",
                    "BBQ Chicken",
                    "Veggie Supreme",
                    "Hawaiian",
                    "Meat Lovers",
                    "Buffalo Chicken",
                    "Four Cheese",
                    "Mediterranean",
                    "Supreme",
                ],
            }
        },
        "required": ["product_type"],
    },
    "process_order": {
        "name": "process_order",
        "description": "Process an order for a specific pizza type and amount",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "product_type": {
                "type": "string",
                "enum": [
                    "Margherita",
                    "Pepperoni",
                    "BBQ Chicken",
                    "Veggie Supreme",
                    "Hawaiian",
                    "Meat Lovers",
                    "Buffalo Chicken",
                    "Four Cheese",
                    "Mediterranean",
                    "Supreme",
                ],
            },
            "amount": {
                "type": "number",
                "description": "The amount of pizzas to order",
            },
        },
        "required": ["product_type", "amount"],
    },
    "get_menu": {
        "name": "get_menu",
        "description": "Get the current pizza menu with available items",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    },
}

created_tools: dict[str, Tool] = {}


@dataclass(frozen=True)
class _ToolCallVerification:
    expected_tool_id: str
    expected_tool_arguments: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class ScenarioMessage:
    customer_message: str
    agent_message: str
    expected_agent_message_content: str
    expected_tool_results: Optional[Sequence[_ToolCallVerification]] = None
    expected_active_guidelines: Optional[set[str]] = None


@dataclass(frozen=True)
class _GuidelineAndTool:
    guideline: GuidelineContent
    associated_tools: Sequence[Tool] = None


@dataclass(frozen=True)
class InteractionScenario:
    messages: Sequence[ScenarioMessage]
    agent_description: Optional[str] = ""
    guidelines_and_tools: Optional[Sequence[_GuidelineAndTool]] = None
    context_variables: Optional[Sequence[ContextVariableValue]] = None
    glossary: Optional[Sequence[Term]] = None


def get_tool(
    tool_name: str,
) -> Tool:
    if tool_name not in created_tools:
        creation_utc = datetime.now(timezone.utc)
        tool_data = tools[tool_name]
        created_tools[tool_name] = Tool(
            creation_utc=creation_utc,
            name=tool_data["name"],
            description=tool_data["description"],
            parameters={
                name: (descriptor, ToolParameterOptions())
                for name, descriptor in tool_data["parameters"].items()
            },
            required=tool_data["required"],
            consequential=False,
        )
    return created_tools[tool_name]


BANKING_SCENARIO = InteractionScenario(
    messages=[
        ScenarioMessage(
            customer_message="I want to transfer 20$ to Alan Johnson",
            agent_message="It seems the PIN code you provided is incorrect, so the transfer could not be completed. Could you please double-check your PIN code?",
            expected_agent_message_content="",
            expected_tool_results=[
                # ToolCallVerification(
                #    expected_tool_id="open_account",
                #    expected_tool_arguments={"account_type": "savings"},
                # )
            ],
            expected_active_guidelines=set(),
        ),
    ],
    guidelines_and_tools=[
        _GuidelineAndTool(
            guideline=GuidelineContent(
                condition="customer needs help unlocking their card",
                action="ask for the card's last 6 digits, try to unlock the card and report the result to the customer",
            ),
            associated_tools=[get_tool("try_unlock_card")],
        ),
        _GuidelineAndTool(
            guideline=GuidelineContent(
                condition="customer wants to transfer money and hasn't specified to whom or how much",
                action="ask them to whom and how much",
            ),
            associated_tools=[],
        ),
        _GuidelineAndTool(
            guideline=GuidelineContent(
                condition="customer wants to transfer money and has specified both how much and to whom",
                action="double-check the recipients' name and account number and confirm with the user",
            ),
            associated_tools=[],
        ),
        _GuidelineAndTool(
            guideline=GuidelineContent(
                condition="user wants to transfer money and has specified both how much and to whom and has confirmed it once and not yet specified or successfully confirmed their PIN Code",
                action="ask for their PIN Code and confirm it",
            ),
            associated_tools=[],  # TODO get_tool("pin_code_verification")
        ),
        _GuidelineAndTool(
            guideline=GuidelineContent(
                condition="user wants to transfer money and has successfully confirmed their PIN code",
                action="transfer money to the recipient and confirm the transaction providing its ID",
            ),
            associated_tools=[],  # TODO get_tool("transfer_money")
        ),
        _GuidelineAndTool(
            guideline=GuidelineContent(
                condition="user wants to know their account balance",
                action="find it and provide it to them",
            ),
            associated_tools=[get_tool("get_account_balance")],
        ),
        _GuidelineAndTool(
            guideline=GuidelineContent(
                condition="user wants to know their transactions",
                action="find the transactions and show it to them",
            ),
            associated_tools=[],  # TODO get_tool("show_transactions")
        ),
    ],
)
model_config = {"arbitrary_types_allowed": True}


def verify_tool_calls(
    emitted_tool_calls: Sequence[EmittedEvent],
    expected_tool_calls: Sequence[_ToolCallVerification],
) -> bool:
    pass


def verify_guidelines(
    active_guidelines: Sequence[GuidelineContent], expected_active_guideline_names: Sequence[str]
) -> None:
    pass


def verify_message_generator(agent_message: str, expected_content: str) -> None:  # NLP test
    pass


def build_test_context(
    context: ContextOfTest, scenario: InteractionScenario, message_n: int
) -> None:
    # Create session
    # engine = context.container[AlphaEngine]
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name="test-agent",
            description=scenario.agent_description,
            max_engine_iterations=2,
        )
    )
    # session_store = context.container[SessionStore]
    # customer_store = context.container[CustomerStore]
    guideline_tool_association_store = context.container[GuidelineToolAssociationStore]

    # utc_now = datetime.now(timezone.utc)

    # customer = context.sync_await(customer_store.create_customer("test_customer"))
    # session = context.sync_await(
    #     session_store.create_session(
    #         creation_utc=utc_now,
    #         customer_id=customer.id,
    #         agent_id=agent.id,
    #     )
    # )

    # Register Guidelines & tools
    guideline_store = context.container[GuidelineStore]
    local_tool_service = context.container[LocalToolService]

    for i, guideline_and_tools in enumerate(scenario.guidelines_and_tools):
        context.guidelines[str(i)] = context.sync_await(
            guideline_store.create_guideline(
                guideline_set=agent.id,
                condition=guideline_and_tools.guideline.condition,
                action=guideline_and_tools.guideline.action,
            )
        )
        for tool_description in guideline_and_tools.associated_tools:
            if tool_description.name not in context.tools:
                tool = context.sync_await(
                    local_tool_service.create_tool(
                        name=tool_description.name,
                        module_path="tests.tool_utilities",
                        description=tool_description.description,
                        parameters=tool_description.parameters,
                        required=tool_description.required,
                    )
                )
                context.tools[tool_description.name] = tool
            context.sync_await(
                guideline_tool_association_store.create_association(
                    guideline_id=context.guidelines[str(i)].id,
                    tool_id=ToolId("local", tool_description.name),
                )
            )

    pass


@mark.parametrize("n", range(10))
def test_banking_scenario(context: ContextOfTest, n: int) -> None:
    build_test_context(context, BANKING_SCENARIO, n)
    # run engine
    # analyze guidelines
    # analyze tool calls
    # analyze message generator
    assert True
