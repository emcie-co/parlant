from dataclasses import Field, dataclass  # noqa
from datetime import datetime, timezone
from itertools import chain  # noqa
import json
from typing import Any, Optional, Sequence, cast  # noqa
from pytest import Session, mark

from lagom import Container  # noqa
from parlant.core.agents import Agent, AgentId, AgentStore  # noqa
from parlant.core.common import DefaultBaseModel, generate_id, JSONSerializable  # noqa
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import Customer, CustomerId, CustomerStore  # noqa
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.engine import AlphaEngine  # noqa
from parlant.core.engines.alpha.message_assembler import MessageAssembler
from parlant.core.engines.alpha.message_event_composer import MessageEventComposer
from parlant.core.engines.alpha.tool_caller import ToolInsights
from parlant.core.engines.types import Context
from parlant.core.glossary import GlossaryStore, Term
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
from parlant.core.sessions import EventSource, MessageEventData, SessionId, SessionStore  # noqa
from parlant.core.loggers import Logger  # noqa
from parlant.core.glossary import TermId  # noqa

from parlant.core.tools import LocalToolService, Tool, ToolId, ToolParameterOptions
from tests.core.common.utils import ContextOfTest, create_event_message  # noqa
from tests.test_utilities import nlp_test, SyncAwaiter  # noqa

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
    "transfer_coins": {
        "name": "transfer_coins",
        "description": "Transfer coins from one account to another",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "amount": {"type": "int", "description": "the number of coins to transfer"},
            "from_account": {
                "type": "string",
                "description": "The account from which coins will be transferred",
            },
            "to_account": {
                "type": "string",
                "description": "The account to which money will be transferred",
            },
            "pincode": {
                "type": "string",
                "description": "the pincode for the account the coins are transfered from",
            },
        },
        "required": ["amount", "from_account", "to_account", "pincode"],
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


class _ResponsePreparationState:  # TODO duplicate, should I keep it here?
    """Helper class to access and update the state needed for responding properly"""

    context_variables: list[tuple[ContextVariable, ContextVariableValue]]
    glossary_terms: set[Term]
    ordinary_guideline_propositions: list[GuidelineProposition]
    tool_enabled_guideline_propositions: dict[GuidelineProposition, list[ToolId]]
    tool_events: list[EmittedEvent]
    tool_insights: ToolInsights
    iterations_completed: int
    prepared_to_respond: bool
    message_events: list[EmittedEvent]


@dataclass(frozen=True)
class ScenarioTurn:
    customer_message: str
    agent_message: str
    expected_agent_message_content: str
    expected_tool_results: Optional[Sequence[_ToolCallVerification]] = None
    expected_active_guidelines: Optional[set[str]] = None


@dataclass(frozen=True)
class _GuidelineAndTool:
    name: str
    guideline: GuidelineContent
    associated_tools: Sequence[Tool] = None


@dataclass(frozen=True)
class _ContextVariableData:
    name: str
    data: JSONSerializable
    description: Optional[str] = ""


@dataclass(frozen=True)
class _TermData:
    name: str
    description: str
    synonyms: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class InteractionScenario:
    messages: Sequence[ScenarioTurn]
    agent_description: Optional[str] = ""
    guidelines_and_tools: Optional[Sequence[_GuidelineAndTool]] = None
    context_variables: Optional[Sequence[_ContextVariableData]] = None
    glossary: Optional[Sequence[_TermData]] = None


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
        ScenarioTurn(
            customer_message="Hello! I want to coinmove to Alan Johnson",
            agent_message="Welcome to Parlant Bank! How many coins would you like to transfer?",
            expected_agent_message_content="Asking how many coins the agent would like to transfer",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_recipient_amount"),
        ),
        ScenarioTurn(
            customer_message="500 coins",
            agent_message="Got it. To confirm the transaction, please provide your PIN code.",
            expected_agent_message_content="asking for the customer's pin code",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_pin_code"),
        ),
        ScenarioTurn(
            customer_message="I don't have it. How can I get it?",
            agent_message="Your pincode was emailed to your address at jez@parlant.io",
            expected_agent_message_content="that the customer's pincode was sent to jez@parlant.io",
            expected_tool_results=[],
            expected_active_guidelines=set("email_pincode"),
        ),
        ScenarioTurn(
            customer_message="Got it! it's 5432",
            agent_message="Thank you! Before proceeding, please let me know if the transaction's details are correct: You wish to transfer 500 coins to Alan Johnson, and your pincode is 5432",
            expected_agent_message_content="asking the customer for confirmation about the following transaction: transferring 500 coins to Alan Johnson, with pin / pincode 5432",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_reiterate"),
        ),
        ScenarioTurn(
            customer_message="that's right",
            agent_message="The transaction was rejected due to an invalid pincode. Can you please double check the provided pin?",
            expected_agent_message_content="informing the customer that the transaction was rejected due to an invalid pincode",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_execute"),
        ),
        ScenarioTurn(
            customer_message="Oh sorry! it's actually 6543",
            agent_message="The transaction was rejected because your account is blocked. Would you like me to unlock your account?",
            expected_agent_message_content="informing the customer that the transaction was rejected because the account is blocked",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_execute"),
        ),
        ScenarioTurn(
            customer_message="Got it! it's 5432",
            agent_message="Thank you! Before proceeding, please let me know if the transaction's details are correct: You wish to transfer 500 coins to Alan Johnson, and your pincode is 6543",
            expected_agent_message_content="asking the customer for confirmation about the following transaction: transferring 500 coins to Alan Johnson, with pin / pincode 6543",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_reiterate"),
        ),
        ScenarioTurn(
            customer_message="Can you reroute it to a different branch instead",
            agent_message="I apologize, but I cannot assist you with this request through this chat. Is there anything else I could do for you instead?",
            expected_agent_message_content="informing the customer that the agent can't help it rerouting to a different branch",
            expected_tool_results=[],
            expected_active_guidelines=set(),
        ),
        ScenarioTurn(
            customer_message="Then yes, just unblock my account...",
            agent_message="Your account has been succesfully unblocked",
            expected_agent_message_content="informing the customer that the transaction was rejected because the account is blocked",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_execute"),
        ),
        ScenarioTurn(
            customer_message="great, can I transfer the money to Alan now?",
            agent_message="500 coins were succesfully transferred to Alan Johnson. The transaction number is 9911827",
            expected_agent_message_content="informing the customer that the transaction succesful, and its number is 9911827",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_execute"),
        ),
        ScenarioTurn(
            customer_message="nice. Now let's give Sophie Chapman 100 coins as well",
            agent_message="Before proceeding, please let me know if the transaction's details are correct: You wish to transfer 100 coins to Sophie Chapman, and your pincode is 6543",
            expected_agent_message_content="asking for confirmation that the customer wishes to transfer 100 coins to Sophie Chapman, using pincode is 6543",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_reiterate"),
        ),
        ScenarioTurn(
            customer_message="make that only 50 actually, she's been rude about my paintings lately",
            agent_message="Before proceeding, please let me know if the transaction's details are correct: You wish to transfer 50 coins to Sophie Chapman, and your pincode is 6543",
            expected_agent_message_content="asking for confirmation that the customer wishes to transfer 50 coins to Sophie Chapman, using pincode is 6543",
            expected_tool_results=[],
            expected_active_guidelines=set("transfer_funds_execute"),
        ),
    ],
    agent_description="You are an AI customer assistant for Parlant Bank. Your role is to assist clients with their banking related needs.",
    guidelines_and_tools=[
        _GuidelineAndTool(
            name="transfer_funds_recipient_amount",
            guideline=GuidelineContent(
                condition="The customer wants to transfer funds and hasn’t specified either to whom or how much",
                action="Get the recipient’s name and transfer amount from the customer",
            ),
            associated_tools=[],
        ),
        _GuidelineAndTool(
            name="transfer_funds_pin_code",
            guideline=GuidelineContent(
                condition="The customer wants to transfer funds, has specified to whom and how much but have yet to provide a confirmed pincode",
                action="Ask for the customer’s pincode",
            ),
            associated_tools=[],
        ),
        _GuidelineAndTool(
            name="transfer_funds_reiterate",
            guideline=GuidelineContent(
                condition="The customer wants to transfer funds, has specified to whom, how much to transfer and has provided a pin code",
                action="Reitirate the details of the transfer to the customer and ask for confirmation",
            ),
            associated_tools=[],
        ),
        _GuidelineAndTool(
            name="transfer_funds_execute",
            guideline=GuidelineContent(
                condition="The customer wants to transfer funds, and has confirmed the transfer details which the agent re-iterated, including the recipient, amount and the pincode",
                action="Try to execute the transfer and report the results to the customer",
            ),
            associated_tools=[],  # TODO get_tool("transfer_funds")
        ),
        _GuidelineAndTool(
            name="unlock_account",
            guideline=GuidelineContent(
                condition="User asks to unlock their account",
                action="Use the customer’s account number to unlock the account",
            ),
            associated_tools=[],  # TODO get_tool("unlock_account")
        ),
        _GuidelineAndTool(
            name="email_pincode",
            guideline=GuidelineContent(
                condition="The customer asks for their pin code",
                action="Email the pincode to the customer and inform them that it has been emailed to their address",
            ),
            associated_tools=[],  # TODO get_tool("email_pinocde")
        ),
    ],
    glossary=[
        _TermData(
            name="coinmove", description="The act of transferring funds from one account to another"
        ),
        _TermData(
            name="coin++",
            description="The act of adding funds to an account from an external source",
            synonyms=["coinadd", "coinincrease"],
        ),
    ],
    context_variables=[
        _ContextVariableData(name="Account Number", data="819663"),
        _ContextVariableData(name="Customer Email", data="jez@parlant.co"),
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


def register_guidelines_and_tools(
    context: ContextOfTest, agent_id: AgentId, guidelines_and_tools: Sequence[_GuidelineAndTool]
) -> None:
    guideline_store = context.container[GuidelineStore]
    local_tool_service = context.container[LocalToolService]
    guideline_tool_association_store = context.container[GuidelineToolAssociationStore]
    for i, guideline_and_tools in enumerate(guidelines_and_tools):
        context.guidelines[str(i)] = context.sync_await(
            guideline_store.create_guideline(
                guideline_set=agent_id,
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


def register_context_variables(
    context: ContextOfTest,
    context_variables: Sequence[_ContextVariableData],
    agent_id: AgentId,
    customer_id: CustomerId,
) -> ContextVariableStore:
    context_variable_store = context.container[ContextVariableStore]
    for context_variable in context_variables:
        variable = context.sync_await(
            context_variable_store.create_variable(
                variable_set=agent_id,
                name=context_variable.name,
                description=context_variable.description,
                tool_id=None,
                freshness_rules=None,
            )
        )
        context.sync_await(
            context_variable_store.update_value(
                variable_set=agent_id,
                key=customer_id,
                variable_id=variable.id,
                data=context_variable.data,
            )
        )
    return context.sync_await(
        context_variable_store.list_variables(
            variable_set=agent_id,
        )
    )


def register_events(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session: SessionStore,
    scenario_messages: Sequence[ScenarioTurn],
) -> None:
    session_store = context.container[SessionStore]
    for i, turn in enumerate(scenario_messages):
        customer_message_data: MessageEventData = {
            "message": turn.customer_message,
            "participant": {
                "id": customer.id,
                "display_name": customer.name,
            },
        }

        customer_message_event = context.sync_await(
            session_store.create_event(
                session_id=session.id,
                source="customer",
                kind="message",
                correlation_id="test_correlation_id",
                data=cast(JSONSerializable, customer_message_data),
            )
        )
        context.events.append(customer_message_event)

        if i < len(scenario_messages) - 1:  # also add agent message and tool results
            agent_message_data: MessageEventData = {
                "message": turn.agent_message,
                "participant": {
                    "id": agent.id,
                    "display_name": agent.name,
                },
            }
            agent_message_event = context.sync_await(
                session_store.create_event(
                    session_id=session.id,
                    source="ai_agent",
                    kind="message",
                    correlation_id="test_correlation_id",
                    data=cast(JSONSerializable, agent_message_data),
                )
            )
            context.events.append(agent_message_event)

            context.sync_await(
                session_store.create_event(
                    session_id=session.id,
                    source="ai_agent",
                    kind="tool",
                    correlation_id="test_correlation_id",
                    data=json.loads(turn.tool_results),
                )
            )


def build_test_context(
    context: ContextOfTest, scenario: InteractionScenario, message_n: int
) -> tuple[SessionId, AgentId]:
    # Get relevant stores
    session_store = context.container[SessionStore]
    customer_store = context.container[CustomerStore]
    glossary_store = context.container[GlossaryStore]
    session_store = context.container[SessionStore]

    # Create agent, customer & session
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name="test-agent",
            description=scenario.agent_description,
            max_engine_iterations=2,
            composition_mode="strict_assembly",
        )
    )
    utc_now = datetime.now(timezone.utc)
    customer = context.sync_await(customer_store.create_customer("test_customer"))
    session = context.sync_await(
        session_store.create_session(
            creation_utc=utc_now,
            customer_id=customer.id,
            agent_id=agent.id,
        )
    )

    # Create guidelines, tools and their associations
    if scenario.guidelines_and_tools:
        register_guidelines_and_tools(context, agent.id, scenario.guidelines_and_tools)

    # Create glossary
    if scenario.glossary:
        for term in scenario.glossary:
            context.sync_await(
                glossary_store.create_term(
                    term_set=agent.id,
                    name=term.name,
                    description=term.description,
                    synonyms=term.synonyms,
                )
            )
        context.terms = context.sync_await(glossary_store.list_terms(term_set=agent.id))

    # Create context variables
    if scenario.context_variables:
        context.context_variables = register_context_variables(
            context,
            scenario.context_variables,
            agent.id,
            customer.id,
        )

    # Create message events
    register_events(context, agent, customer, session, scenario.messages[:message_n])
    return session.id, agent.id


def assmble_message(
    context: ContextOfTest,
    agent: Agent,
    session: Session,
):
    customer = context.sync_await(
        context.container[CustomerStore].read_customer(session.customer_id)
    )

    event_buffer = EventBuffer(
        context.sync_await(
            context.container[AgentStore].read_agent(agent.id),
        )
    )

    message_event_composer: MessageEventComposer = context.container[MessageAssembler]
    result = context.sync_await(
        message_event_composer.generate_events(
            event_emitter=event_buffer,
            agent=agent,
            customer=customer,
            context_variables=[],
            interaction_history=context.events,
            terms=context.glossary,
            ordinary_guideline_propositions=list(context.guideline_propositions.values()),
            tool_enabled_guideline_propositions={},
            tool_insights=ToolInsights(),
            staged_events=[],
        )
    )

    assert len(result) > 0
    assert all(e is not None for e in result[0].events)

    return list(cast(list[EmittedEvent], result[0].events))


@mark.parametrize("n", range(1, 11))
def test_engine_banking_scenario(context: ContextOfTest, n: int) -> None:
    session_id, agent_id = build_test_context(context, BANKING_SCENARIO, n)
    engine = context.container[AlphaEngine]

    # run engine
    buffer = EventBuffer(
        context.sync_await(
            context.container[AgentStore].read_agent(agent_id),
        )
    )
    context.sync_await(
        engine.process(
            Context(
                session_id=session_id,
                agent_id=agent_id,
            ),
            buffer,
        )
    )
    emitted_events = buffer.events

    message_event = next(e for e in emitted_events if e.kind == "message")
    message = cast(MessageEventData, message_event.data)["message"]

    assert context.sync_await(
        nlp_test(
            context=f"Here's a message from an AI agent to a customer, in the context of a conversation: {message}",
            condition=f"The message contains {message}",
        )
    ), f"message: '{message}', expected to contain: '{BANKING_SCENARIO.messages[n].expected_agent_message_content}'"
    # analyze guidelines
    # analyze tool calls
    # analyze message generator
    assert True


@mark.parametrize("n", range(1, 11))
def test_tool_caller_banking_scenario(context: ContextOfTest, n: int) -> None:
    session_id, agent_id = build_test_context(context, BANKING_SCENARIO, n)

    assert True


@mark.parametrize("n", range(1, 11))
def test_assembler_banking_scenario(context: ContextOfTest, n: int) -> None:
    session_id, agent_id = build_test_context(context, BANKING_SCENARIO, n)
    message_composer = context.container[MessageAssembler]
    message_composer.generate_events()
    assert True


# TODO create tool test
# TODO create assembler test
# TODO add fragments
# TODO create full test
