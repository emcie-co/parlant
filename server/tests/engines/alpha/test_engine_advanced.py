from dataclasses import dataclass
import json
from typing import Any, Optional, cast
from lagom import Container
from pytest import fixture
from pytest_bdd import scenarios, given, when, then, parsers
from datetime import datetime, timezone

from emcie.common.tools import Tool, ToolId
from emcie.server.core.agents import AgentId, AgentStore
from emcie.server.core.context_variables import ContextVariableStore, ContextVariableValue
from emcie.server.core.end_users import EndUserId
from emcie.server.core.guideline_connections import (
    ConnectionKind,
    GuidelineConnection,
    GuidelineConnectionStore,
)
from emcie.server.core.tools import LocalToolService, MultiplexedToolService
from emcie.server.engines.alpha.engine import AlphaEngine
from emcie.server.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationStore,
)
from emcie.server.engines.alpha.utils import emitted_tool_event_to_dict
from emcie.server.engines.common import Context
from emcie.server.engines.event_emitter import EmittedEvent
from emcie.server.core.guidelines import Guideline, GuidelineStore
from emcie.server.core.sessions import (
    MessageEventData,
    SessionId,
    SessionStatus,
    SessionStore,
    StatusEventData,
    ToolEventData,
)

from emcie.server.logger import Logger
from tests import tool_utilities
from tests.test_utilities import EventBuffer, SyncAwaiter, nlp_test

scenarios(
    "engines/alpha/tools/single_tool_event.feature",
    "engines/alpha/tools/proactive_agent.feature",
    "engines/alpha/guideline_connections.feature",
)


@dataclass
class _TestContext:
    guidelines: dict[str, Guideline]
    tools: dict[str, Tool]


@fixture
def context() -> _TestContext:
    return _TestContext(
        guidelines=dict(),
        tools=dict(),
    )


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent.id


@given("the alpha engine", target_fixture="engine")
def given_the_alpha_engine(
    container: Container,
) -> AlphaEngine:
    return container[AlphaEngine]


@given("an agent", target_fixture="agent_id")
def given_an_agent(
    agent_id: AgentId,
) -> AgentId:
    return agent_id


@given(parsers.parse("an agent whose job is {description}"), target_fixture="agent_id")
def given_an_agent_with_identity(
    container: Container,
    sync_await: SyncAwaiter,
    description: str,
) -> AgentId:
    agent = sync_await(
        container[AgentStore].create_agent(
            name="test-agent",
            description=f"Your job is {description}",
        )
    )
    return agent.id


@given("an empty session", target_fixture="session_id")
def given_an_empty_session(
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> SessionId:
    store = container[SessionStore]
    utc_now = datetime.now(timezone.utc)
    session = sync_await(
        store.create_session(
            creation_utc=utc_now,
            end_user_id=EndUserId("test_user"),
            agent_id=agent_id,
        )
    )
    return session.id


@given(parsers.parse('a context variable "{variable_name}" with a value of "{variable_value}"'))
def given_a_context_variable(
    variable_name: str,
    variable_value: str,
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
    session_id: SessionId,
) -> ContextVariableValue:
    session_store = container[SessionStore]
    context_variable_store = container[ContextVariableStore]

    end_user_id = sync_await(session_store.read_session(session_id)).end_user_id

    variable = sync_await(
        context_variable_store.create_variable(
            variable_set=agent_id,
            name=variable_name,
            description="",
            tool_id=ToolId(""),
            freshness_rules=None,
        )
    )

    return sync_await(
        context_variable_store.update_value(
            variable_set=agent_id,
            key=end_user_id,
            variable_id=variable.id,
            data={variable_name: variable_value},
        )
    )


@given(parsers.parse("a guideline to {do_something} when {a_condition_holds}"))
def given_a_guideline_to_when(
    do_something: str,
    a_condition_holds: str,
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    sync_await(
        guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=a_condition_holds,
            action=do_something,
        )
    )


@given(parsers.parse('a guideline "{guideline_name}", to {do_something} when {a_condition_holds}'))
def given_a_guideline_name_to_when(
    guideline_name: str,
    do_something: str,
    a_condition_holds: str,
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
    context: _TestContext,
) -> None:
    guideline_store = container[GuidelineStore]

    context.guidelines[guideline_name] = sync_await(
        guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=a_condition_holds,
            action=do_something,
        )
    )


@given(parsers.parse('the guideline called "{guideline_id}"'))
def given_a_guideline(
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
    guideline_id: str,
) -> Guideline:
    guideline_store = container[GuidelineStore]

    async def create_guideline(predicate: str, action: str) -> Guideline:
        return await guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=predicate,
            action=action,
        )

    guidelines = {
        "check_drinks_in_stock": {
            "predicate": "a client asks for a drink",
            "action": "check if the drink is available in stock",
        },
        "check_toppings_in_stock": {
            "predicate": "a client asks about toppings or order pizza with toppings",
            "action": "check what toppings are available in stock",
        },
        "ask_expert_about_Spot": {
            "predicate": "a client asks for information about Spot",
            "action": "ask and get the answer from the expert",
        },
        "check_toppings_or_drinks_in_stock": {
            "predicate": "a client asks for toppings or drinks",
            "action": "check if they are available in stock",
        },
        "calculate_sum": {
            "predicate": "an equation involves adding numbers",
            "action": "calculate the sum",
        },
        "check_drinks_or_toppings_in_stock": {
            "predicate": "a client asks for a drink or toppings",
            "action": "check what drinks or toppings are available in stock",
        },
        "calculate_addition_or_multiplication": {
            "predicate": "an equation contains addition or multiplication",
            "action": "calculate it",
        },
        "retrieve_account_information": {
            "predicate": "asked for information about an account",
            "action": "answer by retrieving the information from the database",
        },
        "calculate_addition": {
            "predicate": "an equation contains an add function",
            "action": "get the result from the add tool",
        },
        "calculate_multiplication": {
            "predicate": "an equation contains a multiply function",
            "action": "get the result from the multiply tool",
        },
        "transfer_money_between_accounts": {
            "predicate": "asked to transfer money from one account to another",
            "action": "check if the account has enough balance to make the transfer"
            "and then proceed with the transfer",
        },
        "retrieve_Spot_information": {
            "predicate": "asked for information about Spot",
            "action": "answer by retrieving the information from the database",
        },
        "retrieve_account_balance": {
            "predicate": "asked for information about an account",
            "action": "answer by retrieving the information from the database",
        },
    }

    return sync_await(create_guideline(**guidelines[guideline_id]))


@given(parsers.parse('the tool "{tool_name}"'))
def given_a_tool(
    sync_await: SyncAwaiter,
    container: Container,
    tool_name: str,
    context: _TestContext,
) -> None:
    tool_store = container[LocalToolService]

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
                "user_query": {"type": "string", "description": "The query from the user"}
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
    }

    tool = sync_await(tool_store.create_tool(**tools[tool_name]))

    multiplexed_tool_service = container[MultiplexedToolService]

    context.tools[tool_name] = sync_await(
        multiplexed_tool_service.read_tool(
            tool.id, next(iter(multiplexed_tool_service.services.keys()))
        )
    )


@given(parsers.parse('an association between "{guideline_name}" and "{tool_name}"'))
def given_a_guideline_tool_association(
    sync_await: SyncAwaiter,
    container: Container,
    tool_name: str,
    guideline_name: str,
    context: _TestContext,
) -> GuidelineToolAssociation:
    guideline_tool_association_store = container[GuidelineToolAssociationStore]

    return sync_await(
        guideline_tool_association_store.create_association(
            guideline_id=context.guidelines[guideline_name].id,
            tool_id=context.tools[tool_name].id,
        )
    )


@given(parsers.parse('a guideline connection whereby "{source}" {kind} "{target}"'))
def given_a_guideline_connection(
    sync_await: SyncAwaiter,
    container: Container,
    source: str,
    target: str,
    kind: str,
    context: _TestContext,
) -> GuidelineConnection:
    store = container[GuidelineConnectionStore]

    source_guideline = context.guidelines[source]
    target_guideline = context.guidelines[target]

    return sync_await(
        store.update_connection(
            source=source_guideline.id,
            target=target_guideline.id,
            kind={"entails": ConnectionKind.ENTAILS, "suggests": ConnectionKind.SUGGESTS}[kind],
        )
    )


@given(parsers.parse('a user message, "{user_message}"'), target_fixture="session_id")
def given_a_user_message(
    sync_await: SyncAwaiter,
    container: Container,
    session_id: SessionId,
    user_message: str,
) -> SessionId:
    store = container[SessionStore]
    session = sync_await(store.read_session(session_id=session_id))

    sync_await(
        store.create_event(
            session_id=session.id,
            source="client",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": user_message},
        )
    )

    return session.id


@given(
    parsers.parse('a server message, "{server_message}"'),
    target_fixture="session_id",
)
def given_a_server_message(
    sync_await: SyncAwaiter,
    container: Container,
    server_message: str,
    session_id: SessionId,
) -> SessionId:
    store = container[SessionStore]
    session = sync_await(store.read_session(session_id=session_id))

    sync_await(
        store.create_event(
            session_id=session.id,
            source="server",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": server_message},
        )
    )

    return session.id


@given(
    parsers.parse("a tool event with data, {tool_event_data}"),
    target_fixture="session_id",
)
def given_a_session_with_tool_event(
    sync_await: SyncAwaiter,
    container: Container,
    session_id: SessionId,
    tool_event_data: str,
) -> SessionId:
    store = container[SessionStore]
    session = sync_await(store.read_session(session_id=session_id))

    sync_await(
        store.create_event(
            session_id=session.id,
            source="server",
            kind="tool",
            correlation_id="test_correlation_id",
            data=json.loads(tool_event_data),
        )
    )

    return session.id


@when("processing is triggered", target_fixture="emitted_events")
def when_processing_is_triggered(
    sync_await: SyncAwaiter,
    engine: AlphaEngine,
    agent_id: AgentId,
    session_id: SessionId,
) -> list[EmittedEvent]:
    buffer = EventBuffer()

    sync_await(
        engine.process(
            Context(
                session_id=session_id,
                agent_id=agent_id,
            ),
            buffer,
        )
    )

    return buffer.events


@then("no tool calls event is emitted")
def then_no_tools_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert 0 == len([e for e in emitted_events if e.kind == "tool"])


@then("a single tool calls event is emitted")
def then_a_single_tool_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert 1 == len([e for e in emitted_events if e.kind == "tool"])


@then(parsers.parse("the tool calls event contains {number_of_tool_calls:d} tool call(s)"))
def then_the_tool_calls_event_contains_n_tool_calls(
    number_of_tool_calls: int,
    emitted_events: list[EmittedEvent],
) -> None:
    tool_calls_event = next(e for e in emitted_events if e.kind == "tool")
    assert number_of_tool_calls == len(cast(ToolEventData, tool_calls_event.data)["tool_calls"])


@then(parsers.parse("the tool calls event contains {expected_content}"))
def then_the_tool_calls_event_contains_expected_content(
    container: Container,
    sync_await: SyncAwaiter,
    expected_content: str,
    emitted_events: list[EmittedEvent],
) -> None:
    tool_calls_event = next(e for e in emitted_events if e.kind == "tool")
    tool_calls = cast(ToolEventData, tool_calls_event.data)["tool_calls"]

    assert sync_await(
        nlp_test(
            logger=container[Logger],
            context=f"The following is the result of tool (function) calls: {tool_calls}",
            predicate=f"The calls contain {expected_content}",
        )
    )


@then(
    parsers.parse(
        "the execution result of {tool_id} is emitted in the "
        "{tool_event_number:d}{ordinal_indicator:s} tool event"
    )
)
def then_drinks_available_in_stock_tool_event_is_emitted(
    emitted_events: list[EmittedEvent],
    tool_id: ToolId,
    tool_event_number: int,
) -> None:
    results = emitted_tool_event_to_dict(emitted_events[tool_event_number - 1])["data"]

    tool_event_functions = {
        "tool_id": tool_utilities.get_available_drinks,
        "tool_id1": tool_utilities.get_available_toppings,
    }

    assert {
        "tool_name": tool_id,
        "parameters": {},
        "result": tool_event_functions[tool_id](),
    } in results


@then(
    parsers.parse(
        "a tool event for product availability of {product_type} is generated "
        "at tool event number {tool_event_number:d}"
    )
)
def then_product_availability_for_toppings_and_drinks_tools_event_is_emitted(
    emitted_events: list[EmittedEvent],
    product_type: str,
    tool_event_number: int,
) -> None:
    types_functions = {
        "drinks": tool_utilities.get_available_drinks,
        "toppings": tool_utilities.get_available_toppings,
    }
    results = emitted_tool_event_to_dict(emitted_events[tool_event_number - 1])["data"]
    assert {
        "tool_name": "get_available_product_by_type",
        "parameters": {"product_type": product_type},
        "result": types_functions[product_type](),
    } in results


@then(
    parsers.parse(
        "an add tool event is emitted with {first_num:d}, {second_num:d} numbers in "
        "tool event number {tool_event_number:d}"
    )
)
def then_add_tool_event_is_emitted(
    emitted_events: list[EmittedEvent],
    first_num: int,
    second_num: int,
    tool_event_number: int,
) -> None:
    results = emitted_tool_event_to_dict(emitted_events[tool_event_number - 1])["data"]

    assert {
        "tool_name": "add",
        "parameters": {
            "first_number": first_num,
            "second_number": second_num,
        },
        "result": tool_utilities.add(first_num, second_num),
    } in results


@then(
    parsers.parse(
        "a multiply tool event is emitted with {first_num:d}, {second_num:d} "
        "numbers in tool event number {tool_event_number:d}"
    )
)
def then_multiply_tool_event_is_emitted(
    emitted_events: list[EmittedEvent],
    first_num: int,
    second_num: int,
    tool_event_number: int,
) -> None:
    results = emitted_tool_event_to_dict(emitted_events[tool_event_number - 1])["data"]

    assert {
        "tool_name": "multiply",
        "parameters": {
            "first_number": first_num,
            "second_number": second_num,
        },
        "result": tool_utilities.multiply(first_num, second_num),
    } in results


@then(
    parsers.parse(
        "a get balance account tool event is emitted for the {name} account "
        "in tool event number {tool_event_number:d}"
    )
)
def then_get_balance_account_tool_event_is_emitted(
    emitted_events: list[EmittedEvent],
    name: str,
    tool_event_number: int,
) -> None:
    results = emitted_tool_event_to_dict(emitted_events[tool_event_number - 1])["data"]

    assert {
        "tool_name": "get_account_balance",
        "parameters": {
            "account_name": name,
        },
        "result": tool_utilities.get_account_balance(name),
    } in results


@then(
    "a get account loans tool event is emitted for {name} "
    "in tool event number {tool_event_number:d}"
)
def then_get_account_loans_tool_event_is_emitted(
    emitted_events: list[EmittedEvent],
    name: str,
    tool_event_number: int,
) -> None:
    results = emitted_tool_event_to_dict(emitted_events[tool_event_number - 1])["data"]

    assert {
        "tool_name": "get_account_loans",
        "parameters": {
            "account_name": name,
        },
        "result": tool_utilities.get_account_loans(name),
    } in results


@then(parsers.parse("the message contains {something}"))
def then_the_message_contains(
    container: Container,
    sync_await: SyncAwaiter,
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == "message")
    message = cast(MessageEventData, message_event.data)["message"]

    assert sync_await(
        nlp_test(
            logger=container[Logger],
            context=message,
            predicate=f"the text contains {something}",
        )
    )


@then("no events are emitted")
def then_no_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len(emitted_events) == 0


@then("no message events are emitted")
def then_no_message_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len([e for e in emitted_events if e.kind == "message"]) == 0


@then("a single message event is emitted")
def then_a_single_message_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len(list(filter(lambda e: e.kind == "message", emitted_events))) == 1


@then("the tool calls event is correlated with the message event")
def then_the_tool_calls_event_is_correlated_with_the_message_event(
    emitted_events: list[EmittedEvent],
) -> None:
    message_event = next(e for e in emitted_events if e.kind == "message")
    tool_calls_event = next(e for e in emitted_events if e.kind == "tool")

    assert message_event.correlation_id == tool_calls_event.correlation_id


def _has_status_event(
    status: SessionStatus,
    acknowledged_event_offset: Optional[int],
    events: list[EmittedEvent],
) -> bool:
    for e in (e for e in events if e.kind == "status"):
        data = cast(StatusEventData, e.data)

        has_same_status = data["status"] == status

        if acknowledged_event_offset is not None:
            has_same_acknowledged_offset = data["acknowledged_offset"] == acknowledged_event_offset

            if has_same_status and has_same_acknowledged_offset:
                return True
        else:
            if has_same_status:
                return True

    return False


@then(parsers.parse("a status event is emitted, acknowledging event {acknowledged_event_offset:d}"))
def then_an_acknowledgement_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("acknowledged", acknowledged_event_offset, emitted_events)


@then(parsers.parse("a status event is emitted, processing event {acknowledged_event_offset:d}"))
def then_a_processing_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("processing", acknowledged_event_offset, emitted_events)


@then(
    parsers.parse(
        "a status event is emitted, typing in response to event {acknowledged_event_offset:d}"
    )
)
def then_a_typing_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("typing", acknowledged_event_offset, emitted_events)


@then(
    parsers.parse(
        "a status event is emitted, cancelling the response to event {acknowledged_event_offset:d}"
    )
)
def then_a_cancelled_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("cancelled", acknowledged_event_offset, emitted_events)


@then(
    parsers.parse(
        "a status event is emitted, ready for further engagement after reacting to event {acknowledged_event_offset:d}"
    )
)
def then_a_ready_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("ready", acknowledged_event_offset, emitted_events)


@then(parsers.parse("a {status_type} status event is not emitted"))
def then_a_status_event_is_not_emitted(
    emitted_events: list[EmittedEvent],
    status_type: SessionStatus,
) -> None:
    assert not _has_status_event(status_type, None, emitted_events)
