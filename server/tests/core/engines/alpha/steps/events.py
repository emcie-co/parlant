from typing import Optional, cast
from pytest_bdd import given, then, parsers, when

from emcie.common.tools import ToolId
from emcie.server.core.engines.alpha.utils import emitted_tool_event_to_dict
from emcie.server.core.emissions import EmittedEvent
from emcie.server.core.nlp.moderation import ModerationTag
from emcie.server.core.sessions import (
    MessageEventData,
    SessionId,
    SessionStatus,
    SessionStore,
    StatusEventData,
    ToolEventData,
)

from tests import tool_utilities
from tests.core.engines.alpha.utils import ContextOfTest, step
from tests.test_utilities import nlp_test


@step(
    given,
    parsers.parse('an agent message, "{server_message}"'),
    target_fixture="session_id",
)
def given_a_server_message(
    context: ContextOfTest,
    server_message: str,
    session_id: SessionId,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    event = context.sync_await(
        store.create_event(
            session_id=session.id,
            source="server",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": server_message},
        )
    )

    context.events.append(event)

    return session.id


@step(given, parsers.parse('a user message, "{user_message}"'), target_fixture="session_id")
def given_a_user_message(
    context: ContextOfTest,
    session_id: SessionId,
    user_message: str,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    event = context.sync_await(
        store.create_event(
            session_id=session.id,
            source="client",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": user_message},
        )
    )

    context.events.append(event)

    return session.id


@step(
    given,
    parsers.parse('a user message, "{user_message}", flagged for {moderation_tag}'),
    target_fixture="session_id",
)
def given_a_flagged_user_message(
    context: ContextOfTest,
    session_id: SessionId,
    user_message: str,
    moderation_tag: ModerationTag,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    event = context.sync_await(
        store.create_event(
            session_id=session.id,
            source="client",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": user_message, "flagged": True, "tags": [moderation_tag]},
        )
    )

    context.events.append(event)

    return session.id


@step(
    when,
    parsers.parse("the last {num_messages:d} messages are deleted"),
    target_fixture="session_id",
)
def when_the_last_few_messages_are_deleted(
    context: ContextOfTest,
    session_id: SessionId,
    num_messages: int,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    events = context.sync_await(store.list_events(session_id=session.id))

    for event in events[-num_messages:]:
        context.sync_await(store.delete_event(event_id=event.id))

    return session.id


@step(then, "a single message event is emitted")
def then_a_single_message_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len(list(filter(lambda e: e.kind == "message", emitted_events))) == 1


@step(then, parsers.parse("the message contains {something}"))
def then_the_message_contains(
    context: ContextOfTest,
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == "message")
    message = cast(MessageEventData, message_event.data)["message"]

    assert context.sync_await(
        nlp_test(
            context=f"Here's a message in the context of a conversation: {message}",
            predicate=f"the text contains {something}",
        )
    )


@step(then, parsers.parse("the message mentions {something}"))
def then_the_message_mentions(
    context: ContextOfTest,
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == "message")
    message = cast(MessageEventData, message_event.data)["message"]

    assert context.sync_await(
        nlp_test(
            context=f"Here's a message in the context of a conversation: {message}",
            predicate=f"the text mentions {something}",
        )
    )


@step(then, "no events are emitted")
def then_no_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len(emitted_events) == 0


@step(then, "no message events are emitted")
def then_no_message_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len([e for e in emitted_events if e.kind == "message"]) == 0


def _has_status_event(
    status: SessionStatus,
    acknowledged_event_offset: Optional[int],
    events: list[EmittedEvent],
) -> bool:
    for e in (e for e in events if e.kind == "status"):
        data = cast(StatusEventData, e.data)

        has_same_status = data["status"] == status

        if acknowledged_event_offset:
            has_same_acknowledged_offset = data["acknowledged_offset"] == acknowledged_event_offset

            if has_same_status and has_same_acknowledged_offset:
                return True
        elif has_same_status:
            return True

    return False


@step(
    then,
    parsers.parse("a status event is emitted, acknowledging event {acknowledged_event_offset:d}"),
)
def then_an_acknowledgement_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("acknowledged", acknowledged_event_offset, emitted_events)


@step(
    then, parsers.parse("a status event is emitted, processing event {acknowledged_event_offset:d}")
)
def then_a_processing_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("processing", acknowledged_event_offset, emitted_events)


@step(
    then,
    parsers.parse(
        "a status event is emitted, typing in response to event {acknowledged_event_offset:d}"
    ),
)
def then_a_typing_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("typing", acknowledged_event_offset, emitted_events)


@step(
    then,
    parsers.parse(
        "a status event is emitted, cancelling the response to event {acknowledged_event_offset:d}"
    ),
)
def then_a_cancelled_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("cancelled", acknowledged_event_offset, emitted_events)


@step(
    then,
    parsers.parse(
        "a status event is emitted, ready for further engagement after reacting to event {acknowledged_event_offset:d}"
    ),
)
def then_a_ready_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("ready", acknowledged_event_offset, emitted_events)


@step(
    then,
    parsers.parse(
        "a status event is emitted, encountering an error while processing event {acknowledged_event_offset:d}"
    ),
)
def then_an_error_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("error", acknowledged_event_offset, emitted_events)


@step(then, parsers.parse("a {status_type} status event is not emitted"))
def then_a_status_event_type_is_not_emitted(
    emitted_events: list[EmittedEvent],
    status_type: SessionStatus,
) -> None:
    assert not _has_status_event(status_type, None, emitted_events)


@step(then, "no tool calls event is emitted")
def then_no_tool_calls_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert 0 == len([e for e in emitted_events if e.kind == "tool"])


@step(then, "a single tool calls event is emitted")
def then_a_single_tool_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert 1 == len([e for e in emitted_events if e.kind == "tool"])


@step(then, parsers.parse("the tool calls event contains {number_of_tool_calls:d} tool call(s)"))
def then_the_tool_calls_event_contains_n_tool_calls(
    number_of_tool_calls: int,
    emitted_events: list[EmittedEvent],
) -> None:
    tool_calls_event = next(e for e in emitted_events if e.kind == "tool")
    assert number_of_tool_calls == len(cast(ToolEventData, tool_calls_event.data)["tool_calls"])


@step(then, parsers.parse("the tool calls event contains {expected_content}"))
def then_the_tool_calls_event_contains_expected_content(
    context: ContextOfTest,
    expected_content: str,
    emitted_events: list[EmittedEvent],
) -> None:
    tool_calls_event = next(e for e in emitted_events if e.kind == "tool")
    tool_calls = cast(ToolEventData, tool_calls_event.data)["tool_calls"]

    assert context.sync_await(
        nlp_test(
            context=f"The following is the result of tool (function) calls: {tool_calls}",
            predicate=f"The calls contain {expected_content}",
        )
    )


@step(
    then,
    parsers.parse(
        "the execution result of {tool_id} is emitted in the "
        "{tool_event_number:d}{ordinal_indicator:s} tool event"
    ),
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


@step(
    then,
    parsers.parse(
        "a tool event for product availability of {product_type} is generated "
        "at tool event number {tool_event_number:d}"
    ),
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


@step(
    then,
    parsers.parse(
        "an add tool event is emitted with {first_num:d}, {second_num:d} numbers in "
        "tool event number {tool_event_number:d}"
    ),
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


@step(
    then,
    parsers.parse(
        "a multiply tool event is emitted with {first_num:d}, {second_num:d} "
        "numbers in tool event number {tool_event_number:d}"
    ),
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


@step(
    then,
    parsers.parse(
        "a get balance account tool event is emitted for the {name} account "
        "in tool event number {tool_event_number:d}"
    ),
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


@step(
    then,
    "a get account loans tool event is emitted for {name} "
    "in tool event number {tool_event_number:d}",
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


@step(then, "the tool calls event is correlated with the message event")
def then_the_tool_calls_event_is_correlated_with_the_message_event(
    emitted_events: list[EmittedEvent],
) -> None:
    message_event = next(e for e in emitted_events if e.kind == "message")
    tool_calls_event = next(e for e in emitted_events if e.kind == "tool")

    assert message_event.correlation_id == tool_calls_event.correlation_id
