# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, sorftware
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pprint import pformat
from typing import cast
from pytest_bdd import given, then, parsers, when

from parlant.core.agents import AgentId, AgentStore
from parlant.core.common import JSONSerializable
from parlant.core.customers import CustomerStore
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.canned_response_generator import DEFAULT_NO_MATCH_CANREP
from parlant.core.nlp.moderation import ModerationTag

from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    SessionId,
    SessionStatus,
    SessionStore,
    StatusEventData,
    ToolCall,
    ToolEventData,
)
from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest
from tests.test_utilities import nlp_test, JournalingEngineHooks


@step(
    given,
    parsers.parse('an agent message, "{agent_message}"'),
    target_fixture="session_id",
)
def given_an_agent_message(
    context: ContextOfTest,
    agent_message: str,
    session_id: SessionId,
    agent_id: AgentId,
) -> SessionId:
    session_store = context.container[SessionStore]
    agent_store = context.container[AgentStore]

    session = context.sync_await(session_store.read_session(session_id=session_id))
    agent = context.sync_await(agent_store.read_agent(agent_id))

    message_data: MessageEventData = {
        "message": agent_message,
        "participant": {
            "id": agent.id,
            "display_name": agent.name,
        },
    }

    event = context.sync_await(
        session_store.create_event(
            session_id=session.id,
            source=EventSource.AI_AGENT,
            kind=EventKind.MESSAGE,
            trace_id="<main>",
            data=cast(JSONSerializable, message_data),
        )
    )

    context.events.append(event)

    return session.id


@step(
    given,
    parsers.parse('a human message on behalf of the agent, "{agent_message}"'),
    target_fixture="session_id",
)
def given_a_human_message_on_behalf_of_the_agent(
    context: ContextOfTest,
    agent_message: str,
    session_id: SessionId,
    agent_id: AgentId,
) -> SessionId:
    session_store = context.container[SessionStore]
    agent_store = context.container[AgentStore]

    session = context.sync_await(session_store.read_session(session_id=session_id))
    agent = context.sync_await(agent_store.read_agent(agent_id))

    message_data: MessageEventData = {
        "message": agent_message,
        "participant": {
            "id": agent.id,
            "display_name": agent.name,
        },
    }

    event = context.sync_await(
        session_store.create_event(
            session_id=session.id,
            source=EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
            kind=EventKind.MESSAGE,
            trace_id="<main>",
            data=cast(JSONSerializable, message_data),
        )
    )

    context.events.append(event)

    return session.id


@step(given, parsers.parse('a customer message, "{customer_message}"'), target_fixture="session_id")
def given_a_customer_message(
    context: ContextOfTest,
    session_id: SessionId,
    customer_message: str,
) -> SessionId:
    session_store = context.container[SessionStore]
    customer_store = context.container[CustomerStore]

    session = context.sync_await(session_store.read_session(session_id=session_id))
    customer = context.sync_await(customer_store.read_customer(customer_id=session.customer_id))

    message_data: MessageEventData = {
        "message": customer_message,
        "participant": {
            "id": customer.id,
            "display_name": customer.name,
        },
    }

    event = context.sync_await(
        session_store.create_event(
            session_id=session.id,
            source=EventSource.CUSTOMER,
            kind=EventKind.MESSAGE,
            trace_id="<main>",
            data=cast(JSONSerializable, message_data),
        )
    )

    context.events.append(event)

    return session.id


@step(
    given,
    parsers.parse('a customer message, "{customer_message}", flagged for {moderation_tag}'),
    target_fixture="session_id",
)
def given_a_flagged_customer_message(
    context: ContextOfTest,
    session_id: SessionId,
    customer_message: str,
    moderation_tag: ModerationTag,
) -> SessionId:
    session_store = context.container[SessionStore]
    customer_store = context.container[CustomerStore]

    session = context.sync_await(session_store.read_session(session_id=session_id))
    customer = context.sync_await(customer_store.read_customer(customer_id=session.customer_id))

    message_data: MessageEventData = {
        "message": customer_message,
        "participant": {
            "id": customer.id,
            "display_name": customer.name,
        },
        "flagged": True,
        "tags": [moderation_tag],
    }

    event = context.sync_await(
        session_store.create_event(
            session_id=session.id,
            source=EventSource.CUSTOMER,
            kind=EventKind.MESSAGE,
            trace_id="<main>",
            data=cast(JSONSerializable, message_data),
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
    assert len(list(filter(lambda e: e.kind == EventKind.MESSAGE, emitted_events))) == 1


@step(then, parsers.parse("a total of {count:d} message events are emitted"))
def then_message_events_are_emitted(
    emitted_events: list[EmittedEvent],
    count: int,
) -> None:
    message_count = sum(1 for e in emitted_events if e.kind == EventKind.MESSAGE)
    assert message_count == count, f"Expected {count} message events, but found {message_count}"


@step(then, parsers.parse('the message contains the text "{something}"'))
def then_the_message_contains_the_text(
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == EventKind.MESSAGE)
    message = cast(MessageEventData, message_event.data)["message"]

    assert something.lower() in message.lower(), (
        f"message: '{message}', expected to contain the text: '{something}'"
    )


@step(then, parsers.parse('the message doesn\'t contain the text "{something}"'))
def then_the_message_does_not_contain_the_text(
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == EventKind.MESSAGE)
    message = cast(MessageEventData, message_event.data)["message"]

    assert something.lower() not in message.lower(), (
        f"message: '{message}', expected to NOT contain the text: '{something}'"
    )


@step(then, parsers.parse("the message contains {something}"))
def then_the_message_contains(
    context: ContextOfTest,
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == EventKind.MESSAGE)
    message = cast(MessageEventData, message_event.data)["message"]

    assert context.sync_await(
        nlp_test(
            context=f"Here's a message from an AI agent to a customer, in the context of a conversation: {message}",
            condition=f"The message contains {something}",
        )
    ), f"message: '{message}', expected to contain: '{something}'"


@step(then, parsers.parse('at least one message contains the text "{something}"'))
def then_the_ith_message_contains(
    context: ContextOfTest,
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_events = [e for e in emitted_events if e.kind == EventKind.MESSAGE]
    messages = [cast(MessageEventData, e.data)["message"] for e in message_events]
    messages_str = " || ".join(messages)

    assert any(something.lower() in m.lower() for m in messages), (
        f"text: '{something} not found in outputted messages {messages_str}'"
    )


@step(then, parsers.parse("the message doesn't contains {something}"))
def then_the_doesnt_message_contains(
    context: ContextOfTest,
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == EventKind.MESSAGE)
    message = cast(MessageEventData, message_event.data)["message"]

    assert context.sync_await(
        nlp_test(
            context=f"Here's a message from an AI agent to a customer, in the context of a conversation: {message}",
            condition=f"The message NOT contains {something}",
        )
    ), f"message: '{message}', expected to contain: '{something}'"


@step(then, parsers.parse("the message mentions {something}"))
def then_the_message_mentions(
    context: ContextOfTest,
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == EventKind.MESSAGE)
    message = cast(MessageEventData, message_event.data)["message"]

    assert context.sync_await(
        nlp_test(
            context=f"Here's a message from an AI agent to a customer, in the context of a conversation: {message}",
            condition=f"The message mentions {something}",
        )
    ), f"message: '{message}', expected to contain: '{something}'"


@step(
    then,
    parsers.parse('the message uses the canned response "{canrep_text}"'),
)
def then_the_message_uses_the_canned_response(
    emitted_events: list[EmittedEvent],
    canned_response_text: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == EventKind.MESSAGE)
    message_data = cast(MessageEventData, message_event.data)
    assert message_data["canned_responses"]

    assert any(canned_response_text in canrep for _, canrep in message_data["canned_responses"])


@step(
    then,
    parsers.parse('the message doesn\'t use the canned response "{canrep_text}"'),
)
def then_the_message_does_not_use_the_canned_response(
    emitted_events: list[EmittedEvent],
    canned_response_text: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == EventKind.MESSAGE)
    message_data = cast(MessageEventData, message_event.data)

    assert all(canned_response_text not in canrep for _, canrep in message_data["canned_responses"])


@step(then, "no events are emitted")
def then_no_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len(emitted_events) == 0


@step(then, "no message events are emitted")
def then_no_message_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len([e for e in emitted_events if e.kind == EventKind.MESSAGE]) == 0


@step(then, "a no-match message is emitted")
def then_a_no_match_message_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    message_event = next(e for e in emitted_events if e.kind == EventKind.MESSAGE)
    message = cast(MessageEventData, message_event.data)["message"]

    assert message == DEFAULT_NO_MATCH_CANREP, (
        f"message: '{message}', expected to be{DEFAULT_NO_MATCH_CANREP}'"
    )


def _has_status_event(
    status: SessionStatus,
    events: list[EmittedEvent],
) -> bool:
    for e in (e for e in events if e.kind == EventKind.STATUS):
        data = cast(StatusEventData, e.data)

        has_same_status = data["status"] == status

        if has_same_status:
            return True

    return False


@step(
    then,
    parsers.parse("a status event is emitted, acknowledging event"),
)
def then_an_acknowledgement_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert _has_status_event("acknowledged", emitted_events)


@step(then, parsers.parse("a status event is emitted, processing event"))
def then_a_processing_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert _has_status_event("processing", emitted_events)


@step(
    then,
    parsers.parse("a status event is emitted, typing in response to event"),
)
def then_a_typing_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert _has_status_event("typing", emitted_events)


@step(
    then,
    parsers.parse("a status event is emitted, cancelling the response to event"),
)
def then_a_cancelled_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert _has_status_event("cancelled", emitted_events)


@step(
    then,
    parsers.parse(
        "a status event is emitted, ready for further engagement after reacting to event"
    ),
)
def then_a_ready_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert _has_status_event("ready", emitted_events)


@step(
    then,
    parsers.parse("a status event is emitted, encountering an error while processing event"),
)
def then_an_error_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert _has_status_event("error", emitted_events)


@step(then, parsers.parse("no tool error has occurred"))
def then_no_tool_error_occurred(emitted_events: list[EmittedEvent]) -> None:
    tool_events = [e for e in emitted_events if e.kind == EventKind.TOOL]
    for tool_event in tool_events:
        tool_event_data = cast(ToolEventData, tool_event.data)
        for tc in tool_event_data["tool_calls"]:
            result_data = tc["result"].get("data", [])
            assert not (isinstance(result_data, str) and "error" in result_data), (
                f"A tool error has occurred in tool: {tc}"
            )


@step(then, parsers.parse("a {status_type} status event is not emitted"))
def then_a_status_event_type_is_not_emitted(
    emitted_events: list[EmittedEvent],
    status_type: SessionStatus,
) -> None:
    assert not _has_status_event(status_type, emitted_events)


@step(then, "no tool calls event is emitted")
def then_no_tool_calls_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    tool_events = [e for e in emitted_events if e.kind == EventKind.TOOL]
    assert 0 == len(tool_events), pformat(tool_events, indent=2)


@step(then, "a single tool calls event is emitted")
def then_a_single_tool_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    tool_events = [e for e in emitted_events if e.kind == EventKind.TOOL]
    assert 1 == len(tool_events), pformat(tool_events, indent=2)


@step(then, parsers.parse("the tool calls event contains {number_of_tool_calls:d} tool call(s)"))
def then_the_tool_calls_event_contains_n_tool_calls(
    number_of_tool_calls: int,
    emitted_events: list[EmittedEvent],
) -> None:
    tool_calls = [
        cast(ToolEventData, e.data)["tool_calls"]
        for e in emitted_events
        if e.kind == EventKind.TOOL
    ]
    assert number_of_tool_calls == len(tool_calls), pformat(tool_calls, indent=2)


def _get_tool_calls(emitted_events: list[EmittedEvent]) -> list[ToolCall]:
    return [
        tool_call
        for e in emitted_events
        if e.kind == EventKind.TOOL
        for tool_call in cast(ToolEventData, e.data)["tool_calls"]
    ]


@step(then, parsers.parse("the tool calls event contains {expected_content}"))
def then_the_tool_calls_event_contains_expected_content(
    context: ContextOfTest,
    expected_content: str,
    emitted_events: list[EmittedEvent],
) -> None:
    tool_calls = _get_tool_calls(emitted_events)

    assert context.sync_await(
        nlp_test(
            context=f"The following is the result of tool (function) calls: {tool_calls}",
            condition=f"The calls contain {expected_content}",
        )
    ), pformat(tool_calls, indent=2)


@step(then, "the tool calls event is traced with the message event")
def then_the_tool_calls_event_is_traced_with_the_message_event(
    emitted_events: list[EmittedEvent],
) -> None:
    tool_events = [e for e in emitted_events if e.kind == EventKind.TOOL]
    message_events = [e for e in emitted_events if e.kind == EventKind.MESSAGE]

    assert len(tool_events) > 0, "No tool event found"
    assert len(message_events) > 0, "No message event found"

    tool_event = tool_events[0]
    message_event = message_events[0]

    assert tool_event.trace_id == message_event.trace_id


@step(then, parsers.parse('the tool calls event contains a call to "{tool_name}"'))
def then_the_tool_calls_event_contains_call(
    emitted_events: list[EmittedEvent],
    tool_name: str,
) -> None:
    tool_calls = _get_tool_calls(emitted_events)

    matching_tool_calls = [
        tc
        for tc in tool_calls
        if tc["tool_id"].endswith(f":{tool_name}") or tc["tool_id"] == f"local:{tool_name}"
    ]

    assert len(matching_tool_calls) > 0, f"No tool call found for {tool_name}"


@step(then, parsers.parse("the number of missing parameters is exactly {number_of_missing:d}"))
def then_the_number_of_missing_is_exactly(
    context: ContextOfTest,
    number_of_missing: int,
) -> None:
    latest_context = next(
        iter(context.container[JournalingEngineHooks].latest_context_per_trace_id.values())
    )
    missing_data = latest_context.state.tool_insights.missing_data

    assert len(missing_data) == number_of_missing, (
        f"Expected {number_of_missing} missing parameters, but found {len(missing_data)}"
    )


@step(then, parsers.parse("the number of invalid parameters is exactly {number_of_invalid:d}"))
def then_the_number_of_invalid_is_exactly(
    context: ContextOfTest,
    number_of_invalid: int,
) -> None:
    latest_context = next(
        iter(context.container[JournalingEngineHooks].latest_context_per_trace_id.values())
    )
    invalid_data = latest_context.state.tool_insights.invalid_data

    assert len(invalid_data) == number_of_invalid, (
        f"Expected {number_of_invalid} missing parameters, but found {len(invalid_data)}"
    )


def _get_staged_events(context: ContextOfTest) -> list[EmittedEvent]:
    return next(
        iter(context.container[JournalingEngineHooks].latest_context_per_trace_id.values())
    ).state.tool_events


@step(then, "a single event is staged")
def then_a_single_event_is_staged(
    context: ContextOfTest,
) -> None:
    staged_events = _get_staged_events(context)

    assert len(staged_events) == 1, f"Expected 1 staged event, but found {len(staged_events)}"


@step(then, parsers.parse("the staged event contains {number_of_tool_calls:d} tool call(s)"))
def then_the_staged_event_contains_n_tool_calls(
    context: ContextOfTest,
    number_of_tool_calls: int,
) -> None:
    staged_tool_events = _get_staged_events(context)
    assert number_of_tool_calls == len(
        cast(ToolEventData, staged_tool_events[0].data)["tool_calls"]
    ), pformat(staged_tool_events, indent=2)


@step(then, parsers.parse("the staged tool calls event contains {expected_content}"))
def then_the_tool_calls_staged_event_contains_expected_content(
    context: ContextOfTest,
    expected_content: str,
) -> None:
    staged_tool_events = _get_staged_events(context)
    tool_calls = cast(ToolEventData, staged_tool_events[0].data)["tool_calls"]

    assert context.sync_await(
        nlp_test(
            context=f"The following is the result of tool (function) calls: {tool_calls}",
            condition=f"The calls contain {expected_content}",
        )
    ), pformat(tool_calls, indent=2)
