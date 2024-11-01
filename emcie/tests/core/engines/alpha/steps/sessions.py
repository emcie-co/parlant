from datetime import datetime, timezone
import json
from pytest_bdd import given, parsers

from emcie.server.core.agents import Agent, AgentId
from emcie.server.core.end_users import EndUser, EndUserId
from emcie.server.core.sessions import Session, SessionId, SessionStore

from tests.core.engines.alpha.utils import ContextOfTest, step


@step(given, "an empty session", target_fixture="session_id")
def given_an_empty_session(
    context: ContextOfTest,
    agent_id: AgentId,
) -> SessionId:
    store = context.container[SessionStore]
    utc_now = datetime.now(timezone.utc)
    session = context.sync_await(
        store.create_session(
            creation_utc=utc_now,
            end_user_id=EndUserId("test_user"),
            agent_id=agent_id,
        )
    )
    return session.id


@step(given, "a session with a single user message", target_fixture="session_id")
def given_a_session_with_a_single_user_message(
    context: ContextOfTest,
    new_session: Session,
    end_user: EndUser,
) -> SessionId:
    store = context.container[SessionStore]

    context.sync_await(
        store.create_event(
            session_id=new_session.id,
            source="end_user",
            kind="message",
            correlation_id="test_correlation_id",
            data={
                "message": "Hey there",
                "participant": {
                    "id": end_user.id,
                    "display_name": end_user.name,
                },
            },
        )
    )

    return new_session.id


@step(given, "a session with a thirsty user", target_fixture="session_id")
def given_a_session_with_a_thirsty_user(
    context: ContextOfTest,
    new_session: Session,
    end_user: EndUser,
) -> SessionId:
    store = context.container[SessionStore]

    context.sync_await(
        store.create_event(
            session_id=new_session.id,
            source="end_user",
            kind="message",
            correlation_id="test_correlation_id",
            data={
                "message": "I'm thirsty",
                "participant": {
                    "id": end_user.id,
                    "display_name": end_user.name,
                },
            },
        )
    )

    return new_session.id


@step(given, "a session with a few messages", target_fixture="session_id")
def given_a_session_with_a_few_messages(
    context: ContextOfTest,
    new_session: Session,
    agent: Agent,
    end_user: EndUser,
) -> SessionId:
    store = context.container[SessionStore]

    messages = [
        {
            "source": "end_user",
            "message": "hey there",
        },
        {
            "source": "ai_agent",
            "message": "Hi, how can I help you today?",
        },
        {
            "source": "end_user",
            "message": "What was the first name of the famous Einstein?",
        },
    ]

    for m in messages:
        context.sync_await(
            store.create_event(
                session_id=new_session.id,
                source=m["source"] == "ai_agent" and "ai_agent" or "end_user",
                kind="message",
                correlation_id="test_correlation_id",
                data={
                    "message": m["message"],
                    "participant": {
                        "end_user": {
                            "id": end_user.id,
                            "display_name": end_user.name,
                        },
                        "ai_agent": {
                            "id": agent.id,
                            "display_name": agent.name,
                        },
                    }[m["source"]],
                },
            )
        )

    return new_session.id


@step(
    given,
    parsers.parse("a tool event with data, {tool_event_data}"),
    target_fixture="session_id",
)
def given_a_session_with_tool_event(
    context: ContextOfTest,
    session_id: SessionId,
    tool_event_data: str,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    context.sync_await(
        store.create_event(
            session_id=session.id,
            source="ai_agent",
            kind="tool",
            correlation_id="test_correlation_id",
            data=json.loads(tool_event_data),
        )
    )

    return session.id
