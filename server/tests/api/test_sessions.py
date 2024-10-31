import time
from typing import Any, cast
import dateutil
from fastapi.testclient import TestClient
from fastapi import status
from lagom import Container
from pytest import fixture, mark
from datetime import datetime, timezone

from emcie.common.tools import ToolResult
from emcie.server.core.agents import AgentId
from emcie.server.core.async_utils import Timeout
from emcie.server.core.end_users import EndUserId
from emcie.server.core.sessions import EventSource, MessageEventData, SessionId, SessionStore
from tests.api.utils import (
    create_agent,
    create_context_variable,
    create_end_user,
    create_guideline,
    create_session,
    create_term,
    post_message,
    read_reply,
    set_context_variable_value,
)


@fixture
async def long_session_id(
    client: TestClient,
    container: Container,
    session_id: SessionId,
) -> SessionId:
    await populate_session_id(
        container,
        session_id,
        [
            make_event_params("end_user"),
            make_event_params("ai_agent"),
            make_event_params("end_user"),
            make_event_params("ai_agent"),
            make_event_params("ai_agent"),
            make_event_params("end_user"),
        ],
    )

    return session_id


def make_event_params(
    source: EventSource,
    data: dict[str, Any] = {},
    kind: str = "custom",
) -> dict[str, Any]:
    return {
        "source": source,
        "kind": kind,
        "creation_utc": str(datetime.now(timezone.utc)),
        "correlation_id": "dummy_correlation_id",
        "data": data,
        "deleted": False,
    }


async def populate_session_id(
    container: Container,
    session_id: SessionId,
    events: list[dict[str, Any]],
) -> None:
    session_store = container[SessionStore]

    for e in events:
        await session_store.create_event(
            session_id=session_id,
            source=e["source"],
            kind=e["kind"],
            correlation_id=e["correlation_id"],
            data=e["data"],
        )


def event_is_according_to_params(
    event: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    tested_properties = ["source", "kind", "data"]

    for p in tested_properties:
        if event[p] != params[p]:
            return False

    return True


def get_cow_uttering() -> ToolResult:
    return ToolResult("moo")


###############################################################################
## Session CRUD API
###############################################################################


def test_that_a_session_can_be_created_without_a_title(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    response = client.post(
        "/sessions",
        json={
            "end_user_id": "test_user",
            "agent_id": agent_id,
        },
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()["session"]

    assert "id" in data
    assert "agent_id" in data
    assert data["agent_id"] == agent_id
    assert "title" in data
    assert data["title"] is None


def test_that_a_session_can_be_created_with_title(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    title = "Test Session Title"

    response = client.post(
        "/sessions",
        json={
            "end_user_id": "test_user",
            "agent_id": agent_id,
            "title": title,
        },
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()["session"]

    assert "id" in data
    assert "agent_id" in data
    assert data["agent_id"] == agent_id
    assert data["title"] == title


def test_that_a_created_session_has_meaningful_creation_utc(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    time_before_creation = datetime.now(timezone.utc)

    data = (
        client.post(
            "/sessions",
            json={
                "end_user_id": "test_user",
                "agent_id": agent_id,
            },
        )
        .raise_for_status()
        .json()["session"]
    )

    assert "creation_utc" in data
    creation_utc = dateutil.parser.isoparse(data["creation_utc"])

    time_after_creation = datetime.now(timezone.utc)

    assert time_before_creation <= creation_utc <= time_after_creation, (
        f"Expected creation_utc to be between {time_before_creation} and {time_after_creation}, "
        f"but got {creation_utc}."
    )


async def test_that_a_session_can_be_read(
    client: TestClient,
    container: Container,
) -> None:
    pass


async def test_that_sessions_can_be_listed(
    client: TestClient,
    container: Container,
) -> None:
    agents = [
        await create_agent(container, "first-agent"),
        await create_agent(container, "second-agent"),
    ]

    sessions = [
        await create_session(container, agent_id=agents[0].id, title="first-session"),
        await create_session(container, agent_id=agents[0].id, title="second-session"),
        await create_session(container, agent_id=agents[1].id, title="third-session"),
    ]

    data = client.get("/sessions").raise_for_status().json()

    assert len(data["sessions"]) == len(sessions)

    for listed_session, created_session in zip(data["sessions"], sessions):
        assert listed_session["title"] == created_session.title
        assert listed_session["end_user_id"] == created_session.end_user_id


async def test_that_sessions_can_be_listed_by_agent_id(
    client: TestClient,
    container: Container,
) -> None:
    agents = [
        await create_agent(container, "first-agent"),
        await create_agent(container, "second-agent"),
    ]

    sessions = [
        await create_session(container, agent_id=agents[0].id, title="first-session"),
        await create_session(container, agent_id=agents[0].id, title="second-session"),
        await create_session(container, agent_id=agents[1].id, title="third-session"),
    ]

    for agent in agents:
        agent_sessions = [s for s in sessions if s.agent_id == agent.id]

        data = client.get("/sessions", params={"agent_id": agent.id}).raise_for_status().json()

        assert len(data["sessions"]) == len(agent_sessions)

        for listed_session, created_session in zip(data["sessions"], agent_sessions):
            assert listed_session["agent_id"] == agent.id
            assert listed_session["title"] == created_session.title
            assert listed_session["end_user_id"] == created_session.end_user_id


async def test_that_sessions_can_be_listed_by_end_user_id(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    _ = await create_session(container, agent_id=agent_id, title="first-session")
    _ = await create_session(container, agent_id=agent_id, title="second-session")
    _ = await create_session(
        container, agent_id=agent_id, title="three-session", end_user_id=EndUserId("Joe")
    )

    data = client.get("/sessions", params={"end_user_id": "Joe"}).raise_for_status().json()

    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["end_user_id"] == "Joe"


def test_that_a_session_is_created_with_zeroed_out_consumption_offsets(
    client: TestClient,
    long_session_id: SessionId,
) -> None:
    data = client.get(f"/sessions/{long_session_id}").raise_for_status().json()

    assert "consumption_offsets" in data
    assert "client" in data["consumption_offsets"]
    assert data["consumption_offsets"]["client"] == 0


@mark.parametrize("consumer_id", ["client"])
def test_that_consumption_offsets_can_be_updated(
    client: TestClient,
    long_session_id: SessionId,
    consumer_id: str,
) -> None:
    response = client.patch(
        f"/sessions/{long_session_id}",
        json={
            "consumption_offsets": {
                consumer_id: 1,
            }
        },
    )
    assert response.status_code == status.HTTP_204_NO_CONTENT

    data = client.get(f"/sessions/{long_session_id}").raise_for_status().json()

    assert data["consumption_offsets"][consumer_id] == 1


def test_that_title_can_be_updated(
    client: TestClient,
    session_id: SessionId,
) -> None:
    response = client.patch(
        f"/sessions/{session_id}",
        json={"title": "new session title"},
    )
    assert response.status_code == status.HTTP_204_NO_CONTENT

    data = client.get(f"/sessions/{session_id}").raise_for_status().json()

    assert data["title"] == "new session title"


def test_that_deleting_a_nonexistent_session_returns_404(
    client: TestClient,
) -> None:
    response = client.delete("/sessions/nonexistent-session-id")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_that_a_session_can_be_deleted(
    client: TestClient,
    session_id: SessionId,
) -> None:
    delete_response = client.delete(f"/sessions/{session_id}").raise_for_status().json()
    assert delete_response["session_id"] == session_id

    get_response = client.get(f"/sessions/{session_id}")
    assert get_response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_deleted_session_is_removed_from_the_session_list(
    client: TestClient,
    session_id: SessionId,
) -> None:
    sessions = client.get("/sessions").raise_for_status().json()["sessions"]
    assert any(session["id"] == str(session_id) for session in sessions)

    client.delete(f"/sessions/{session_id}").raise_for_status()

    sessions_after_deletion = client.get("/sessions").raise_for_status().json()["sessions"]
    assert not any(session["session_id"] == str(session_id) for session in sessions_after_deletion)


async def test_that_all_sessions_can_be_deleted_with_one_request(
    client: TestClient,
    agent_id: AgentId,
    container: Container,
) -> None:
    for _ in range(5):
        await create_session(
            container=container,
            agent_id=agent_id,
            end_user_id=EndUserId("test-user"),
        )

    response = client.delete("/sessions", params={"agent_id": agent_id})

    assert response.status_code == status.HTTP_204_NO_CONTENT

    stored_sessions = await container[SessionStore].list_sessions(agent_id)

    assert len(stored_sessions) == 0


async def test_that_deleting_a_session_also_deletes_its_events(
    client: TestClient,
    container: Container,
    session_id: SessionId,
) -> None:
    session_events = [
        make_event_params("end_user"),
        make_event_params("ai_agent"),
    ]

    await populate_session_id(container, session_id, session_events)

    events = client.get(f"/sessions/{session_id}/events").raise_for_status().json()["events"]
    assert len(events) == len(session_events)

    client.delete(f"/sessions/{session_id}").raise_for_status()

    events_after_deletion = client.get(f"/sessions/{session_id}/events")
    assert events_after_deletion.status_code == status.HTTP_404_NOT_FOUND


###############################################################################
## Events CRUD API
###############################################################################


async def test_that_events_can_be_listed(
    client: TestClient,
    container: Container,
    session_id: SessionId,
) -> None:
    session_events = [
        make_event_params("end_user"),
        make_event_params("ai_agent"),
        make_event_params("ai_agent"),
        make_event_params("end_user"),
        make_event_params("ai_agent"),
    ]

    await populate_session_id(container, session_id, session_events)

    data = client.get(f"/sessions/{session_id}/events").raise_for_status().json()

    assert data["session_id"] == session_id
    assert len(data["events"]) == len(session_events)

    for i, (event_params, listed_event) in enumerate(zip(session_events, data["events"])):
        assert listed_event["offset"] == i
        assert event_is_according_to_params(event=listed_event, params=event_params)


@mark.parametrize("offset", (0, 2, 4))
async def test_that_events_can_be_filtered_by_offset(
    client: TestClient,
    container: Container,
    session_id: SessionId,
    offset: int,
) -> None:
    session_events = [
        make_event_params("end_user"),
        make_event_params("ai_agent"),
        make_event_params("end_user"),
        make_event_params("ai_agent"),
        make_event_params("end_user"),
    ]

    await populate_session_id(container, session_id, session_events)

    retrieved_events = (
        client.get(
            f"/sessions/{session_id}/events",
            params={
                "min_offset": offset,
            },
        )
        .raise_for_status()
        .json()["events"]
    )

    for event_params, listed_event in zip(session_events, retrieved_events):
        assert event_is_according_to_params(event=listed_event, params=event_params)


def test_that_posting_problematic_messages_with_moderation_enabled_causes_them_to_be_flagged_and_tagged_as_such(
    client: TestClient,
    session_id: SessionId,
) -> None:
    response = client.post(
        f"/sessions/{session_id}/events",
        params={"moderation": "auto"},
        json={"content": "Fuck all those guys"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    event = response.json()["event"]

    assert event["data"].get("flagged")
    assert "harassment" in event["data"].get("tags")


def test_that_posting_a_message_elicits_a_response(
    client: TestClient,
    session_id: SessionId,
) -> None:
    response = client.post(
        f"/sessions/{session_id}/events",
        json={"content": "Hello there!"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    event = response.json()["event"]

    events_in_session = (
        client.get(
            f"/sessions/{session_id}/events",
            params={
                "min_offset": event["offset"] + 1,
                "kinds": "message",
                "wait": True,
            },
        )
        .raise_for_status()
        .json()["events"]
    )

    assert events_in_session


async def test_that_status_updates_can_be_retrieved_separately_after_posting_a_message(
    client: TestClient,
    container: Container,
    session_id: SessionId,
) -> None:
    event = await post_message(
        container=container,
        session_id=session_id,
        message="Hello there!",
        response_timeout=Timeout(30),
    )

    events = (
        client.get(
            f"/sessions/{session_id}/events",
            params={
                "min_offset": event.offset + 1,
                "kinds": "status",
                "wait": True,
            },
        )
        .raise_for_status()
        .json()["events"]
    )

    assert events
    assert all(e["kind"] == "status" for e in events)


def test_that_not_waiting_for_a_response_does_in_fact_return_immediately(
    client: TestClient,
    session_id: SessionId,
) -> None:
    posted_event = (
        client.post(
            f"/sessions/{session_id}/events",
            json={"content": "Hello there!"},
        )
        .raise_for_status()
        .json()
    )

    t_start = time.time()

    client.get(
        f"/sessions/{session_id}/events",
        params={
            "min_offset": posted_event["event"]["offset"] + 1,
            "wait": False,
        },
    )

    t_end = time.time()

    assert (t_end - t_start) < 1


async def test_that_tool_events_are_correlated_with_message_events(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    session_id: SessionId,
) -> None:
    await create_guideline(
        container=container,
        agent_id=agent_id,
        predicate="a user says hello",
        action="answer like a cow",
        tool_function=get_cow_uttering,
    )

    event = await post_message(
        container=container,
        session_id=session_id,
        message="Hello there!",
        response_timeout=Timeout(30),
    )

    events_in_session = (
        client.get(
            f"/sessions/{session_id}/events",
            params={"min_offset": event.offset + 1},
        )
        .raise_for_status()
        .json()["events"]
    )

    message_event = next(e for e in events_in_session if e["kind"] == "message")
    tool_call_event = next(e for e in events_in_session if e["kind"] == "tool")
    assert message_event["correlation_id"] == tool_call_event["correlation_id"]


async def test_that_deleted_events_no_longer_show_up_in_the_listing(
    client: TestClient,
    container: Container,
    session_id: SessionId,
) -> None:
    session_events = [
        make_event_params("end_user"),
        make_event_params("ai_agent"),
        make_event_params("end_user"),
        make_event_params("ai_agent"),
        make_event_params("end_user"),
    ]
    await populate_session_id(container, session_id, session_events)

    initial_events = (
        client.get(f"/sessions/{session_id}/events").raise_for_status().json()["events"]
    )
    assert len(initial_events) == len(session_events)

    event_to_delete = initial_events[1]
    deleted_event_ids = (
        client.delete(f"/sessions/{session_id}/events?min_offset={event_to_delete['offset']}")
        .raise_for_status()
        .json()["event_ids"]
    )

    for d_id, e in zip(deleted_event_ids, initial_events[1:]):
        assert d_id == e["id"]

    remaining_events = (
        client.get(f"/sessions/{session_id}/events").raise_for_status().json()["events"]
    )

    assert len(remaining_events) == 1
    assert event_is_according_to_params(remaining_events[0], session_events[0])

    assert all(e["offset"] > event_to_delete["offset"] for e in remaining_events) is False


###############################################################################
## Interaction API
###############################################################################


def test_that_no_interaction_is_found_for_an_empty_session(
    client: TestClient,
    session_id: SessionId,
) -> None:
    data = (
        client.get(
            f"/sessions/{session_id}/interactions",
            params={
                "min_event_offset": 0,
                "source": "ai_agent",
            },
        )
        .raise_for_status()
        .json()
    )

    assert data["session_id"] == session_id
    assert len(data["interactions"]) == 0


async def test_that_a_server_interaction_is_found_for_a_session_with_a_user_message(
    client: TestClient,
    container: Container,
    session_id: SessionId,
) -> None:
    event = await post_message(
        container=container,
        session_id=session_id,
        message="Hello there!",
        response_timeout=Timeout(30),
    )

    interactions = (
        client.get(
            f"/sessions/{session_id}/interactions",
            params={
                "min_event_offset": event.offset,
                "source": "ai_agent",
                "wait": True,
            },
        )
        .raise_for_status()
        .json()
    )["interactions"]

    assert len(interactions) == 1
    assert interactions[0]["source"] == "ai_agent"
    assert interactions[0]["kind"] == "message"
    assert isinstance(interactions[0]["data"], str)
    assert len(interactions[0]["data"]) > 0


async def test_that_a_message_interaction_can_be_inspected_using_the_message_event_id(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    end_user = await create_end_user(
        container=container,
        name="John Smith",
    )

    session = await create_session(
        container=container,
        agent_id=agent_id,
        end_user_id=end_user.id,
    )

    guideline = await create_guideline(
        container=container,
        agent_id=agent_id,
        predicate="a user mentions cows",
        action="answer like a cow while mentioning the user's full name",
        tool_function=get_cow_uttering,
    )

    term = await create_term(
        container=container,
        agent_id=agent_id,
        name="Flubba",
        description="A type of cow",
        synonyms=["Bobo"],
    )

    context_variable = await create_context_variable(
        container=container,
        agent_id=agent_id,
        name="User full name",
    )

    await set_context_variable_value(
        container=container,
        agent_id=agent_id,
        variable_id=context_variable.id,
        key=session.end_user_id,
        data=end_user.name,
    )

    user_event = await post_message(
        container=container,
        session_id=session.id,
        message="Bobo!",
        response_timeout=Timeout(60),
    )

    reply_event = await read_reply(
        container=container,
        session_id=session.id,
        user_event_offset=user_event.offset,
    )

    inspection_data = (
        client.get(f"/sessions/{session.id}/interactions/{reply_event.correlation_id}")
        .raise_for_status()
        .json()
    )

    assert end_user.name in cast(MessageEventData, reply_event.data)["message"]

    iterations = inspection_data["preparation_iterations"]
    assert len(iterations) >= 1

    assert len(iterations[0]["guideline_propositions"]) == 1
    assert iterations[0]["guideline_propositions"][0]["guideline_id"] == guideline.id
    assert iterations[0]["guideline_propositions"][0]["predicate"] == guideline.content.predicate
    assert iterations[0]["guideline_propositions"][0]["action"] == guideline.content.action

    assert len(iterations[0]["tool_calls"]) == 1
    assert "get_cow_uttering" in iterations[0]["tool_calls"][0]["tool_id"]
    assert iterations[0]["tool_calls"][0]["result"]["data"] == "moo"

    assert len(iterations[0]["terms"]) == 1
    assert iterations[0]["terms"][0]["name"] == term.name
    assert iterations[0]["terms"][0]["description"] == term.description
    assert iterations[0]["terms"][0]["synonyms"] == term.synonyms

    assert len(iterations[0]["context_variables"]) == 1
    assert iterations[0]["context_variables"][0]["name"] == context_variable.name
    assert iterations[0]["context_variables"][0]["key"] == end_user.id
    assert iterations[0]["context_variables"][0]["value"] == end_user.name
