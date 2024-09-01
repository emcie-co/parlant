import time
from typing import Any, Callable, Optional
import dateutil
from fastapi.testclient import TestClient
from fastapi import status
from lagom import Container
from pytest import fixture, mark
from datetime import datetime, timezone

from emcie.common.tools import ToolId, ToolResult
from emcie.server.core.agents import AgentId
from emcie.server.core.guideline_tool_associations import GuidelineToolAssociationStore
from emcie.server.core.guidelines import GuidelineStore
from emcie.server.core.sessions import EventSource, SessionId, SessionStore
from emcie.server.core.tools import LocalToolService


def _get_cow_uttering() -> ToolResult:
    return ToolResult("moo")


class ToolFunctions:
    GET_COW_UTTERING = _get_cow_uttering


@fixture
def agent_id(client: TestClient) -> AgentId:
    response = client.post(
        "/agents",
        json={"agent_name": "test-agent"},
    )
    return AgentId(response.json()["agent_id"])


@fixture
def session_id(
    client: TestClient,
    agent_id: AgentId,
) -> SessionId:
    response = client.post(
        "/sessions?allow_greeting=False",
        json={
            "end_user_id": "test_user",
            "agent_id": agent_id,
        },
    ).raise_for_status()
    return SessionId(response.json()["session_id"])


@fixture
async def long_session_id(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> SessionId:
    response = client.post(
        "/sessions?allow_greeting=False",
        json={
            "end_user_id": "test_user",
            "agent_id": agent_id,
        },
    ).raise_for_status()
    session_id = SessionId(response.json()["session_id"])

    await populate_session_id(
        container,
        session_id,
        [
            make_event_params("client"),
            make_event_params("server"),
            make_event_params("client"),
            make_event_params("server"),
            make_event_params("server"),
            make_event_params("client"),
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
    }


async def add_guideline(
    container: Container,
    agent_id: AgentId,
    predicate: str,
    action: str,
    tool_function: Optional[Callable[[], ToolResult]] = None,
) -> None:
    guideline = await container[GuidelineStore].create_guideline(
        guideline_set=agent_id,
        predicate=predicate,
        action=action,
    )

    if tool_function:
        tool = await container[LocalToolService].create_tool(
            name=tool_function.__name__,
            module_path=tool_function.__module__,
            description="",
            parameters={},
            required=[],
        )

        await container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId(f"local__{tool.id}"),
        )


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


def test_that_a_session_can_be_created(
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
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "session_id" in data
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
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "session_id" in data
    assert data["title"] == title


def test_that_session_has_meaningful_creation_utc(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    time_before_creation = datetime.now(timezone.utc)

    response = client.post(
        "/sessions",
        json={
            "end_user_id": "test_user",
            "agent_id": agent_id,
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "creation_utc" in data
    creation_utc = dateutil.parser.isoparse(data["creation_utc"])

    time_after_creation = datetime.now(timezone.utc)

    assert time_before_creation <= creation_utc <= time_after_creation, (
        f"Expected creation_utc to be between {time_before_creation} and {time_after_creation}, "
        f"but got {creation_utc}."
    )


def test_that_sessions_can_be_listed(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"agent_name": "first_test-agent"},
    )
    first_agent_id = AgentId(response.json()["agent_id"])

    response = client.post(
        "/agents",
        json={"agent_name": "second_test-agent"},
    )
    second_agent_id = AgentId(response.json()["agent_id"])

    client.post(
        "/sessions",
        json={
            "end_user_id": "test_user1",
            "agent_id": first_agent_id,
            "title": "Test Session1 Title",
        },
    )

    client.post(
        "/sessions",
        json={
            "end_user_id": "test_user2",
            "agent_id": first_agent_id,
            "title": "Test Session2 Title",
        },
    )

    client.post(
        "/sessions",
        json={
            "end_user_id": "test_user3",
            "agent_id": second_agent_id,
            "title": "Test Session3 Title",
        },
    )

    response = client.get("/sessions")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data

    sessions = data["sessions"]

    assert len(sessions) == 3

    assert sessions[0]["title"] == "Test Session1 Title"
    assert sessions[1]["title"] == "Test Session2 Title"
    assert sessions[2]["title"] == "Test Session3 Title"

    assert sessions[0]["end_user_id"] == "test_user1"
    assert sessions[1]["end_user_id"] == "test_user2"
    assert sessions[2]["end_user_id"] == "test_user3"


def test_that_sessions_can_be_listed_by_agent_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"agent_name": "first_test-agent"},
    )
    first_agent_id = AgentId(response.json()["agent_id"])

    response = client.post(
        "/agents",
        json={"agent_name": "second_test-agent"},
    )
    second_agent_id = AgentId(response.json()["agent_id"])

    client.post(
        "/sessions",
        json={
            "end_user_id": "test_user1",
            "agent_id": first_agent_id,
            "title": "Test Session1 Title",
        },
    )

    client.post(
        "/sessions",
        json={
            "end_user_id": "test_user2",
            "agent_id": first_agent_id,
            "title": "Test Session2 Title",
        },
    )

    client.post(
        "/sessions",
        json={
            "end_user_id": "test_user3",
            "agent_id": second_agent_id,
            "title": "Test Session3 Title",
        },
    )

    response = client.get(f"/sessions/?agent_id={first_agent_id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data

    sessions = data["sessions"]

    assert len(sessions) == 2

    assert sessions[0]["title"] == "Test Session1 Title"
    assert sessions[1]["title"] == "Test Session2 Title"

    assert sessions[0]["end_user_id"] == "test_user1"
    assert sessions[1]["end_user_id"] == "test_user2"

    response = client.get(f"/sessions/?agent_id={second_agent_id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data

    sessions = data["sessions"]

    assert len(sessions) == 1

    assert sessions[0]["title"] == "Test Session3 Title"

    assert sessions[0]["end_user_id"] == "test_user3"


async def test_that_events_can_be_listed(
    client: TestClient,
    container: Container,
    session_id: SessionId,
) -> None:
    session_events = [
        make_event_params("client"),
        make_event_params("server"),
        make_event_params("server"),
        make_event_params("client"),
        make_event_params("server"),
    ]

    await populate_session_id(container, session_id, session_events)

    response = client.get(f"/sessions/{session_id}/events")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["session_id"] == session_id

    assert len(data["events"]) == len(session_events)

    for i, (event_params, listed_event) in enumerate(zip(session_events, data["events"])):
        assert listed_event["offset"] == i
        assert event_is_according_to_params(event=listed_event, params=event_params)


async def test_that_tool_events_are_correlated_with_message_events(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    session_id: SessionId,
) -> None:
    await add_guideline(
        container=container,
        agent_id=agent_id,
        predicate="a user says hello",
        action="answer like a cow",
        tool_function=ToolFunctions.GET_COW_UTTERING,
    )

    posted_event = (
        client.post(
            f"/sessions/{session_id}/events",
            json={"content": "Hello there!"},
        )
        .raise_for_status()
        .json()
    )

    events_in_session = (
        client.get(
            f"/sessions/{session_id}/events",
            params={
                "min_offset": posted_event["event_offset"] + 1,
                "wait": True,
            },
        )
        .raise_for_status()
        .json()["events"]
    )

    message_event = next(e for e in events_in_session if e["kind"] == "message")
    tool_call_event = next(e for e in events_in_session if e["kind"] == "tool")
    assert message_event["correlation_id"] == tool_call_event["correlation_id"]


def test_that_a_session_is_created_with_zeroed_out_consumption_offsets(
    client: TestClient,
    long_session_id: SessionId,
) -> None:
    response = client.get(f"/sessions/{long_session_id}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

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

    response = client.get(f"/sessions/{long_session_id}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["consumption_offsets"][consumer_id] == 1


@mark.parametrize("offset", (0, 2, 4))
async def test_that_events_can_be_filtered_by_offset(
    client: TestClient,
    container: Container,
    session_id: SessionId,
    offset: int,
) -> None:
    session_events = [
        make_event_params("client"),
        make_event_params("server"),
        make_event_params("client"),
        make_event_params("server"),
        make_event_params("client"),
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


def test_that_posting_a_message_elicits_a_response(
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

    events_in_session = (
        client.get(
            f"/sessions/{session_id}/events",
            params={
                "min_offset": posted_event["event_offset"] + 1,
                "wait": True,
            },
        )
        .raise_for_status()
        .json()["events"]
    )

    assert events_in_session


def test_that_not_waiting_for_a_response_does_in_fact_return_immediately(
    client: TestClient,
    agent_id: AgentId,
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
            "min_offset": posted_event["event_offset"] + 1,
            "wait": False,
        },
    )

    t_end = time.time()

    assert (t_end - t_start) < 1


def test_that_deleting_a_nonexistent_session_returns_404(
    client: TestClient,
) -> None:
    response = client.delete("/sessions/nonexistent-session-id")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_that_a_session_can_be_deleted(
    client: TestClient,
    session_id: SessionId,
) -> None:
    response = client.delete(f"/sessions/{session_id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    get_response = client.get(f"/sessions/{session_id}")
    assert get_response.status_code == status.HTTP_404_NOT_FOUND


def test_that_a_deleted_session_is_removed_from_the_session_list(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    # Create a session
    session_id = SessionId(
        client.post(
            "/sessions",
            json={
                "end_user_id": "test_user",
                "agent_id": agent_id,
                "title": "Session to be deleted",
            },
        )
        .raise_for_status()
        .json()["session_id"]
    )

    sessions = client.get("/sessions").raise_for_status().json()["sessions"]
    assert any(session["session_id"] == str(session_id) for session in sessions)

    client.delete(f"/sessions/{session_id}").raise_for_status()

    sessions_after_deletion = client.get("/sessions").raise_for_status().json()["sessions"]
    assert not any(session["session_id"] == str(session_id) for session in sessions_after_deletion)


async def test_that_deleting_a_session_also_deletes_its_events(
    client: TestClient,
    container: Container,
    session_id: SessionId,
) -> None:
    session_events = [
        make_event_params("client"),
        make_event_params("server"),
    ]
    await populate_session_id(container, session_id, session_events)

    events = client.get(f"/sessions/{session_id}/events").raise_for_status().json()["events"]
    assert len(events) == len(session_events)

    delete_response = client.delete(f"/sessions/{session_id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    events_response_after = client.get(f"/sessions/{session_id}/events")
    assert events_response_after.status_code == status.HTTP_404_NOT_FOUND
