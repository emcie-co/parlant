from typing import Any
from fastapi.testclient import TestClient
from fastapi import status
from lagom import Container
from pytest import mark

from emcie.server.core.agents import AgentStore


def test_that_an_agent_can_be_created_without_description(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"name": "test-agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()["agent"]

    assert agent["name"] == "test-agent"
    assert agent["description"] is None


def test_that_an_agent_can_be_created_with_description(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"name": "test-agent", "description": "You are a test agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()["agent"]

    assert agent["name"] == "test-agent"
    assert agent["description"] == "You are a test agent"


def test_that_an_agent_can_be_created_without_max_engine_iterations(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"name": "test-agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()["agent"]

    assert agent["name"] == "test-agent"
    assert agent["max_engine_iterations"] == 3


def test_that_an_agent_can_be_created_with_max_engine_iterations(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"name": "test-agent", "max_engine_iterations": 1},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()["agent"]

    assert agent["name"] == "test-agent"
    assert agent["max_engine_iterations"] == 1


@mark.parametrize(
    "patch_request",
    (
        {"description": None},
        {"description": "You are a test agent"},
        {"description": "You are a test agent", "max_engine_iterations": 1},
        {"max_engine_iterations": 1},
    ),
)
async def test_that_agent_can_be_updated(
    client: TestClient,
    container: Container,
    patch_request: dict[str, Any],
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent("test-agent")

    patch_response = client.patch(
        f"/agents/{agent.id}",
        json=patch_request,
    )
    assert patch_response.status_code == status.HTTP_204_NO_CONTENT

    response = client.get("/agents")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data["agents"]) == 1
    agent_dto = data["agents"][0]

    assert agent_dto["description"] == patch_request.get("description")
    assert agent_dto["max_engine_iterations"] == patch_request.get("max_engine_iterations", 3)
