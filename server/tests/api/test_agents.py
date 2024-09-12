from fastapi.testclient import TestClient
from fastapi import status


def test_that_an_agent_can_be_created_without_description(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"agent_name": "test-agent"},
    )

    assert response.status_code == status.HTTP_200_OK

    response = client.get("/agents")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data["agents"]) == 1
    assert data["agents"][0]["name"] == "test-agent"
    assert data["agents"][0]["description"] is None


def test_that_an_agent_can_be_created_with_description(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"agent_name": "test-agent", "agent_description": "You are a test agent"},
    )

    assert response.status_code == status.HTTP_200_OK

    response = client.get("/agents")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data["agents"]) == 1
    assert data["agents"][0]["name"] == "test-agent"
    assert data["agents"][0]["description"] == "You are a test agent"
