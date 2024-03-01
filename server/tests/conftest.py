from typing import Any, AsyncIterator, Dict
from fastapi import status
from fastapi.testclient import TestClient
from pytest import fixture, Config

from emcie.server import main


@fixture
def test_config(pytestconfig: Config) -> Dict[str, Any]:
    return {"patience": 10}


@fixture
async def app_configuration() -> Dict[str, Any]:
    return {}


@fixture
async def client(
    app_configuration: Dict[str, Any],
) -> AsyncIterator[TestClient]:
    app = await main.create_app(**app_configuration)

    with TestClient(app) as client:
        yield client


@fixture
def agent_id(client: TestClient) -> str:
    return str(client.post("/agents").json()["agent_id"])


@fixture
def new_thread_id(client: TestClient) -> str:
    return str(client.post("/threads").json()["thread_id"])


@fixture
def user_question_thread_id(
    client: TestClient,
    new_thread_id: str,
) -> str:
    response = client.post(
        f"/threads/{new_thread_id}/messages",
        json={
            "role": "user",
            "content": "Is 42 a number?",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    return new_thread_id


@fixture
def multiplication_thread_id(
    client: TestClient,
    new_thread_id: str,
) -> str:
    response = client.post(
        f"/threads/{new_thread_id}/messages",
        json={
            "role": "user",
            "content": "What is 1985 * 53.5?",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    return new_thread_id
