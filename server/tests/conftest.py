from typing import Any, AsyncIterator, Dict
from fastapi import status
from fastapi.testclient import TestClient
from lagom import Container
from pytest import fixture, Config

from emcie.server import main
from emcie.server.agents import AgentStore
from emcie.server.models import ModelRegistry
from emcie.server.sessions import SessionStore
from emcie.server.threads import ThreadStore


@fixture
def test_config(pytestconfig: Config) -> Dict[str, Any]:
    return {"patience": 10}


@fixture
def container() -> Container:
    container = Container()

    container[AgentStore] = AgentStore()
    container[ThreadStore] = ThreadStore()
    container[SessionStore] = SessionStore()
    container[ModelRegistry] = ModelRegistry()

    return container


@fixture
async def client(container: Container) -> AsyncIterator[TestClient]:
    app = await main.create_app(container)

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
