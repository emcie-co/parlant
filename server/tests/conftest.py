# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co

from typing import Any, AsyncIterator, Dict
from fastapi import status
from fastapi.testclient import TestClient
from pytest import fixture, Config
from tinydb import TinyDB
from tinydb.storages import MemoryStorage
from emcie.server import main
from emcie.server.embedders import OpenAIEmbedder
from emcie.server.rag import RagStore


@fixture
def test_config(pytestconfig: Config) -> Dict[str, Any]:
    return {"patience": 10}


@fixture
async def app_configuration() -> Dict[str, Any]:
    return {}


@fixture
def rag_store() -> RagStore:
    return RagStore(TinyDB(storage=MemoryStorage), OpenAIEmbedder())


@fixture
async def client(
    app_configuration: Dict[str, Any],
    rag_store: RagStore,
) -> AsyncIterator[TestClient]:
    app = await main.create_app(rag_store=rag_store, **app_configuration)


def rag_store() -> RagStore:
    return RagStore(TinyDB(storage=MemoryStorage))


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
