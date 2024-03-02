# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co
from datetime import datetime
from dateutil import parser
from fastapi.testclient import TestClient
from fastapi import status
from pytest import fixture, mark

from emcie.server.api.threads import MessageDTO


@fixture
async def new_assistant_message(
    new_thread_id: str,
    client: TestClient,
) -> MessageDTO:
    response = client.post(
        f"/threads/{new_thread_id}/messages",
        json={
            "role": "assistant",
            "content": "",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    message = response.json()["message"]

    return MessageDTO(**message)


def test_that_a_thread_can_be_created(
    client: TestClient,
) -> None:
    response = client.post("/threads")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "thread_id" in data


@mark.parametrize("role", ("user", "assistant"))
def test_that_a_user_message_can_be_added_to_a_thread(
    client: TestClient,
    new_thread_id: str,
    role: str,
) -> None:
    before_creating_the_message = datetime.utcnow()

    response = client.post(
        f"/threads/{new_thread_id}/messages",
        json={
            "role": role,
            "content": "Hello",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    response = client.get(f"/threads/{new_thread_id}/messages")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data["messages"]) == 1
    assert data["messages"][0]["role"] == role
    assert data["messages"][0]["content"] == "Hello"
    assert parser.parse(data["messages"][0]["creation_utc"]) >= before_creating_the_message


def test_that_an_assistant_message_can_be_updated_with_new_tokens(
    client: TestClient,
    new_thread_id: str,
    new_assistant_message: MessageDTO,
) -> None:
    response = client.patch(
        f"/threads/{new_thread_id}/messages/{new_assistant_message.id}",
        json={
            "target_revision": new_assistant_message.revision,
            "content_delta": "Hello",
            "completed": False,
        },
    )

    assert response.status_code == status.HTTP_200_OK

    response = client.get(f"/threads/{new_thread_id}/messages")

    message = MessageDTO(**response.json()["messages"][0])

    assert message.content == "Hello"
    assert message.revision == (new_assistant_message.revision + 1)


def test_that_an_assistant_message_cannot_be_updated_when_the_target_revision_isnt_the_current_one(  # noqa
    client: TestClient,
    new_thread_id: str,
    new_assistant_message: MessageDTO,
) -> None:
    response = client.patch(
        f"/threads/{new_thread_id}/messages/{new_assistant_message.id}",
        json={
            "target_revision": new_assistant_message.revision + 1,
            "content_delta": "Hello",
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_that_a_specific_message_can_be_read(
    client: TestClient,
    new_thread_id: str,
    new_assistant_message: MessageDTO,
) -> None:
    response = client.get(f"/threads/{new_thread_id}/messages/{new_assistant_message.id}")

    assert response.status_code == status.HTTP_200_OK

    read_message = MessageDTO(**response.json()["message"])

    assert read_message == new_assistant_message
