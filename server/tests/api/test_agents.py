# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co
from typing import Any, Dict
from fastapi.testclient import TestClient
from fastapi import status
import time

from emcie.server.api.agents import ReactionDTO
from emcie.server.api.threads import MessageDTO


def get_message(
    client: TestClient,
    thread_id: str,
    message_id: str,
) -> MessageDTO:
    response = client.get(f"/threads/{thread_id}/messages/{message_id}")
    assert response.status_code == status.HTTP_200_OK
    return MessageDTO(**response.json()["message"])


def test_that_an_agent_can_be_created(
    client: TestClient,
) -> None:
    response = client.post(
        "/agents",
        json={"id": "test-agent"},
    )

    assert response.status_code == status.HTTP_200_OK

    response = client.get("/agents")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data["agents"]) == 1


def test_that_an_agent_can_respond_to_a_thread(
    client: TestClient,
    agent_id: str,
    user_question_thread_id: str,
    test_config: Dict[str, Any],
) -> None:
    response = client.post(
        f"/agents/{agent_id}/reactions",
        json={
            "thread_id": user_question_thread_id,
        },
    )

    assert response.status_code == status.HTTP_200_OK

    reaction = ReactionDTO(**response.json()["reaction"])

    assert reaction.thread_id == user_question_thread_id
    assert reaction.message_id

    for _ in range(test_config["patience"]):
        message = get_message(
            client,
            thread_id=user_question_thread_id,
            message_id=reaction.message_id,
        )

        if message.completed:
            break
        else:
            time.sleep(1)

    assert message.completed
    assert len(message.content) > 0

    response = client.get(f"/threads/{user_question_thread_id}/messages")
    assert response.status_code == status.HTTP_200_OK
    messages = response.json()["messages"]
    assert len(messages) == 2
