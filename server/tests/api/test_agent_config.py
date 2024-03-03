# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co
from typing import Any, Dict, List
from fastapi.testclient import TestClient
from fastapi import status
from pytest import fixture
import time

from emcie.server.api.agents import ReactionDTO
from emcie.server.api.threads import MessageDTO
from emcie.server.api.agents import ReactionMiddleware, ReactionMiddlewareLink
from emcie.server.threads import Message


def get_message(
    client: TestClient,
    thread_id: str,
    message_id: str,
) -> MessageDTO:
    response = client.get(f"/threads/{thread_id}/messages/{message_id}")
    assert response.status_code == status.HTTP_200_OK
    return MessageDTO(**response.json()["message"])


async def base_reaction_middleware(next: ReactionMiddlewareLink) -> Message:
    message = await next()

    return Message(
        id=message.id,
        thread_id=message.thread_id,
        role=message.role,
        content="$ " + message.content,
        completed=message.completed,
        creation_utc=message.creation_utc,
        revision=message.revision,
    )


@fixture
def reaction_middlewares() -> List[ReactionMiddleware]:
    async def f(next: ReactionMiddlewareLink) -> Message:
        message = await next()

        return Message(
            id=message.id,
            thread_id=message.thread_id,
            role=message.role,
            content=message.content.format(prefix="Message"),
            completed=message.completed,
            creation_utc=message.creation_utc,
            revision=message.revision,
        )

    return [f, base_reaction_middleware]


@fixture
async def app_configuration(reaction_middlewares: ReactionMiddleware) -> Dict[str, Any]:
    return {
        "reaction_middlewares": reaction_middlewares,
        "skills": {
            "multiply_numbers": {
                "name": "multiply_numbers",
                "description": "multiply two numbers",
                "module_path": "emcie.server.skills.multiply",
                "parameters": {
                    "first_number": {
                        "type": "number",
                        "description": "the first number",
                    },
                    "second_number": {
                        "type": "number",
                        "description": "the second number",
                    },
                },
                "required": ["first_number", "second_number"],
            },
            "pizza_toppings": {
                "name": "pizza_toppings",
                "description": "gets the types of toppings we have on pizzas",
                "module_path": "emcie.server.skills.pizza_toppings",
                "parameters": {},
                "required": [],
            },
        },
        "rules": [
            {"when": "always", "then": "start math related answers with '{prefix}: '"},
        ],
    }


def test_that_agent_config_can_be_loaded_and_used(
    client: TestClient,
    agent_id: str,
    multiplication_thread_id: str,
    test_config: Dict[str, Any],
) -> None:
    response = client.post(
        f"/agents/{agent_id}/reactions",
        json={
            "thread_id": multiplication_thread_id,
        },
    )

    assert response.status_code == status.HTTP_200_OK

    reaction = ReactionDTO(**response.json()["reaction"])

    assert reaction.thread_id == multiplication_thread_id
    assert reaction.message_id

    for _ in range(test_config["patience"]):
        message = get_message(
            client,
            thread_id=multiplication_thread_id,
            message_id=reaction.message_id,
        )

        if message.completed:
            break
        else:
            time.sleep(1)

    assert message.completed
    assert len(message.content) > 0
    assert message.content.startswith("$ Message: ")

    response = client.get(f"/threads/{multiplication_thread_id}/messages")
    assert response.status_code == status.HTTP_200_OK
    messages = response.json()["messages"]
    assert len(messages) == 2
