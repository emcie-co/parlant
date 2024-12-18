# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

from tests.core.common.conftest import REASONABLE_AMOUNT_OF_TIME, ApplicationContextOfTest
from parlant.core.async_utils import Timeout
from parlant.core.agents import AgentId
from parlant.core.sessions import Session, SessionStore
from parlant.core.tools import ToolResult
from tests.test_utilities import create_guideline, nlp_test
from tests.core.common.conftest import (
    application_context,
    customer_id,
    agent_id,
    proactive_agent_id,
    session,
)


async def test_that_a_new_customer_session_can_be_created(
    application_context: ApplicationContextOfTest,
    agent_id: AgentId,
) -> None:
    created_session = await application_context.app.create_customer_session(
        customer_id=application_context.customer_id,
        agent_id=agent_id,
    )

    session_in_db = await application_context.container[SessionStore].read_session(
        created_session.id,
    )

    assert created_session == session_in_db


async def test_that_a_new_customer_session_with_a_proactive_agent_contains_a_message(
    application_context: ApplicationContextOfTest,
    proactive_agent_id: AgentId,
) -> None:
    session = await application_context.app.create_customer_session(
        customer_id=application_context.customer_id,
        agent_id=proactive_agent_id,
        allow_greeting=True,
    )

    assert await application_context.app.wait_for_update(
        session_id=session.id,
        min_offset=0,
        kinds=["message"],
        timeout=Timeout(REASONABLE_AMOUNT_OF_TIME),
    )

    events = list(await application_context.container[SessionStore].list_events(session.id))

    assert len([e for e in events if e.kind == "message"]) == 1


async def test_that_when_a_client_event_is_posted_then_new_server_events_are_emitted(
    application_context: ApplicationContextOfTest,
    session: Session,
) -> None:
    event = await application_context.app.post_event(
        session_id=session.id,
        kind="message",
        data={
            "message": "Hey there",
            "participant": {
                "display_name": "Johnny Boy",
            },
        },
    )

    await application_context.app.wait_for_update(
        session_id=session.id,
        min_offset=1 + event.offset,
        kinds=["message"],
        timeout=Timeout(REASONABLE_AMOUNT_OF_TIME),
    )

    events = list(await application_context.container[SessionStore].list_events(session.id))

    assert len(events) > 1


async def test_that_a_session_update_is_detected_as_soon_as_a_client_event_is_posted(
    application_context: ApplicationContextOfTest,
    session: Session,
) -> None:
    event = await application_context.app.post_event(
        session_id=session.id,
        kind="message",
        data={
            "message": "Hey there",
            "participant": {
                "display_name": "Johnny Boy",
            },
        },
    )

    assert await application_context.app.wait_for_update(
        session_id=session.id,
        min_offset=event.offset,
        kinds=[],
        timeout=Timeout.none(),
    )


async def test_that_when_a_customer_quickly_posts_more_than_one_message_then_only_one_message_is_emitted_as_a_reply_to_the_last_message(
    application_context: ApplicationContextOfTest,
    session: Session,
) -> None:
    messages = [
        "What are bananas?",
        "Scratch that; what are apples?",
        "Actually scratch that too. What are pineapples?",
    ]

    for m in messages:
        await application_context.app.post_event(
            session_id=session.id,
            kind="message",
            data={
                "message": m,
                "participant": {
                    "display_name": "Johnny Boy",
                },
            },
        )

        await asyncio.sleep(1)

    await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

    events = list(await application_context.container[SessionStore].list_events(session.id))
    message_events = [e for e in events if e.kind == "message"]

    assert len(message_events) == 4
    assert await nlp_test(str(message_events[-1].data), "It talks about pineapples")


def hand_off_to_human_operator() -> ToolResult:
    return ToolResult(data=None, control={"mode": "manual"})


async def test_that_a_response_is_not_generated_automatically_after_a_tool_switches_the_session_to_manual_mode(
    application_context: ApplicationContextOfTest,
    session: Session,
) -> None:
    await create_guideline(
        container=application_context.container,
        agent_id=session.agent_id,
        condition="the customer expresses dissatisfaction",
        action="immediately hand off to a human operator, explaining this just before you sign off",
        tool_function=hand_off_to_human_operator,
    )

    event = await application_context.app.post_event(
        session_id=session.id,
        kind="message",
        data={
            "message": "I'm extremely dissatisfied with your service!",
            "participant": {
                "display_name": "Johnny Boy",
            },
        },
    )

    await application_context.app.wait_for_update(
        session_id=session.id,
        min_offset=event.offset,
        kinds=["message"],
        source="ai_agent",
        timeout=Timeout(30),
    )

    updated_session = await application_context.container[SessionStore].read_session(session.id)

    assert session.mode == "auto"
    assert updated_session.mode == "manual"
