# Copyright 2026 Emcie Co Ltd.
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
from dataclasses import dataclass
from lagom import Container
from pytest import fixture

from parlant.core.async_utils import Timeout
from parlant.core.application import Application
from parlant.core.agents import AgentId, AgentStore
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.sessions import EventKind, EventSource, Session, SessionStore
from parlant.core.tags import Tag
from parlant.core.tools import ToolResult

from tests.test_utilities import create_guideline, nlp_test

REASONABLE_AMOUNT_OF_TIME = 10


@dataclass
class ContextOfTest:
    container: Container
    app: Application
    customer_id: CustomerId


@fixture
async def context(
    container: Container,
    customer_id: CustomerId,
) -> ContextOfTest:
    return ContextOfTest(
        container=container,
        app=container[Application],
        customer_id=customer_id,
    )


@fixture
async def agent_id(container: Container) -> AgentId:
    store = container[AgentStore]
    agent = await store.create_agent(
        name="test-agent",
        max_engine_iterations=2,
    )
    return agent.id


@fixture
async def proactive_agent_id(
    container: Container,
    agent_id: AgentId,
) -> AgentId:
    guideline = await container[GuidelineStore].create_guideline(
        condition="The customer hasn't engaged yet",
        action="Greet the customer",
    )

    await container[GuidelineStore].upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    return agent_id


@fixture
async def session(
    container: Container,
    customer_id: CustomerId,
    agent_id: AgentId,
) -> Session:
    store = container[SessionStore]
    session = await store.create_session(
        customer_id=customer_id,
        agent_id=agent_id,
    )
    return session


@fixture
async def customer_id(container: Container) -> CustomerId:
    store = container[CustomerStore]
    customer = await store.create_customer("Larry David", extra={"email": "larry@seinfeld.com"})
    return customer.id


async def test_that_a_new_customer_session_can_be_created(
    context: ContextOfTest,
    agent_id: AgentId,
) -> None:
    created_session = await context.app.sessions.create(
        customer_id=context.customer_id,
        agent_id=agent_id,
    )

    session_in_db = await context.container[SessionStore].read_session(
        created_session.id,
    )

    assert created_session == session_in_db


async def test_that_a_new_customer_session_with_a_proactive_agent_contains_a_message(
    context: ContextOfTest,
    proactive_agent_id: AgentId,
) -> None:
    session = await context.app.sessions.create(
        customer_id=context.customer_id,
        agent_id=proactive_agent_id,
        allow_greeting=True,
    )

    assert await context.app.sessions.wait_for_update(
        session_id=session.id,
        min_offset=0,
        kinds=[EventKind.MESSAGE],
        timeout=Timeout(REASONABLE_AMOUNT_OF_TIME),
    )

    events = list(await context.container[SessionStore].list_events(session.id))

    assert len([e for e in events if e.kind == EventKind.MESSAGE]) == 1


async def test_that_when_a_client_event_is_posted_then_new_server_events_are_emitted(
    context: ContextOfTest,
    session: Session,
) -> None:
    event = await context.app.sessions.create_event(
        session_id=session.id,
        kind=EventKind.MESSAGE,
        data={
            "message": "Hey there",
            "participant": {
                "display_name": "Johnny Boy",
            },
        },
        metadata={},
    )

    await context.app.sessions.wait_for_update(
        session_id=session.id,
        min_offset=1 + event.offset,
        kinds=[EventKind.MESSAGE],
        timeout=Timeout(REASONABLE_AMOUNT_OF_TIME),
    )

    events = list(await context.container[SessionStore].list_events(session.id))

    assert len(events) > 1


async def test_that_a_session_update_is_detected_as_soon_as_a_client_event_is_posted(
    context: ContextOfTest,
    session: Session,
) -> None:
    event = await context.app.sessions.create_event(
        session_id=session.id,
        kind=EventKind.MESSAGE,
        data={
            "message": "Hey there",
            "participant": {
                "display_name": "Johnny Boy",
            },
        },
        metadata={},
    )

    assert await context.app.sessions.wait_for_update(
        session_id=session.id,
        min_offset=event.offset,
        kinds=[],
        timeout=Timeout.none(),
    )


async def test_that_when_a_customer_quickly_posts_more_than_one_message_then_only_one_message_is_emitted_as_a_reply_to_the_last_message(
    context: ContextOfTest,
    session: Session,
) -> None:
    messages = [
        "What are bananas?",
        "Scratch that; what are apples?",
        "Actually scratch that too. What are pineapples?",
    ]

    for m in messages:
        await context.app.sessions.create_event(
            session_id=session.id,
            kind=EventKind.MESSAGE,
            data={
                "message": m,
                "participant": {
                    "display_name": "Johnny Boy",
                },
            },
            metadata={},
        )

        await asyncio.sleep(1)

    await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

    events = list(await context.container[SessionStore].list_events(session.id))
    message_events = [e for e in events if e.kind == EventKind.MESSAGE]

    assert len(message_events) == 4
    assert await nlp_test(str(message_events[-1].data), "It talks about pineapples")


def hand_off_to_human_operator() -> ToolResult:
    return ToolResult(data=None, control={"mode": "manual"})


async def test_that_a_response_is_not_generated_automatically_after_a_tool_switches_the_session_to_manual_mode(
    context: ContextOfTest,
    session: Session,
) -> None:
    await create_guideline(
        container=context.container,
        agent_id=session.agent_id,
        condition="the customer expresses dissatisfaction",
        action="immediately hand off to a human operator, explaining this just before you sign off",
        tool_function=hand_off_to_human_operator,
    )

    event = await context.app.sessions.create_event(
        session_id=session.id,
        kind=EventKind.MESSAGE,
        data={
            "message": "I'm extremely dissatisfied with your service!",
            "participant": {
                "display_name": "Johnny Boy",
            },
        },
        metadata={},
    )

    await context.app.sessions.wait_for_update(
        session_id=session.id,
        min_offset=event.offset,
        kinds=[EventKind.MESSAGE],
        source=EventSource.AI_AGENT,
        timeout=Timeout(30),
    )

    updated_session = await context.container[SessionStore].read_session(session.id)

    assert session.mode == "auto"
    assert updated_session.mode == "manual"

    event = await context.app.sessions.create_event(
        session_id=session.id,
        kind=EventKind.MESSAGE,
        data={
            "message": "Well?",
            "participant": {
                "display_name": "Johnny Boy",
            },
        },
        metadata={},
    )

    assert not await context.app.sessions.wait_for_update(
        session_id=session.id,
        min_offset=event.offset + 1,
        timeout=Timeout(3),
    )
