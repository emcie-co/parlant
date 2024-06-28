from dataclasses import dataclass
from typing import AsyncGenerator
from lagom import Container
from pytest import fixture

from emcie.server.async_utils import Timeout
from emcie.server.mc import MC
from emcie.server.core.agents import AgentId, AgentStore
from emcie.server.core.end_users import EndUserId, EndUserStore
from emcie.server.core.guidelines import GuidelineStore
from emcie.server.core.sessions import Event, Session, SessionStore

REASONABLE_AMOUNT_OF_TIME = 10


@dataclass
class _TestContext:
    container: Container
    mc: MC
    end_user_id: EndUserId


@fixture
async def context(
    container: Container,
    end_user_id: EndUserId,
) -> _TestContext:
    return _TestContext(
        container=container,
        mc=container[MC],
        end_user_id=end_user_id,
    )


@fixture
async def agent_id(container: Container) -> AgentId:
    store = container[AgentStore]
    agent = await store.create_agent()
    return agent.id


@fixture
async def proactive_agent_id(
    container: Container,
    agent_id: AgentId,
) -> AgentId:
    await container[GuidelineStore].create_guideline(
        guideline_set=agent_id,
        predicate="The user hasn't engaged yet",
        content="Greet the user",
    )

    return agent_id


@fixture
async def session(
    container: Container,
    end_user_id: EndUserId,
    agent_id: AgentId,
) -> Session:
    store = container[SessionStore]
    session = await store.create_session(
        end_user_id=end_user_id,
        agent_id=agent_id,
    )
    return session


@fixture
async def end_user_id(container: Container) -> EndUserId:
    store = container[EndUserStore]
    user = await store.create_end_user("Larry David", email="larry@seinfeld.com")
    return user.id


async def test_that_a_new_end_user_session_can_be_created(
    context: _TestContext,
    agent_id: AgentId,
) -> None:
    created_session = await context.mc.create_end_user_session(
        end_user_id=context.end_user_id,
        agent_id=agent_id,
    )

    session_in_db = await context.container[SessionStore].read_session(
        created_session.id,
    )

    assert created_session == session_in_db


async def test_that_a_new_user_session_with_a_proactive_agent_contains_a_message(
    context: _TestContext,
    proactive_agent_id: AgentId,
) -> None:
    session = await context.mc.create_end_user_session(
        end_user_id=context.end_user_id,
        agent_id=proactive_agent_id,
    )

    assert await context.mc.wait_for_update(
        session_id=session.id,
        latest_known_offset=session.consumption_offsets["client"],
        timeout=Timeout(REASONABLE_AMOUNT_OF_TIME),
    )

    events = list(await context.container[SessionStore].list_events(session.id))

    assert len(events) == 1


async def test_that_when_a_client_event_is_posted_then_new_server_events_are_produced(
    context: _TestContext,
    session: Session,
) -> None:
    await context.mc.post_client_event(
        session_id=session.id,
        type=Event.MESSAGE_TYPE,
        data={"message": "Hey there"},
    )

    await context.mc.wait_for_update(
        session_id=session.id,
        latest_known_offset=session.consumption_offsets["client"],
        timeout=Timeout(REASONABLE_AMOUNT_OF_TIME),
    )

    events = list(await context.container[SessionStore].list_events(session.id))

    assert len(events) > 1
