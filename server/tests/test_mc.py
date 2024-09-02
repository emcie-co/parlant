import asyncio
from dataclasses import dataclass
from lagom import Container
from pytest import fixture

from emcie.server.async_utils import Timeout
from emcie.server.mc import MC
from emcie.server.core.agents import AgentId, AgentStore
from emcie.server.core.end_users import EndUserId, EndUserStore
from emcie.server.core.guidelines import GuidelineStore
from emcie.server.core.sessions import Event, Session, SessionStore
from tests.test_utilities import nlp_test

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
    agent = await store.create_agent(name="test-agent")
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
        min_offset=0,
        timeout=Timeout(REASONABLE_AMOUNT_OF_TIME),
    )

    events = list(await context.container[SessionStore].list_events(session.id))

    assert len(events) == 1


async def test_that_when_a_client_event_is_posted_then_new_server_events_are_produced(
    context: _TestContext,
    session: Session,
) -> None:
    event = await context.mc.post_client_event(
        session_id=session.id,
        kind=Event.MESSAGE_KIND,
        data={"message": "Hey there"},
    )

    await context.mc.wait_for_update(
        session_id=session.id,
        min_offset=1 + event.offset,
        timeout=Timeout(REASONABLE_AMOUNT_OF_TIME),
    )

    events = list(await context.container[SessionStore].list_events(session.id))

    assert len(events) > 1


async def test_that_a_session_update_is_detected_as_soon_as_a_client_event_is_posted(
    context: _TestContext,
    session: Session,
) -> None:
    event = await context.mc.post_client_event(
        session_id=session.id,
        kind=Event.MESSAGE_KIND,
        data={"message": "Hey there"},
    )

    assert await context.mc.wait_for_update(
        session_id=session.id,
        min_offset=event.offset,
        timeout=Timeout.none(),
    )


async def test_that_when_a_user_quickly_posts_more_than_one_message_then_only_one_message_is_produced_as_a_reply_to_the_last_message(
    context: _TestContext,
    session: Session,
) -> None:
    messages = [
        "What are bananas?",
        "Scratch that; what are apples?",
        "Actually scratch that too. What are pineapples?",
    ]

    for m in messages:
        await context.mc.post_client_event(
            session_id=session.id,
            kind=Event.MESSAGE_KIND,
            data={"message": m},
        )

        await asyncio.sleep(1)

    await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

    events = list(await context.container[SessionStore].list_events(session.id))

    assert len(events) == 4
    assert nlp_test(str(events[-1].data), "It talks about pineapples")
