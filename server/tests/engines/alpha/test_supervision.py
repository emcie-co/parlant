from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, cast
from lagom import Container
from pytest import fixture
from pytest_bdd import scenarios, given, when, then, parsers

from emcie.server.contextual_correlator import ContextualCorrelator
from emcie.server.core.agents import Agent, AgentId, AgentStore
from emcie.server.core.end_users import EndUserId
from emcie.server.core.guidelines import Guideline, GuidelineStore
from emcie.server.core.sessions import Event, MessageEventData, SessionId, SessionStore
from emcie.server.engines.alpha.message_event_producer import MessageEventProducer
from emcie.server.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.engines.event_emitter import EmittedEvent

from emcie.server.logger import Logger
from tests.test_utilities import SyncAwaiter, nlp_test

roles = Literal["client", "server"]

scenarios(
    "engines/alpha/supervision.feature",
)


@dataclass
class _TestContext:
    sync_await: SyncAwaiter
    container: Container
    agent_id: AgentId
    guidelines: dict[str, Guideline]
    guideline_proposition: dict[str, GuidelineProposition]
    intercations_history: list[Event]


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent.id


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> _TestContext:
    return _TestContext(
        sync_await,
        container,
        agent_id,
        guidelines=dict(),
        guideline_proposition=dict(),
        intercations_history=list(),
    )


@given("an empty session", target_fixture="session_id")
def given_an_empty_session(
    context: _TestContext,
) -> SessionId:
    store = context.container[SessionStore]
    utc_now = datetime.now(timezone.utc)
    session = context.sync_await(
        store.create_session(
            creation_utc=utc_now,
            end_user_id=EndUserId("test_user"),
            agent_id=context.agent_id,
        )
    )
    return session.id


@given(parsers.parse('a user message, "{user_message}"'), target_fixture="session_id")
def given_a_user_message(
    context: _TestContext,
    session_id: SessionId,
    user_message: str,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    context.intercations_history.append(
        context.sync_await(
            store.create_event(
                session_id=session.id,
                source="client",
                kind="message",
                correlation_id="test_correlation_id",
                data={"message": user_message},
            )
        )
    )

    return session.id


@given(parsers.parse("a guideline {guideline_name}, to {do_something} when {a_condition_holds}"))
def given_a_guideline_to_when(
    context: _TestContext,
    guideline_name: str,
    do_something: str,
    a_condition_holds: str,
) -> None:
    guideline_store = context.container[GuidelineStore]
    context.guidelines[guideline_name] = context.sync_await(
        guideline_store.create_guideline(
            guideline_set=context.agent_id,
            predicate=a_condition_holds,
            action=do_something,
        )
    )


@given(
    parsers.parse(
        "that the {guideline_name} guideline is proposed with a priority of {score} because {rationale}"  # noqa
    )
)
def given_a_guideline_proposition(
    context: _TestContext,
    guideline_name: str,
    score: int,
    rationale: str,
) -> None:
    guideline = context.guidelines[guideline_name]
    context.guideline_proposition[guideline_name] = GuidelineProposition(
        guideline=guideline,
        score=score,
        rationale=rationale,
    )


@when("messages are emitted", target_fixture="emitted_events")
def when_processing_is_triggered(
    context: _TestContext,
) -> list[EmittedEvent]:
    agents = [
        Agent(
            id=AgentId("123"),
            creation_utc=datetime.now(timezone.utc),
            name="Test Agent",
            description="You are an agent that works for Emcie",
        )
    ]

    message_event_producer = MessageEventProducer(
        context.container[Logger],
        context.container[ContextualCorrelator],
    )

    message_events = context.sync_await(
        message_event_producer.produce_events(
            agents=agents,
            context_variables=[],
            interaction_history=context.intercations_history,
            terms=[],
            ordinary_guideline_propositions=list(context.guideline_proposition.values()),
            tool_enabled_guideline_propositions={},
            staged_events=[],
        )
    )

    return list(message_events)


@then(parsers.parse("the message should contain {something}"))
def then_the_message_contains(
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == "message")
    message = cast(MessageEventData, message_event.data)["message"]

    assert nlp_test(
        context=message,
        predicate=f"the text contains {something}",
    )
