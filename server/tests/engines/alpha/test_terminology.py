from dataclasses import dataclass
from typing import Literal, cast
from lagom import Container
from pytest import fixture
from pytest_bdd import scenarios, given, when, then, parsers

from emcie.server.core.agents import AgentId, AgentStore
from emcie.server.core.end_users import EndUserId
from emcie.server.core.guidelines import GuidelineStore
from emcie.server.core.sessions import Event, MessageEventData, SessionId, SessionStore
from emcie.server.engines.alpha.engine import AlphaEngine
from emcie.server.engines.alpha.terminology import TerminologyStore
from emcie.server.engines.common import Context, ProducedEvent

from tests.test_utilities import SyncAwaiter, nlp_test

roles = Literal["client", "server"]

scenarios(
    "engines/alpha/terminology.feature",
)


@dataclass
class _TestContext:
    sync_await: SyncAwaiter
    container: Container
    agent_id: AgentId


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
    )


@given("the alpha engine", target_fixture="engine")
def given_the_alpha_engine(
    container: Container,
) -> AlphaEngine:
    return container[AlphaEngine]


@given("an agent", target_fixture="agent_id")
def given_an_agent(
    agent_id: AgentId,
) -> AgentId:
    return agent_id


@given("an empty session", target_fixture="session_id")
def given_an_empty_session(
    context: _TestContext,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(
        store.create_session(
            end_user_id=EndUserId("test_user"),
            agent_id=context.agent_id,
        )
    )
    return session.id


@given(parsers.parse("a guideline to {do_something} when {a_condition_holds}"))
def given_a_guideline_to_when(
    do_something: str,
    a_condition_holds: str,
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    sync_await(
        guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=a_condition_holds,
            content=do_something,
        )
    )


@given(parsers.parse('a user message, "{user_message}"'), target_fixture="session_id")
def given_a_user_message(
    context: _TestContext,
    session_id: SessionId,
    user_message: str,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    context.sync_await(
        store.create_event(
            session_id=session.id,
            source="client",
            kind=Event.MESSAGE_KIND,
            data={"message": user_message},
        )
    )

    return session.id


@given(parsers.parse("a term of {term}"), target_fixture="terminology")
def given_a_term_of(
    context: _TestContext,
    agent_id: AgentId,
    term: str,
) -> None:
    cryptocurrency_terminology = {
        "token": {
            "name": "token",
            "description": "A digital asset issued on a blockchain, often representing an asset or utility.",  # noqa
            "synonyms": ["crypto token", "digital token"],
        },
        "smart contract": {
            "name": "smart contract",
            "description": "A self-executing contract with the terms directly written into lines of code.",  # noqa
            "synonyms": ["self-executing contract", "blockchain contract"],
        },
        "mining": {
            "name": "mining",
            "description": "The process of validating and adding transactions to the blockchain, typically requiring significant computational power.",  # noqa
            "synonyms": ["crypto mining", "bitcoin mining"],
        },
        "wallet": {
            "name": "wallet",
            "description": "A digital tool that allows users to store, send, and receive cryptocurrencies.",  # noqa
            "synonyms": ["crypto wallet", "digital wallet"],
        },
        "private key": {
            "name": "private key",
            "description": "A secret key that allows the owner to access and manage their cryptocurrency.",  # noqa
            "synonyms": ["secret key"],
        },
        "public key": {
            "name": "public key",
            "description": "A cryptographic key that is used to receive cryptocurrency and is shared publicly.",  # noqa
            "synonyms": ["crypto address", "public address"],
        },
        "gas": {
            "name": "gas",
            "description": "A fee required to conduct a transaction or execute a smart contract on the Ethereum blockchain.",  # noqa
            "synonyms": ["transaction fee", "ethereum fee"],
        },
        "walnut": {
            "name": "walnut",
            "description": "A name of an altcoin.",
        },
    }

    terminology_store = context.container[TerminologyStore]
    agent_name = context.sync_await(context.container[AgentStore].read_agent(agent_id)).name
    context.sync_await(
        terminology_store.create_term(
            **{"term_set": agent_name, **cryptocurrency_terminology[term]}  # type: ignore
        )
    )


@when("processing is triggered", target_fixture="produced_events")
def when_processing_is_triggered(
    context: _TestContext,
    engine: AlphaEngine,
    session_id: SessionId,
) -> list[ProducedEvent]:
    events = context.sync_await(
        engine.process(
            Context(
                session_id=session_id,
                agent_id=context.agent_id,
            )
        )
    )

    return list(events)


@then("a single message event is produced")
def then_a_single_message_event_is_produced(
    produced_events: list[ProducedEvent],
) -> None:
    assert len(list(filter(lambda e: e.kind == Event.MESSAGE_KIND, produced_events))) == 1


@then(parsers.parse("the message contains {something}"))
def then_the_message_contains(
    produced_events: list[ProducedEvent],
    something: str,
) -> None:
    message = cast(MessageEventData, produced_events[-1].data)["message"]

    assert nlp_test(
        context=message,
        predicate=f"the text contains {something}",
    )
