import asyncio
from typing import Callable, cast
from lagom import Container
from pytest import fixture
from pytest_bdd import scenarios, given, when, then, parsers
from datetime import datetime, timezone

from emcie.server.core.agents import AgentId, AgentStore
from emcie.server.core.end_users import EndUserId
from emcie.server.core.engines.alpha.engine import AlphaEngine
from emcie.server.core.engines.types import Context
from emcie.server.core.engines.emission import EmittedEvent
from emcie.server.core.guidelines import Guideline, GuidelineStore
from emcie.server.core.sessions import (
    MessageEventData,
    Session,
    SessionId,
    SessionStatus,
    SessionStore,
    StatusEventData,
)

from emcie.server.logger import Logger
from tests.test_utilities import EventBuffer, SyncAwaiter, nlp_test

scenarios(
    "engines/alpha/vanilla_agent.feature",
    "engines/alpha/message_agent_with_rules.feature",
)


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent.id


@fixture
def new_session(
    container: Container,
    sync_await: SyncAwaiter,
    agent_id: AgentId,
) -> Session:
    store = container[SessionStore]
    utc_now = datetime.now(timezone.utc)
    return sync_await(
        store.create_session(
            creation_utc=utc_now,
            end_user_id=EndUserId("test_user"),
            agent_id=agent_id,
        )
    )


@given("the alpha engine", target_fixture="engine")
def given_the_alpha_engine(
    container: Container,
) -> AlphaEngine:
    return container[AlphaEngine]


@given(parsers.parse("an agent whose job is {description}"), target_fixture="agent_id")
def given_an_agent_with_identity(
    container: Container,
    sync_await: SyncAwaiter,
    description: str,
) -> AgentId:
    agent = sync_await(
        container[AgentStore].create_agent(
            name="test-agent",
            description=f"Your job is {description}",
        )
    )
    return agent.id


@given("an agent", target_fixture="agent_id")
def given_an_agent(
    agent_id: AgentId,
) -> AgentId:
    return agent_id


@given(parsers.parse('a user message, "{user_message}"'), target_fixture="session_id")
def given_a_user_message(
    sync_await: SyncAwaiter,
    container: Container,
    session_id: SessionId,
    user_message: str,
) -> SessionId:
    store = container[SessionStore]

    sync_await(
        store.create_event(
            session_id=session_id,
            source="client",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": user_message},
        )
    )

    return session_id


@given(parsers.parse("a guideline to {do_something}"))
def given_a_guideline_to(
    do_something: str,
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> Guideline:
    guideline_store = container[GuidelineStore]

    guidelines: dict[str, Callable[[], Guideline]] = {
        "greet with 'Howdy'": lambda: sync_await(
            guideline_store.create_guideline(
                guideline_set=agent_id,
                predicate="The user hasn't engaged yet",
                action="Greet the user with the word 'Howdy'",
            )
        ),
        "offer thirsty users a Pepsi": lambda: sync_await(
            guideline_store.create_guideline(
                guideline_set=agent_id,
                predicate="The user is thirsty",
                action="Offer the user a Pepsi",
            )
        ),
        "do your job when the user says hello": lambda: sync_await(
            guideline_store.create_guideline(
                guideline_set=agent_id,
                predicate="greeting the user",
                action="do your job when the user says hello",
            )
        ),
    }

    return guidelines[do_something]()


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
            action=do_something,
        )
    )


@given("50 other random guidelines")
def given_50_other_random_guidelines(
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> list[Guideline]:
    guideline_store = container[GuidelineStore]

    async def create_guideline(predicate: str, action: str) -> Guideline:
        return await guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=predicate,
            action=action,
        )

    guidelines: list[Guideline] = []

    for guideline_params in [
        {
            "predicate": "The user mentions being hungry",
            "action": "Suggest our pizza specials to the user",
        },
        {
            "predicate": "The user asks about vegetarian options",
            "action": "list all vegetarian pizza options",
        },
        {
            "predicate": "The user inquires about delivery times",
            "action": "Provide the estimated delivery time based on their location",
        },
        {
            "predicate": "The user seems undecided",
            "action": "Recommend our top three most popular pizzas",
        },
        {
            "predicate": "The user asks for discount or promotions",
            "action": "Inform the user about current deals or coupons",
        },
        {
            "predicate": "The conversation starts",
            "action": "Greet the user and ask if they'd like to order a pizza",
        },
        {
            "predicate": "The user mentions a food allergy",
            "action": "Ask for specific allergies and recommend safe menu options",
        },
        {
            "predicate": "The user requests a custom pizza",
            "action": "Guide the user through choosing base, sauce, toppings, and cheese",
        },
        {
            "predicate": "The user wants to repeat a previous order",
            "action": "Retrieve the user’s last order details and confirm if they want the same",
        },
        {
            "predicate": "The user asks about portion sizes",
            "action": "Describe the different pizza sizes and how many they typically serve",
        },
        {
            "predicate": "The user requests a drink",
            "action": "list available beverages and suggest popular pairings with "
            "their pizza choice",
        },
        {
            "predicate": "The user asks for the price",
            "action": "Provide the price of the selected items and any additional costs",
        },
        {
            "predicate": "The user expresses concern about calories",
            "action": "Offer information on calorie content and suggest lighter "
            "options if desired",
        },
        {
            "predicate": "The user mentions a special occasion",
            "action": "Suggest our party meal deals and ask if they would like "
            "to include desserts",
        },
        {
            "predicate": "The user wants to know the waiting area",
            "action": "Inform about the waiting facilities at our location or "
            "suggest comfortable seating arrangements",
        },
        {
            "predicate": "The user is comparing pizza options",
            "action": "Highlight the unique features of different pizzas we offer",
        },
        {
            "predicate": "The user asks for recommendations",
            "action": "Suggest pizzas based on their previous orders or popular trends",
        },
        {
            "predicate": "The user is interested in combo deals",
            "action": "Explain the different combo offers and their benefits",
        },
        {
            "predicate": "The user asks if ingredients are fresh",
            "action": "Assure them of the freshness and quality of our ingredients",
        },
        {
            "predicate": "The user wants to modify an order",
            "action": "Assist in making the desired changes and confirm the new order details",
        },
        {
            "predicate": "The user has connectivity issues during ordering",
            "action": "Suggest completing the order via a different method (phone, app)",
        },
        {
            "predicate": "The user expresses dissatisfaction with a previous order",
            "action": "Apologize and offer a resolution (discount, replacement)",
        },
        {
            "predicate": "The user inquires about loyalty programs",
            "action": "Describe our loyalty program benefits and enrollment process",
        },
        {
            "predicate": "The user is about to end the conversation without ordering",
            "action": "Offer a quick summary of unique selling points or a one-time "
            "discount to encourage purchase",
        },
        {
            "predicate": "The user asks for gluten-free options",
            "action": "list our gluten-free pizza bases and toppings",
        },
        {
            "predicate": "The user is looking for side orders",
            "action": "Recommend complementary side dishes like garlic bread or salads",
        },
        {
            "predicate": "The user mentions children",
            "action": "Suggest our kids' menu or family-friendly options",
        },
        {
            "predicate": "The user is having trouble with the online payment",
            "action": "Offer assistance with the payment process or propose an "
            "alternative payment method",
        },
        {
            "predicate": "The user wants to know the origin of ingredients",
            "action": "Provide information about the source and quality assurance "
            "of our ingredients",
        },
        {
            "predicate": "The user asks for a faster delivery option",
            "action": "Explain express delivery options and any associated costs",
        },
        {
            "predicate": "The user seems interested in healthy eating",
            "action": "Highlight our health-conscious options like salads or "
            "pizzas with whole wheat bases",
        },
        {
            "predicate": "The user wants a contactless delivery",
            "action": "Confirm the address and explain the process for contactless delivery",
        },
        {
            "predicate": "The user is a returning customer",
            "action": "Welcome them back and ask if they would like to order their "
            "usual or try something new",
        },
        {
            "predicate": "The user inquires about our environmental impact",
            "action": "Share information about our sustainability practices and "
            "eco-friendly packaging",
        },
        {
            "predicate": "The user is planning a large event",
            "action": "Offer catering services and discuss bulk order discounts",
        },
        {
            "predicate": "The user seems in a rush",
            "action": "Suggest our quickest delivery option and process the order promptly",
        },
        {
            "predicate": "The user wants to pick up the order",
            "action": "Provide the pickup location and expected time until the order is ready",
        },
        {
            "predicate": "The user expresses interest in a specific topping",
            "action": "Offer additional information about that topping and suggest "
            "other complementary toppings",
        },
        {
            "predicate": "The user is making a business order",
            "action": "Propose our corporate deals and ask about potential regular "
            "orders for business meetings",
        },
        {
            "predicate": "The user asks for cooking instructions",
            "action": "Provide details on how our pizzas are made or instructions "
            "for reheating if applicable",
        },
        {
            "predicate": "The user inquires about the chefs",
            "action": "Share background information on our chefs’ expertise and experience",
        },
        {
            "predicate": "The user asks about non-dairy options",
            "action": "list our vegan cheese alternatives and other non-dairy products",
        },
        {
            "predicate": "The user expresses excitement about a new menu item",
            "action": "Provide more details about the item and suggest adding it to their order",
        },
        {
            "predicate": "The user wants a quiet place to eat",
            "action": "Describe the ambiance of our quieter dining areas or "
            "recommend off-peak times",
        },
        {
            "predicate": "The user asks about our app",
            "action": "Explain the features of our app and benefits of ordering through it",
        },
        {
            "predicate": "The user has difficulty deciding",
            "action": "Offer to make a selection based on their preferences or "
            "our chef’s recommendations",
        },
        {
            "predicate": "The user mentions they are in a specific location",
            "action": "Check if we deliver to that location and inform them about "
            "the nearest outlet",
        },
        {
            "predicate": "The user is concerned about food safety",
            "action": "Reassure them about our health and safety certifications and practices",
        },
        {
            "predicate": "The user is looking for a quiet place to eat",
            "action": "Describe the ambiance of our quieter dining areas or "
            "recommend off-peak times",
        },
        {
            "predicate": "The user shows interest in repeat orders",
            "action": "Introduce features like scheduled deliveries or subscription "
            "services to simplify their future orders",
        },
    ]:
        guidelines.append(sync_await(create_guideline(**guideline_params)))

    return guidelines


@given("an empty session", target_fixture="session_id")
def given_an_empty_session(
    sync_await: SyncAwaiter,
    container: Container,
    new_session: Session,
) -> SessionId:
    return new_session.id


@given("a session with a single user message", target_fixture="session_id")
def given_a_session_with_a_single_user_message(
    sync_await: SyncAwaiter,
    container: Container,
    new_session: Session,
) -> SessionId:
    store = container[SessionStore]

    sync_await(
        store.create_event(
            session_id=new_session.id,
            source="client",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": "Hey there"},
        )
    )

    return new_session.id


@given("a session with a thirsty user", target_fixture="session_id")
def given_a_session_with_a_thirsty_user(
    sync_await: SyncAwaiter,
    container: Container,
    new_session: Session,
) -> SessionId:
    store = container[SessionStore]

    sync_await(
        store.create_event(
            session_id=new_session.id,
            source="client",
            kind="message",
            correlation_id="test_correlation_id",
            data={"message": "I'm thirsty"},
        )
    )

    return new_session.id


@given("a session with a few messages", target_fixture="session_id")
def given_a_session_with_a_few_messages(
    sync_await: SyncAwaiter,
    container: Container,
    new_session: Session,
) -> SessionId:
    store = container[SessionStore]

    messages = [
        {
            "source": "client",
            "message": "hey there",
        },
        {
            "source": "server",
            "message": "Hi, how can I help you today?",
        },
        {
            "source": "client",
            "message": "What was the first name of the famous Einstein?",
        },
    ]

    for m in messages:
        sync_await(
            store.create_event(
                session_id=new_session.id,
                source=m["source"] == "server" and "server" or "client",
                kind="message",
                correlation_id="test_correlation_id",
                data={"message": m["message"]},
            )
        )

    return new_session.id


@when("processing is triggered", target_fixture="emitted_events")
def when_processing_is_triggered(
    sync_await: SyncAwaiter,
    engine: AlphaEngine,
    agent_id: AgentId,
    session_id: SessionId,
) -> list[EmittedEvent]:
    event_buffer = EventBuffer()

    sync_await(
        engine.process(
            Context(
                session_id=session_id,
                agent_id=agent_id,
            ),
            event_buffer,
        )
    )

    return event_buffer.events


@when("processing is triggered and cancelled in the middle", target_fixture="emitted_events")
def when_processing_is_triggered_and_cancelled_in_the_middle(
    sync_await: SyncAwaiter,
    engine: AlphaEngine,
    agent_id: AgentId,
    session_id: SessionId,
) -> list[EmittedEvent]:
    event_buffer = EventBuffer()

    processing_task = sync_await.event_loop.create_task(
        engine.process(
            Context(
                session_id=session_id,
                agent_id=agent_id,
            ),
            event_buffer,
        )
    )

    sync_await(asyncio.sleep(1))

    processing_task.cancel()

    assert not sync_await(processing_task)

    return event_buffer.events


@then("no events are emitted")
def then_no_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len(emitted_events) == 0


@then("no message events are emitted")
def then_no_message_events_are_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len([e for e in emitted_events if e.kind == "message"]) == 0


@then("a single message event is emitted")
def then_a_single_message_event_is_emitted(
    emitted_events: list[EmittedEvent],
) -> None:
    assert len(list(filter(lambda e: e.kind == "message", emitted_events))) == 1


@then(parsers.parse("the message contains {something}"))
def then_the_message_contains(
    container: Container,
    sync_await: SyncAwaiter,
    emitted_events: list[EmittedEvent],
    something: str,
) -> None:
    message_event = next(e for e in emitted_events if e.kind == "message")

    assert sync_await(
        nlp_test(
            logger=container[Logger],
            context=cast(MessageEventData, message_event.data)["message"],
            predicate=f"the text contains {something}",
        )
    )


def _has_status_event(
    status: SessionStatus,
    acknowledged_event_offset: int,
    events: list[EmittedEvent],
) -> bool:
    for e in (e for e in events if e.kind == "status"):
        data = cast(StatusEventData, e.data)

        has_same_status = data["status"] == status
        has_same_acknowledged_offset = data["acknowledged_offset"] == acknowledged_event_offset

        if has_same_status and has_same_acknowledged_offset:
            return True

    return False


@then(parsers.parse("a status event is emitted, acknowledging event {acknowledged_event_offset:d}"))
def then_an_acknowledgement_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("acknowledged", acknowledged_event_offset, emitted_events)


@then(parsers.parse("a status event is emitted, processing event {acknowledged_event_offset:d}"))
def then_a_processing_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("processing", acknowledged_event_offset, emitted_events)


@then(
    parsers.parse(
        "a status event is emitted, typing in response to event {acknowledged_event_offset:d}"
    )
)
def then_a_typing_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("typing", acknowledged_event_offset, emitted_events)


@then(
    parsers.parse(
        "a status event is emitted, cancelling the response to event {acknowledged_event_offset:d}"
    )
)
def then_a_cancelled_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("cancelled", acknowledged_event_offset, emitted_events)


@then(
    parsers.parse(
        "a status event is emitted, ready for further engagement after reacting to event {acknowledged_event_offset:d}"
    )
)
def then_a_ready_status_event_is_emitted(
    emitted_events: list[EmittedEvent],
    acknowledged_event_offset: int,
) -> None:
    assert _has_status_event("ready", acknowledged_event_offset, emitted_events)