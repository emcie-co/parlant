from dataclasses import dataclass
from typing import Sequence
from lagom import Container
from pytest import fixture, mark
from emcie.server.core.agents import Agent
from emcie.server.core.guideline_connections import ConnectionKind
from emcie.server.core.guidelines import GuidelineContent
from emcie.server.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
)
from emcie.server.core.terminology import TerminologyStore
from tests.test_utilities import SyncAwaiter


@dataclass
class _TestContext:
    sync_await: SyncAwaiter
    container: Container


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> _TestContext:
    return _TestContext(sync_await, container)


def _create_guideline_content(
    predicate: str,
    action: str,
) -> GuidelineContent:
    return GuidelineContent(predicate=predicate, action=action)


@mark.parametrize(
    (
        "source_guideline_definition",
        "target_guideline_definition",
    ),
    [
        (
            {
                "predicate": "the user asks about the weather",
                "action": "provide the current weather update",
            },
            {
                "predicate": "providing the weather update",
                "action": "mention the best time to go for a walk",
            },
        ),
        (
            {
                "predicate": "the user asks about nearby restaurants",
                "action": "provide a list of popular restaurants",
            },
            {
                "predicate": "listing restaurants",
                "action": "highlight the one with the best reviews",
            },
        ),
    ],
)
def test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_predicate_of_the_other(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["predicate"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["predicate"],
        target_guideline_definition["action"],
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [target_guideline_content, source_guideline_content],
            )
        )
    )

    assert len(connection_propositions) == 1
    assert connection_propositions[0].source == source_guideline_content
    assert connection_propositions[0].target == target_guideline_content
    assert connection_propositions[0].kind == ConnectionKind.ENTAILS


@mark.parametrize(
    (
        "source_guideline_definition",
        "target_guideline_definition",
    ),
    [
        (
            {
                "guideline_set": "test-agent",
                "predicate": "the user requests technical support",
                "action": "provide the support contact details",
            },
            {
                "guideline_set": "test-agent",
                "predicate": "providing support contact details",
                "action": "consider checking the troubleshooting guide first",
            },
        ),
        (
            {
                "guideline_set": "test-agent",
                "predicate": "the user inquires about office hours",
                "action": "tell them the office hours",
            },
            {
                "guideline_set": "test-agent",
                "predicate": "mentioning office hours",
                "action": "you may suggest the best time to visit for quicker service",
            },
        ),
    ],
)
def test_that_a_suggestion_connection_is_proposed_for_two_guidelines_where_the_content_of_one_suggests_a_follow_up_to_the_predicate_of_the_other(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["predicate"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["predicate"],
        target_guideline_definition["action"],
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [
                    source_guideline_content,
                    target_guideline_content,
                ],
            )
        )
    )

    assert len(connection_propositions) == 1
    assert connection_propositions[0].source == source_guideline_content
    assert connection_propositions[0].target == target_guideline_content
    assert connection_propositions[0].kind == ConnectionKind.SUGGESTS


def test_that_multiple_connections_are_detected_and_proposed_at_the_same_time(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(predicate=i["predicate"], action=i["action"])
        for i in [
            {
                "predicate": "the user requests technical support",
                "action": "provide the support contact details",
            },
            {
                "predicate": "providing support contact details",
                "action": "consider checking the troubleshooting guide first",
            },
            {
                "predicate": "the user inquires about office hours",
                "action": "tell them the office hours",
            },
            {
                "predicate": "mentioning office hours",
                "action": "suggest the best time to visit for quicker service",
            },
            {
                "predicate": "the user asks about the weather",
                "action": "provide the current weather update",
            },
            {
                "predicate": "providing the weather update",
                "action": "mention the best time to go for a walk",
            },
            {
                "predicate": "the user asks about nearby restaurants",
                "action": "provide a list of popular restaurants",
            },
            {
                "predicate": "listing restaurants",
                "action": "highlight the one with the best reviews",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(agent, introduced_guidelines, [])
        )
    )

    assert len(connection_propositions) == len(introduced_guidelines) // 2

    pairs = [
        (introduced_guidelines[i], introduced_guidelines[i + 1])
        for i in range(0, len(introduced_guidelines), 2)
    ]

    for i, connection in enumerate(connection_propositions):
        assert connection.source == pairs[i][0]
        assert connection.target == pairs[i][1]


def test_that_possible_connections_between_existing_guidelines_are_not_proposed(
    context: _TestContext,
    agent: Agent,
) -> None:
    existing_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(predicate=i["predicate"], action=i["action"])
        for i in [
            {
                "predicate": "the user requests technical support",
                "action": "provide the support contact details",
            },
            {
                "predicate": "providing support contact details",
                "action": "consider checking the troubleshooting guide first",
            },
            {
                "predicate": "the user inquires about office hours",
                "action": "tell them the office hours",
            },
            {
                "predicate": "mentioning office hours",
                "action": "suggest the best time to visit for quicker service",
            },
            {
                "predicate": "the user asks about the weather",
                "action": "provide the current weather update",
            },
            {
                "predicate": "providing the weather update",
                "action": "mention the best time to go for a walk",
            },
            {
                "predicate": "the user asks about nearby restaurants",
                "action": "provide a list of popular restaurants",
            },
            {
                "predicate": "listing restaurants",
                "action": "highlight the one with the best reviews",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(connection_proposer.propose_connections(agent, [], existing_guidelines))
    )

    assert len(connection_propositions) == 0


def test_that_a_connection_is_proposed_based_on_given_terminology(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    terminology_store = context.container[TerminologyStore]

    context.sync_await(
        terminology_store.create_term(
            term_set=agent.id,
            name="walnut",
            description="walnut is an altcoin",
        )
    )

    source_guideline_content = _create_guideline_content(
        "the user asks about walnut prices",
        "provide the current walnut prices",
    )

    target_guideline_content = _create_guideline_content(
        "providing altcoin prices",
        "mention that between exchanges, there can be minor differences",
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [source_guideline_content, target_guideline_content],
            )
        )
    )

    assert len(connection_propositions) == 1
    assert connection_propositions[0].source == source_guideline_content
    assert connection_propositions[0].target == target_guideline_content
    assert connection_propositions[0].kind == ConnectionKind.ENTAILS


def test_that_a_connection_is_proposed_based_on_multiple_terminology_terms(  # TODO ask Dor how to debug this
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    terminology_store = context.container[TerminologyStore]

    context.sync_await(
        terminology_store.create_term(
            term_set=agent.id,
            name="walnut",
            description="walnut is an altcoin",
        )
    )
    context.sync_await(
        terminology_store.create_term(
            term_set=agent.id,
            name="the tall tree",
            description="the tall tree is a German website for virtual goods",
        )
    )

    source_guideline_content = _create_guideline_content(
        "the user asks about getting walnuts",
        "reply that the user can get walnuts from the tall tree",
    )

    target_guideline_content = _create_guideline_content(
        "offering to purchase altcoins from a european service",
        "warn about EU regulations",
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [source_guideline_content, target_guideline_content],
            )
        )
    )

    assert len(connection_propositions) == 1
    assert connection_propositions[0].source == source_guideline_content
    assert connection_propositions[0].target == target_guideline_content
    assert connection_propositions[0].kind == ConnectionKind.ENTAILS


def test_that_one_guideline_can_entail_multiple_guidelines(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        _create_guideline_content(predicate=i["predicate"], action=i["action"])
        for i in [
            {
                "predicate": "the user asks for our catalouge",
                "action": "list the store's product and their pricings",
            },
            {
                "predicate": "listing store items",
                "action": "recommend promoted items",
            },
            {
                "predicate": "mentioning an item's price",
                "action": "remind the user about our summer discounts",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(agent, introduced_guidelines, [])
        )
    )
    assert len(connection_propositions) == 2
    assert connection_propositions[0].source == introduced_guidelines[0]
    assert connection_propositions[0].target == introduced_guidelines[1]
    assert connection_propositions[0].kind == ConnectionKind.ENTAILS
    assert connection_propositions[1].source == introduced_guidelines[0]
    assert connection_propositions[1].target == introduced_guidelines[2]
    assert connection_propositions[1].kind == ConnectionKind.ENTAILS


@mark.parametrize(
    (
        "source_guideline_definition",
        "target_guideline_definition",
    ),
    [
        (
            {
                "predicate": "the user places an order",
                "action": "direct the user to the electronic store",
            },
            {
                "predicate": "an order is being place",
                "action": "remind the user about our discounts",
            },
        ),
        (
            {
                "predicate": "A language other than English is used",
                "action": "Explain that English is the only supported language",
            },
            {
                "predicate": "Asked something in French",
                "action": "Ask the user to speak English",
            },
        ),
    ],
)
def test_that_entailing_whens_are_not_connected(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["predicate"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["predicate"],
        target_guideline_definition["action"],
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [
                    source_guideline_content,
                    target_guideline_content,
                ],
            )
        )
    )

    assert len(connection_propositions) == 0


@mark.parametrize(
    (
        "source_guideline_definition",
        "target_guideline_definition",
    ),
    [
        (
            {
                "predicate": "mentioning office hours",
                "action": "clarify that the store is closed on weekends",
            },
            {
                "predicate": "attempting to make an order on Saturday",
                "action": "clarify that the store is closed on Saturdays",
            },
        ),
        (
            {
                "predicate": "asked if an item is available in red",
                "action": "mention that the color could be changed by request",
            },
            {
                "predicate": "Asked if an item can be colored green",
                "action": "explain that it can be colored green",
            },
        ),
    ],
)
def test_that_entailing_thens_are_not_connected(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["predicate"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["predicate"],
        target_guideline_definition["action"],
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [
                    source_guideline_content,
                    target_guideline_content,
                ],
            )
        )
    )

    assert len(connection_propositions) == 0


def test_that_connection_is_proposed_for_a_sequence_where_each_guideline_entails_the_next_one(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(predicate=i["predicate"], action=i["action"])
        for i in [
            {
                "predicate": "referencing a guide",
                "action": "explain how our guides directory works",
            },
            {
                "predicate": "mentioning our guide directory",
                "action": "check the operational guide",
            },
            {
                "predicate": "checking a guide",
                "action": "Make sure that it was updated within the last year",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(agent, introduced_guidelines, [])
        )
    )

    assert len(connection_propositions) == 2
    assert connection_propositions[0].source == introduced_guidelines[0]
    assert connection_propositions[0].target == introduced_guidelines[1]
    assert connection_propositions[0].kind == ConnectionKind.ENTAILS
    assert connection_propositions[1].source == introduced_guidelines[1]
    assert connection_propositions[1].target == introduced_guidelines[2]
    assert connection_propositions[1].kind == ConnectionKind.ENTAILS


def test_that_connection_is_proposed_for_a_sequence_where_each_guideline_suggests_the_next_one(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(predicate=i["predicate"], action=i["action"])
        for i in [
            {
                "predicate": "discussing sandwiches",
                "action": "consider suggesting the daily specials",
            },
            {
                "predicate": "listing the daily specials",
                "action": "mention ingridients that may cause allergic reactions, if discussed earlier in the conversation",
            },
            {
                "predicate": "discussing anything related to a peanut allergy",  # TODO ask if this is a valid suggest connection
                "action": "Note that all dishes may contain peanut residues",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(agent, introduced_guidelines, [])
        )
    )

    assert len(connection_propositions) == 2
    assert connection_propositions[0].source == introduced_guidelines[0]
    assert connection_propositions[0].target == introduced_guidelines[1]
    assert connection_propositions[0].kind == ConnectionKind.SUGGESTS
    assert connection_propositions[1].source == introduced_guidelines[1]
    assert connection_propositions[1].target == introduced_guidelines[2]
    assert connection_propositions[1].kind == ConnectionKind.SUGGESTS


def test_that_circular_connection_is_proposed_for_three_guidelines_where_each_action_entails_the_following_predicate(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(predicate=i["predicate"], action=i["action"])
        for i in [
            {
                "predicate": "referencing a guide",
                "action": "explain how our guides directory works",
            },
            {
                "predicate": "mentioning our guide directory",
                "action": "check the operational guide",
            },
            {
                "predicate": "checking a guide",
                "action": "give reference to the guide when replying",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(agent, introduced_guidelines, [])
        )
    )

    assert len(connection_propositions) == 3
    for i, p in enumerate(connection_propositions):
        assert p.source == introduced_guidelines[i]
        assert p.target == introduced_guidelines[(1 + 1) % 3]
        assert p.kind == ConnectionKind.ENTAILS


@mark.parametrize(
    (
        "source_guideline_definition",
        "target_guideline_definition",
    ),
    [
        (
            {
                "predicate": "user is asking for specific instructions",
                "action": "consider redirecting the user to our video guides",
            },
            {
                "predicate": "mentioning a video",
                "action": "notify the user about supported video formats",
            },
        ),
    ],
)
def test_that_a_suggestive_guideline_which_entails_another_guideline_are_connected(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["predicate"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["predicate"],
        target_guideline_definition["action"],
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [source_guideline_content, target_guideline_content],
            )
        )
    )

    assert len(connection_propositions) == 1
    assert connection_propositions[0].source == source_guideline_content
    assert connection_propositions[0].target == target_guideline_content
    assert connection_propositions[0].kind == ConnectionKind.ENTAILS


@mark.parametrize(
    (
        "source_guideline_definition",
        "target_guideline_definition",
    ),
    [
        (
            {
                "predicate": "the user refers to a past interaction",
                "action": "ask for the date of this previous interaction",
            },
            {
                "predicate": "the user refers to a quota offered in a past interaction",
                "action": "answer that that quota is no longer relevant",
            },
        ),
    ],
)
def test_that_no_connection_is_made_for_a_guidelines_whose_predicate_entails_another_guidelines_predicate(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["predicate"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["predicate"],
        target_guideline_definition["action"],
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [source_guideline_content, target_guideline_content],
            )
        )
    )
    assert len(connection_propositions) == 0


@mark.parametrize(
    (
        "source_guideline_definition",
        "target_guideline_definition",
    ),
    [
        (
            {
                "predicate": "the user refers to a past interaction",
                "action": "ask the user for the date of this previous interaction",
            },
            {
                "predicate": "the user asks about a past conversation with this agent",
                "action": "ask when that conversation occured",
            },
        ),
    ],
)
def test_that_two_rephrased_guidelines_arent_connected(  # Tests both that entailing predicates and entailing actions aren't connected
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["predicate"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["predicate"],
        target_guideline_definition["action"],
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [source_guideline_content, target_guideline_content],
            )
        )
    )
    assert len(connection_propositions) == 0


@mark.parametrize(
    (
        "source_guideline_definition",
        "target_guideline_definition",
    ),
    [
        (
            {
                "predicate": "asked about pizza toppings",
                "action": "list our pizza toppings",
            },
            {
                "predicate": "asked about our menu",
                "action": "list our pizza toppings",
            },
        ),
    ],
)
def test_that_identical_actions_arent_connected(  # Tests both that entailing predicates and entailing actions aren't connected
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["predicate"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["predicate"],
        target_guideline_definition["action"],
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [source_guideline_content, target_guideline_content],
            )
        )
    )
    assert len(connection_propositions) == 0


def test_that_mispelled_entailing_guidelines_are_connected(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    terminology_store = context.container[TerminologyStore]

    context.sync_await(
        terminology_store.create_term(
            term_set=agent.id,
            name="walnut",
            description="walnut is an altcoin",
        )
    )

    source_guideline_content = _create_guideline_content(
        "the user ask about wallnut prices",
        "provide the curent walnut prices",
    )

    target_guideline_content = _create_guideline_content(
        "provding altcoinn prices",
        "mention that between exchanges, there can be minor differences",
    )

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [source_guideline_content, target_guideline_content],
            )
        )
    )

    assert len(connection_propositions) == 1
    assert connection_propositions[0].source == source_guideline_content
    assert connection_propositions[0].target == target_guideline_content
    assert connection_propositions[0].kind == ConnectionKind.ENTAILS
