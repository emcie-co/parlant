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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent, AgentId
from parlant.core.glossary import GlossaryStore
from parlant.core.guideline_connections import ConnectionKind
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
)

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
    condition: str,
    action: str,
) -> GuidelineContent:
    return GuidelineContent(condition=condition, action=action)


def base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["condition"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["condition"],
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


def test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other_parametrized_1(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "condition": "the customer asks about the weather",
        "action": "provide the current weather update",
    }
    target_guideline_definition: dict[str, str] = {
        "condition": "providing the weather update",
        "action": "mention the best time to go for a walk",
    }
    base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other_parametrized_2(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "condition": "the customer asks about nearby restaurants",
        "action": "provide a list of popular restaurants",
    }
    target_guideline_definition: dict[str, str] = {
        "condition": "listing restaurants",
        "action": "highlight the one with the best reviews",
    }
    base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def base_test_that_a_suggestion_connection_is_proposed_for_two_guidelines_where_the_content_of_one_suggests_a_follow_up_to_the_condition_of_the_other(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["condition"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["condition"],
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


def test_that_a_suggestion_connection_is_proposed_for_two_guidelines_where_the_content_of_one_suggests_a_follow_up_to_the_condition_of_the_other_parametrized_1(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "guideline_set": "test-agent",
        "condition": "the customer requests technical support",
        "action": "provide the support contact details",
    }
    target_guideline_definition: dict[str, str] = {
        "guideline_set": "test-agent",
        "condition": "providing support contact details",
        "action": "consider checking the troubleshooting guide first",
    }
    base_test_that_a_suggestion_connection_is_proposed_for_two_guidelines_where_the_content_of_one_suggests_a_follow_up_to_the_condition_of_the_other(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_a_suggestion_connection_is_proposed_for_two_guidelines_where_the_content_of_one_suggests_a_follow_up_to_the_condition_of_the_other_parametrized_2(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "guideline_set": "test-agent",
        "condition": "the customer inquires about office hours",
        "action": "tell them the office hours",
    }
    target_guideline_definition: dict[str, str] = {
        "guideline_set": "test-agent",
        "condition": "mentioning office hours",
        "action": "you may suggest the best time to visit for quicker service",
    }
    base_test_that_a_suggestion_connection_is_proposed_for_two_guidelines_where_the_content_of_one_suggests_a_follow_up_to_the_condition_of_the_other(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_multiple_connections_are_detected_and_proposed_at_the_same_time(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "the customer requests technical support",
                "action": "provide the support contact details",
            },
            {
                "condition": "providing support contact details",
                "action": "consider checking the troubleshooting guide first",
            },
            {
                "condition": "the customer inquires about office hours",
                "action": "tell them the office hours",
            },
            {
                "condition": "mentioning office hours",
                "action": "suggest the best time to visit for quicker service",
            },
            {
                "condition": "the customer asks about the weather",
                "action": "provide the current weather update",
            },
            {
                "condition": "providing the weather update",
                "action": "mention the best time to go for a walk",
            },
            {
                "condition": "the customer asks about nearby restaurants",
                "action": "provide a list of popular restaurants",
            },
            {
                "condition": "listing restaurants",
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
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "the customer requests technical support",
                "action": "provide the support contact details",
            },
            {
                "condition": "providing support contact details",
                "action": "consider checking the troubleshooting guide first",
            },
            {
                "condition": "the customer inquires about office hours",
                "action": "tell them the office hours",
            },
            {
                "condition": "mentioning office hours",
                "action": "suggest the best time to visit for quicker service",
            },
            {
                "condition": "the customer asks about the weather",
                "action": "provide the current weather update",
            },
            {
                "condition": "providing the weather update",
                "action": "mention the best time to go for a walk",
            },
            {
                "condition": "the customer asks about nearby restaurants",
                "action": "provide a list of popular restaurants",
            },
            {
                "condition": "listing restaurants",
                "action": "highlight the one with the best reviews",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(connection_proposer.propose_connections(agent, [], existing_guidelines))
    )

    assert len(connection_propositions) == 0


def test_that_a_connection_is_proposed_based_on_given_glossary(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    glossary_store = context.container[GlossaryStore]

    context.sync_await(
        glossary_store.create_term(
            term_set=agent.id,
            name="walnut",
            description="walnut is an altcoin",
        )
    )

    source_guideline_content = _create_guideline_content(
        "the customer asks about walnut prices",
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


def test_that_a_connection_is_proposed_based_on_multiple_glossary_terms(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    glossary_store = context.container[GlossaryStore]

    context.sync_await(
        glossary_store.create_term(
            term_set=agent.id,
            name="walnut",
            description="walnut is an altcoin",
        )
    )
    context.sync_await(
        glossary_store.create_term(
            term_set=agent.id,
            name="the tall tree",
            description="the tall tree is a German website for purchasing virtual goods",
        )
    )

    source_guideline_content = _create_guideline_content(
        "the customer asks about getting walnuts",
        "reply that the customer can buy walnuts from the tall tree",
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
        _create_guideline_content(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "the customer asks for our catalouge",
                "action": "list the store's product and their pricings",
            },
            {
                "condition": "listing store items",
                "action": "recommend promoted items",
            },
            {
                "condition": "mentioning an item's price",
                "action": "remind the customer about our summer discounts",
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


def base_test_that_entailing_whens_are_not_connected(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["condition"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["condition"],
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


def test_that_entailing_whens_are_not_connected_parametrized_1(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "condition": "the customer places an order",
        "action": "direct the customer to the electronic store",
    }
    target_guideline_definition: dict[str, str] = {
        "condition": "the customer is ordering electronic goods",
        "action": "remind the customer about our discounts",
    }
    base_test_that_entailing_whens_are_not_connected(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_entailing_whens_are_not_connected_parametrized_2(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "condition": "asked about supported languages",
        "action": "explain that English is the only supported language",
    }
    target_guideline_definition: dict[str, str] = {
        "condition": "the customer uses a language other than English",
        "action": "refer them to our international website",
    }
    base_test_that_entailing_whens_are_not_connected(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def base_test_that_entailing_thens_are_not_connected(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["condition"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["condition"],
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


def test_that_entailing_thens_are_not_connected_parametrized_1(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "condition": "mentioning office hours",
        "action": "clarify that the store is closed on weekends",
    }
    target_guideline_definition: dict[str, str] = {
        "condition": "attempting to make an order on Saturday",
        "action": "clarify that the store is closed on Saturdays",
    }
    base_test_that_entailing_thens_are_not_connected(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_entailing_thens_are_not_connected_parametrized_2(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "condition": "asked if an item is available in red",
        "action": "mention that the color could be changed by request",
    }
    target_guideline_definition: dict[str, str] = {
        "condition": "Asked if an item can be colored green",
        "action": "explain that it can be colored green",
    }
    base_test_that_entailing_thens_are_not_connected(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_connection_is_proposed_for_a_sequence_where_each_guideline_entails_the_next_one_using_pronouns_from_then_to_when(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "the customer says hello",
                "action": "say you like bananas",
            },
            {
                "condition": "talking about bananas",
                "action": "say that they're sweet this season",
            },
            {
                "condition": "you say that bananas are sweet",
                "action": "say they're even sweeter than mangoes",
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


def test_that_connection_is_proposed_for_a_sequence_where_each_guideline_entails_the_next_one(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "directing the customer to a guide",
                "action": "explain how our guides directory works",
            },
            {
                "condition": "mentioning our guide directory",
                "action": "check the operational guide",
            },
            {
                "condition": "checking a guide",
                "action": "Make sure that the guide was updated within the last year",
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
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "discussing sandwiches",
                "action": "recommend the daily specials",
            },
            {
                "condition": "listing the daily specials",
                "action": "consider mentioning ingredients that may cause allergic reactions",
            },
            {
                "condition": "discussing anything related to a food allergies",
                "action": "you may note that all dishes may contain peanut residues",
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


def test_that_circular_connection_is_proposed_for_three_guidelines_where_each_action_entails_the_following_condition(
    context: _TestContext,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "referencing a guide to the customer",
                "action": "explain how our guides directory works",
            },
            {
                "condition": "mentioning our guide directory",
                "action": "check the operational guide",
            },
            {
                "condition": "checking a guide",
                "action": "direct the customer to the guide when replying",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(agent, introduced_guidelines, [])
        )
    )

    correct_propositions_set = {
        (introduced_guidelines[i], introduced_guidelines[(i + 1) % 3]) for i in range(3)
    }
    suggested_propositions_set = {(p.source, p.target) for p in connection_propositions}
    assert correct_propositions_set == suggested_propositions_set


def base_test_that_a_suggestive_guideline_which_entails_another_guideline_are_connected_as_suggestive(
    context: _TestContext,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        source_guideline_definition["condition"],
        source_guideline_definition["action"],
    )

    target_guideline_content = _create_guideline_content(
        target_guideline_definition["condition"],
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
    assert connection_propositions[0].kind == ConnectionKind.SUGGESTS


def test_that_a_suggestive_guideline_which_entails_another_guideline_are_connected_as_suggestive_parametrized_1(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "condition": "customer is asking for specific instructions",
        "action": "consider redirecting the customer to our video guides",
    }
    target_guideline_definition: dict[str, str] = {
        "condition": "mentioning a video",
        "action": "notify the customer about supported video formats",
    }
    base_test_that_a_suggestive_guideline_which_entails_another_guideline_are_connected_as_suggestive(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_a_suggestive_guideline_which_entails_another_guideline_are_connected_as_suggestive_parametrized_2(
    context: _TestContext,
    agent: Agent,
) -> None:
    source_guideline_definition: dict[str, str] = {
        "guideline_set": "test-agent",
        "condition": "the customer asks for express shipping",
        "action": "check if express delivery is avialable and reply positively only if it is",  # Keeping the mispelling intentionally
    }
    target_guideline_definition: dict[str, str] = {
        "guideline_set": "test-agent",
        "condition": "offering express delivery",
        "action": "mention it takes up to 48 hours",
    }
    base_test_that_a_suggestive_guideline_which_entails_another_guideline_are_connected_as_suggestive(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_no_connection_is_made_for_a_guidelines_whose_condition_entails_another_guidelines_condition(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        "the customer refers to a past interaction",
        "ask for the date of this previous interaction",
    )

    target_guideline_content = _create_guideline_content(
        "the customer refers to a quota offered in a past interaction",
        "answer that that quota is no longer relevant",
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


def test_that_no_connection_is_made_for_a_guideline_which_implies_but_not_causes_another_guideline(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        "The customer complains that the phrases in the photograph are blurry",
        "clarify what the unclear phrases mean",
    )

    target_guideline_content = _create_guideline_content(
        "a word is misunderstood",
        "reply with its dictionary definition",
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


def test_that_guidelines_with_similar_thens_arent_connected(  # Tests both that entailing conditions and entailing actions aren't connected
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        "the customer refers to a past interaction",
        "ask the customer for the date of this interaction",
    )

    target_guideline_content = _create_guideline_content(
        "the customer asks about a solution suggested in a previous interaction",
        "ask when that conversation occurred",
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


def test_that_identical_actions_arent_connected(  # Tests both that entailing conditions and entailing actions aren't connected
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        "asked about pizza toppings",
        "list our pizza toppings",
    )

    target_guideline_content = _create_guideline_content(
        "asked about our menu",
        "list our pizza toppings",
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


def test_that_misspelled_entailing_guidelines_are_connected(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    glossary_store = context.container[GlossaryStore]

    context.sync_await(
        glossary_store.create_term(
            term_set=agent.id,
            name="walnut",
            description="walnut is an altcoin",
        )
    )

    source_guideline_content = _create_guideline_content(
        "the customer ask about wallnut prices",
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


def test_that_try_actions_are_connected_but_not_suggestive(  # Tests both that entailing conditions and entailing actions aren't connected
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        "the customer complains that a suggested solution did not work",
        "reply in a hostile manner",
    )

    target_guideline_content = _create_guideline_content(
        "the conversational tone is hostile",
        "try to de-escalate the situation",
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


def test_that_agent_based_connection_is_detected(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    agent = Agent(
        id=AgentId("sparkling-water-agent"),
        name="sparkling-water-agent",
        description="You are a helpful AI assistant for a sparkling water company. Our philosophy dictates that oranges are vegetables and not fruit, despite what other may say.",
        creation_utc=datetime.now(timezone.utc),
        max_engine_iterations=3,
    )
    source_guideline_content = _create_guideline_content(
        "the customer asks for drink recommendation",
        "recommend orange soda",
    )

    target_guideline_content = _create_guideline_content(
        "recommending a vegetable based soda",
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


def test_that_many_guidelines_with_agent_description_and_glossary_arent_detected_as_false_positives(  # This test fails occasionally
    context: _TestContext,
) -> None:
    agent = Agent(
        id=AgentId("Sparkleton Agent"),
        creation_utc=datetime.now(timezone.utc),
        name="Sparkleton Agent",
        description="You're an AI assistant to a sparkling water expert at Sparkleton. The expert may consult you while talking to potential clients to retrieve important information from Sparkleton's documentation.",
        max_engine_iterations=3,
    )

    glossary_store = context.container[GlossaryStore]

    context.sync_await(
        glossary_store.create_term(
            term_set=agent.id,
            name="Sparkleton",
            description="The top sparkling water company in the world",
            synonyms=["sparkleton", "sparkletown", "the company"],
        )
    )
    context.sync_await(
        glossary_store.create_term(
            term_set=agent.id,
            name="tomatola",
            description="A type of cola made out of tomatoes",
            synonyms=["tomato cola"],
        )
    )
    context.sync_await(
        glossary_store.create_term(
            term_set=agent.id,
            name="carbon coin",
            description="a virtual currency awarded to customers. Can be used to buy any Sparkleton product",
            synonyms=["cc", "C coin"],
        )
    )

    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "asked a clarifying question",
                "action": "Keep your answer short and direct",
            },
            {
                "condition": "The customer asks about carbon coin",
                "action": "Always check the carbon coin terms of use before replying. Do not reply with anything that is not explicitly mentioned in the terms of use.",
            },
            {
                "condition": "The customer seems to be short on time",
                "action": "suggest continuing the conversation at another time",
            },
            {
                "condition": "The customer asked a question that's not mentioned in the terms of use document",
                "action": "Forward the customer's question to management and inform them that you'll get back to them later",
            },
            {
                "condition": "The customer asks you if you're confident in your reply",
                "action": "Reply that you are extremely confident, as you're the best ai agent in the world",
            },
            {
                "condition": "The customer asks for ways of earning carbon coin",
                "action": "Answer the customer's question based on the documentation. Be clear that the coin can only be used on Sparkleton products",
            },
            {
                "condition": "The customer asks if tomatola is available",
                "action": "Check the inventory and reply accordingly",
            },
            {
                "condition": "The customer inquires about anything that doesn't have to do with sparkling drinks",
                "action": "Let the customer know that you are not trained to help with subjects not related to Sparkleton.",
            },
            {
                "condition": "The customer asks further question about an answer you previously provided",
                "action": "Answer the question, even if it's not related to Sparkleton",
            },
            {
                "condition": "The customer asks multiple questions in one message",
                "action": "Split the message into each individual question, and reply to each question in a new message.",
            },
            {
                "condition": "The customer asks for further clarification",
                "action": "Provide a link to the relevant document in full",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(agent, introduced_guidelines, [])
        )
    )

    assert len(connection_propositions) == 0


def test_that_strictly_entailed_predicate_describing_an_agent_action_is_detected(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        condition="Asked for our pizza topping selection",
        action="list the possible toppings and recommend olives",
    )
    target_guideline_content = _create_guideline_content(
        condition="Recommending pizza toppings", action="Recommend mushrooms as they are healthy"
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


def test_that_strict_entailment_due_to_the_sources_condition_is_detected(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        condition="Asked for our pizza topping selection",
        action="list the possible toppings and recommend olives",
    )
    target_guideline_content = _create_guideline_content(
        condition="Recommending pizza toppings", action="Recommend mushrooms as they are healthy"
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


def test_that_suggestive_entailment_due_to_the_sources_condition_is_detected(
    context: _TestContext,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = _create_guideline_content(
        condition="Asked for our pizza topping selection",
        action="list the possible toppings and recommend olives",
    )
    target_guideline_content = _create_guideline_content(
        condition="Recommending pizza toppings", action="Recommend mushrooms as they are healthy"
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
