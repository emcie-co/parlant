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

from typing import Sequence


from parlant.core.agents import Agent
from parlant.core.glossary import GlossaryStore
from parlant.core.guideline_connections import ConnectionKind
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
)

from tests.core.common.services.indexing.conftest import (
    _TestContext,
    _create_guideline_content,
    context,
)
from tests.core.common.conftest import agent


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
