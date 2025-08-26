# Copyright 2025 Emcie Co Ltd.
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

from datetime import datetime, timezone
from typing import Sequence


from parlant.core.agents import Agent, AgentId
from parlant.core.glossary import GlossaryStore
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
)
from parlant.core.tags import Tag
from tests.core.common.utils import ContextOfTest


def base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other(
    context: ContextOfTest,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        source_guideline_definition["condition"],
        source_guideline_definition["action"],
    )

    target_guideline_content = GuidelineContent(
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


def test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other_parametrized_1(
    context: ContextOfTest,
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
    context: ContextOfTest,
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


def test_that_a_connection_is_proposed_for_two_guidelines_where_the_content_of_one_suggests_a_follow_up_to_the_condition_of_the_other_parametrized_3(
    context: ContextOfTest,
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
    base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_multiple_connections_are_detected_and_proposed_at_the_same_time(
    context: ContextOfTest,
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
    context: ContextOfTest,
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
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    glossary_store = context.container[GlossaryStore]

    term = context.sync_await(
        glossary_store.create_term(
            name="walnut",
            description="walnut is an altcoin",
        )
    )

    context.sync_await(
        glossary_store.upsert_tag(
            term_id=term.id,
            tag_id=Tag.for_agent_id(agent.id),
        )
    )
    source_guideline_content = GuidelineContent(
        "the customer asks about walnut prices",
        "provide the current walnut prices",
    )

    target_guideline_content = GuidelineContent(
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


def test_that_a_connection_is_proposed_based_on_multiple_glossary_terms(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    glossary_store = context.container[GlossaryStore]

    first_term = context.sync_await(
        glossary_store.create_term(
            name="walnut",
            description="walnut is an altcoin",
        )
    )
    second_term = context.sync_await(
        glossary_store.create_term(
            name="the tall tree",
            description="the tall tree is a German website for purchasing virtual goods",
        )
    )

    context.sync_await(
        glossary_store.upsert_tag(
            term_id=first_term.id,
            tag_id=Tag.for_agent_id(agent.id),
        )
    )

    context.sync_await(
        glossary_store.upsert_tag(
            term_id=second_term.id,
            tag_id=Tag.for_agent_id(agent.id),
        )
    )

    source_guideline_content = GuidelineContent(
        "the customer asks about getting walnuts",
        "reply that the customer can buy walnuts from the tall tree",
    )

    target_guideline_content = GuidelineContent(
        "suggesting to purchase altcoins from a european service",
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


def test_that_one_guideline_can_entail_multiple_guidelines(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(condition=i["condition"], action=i["action"])
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
    assert connection_propositions[1].source == introduced_guidelines[0]
    assert connection_propositions[1].target == introduced_guidelines[2]


def base_test_that_entailing_whens_are_not_connected(
    context: ContextOfTest,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        source_guideline_definition["condition"],
        source_guideline_definition["action"],
    )

    target_guideline_content = GuidelineContent(
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
    context: ContextOfTest,
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
    context: ContextOfTest,
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
    context: ContextOfTest,
    agent: Agent,
    source_guideline_definition: dict[str, str],
    target_guideline_definition: dict[str, str],
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        source_guideline_definition["condition"],
        source_guideline_definition["action"],
    )

    target_guideline_content = GuidelineContent(
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
    context: ContextOfTest,
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
    context: ContextOfTest,
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
    context: ContextOfTest,
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
    assert connection_propositions[1].source == introduced_guidelines[1]
    assert connection_propositions[1].target == introduced_guidelines[2]


def test_that_connection_is_proposed_for_a_sequence_where_each_guideline_entails_the_next_one(
    context: ContextOfTest,
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
    assert connection_propositions[1].source == introduced_guidelines[1]
    assert connection_propositions[1].target == introduced_guidelines[2]


def test_that_circular_connection_is_proposed_for_three_guidelines_where_each_action_entails_the_following_condition(
    context: ContextOfTest,
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


def base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other_parametrized_4(
    context: ContextOfTest,
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
    base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other_parametrized_5(
    context: ContextOfTest,
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
    base_test_that_an_entailment_connection_is_proposed_for_two_guidelines_where_the_content_of_one_entails_the_condition_of_the_other(
        context, agent, source_guideline_definition, target_guideline_definition
    )


def test_that_no_connection_is_made_for_a_guidelines_whose_condition_entails_another_guidelines_condition(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        "the customer refers to a past interaction",
        "ask for the date of this previous interaction",
    )

    target_guideline_content = GuidelineContent(
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
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        "The customer complains that the phrases in the photograph are blurry",
        "clarify what the unclear phrases mean",
    )

    target_guideline_content = GuidelineContent(
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
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        "the customer refers to a past interaction",
        "ask the customer for the date of this interaction",
    )

    target_guideline_content = GuidelineContent(
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
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        "asked about pizza toppings",
        "list our pizza toppings",
    )

    target_guideline_content = GuidelineContent(
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
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    glossary_store = context.container[GlossaryStore]

    term = context.sync_await(
        glossary_store.create_term(
            name="walnut",
            description="walnut is an altcoin",
        )
    )

    context.sync_await(
        glossary_store.upsert_tag(
            term_id=term.id,
            tag_id=Tag.for_agent_id(agent.id),
        )
    )

    source_guideline_content = GuidelineContent(
        "the customer ask about wallnut prices",
        "provide the curent walnut prices",
    )

    target_guideline_content = GuidelineContent(
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


def test_that_try_actions_are_connected(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        "the customer complains that a suggested solution did not work",
        "reply in a hostile manner",
    )

    target_guideline_content = GuidelineContent(
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


def test_that_agent_based_connection_is_detected(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]
    agent = Agent(
        id=AgentId("sparkling-water-agent"),
        name="sparkling-water-agent",
        description="You are a helpful AI assistant for a sparkling water company. Our philosophy dictates that oranges are vegetables and not fruit, despite what other may say.",
        creation_utc=datetime.now(timezone.utc),
        max_engine_iterations=3,
        tags=[],
    )
    source_guideline_content = GuidelineContent(
        "the customer asks for drink recommendation",
        "recommend orange soda",
    )

    target_guideline_content = GuidelineContent(
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


def test_that_many_guidelines_with_agent_description_and_glossary_arent_detected_as_false_positives(
    context: ContextOfTest,
) -> None:
    agent = Agent(
        id=AgentId("Sparkleton Agent"),
        creation_utc=datetime.now(timezone.utc),
        name="Sparkleton Agent",
        description="You're an AI assistant to a sparkling water expert at Sparkleton. The expert may consult you while talking to potential clients to retrieve important information from Sparkleton's documentation.",
        max_engine_iterations=3,
        tags=[],
    )

    glossary_store = context.container[GlossaryStore]

    first_term = context.sync_await(
        glossary_store.create_term(
            name="Sparkleton",
            description="The top sparkling water company in the world",
            synonyms=["sparkleton", "sparkletown", "the company"],
        )
    )
    context.sync_await(
        glossary_store.upsert_tag(
            term_id=first_term.id,
            tag_id=Tag.for_agent_id(agent.id),
        )
    )

    second_term = context.sync_await(
        glossary_store.create_term(
            name="tomatola",
            description="A type of cola made out of tomatoes",
            synonyms=["tomato cola"],
        )
    )
    context.sync_await(
        glossary_store.upsert_tag(
            term_id=second_term.id,
            tag_id=Tag.for_agent_id(agent.id),
        )
    )

    third_term = context.sync_await(
        glossary_store.create_term(
            name="carbon coin",
            description="a virtual currency awarded to customers. Can be used to buy any Sparkleton product",
            synonyms=["cc", "C coin"],
        )
    )
    context.sync_await(
        glossary_store.upsert_tag(
            term_id=third_term.id,
            tag_id=Tag.for_agent_id(agent.id),
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


def test_that_entailed_predicate_describing_an_agent_action_is_detected(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        condition="Asked for our pizza topping selection",
        action="list the possible toppings and recommend olives",
    )
    target_guideline_content = GuidelineContent(
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


def test_that_strict_entailment_due_to_the_sources_condition_is_detected_1(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        condition="A US based customer attempts to purchase a ladder",
        action="Ask how tall the ladder should be",
    )
    target_guideline_content = GuidelineContent(
        condition="discussing sizes with an American customer", action="Use imperial units"
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


def test_that_strict_entailment_due_to_the_sources_condition_is_detected_2(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        condition="a non-guest customer greets you",
        action="refer to them by their first name, and welcome them 'back'",
    )
    target_guideline_content = GuidelineContent(
        condition="you're welcoming non-guest customers",
        action="refer to them by their first name, and welcome them 'back'",
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


def test_that_an_action_that_causes_a_more_general_case_of_another_guidelines_source_is_not_detected(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "the customer needs help unlocking their card",
                "action": "ask for the last 6 digits and help them unlock",
            },
            {
                "condition": "you have tried to unlock the customer's card but failed due to invalid customer ID",
                "action": "explain the issue and ask them to try to log out and back into the app again",
            },
            {
                "condition": "you have tried to unlock the customer's card but failed due to customer 'ineligibility'",
                "action": "explain the issue and ask them to have a parent contact the bank",
            },
            {
                "condition": "you have tried to unlock the customer's card but failed due to not finding the card",
                "action": "explain the issue and ask them to double check the card number and that it is indeed a card that's associated with their account",
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
