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

from datetime import datetime, timezone
from typing import Sequence


from parlant.core.agents import Agent, AgentId
from parlant.core.glossary import GlossaryStore
from parlant.core.guideline_connections import ConnectionKind
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
)

from tests.core.stable.services.indexing.test_guideline_connection_proposer import (
    base_test_that_entailing_whens_are_not_connected,
    base_test_that_a_suggestive_guideline_which_entails_another_guideline_are_connected_as_suggestive,
)

from tests.core.common.services.indexing.conftest import (
    _TestContext,
    _create_guideline_content,
    context,
)
from tests.core.common.conftest import agent


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
