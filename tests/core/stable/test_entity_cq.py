# Copyright 2026 Emcie Co Ltd.
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

import random
from lagom import Container

from parlant.core.agents import Agent, AgentStore
from parlant.core.capabilities import CapabilityStore
from parlant.core.entity_cq import EntityQueries
from parlant.core.glossary import GlossaryStore
from parlant.core.journey_guideline_projection import JourneyGuidelineProjection
from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipStore,
    RelationshipKind,
    RelationshipEntityKind,
)
from parlant.core.canned_responses import CannedResponseStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore
from parlant.core.tags import Tag, TagId, TagStore


async def test_that_list_guidelines_with_mutual_agent_tag_are_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    agent_store = container[AgentStore]
    guideline_store = container[GuidelineStore]

    await agent_store.upsert_tag(
        agent_id=agent.id,
        tag_id=TagId("tag_1"),
    )

    first_guideline = await guideline_store.create_guideline(
        condition="condition 1",
        action="action 1",
    )

    second_guideline = await guideline_store.create_guideline(
        condition="condition 2",
        action="action 2",
    )

    await guideline_store.upsert_tag(
        guideline_id=first_guideline.id,
        tag_id=TagId("tag_1"),
    )

    await guideline_store.upsert_tag(
        guideline_id=second_guideline.id,
        tag_id=TagId("tag_2"),
    )

    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    assert len(result) == 1
    assert result[0].id == first_guideline.id


async def test_that_list_guidelines_global_guideline_is_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]

    global_guideline = await guideline_store.create_guideline(
        condition="condition 1",
        action="action 1",
    )

    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    assert len(result) == 1
    assert result[0].id == global_guideline.id


async def test_that_guideline_with_not_hierarchy_tag_is_not_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]

    first_guideline = await guideline_store.create_guideline(
        condition="condition 1",
        action="action 1",
    )

    second_guideline = await guideline_store.create_guideline(
        condition="condition 2",
        action="action 2",
    )

    await guideline_store.upsert_tag(
        guideline_id=first_guideline.id,
        tag_id=Tag.for_agent_id(agent.id),
    )

    await guideline_store.upsert_tag(
        guideline_id=second_guideline.id,
        tag_id=TagId("tag_2"),
    )

    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    assert len(result) == 1
    assert result[0].id == first_guideline.id


async def test_that_guideline_matches_are_not_filtered_by_enabled_journeys(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    journey_guideline = await guideline_store.create_guideline(
        condition="condition 1",
    )

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[journey_guideline.id],
    )

    guideline = await guideline_store.create_guideline(
        condition="condition 2",
    )

    await guideline_store.upsert_tag(
        guideline_id=journey_guideline.id,
        tag_id=Tag.for_journey_id(journey.id),
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_journey_id(journey.id),
    )

    result = await entity_queries.find_guidelines_for_context(
        agent.id,
        [journey],
    )

    assert len(result) == 2
    assert any(journey_guideline.id == g.id for g in result)
    assert any(guideline.id == g.id for g in result)


async def test_that_guideline_tagged_with_disabled_journey_is_filtered_out_when_matched(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    journey_guideline = await guideline_store.create_guideline(
        condition="condition 1",
    )

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[journey_guideline.id],
    )

    guideline = await guideline_store.create_guideline(
        condition="condition 2",
    )

    await guideline_store.upsert_tag(
        guideline_id=journey_guideline.id,
        tag_id=Tag.for_journey_id(journey.id),
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_journey_id(journey.id),
    )

    result = await entity_queries.find_guidelines_for_context(
        agent.id,
        [],
    )

    assert len(result) == 0


async def test_that_find_canned_responses_for_agent_returns_global_canned_responses(
    container: Container,
    agent: Agent,
) -> None:
    canrep_store: CannedResponseStore = container[CannedResponseStore]
    entity_queries = container[EntityQueries]

    untagged_canrep = await canrep_store.create_canned_response(
        value="Hello world",
        fields=[],
    )

    results = await entity_queries.find_canned_responses_for_context(
        agent=agent,
        journeys=[],
        guidelines=[],
    )
    assert len(results) == 1
    assert results[0].id == untagged_canrep.id


async def test_that_find_canned_responses_for_agent_returns_none_for_non_matching_tag(
    container: Container, agent: Agent
) -> None:
    canrep_store: CannedResponseStore = container[CannedResponseStore]
    entity_queries = container[EntityQueries]

    tag1 = TagId("tag1")
    await canrep_store.create_canned_response(
        value="Tagged canned response",
        fields=[],
        tags=[tag1],
    )

    await container[AgentStore].upsert_tag(agent_id=agent.id, tag_id=TagId("non_matching_tag"))

    results = await entity_queries.find_canned_responses_for_context(
        agent=agent,
        journeys=[],
        guidelines=[],
    )
    assert len(results) == 0


async def test_that_find_canned_responses_for_agent_and_journey_returns_journey_canned_responses(
    container: Container, agent: Agent
) -> None:
    canrep_store: CannedResponseStore = container[CannedResponseStore]
    journey_store = container[JourneyStore]
    entity_queries = container[EntityQueries]

    journey = await journey_store.create_journey(
        title="Test Journey",
        description="A test journey",
        conditions=[],
    )

    journey_tag = Tag.for_journey_id(journey.id)
    journey_canrep = await canrep_store.create_canned_response(
        value="Journey canrep",
        fields=[],
        tags=[journey_tag],
    )

    results = await entity_queries.find_canned_responses_for_context(
        agent=agent,
        journeys=[journey],
        guidelines=[],
    )
    assert len(results) == 1
    assert results[0].id == journey_canrep.id


async def test_that_find_glossary_terms_for_agent_returns_all_when_no_tags(
    container: Container,
    agent: Agent,
) -> None:
    glossary_store = container[GlossaryStore]
    entity_queries = container[EntityQueries]

    untagged_term = await glossary_store.create_term(
        name="Hello world",
        description="A greeting",
        tags=[],
    )

    tag = TagId("tag1")
    await glossary_store.create_term(
        name="Tagged term",
        description="A tagged glossary entry",
        tags=[tag],
    )

    results = await entity_queries.find_glossary_terms_for_context(agent_id=agent.id, query="Hello")
    assert len(results) == 1
    assert results[0].id == untagged_term.id


async def test_that_find_glossary_terms_for_agent_returns_none_for_non_matching_tag(
    container: Container,
    agent: Agent,
) -> None:
    glossary_store = container[GlossaryStore]
    entity_queries = container[EntityQueries]

    tag1 = TagId("tag1")
    await glossary_store.create_term(
        name="Tagged term",
        description="A tagged glossary entry",
        tags=[tag1],
    )

    await container[AgentStore].upsert_tag(agent_id=agent.id, tag_id=TagId("non_matching_tag"))

    results = await entity_queries.find_glossary_terms_for_context(
        agent_id=agent.id, query="Tagged"
    )
    assert len(results) == 0


async def test_that_find_capabilities_for_agent_returns_unique_capabilities(
    container: Container,
    agent: Agent,
) -> None:
    def random_unicode_string() -> str:
        return "".join(chr(random.randint(0, 255)) for _ in range(10))

    capability_store = container[CapabilityStore]
    entity_queries = container[EntityQueries]

    for i in range(10):
        capability = {
            "title": random_unicode_string(),
            "description": random_unicode_string(),
            "signals": [random_unicode_string() for _ in range(5)],
        }

        await capability_store.create_capability(
            title=str(capability["title"]),
            description=str(capability["description"]),
            signals=capability["signals"],
        )

    relevant_capabilities = await entity_queries.find_capabilities_for_agent(
        agent_id=agent.id,
        query=random_unicode_string(),
        max_count=3,
    )

    assert len(relevant_capabilities) == 3
    assert len({c.id for c in relevant_capabilities}) == 3


async def test_find_relevant_journeys_for_agent_returns_most_relevant(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    condition = await guideline_store.create_guideline(
        condition="the customer wants to reset their password",
    )

    onboarding_journey = await journey_store.create_journey(
        title="Reset Password Journey",
        description="""follow these steps to reset a customers password:
        1. ask for their account name
        2. ask for their email or phone number
        3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort.
        4. use the tool reset_password with the provided information
        5. report the result to the customer""",
        conditions=[condition.id],
    )

    support_journey = await journey_store.create_journey(
        title="Change Credit Limits",
        description="Remember that credit limits can be decreased through this chat, using the decrease_limits tool, but that to increase credit limits you must visit a physical branch",
        conditions=[],
    )

    results = await entity_queries.sort_journeys_by_contextual_relevance(
        [onboarding_journey, support_journey], "I'd like to reset my password"
    )

    assert len(results) == 2
    assert results[0].id == onboarding_journey.id
    assert results[1].id == support_journey.id


async def test_list_guidelines_dependent_directly_on_journey(
    container: Container,
) -> None:
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    relationship_store = container[RelationshipStore]

    journey = await journey_store.create_journey(
        title="Test Journey",
        description="A journey for testing dependencies",
        conditions=[],
    )

    guideline1 = await guideline_store.create_guideline(
        condition="condition 1",
        action="action 1",
    )
    _ = await guideline_store.create_guideline(
        condition="condition 2",
        action="action 2",
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=guideline1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(
            id=Tag.for_journey_id(journey.id), kind=RelationshipEntityKind.TAG
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await entity_queries.find_journey_related_guidelines(journey)

    assert len(result) == 1
    assert result[0] == guideline1.id


async def test_list_guidelines_dependent_indirectly_on_journey(
    container: Container,
) -> None:
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    relationship_store = container[RelationshipStore]
    tag_store = container[TagStore]

    journey = await journey_store.create_journey(
        title="Test Journey",
        description="A journey for testing dependencies",
        conditions=[],
    )

    guideline1 = await guideline_store.create_guideline(
        condition="condition 1",
        action="action 1",
    )
    guideline2 = await guideline_store.create_guideline(
        condition="condition 2",
        action="action 2",
    )
    guideline3 = await guideline_store.create_guideline(
        condition="condition 3",
        action="action 3",
    )
    tag = await tag_store.create_tag(name="test tag")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=guideline1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(
            id=Tag.for_journey_id(journey.id), kind=RelationshipEntityKind.TAG
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=guideline2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=guideline1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=guideline3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=tag.id, kind=RelationshipEntityKind.TAG),
        kind=RelationshipKind.DEPENDENCY,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=tag.id, kind=RelationshipEntityKind.TAG),
        target=RelationshipEntity(
            id=Tag.for_journey_id(journey.id), kind=RelationshipEntityKind.TAG
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await entity_queries.find_journey_related_guidelines(journey)

    assert len(result) == 3

    assert any(guideline1.id == g for g in result)
    assert any(guideline2.id == g for g in result)
    assert any(guideline3.id == g for g in result)


async def test_that_canned_responses_can_be_found_for_a_guideline(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    canned_response_store = container[CannedResponseStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    g1 = await guideline_store.create_guideline(
        condition="condition 1",
        action="action 1",
    )

    g2 = await guideline_store.create_guideline(
        condition="condition 2",
        action="action 2",
    )

    journey = await journey_store.create_journey(
        title="Test Journey",
        description="A journey for testing canned responses",
        conditions=[],
    )

    node = await journey_store.create_node(
        journey_id=journey.id,
        action="Test Node",
        tools=[],
    )

    await journey_store.create_edge(
        journey_id=journey.id,
        source=journey.root_id,
        target=node.id,
        condition=None,
    )

    projection = await container[JourneyGuidelineProjection].project_journey_to_guidelines(
        journey_id=journey.id,
    )

    assert len(projection) == 2

    canrep_1 = await canned_response_store.create_canned_response(
        value="Canned response for guideline",
        fields=[],
    )

    canrep_2 = await canned_response_store.create_canned_response(
        value="Another canned response",
        fields=[],
    )

    canrep_3 = await canned_response_store.create_canned_response(
        value="Canned response not for guideline",
        fields=[],
    )

    canrep_4 = await canned_response_store.create_canned_response(
        value="Canned response for journey",
        fields=[],
    )

    await canned_response_store.upsert_tag(
        canned_response_id=canrep_1.id,
        tag_id=Tag.for_guideline_id(g1.id),
    )

    await canned_response_store.upsert_tag(
        canned_response_id=canrep_2.id,
        tag_id=Tag.for_guideline_id(g2.id),
    )

    await canned_response_store.upsert_tag(
        canned_response_id=canrep_4.id,
        tag_id=Tag.for_journey_node_id(node.id),
    )

    results = await entity_queries.find_canned_responses_for_guidelines(
        guidelines=[
            g1,
            g2,
            projection[1],
        ]
    )

    assert len(results) == 3
    assert any(canrep_1.id == r.id for r in results)
    assert any(canrep_2.id == r.id for r in results)
    assert any(canrep_4.id == r.id for r in results)

    assert all(canrep_3.id != r.id for r in results)
