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

from lagom import Container

from parlant.core.agents import Agent, AgentStore
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore
from parlant.core.tags import Tag, TagId


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

    result = await entity_queries.find_guidelines_for_agent(agent.id, [])

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

    result = await entity_queries.find_guidelines_for_agent(agent.id, [])

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

    result = await entity_queries.find_guidelines_for_agent(agent.id, [])

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

    result = await entity_queries.find_guidelines_for_agent(
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

    result = await entity_queries.find_guidelines_for_agent(
        agent.id,
        [],
    )

    assert len(result) == 0
