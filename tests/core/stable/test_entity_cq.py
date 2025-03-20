from lagom import Container

from parlant.core.agents import Agent, AgentStore
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import GuidelineStore
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

    result = await entity_queries.find_guidelines_for_agent(agent.id)

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

    result = await entity_queries.find_guidelines_for_agent(agent.id)

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

    result = await entity_queries.find_guidelines_for_agent(agent.id)

    assert len(result) == 1
    assert result[0].id == first_guideline.id
