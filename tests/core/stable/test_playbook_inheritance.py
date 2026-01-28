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

"""Tests for playbook inheritance resolution in EntityQueries."""

from lagom import Container

from parlant.core.agents import Agent, AgentStore
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import GuidelineStore
from parlant.core.playbooks import DisabledRuleRef, PlaybookStore
from parlant.core.tags import Tag


async def test_that_guidelines_from_playbook_are_returned(
    container: Container,
    agent: Agent,
) -> None:
    """Agent with playbook should get guidelines tagged with that playbook."""
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    playbook_store = container[PlaybookStore]
    agent_store = container[AgentStore]

    # Create a playbook
    playbook = await playbook_store.create_playbook(name="Test Playbook")

    # Create a guideline tagged with the playbook
    guideline = await guideline_store.create_guideline(
        condition="playbook condition",
        action="playbook action",
    )
    await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_playbook_id(playbook.id),
    )

    # Assign playbook to agent
    await agent_store.update_agent(
        agent_id=agent.id,
        params={"playbook_id": playbook.id},
    )

    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    assert len(result) == 1
    assert result[0].id == guideline.id


async def test_that_guidelines_from_parent_playbook_are_inherited(
    container: Container,
    agent: Agent,
) -> None:
    """Agent with playbook should inherit guidelines from parent playbook."""
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    playbook_store = container[PlaybookStore]
    agent_store = container[AgentStore]

    # Create parent playbook with a guideline
    parent_playbook = await playbook_store.create_playbook(name="Parent Playbook")
    parent_guideline = await guideline_store.create_guideline(
        condition="parent condition",
        action="parent action",
    )
    await guideline_store.upsert_tag(
        guideline_id=parent_guideline.id,
        tag_id=Tag.for_playbook_id(parent_playbook.id),
    )

    # Create child playbook with its own guideline
    child_playbook = await playbook_store.create_playbook(
        name="Child Playbook",
        parent_id=parent_playbook.id,
    )
    child_guideline = await guideline_store.create_guideline(
        condition="child condition",
        action="child action",
    )
    await guideline_store.upsert_tag(
        guideline_id=child_guideline.id,
        tag_id=Tag.for_playbook_id(child_playbook.id),
    )

    # Assign child playbook to agent
    await agent_store.update_agent(
        agent_id=agent.id,
        params={"playbook_id": child_playbook.id},
    )

    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    assert len(result) == 2
    assert any(g.id == parent_guideline.id for g in result)
    assert any(g.id == child_guideline.id for g in result)


async def test_that_disabled_rules_filter_out_inherited_guidelines(
    container: Container,
    agent: Agent,
) -> None:
    """Disabled rules in playbook should exclude inherited guidelines."""
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    playbook_store = container[PlaybookStore]
    agent_store = container[AgentStore]

    # Create parent playbook with two guidelines
    parent_playbook = await playbook_store.create_playbook(name="Parent Playbook")
    guideline_to_keep = await guideline_store.create_guideline(
        condition="keep this",
        action="keep action",
    )
    guideline_to_disable = await guideline_store.create_guideline(
        condition="disable this",
        action="disable action",
    )
    await guideline_store.upsert_tag(
        guideline_id=guideline_to_keep.id,
        tag_id=Tag.for_playbook_id(parent_playbook.id),
    )
    await guideline_store.upsert_tag(
        guideline_id=guideline_to_disable.id,
        tag_id=Tag.for_playbook_id(parent_playbook.id),
    )

    # Create child playbook that disables one guideline
    child_playbook = await playbook_store.create_playbook(
        name="Child Playbook",
        parent_id=parent_playbook.id,
    )
    await playbook_store.add_disabled_rule(
        playbook_id=child_playbook.id,
        rule_ref=DisabledRuleRef(f"guideline:{guideline_to_disable.id}"),
    )

    # Assign child playbook to agent
    await agent_store.update_agent(
        agent_id=agent.id,
        params={"playbook_id": child_playbook.id},
    )

    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    assert len(result) == 1
    assert result[0].id == guideline_to_keep.id


async def test_that_agent_without_playbook_works_unchanged(
    container: Container,
    agent: Agent,
) -> None:
    """Agent without playbook should work as before (backward compatible)."""
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    playbook_store = container[PlaybookStore]

    # Create a playbook with a guideline (not assigned to agent)
    playbook = await playbook_store.create_playbook(name="Unassigned Playbook")
    playbook_guideline = await guideline_store.create_guideline(
        condition="playbook condition",
        action="playbook action",
    )
    await guideline_store.upsert_tag(
        guideline_id=playbook_guideline.id,
        tag_id=Tag.for_playbook_id(playbook.id),
    )

    # Create a global guideline
    global_guideline = await guideline_store.create_guideline(
        condition="global condition",
        action="global action",
    )

    # Agent has no playbook assigned
    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    # Should only get global guideline, not playbook guideline
    assert len(result) == 1
    assert result[0].id == global_guideline.id


async def test_that_agent_with_playbook_also_gets_agent_tagged_guidelines(
    container: Container,
    agent: Agent,
) -> None:
    """Agent with playbook should still get guidelines tagged directly to agent."""
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    playbook_store = container[PlaybookStore]
    agent_store = container[AgentStore]

    # Create playbook guideline
    playbook = await playbook_store.create_playbook(name="Test Playbook")
    playbook_guideline = await guideline_store.create_guideline(
        condition="playbook condition",
        action="playbook action",
    )
    await guideline_store.upsert_tag(
        guideline_id=playbook_guideline.id,
        tag_id=Tag.for_playbook_id(playbook.id),
    )

    # Create agent-tagged guideline
    agent_guideline = await guideline_store.create_guideline(
        condition="agent condition",
        action="agent action",
    )
    await guideline_store.upsert_tag(
        guideline_id=agent_guideline.id,
        tag_id=Tag.for_agent_id(agent.id),
    )

    # Assign playbook to agent
    await agent_store.update_agent(
        agent_id=agent.id,
        params={"playbook_id": playbook.id},
    )

    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    # Should get both
    assert len(result) == 2
    assert any(g.id == playbook_guideline.id for g in result)
    assert any(g.id == agent_guideline.id for g in result)


async def test_that_three_level_inheritance_works(
    container: Container,
    agent: Agent,
) -> None:
    """Guidelines from grandparent -> parent -> child should all be inherited."""
    entity_queries = container[EntityQueries]
    guideline_store = container[GuidelineStore]
    playbook_store = container[PlaybookStore]
    agent_store = container[AgentStore]

    # Create grandparent playbook
    grandparent = await playbook_store.create_playbook(name="Grandparent")
    grandparent_guideline = await guideline_store.create_guideline(
        condition="grandparent",
        action="grandparent action",
    )
    await guideline_store.upsert_tag(
        guideline_id=grandparent_guideline.id,
        tag_id=Tag.for_playbook_id(grandparent.id),
    )

    # Create parent playbook
    parent = await playbook_store.create_playbook(
        name="Parent",
        parent_id=grandparent.id,
    )
    parent_guideline = await guideline_store.create_guideline(
        condition="parent",
        action="parent action",
    )
    await guideline_store.upsert_tag(
        guideline_id=parent_guideline.id,
        tag_id=Tag.for_playbook_id(parent.id),
    )

    # Create child playbook
    child = await playbook_store.create_playbook(
        name="Child",
        parent_id=parent.id,
    )
    child_guideline = await guideline_store.create_guideline(
        condition="child",
        action="child action",
    )
    await guideline_store.upsert_tag(
        guideline_id=child_guideline.id,
        tag_id=Tag.for_playbook_id(child.id),
    )

    # Assign child playbook to agent
    await agent_store.update_agent(
        agent_id=agent.id,
        params={"playbook_id": child.id},
    )

    result = await entity_queries.find_guidelines_for_context(agent.id, [])

    assert len(result) == 3
    assert any(g.id == grandparent_guideline.id for g in result)
    assert any(g.id == parent_guideline.id for g in result)
    assert any(g.id == child_guideline.id for g in result)
