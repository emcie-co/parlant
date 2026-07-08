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
from parlant.core.common import generate_id
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCallEvaluation,
    ToolCallId,
    ToolInsights,
)
from parlant.core.entity_cq import EntityQueries
from parlant.core.glossary import GlossaryStore
from parlant.core.journey_rule_projection import JourneyRuleProjection
from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipStore,
    RelationshipKind,
    RelationshipEntityKind,
)
from parlant.core.canned_responses import CannedResponseStore
from parlant.core.rules import RuleStore
from parlant.core.journeys import JourneyNodeKind, JourneyStore
from parlant.core.groups import GroupIds, GroupId, GroupStore
from parlant.core.tools import ToolId


async def test_that_list_rules_with_mutual_agent_group_are_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    agent_store = container[AgentStore]
    rule_store = container[RuleStore]

    await agent_store.upsert_group(
        agent_id=agent.id,
        group_id=GroupId("tag_1"),
    )

    first_rule = await rule_store.create_rule(
        condition="condition 1",
        action="action 1",
    )

    second_rule = await rule_store.create_rule(
        condition="condition 2",
        action="action 2",
    )

    await rule_store.upsert_group(
        rule_id=first_rule.id,
        group_id=GroupId("tag_1"),
    )

    await rule_store.upsert_group(
        rule_id=second_rule.id,
        group_id=GroupId("tag_2"),
    )

    result = await entity_queries.find_rules_for_context(agent.id, [])

    assert len(result) == 1
    assert result[0].id == first_rule.id


async def test_that_list_rules_global_rule_is_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]

    global_rule = await rule_store.create_rule(
        condition="condition 1",
        action="action 1",
    )

    result = await entity_queries.find_rules_for_context(agent.id, [])

    assert len(result) == 1
    assert result[0].id == global_rule.id


async def test_that_rule_with_not_hierarchy_tag_is_not_returned(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]

    first_rule = await rule_store.create_rule(
        condition="condition 1",
        action="action 1",
    )

    second_rule = await rule_store.create_rule(
        condition="condition 2",
        action="action 2",
    )

    await rule_store.upsert_group(
        rule_id=first_rule.id,
        group_id=GroupIds.for_agent_id(agent.id),
    )

    await rule_store.upsert_group(
        rule_id=second_rule.id,
        group_id=GroupId("tag_2"),
    )

    result = await entity_queries.find_rules_for_context(agent.id, [])

    assert len(result) == 1
    assert result[0].id == first_rule.id


async def test_that_rule_matches_are_not_filtered_by_enabled_journeys(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]

    journey_rule = await rule_store.create_rule(
        condition="condition 1",
    )

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        triggers=[journey_rule.id],
    )

    rule = await rule_store.create_rule(
        condition="condition 2",
    )

    await rule_store.upsert_group(
        rule_id=journey_rule.id,
        group_id=GroupIds.for_journey_id(journey.id),
    )

    await rule_store.upsert_group(
        rule_id=rule.id,
        group_id=GroupIds.for_journey_id(journey.id),
    )

    result = await entity_queries.find_rules_for_context(
        agent.id,
        [journey],
    )

    assert len(result) == 3
    assert any(journey_rule.id == g.id for g in result)
    assert any(rule.id == g.id for g in result)


async def test_that_rule_groupged_with_disabled_journey_is_filtered_out_when_matched(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]

    journey_rule = await rule_store.create_rule(
        condition="condition 1",
    )

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        triggers=[journey_rule.id],
    )

    rule = await rule_store.create_rule(
        condition="condition 2",
    )

    await rule_store.upsert_group(
        rule_id=journey_rule.id,
        group_id=GroupIds.for_journey_id(journey.id),
    )

    await rule_store.upsert_group(
        rule_id=rule.id,
        group_id=GroupIds.for_journey_id(journey.id),
    )

    result = await entity_queries.find_rules_for_context(
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

    lookup = await entity_queries.find_canned_responses_for_context(
        agent=agent,
        journeys=[],
        rules=[],
    )
    assert len(lookup.canned_responses) == 1
    assert lookup.canned_responses[0].id == untagged_canrep.id


async def test_that_find_canned_responses_for_agent_returns_none_for_non_matching_tag(
    container: Container, agent: Agent
) -> None:
    canrep_store: CannedResponseStore = container[CannedResponseStore]
    entity_queries = container[EntityQueries]

    group1 = GroupId("group1")
    await canrep_store.create_canned_response(
        value="Grouped canned response",
        fields=[],
        groups=[group1],
    )

    await container[AgentStore].upsert_group(
        agent_id=agent.id, group_id=GroupId("non_matching_tag")
    )

    lookup = await entity_queries.find_canned_responses_for_context(
        agent=agent,
        journeys=[],
        rules=[],
    )
    assert len(lookup.canned_responses) == 0


async def test_that_find_canned_responses_for_agent_and_journey_returns_journey_canned_responses(
    container: Container, agent: Agent
) -> None:
    canrep_store: CannedResponseStore = container[CannedResponseStore]
    journey_store = container[JourneyStore]
    entity_queries = container[EntityQueries]

    journey = await journey_store.create_journey(
        title="Test Journey",
        description="A test journey",
        triggers=[],
    )

    journey_group = GroupIds.for_journey_id(journey.id)
    journey_canrep = await canrep_store.create_canned_response(
        value="Journey canrep",
        fields=[],
        groups=[journey_group],
    )

    lookup = await entity_queries.find_canned_responses_for_context(
        agent=agent,
        journeys=[journey],
        rules=[],
    )
    assert len(lookup.canned_responses) == 1
    assert lookup.canned_responses[0].id == journey_canrep.id


async def test_that_find_glossary_terms_for_agent_returns_all_when_no_tags(
    container: Container,
    agent: Agent,
) -> None:
    glossary_store = container[GlossaryStore]
    entity_queries = container[EntityQueries]

    untagged_term = await glossary_store.create_term(
        name="Hello world",
        description="A greeting",
        groups=[],
    )

    group = GroupId("group1")
    await glossary_store.create_term(
        name="Grouped term",
        description="A grouped glossary entry",
        groups=[group],
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

    group1 = GroupId("group1")
    await glossary_store.create_term(
        name="Grouped term",
        description="A grouped glossary entry",
        groups=[group1],
    )

    await container[AgentStore].upsert_group(
        agent_id=agent.id, group_id=GroupId("non_matching_tag")
    )

    results = await entity_queries.find_glossary_terms_for_context(
        agent_id=agent.id, query="Grouped"
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
    rule_store = container[RuleStore]

    condition = await rule_store.create_rule(
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
        triggers=[condition.id],
    )

    support_journey = await journey_store.create_journey(
        title="Change Credit Limits",
        description="Remember that credit limits can be decreased through this chat, using the decrease_limits tool, but that to increase credit limits you must visit a physical branch",
        triggers=[],
    )

    results = await entity_queries.sort_journeys_by_contextual_relevance(
        [onboarding_journey, support_journey], "I'd like to reset my password"
    )

    assert len(results) == 2
    assert results[0].id == onboarding_journey.id
    assert results[1].id == support_journey.id


async def test_list_rules_dependent_directly_on_journey(
    container: Container,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]
    relationship_store = container[RelationshipStore]

    journey = await journey_store.create_journey(
        title="Test Journey",
        description="A journey for testing dependencies",
        triggers=[],
    )

    rule1 = await rule_store.create_rule(
        condition="condition 1",
        action="action 1",
    )
    _ = await rule_store.create_rule(
        condition="condition 2",
        action="action 2",
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=rule1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(
            id=GroupIds.for_journey_id(journey.id), kind=RelationshipEntityKind.GROUP_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await entity_queries.find_journey_related_rules(journey)

    assert len(result) == 2
    assert any([rule1.id in g for g in result])
    assert any([journey.root_id in g for g in result])


async def test_list_rules_dependent_indirectly_on_journey(
    container: Container,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]
    relationship_store = container[RelationshipStore]
    group_store = container[GroupStore]

    journey = await journey_store.create_journey(
        title="Test Journey",
        description="A journey for testing dependencies",
        triggers=[],
    )

    rule1 = await rule_store.create_rule(
        condition="condition 1",
        action="action 1",
    )
    rule2 = await rule_store.create_rule(
        condition="condition 2",
        action="action 2",
    )
    rule3 = await rule_store.create_rule(
        condition="condition 3",
        action="action 3",
    )
    group = await group_store.create_group(name="test group")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=rule1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(
            id=GroupIds.for_journey_id(journey.id), kind=RelationshipEntityKind.GROUP_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=rule2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=rule1.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=rule3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(
            id=GroupIds.for_journey_id(journey.id), kind=RelationshipEntityKind.GROUP_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await entity_queries.find_journey_related_rules(journey)

    assert len(result) == 4

    assert any(rule1.id == g for g in result)
    assert any(rule2.id == g for g in result)
    assert any(rule3.id == g for g in result)


async def test_that_canned_responses_can_be_found_for_a_rule(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    canned_response_store = container[CannedResponseStore]
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]

    g1 = await rule_store.create_rule(
        condition="condition 1",
        action="action 1",
    )

    g2 = await rule_store.create_rule(
        condition="condition 2",
        action="action 2",
    )

    journey = await journey_store.create_journey(
        title="Test Journey",
        description="A journey for testing canned responses",
        triggers=[],
    )

    node = await journey_store.create_node(
        journey_id=journey.id,
        kind=JourneyNodeKind.CHAT,
        action="Test Node",
        tools=[],
    )

    await journey_store.create_edge(
        journey_id=journey.id,
        source=journey.root_id,
        target=node.id,
        condition=None,
    )

    projection = await container[JourneyRuleProjection].project_journey_to_rules(
        journey_id=journey.id,
    )

    assert len(projection) == 2

    canrep_1 = await canned_response_store.create_canned_response(
        value="Canned response for rule",
        fields=[],
    )

    canrep_2 = await canned_response_store.create_canned_response(
        value="Another canned response",
        fields=[],
    )

    canrep_3 = await canned_response_store.create_canned_response(
        value="Canned response not for rule",
        fields=[],
    )

    canrep_4 = await canned_response_store.create_canned_response(
        value="Canned response for journey",
        fields=[],
    )

    await canned_response_store.upsert_group(
        canned_response_id=canrep_1.id,
        group_id=GroupIds.for_rule_id(g1.id),
    )

    await canned_response_store.upsert_group(
        canned_response_id=canrep_2.id,
        group_id=GroupIds.for_rule_id(g2.id),
    )

    await canned_response_store.upsert_group(
        canned_response_id=canrep_4.id,
        group_id=GroupIds.for_journey_node_id(node.id),
    )

    results = await entity_queries.find_canned_responses_for_rules(
        rules=[
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


async def test_that_find_rules_that_need_reevaluation_finds_rules_by_tag(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]
    agent_store = container[AgentStore]

    custom_group_id = GroupId("custom-group")
    tool_id = ToolId(service_name="built-in", tool_name="verify_account")

    await agent_store.upsert_group(
        agent_id=agent.id,
        group_id=GroupId("agent-group"),
    )

    rule = await rule_store.create_rule(
        condition="the customer's account has been verified",
        action="Offer a Pepsi",
    )

    await rule_store.upsert_group(
        rule_id=rule.id,
        group_id=GroupId("agent-group"),
    )

    await rule_store.upsert_group(
        rule_id=rule.id,
        group_id=custom_group_id,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=custom_group_id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        target=RelationshipEntity(
            id=tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        kind=RelationshipKind.REEVALUATION,
    )

    tool_insights = ToolInsights(
        evaluations={tool_id: {ToolCallId(generate_id()): ToolCallEvaluation.NEEDS_TO_RUN}},
    )

    # Re-read the rule after groups were upserted
    rule = await rule_store.read_rule(rule.id)

    available_rules = {rule.id: rule}

    result = await entity_queries.find_rules_that_need_reevaluation(
        available_rules=available_rules,
        active_journeys=[],
        tool_insights=tool_insights,
    )

    assert len(result) == 1
    assert result[0].id == rule.id
