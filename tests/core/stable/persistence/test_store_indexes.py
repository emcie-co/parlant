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

"""Stores declare the fields they look documents up by.

With many agents in one server, reading one agent's records must not examine
every other agent's. The transient database counts the documents each query
examines, which is the only place the difference is visible: an indexed lookup
and a full scan return the same entities.
"""

from typing import Any, cast

from parlant.adapters.db.transient import TransientDocumentCollection, TransientDocumentDatabase
from parlant.core.agents import AgentDocumentStore, AgentId
from parlant.core.common import IdGenerator
from parlant.core.groups import GroupDocumentStore, GroupId
from parlant.core.relationships import (
    RelationshipDocumentStore,
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipKind,
)
from parlant.core.rule_tool_associations import RuleToolAssociationDocumentStore
from parlant.core.rules import RuleId
from parlant.core.tools import ToolId

AGENTS = 200


def _examined(collection: Any) -> int:
    transient = cast(TransientDocumentCollection[Any], collection)
    examined = transient.examined_document_count
    transient.reset_examined_document_count()
    return examined


async def test_that_reading_one_agent_examines_only_that_agent_and_its_groups() -> None:
    async with AgentDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        for number in range(AGENTS):
            await store.create_agent(
                name=f"Agent {number}",
                id=AgentId(f"agent-{number}"),
                groups=[GroupId(f"group-{number}"), GroupId("shared")],
            )
        _examined(store._agents_collection)
        _examined(store._tag_association_collection)

        agent = await store.read_agent(AgentId("agent-117"))

        assert sorted(agent.groups) == ["group-117", "shared"]
        assert _examined(store._agents_collection) == 1
        assert _examined(store._tag_association_collection) == 2


async def test_that_reading_one_group_examines_only_that_group() -> None:
    async with GroupDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        groups = [await store.create_group(name=f"Group {number}") for number in range(AGENTS)]
        _examined(store._collection)

        group = await store.read_group(groups[117].id)

        assert group.name == "Group 117"
        assert _examined(store._collection) == 1


async def test_that_listing_groups_by_name_examines_only_the_groups_with_that_name() -> None:
    async with GroupDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        for number in range(AGENTS):
            await store.create_group(name=f"Group {number}")
        _examined(store._collection)

        groups = await store.list_groups(name="Group 117")

        assert [group.name for group in groups] == ["Group 117"]
        assert _examined(store._collection) == 1


async def test_that_a_rules_direct_relationships_examine_only_that_rules_relationships() -> None:
    async with RelationshipDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        for number in range(AGENTS):
            await store.create_relationship(
                source=RelationshipEntity(
                    id=RuleId(f"rule-{number}-a"), kind=RelationshipEntityKind.RULE
                ),
                target=RelationshipEntity(
                    id=RuleId(f"rule-{number}-b"), kind=RelationshipEntityKind.RULE
                ),
                kind=RelationshipKind.PRIORITY,
            )
        _examined(store._collection)

        by_source = await store.list_relationships(
            kind=RelationshipKind.PRIORITY,
            indirect=False,
            source_id=RuleId("rule-117-a"),
        )
        examined_by_source = _examined(store._collection)
        by_target = await store.list_relationships(
            kind=RelationshipKind.PRIORITY,
            indirect=False,
            target_id=RuleId("rule-117-b"),
        )
        examined_by_target = _examined(store._collection)

        assert [r.target.id for r in by_source] == ["rule-117-b"]
        assert [r.source.id for r in by_target] == ["rule-117-a"]
        assert examined_by_source == 1
        assert examined_by_target == 1


async def test_that_reading_one_tool_association_examines_only_that_association() -> None:
    async with RuleToolAssociationDocumentStore(
        IdGenerator(), TransientDocumentDatabase()
    ) as store:
        associations = [
            await store.create_association(
                rule_id=RuleId(f"rule-{number}"),
                tool_id=ToolId(service_name="service", tool_name=f"tool_{number}"),
            )
            for number in range(AGENTS)
        ]
        _examined(store._collection)

        association = await store.read_association(associations[117].id)

        assert association.rule_id == "rule-117"
        assert _examined(store._collection) == 1


async def test_that_the_tool_associations_of_some_rules_examine_only_those_rules() -> None:
    async with RuleToolAssociationDocumentStore(
        IdGenerator(), TransientDocumentDatabase()
    ) as store:
        for number in range(AGENTS):
            for tool in ("lookup", "save"):
                await store.create_association(
                    rule_id=RuleId(f"rule-{number}"),
                    tool_id=ToolId(service_name="service", tool_name=tool),
                )
        _examined(store._collection)

        associations = await store.list_associations(
            rule_ids=[RuleId("rule-117"), RuleId("rule-3"), RuleId("rule-without-tools")]
        )

        assert sorted((a.rule_id, a.tool_id.tool_name) for a in associations) == [
            ("rule-117", "lookup"),
            ("rule-117", "save"),
            ("rule-3", "lookup"),
            ("rule-3", "save"),
        ]
        assert _examined(store._collection) == 4


async def test_that_asking_for_the_tool_associations_of_no_rules_returns_none_of_them() -> None:
    async with RuleToolAssociationDocumentStore(
        IdGenerator(), TransientDocumentDatabase()
    ) as store:
        await store.create_association(
            rule_id=RuleId("rule-1"),
            tool_id=ToolId(service_name="service", tool_name="lookup"),
        )
        _examined(store._collection)

        assert await store.list_associations(rule_ids=[]) == []
        assert _examined(store._collection) == 0


async def test_that_listing_tool_associations_without_rules_still_returns_all_of_them() -> None:
    async with RuleToolAssociationDocumentStore(
        IdGenerator(), TransientDocumentDatabase()
    ) as store:
        for number in range(3):
            await store.create_association(
                rule_id=RuleId(f"rule-{number}"),
                tool_id=ToolId(service_name="service", tool_name="lookup"),
            )

        assert len(await store.list_associations()) == 3


def _rule_entity(rule_id: str) -> RelationshipEntity:
    return RelationshipEntity(id=RuleId(rule_id), kind=RelationshipEntityKind.RULE)


async def test_that_the_relationships_of_some_entities_examine_only_those_entities() -> None:
    async with RelationshipDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        for number in range(AGENTS):
            await store.create_relationship(
                source=_rule_entity(f"rule-{number}-a"),
                target=_rule_entity(f"rule-{number}-b"),
                kind=RelationshipKind.PRIORITY,
            )
        await store.create_relationship(
            source=_rule_entity("rule-117-a"),
            target=RelationshipEntity(
                id=GroupId("group-117"), kind=RelationshipEntityKind.GROUP_ALL
            ),
            kind=RelationshipKind.DEPENDENCY,
        )
        await store.create_relationship(
            source=RelationshipEntity(
                id=ToolId(service_name="service", tool_name="lookup"),
                kind=RelationshipEntityKind.TOOL,
            ),
            target=RelationshipEntity(
                id=ToolId(service_name="service", tool_name="search"),
                kind=RelationshipEntityKind.TOOL,
            ),
            kind=RelationshipKind.OVERLAP,
        )
        _examined(store._collection)

        of_any_kind = await store.list_relationships(
            entity_ids=[RuleId("rule-117-a"), RuleId("rule-3-b"), RuleId("rule-without-any")],
        )
        examined_of_any_kind = _examined(store._collection)
        of_one_kind = await store.list_relationships(
            kind=RelationshipKind.PRIORITY,
            indirect=False,
            entity_ids=[RuleId("rule-117-a"), RuleId("rule-3-b"), GroupId("group-117")],
        )
        examined_of_one_kind = _examined(store._collection)
        of_a_tool = await store.list_relationships(
            kind=RelationshipKind.OVERLAP,
            entity_ids=[ToolId(service_name="service", tool_name="search")],
        )
        examined_of_a_tool = _examined(store._collection)

        assert sorted((r.source.id, r.target.id, r.kind) for r in of_any_kind) == [
            ("rule-117-a", "group-117", RelationshipKind.DEPENDENCY),
            ("rule-117-a", "rule-117-b", RelationshipKind.PRIORITY),
            ("rule-3-a", "rule-3-b", RelationshipKind.PRIORITY),
        ]
        assert examined_of_any_kind == 3
        assert sorted((r.source.id, r.target.id) for r in of_one_kind) == [
            ("rule-117-a", "rule-117-b"),
            ("rule-3-a", "rule-3-b"),
        ]
        assert examined_of_one_kind == 3
        assert [r.source.id_to_string() for r in of_a_tool] == ["service:lookup"]
        assert examined_of_a_tool == 1


async def test_that_asking_for_the_relationships_of_no_entities_returns_none_of_them() -> None:
    async with RelationshipDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        await store.create_relationship(
            source=_rule_entity("rule-a"),
            target=_rule_entity("rule-b"),
            kind=RelationshipKind.PRIORITY,
        )
        _examined(store._collection)

        assert await store.list_relationships(entity_ids=[]) == []
        assert await store.list_relationships(kind=RelationshipKind.PRIORITY, entity_ids=[]) == []
        assert _examined(store._collection) == 0


async def test_that_listing_relationships_without_entities_still_returns_all_of_them() -> None:
    async with RelationshipDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        for number in range(3):
            await store.create_relationship(
                source=_rule_entity(f"rule-{number}-a"),
                target=_rule_entity(f"rule-{number}-b"),
                kind=RelationshipKind.PRIORITY,
            )

        assert len(await store.list_relationships()) == 3
        assert len(await store.list_relationships(kind=RelationshipKind.PRIORITY)) == 3


async def test_that_listing_some_groups_by_id_examines_only_those_groups() -> None:
    async with GroupDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        groups = [await store.create_group(name=f"Group {number}") for number in range(AGENTS)]
        _examined(store._collection)

        found = await store.list_groups(ids=[groups[117].id, groups[3].id, GroupId("no-group")])
        examined = _examined(store._collection)

        assert sorted(group.name for group in found) == ["Group 117", "Group 3"]
        assert examined == 2


async def test_that_asking_for_no_groups_by_id_returns_none_of_them() -> None:
    async with GroupDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        await store.create_group(name="Group")
        _examined(store._collection)

        assert await store.list_groups(ids=[]) == []
        assert _examined(store._collection) == 0


async def test_that_listing_groups_without_ids_still_returns_all_of_them() -> None:
    async with GroupDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        for number in range(3):
            await store.create_group(name=f"Group {number}")

        assert len(await store.list_groups()) == 3
