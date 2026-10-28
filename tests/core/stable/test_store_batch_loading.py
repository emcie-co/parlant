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

"""Every store that owns a group-association collection reconstructs its
entities from it. Listing N entities must cost one association query, not N.

The entities a batched read returns are identical to the ones a per-entity read
returns, which is exactly why this needs its own tests: the only observable
difference is the number of queries. See ``query_instrumentation``.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence

import pytest

from parlant.core.agents import AgentStore
from parlant.core.canned_responses import CannedResponseStore
from parlant.core.container import Container
from parlant.core.context_variables import ContextVariableStore
from parlant.core.customers import CustomerStore
from parlant.core.glossary import GlossaryStore
from parlant.core.groups import GroupId
from parlant.core.rules import RuleStore
from tests.core.stable.query_instrumentation import QueryLog, instrument_queries


@dataclass(frozen=True)
class _Store:
    """One store's group-association surface, so every store is tested alike."""

    name: str
    association_collection: str
    create: Callable[[Container, str, Sequence[GroupId]], Awaitable[Any]]
    # Not Sequence: CustomerStore returns a listing object. Both iterate.
    list_all: Callable[[Container], Awaitable[Any]]
    read: Callable[[Container, Any], Awaitable[Any]]
    list_by_group: Optional[Callable[[Container, GroupId], Awaitable[Any]]]


async def _create_rule(c: Container, name: str, groups: Sequence[GroupId]) -> Any:
    return await c[RuleStore].create_rule(content=name, groups=groups)


async def _create_term(c: Container, name: str, groups: Sequence[GroupId]) -> Any:
    return await c[GlossaryStore].create_term(name=name, description=name, groups=groups)


async def _create_canned_response(c: Container, name: str, groups: Sequence[GroupId]) -> Any:
    return await c[CannedResponseStore].create_canned_response(value=name, groups=groups)


async def _create_variable(c: Container, name: str, groups: Sequence[GroupId]) -> Any:
    return await c[ContextVariableStore].create_variable(name=name, groups=groups)


async def _create_agent(c: Container, name: str, groups: Sequence[GroupId]) -> Any:
    return await c[AgentStore].create_agent(name=name, groups=groups)


async def _create_customer(c: Container, name: str, groups: Sequence[GroupId]) -> Any:
    return await c[CustomerStore].create_customer(name=name, groups=groups)


STORES = [
    _Store(
        name="rules",
        association_collection="rule_group_associations",
        create=_create_rule,
        list_all=lambda c: c[RuleStore].list_rules(),
        read=lambda c, id: c[RuleStore].read_rule(id),
        list_by_group=lambda c, group: c[RuleStore].list_rules(groups=[group]),
    ),
    _Store(
        name="glossary",
        association_collection="glossary_tags",
        create=_create_term,
        list_all=lambda c: c[GlossaryStore].list_terms(),
        read=lambda c, id: c[GlossaryStore].read_term(id),
        list_by_group=lambda c, group: c[GlossaryStore].list_terms(groups=[group]),
    ),
    _Store(
        name="canned_responses",
        association_collection="canned_response_tag_associations",
        create=_create_canned_response,
        list_all=lambda c: c[CannedResponseStore].list_canned_responses(),
        read=lambda c, id: c[CannedResponseStore].read_canned_response(id),
        list_by_group=lambda c, group: c[CannedResponseStore].list_canned_responses(groups=[group]),
    ),
    _Store(
        name="context_variables",
        association_collection="variable_tag_associations",
        create=_create_variable,
        list_all=lambda c: c[ContextVariableStore].list_variables(),
        read=lambda c, id: c[ContextVariableStore].read_variable(id),
        list_by_group=lambda c, group: c[ContextVariableStore].list_variables(groups=[group]),
    ),
    _Store(
        name="agents",
        association_collection="agent_groups",
        create=_create_agent,
        list_all=lambda c: c[AgentStore].list_agents(),
        read=lambda c, id: c[AgentStore].read_agent(id),
        list_by_group=None,  # list_agents takes no group filter
    ),
    _Store(
        name="customers",
        association_collection="customer_tag_associations",
        create=_create_customer,
        list_all=lambda c: c[CustomerStore].list_customers(),
        read=lambda c, id: c[CustomerStore].read_customer(id),
        list_by_group=lambda c, group: c[CustomerStore].list_customers(groups=[group]),
    ),
]

across_stores = pytest.mark.parametrize("store", STORES, ids=[s.name for s in STORES])


def _entities(listing: Any) -> list[Any]:
    # CustomerStore returns a listing object; the rest return plain sequences.
    # Both iterate over their entities.
    return list(listing)


@across_stores
async def test_that_listing_entities_issues_one_association_query_regardless_of_count(
    container: Container,
    store: _Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The regression under test: one query per entity turns a page of rules on
    # the matcher's hot path into a storm of round trips.
    created = [
        await store.create(container, f"entity-{index}", [GroupId(f"group-{index}")])
        for index in range(8)
    ]

    log: QueryLog = instrument_queries(monkeypatch)

    listed = {entity.id for entity in _entities(await store.list_all(container))}

    assert {entity.id for entity in created} <= listed
    assert log.count_for(store.association_collection) == 1


@across_stores
async def test_that_listing_an_empty_store_issues_no_association_query(
    container: Container,
    store: _Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _entities(await store.list_all(container)):
        pytest.skip(f"{store.name} is seeded with built-in entities, so it is never empty")

    log: QueryLog = instrument_queries(monkeypatch)

    assert _entities(await store.list_all(container)) == []
    assert log.count_for(store.association_collection) == 0


@across_stores
async def test_that_listing_returns_the_same_entities_as_reading_them_one_by_one(
    container: Container,
    store: _Store,
) -> None:
    created = [
        await store.create(container, "shared-group", [GroupId("a"), GroupId("b")]),
        await store.create(container, "one-group", [GroupId("a")]),
        await store.create(container, "no-groups", []),
    ]

    listed = {entity.id: entity for entity in _entities(await store.list_all(container))}

    assert {entity.id for entity in created} <= set(listed)

    for entity in created:
        read = await store.read(container, entity.id)

        assert listed[entity.id] == read
        assert set(listed[entity.id].groups) == set(read.groups)


@across_stores
async def test_that_an_entity_without_groups_is_listed_with_no_groups(
    container: Container,
    store: _Store,
) -> None:
    # An ungrouped entity must survive the batch: it has no association rows at
    # all, so a batched read that only walks the rows it found would drop it.
    await store.create(container, "grouped", [GroupId("a")])
    ungrouped = await store.create(container, "ungrouped", [])

    listed = {entity.id: entity for entity in _entities(await store.list_all(container))}

    assert ungrouped.id in listed
    assert list(listed[ungrouped.id].groups) == []


@across_stores
async def test_that_a_group_filtered_listing_still_reports_each_entity_full_groups(
    container: Container,
    store: _Store,
) -> None:
    # Filtering by one group must not truncate the groups on what comes back —
    # the batch query has to ask by entity id, not reuse the filter's rows.
    if store.list_by_group is None:
        pytest.skip(f"{store.name} has no group-filtered listing")

    entity = await store.create(container, "multi", [GroupId("a"), GroupId("b")])

    listed = _entities(await store.list_by_group(container, GroupId("a")))

    assert [e.id for e in listed] == [entity.id]
    assert set(listed[0].groups) == {GroupId("a"), GroupId("b")}
