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

from datetime import datetime, timezone
from typing import Optional, Sequence

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.common import Version
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    group_ids_by_entity,
)
from tests.core.stable.query_instrumentation import QueryLog, instrument_queries

COLLECTION_NAME = "rule_group_associations"


class _AssociationDocument(BaseDocument, total=False):
    rule_id: str
    group_id: str


async def _loader(doc: BaseDocument) -> Optional[_AssociationDocument]:
    return doc  # type: ignore[return-value]


async def _collection(
    *associations: tuple[str, str],
) -> DocumentCollection[_AssociationDocument]:
    database = TransientDocumentDatabase()

    collection = await database.get_or_create_collection(
        COLLECTION_NAME,
        _AssociationDocument,
        _loader,
    )

    for index, (rule_id, group_id) in enumerate(associations):
        await collection.insert_one(
            _AssociationDocument(
                id=ObjectId(f"assoc-{index}"),
                version=Version.String("0.1.0"),
                creation_utc=datetime.now(timezone.utc).isoformat(),
                rule_id=rule_id,
                group_id=group_id,
            )
        )

    return collection


async def _group_ids(
    collection: DocumentCollection[_AssociationDocument],
    entity_ids: Sequence[str],
) -> dict[str, list[str]]:
    result = await group_ids_by_entity(
        collection,
        entity_field="rule_id",
        value_field="group_id",
        entity_ids=entity_ids,
    )

    return {entity_id: sorted(groups) for entity_id, groups in result.items()}


async def test_that_no_entity_ids_issues_no_query_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A store listing an empty result must not pay for a round trip it can't use.
    collection = await _collection(("rule-1", "group-a"))

    log: QueryLog = instrument_queries(monkeypatch)

    assert await _group_ids(collection, []) == {}
    assert log.collections == []


async def test_that_an_entity_without_associations_is_returned_with_no_group_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every requested id has to appear, or the caller can't tell "no groups"
    # apart from "I forgot to ask", and an ungrouped entity silently vanishes.
    collection = await _collection(("rule-1", "group-a"))

    assert await _group_ids(collection, ["rule-1", "rule-2"]) == {
        "rule-1": ["group-a"],
        "rule-2": [],
    }


async def test_that_every_group_of_one_entity_is_collected() -> None:
    collection = await _collection(
        ("rule-1", "group-a"),
        ("rule-1", "group-b"),
        ("rule-1", "group-c"),
    )

    assert await _group_ids(collection, ["rule-1"]) == {
        "rule-1": ["group-a", "group-b", "group-c"],
    }


async def test_that_a_group_shared_by_several_entities_reaches_each_of_them() -> None:
    collection = await _collection(
        ("rule-1", "shared"),
        ("rule-2", "shared"),
    )

    assert await _group_ids(collection, ["rule-1", "rule-2"]) == {
        "rule-1": ["shared"],
        "rule-2": ["shared"],
    }


async def test_that_associations_of_other_entities_are_not_mixed_in() -> None:
    collection = await _collection(
        ("rule-1", "group-a"),
        ("rule-2", "group-b"),
        ("rule-3", "group-c"),
    )

    assert await _group_ids(collection, ["rule-1", "rule-3"]) == {
        "rule-1": ["group-a"],
        "rule-3": ["group-c"],
    }


async def test_that_one_query_is_issued_regardless_of_how_many_entities_are_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole point: cost is independent of the number of entities.
    collection = await _collection(*[(f"rule-{i}", f"group-{i}") for i in range(25)])

    log: QueryLog = instrument_queries(monkeypatch)

    await _group_ids(collection, [f"rule-{i}" for i in range(25)])

    assert log.count_for(COLLECTION_NAME) == 1
