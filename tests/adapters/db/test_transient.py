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
from typing import Any, cast

import pytest

from parlant.adapters.db.transient import TransientDocumentCollection, TransientDocumentDatabase
from parlant.core.common import Version
from parlant.core.persistence.common import ObjectId, SortDirection, Where, matches_filters
from parlant.core.persistence.document_database import (
    BaseDocument,
    CollectionIndex,
    identity_loader_for,
)


class _Document(BaseDocument, total=False):
    owner_id: str
    kind: str
    rank: int


def _document(number: int, owners: int = 10) -> _Document:
    return _Document(
        id=ObjectId(f"doc-{number}"),
        version=Version.String("0.1.0"),
        creation_utc=f"2026-01-{(number % 28) + 1:02d}",
        owner_id=f"owner-{number % owners}",
        kind=("a", "b", "c")[number % 3],
        rank=number % 4,
    )


async def _collection(
    documents: int,
    indexed_fields: tuple[str, ...] = (),
) -> TransientDocumentCollection[_Document]:
    collection = await TransientDocumentDatabase().get_or_create_collection(
        name="documents",
        schema=_Document,
        document_loader=identity_loader_for(_Document),
    )
    await collection.ensure_indexes(
        [CollectionIndex(fields=((field, SortDirection.ASC),)) for field in indexed_fields]
    )

    for number in range(documents):
        await collection.insert_one(_document(number))

    return collection


def _examined(collection: TransientDocumentCollection[_Document]) -> int:
    examined = collection.examined_document_count
    collection.reset_examined_document_count()
    return examined


async def test_that_a_lookup_by_id_examines_only_the_document_with_that_id() -> None:
    collection = await _collection(documents=1000)
    _examined(collection)

    found = await collection.find_one({"id": {"$eq": "doc-734"}})

    assert found is not None and found["id"] == "doc-734"
    assert _examined(collection) == 1


async def test_that_a_field_without_an_index_is_found_by_examining_every_document() -> None:
    collection = await _collection(documents=300)
    _examined(collection)

    found = await collection.find({"owner_id": {"$eq": "owner-3"}})

    assert len(found.items) == 30
    assert _examined(collection) == 300


async def test_that_a_declared_index_limits_a_lookup_to_the_documents_holding_the_value() -> None:
    collection = await _collection(documents=300, indexed_fields=("owner_id",))
    _examined(collection)

    found = await collection.find({"owner_id": {"$eq": "owner-3"}})

    assert len(found.items) == 30
    assert _examined(collection) == 30


async def test_that_an_index_declared_after_documents_exist_covers_them() -> None:
    collection = await _collection(documents=300)
    await collection.ensure_indexes([CollectionIndex(fields=(("owner_id", SortDirection.ASC),))])
    _examined(collection)

    found = await collection.find({"owner_id": {"$eq": "owner-3"}})

    assert len(found.items) == 30
    assert _examined(collection) == 30


async def test_that_only_the_leading_field_of_a_compound_index_is_used_for_lookup() -> None:
    collection = await _collection(documents=300)
    await collection.ensure_indexes(
        [
            CollectionIndex(
                fields=(("owner_id", SortDirection.ASC), ("creation_utc", SortDirection.ASC))
            )
        ]
    )
    _examined(collection)

    await collection.find({"owner_id": {"$eq": "owner-3"}})
    by_leading_field = _examined(collection)
    await collection.find({"creation_utc": {"$eq": "2026-01-05"}})
    by_second_field = _examined(collection)

    assert by_leading_field == 30
    assert by_second_field == 300


async def test_that_alternatives_on_indexed_fields_examine_only_their_own_documents() -> None:
    collection = await _collection(documents=1000)
    _examined(collection)

    found = await collection.find(
        {"$or": [{"id": {"$eq": "doc-1"}}, {"id": {"$eq": "doc-2"}}, {"id": {"$eq": "doc-3"}}]}
    )

    assert sorted(document["id"] for document in found.items) == ["doc-1", "doc-2", "doc-3"]
    assert _examined(collection) == 3


async def test_that_the_most_selective_indexed_condition_decides_what_is_examined() -> None:
    collection = await _collection(documents=900, indexed_fields=("owner_id", "kind"))
    _examined(collection)

    found = await collection.find(
        {"$and": [{"kind": {"$eq": "a"}}, {"owner_id": {"$eq": "owner-3"}}]}
    )

    assert all(d["kind"] == "a" and d["owner_id"] == "owner-3" for d in found.items)
    assert _examined(collection) <= 90


async def test_that_an_alternative_without_an_index_falls_back_to_examining_everything() -> None:
    collection = await _collection(documents=200)
    _examined(collection)

    found = await collection.find({"$or": [{"id": {"$eq": "doc-1"}}, {"rank": {"$eq": 3}}]})

    assert len(found.items) == 51
    assert _examined(collection) == 200


async def test_that_updating_an_indexed_field_moves_the_document_in_the_index() -> None:
    collection = await _collection(documents=50, indexed_fields=("owner_id",))

    await collection.update_one(
        {"id": {"$eq": "doc-7"}}, cast(_Document, {"owner_id": "owner-new"})
    )

    assert [d["id"] for d in (await collection.find({"owner_id": {"$eq": "owner-new"}})).items] == [
        "doc-7"
    ]
    assert "doc-7" not in [
        d["id"] for d in (await collection.find({"owner_id": {"$eq": "owner-7"}})).items
    ]


async def test_that_a_deleted_document_leaves_the_index() -> None:
    collection = await _collection(documents=50, indexed_fields=("owner_id",))

    deleted = await collection.delete_one({"id": {"$eq": "doc-7"}})

    assert deleted.deleted_count == 1
    assert await collection.find_one({"id": {"$eq": "doc-7"}}) is None
    assert "doc-7" not in [
        d["id"] for d in (await collection.find({"owner_id": {"$eq": "owner-7"}})).items
    ]


def _random_filter(rng: random.Random, depth: int = 0) -> Where:
    if depth < 2 and rng.random() < 0.4:
        operator = rng.choice(["$and", "$or"])
        return cast(
            Where,
            {operator: [_random_filter(rng, depth + 1) for _ in range(rng.randint(1, 3))]},
        )

    return cast(
        Where,
        rng.choice(
            [
                {"owner_id": {"$eq": f"owner-{rng.randint(0, 11)}"}},
                {"id": {"$eq": f"doc-{rng.randint(0, 70)}"}},
                {"id": {"$in": [f"doc-{rng.randint(0, 70)}" for _ in range(rng.randint(0, 3))]}},
                {"kind": {"$in": rng.sample(["a", "b", "c", "z"], rng.randint(0, 3))}},
                {"rank": {"$gte": rng.randint(0, 3)}},
                {"kind": {"$ne": "a"}},
                {"kind": {"$eq": "b"}, "rank": {"$lt": 2}},
                {"owner_id": {"$nin": ["owner-1", "owner-2"]}},
                {},
            ]
        ),
    )


@pytest.mark.parametrize("seed", range(25))
async def test_that_indexed_queries_return_what_examining_every_document_would(
    seed: int,
) -> None:
    rng = random.Random(seed)
    collection = await _collection(documents=60, indexed_fields=("owner_id", "kind"))
    reference: list[_Document] = [_document(number) for number in range(60)]

    def matching(where: Where) -> list[_Document]:
        return [document for document in reference if matches_filters(where, document)]

    for _ in range(80):
        where = _random_filter(rng)
        action = rng.choice(["find", "find", "find_one", "update", "delete"])
        expected = matching(where)

        if action == "find":
            direction = rng.choice([SortDirection.ASC, SortDirection.DESC])
            found = await collection.find(where, sort_direction=direction)
            ordered = sorted(
                expected,
                key=lambda d: (d["creation_utc"], d["id"]),
                reverse=direction == SortDirection.DESC,
            )
            assert list(found.items) == ordered
            assert found.total_count == len(expected)
        elif action == "find_one":
            assert await collection.find_one(where) == (expected[0] if expected else None)
        elif action == "update":
            changes: dict[str, Any] = {
                "kind": rng.choice(["a", "b", "c"]),
                "rank": rng.randint(0, 3),
            }
            result = await collection.update_one(where, cast(_Document, changes))
            assert result.matched_count == (1 if expected else 0)
            if expected:
                position = reference.index(expected[0])
                reference[position] = cast(_Document, {**reference[position], **changes})
        else:
            result_of_delete = await collection.delete_one(where)
            assert result_of_delete.deleted_count == (1 if expected else 0)
            if expected:
                reference.remove(expected[0])


async def test_that_a_page_of_results_continues_from_its_cursor() -> None:
    collection = await _collection(documents=30, indexed_fields=("owner_id",))
    where: Where = {"owner_id": {"$in": ["owner-1", "owner-2", "owner-3"]}}

    first = await collection.find(where, limit=4)
    assert first.has_more and first.next_cursor is not None
    second = await collection.find(where, limit=4, cursor=first.next_cursor)

    everything = await collection.find(where)
    assert [*first.items, *second.items] == list(everything.items)[:8]
