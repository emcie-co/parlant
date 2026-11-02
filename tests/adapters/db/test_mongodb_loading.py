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

from typing import Any, AsyncIterator, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from parlant.adapters.db.mongo_db import MongoDocumentDatabase
from parlant.core.loggers import Logger
from parlant.core.persistence.document_database import BaseDocument


class DocumentCursor:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.events: list[str] = []
        self.materialized = False

    async def to_list(self) -> list[dict[str, Any]]:
        self.materialized = True
        return self.documents

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        for document in self.documents:
            self.events.append(f"read:{document['id']}")
            yield document


def database_fixture(documents: list[dict[str, Any]]) -> tuple[Any, ...]:
    cursor = DocumentCursor(documents)
    source = Mock(
        find=Mock(return_value=cursor),
        replace_one=AsyncMock(),
        delete_one=AsyncMock(),
        create_index=AsyncMock(),
    )
    failed = Mock(insert_one=AsyncMock(), create_index=AsyncMock())
    mongo = Mock(
        get_collection=Mock(return_value=source),
        list_collection_names=AsyncMock(return_value=["test_db_customers_failed_migrations"]),
        create_collection=AsyncMock(return_value=failed),
        drop_collection=AsyncMock(),
    )
    database = MongoDocumentDatabase(
        cast(Any, {"test_db": mongo}), "test_db", MagicMock(spec=Logger)
    )
    return database, source, failed, mongo, cursor


async def test_loads_documents_incrementally_without_materializing_the_cursor() -> None:
    database, source, _, _, cursor = database_fixture([{"id": "a"}, {"id": "b"}])

    async def loader(document: BaseDocument) -> BaseDocument:
        cursor.events.append(f"load:{document['id']}")
        return document

    async with database:
        await database.get_collection("customers", BaseDocument, loader)

    assert not cursor.materialized
    assert cursor.events == ["read:a", "load:a", "read:b", "load:b"]
    source.create_index.assert_awaited_once()


@pytest.mark.parametrize("copy_document", [False, True])
async def test_unchanged_documents_are_not_replaced(copy_document: bool) -> None:
    original = {"id": "a", "version": "1.0.0", "metadata": {"tags": ["customer"]}}
    database, source, failed, mongo, _ = database_fixture([original])

    async def loader(document: BaseDocument) -> BaseDocument:
        return cast(BaseDocument, dict(document)) if copy_document else document

    async with database:
        await database.get_collection("customers", BaseDocument, loader)

    source.replace_one.assert_not_awaited()
    source.delete_one.assert_not_awaited()
    failed.insert_one.assert_not_awaited()
    mongo.drop_collection.assert_awaited_once_with("test_db_customers_failed_migrations")


async def test_changed_documents_are_replaced_with_the_loaded_version() -> None:
    original = {"id": "a", "version": "1.0.0", "name": "old"}
    converted = {**original, "version": "2.0.0", "name": "converted"}
    database, source, failed, _, _ = database_fixture([original])

    async def loader(document: BaseDocument) -> BaseDocument:
        return cast(BaseDocument, converted)

    async with database:
        await database.get_collection("customers", BaseDocument, loader)

    source.replace_one.assert_awaited_once_with(original, converted)
    source.delete_one.assert_not_awaited()
    failed.insert_one.assert_not_awaited()


@pytest.mark.parametrize("raises_error", [False, True])
async def test_failed_loads_keep_the_existing_archive_and_deletion_behavior(
    raises_error: bool,
) -> None:
    rejected = {"id": "a", "version": "unknown"}
    accepted = {"id": "b", "version": "1.0.0"}
    database, source, failed, mongo, cursor = database_fixture([rejected, accepted])
    loaded = []

    async def loader(document: BaseDocument) -> BaseDocument | None:
        loaded.append(document["id"])
        if document["id"] == "a":
            if raises_error:
                raise ValueError("Cannot convert document")
            return None
        return document

    async with database:
        await database.get_collection("customers", BaseDocument, loader)

    assert loaded == ["a", "b"]
    assert not cursor.materialized
    failed.insert_one.assert_awaited_once_with(rejected)
    assert mongo.create_collection.await_count == 1
    if raises_error:
        source.delete_one.assert_not_awaited()
    else:
        source.delete_one.assert_awaited_once_with(rejected)


async def test_a_failed_conversion_write_is_archived_without_deleting_the_source() -> None:
    original = {"id": "a", "version": "1.0.0"}
    database, source, failed, _, _ = database_fixture([original])
    source.replace_one.side_effect = RuntimeError("Write failed")

    async def loader(document: BaseDocument) -> BaseDocument:
        return cast(BaseDocument, {**document, "version": "2.0.0"})

    async with database:
        await database.get_collection("customers", BaseDocument, loader)

    failed.insert_one.assert_awaited_once_with(original)
    source.delete_one.assert_not_awaited()


async def test_in_place_loader_changes_are_compared_against_the_original_document() -> None:
    original = {"id": "a", "version": "1.0.0", "metadata": {"tags": ["old"]}}
    database, source, _, _, _ = database_fixture([original])

    async def loader(document: BaseDocument) -> BaseDocument:
        cast(dict[str, Any], document)["metadata"]["tags"].append("converted")
        return document

    async with database:
        await database.get_collection("customers", BaseDocument, loader)

    source.replace_one.assert_awaited_once_with(
        {"id": "a", "version": "1.0.0", "metadata": {"tags": ["old"]}},
        {"id": "a", "version": "1.0.0", "metadata": {"tags": ["old", "converted"]}},
    )
