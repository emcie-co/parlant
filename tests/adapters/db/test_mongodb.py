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

import asyncio
import os
from typing import Any, AsyncIterator, Optional, TypedDict, cast
from pymongo import AsyncMongoClient
import pytest
from typing_extensions import Self
from lagom import Container
from pytest import fixture, raises

from parlant.core.common import Version
from parlant.adapters.db.mongo_db import MongoDocumentDatabase
from parlant.core.persistence.common import Cursor, MigrationRequired, ObjectId, SortDirection
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    FindResult,
    identity_loader,
    identity_loader_for,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.loggers import Logger


@fixture
async def test_database_name() -> AsyncIterator[str]:
    yield "test_db"


async def pymongo_tasks_still_running() -> None:
    while any("pymongo" in str(t) for t in asyncio.all_tasks()):
        print(str(t) for t in asyncio.all_tasks())
        await asyncio.sleep(1)


@fixture
async def test_mongo_client() -> AsyncIterator[AsyncMongoClient[Any]]:
    test_mongo_server = os.environ.get("TEST_MONGO_SERVER")
    if test_mongo_server:
        client = AsyncMongoClient[Any](test_mongo_server)
        yield client
        await client.close()
        await pymongo_tasks_still_running()
    else:
        print("could not find `TEST_MONGO_SERVER` in environment, skipping mongo tests...")
        raise pytest.skip()


class MongoTestDocument(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    version: Version.String
    name: str


class DummyStore:
    VERSION = Version.from_string("2.0.0")

    class DummyDocumentV1(TypedDict, total=False):
        id: ObjectId
        creation_utc: str
        version: Version.String
        name: str

    class DummyDocumentV2(TypedDict, total=False):
        id: ObjectId
        creation_utc: str
        version: Version.String
        name: str
        additional_field: str

    def __init__(self, database: MongoDocumentDatabase, allow_migration: bool = True):
        self._database: MongoDocumentDatabase = database
        self._collection: DocumentCollection[DummyStore.DummyDocumentV2]
        self.allow_migration = allow_migration

    async def _document_loader(self, doc: BaseDocument) -> Optional[DummyDocumentV2]:
        if doc["version"] == "1.0.0":
            doc = cast(DummyStore.DummyDocumentV1, doc)
            return self.DummyDocumentV2(
                id=doc["id"],
                version=Version.String("2.0.0"),
                name=doc["name"],
                additional_field="default_value",
                creation_utc=str(doc.get("creation_utc", "2023-01-01T00:00:00Z")),
            )
        elif doc["version"] == "2.0.0":
            # Ensure creation_utc field exists for existing documents
            doc_with_creation = dict(doc)
            if "creation_utc" not in doc_with_creation:
                doc_with_creation["creation_utc"] = "2023-01-01T00:00:00Z"
            return cast(DummyStore.DummyDocumentV2, doc_with_creation)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self.allow_migration,
        ):
            self._collection = await self._database.get_or_create_collection(
                name="dummy_collection",
                schema=DummyStore.DummyDocumentV2,
                document_loader=self._document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    async def list_dummy(
        self,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[DummyDocumentV2]:
        if sort_direction is not None:
            return await self._collection.find(
                {}, limit=limit, cursor=cursor, sort_direction=sort_direction
            )
        return await self._collection.find({}, limit=limit, cursor=cursor)

    async def create_dummy(self, name: str, additional_field: str = "default") -> DummyDocumentV2:
        from datetime import datetime, timezone

        doc = self.DummyDocumentV2(
            id=ObjectId(f"dummy_{name}"),
            version=Version.String("2.0.0"),
            name=name,
            additional_field=additional_field,
            creation_utc=datetime.now(timezone.utc).isoformat(),
        )
        await self._collection.insert_one(doc)
        return doc

    async def read_dummy(self, doc_id: str) -> Optional[DummyDocumentV2]:
        return await self._collection.find_one({"id": {"$eq": doc_id}})

    async def update_dummy(self, doc_id: str, name: str) -> Optional[DummyDocumentV2]:
        # First get the existing document to preserve other fields
        existing = await self._collection.find_one({"id": {"$eq": doc_id}})
        if existing is None:
            return None

        # Create updated document with changed name
        updated_doc = self.DummyDocumentV2(
            id=existing["id"],
            version=existing["version"],
            name=name,
            additional_field=existing["additional_field"],
            creation_utc=existing["creation_utc"],
        )

        result = await self._collection.update_one({"id": {"$eq": doc_id}}, updated_doc)
        return result.updated_document

    async def delete_dummy(self, doc_id: str) -> bool:
        result = await self._collection.delete_one({"id": {"$eq": doc_id}})
        return result.acknowledged and result.deleted_count > 0


async def test_that_dummy_documents_can_be_created_and_persisted(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    created_dummy = None

    async with MongoDocumentDatabase(
        test_mongo_client,
        test_database_name,
        container[Logger],
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            created_dummy = await dummy_store.create_dummy(name="test-dummy")

            dummies = await dummy_store.list_dummy()
            assert dummies.total_count == 1
            assert dummies.items[0] == created_dummy

    assert created_dummy
    assert created_dummy["name"] == "test-dummy"
    assert created_dummy["additional_field"] == "default"

    # Verify persistence after reopening
    async with MongoDocumentDatabase(
        test_mongo_client,
        test_database_name,
        container[Logger],
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            actual_dummies = await dummy_store.list_dummy()
            assert actual_dummies.total_count == 1

            db_dummy = actual_dummies.items[0]
            assert db_dummy["id"] == created_dummy["id"]
            assert db_dummy["name"] == created_dummy["name"]
            assert db_dummy["additional_field"] == created_dummy["additional_field"]


async def test_that_dummy_documents_can_be_retrieved_by_id(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    created_dummy = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            created_dummy = await dummy_store.create_dummy(
                name="retrievable_dummy", additional_field="custom_value"
            )

            retrieved_dummy = await dummy_store.read_dummy(created_dummy["id"])

            assert created_dummy == retrieved_dummy


async def test_that_multiple_dummy_documents_can_be_created_and_retrieved(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    first_dummy = None
    second_dummy = None

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            first_dummy = await dummy_store.create_dummy(
                name="first_dummy", additional_field="first_value"
            )

            second_dummy = await dummy_store.create_dummy(
                name="second_dummy", additional_field="second_value"
            )

    assert first_dummy
    assert second_dummy

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            dummies = await dummy_store.list_dummy()
            assert dummies.total_count == 2

            dummy_ids = [d["id"] for d in dummies.items]
            assert first_dummy["id"] in dummy_ids
            assert second_dummy["id"] in dummy_ids

            for dummy in dummies.items:
                if dummy["id"] == first_dummy["id"]:
                    assert dummy["name"] == "first_dummy"
                    assert dummy["additional_field"] == "first_value"
                elif dummy["id"] == second_dummy["id"]:
                    assert dummy["name"] == "second_dummy"
                    assert dummy["additional_field"] == "second_value"


async def test_that_dummy_documents_can_be_updated(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            original_dummy = await dummy_store.create_dummy(
                name="original_name", additional_field="original_value"
            )

            updated_dummy = await dummy_store.update_dummy(original_dummy["id"], "updated_name")

            assert updated_dummy
            assert updated_dummy["id"] == original_dummy["id"]
            assert updated_dummy["name"] == "updated_name"
            assert updated_dummy["additional_field"] == "original_value"  # Should remain unchanged

            # Verify the update persisted
            retrieved_dummy = await dummy_store.read_dummy(original_dummy["id"])
            assert retrieved_dummy
            assert retrieved_dummy["name"] == "updated_name"


async def test_that_dummy_documents_can_be_deleted(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            dummy_to_delete = await dummy_store.create_dummy(
                name="deletable_dummy", additional_field="will_be_deleted"
            )

            # Verify it exists
            dummies_before = await dummy_store.list_dummy()
            assert dummies_before.total_count == 1

            # Delete it
            deletion_result = await dummy_store.delete_dummy(dummy_to_delete["id"])
            assert deletion_result is True

            # Verify it's gone
            dummies_after = await dummy_store.list_dummy()
            assert dummies_after.total_count == 0

            # Verify we can't retrieve it
            retrieved_dummy = await dummy_store.read_dummy(dummy_to_delete["id"])
            assert retrieved_dummy is None


async def test_that_database_initialization_creates_collections(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            await dummy_store.create_dummy(
                name="initialization_test", additional_field="test_value"
            )

    collections = await test_mongo_client[test_database_name].list_collection_names()
    assert "dummy_collection" in collections


async def test_that_document_upgrade_happens_during_loading_of_store(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    adb = test_mongo_client[test_database_name]
    await adb.metadata.insert_one({"id": "123", "version": "1.0.0"})
    await adb.dummy_collection.insert_one(
        {"id": "dummy_id", "version": "1.0.0", "name": "Test Document"}
    )

    logger = container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        async with DummyStore(db, allow_migration=True) as store:
            result = await store.list_dummy()

            assert result.total_count == 1
            upgraded_doc = result.items[0]
            assert upgraded_doc["version"] == "2.0.0"
            assert upgraded_doc["name"] == "Test Document"
            assert upgraded_doc["additional_field"] == "default_value"


async def test_that_migration_is_not_needed_for_new_store(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    logger = container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata",
                schema=BaseDocument,
                document_loader=identity_loader,
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_that_failed_migrations_are_tracked_in_separate_collection(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    adb = test_mongo_client[test_database_name]
    await adb.metadata.insert_one({"id": "meta_id", "version": "1.0.0"})
    await adb.dummy_collection.insert_one(
        {
            "id": "invalid_dummy_id",
            "version": "3.0",
            "name": "Unmigratable Document",
        }
    )

    logger = container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        async with DummyStore(db, allow_migration=True) as store:
            result = await store.list_dummy()

            assert result.total_count == 0

            failed_migrations_collection = await db.get_collection(
                "test_db_dummy_collection_failed_migrations",
                BaseDocument,
                identity_loader,
            )
            result_of_failed_migrations = await failed_migrations_collection.find({})

            assert result_of_failed_migrations.total_count == 1
            failed_doc = result_of_failed_migrations.items[0]
            assert failed_doc["id"] == "invalid_dummy_id"
            assert failed_doc["version"] == "3.0"
            assert failed_doc.get("name") == "Unmigratable Document"


async def test_that_version_mismatch_raises_error_when_migration_is_required_but_disabled(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    adb = test_mongo_client[test_database_name]
    await adb.metadata.insert_one({"id": "meta_id", "version": "1.5.0"})

    logger = container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        with raises(MigrationRequired) as exc_info:
            async with DummyStore(db, allow_migration=False) as _:
                pass

        assert "Migration required for DummyStore." in str(exc_info.value)


async def test_that_persistence_and_store_version_match_allows_store_to_open_when_migrate_is_disabled(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    adb = test_mongo_client[test_database_name]
    await adb.metadata.insert_one({"id": "meta_id", "version": "2.0.0"})

    logger = container[Logger]

    async with MongoDocumentDatabase(test_mongo_client, "test_db", logger) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata",
                schema=BaseDocument,
                document_loader=identity_loader,
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_that_collections_can_be_deleted(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    await test_mongo_client.drop_database(test_database_name)

    logger = container[Logger]

    async def test_document_loader(doc: BaseDocument) -> Optional[MongoTestDocument]:
        return cast(MongoTestDocument, doc)

    async with MongoDocumentDatabase(test_mongo_client, test_database_name, logger) as mongo_db:
        # Create a simple collection
        await mongo_db.get_or_create_collection(
            name="test_collection",
            schema=MongoTestDocument,
            document_loader=test_document_loader,
        )

        # Insert a test document using the raw pymongo client
        await test_mongo_client[test_database_name]["test_collection"].insert_one(
            {"id": "test_id", "version": "1.0.0", "name": "Test Document"}
        )

        collections = await test_mongo_client[test_database_name].list_collection_names()
        assert "test_collection" in collections

        await mongo_db.delete_collection("test_collection")

        collections = await test_mongo_client[test_database_name].list_collection_names()
        assert "test_collection" not in collections


async def test_that_dummy_documents_can_be_listed_with_pagination_limit(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that dummy documents can be listed with a limit for pagination."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            # Create multiple documents
            for i in range(5):
                await dummy_store.create_dummy(f"doc{i}", f"value{i}")

            # List with limit
            result = await dummy_store.list_dummy(limit=3)

            assert len(result.items) == 3
            assert result.total_count == 4  # 3 returned items + 1 extra for has_more check
            assert result.has_more
            assert result.next_cursor is not None


async def test_that_dummy_documents_are_sorted_by_creation_time_descending(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that dummy documents are automatically sorted by creation_utc in descending order."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            # Create documents with small delays to ensure different timestamps
            import asyncio

            await dummy_store.create_dummy("first", "field1")
            await asyncio.sleep(0.01)
            await dummy_store.create_dummy("second", "field2")
            await asyncio.sleep(0.01)
            await dummy_store.create_dummy("third", "field3")

            result = await dummy_store.list_dummy(sort_direction=SortDirection.DESC)

            assert len(result.items) == 3
            # Most recent first (descending order)
            assert result.items[0]["name"] == "third"
            assert result.items[1]["name"] == "second"
            assert result.items[2]["name"] == "first"


async def test_that_dummy_documents_can_be_paginated_using_cursor(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that dummy documents can be paginated using cursor-based pagination."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            # Create documents with small delays to ensure different timestamps
            import asyncio

            doc1 = await dummy_store.create_dummy("first", "field1")
            await asyncio.sleep(0.01)
            await dummy_store.create_dummy("second", "field2")
            await asyncio.sleep(0.01)
            await dummy_store.create_dummy("third", "field3")

            # Create cursor from doc1 (the oldest document, which will be first in asc order)
            # This should return the documents that come after it in the sorted list
            cursor = Cursor(creation_utc=doc1["creation_utc"], id=doc1["id"])

            # Find documents after cursor
            result = await dummy_store.list_dummy(cursor=cursor)

            assert len(result.items) == 2
            # Should get the documents created after doc1 in ascending order (second, then third)
            assert result.items[0]["name"] == "second"
            assert result.items[1]["name"] == "third"


async def test_that_dummy_documents_support_multi_page_cursor_pagination(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that dummy documents support cursor-based pagination across multiple pages."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            # Create 5 dummy documents with small delays
            import asyncio

            docs = []
            for i in range(5):
                doc = await dummy_store.create_dummy(f"doc{i:02d}", f"field{i}")
                docs.append(doc)
                if i < 4:  # Don't sleep after the last one
                    await asyncio.sleep(0.01)

            # First page: get first 2 documents
            result1 = await dummy_store.list_dummy(limit=2)

            assert len(result1.items) == 2
            assert result1.has_more
            assert result1.next_cursor is not None

            # Second page: use cursor from first page
            result2 = await dummy_store.list_dummy(limit=2, cursor=result1.next_cursor)

            assert len(result2.items) == 2
            assert result2.has_more
            assert result2.next_cursor is not None

            # Third page: use cursor from second page
            result3 = await dummy_store.list_dummy(limit=2, cursor=result2.next_cursor)

            assert len(result3.items) == 1
            assert not result3.has_more
            assert result3.next_cursor is None


async def test_that_all_operations_can_be_cleaned_up_properly(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that we properly clean up all operations in each test."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        async with DummyStore(dummy_db) as dummy_store:
            # Create some dummy data
            dummy1 = await dummy_store.create_dummy("test1", "value1")
            dummy2 = await dummy_store.create_dummy("test2", "value2")
            await dummy_store.create_dummy("test3", "value3")

            # Verify creation
            dummies = await dummy_store.list_dummy()
            assert dummies.total_count == 3

            # Update one
            updated = await dummy_store.update_dummy(dummy1["id"], "updated_name")
            assert updated
            assert updated["name"] == "updated_name"

            # Delete one
            deleted = await dummy_store.delete_dummy(dummy2["id"])
            assert deleted is True

            # Verify final state has 2 items
            final_dummies = await dummy_store.list_dummy()
            assert final_dummies.total_count == 2

            # Clean up all remaining items
            for dummy in final_dummies.items:
                await dummy_store.delete_dummy(dummy["id"])

            # Verify all cleaned up
            after_cleanup = await dummy_store.list_dummy()
            assert after_cleanup.total_count == 0

    # Verify we can drop the database completely
    await test_mongo_client.drop_database(test_database_name)

    # After drop, database should not exist or be empty
    try:
        collections_after_drop = await test_mongo_client[test_database_name].list_collection_names()
        assert len(collections_after_drop) == 0
    except Exception:
        # Database might not exist anymore, which is also acceptable
        pass


async def test_that_documents_can_be_sorted_in_ascending_order(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that documents can be sorted by creation_utc in ascending order (oldest first)."""
    await test_mongo_client.drop_database(test_database_name)

    async def mongo_test_document_loader(doc: BaseDocument) -> Optional[MongoTestDocument]:
        return cast(MongoTestDocument, doc)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        collection = await dummy_db.get_or_create_collection(
            name="test_collection",
            schema=MongoTestDocument,
            document_loader=mongo_test_document_loader,
        )

        # Create documents with different timestamps
        doc1 = MongoTestDocument(
            id=ObjectId("doc1"),
            creation_utc="2023-01-01T10:00:00Z",
            version=Version.String("1.0.0"),
            name="first",
        )
        doc2 = MongoTestDocument(
            id=ObjectId("doc2"),
            creation_utc="2023-01-01T11:00:00Z",
            version=Version.String("1.0.0"),
            name="second",
        )
        doc3 = MongoTestDocument(
            id=ObjectId("doc3"),
            creation_utc="2023-01-01T12:00:00Z",
            version=Version.String("1.0.0"),
            name="third",
        )

        await collection.insert_one(doc1)
        await collection.insert_one(doc2)
        await collection.insert_one(doc3)

        # Test ascending sort (oldest first)
        result = await collection.find({}, sort_direction=SortDirection.ASC)

        assert len(result.items) == 3
        assert result.items[0]["name"] == "first"  # Oldest
        assert result.items[1]["name"] == "second"  # Middle
        assert result.items[2]["name"] == "third"  # Newest


async def test_that_documents_can_be_sorted_in_descending_order(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that documents can be sorted by creation_utc in descending order (newest first)."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        collection = await dummy_db.get_or_create_collection(
            name="test_collection",
            schema=MongoTestDocument,
            document_loader=identity_loader_for(MongoTestDocument),
        )

        # Create documents with different timestamps
        doc1 = MongoTestDocument(
            id=ObjectId("doc1"),
            creation_utc="2023-01-01T10:00:00Z",
            version=Version.String("1.0.0"),
            name="first",
        )
        doc2 = MongoTestDocument(
            id=ObjectId("doc2"),
            creation_utc="2023-01-01T11:00:00Z",
            version=Version.String("1.0.0"),
            name="second",
        )
        doc3 = MongoTestDocument(
            id=ObjectId("doc3"),
            creation_utc="2023-01-01T12:00:00Z",
            version=Version.String("1.0.0"),
            name="third",
        )

        await collection.insert_one(doc1)
        await collection.insert_one(doc2)
        await collection.insert_one(doc3)

        # Test descending sort (newest first)
        result = await collection.find({}, sort_direction=SortDirection.DESC)

        assert len(result.items) == 3
        assert result.items[0]["name"] == "third"  # Newest
        assert result.items[1]["name"] == "second"  # Middle
        assert result.items[2]["name"] == "first"  # Oldest


async def test_that_cursor_pagination_works_with_ascending_sort(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that cursor-based pagination works correctly with ascending sort."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        collection = await dummy_db.get_or_create_collection(
            name="test_collection",
            schema=MongoTestDocument,
            document_loader=identity_loader_for(MongoTestDocument),
        )

        # Create documents with different timestamps
        doc1 = MongoTestDocument(
            id=ObjectId("doc1"),
            creation_utc="2023-01-01T10:00:00Z",
            version=Version.String("1.0.0"),
            name="first",
        )
        doc2 = MongoTestDocument(
            id=ObjectId("doc2"),
            creation_utc="2023-01-01T11:00:00Z",
            version=Version.String("1.0.0"),
            name="second",
        )
        doc3 = MongoTestDocument(
            id=ObjectId("doc3"),
            creation_utc="2023-01-01T12:00:00Z",
            version=Version.String("1.0.0"),
            name="third",
        )

        await collection.insert_one(doc1)
        await collection.insert_one(doc2)
        await collection.insert_one(doc3)

        # Get first page with ascending sort
        first_page = await collection.find({}, limit=1, sort_direction=SortDirection.ASC)

        assert len(first_page.items) == 1
        assert first_page.items[0]["name"] == "first"  # Oldest first
        assert first_page.has_more is True
        assert first_page.next_cursor is not None

        # Get second page using cursor
        second_page = await collection.find(
            {}, limit=1, cursor=first_page.next_cursor, sort_direction=SortDirection.ASC
        )

        assert len(second_page.items) == 1
        assert second_page.items[0]["name"] == "second"  # Next oldest
        assert second_page.has_more is True
        assert second_page.next_cursor is not None

        # Get third page using cursor
        third_page = await collection.find(
            {}, limit=1, cursor=second_page.next_cursor, sort_direction=SortDirection.ASC
        )

        assert len(third_page.items) == 1
        assert third_page.items[0]["name"] == "third"  # Newest
        assert third_page.has_more is False
        assert third_page.next_cursor is None


async def test_that_cursor_pagination_works_with_descending_sort(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that cursor-based pagination works correctly with descending sort."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        collection = await dummy_db.get_or_create_collection(
            name="test_collection",
            schema=MongoTestDocument,
            document_loader=identity_loader_for(MongoTestDocument),
        )

        # Create documents with different timestamps
        doc1 = MongoTestDocument(
            id=ObjectId("doc1"),
            creation_utc="2023-01-01T10:00:00Z",
            version=Version.String("1.0.0"),
            name="first",
        )
        doc2 = MongoTestDocument(
            id=ObjectId("doc2"),
            creation_utc="2023-01-01T11:00:00Z",
            version=Version.String("1.0.0"),
            name="second",
        )
        doc3 = MongoTestDocument(
            id=ObjectId("doc3"),
            creation_utc="2023-01-01T12:00:00Z",
            version=Version.String("1.0.0"),
            name="third",
        )

        await collection.insert_one(doc1)
        await collection.insert_one(doc2)
        await collection.insert_one(doc3)

        # Get first page with descending sort
        first_page = await collection.find({}, limit=1, sort_direction=SortDirection.DESC)

        assert len(first_page.items) == 1
        assert first_page.items[0]["name"] == "third"  # Newest first
        assert first_page.has_more is True
        assert first_page.next_cursor is not None

        # Get second page using cursor
        second_page = await collection.find(
            {}, limit=1, cursor=first_page.next_cursor, sort_direction=SortDirection.DESC
        )

        assert len(second_page.items) == 1
        assert second_page.items[0]["name"] == "second"  # Next newest
        assert second_page.has_more is True
        assert second_page.next_cursor is not None

        # Get third page using cursor
        third_page = await collection.find(
            {}, limit=1, cursor=second_page.next_cursor, sort_direction=SortDirection.DESC
        )

        assert len(third_page.items) == 1
        assert third_page.items[0]["name"] == "first"  # Oldest
        assert third_page.has_more is False
        assert third_page.next_cursor is None


async def test_that_default_sort_direction_is_ascending(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that the default sort direction is ascending (oldest first)."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        collection = await dummy_db.get_or_create_collection(
            name="test_collection",
            schema=MongoTestDocument,
            document_loader=identity_loader_for(MongoTestDocument),
        )

        # Create documents with different timestamps
        doc1 = MongoTestDocument(
            id=ObjectId("doc1"),
            creation_utc="2023-01-01T10:00:00Z",
            version=Version.String("1.0.0"),
            name="first",
        )
        doc2 = MongoTestDocument(
            id=ObjectId("doc2"),
            creation_utc="2023-01-01T11:00:00Z",
            version=Version.String("1.0.0"),
            name="second",
        )

        await collection.insert_one(doc1)
        await collection.insert_one(doc2)

        # Test default sort (should be ascending)
        result = await collection.find({})

        assert len(result.items) == 2
        assert result.items[0]["name"] == "first"  # Older document first (ascending)
        assert result.items[1]["name"] == "second"  # Newer document second


async def test_that_creation_utc_index_is_created_for_new_collections(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that creation_utc field is automatically indexed when creating a new collection."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        collection = await dummy_db.create_collection(
            name="test_new_collection",
            schema=MongoTestDocument,
        )

        # Access the underlying PyMongo collection to check indexes
        from parlant.adapters.db.mongo_db import MongoDocumentCollection

        mongo_collection = cast(MongoDocumentCollection[MongoTestDocument], collection)

        # Get index information
        indexes = await mongo_collection._collection.index_information()

        # Check that creation_utc index exists
        creation_utc_index_found = False
        for index_name, index_info in indexes.items():
            if index_name != "_id_":  # Skip the default _id index
                # Check if this index includes creation_utc field
                index_keys = index_info.get("key", [])
                for field_name, _ in index_keys:
                    if field_name == "creation_utc":
                        creation_utc_index_found = True
                        break

        assert creation_utc_index_found, "creation_utc index should be created for new collections"


async def test_that_creation_utc_index_is_created_for_existing_collections(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that creation_utc field is automatically indexed when accessing existing collections."""
    await test_mongo_client.drop_database(test_database_name)

    # First, create a collection directly with PyMongo (without our wrapper)
    database = test_mongo_client[test_database_name]
    raw_collection = database["test_existing_collection"]

    # Insert a document to ensure the collection exists
    await raw_collection.insert_one(
        {
            "id": "test_doc",
            "creation_utc": "2023-01-01T00:00:00Z",
            "version": "1.0.0",
            "name": "test",
        }
    )

    # Verify there's no creation_utc index initially
    initial_indexes = await raw_collection.index_information()
    creation_utc_index_exists_initially = any(
        any(field_name == "creation_utc" for field_name, _ in index_info.get("key", []))
        for index_name, index_info in initial_indexes.items()
        if index_name != "_id_"
    )
    assert not creation_utc_index_exists_initially, "creation_utc index should not exist initially"

    # Now access the collection through our wrapper
    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        collection = await dummy_db.get_collection(
            name="test_existing_collection",
            schema=MongoTestDocument,
            document_loader=identity_loader_for(MongoTestDocument),
        )

        # Access the underlying PyMongo collection to check indexes
        from parlant.adapters.db.mongo_db import MongoDocumentCollection

        mongo_collection = cast(MongoDocumentCollection[MongoTestDocument], collection)

        # Get index information after our wrapper processed the collection
        indexes = await mongo_collection._collection.index_information()

        # Check that creation_utc index now exists
        creation_utc_index_found = False
        for index_name, index_info in indexes.items():
            if index_name != "_id_":  # Skip the default _id index
                # Check if this index includes creation_utc field
                index_keys = index_info.get("key", [])
                for field_name, _ in index_keys:
                    if field_name == "creation_utc":
                        creation_utc_index_found = True
                        break

        assert creation_utc_index_found, (
            "creation_utc index should be created for existing collections"
        )


async def test_that_creation_utc_index_is_created_for_get_or_create_collections(
    container: Container,
    test_mongo_client: AsyncMongoClient[Any],
    test_database_name: str,
) -> None:
    """Test that creation_utc field is automatically indexed when using get_or_create_collection."""
    await test_mongo_client.drop_database(test_database_name)

    async with MongoDocumentDatabase(
        test_mongo_client, test_database_name, container[Logger]
    ) as dummy_db:
        collection = await dummy_db.get_or_create_collection(
            name="test_get_or_create_collection",
            schema=MongoTestDocument,
            document_loader=identity_loader_for(MongoTestDocument),
        )

        # Access the underlying PyMongo collection to check indexes
        from parlant.adapters.db.mongo_db import MongoDocumentCollection

        mongo_collection = cast(MongoDocumentCollection[MongoTestDocument], collection)

        # Get index information
        indexes = await mongo_collection._collection.index_information()

        # Check that creation_utc index exists
        creation_utc_index_found = False
        for index_name, index_info in indexes.items():
            if index_name != "_id_":  # Skip the default _id index
                # Check if this index includes creation_utc field
                index_keys = index_info.get("key", [])
                for field_name, _ in index_keys:
                    if field_name == "creation_utc":
                        creation_utc_index_found = True
                        break

        assert creation_utc_index_found, (
            "creation_utc index should be created for get_or_create collections"
        )
