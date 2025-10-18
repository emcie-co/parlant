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

import asyncio
import os
from typing import Any, AsyncIterator, Optional, cast
from pymongo import AsyncMongoClient
import pytest
from typing_extensions import Self
from lagom import Container
from pytest import fixture, raises

from parlant.core.common import Version
from parlant.adapters.db.mongo_db import MongoDocumentDatabase
from parlant.core.persistence.common import MigrationRequired, ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    FindResult,
    identity_loader,
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


class MongoTestDocument(BaseDocument):
    name: str


class DummyStore:
    VERSION = Version.from_string("2.0.0")

    class DummyDocumentV1(BaseDocument):
        name: str

    class DummyDocumentV2(BaseDocument):
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
            )
        elif doc["version"] == "2.0.0":
            return cast(DummyStore.DummyDocumentV2, doc)
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

    async def list_dummy(self) -> FindResult[DummyDocumentV2]:
        return await self._collection.find({})

    async def create_dummy(self, name: str, additional_field: str = "default") -> DummyDocumentV2:
        doc = self.DummyDocumentV2(
            id=ObjectId(f"dummy_{name}"),
            version=Version.String("2.0.0"),
            name=name,
            additional_field=additional_field,
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
                name="metadata", schema=BaseDocument, document_loader=identity_loader
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
