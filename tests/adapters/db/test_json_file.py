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
from pathlib import Path
from typing import AsyncIterator, Optional, TypedDict, cast
from typing_extensions import Self
import tempfile
from pytest import fixture

from parlant.core.common import Version
from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    FindResult,
    identity_loader_for,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.loggers import Logger
from parlant.core.persistence.common import MigrationRequired
from parlant.core.persistence.document_database import identity_loader
import json
from pytest import raises


@fixture
async def new_file() -> AsyncIterator[Path]:
    with tempfile.NamedTemporaryFile() as file:
        yield Path(file.name)


@fixture
def logger() -> Logger:
    """Simple logger for testing."""

    class TestLogger:
        def info(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            pass

        def debug(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

    return TestLogger()  # type: ignore


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

    def __init__(self, database: JSONFileDocumentDatabase, allow_migration: bool = True):
        self._database = database
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


async def test_that_dummy_documents_can_be_created(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that dummy documents can be created with all required fields."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            doc = await store.create_dummy("test_doc", "test_value")

            assert doc["name"] == "test_doc"
            assert doc["additional_field"] == "test_value"
            assert doc["version"] == "2.0.0"
            assert doc["id"] == "dummy_test_doc"
            assert doc["creation_utc"]


async def test_that_dummy_documents_can_be_read_by_id(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that dummy documents can be retrieved by ID and non-existent IDs return None."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            # Create a document first
            created_doc = await store.create_dummy("read_test", "read_value")

            # Read it back
            retrieved_doc = await store.read_dummy(created_doc["id"])

            assert retrieved_doc is not None
            assert retrieved_doc["name"] == "read_test"
            assert retrieved_doc["additional_field"] == "read_value"
            assert retrieved_doc["id"] == created_doc["id"]

            # Test reading non-existent document
            non_existent = await store.read_dummy("dummy_non_existent")
            assert non_existent is None


async def test_that_dummy_documents_can_be_updated_by_id(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that dummy documents can be updated by ID while preserving other fields."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            # Create a document first
            created_doc = await store.create_dummy("update_test", "original_value")

            # Update it
            updated_doc = await store.update_dummy(created_doc["id"], "updated_name")

            assert updated_doc is not None
            assert updated_doc["name"] == "updated_name"
            assert updated_doc["additional_field"] == "original_value"  # Should remain unchanged
            assert updated_doc["id"] == created_doc["id"]

            # Verify the update persisted
            retrieved_doc = await store.read_dummy(created_doc["id"])
            assert retrieved_doc is not None
            assert retrieved_doc["name"] == "updated_name"

            # Test updating non-existent document
            non_updated = await store.update_dummy("dummy_non_existent", "new_name")
            assert non_updated is None


async def test_that_dummy_documents_can_be_deleted_by_id(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that dummy documents can be deleted by ID and deletion is persisted."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            # Create a document first
            created_doc = await store.create_dummy("delete_test", "delete_value")

            # Verify it exists
            retrieved_doc = await store.read_dummy(created_doc["id"])
            assert retrieved_doc is not None

            # Delete it
            delete_result = await store.delete_dummy(created_doc["id"])
            assert delete_result is True

            # Verify it's gone
            deleted_doc = await store.read_dummy(created_doc["id"])
            assert deleted_doc is None

            # Test deleting non-existent document
            delete_non_existent = await store.delete_dummy("dummy_non_existent")
            assert delete_non_existent is False


async def test_that_all_dummy_documents_can_be_listed(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that all dummy documents can be listed without pagination."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            # Create multiple documents
            docs = []
            for i in range(3):
                doc = await store.create_dummy(f"doc{i}", f"value{i}")
                docs.append(doc)

            # List all documents
            result = await store.list_dummy()

            assert len(result.items) == 3
            assert result.total_count == 3
            assert not result.has_more
            assert result.next_cursor is None


async def test_that_dummy_documents_can_be_listed_with_pagination_limit(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that dummy documents can be listed with a limit for pagination."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            # Create multiple documents
            for i in range(5):
                await store.create_dummy(f"doc{i}", f"value{i}")

            # List with limit
            result = await store.list_dummy(limit=3)

            assert len(result.items) == 3
            assert result.total_count == 5
            assert result.has_more
            assert result.next_cursor is not None


async def test_that_dummy_documents_are_sorted_by_creation_time_descending(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that dummy documents are automatically sorted by creation_utc in descending order."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            _ = await store.create_dummy("charlie", "field1")
            _ = await store.create_dummy("alice", "field2")
            _ = await store.create_dummy("bob", "field3")

            result = await store.list_dummy(sort_direction=SortDirection.DESC)

            assert len(result.items) == 3
            assert result.items[0]["name"] == "bob"
            assert result.items[1]["name"] == "alice"
            assert result.items[2]["name"] == "charlie"


async def test_that_dummy_documents_can_be_paginated_using_cursor(
    new_file: Path,
    logger: Logger,
) -> None:
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            # Create documents with different names for sorting
            doc1 = await store.create_dummy("first", "field1")
            await store.create_dummy("second", "field2")
            await store.create_dummy("third", "field3")

            # Create cursor from doc1 (the oldest document, which will be first in asc order)
            # This should return the documents that come after it in the sorted list
            cursor = Cursor(creation_utc=doc1["creation_utc"], id=doc1["id"])

            # Find documents after cursor
            result = await store.list_dummy(cursor=cursor)

            assert len(result.items) == 2
            # Should get the documents created after doc1 in ascending order (second, then third)
            assert result.items[0]["name"] == "second"
            assert result.items[1]["name"] == "third"


async def test_that_dummy_documents_support_multi_page_cursor_pagination(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that dummy documents support cursor-based pagination across multiple pages."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            # Create 5 dummy documents
            docs = []
            for i in range(5):
                doc = await store.create_dummy(f"doc{i:02d}", f"field{i}")
                docs.append(doc)

            # First page: get first 2 documents
            result1 = await store.list_dummy(limit=2)

            assert len(result1.items) == 2
            assert result1.has_more
            assert result1.next_cursor is not None

            # Second page: use cursor from first page
            result2 = await store.list_dummy(limit=2, cursor=result1.next_cursor)

            assert len(result2.items) == 2
            assert result2.has_more
            assert result2.next_cursor is not None

            # Third page: use cursor from second page
            result3 = await store.list_dummy(limit=2, cursor=result2.next_cursor)

            assert len(result3.items) == 1
            assert not result3.has_more
            assert result3.next_cursor is None


async def test_that_documents_are_migrated_from_v1_to_v2_during_store_loading(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that documents are automatically migrated from v1 to v2 format during store loading."""
    with open(new_file, "w") as f:
        json.dump(
            {
                "metadata": [
                    {
                        "id": "123",
                        "version": "1.0.0",
                    }
                ],
                "dummy_collection": [
                    {
                        "id": "dummy_id",
                        "version": "1.0.0",
                        "name": "Test Document",
                    }
                ],
            },
            f,
        )

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db, allow_migration=True) as store:
            result = await store.list_dummy()

            assert result.total_count == 1
            upgraded_doc = result.items[0]
            assert upgraded_doc["version"] == "2.0.0"
            assert upgraded_doc["name"] == "Test Document"
            assert upgraded_doc["additional_field"] == "default_value"


async def test_that_migration_is_not_needed_for_new_store(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that new stores don't require migration."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata", schema=BaseDocument, document_loader=identity_loader
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_that_failed_migrations_are_stored_in_separate_collection(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that failed migrations are stored in a separate failed_migrations collection."""
    with open(new_file, "w") as f:
        json.dump(
            {
                "metadata": [
                    {
                        "id": "meta_id",
                        "creation_utc": "2023-01-01T00:00:00Z",
                        "version": "1.0.0",
                    },
                ],
                "dummy_collection": [
                    {
                        "id": "invalid_dummy_id",
                        "creation_utc": "2023-01-01T00:00:00Z",
                        "version": "3.0",
                        "name": "Unmigratable Document",
                    }
                ],
            },
            f,
        )

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db, allow_migration=True) as store:
            result = await store.list_dummy()

            assert result.total_count == 0

            failed_migrations_collection = await db.get_collection(
                "failed_migrations", BaseDocument, identity_loader
            )
            result_of_failed_migrations = await failed_migrations_collection.find({})

            assert result_of_failed_migrations.total_count == 1
            failed_doc = result_of_failed_migrations.items[0]
            assert failed_doc["id"] == "invalid_dummy_id"
            assert failed_doc["version"] == "3.0"
            assert failed_doc.get("name") == "Unmigratable Document"


async def test_that_version_mismatch_raises_error_when_migration_is_required_but_disabled(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that version mismatch raises error when migration is disabled."""
    with open(new_file, "w") as f:
        json.dump(
            {
                "metadata": [
                    {"id": "meta_id", "version": "0.0.1"},
                ]
            },
            f,
        )

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        with raises(MigrationRequired) as exc_info:
            async with DummyStore(db, allow_migration=False) as _:
                pass

        assert "Migration required for DummyStore." in str(exc_info.value)


async def test_that_persistence_and_store_version_match_allows_store_to_open_when_migrate_is_disabled(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that matching versions allow store to open without migration."""
    with open(new_file, "w") as f:
        json.dump(
            {
                "metadata": [
                    {"id": "meta_id", "version": "2.0.0"},
                ]
            },
            f,
        )

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db, allow_migration=False):
            meta_collection = await db.get_or_create_collection(
                name="metadata",
                schema=BaseDocument,
                document_loader=identity_loader,
            )
            meta_document = await meta_collection.find_one({})

            assert meta_document
            assert meta_document["version"] == "2.0.0"


async def test_that_empty_json_files_can_be_loaded_successfully(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that empty JSON files can be loaded and used to create new documents."""
    # Create an empty file
    new_file.touch()

    async with JSONFileDocumentDatabase(logger, new_file) as db:
        async with DummyStore(db) as store:
            doc = await store.create_dummy("test_doc", "test_value")

            assert doc["name"] == "test_doc"
            assert doc["additional_field"] == "test_value"

            # Verify it was saved
            result = await store.list_dummy()
            assert result.total_count == 1


async def test_that_documents_can_be_sorted_in_ascending_order(
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that documents can be sorted by creation_utc in ascending order (oldest first)."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        collection = await db.get_or_create_collection(
            name="test_collection",
            schema=DummyStore.DummyDocumentV2,
            document_loader=identity_loader_for(DummyStore.DummyDocumentV2),
        )

        # Create documents with different timestamps
        doc1 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc1"),
            version=Version.String("2.0.0"),
            name="first",
            additional_field="field1",
            creation_utc="2023-01-01T10:00:00Z",
        )
        doc2 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc2"),
            version=Version.String("2.0.0"),
            name="second",
            additional_field="field2",
            creation_utc="2023-01-01T11:00:00Z",
        )
        doc3 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc3"),
            version=Version.String("2.0.0"),
            name="third",
            additional_field="field3",
            creation_utc="2023-01-01T12:00:00Z",
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
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that documents can be sorted by creation_utc in descending order (newest first)."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        collection = await db.get_or_create_collection(
            name="test_collection",
            schema=DummyStore.DummyDocumentV2,
            document_loader=identity_loader_for(DummyStore.DummyDocumentV2),
        )

        # Create documents with different timestamps
        doc1 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc1"),
            version=Version.String("2.0.0"),
            name="first",
            additional_field="field1",
            creation_utc="2023-01-01T10:00:00Z",
        )
        doc2 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc2"),
            version=Version.String("2.0.0"),
            name="second",
            additional_field="field2",
            creation_utc="2023-01-01T11:00:00Z",
        )
        doc3 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc3"),
            version=Version.String("2.0.0"),
            name="third",
            additional_field="field3",
            creation_utc="2023-01-01T12:00:00Z",
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
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that cursor-based pagination works correctly with ascending sort."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        collection = await db.get_or_create_collection(
            name="test_collection",
            schema=DummyStore.DummyDocumentV2,
            document_loader=identity_loader_for(DummyStore.DummyDocumentV2),
        )

        # Create documents with different timestamps
        doc1 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc1"),
            version=Version.String("2.0.0"),
            name="first",
            additional_field="field1",
            creation_utc="2023-01-01T10:00:00Z",
        )
        doc2 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc2"),
            version=Version.String("2.0.0"),
            name="second",
            additional_field="field2",
            creation_utc="2023-01-01T11:00:00Z",
        )
        doc3 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc3"),
            version=Version.String("2.0.0"),
            name="third",
            additional_field="field3",
            creation_utc="2023-01-01T12:00:00Z",
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
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that cursor-based pagination works correctly with descending sort."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        collection = await db.get_or_create_collection(
            name="test_collection",
            schema=DummyStore.DummyDocumentV2,
            document_loader=identity_loader_for(DummyStore.DummyDocumentV2),
        )

        # Create documents with different timestamps
        doc1 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc1"),
            version=Version.String("2.0.0"),
            name="first",
            additional_field="field1",
            creation_utc="2023-01-01T10:00:00Z",
        )
        doc2 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc2"),
            version=Version.String("2.0.0"),
            name="second",
            additional_field="field2",
            creation_utc="2023-01-01T11:00:00Z",
        )
        doc3 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc3"),
            version=Version.String("2.0.0"),
            name="third",
            additional_field="field3",
            creation_utc="2023-01-01T12:00:00Z",
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
    new_file: Path,
    logger: Logger,
) -> None:
    """Test that the default sort direction is ascending (oldest first)."""
    async with JSONFileDocumentDatabase(logger, new_file) as db:
        collection = await db.get_or_create_collection(
            name="test_collection",
            schema=DummyStore.DummyDocumentV2,
            document_loader=identity_loader_for(DummyStore.DummyDocumentV2),
        )

        # Create documents with different timestamps
        doc1 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc1"),
            version=Version.String("2.0.0"),
            name="first",
            additional_field="field1",
            creation_utc="2023-01-01T10:00:00Z",
        )
        doc2 = DummyStore.DummyDocumentV2(
            id=ObjectId("doc2"),
            version=Version.String("2.0.0"),
            name="second",
            additional_field="field2",
            creation_utc="2023-01-01T11:00:00Z",
        )

        await collection.insert_one(doc1)
        await collection.insert_one(doc2)

        # Test default sort (should be ascending)
        result = await collection.find({})

        assert len(result.items) == 2
        assert result.items[0]["name"] == "first"  # Older document first (ascending)
        assert result.items[1]["name"] == "second"  # Newer document second
