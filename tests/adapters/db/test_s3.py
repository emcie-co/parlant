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

import json
from typing import AsyncIterator, Optional, TypedDict, cast
from typing_extensions import Self
from pytest import fixture, raises
from moto import mock_aws
import boto3

from parlant.core.common import Version
from parlant.adapters.db.s3 import S3DocumentDatabase
from parlant.core.persistence.common import Cursor, MigrationRequired, ObjectId, SortDirection
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    FindResult,
    identity_loader,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.loggers import Logger


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


@fixture
async def s3_bucket(logger: Logger) -> AsyncIterator[str]:
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket_name = "test-bucket"
        s3.create_bucket(Bucket=bucket_name)
        yield bucket_name


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

    def __init__(self, database: S3DocumentDatabase, allow_migration: bool = True):
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


async def test_that_dummy_documents_can_be_created(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that dummy documents can be created with all required fields."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            doc = await store.create_dummy("test_doc", "test_value")

            assert doc["name"] == "test_doc"
            assert doc["additional_field"] == "test_value"
            assert doc["version"] == "2.0.0"
            assert doc["id"] == "dummy_test_doc"


async def test_that_data_persists_to_s3(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that data is actually written to S3 as individual objects."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            await store.create_dummy("persist_test", "persist_value")

    # Verify directly via boto3
    s3 = boto3.client("s3", region_name="us-east-1")
    # Key should be collection_name/document_id.json
    key = "dummy_collection/dummy_persist_test.json"
    response = s3.get_object(Bucket=s3_bucket, Key=key)
    content = json.loads(response["Body"].read().decode("utf-8"))
    
    assert content["name"] == "persist_test"
    assert content["id"] == "dummy_persist_test"


async def test_that_data_loads_from_s3(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that data is loaded from S3 on startup (via find/list)."""
    # Pre-populate S3
    s3 = boto3.client("s3", region_name="us-east-1")
    doc = {
        "id": "dummy_loaded",
        "version": "2.0.0",
        "name": "loaded_doc",
        "additional_field": "loaded_value",
        "creation_utc": "2023-01-01T00:00:00Z",
    }
    s3.put_object(
        Bucket=s3_bucket, 
        Key="dummy_collection/dummy_loaded.json", 
        Body=json.dumps(doc).encode("utf-8")
    )

    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            result = await store.list_dummy()
            assert len(result.items) == 1
            assert result.items[0]["name"] == "loaded_doc"


async def test_that_bucket_is_created_if_missing(
    logger: Logger,
) -> None:
    """Test that the bucket is created if it doesn't exist."""
    with mock_aws():
        bucket_name = "new-bucket"
        s3 = boto3.client("s3", region_name="us-east-1")
        
        async with S3DocumentDatabase(logger, bucket_name, region_name="us-east-1") as db:
            pass
            
        # Verify bucket exists
        s3.head_bucket(Bucket=bucket_name)


async def test_that_dummy_documents_can_be_retrieved_by_id(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that dummy documents can be retrieved by ID."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            created_doc = await store.create_dummy("read_test", "read_value")
            
            retrieved_doc = await store.read_dummy(created_doc["id"])
            
            assert retrieved_doc is not None
            assert retrieved_doc["name"] == "read_test"
            assert retrieved_doc["id"] == created_doc["id"]


async def test_that_dummy_documents_can_be_updated(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that dummy documents can be updated."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            original_doc = await store.create_dummy("original_name", "original_value")
            
            updated_doc = await store.update_dummy(original_doc["id"], "updated_name")
            
            assert updated_doc is not None
            assert updated_doc["name"] == "updated_name"
            
            # Verify persistence
            retrieved_doc = await store.read_dummy(original_doc["id"])
            assert retrieved_doc is not None
            assert retrieved_doc["name"] == "updated_name"


async def test_that_dummy_documents_can_be_deleted(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that dummy documents can be deleted."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            doc = await store.create_dummy("delete_test", "delete_value")
            
            # Verify exists
            assert await store.read_dummy(doc["id"]) is not None
            
            # Delete
            result = await store.delete_dummy(doc["id"])
            assert result is True
            
            # Verify gone
            assert await store.read_dummy(doc["id"]) is None
            
            # Verify gone from S3
            s3 = boto3.client("s3", region_name="us-east-1")
            with raises(s3.exceptions.NoSuchKey):
                s3.get_object(Bucket=s3_bucket, Key=f"dummy_collection/{doc['id']}.json")


async def test_that_dummy_documents_can_be_listed_with_pagination_limit(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that dummy documents can be listed with a limit."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            for i in range(5):
                await store.create_dummy(f"doc{i}", f"value{i}")
                
            result = await store.list_dummy(limit=3)
            
            assert len(result.items) == 3
            assert result.total_count == 5
            assert result.has_more
            assert result.next_cursor is not None


async def test_that_dummy_documents_are_sorted_by_creation_time_descending(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that dummy documents are sorted by creation_utc descending."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            import asyncio
            await store.create_dummy("first", "field1")
            await asyncio.sleep(0.01)
            await store.create_dummy("second", "field2")
            await asyncio.sleep(0.01)
            await store.create_dummy("third", "field3")
            
            result = await store.list_dummy(sort_direction=SortDirection.DESC)
            
            assert len(result.items) == 3
            assert result.items[0]["name"] == "third"
            assert result.items[1]["name"] == "second"
            assert result.items[2]["name"] == "first"


async def test_that_dummy_documents_can_be_paginated_using_cursor(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that dummy documents can be paginated using cursor."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            import asyncio
            doc1 = await store.create_dummy("first", "field1")
            await asyncio.sleep(0.01)
            await store.create_dummy("second", "field2")
            await asyncio.sleep(0.01)
            await store.create_dummy("third", "field3")
            
            cursor = Cursor(creation_utc=doc1["creation_utc"], id=doc1["id"])
            
            result = await store.list_dummy(cursor=cursor)
            
            assert len(result.items) == 2
            assert result.items[0]["name"] == "second"
            assert result.items[1]["name"] == "third"


async def test_that_document_upgrade_happens_during_loading_of_store(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that documents are upgraded during loading."""
    s3 = boto3.client("s3", region_name="us-east-1")
    doc = {
        "id": "dummy_id",
        "version": "1.0.0",
        "name": "Test Document",
    }
    s3.put_object(
        Bucket=s3_bucket, 
        Key="dummy_collection/dummy_id.json", 
        Body=json.dumps(doc).encode("utf-8")
    )
    
    # Also need metadata
    meta = {"id": "meta_id", "version": "1.0.0"}
    s3.put_object(
        Bucket=s3_bucket,
        Key="metadata/meta_id.json",
        Body=json.dumps(meta).encode("utf-8")
    )
    
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db, allow_migration=True) as store:
            result = await store.list_dummy()
            
            assert result.total_count == 1
            upgraded_doc = result.items[0]
            assert upgraded_doc["version"] == "2.0.0"
            assert upgraded_doc["name"] == "Test Document"
            assert upgraded_doc["additional_field"] == "default_value"


async def test_that_failed_migrations_are_tracked_in_separate_collection(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that failed migrations are tracked."""
    s3 = boto3.client("s3", region_name="us-east-1")
    doc = {
        "id": "invalid_dummy_id",
        "version": "3.0",
        "name": "Unmigratable Document",
        "creation_utc": "2023-01-01T00:00:00Z",
    }
    s3.put_object(
        Bucket=s3_bucket, 
        Key="dummy_collection/invalid_dummy_id.json", 
        Body=json.dumps(doc).encode("utf-8")
    )
    
    # Metadata
    meta = {"id": "meta_id", "version": "1.0.0"}
    s3.put_object(
        Bucket=s3_bucket,
        Key="metadata/meta_id.json",
        Body=json.dumps(meta).encode("utf-8")
    )
    
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db, allow_migration=True) as store:
            result = await store.list_dummy()
            assert result.total_count == 0
            
            failed_collection = await db.get_collection(
                "failed_migrations", BaseDocument, identity_loader
            )
            failed_docs = await failed_collection.find({})
            assert failed_docs.total_count == 1
            assert failed_docs.items[0]["id"] == "invalid_dummy_id"


async def test_that_version_mismatch_raises_error_when_migration_is_required_but_disabled(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that version mismatch raises error when migration is disabled."""
    s3 = boto3.client("s3", region_name="us-east-1")
    doc = {
        "id": "dummy_id",
        "version": "1.0.0",
        "name": "Test Document",
    }
    s3.put_object(
        Bucket=s3_bucket, 
        Key="dummy_collection/dummy_id.json", 
        Body=json.dumps(doc).encode("utf-8")
    )
    
    # Metadata
    meta = {"id": "meta_id", "version": "1.0.0"}
    s3.put_object(
        Bucket=s3_bucket,
        Key="metadata/meta_id.json",
        Body=json.dumps(meta).encode("utf-8")
    )
    
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        with raises(MigrationRequired) as exc_info:
            async with DummyStore(db, allow_migration=False) as _:
                pass
        
        assert "Migration required for DummyStore." in str(exc_info.value)


async def test_that_collections_can_be_deleted(
    s3_bucket: str,
    logger: Logger,
) -> None:
    """Test that collections can be deleted."""
    async with S3DocumentDatabase(logger, s3_bucket, region_name="us-east-1") as db:
        async with DummyStore(db) as store:
            await store.create_dummy("test_doc")
            
        await db.delete_collection("dummy_collection")
        
        # Verify objects are gone
        s3 = boto3.client("s3", region_name="us-east-1")
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix="dummy_collection/")
        assert "Contents" not in response


async def test_that_s3_can_be_initialized_with_explicit_credentials(
    logger: Logger,
) -> None:
    """Test that S3 can be initialized with explicit credentials."""
    with mock_aws():
        # Create a bucket in a specific region
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket="creds-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )

        # Initialize DB with explicit credentials
        async with S3DocumentDatabase(
            logger,
            "creds-bucket",
            aws_access_key_id="fake_key",
            aws_secret_access_key="fake_secret",
            region_name="us-west-2",
        ) as db:
            # Verify we can perform operations
            async with DummyStore(db) as store:
                await store.create_dummy("creds_test")
                
            # Verify object exists in the correct region/bucket
            s3_chk = boto3.client(
                "s3",
                region_name="us-west-2",
                aws_access_key_id="fake_key",
                aws_secret_access_key="fake_secret"
            )
            response = s3_chk.get_object(
                Bucket="creds-bucket", 
                Key="dummy_collection/dummy_creds_test.json"
            )
            assert response["Body"] is not None
