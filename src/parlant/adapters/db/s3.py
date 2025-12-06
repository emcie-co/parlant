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

from __future__ import annotations
import json
import asyncio
from typing import Any, Awaitable, Callable, List, Optional, Sequence, cast
from typing_extensions import override, Self
import boto3
from botocore.exceptions import ClientError

from parlant.core.persistence.common import (
    Cursor,
    SortDirection,
    Where,
    matches_filters,
    ensure_is_total,
    ObjectId,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
    FindResult,
    InsertResult,
    TDocument,
    UpdateResult,
    identity_loader,
)
from parlant.core.loggers import Logger


class S3DocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        logger: Logger,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.bucket_name = bucket_name
        self._logger = logger

        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            endpoint_url=endpoint_url,
        )

        self._collections: dict[str, S3DocumentCollection[Any]] = {}

    async def _ensure_bucket_exists(self) -> None:
        def check_and_create() -> None:
            try:
                self._s3_client.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code == "404":
                    self._s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    raise

        await asyncio.to_thread(check_and_create)

    async def __aenter__(self) -> Self:
        await self._ensure_bucket_exists()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> S3DocumentCollection[TDocument]:
        if name not in self._collections:
            self._collections[name] = S3DocumentCollection(
                database=self,
                name=name,
                schema=schema,
                document_loader=identity_loader,
            )
        return cast(S3DocumentCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> S3DocumentCollection[TDocument]:
        # Check if collection "exists" by checking if any objects exist with prefix
        exists = await self._collection_exists(name)
        if not exists and name not in self._collections:
            raise ValueError(f'Collection "{name}" does not exists')

        if name not in self._collections:
            self._collections[name] = S3DocumentCollection(
                database=self,
                name=name,
                schema=schema,
                document_loader=document_loader,
            )
        return cast(S3DocumentCollection[TDocument], self._collections[name])

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> S3DocumentCollection[TDocument]:
        if name not in self._collections:
            self._collections[name] = S3DocumentCollection(
                database=self,
                name=name,
                schema=schema,
                document_loader=document_loader,
            )
        return cast(S3DocumentCollection[TDocument], self._collections[name])

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        await self._delete_prefix(name)
        if name in self._collections:
            del self._collections[name]

    async def _collection_exists(self, name: str) -> bool:
        def check() -> bool:
            response = self._s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=f"{name}/", MaxKeys=1
            )
            return "Contents" in response

        return await asyncio.to_thread(check)

    async def _delete_prefix(self, prefix: str) -> None:
        def delete() -> None:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=f"{prefix}/")

            for page in pages:
                if "Contents" in page:
                    objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                    # Delete in batches of 1000 (S3 limit)
                    for i in range(0, len(objects), 1000):
                        batch = objects[i : i + 1000]
                        self._s3_client.delete_objects(
                            Bucket=self.bucket_name, Delete={"Objects": batch}
                        )

        await asyncio.to_thread(delete)


class S3DocumentCollection(DocumentCollection[TDocument]):
    def __init__(
        self,
        database: S3DocumentDatabase,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        self._database = database
        self._name = name
        self._schema = schema
        self._document_loader = document_loader
        self._s3_client = database._s3_client
        self._bucket = database.bucket_name

    def _get_key(self, doc_id: str) -> str:
        return f"{self._name}/{doc_id}.json"

    async def _get_object_content(self, key: str) -> Optional[TDocument]:
        def fetch() -> Optional[dict[str, Any]]:
            try:
                response = self._s3_client.get_object(Bucket=self._bucket, Key=key)
                return json.loads(response["Body"].read().decode("utf-8"))
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    return None
                raise

        raw_doc = await asyncio.to_thread(fetch)
        if raw_doc is None:
            return None

        # Apply document loader (migration)
        try:
            loaded_doc = await self._document_loader(cast(BaseDocument, raw_doc))
            if loaded_doc:
                return loaded_doc
            
            # Migration failed (loader returned None)
            self._database._logger.warning(f'Failed to load document "{raw_doc}"')
            await self._handle_failed_migration(cast(BaseDocument, raw_doc))
            return None
        except Exception as e:
            self._database._logger.error(f"Failed to load document '{raw_doc}': {e}")
            await self._handle_failed_migration(cast(BaseDocument, raw_doc))
            return None

    async def _handle_failed_migration(self, doc: BaseDocument) -> None:
        # Save to failed_migrations collection
        # We use a direct S3 write to avoid recursion or complex dependency
        failed_collection_name = f"{self._database.bucket_name}_failed_migrations"
        # Actually, standard pattern is usually "failed_migrations" collection in the same DB
        # But to match other adapters, we should probably use a standard name.
        # Mongo adapter uses: f"{self.database_name}_{name}_failed_migrations" ?
        # No, it uses "failed_migrations" usually or specific name.
        # Let's check the test expectation.
        # Test expects: db.get_collection("failed_migrations", ...)
        
        # We can just write to "failed_migrations/{id}.json"
        doc_id = doc.get("id")
        if doc_id:
            key = f"failed_migrations/{doc_id}.json"
            def put() -> None:
                self._s3_client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=json.dumps(doc, ensure_ascii=False).encode("utf-8"),
                )
            await asyncio.to_thread(put)

    async def _put_object_content(self, key: str, content: dict[str, Any]) -> None:
        def put() -> None:
            self._s3_client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(content, ensure_ascii=False).encode("utf-8"),
            )

        await asyncio.to_thread(put)

    async def _delete_object(self, key: str) -> None:
        def delete() -> None:
            self._s3_client.delete_object(Bucket=self._bucket, Key=key)

        await asyncio.to_thread(delete)

    async def _list_keys(self) -> List[str]:
        def list_objs() -> List[str]:
            keys = []
            paginator = self._s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=f"{self._name}/"):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        keys.append(obj["Key"])
            return keys

        return await asyncio.to_thread(list_objs)

    @override
    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        # 1. List all keys in the collection
        keys = await self._list_keys()

        # 2. Fetch all documents (in parallel)
        # This applies the loader, so documents are migrated
        tasks = [self._get_object_content(key) for key in keys]
        docs_or_none = await asyncio.gather(*tasks)
        documents = [doc for doc in docs_or_none if doc is not None]

        # 3. Apply filtering
        filtered_docs = [doc for doc in documents if matches_filters(filters, doc)]

        # 4. Apply sorting
        sort_direction = sort_direction or SortDirection.ASC
        filtered_docs = self._apply_sort(filtered_docs, sort_direction)

        # 5. Apply cursor
        if cursor:
            filtered_docs = self._apply_cursor_filter(filtered_docs, cursor, sort_direction)

        total_count = len(filtered_docs)

        # 6. Apply limit
        has_more = False
        next_cursor = None

        if limit is not None and len(filtered_docs) > limit:
            has_more = True
            result_docs = filtered_docs[:limit]
            if result_docs:
                last_doc = result_docs[-1]
                next_cursor = Cursor(
                    creation_utc=str(last_doc.get("creation_utc", "")),
                    id=ObjectId(str(last_doc.get("id", ""))),
                )
        else:
            result_docs = filtered_docs

        return FindResult(
            items=result_docs,
            total_count=total_count,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    def _apply_sort(
        self, documents: list[TDocument], sort_direction: SortDirection
    ) -> list[TDocument]:
        docs = list(documents)
        reverse_order = sort_direction == SortDirection.DESC
        docs.sort(
            key=lambda d: (
                d.get("creation_utc") or "",
                d.get("id") or "",
            ),
            reverse=reverse_order,
        )
        return docs

    def _apply_cursor_filter(
        self, documents: list[TDocument], cursor: Cursor, sort_direction: SortDirection
    ) -> list[TDocument]:
        result = []
        for doc in documents:
            doc_creation_utc = str(doc.get("creation_utc", ""))
            doc_id = str(doc.get("id", ""))
            if sort_direction == SortDirection.DESC:
                if doc_creation_utc < cursor.creation_utc or (
                    doc_creation_utc == cursor.creation_utc and doc_id < cursor.id
                ):
                    result.append(doc)
            else:
                if doc_creation_utc > cursor.creation_utc or (
                    doc_creation_utc == cursor.creation_utc and doc_id > cursor.id
                ):
                    result.append(doc)
        return result

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        # Optimization: check if 'id' is in filters with equality
        doc_id = None
        if "id" in filters and isinstance(filters["id"], dict) and "$eq" in filters["id"]:
            doc_id = filters["id"]["$eq"]

        if doc_id:
            key = self._get_key(str(doc_id))
            doc = await self._get_object_content(key)
            if doc and matches_filters(filters, doc):
                return doc
            return None

        # Fallback to scan
        keys = await self._list_keys()
        tasks = [self._get_object_content(key) for key in keys]
        docs_or_none = await asyncio.gather(*tasks)
        documents = [doc for doc in docs_or_none if doc is not None]

        for doc in documents:
            if matches_filters(filters, doc):
                return doc
        return None

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        doc_id = document.get("id")
        if not doc_id:
            raise ValueError("Document must have an 'id' field")

        key = self._get_key(str(doc_id))
        await self._put_object_content(key, document)

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        existing_doc = await self.find_one(filters)

        if existing_doc:
            updated_doc = cast(TDocument, {**existing_doc, **params})
            doc_id = updated_doc.get("id")
            key = self._get_key(str(doc_id))
            await self._put_object_content(key, updated_doc)

            return UpdateResult(
                acknowledged=True,
                matched_count=1,
                modified_count=1,
                updated_document=updated_doc,
            )

        if upsert:
            await self.insert_one(params)
            return UpdateResult(
                acknowledged=True,
                matched_count=0,
                modified_count=0,
                updated_document=params,
            )

        return UpdateResult(
            acknowledged=True,
            matched_count=0,
            modified_count=0,
            updated_document=None,
        )

    @override
    async def delete_one(
        self,
        filters: Where,
    ) -> DeleteResult[TDocument]:
        existing_doc = await self.find_one(filters)
        if existing_doc:
            doc_id = existing_doc.get("id")
            key = self._get_key(str(doc_id))
            await self._delete_object(key)
            return DeleteResult(
                deleted_count=1,
                acknowledged=True,
                deleted_document=existing_doc,
            )

        return DeleteResult(
            acknowledged=True,
            deleted_count=0,
            deleted_document=None,
        )
