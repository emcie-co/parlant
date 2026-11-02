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
import asyncio
import json
from copy import deepcopy
from typing import Any, Awaitable, Callable, Optional, Sequence, cast
from typing_extensions import Self, override
import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from parlant.core.persistence.common import (
    Cursor,
    SortDirection,
    Where,
    matches_filters,
    ensure_is_total,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    CollectionIndex,
    CollectionSort,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
    FindResult,
    InsertResult,
    TDocument,
    UpdateResult,
)
from parlant.core.loggers import Logger


def _conflict(error: ClientError) -> bool:
    return error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in (409, 412)


class S3DocumentDatabase(DocumentDatabase):
    """Document-per-object storage. General queries scan the collection; no secondary indexes.

    The bucket/endpoint must support S3 conditional PUT and DELETE requests. Credentials
    default to boto3's standard credential chain. Use a dedicated bucket for this database.
    """

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
        self._s3_client: Any = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            endpoint_url=endpoint_url,
        )
        self._collections: dict[str, S3DocumentCollection[Any]] = {}
        self._read_slots = asyncio.Semaphore(16)

    async def __aenter__(self) -> Self:
        def check_and_create() -> None:
            try:
                self._s3_client.head_bucket(Bucket=self.bucket_name)
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket"):
                    raise
                region = self._s3_client.meta.region_name
                configuration = (
                    {"CreateBucketConfiguration": {"LocationConstraint": region}}
                    if region and region != "us-east-1"
                    else {}
                )
                self._s3_client.create_bucket(Bucket=self.bucket_name, **configuration)

        try:
            await asyncio.to_thread(check_and_create)
        except BaseException:
            self._s3_client.close()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        await asyncio.to_thread(self._s3_client.close)
        return False

    async def _read(self, key: str) -> tuple[BaseDocument, str] | None:
        def fetch() -> tuple[BaseDocument, str] | None:
            try:
                response = self._s3_client.get_object(Bucket=self.bucket_name, Key=key)
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                    return None
                raise
            body = response["Body"]
            try:
                value = json.loads(body.read())
            finally:
                body.close()
            if not isinstance(value, dict):
                raise ValueError("S3 document must contain a JSON object")
            return cast(BaseDocument, value), str(response["ETag"])

        async with self._read_slots:
            return await asyncio.to_thread(fetch)

    async def _put(self, key: str, document: BaseDocument, etag: str | None = None) -> None:
        condition = {"IfMatch": etag} if etag is not None else {"IfNoneMatch": "*"}
        await asyncio.to_thread(
            self._s3_client.put_object,
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(document, ensure_ascii=False).encode("utf-8"),
            **condition,
        )

    async def _delete(self, key: str, etag: str) -> None:
        await asyncio.to_thread(
            self._s3_client.delete_object,
            Bucket=self.bucket_name,
            Key=key,
            IfMatch=etag,
        )

    async def _list_keys(self, name: str) -> list[str]:
        def list_keys() -> list[str]:
            return [
                str(item["Key"])
                for page in self._s3_client.get_paginator("list_objects_v2").paginate(
                    Bucket=self.bucket_name, Prefix=f"{name}/"
                )
                for item in page.get("Contents", [])
                if str(item["Key"]).endswith(".json")
            ]

        return await asyncio.to_thread(list_keys)

    async def _collection_exists(self, name: str) -> bool:
        def check() -> bool:
            result = self._s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{name}/",
                MaxKeys=1,
            )
            return bool(result.get("Contents"))

        return await asyncio.to_thread(check)

    async def _mark_collection(self, name: str) -> None:
        try:
            await self._put(f"{name}/.collection", {})
        except ClientError as error:
            if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                raise

    @override
    async def create_collection(
        self, name: str, schema: type[TDocument]
    ) -> S3DocumentCollection[TDocument]:
        if not name or name.startswith("/") or ".." in name.split("/"):
            raise ValueError(
                "Provide a non-empty S3 collection name without parent path components"
            )
        await self._mark_collection(name)
        if name not in self._collections:
            self._collections[name] = S3DocumentCollection(self, name, schema)
        return cast(S3DocumentCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> S3DocumentCollection[TDocument]:
        if name not in self._collections:
            if not await self._collection_exists(name):
                raise ValueError(f'Collection "{name}" does not exist')
            await self._migrate_collection_documents(name, document_loader)
            return await self.create_collection(name, schema)
        return cast(S3DocumentCollection[TDocument], self._collections[name])

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> S3DocumentCollection[TDocument]:
        if name not in self._collections:
            if await self._collection_exists(name):
                await self._migrate_collection_documents(name, document_loader)
            return await self.create_collection(name, schema)
        return cast(S3DocumentCollection[TDocument], self._collections[name])

    @override
    async def delete_collection(self, name: str) -> None:
        def delete() -> None:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=f"{name}/"):
                objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                if objects:
                    result = self._s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={"Objects": objects},
                    )
                    if result.get("Errors"):
                        raise RuntimeError("S3 could not delete every document in the collection")

        await asyncio.to_thread(delete)
        self._collections.pop(name, None)

    async def _migrate_collection_documents(
        self,
        name: str,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        for key in await self._list_keys(name):
            entry = await self._read(key)
            if entry is None:
                continue
            original, etag = entry
            try:
                loaded = await document_loader(deepcopy(original))
            except Exception:
                # Only conversion failures are quarantined. Storage errors propagate
                # outside this block and never cause the original to be removed.
                loaded = None
            if loaded is not None:
                if loaded != original:
                    await self._put(key, loaded, etag)
            else:
                quarantine = f"failed_migrations/{name}/{original['id']}/{etag.strip(chr(34))}.json"
                try:
                    await self._put(quarantine, original)
                except ClientError as error:
                    if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                        raise
                await self._delete(key, etag)
                self._logger.warning(
                    f"Quarantined an unmigratable S3 document in collection '{name}'"
                )


class S3DocumentCollection(DocumentCollection[TDocument]):
    def __init__(self, database: S3DocumentDatabase, name: str, schema: type[TDocument]) -> None:
        self._database = database
        self._name = name
        self._schema = schema

    def _get_key(self, document_id: str) -> str:
        return f"{self._name}/{document_id}.json"

    async def _entries(self, filters: Where) -> list[tuple[TDocument, str]]:
        condition = filters.get("id")
        if isinstance(condition, dict) and isinstance(condition.get("$eq"), str):
            keys = [self._get_key(str(condition["$eq"]))]
        else:
            keys = await self._database._list_keys(self._name)
        entries: list[tuple[TDocument, str]] = []
        # Bound both scheduled tasks and concurrent requests, even for large collections.
        for start in range(0, len(keys), 16):
            batch = await asyncio.gather(
                *(self._database._read(key) for key in keys[start : start + 16])
            )
            entries.extend(
                (cast(TDocument, entry[0]), entry[1])
                for entry in batch
                if entry is not None and matches_filters(filters, entry[0])
            )
        return entries

    @override
    async def ensure_indexes(self, indexes: Sequence[CollectionIndex]) -> None:
        # Non-unique indexes are query hints, as with the JSON/transient adapters.
        if any(index.unique for index in indexes):
            raise NotImplementedError(
                "S3 document collections do not support unique secondary indexes"
            )

    @override
    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        documents = [doc for doc, _ in await self._entries(filters)]
        descending = sort_direction == SortDirection.DESC
        documents.sort(
            key=lambda d: (str(d.get("creation_utc", "")), str(d["id"])), reverse=descending
        )
        total_count = len(documents)
        if cursor:
            position = (cursor.sort_value, str(cursor.id))
            documents = [
                d
                for d in documents
                if (
                    (str(d.get("creation_utc", "")), str(d["id"])) < position
                    if descending
                    else (str(d.get("creation_utc", "")), str(d["id"])) > position
                )
            ]
        items = documents[:limit] if limit is not None else documents
        more = limit is not None and len(documents) > limit
        next_cursor = (
            Cursor(sort_value=str(items[-1].get("creation_utc", "")), id=items[-1]["id"])
            if more and items
            else None
        )
        return FindResult(
            items=items, total_count=total_count, has_more=more, next_cursor=next_cursor
        )

    @override
    async def find_one(
        self, filters: Where, sort: Optional[CollectionSort] = None
    ) -> Optional[TDocument]:
        entries = await self._entries(filters)
        for field, direction in reversed(sort or []):
            entries.sort(
                key=lambda e: cast(Any, e[0].get(field)), reverse=direction == SortDirection.DESC
            )
        return entries[0][0] if entries else None

    @override
    async def insert_one(self, document: TDocument) -> InsertResult:
        ensure_is_total(document, self._schema)
        if not document.get("id"):
            raise ValueError("Document must have an 'id' field")
        await self._database._put(self._get_key(document["id"]), document)
        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self, filters: Where, params: TDocument, upsert: bool = False
    ) -> UpdateResult[TDocument]:
        for _ in range(8):
            entries = await self._entries(filters)
            if not entries:
                if upsert:
                    await self.insert_one(params)
                    return UpdateResult(True, 0, 0, params)
                return UpdateResult(True, 0, 0, None)
            original, etag = entries[0]
            if "id" in params and params["id"] != original["id"]:
                raise ValueError("A document's ID cannot be changed")
            updated = cast(TDocument, {**original, **params})
            try:
                await self._database._put(self._get_key(original["id"]), updated, etag)
                return UpdateResult(True, 1, int(updated != original), updated)
            except ClientError as error:
                if not _conflict(error):
                    raise
        raise RuntimeError("S3 document changed concurrently; retry the update")

    @override
    async def delete_one(self, filters: Where) -> DeleteResult[TDocument]:
        for _ in range(8):
            entries = await self._entries(filters)
            if not entries:
                return DeleteResult(True, 0, None)
            document, etag = entries[0]
            try:
                await self._database._delete(self._get_key(document["id"]), etag)
                return DeleteResult(True, 1, document)
            except ClientError as error:
                if not _conflict(error):
                    raise
        raise RuntimeError("S3 document changed concurrently; retry the deletion")
