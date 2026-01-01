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

from typing import Any, Awaitable, Callable, Optional
from bson import CodecOptions
from typing_extensions import Self
from parlant.core.loggers import Logger
from parlant.core.persistence.common import Cursor, SortDirection, Where, ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
    FindResult,
    InsertResult,
    TDocument,
    UpdateResult,
)
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.asynchronous.collection import AsyncCollection


class MongoDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        mongo_client: AsyncMongoClient[Any],
        database_name: str,
        logger: Logger,
    ):
        self.mongo_client: AsyncMongoClient[Any] = mongo_client
        self.database_name = database_name

        self._logger = logger

        self._database: Optional[AsyncDatabase[Any]] = None
        self._collections: dict[str, MongoDocumentCollection[Any]] = {}

    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> DocumentCollection[TDocument]:
        if self._database is None:
            raise Exception("underlying database missing.")

        collection = await self._database.create_collection(
            name=name,
            codec_options=CodecOptions(document_class=schema),
        )

        # Create index on creation_utc field
        await collection.create_index([("creation_utc", 1)])

        self._collections[name] = MongoDocumentCollection(self, collection)
        return self._collections[name]

    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[TDocument | None]],
    ) -> DocumentCollection[TDocument]:
        if self._database is None:
            raise Exception("underlying database missing.")

        result_collection = self._database.get_collection(
            name=name,
            codec_options=CodecOptions(document_class=schema),
        )

        failed_migrations_collection_name = f"{self.database_name}_{name}_failed_migrations"
        collection_existing_documents = result_collection.find({})
        if failed_migrations_collection_name in await self._database.list_collection_names():
            self._logger.info(f"deleting old `{failed_migrations_collection_name}` collection")
            await self.delete_collection(failed_migrations_collection_name)

        failed_migration_collection: Optional[DocumentCollection[TDocument]] = None
        for doc in await collection_existing_documents.to_list():
            try:
                if loaded_doc := await document_loader(doc):
                    await result_collection.replace_one(doc, loaded_doc)
                    continue

                if failed_migration_collection is None:
                    self._logger.warning(
                        f"creating: `{failed_migrations_collection_name}` collection to store failed migrations..."
                    )
                    failed_migration_collection = await self.create_collection(
                        failed_migrations_collection_name, schema
                    )

                self._logger.warning(f'failed to load document "{doc}"')
                await failed_migration_collection.insert_one(doc)
                await result_collection.delete_one(doc)
            except Exception as e:
                if failed_migration_collection is None:
                    self._logger.warning(
                        f"creating: `{failed_migrations_collection_name}` collection to store failed migrations..."
                    )
                    failed_migration_collection = await self.create_collection(
                        failed_migrations_collection_name, schema
                    )

                self._logger.error(
                    f"failed to load document '{doc}' with error: {e}. Added to `{failed_migrations_collection_name}` collection."
                )
                await failed_migration_collection.insert_one(doc)

        # Create index on creation_utc field
        await result_collection.create_index([("creation_utc", 1)])

        self._collections[name] = MongoDocumentCollection(self, result_collection)
        return self._collections[name]

    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[TDocument | None]],
    ) -> DocumentCollection[TDocument]:
        return await self.get_collection(name, schema, document_loader)

    async def delete_collection(self, name: str) -> None:
        if self._database is None:
            raise Exception("underlying database missing.")

        await self._database.drop_collection(name)

    async def __aenter__(self) -> Self:
        self._database = self.mongo_client[self.database_name]
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        if self._database is not None:
            self._database = None

        return False


class MongoDocumentCollection(DocumentCollection[TDocument]):
    def __init__(
        self,
        mongo_document_database: MongoDocumentDatabase,
        mongo_collection: AsyncCollection[TDocument],
    ) -> None:
        self._database = mongo_document_database
        self._collection = mongo_collection

    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        query = dict(filters) if filters else {}
        sort_direction = sort_direction or SortDirection.ASC

        if cursor is not None:
            if sort_direction == SortDirection.DESC:
                cursor_conditions = [
                    {"creation_utc": {"$lt": cursor.creation_utc}},
                    {
                        "$and": [
                            {"creation_utc": cursor.creation_utc},
                            {"_id": {"$lt": cursor.id}},
                        ]
                    },
                ]
            else:
                cursor_conditions = [
                    {"creation_utc": {"$gt": cursor.creation_utc}},
                    {
                        "$and": [
                            {"creation_utc": cursor.creation_utc},
                            {"_id": {"$gt": cursor.id}},
                        ]
                    },
                ]
            query["$or"] = cursor_conditions

        # Sort by creation_utc with _id as tiebreaker according to sort_direction
        sort_order = -1 if sort_direction == SortDirection.DESC else 1
        sort_spec = [("creation_utc", sort_order), ("_id", sort_order)]

        # Get one extra document to check if there are more
        query_limit = (limit + 1) if limit else None

        mongo_cursor = self._collection.find(query).sort(sort_spec)
        if query_limit:
            mongo_cursor = mongo_cursor.limit(query_limit)

        items = await mongo_cursor.to_list(length=query_limit)

        # Calculate pagination metadata
        has_more = False
        next_cursor = None
        total_count = len(items)

        if limit and len(items) > limit:
            has_more = True
            items = items[:limit]  # Remove the extra item

            # Create cursor from the last item
            if items:
                last_item = items[-1]
                next_cursor = Cursor(
                    creation_utc=str(last_item.get("creation_utc", "")),
                    id=ObjectId(str(last_item.get("_id", last_item.get("id", "")))),
                )

        return FindResult(
            items=items, total_count=total_count, has_more=has_more, next_cursor=next_cursor
        )

    async def find_one(self, filters: Where) -> TDocument | None:
        result = await self._collection.find_one(filters)
        return result

    async def insert_one(self, document: TDocument) -> InsertResult:
        insert_result = await self._collection.insert_one(document)
        return InsertResult(acknowledged=insert_result.acknowledged)

    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        update_result = await self._collection.update_one(filters, {"$set": params}, upsert)
        result_document = await self._collection.find_one(filters)
        return UpdateResult[TDocument](
            update_result.acknowledged,
            update_result.matched_count,
            update_result.modified_count,
            result_document,
        )

    async def delete_one(self, filters: Where) -> DeleteResult[TDocument]:
        result_document = await self._collection.find_one(filters)
        if result_document is None:
            return DeleteResult(True, 0, None)

        delete_result = await self._collection.delete_one(filters)
        return DeleteResult(
            delete_result.acknowledged,
            deleted_count=delete_result.deleted_count,
            deleted_document=result_document,
        )
