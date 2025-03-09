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

from typing import Any, Awaitable, Callable, Optional, Sequence
from bson import CodecOptions
from typing_extensions import Self
from parlant.core.loggers import Logger
from parlant.core.persistence.common import Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
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
        mongo_client_async: AsyncMongoClient[Any],
        database_name: str,
        logger: Logger,
    ):
        self.mongo_client: AsyncMongoClient[Any] = mongo_client_async
        self.database_name = database_name

        self._logger = logger

        self._database: Optional[AsyncDatabase[Any]] = None
        self._collections: dict[str, MongoDocumentCollection[Any]] = {}

    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> DocumentCollection[TDocument]:
        self._logger.debug(f'create collection "{name}"')
        if self._database is None:
            raise Exception("underlying database missing.")

        self._collections[name] = MongoDocumentCollection(
            self,
            await self._database.create_collection(
                name=name,
                codec_options=CodecOptions(document_class=schema),
            ),
        )
        return self._collections[name]

    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[TDocument | None]],
    ) -> DocumentCollection[TDocument]:
        self._logger.debug(f'get collection "{name}"')
        if self._database is None:
            raise Exception("underlying database missing.")

        result_collection = self._database.get_collection(
            name=name,
            codec_options=CodecOptions(document_class=schema),
        )

        collection_existing_documents = result_collection.find({})
        failed_migrations: list[TDocument] = []
        for doc in await collection_existing_documents.to_list():
            try:
                if loaded_doc := await document_loader(doc):
                    await result_collection.replace_one(doc, loaded_doc)
                else:
                    self._logger.warning(f'Failed to load document "{doc}"')
                    failed_migrations.append(doc)
                    await result_collection.delete_one(doc)
            except Exception as e:
                self._logger.error(
                    f"Failed to load document '{doc}' with error: {e}. Added to failed migrations collection."
                )
                failed_migrations.append(doc)

        if len(failed_migrations):
            failed_migration_collection = await self.create_collection("failed_migrations", schema)
            for doc in failed_migrations:
                await failed_migration_collection.insert_one(doc)

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

    async def find(self, filters: Where) -> Sequence[TDocument]:
        mongo_cursor = self._collection.find(filters)
        result = await mongo_cursor.to_list()
        await mongo_cursor.close()
        return result

    async def find_one(self, filters: Where) -> TDocument | None:
        result = await self._collection.find_one(filters)
        return result

    async def insert_one(self, document: TDocument) -> InsertResult:
        await self._collection.insert_one(document)
        return InsertResult(acknowledged=True)

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
