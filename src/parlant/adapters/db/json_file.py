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
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, cast
from typing_extensions import override, Self
import aiofiles

from parlant.core.persistence.common import (
    Where,
    matches_filters,
    ensure_is_total,
)
from parlant.core.async_utils import ReaderWriterLock
from parlant.core.persistence.document_database import (
    BaseDocument,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
    InsertResult,
    TDocument,
    UpdateResult,
    identity_loader,
)
from parlant.core.loggers import Logger


class JSONFileDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        logger: Logger,
        file_path: Path,
    ) -> None:
        self.file_path = file_path

        self._logger = logger
        self._op_counter = 0

        self._lock = ReaderWriterLock()

        if not self.file_path.exists():
            self.file_path.write_text(json.dumps({}))

        self._raw_data: dict[str, Any] = {}
        self._collections: dict[str, JSONFileDocumentCollection[BaseDocument]] = {}

    async def flush(self) -> None:
        async with self._lock.writer_lock:
            await self._flush_unlocked()

    async def __aenter__(self) -> Self:
        async with self._lock.reader_lock:
            self._raw_data = await self._load_raw_data()

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        async with self._lock.writer_lock:
            await self._flush_unlocked()
        return False

    async def _load_raw_data(
        self,
    ) -> dict[str, Any]:
        # Return an empty JSON object if the file is empty
        if self.file_path.stat().st_size == 0:
            return {}

        async with aiofiles.open(self.file_path, "r", encoding="utf-8") as file:
            return cast(dict[str, Any], json.loads(await file.read()))

    async def _save_data(
        self,
        data: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None:
        async with aiofiles.open(self.file_path, mode="w", encoding="utf-8") as file:
            json_string = json.dumps(
                {
                    **self._raw_data,
                    **data,
                },
                ensure_ascii=False,
                indent=2,
            )
            await file.write(json_string)

    async def load_documents_with_loader(
        self,
        name: str,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
        documents: Sequence[BaseDocument] | None = None,
    ) -> Sequence[TDocument]:
        data: list[TDocument] = []
        failed_migrations: list[BaseDocument] = []

        collection_documents = documents or self._raw_data.get(name, [])

        for doc in collection_documents:
            try:
                if loaded_doc := await document_loader(doc):
                    data.append(loaded_doc)
                else:
                    self._logger.warning(f'Failed to load document "{doc}"')
                    failed_migrations.append(doc)
            except Exception as e:
                self._logger.error(
                    f"Failed to load document '{doc}' with error: {e}. Added to failed migrations collection."
                )
                failed_migrations.append(doc)

        if failed_migrations:
            failed_migrations_collection = await self.get_or_create_collection(
                "failed_migrations", BaseDocument, identity_loader
            )

            for doc in failed_migrations:
                await failed_migrations_collection.insert_one(doc)

        return data

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> JSONFileDocumentCollection[TDocument]:
        self._collections[name] = JSONFileDocumentCollection(
            database=self,
            name=name,
            schema=schema,
        )

        return cast(JSONFileDocumentCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> JSONFileDocumentCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(JSONFileDocumentCollection[TDocument], collection)

        elif name in self._raw_data:
            self._collections[name] = JSONFileDocumentCollection(
                database=self,
                name=name,
                schema=schema,
                data=await self.load_documents_with_loader(name, document_loader),
            )
            return cast(JSONFileDocumentCollection[TDocument], self._collections[name])

        raise ValueError(f'Collection "{name}" does not exists')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> JSONFileDocumentCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(JSONFileDocumentCollection[TDocument], collection)

        elif name in self._raw_data:
            self._collections[name] = JSONFileDocumentCollection(
                database=self,
                name=name,
                schema=schema,
                data=await self.load_documents_with_loader(name, document_loader),
            )
            return cast(JSONFileDocumentCollection[TDocument], self._collections[name])

        self._collections[name] = JSONFileDocumentCollection(
            database=self,
            name=name,
            schema=schema,
            data=await self.load_documents_with_loader(name, document_loader),
        )

        return cast(JSONFileDocumentCollection[TDocument], self._collections[name])

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name in self._collections:
            del self._collections[name]
            return

        raise ValueError(f'Collection "{name}" does not exists')

    async def _flush_unlocked(self) -> None:
        data = {}
        for collection_name in self._collections:
            data[collection_name] = self._collections[collection_name].documents
        await self._save_data(data)


class JSONFileDocumentCollection(DocumentCollection[TDocument]):
    def __init__(
        self,
        database: JSONFileDocumentDatabase,
        name: str,
        schema: type[TDocument],
        data: Sequence[TDocument] | None = None,
    ) -> None:
        self._database = database
        self._name = name
        self._schema = schema
        self._op_counter = 0

        self._lock = ReaderWriterLock()

        self.documents = list(data) if data else []

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        result = []
        async with self._lock.reader_lock:
            for doc in filter(
                lambda d: matches_filters(filters, d),
                self.documents,
            ):
                result.append(doc)

        return result

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            for doc in self.documents:
                if matches_filters(filters, doc):
                    return doc

        return None

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        async with self._lock.writer_lock:
            self.documents.append(document)

        await self._database.flush()

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        async with self._lock.writer_lock:
            for i, d in enumerate(self.documents):
                if matches_filters(filters, d):
                    self.documents[i] = cast(TDocument, {**self.documents[i], **params})

                    await self._database.flush()

                    return UpdateResult(
                        acknowledged=True,
                        matched_count=1,
                        modified_count=1,
                        updated_document=self.documents[i],
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
        async with self._lock.writer_lock:
            for i, d in enumerate(self.documents):
                if matches_filters(filters, d):
                    document = self.documents.pop(i)

                    await self._database.flush()

                    return DeleteResult(
                        deleted_count=1, acknowledged=True, deleted_document=document
                    )

        return DeleteResult(
            acknowledged=True,
            deleted_count=0,
            deleted_document=None,
        )
