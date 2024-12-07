# Copyright 2024 Emcie Co Ltd.
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
import importlib
import json
import operator
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast
from typing_extensions import override
import aiofiles

from parlant.core.persistence.document_database import (
    BaseDocument,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
    InsertResult,
    TDocument,
    UpdateResult,
    Where,
    ensure_is_total,
    matches_filters,
)
from parlant.core.logging import Logger


class JSONFileDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        logger: Logger,
        file_path: Path,
    ) -> None:
        self.file_path = file_path

        self._logger = logger
        self._lock = asyncio.Lock()
        self._op_counter = 0

        if not self.file_path.exists():
            self.file_path.write_text(json.dumps({}))
        self._collections: dict[str, JSONFileDocumentCollection[BaseDocument]]

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_unlocked()

    async def __aenter__(self) -> JSONFileDocumentDatabase:
        async with self._lock:
            raw_data = await self._load_data()

        schemas: dict[str, Any] = raw_data.get("__schemas__", {})
        self._collections = (
            {
                c_name: JSONFileDocumentCollection(
                    database=self,
                    name=c_name,
                    schema=operator.attrgetter(c_schema["model_path"])(
                        importlib.import_module(c_schema["module_path"])
                    ),
                    data=raw_data[c_name],
                )
                for c_name, c_schema in schemas.items()
            }
            if raw_data
            else {}
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        async with self._lock:
            await self._flush_unlocked()
        return False

    async def _load_data(
        self,
    ) -> dict[str, Any]:
        # Return an empty JSON object if the file is empty
        if self.file_path.stat().st_size == 0:
            return {}

        async with aiofiles.open(self.file_path, "r") as file:
            data: dict[str, Any] = json.loads(await file.read())
            return data

    async def _save_data(
        self,
        data: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None:
        async with aiofiles.open(self.file_path, mode="w") as file:
            json_string = json.dumps(
                {
                    "__schemas__": {
                        name: {
                            "module_path": c._schema.__module__,
                            "model_path": c._schema.__qualname__,
                        }
                        for name, c in self._collections.items()
                    },
                    **data,
                },
                ensure_ascii=False,
                indent=2,
            )
            await file.write(json_string)

    @override
    def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> JSONFileDocumentCollection[TDocument]:
        self._logger.debug(f'Create collection "{name}"')

        self._collections[name] = JSONFileDocumentCollection(
            database=self,
            name=name,
            schema=schema,
        )

        return cast(JSONFileDocumentCollection[TDocument], self._collections[name])

    @override
    def get_collection(
        self,
        name: str,
    ) -> JSONFileDocumentCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(JSONFileDocumentCollection[TDocument], collection)
        raise ValueError(f'Collection "{name}" does not exists')

    @override
    def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> JSONFileDocumentCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(JSONFileDocumentCollection[TDocument], collection)

        self._collections[name] = JSONFileDocumentCollection(
            database=self,
            name=name,
            schema=schema,
        )

        return cast(JSONFileDocumentCollection[TDocument], self._collections[name])

    @override
    def delete_collection(
        self,
        name: str,
    ) -> None:
        if name in self._collections:
            del self._collections[name]
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
        data: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        self._database = database
        self._name = name
        self._schema = schema
        self._op_counter = 0
        self._lock = asyncio.Lock()

        self.documents = [cast(TDocument, doc) for doc in data] if data else []

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        result = []
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

        async with self._lock:
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
        for i, d in enumerate(self.documents):
            if matches_filters(filters, d):
                async with self._lock:
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
        for i, d in enumerate(self.documents):
            if matches_filters(filters, d):
                async with self._lock:
                    document = self.documents.pop(i)

                await self._database.flush()
                return DeleteResult(deleted_count=1, acknowledged=True, deleted_document=document)

        return DeleteResult(
            acknowledged=True,
            deleted_count=0,
            deleted_document=None,
        )
