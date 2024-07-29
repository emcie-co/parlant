from __future__ import annotations
import asyncio
import importlib
import json
import operator
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type

import aiofiles
from loguru import logger
from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.persistence.common import ObjectId, Where, matches_filters
from emcie.server.core.persistence.document_database import DocumentCollection, DocumentDatabase


class JSONFileDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        file_path: Path,
    ) -> None:
        self.file_path = file_path
        self._lock = asyncio.Lock()
        self.op_counter = 0
        if not self.file_path.exists():
            self.file_path.write_text(json.dumps({}))
        self._collections: dict[str, JSONFileDocumentCollection]

    async def __aenter__(self) -> JSONFileDocumentDatabase:
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
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        await self.flush()
        return False

    async def _load_data(
        self,
    ) -> dict[str, Any]:
        async with self._lock:
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
        async with self._lock:
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
                    indent=4,
                )
                await file.write(json_string)

    def create_collection(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> JSONFileDocumentCollection:
        logger.debug(f'Create collection "{name}"')
        new_collection = JSONFileDocumentCollection(
            database=self,
            name=name,
            schema=schema,
        )
        self._collections[name] = new_collection
        return new_collection

    def get_collection(
        self,
        name: str,
    ) -> JSONFileDocumentCollection:
        if collection := self._collections.get(name):
            return collection
        raise ValueError(f'Collection "{name}" does not exists')

    def get_or_create_collection(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> JSONFileDocumentCollection:
        if collection := self._collections.get(name):
            return collection
        new_collection = JSONFileDocumentCollection(
            database=self,
            name=name,
            schema=schema,
        )
        self._collections[name] = new_collection
        return new_collection

    def delete_collection(
        self,
        name: str,
    ) -> None:
        if name in self._collections:
            del self._collections[name]
        raise ValueError(f'Collection "{name}" does not exists')

    async def flush(self) -> None:
        data = {}
        for collection_name in self._collections:
            data[collection_name] = self._collections[collection_name]._documents
        await self._save_data(data)


class JSONFileDocumentCollection(DocumentCollection):

    def __init__(
        self,
        database: JSONFileDocumentDatabase,
        name: str,
        schema: Type[DefaultBaseModel],
        data: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        self._database = database
        self._name = name
        self._schema = schema
        self._documents = [doc for doc in data] if data else []
        self._op_counter = 0

    async def _process_operation_counter(self) -> None:
        self._op_counter += 1
        if self._op_counter % 5:
            await self._database.flush()

    async def find(
        self,
        filters: Where,
    ) -> Sequence[Mapping[str, Any]]:
        return [
            self._schema.model_validate(doc).model_dump()
            for doc in filter(
                lambda d: matches_filters(filters, d),
                self._documents,
            )
        ]

    async def find_one(
        self,
        filters: Where,
    ) -> Mapping[str, Any]:
        matched_documents = await self.find(filters)
        if not matched_documents:
            raise ValueError("No document found matching the provided filters.")
        result = matched_documents[0]
        return self._schema.model_validate(result).model_dump()

    async def insert_one(
        self,
        document: Mapping[str, Any],
    ) -> ObjectId:
        self._documents.append(self._schema(**document).model_dump(mode="json"))

        await self._process_operation_counter()

        document_id: ObjectId = document["id"]
        return document_id

    async def update_one(
        self,
        filters: Where,
        updated_document: Mapping[str, Any],
        upsert: bool = False,
    ) -> ObjectId:
        for i, d in enumerate(self._documents):
            if matches_filters(filters, d):
                self._documents[i] = updated_document

                await self._process_operation_counter()

                document_id: ObjectId = updated_document["id"]
                return document_id

        if upsert:
            document_id = await self.insert_one(updated_document)

            await self._process_operation_counter()

            return document_id

        raise ValueError("No document found matching the provided filters.")

    async def delete_one(
        self,
        filters: Where,
    ) -> None:
        for i, d in enumerate(self._documents):
            if matches_filters(filters, d):
                del self._documents[i]

                await self._process_operation_counter()
                return
        raise ValueError("No document found matching the provided filters.")
