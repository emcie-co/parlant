from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Type, cast
import aiofiles

from emcie.server.core.persistence.common import (
    CollectionDescriptor,
    ObjectId,
    Where,
)


class DocumentDatabase(ABC):

    @abstractmethod
    async def find(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> Sequence[Mapping[str, Any]]: ...

    @abstractmethod
    async def find_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> Mapping[str, Any]:
        """
        Returns the first document that matches the query criteria.
        """

    ...

    @abstractmethod
    async def insert_one(
        self,
        collection: CollectionDescriptor,
        document: Mapping[str, Any],
    ) -> ObjectId: ...

    @abstractmethod
    async def update_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
        updated_document: Mapping[str, Any],
        upsert: bool = False,
    ) -> ObjectId:
        """
        Updates the first document that matches the query criteria.
        """
        ...

    @abstractmethod
    async def delete_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> None:
        """
        Deletes the first document that matches the query criteria.
        """
        ...

    async def flush(self) -> None: ...


def _matches_filters(
    field_filters: Where,
    candidate: Mapping[str, Any],
) -> bool:
    tests: dict[str, Callable[[Any, Any], bool]] = {
        "$eq": lambda candidate, filter_value: candidate == filter_value,
        "$ne": lambda candidate, filter_value: candidate != filter_value,
        "$gt": lambda candidate, filter_value: candidate > filter_value,
        "$gte": lambda candidate, filter_value: candidate >= filter_value,
        "$lt": lambda candidate, filter_value: candidate < filter_value,
        "$lte": lambda candidate, filter_value: candidate <= filter_value,
    }
    for field, field_filter in field_filters.items():
        for filter_name, filter_value in field_filter.items():
            if not tests[filter_name](candidate.get(field), filter_value):
                return False
    return True


class TransientDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        collections: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    ) -> None:
        self._collections = (
            {name: list(docs) for name, docs in collections.items()}
            if collections
            else defaultdict(list)
        )

    async def find(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> Sequence[Mapping[str, Any]]:
        return list(
            filter(
                lambda d: _matches_filters(filters, d),
                self._collections[collection.name],
            )
        )

    async def find_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> Mapping[str, Any]:
        matched_documents = await self.find(collection, filters)
        if len(matched_documents) >= 1:
            return matched_documents[0]
        raise ValueError("No document found matching the provided filters.")

    async def insert_one(
        self,
        collection: CollectionDescriptor,
        document: Mapping[str, Any],
    ) -> ObjectId:
        self._collections[collection.name].append(document)
        return cast(ObjectId, document["id"])

    async def update_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
        updated_document: Mapping[str, Any],
        upsert: bool = False,
    ) -> ObjectId:
        for i, d in enumerate(self._collections[collection.name]):
            if _matches_filters(filters, d):
                self._collections[collection.name][i] = updated_document
                return cast(ObjectId, updated_document["id"])
        if upsert:
            document_id = await self.insert_one(collection, updated_document)
            return document_id

        raise ValueError("No document found matching the provided filters.")

    async def delete_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> None:
        for i, d in enumerate(self._collections[collection.name]):
            if _matches_filters(filters, d):
                del self._collections[collection.name][i]
                return
        raise ValueError("No document found matching the provided filters.")


class JSONFileDocumentDatabase(DocumentDatabase):

    def __init__(self, file_path: Path) -> None:
        self._collection_descriptors: dict[str, CollectionDescriptor] = {}
        self.file_path = file_path
        self._lock = asyncio.Lock()
        self.op_counter = 0
        if not self.file_path.exists():
            self.file_path.write_text(json.dumps({}))
        self.transient_db: TransientDocumentDatabase

    async def __aenter__(self) -> JSONFileDocumentDatabase:
        self.transient_db = TransientDocumentDatabase(await self._load_data())
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        await self.flush()
        return False

    async def _process_operation_counter(self) -> None:
        self.op_counter += 1
        if self.op_counter % 5:
            await self.flush()

    async def _load_data(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
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
                    data,
                    ensure_ascii=False,
                    indent=4,
                )
                await file.write(json_string)

    async def find(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> Sequence[Mapping[str, Any]]:
        # TODO MC-101
        self._collection_descriptors[collection.name] = collection
        return [
            collection.schema.model_validate(doc).model_dump()
            for doc in await self.transient_db.find(collection, filters)
        ]

    async def find_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> Mapping[str, Any]:
        # TODO MC-101
        self._collection_descriptors[collection.name] = collection
        result = await self.transient_db.find_one(collection, filters)
        return collection.schema.model_validate(result).model_dump()

    async def insert_one(
        self,
        collection: CollectionDescriptor,
        document: Mapping[str, Any],
    ) -> ObjectId:
        # TODO MC-101
        self._collection_descriptors[collection.name] = collection
        document_id = await self.transient_db.insert_one(
            collection,
            collection.schema(**document).model_dump(mode="json"),
        )
        await self._process_operation_counter()
        return document_id

    async def update_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
        updated_document: Mapping[str, Any],
        upsert: bool = False,
    ) -> ObjectId:
        # TODO MC-101
        self._collection_descriptors[collection.name] = collection
        result = await self.transient_db.update_one(
            collection,
            filters,
            collection.schema(**updated_document).model_dump(mode="json"),
            upsert,
        )
        await self._process_operation_counter()
        return result

    async def delete_one(
        self,
        collection: CollectionDescriptor,
        filters: Where,
    ) -> None:
        # TODO MC-101
        self._collection_descriptors[collection.name] = collection
        await self.transient_db.delete_one(collection, filters)
        await self._process_operation_counter()

    async def _list_collections(self) -> list[str]:
        return list(self._collection_descriptors.keys())

    async def _get_collection(
        self,
        collection_name: str,
    ) -> CollectionDescriptor:
        return self._collection_descriptors[collection_name]

    async def flush(self) -> None:
        if self.transient_db:
            data = {}
            for collection_name in await self._list_collections():
                data[collection_name] = list(
                    await self.transient_db.find(
                        collection=await self._get_collection(collection_name),
                        filters={},
                    )
                )

            await self._save_data(data)
