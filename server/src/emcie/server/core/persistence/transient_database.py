from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Type
from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.persistence.common import ObjectId, Where, matches_filters
from emcie.server.core.persistence.document_database import DocumentCollection, DocumentDatabase


class TransientDocumentDatabase(DocumentDatabase):
    def __init__(self) -> None:
        self._collections: dict[str, _TransientDocumentCollection] = {}

    def create_collection(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> _TransientDocumentCollection:
        collection = _TransientDocumentCollection(
            name=name,
            schema=schema,
        )
        self._collections[name] = collection
        return collection

    def get_collection(
        self,
        name: str,
    ) -> _TransientDocumentCollection:
        if name in self._collections:
            return self._collections[name]
        raise ValueError(f'Collection "{name}" does not exist')

    def get_or_create_collection(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> _TransientDocumentCollection:
        if collection := self._collections.get(name):
            return collection
        return self.create_collection(
            name=name,
            schema=schema,
        )

    def delete_collection(
        self,
        name: str,
    ) -> None:
        if name in self._collections:
            del self._collections[name]
        else:
            raise ValueError(f'Collection "{name}" does not exist')


class _TransientDocumentCollection(DocumentCollection):
    def __init__(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
        data: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        self._name = name
        self._schema = schema
        self._documents = list(data) if data else []

    async def find(
        self,
        filters: Where,
    ) -> Sequence[Mapping[str, Any]]:
        return list(
            filter(
                lambda d: matches_filters(filters, d),
                self._documents,
            )
        )

    async def find_one(
        self,
        filters: Where,
    ) -> Mapping[str, Any]:
        matched_documents = await self.find(filters)
        if len(matched_documents) >= 1:
            return matched_documents[0]
        raise ValueError("No document found matching the provided filters.")

    async def insert_one(
        self,
        document: Mapping[str, Any],
    ) -> ObjectId:
        self._documents.append(self._schema(**document).model_dump(mode="json"))
        document_id: ObjectId = document["id"]
        return document_id

    async def update_one(
        self,
        filters: Where,
        updated_document: Mapping[str, Any],
        upsert: bool = False,
    ) -> ObjectId:
        document_id: ObjectId

        for i, d in enumerate(self._documents):
            if matches_filters(filters, d):
                self._documents[i] = updated_document
                document_id = updated_document["id"]
                return document_id

        if upsert:
            document_id = await self.insert_one(updated_document)
            return document_id

        raise ValueError("No document found matching the provided filters.")

    async def delete_one(
        self,
        filters: Where,
    ) -> None:
        for i, d in enumerate(self._documents):
            if matches_filters(filters, d):
                del self._documents[i]
                return
        raise ValueError("No document found matching the provided filters.")
