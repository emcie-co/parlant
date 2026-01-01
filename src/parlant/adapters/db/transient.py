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

from __future__ import annotations
from typing import Awaitable, Callable, Optional, Sequence, cast
from typing_extensions import override
from typing_extensions import get_type_hints

from parlant.core.persistence.common import (
    Cursor,
    SortDirection,
    matches_filters,
    Where,
    ObjectId,
    ensure_is_total,
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
)


class TransientDocumentDatabase(DocumentDatabase):
    def __init__(self) -> None:
        self._collections: dict[str, TransientDocumentCollection[BaseDocument]] = {}

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> TransientDocumentCollection[TDocument]:
        annotations = get_type_hints(schema)
        assert "id" in annotations and annotations["id"] == ObjectId

        self._collections[name] = TransientDocumentCollection(
            name=name,
            schema=schema,
        )

        return cast(TransientDocumentCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> TransientDocumentCollection[TDocument]:
        if name in self._collections:
            return cast(TransientDocumentCollection[TDocument], self._collections[name])
        raise ValueError(f'Collection "{name}" does not exist')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> TransientDocumentCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(TransientDocumentCollection[TDocument], collection)

        annotations = get_type_hints(schema)
        assert "id" in annotations and annotations["id"] == ObjectId

        return await self.create_collection(
            name=name,
            schema=schema,
        )

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name in self._collections:
            del self._collections[name]
        else:
            raise ValueError(f'Collection "{name}" does not exist')


class TransientDocumentCollection(DocumentCollection[TDocument]):
    def __init__(
        self,
        name: str,
        schema: type[TDocument],
        data: Optional[Sequence[TDocument]] = None,
    ) -> None:
        self._name = name
        self._schema = schema
        self._documents = list(data) if data else []

    @override
    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        # First, filter documents
        filtered_docs = [doc for doc in self._documents if matches_filters(filters, doc)]

        # Sort by creation_utc with id as tiebreaker according to sort_direction
        sort_direction = sort_direction or SortDirection.ASC
        filtered_docs = self._apply_sort(filtered_docs, sort_direction)

        # Apply cursor-based pagination if cursor is provided
        if cursor:
            filtered_docs = self._apply_cursor_filter(filtered_docs, cursor, sort_direction)

        total_count = len(filtered_docs)

        # Apply limit
        has_more = False
        next_cursor = None

        if limit is not None and len(filtered_docs) > limit:
            # There are more items beyond the limit
            has_more = True
            result_docs = filtered_docs[:limit]

            # Generate next cursor from the last item if we have results
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
        self,
        documents: list[TDocument],
        sort_direction: SortDirection,
    ) -> list[TDocument]:
        docs = list(documents)  # don't mutate input

        # Sort by creation_utc with id as tiebreaker according to sort_direction
        reverse_order = sort_direction == SortDirection.DESC
        docs.sort(
            key=lambda d: (
                d.get("creation_utc") or "",  # Primary sort: creation_utc
                d.get("id") or "",  # Tiebreaker: id
            ),
            reverse=reverse_order,
        )

        return docs

    def _apply_cursor_filter(
        self,
        documents: list[TDocument],
        cursor: Cursor,
        sort_direction: SortDirection,
    ) -> list[TDocument]:
        cursor_creation_utc = str(cursor.creation_utc)
        cursor_id = str(cursor.id)

        result = []
        for doc in documents:
            doc_creation_utc = str(doc.get("creation_utc", ""))
            doc_id = str(doc.get("id", ""))

            if sort_direction == SortDirection.DESC:
                # For descending order pagination, include documents that come after the cursor
                # This matches the MongoDB query pattern:
                # { "$or": [
                #     { "creation_utc": { "$lt": cursor_creation_utc } },
                #     { "creation_utc": cursor_creation_utc, "id": { "$lt": cursor_id } }
                # ]}
                if doc_creation_utc < cursor_creation_utc or (
                    doc_creation_utc == cursor_creation_utc and doc_id < cursor_id
                ):
                    result.append(doc)
            else:  # SortDirection.ASC
                # For ascending order pagination, include documents that come after the cursor
                # { "$or": [
                #     { "creation_utc": { "$gt": cursor_creation_utc } },
                #     { "creation_utc": cursor_creation_utc, "id": { "$gt": cursor_id } }
                # ]}
                if doc_creation_utc > cursor_creation_utc or (
                    doc_creation_utc == cursor_creation_utc and doc_id > cursor_id
                ):
                    result.append(doc)

        return result

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        for doc in self._documents:
            if matches_filters(filters, doc):
                return doc

        return None

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        self._documents.append(document)

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        for i, d in enumerate(self._documents):
            if matches_filters(filters, d):
                self._documents[i] = cast(TDocument, {**self._documents[i], **params})

                return UpdateResult(
                    acknowledged=True,
                    matched_count=1,
                    modified_count=1,
                    updated_document=self._documents[i],
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
        for i, d in enumerate(self._documents):
            if matches_filters(filters, d):
                document = self._documents.pop(i)

                return DeleteResult(deleted_count=1, acknowledged=True, deleted_document=document)

        return DeleteResult(
            acknowledged=True,
            deleted_count=0,
            deleted_document=None,
        )
