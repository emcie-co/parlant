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
from typing import Any, Awaitable, Callable, Generic, Mapping, Optional, Sequence, cast
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
    CollectionIndex,
    CollectionSort,
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


_NO_POSITIONS: Mapping[int, None] = {}


class _EqualityIndexes(Generic[TDocument]):
    """Documents in insertion order, with a value-to-documents index per indexed field.

    A filter that pins an indexed field to a value (``$eq`` or ``$in``, alone or
    inside ``$and``/``$or``) is answered from those documents only. Every other
    filter still examines each document, exactly as before; an index never
    changes what a query returns, only how many documents it looks at.

    ``id`` is always indexed. Other fields are indexed when a store declares
    them through ``ensure_indexes``.
    """

    def __init__(self) -> None:
        self._documents: dict[int, TDocument] = {}
        self._next_position = 0
        self._positions_by_value: dict[str, dict[Any, dict[int, None]]] = {"id": {}}
        # Positions whose value for an indexed field cannot be a dictionary key
        # (or is missing); they stay candidates for every lookup on that field.
        self._unindexed_positions: dict[str, dict[int, None]] = {"id": {}}
        self._indexed_values: dict[int, list[tuple[str, Any]]] = {}
        self.examined_document_count = 0

    def __len__(self) -> int:
        return len(self._documents)

    def all(self) -> list[TDocument]:
        return list(self._documents.values())

    def get(self, position: int) -> TDocument:
        return self._documents[position]

    def index_field(self, field: str) -> None:
        if field in self._positions_by_value:
            return

        self._positions_by_value[field] = {}
        self._unindexed_positions[field] = {}

        for position, document in self._documents.items():
            self._indexed_values[position].append(self._enter(field, position, document))

    def add(self, document: TDocument) -> int:
        position = self._next_position
        self._next_position += 1
        self._documents[position] = document
        self._indexed_values[position] = [
            self._enter(field, position, document) for field in self._positions_by_value
        ]
        return position

    def replace(self, position: int, document: TDocument) -> None:
        self._leave(position)
        self._documents[position] = document
        self._indexed_values[position] = [
            self._enter(field, position, document) for field in self._positions_by_value
        ]

    def remove(self, position: int) -> TDocument:
        self._leave(position)
        return self._documents.pop(position)

    def _enter(self, field: str, position: int, document: Mapping[str, Any]) -> tuple[str, Any]:
        # What was entered is remembered per document, so leaving the index never
        # depends on the document still holding the same values.
        if field in document:
            value = document[field]
            try:
                self._positions_by_value[field].setdefault(value, {})[position] = None
                return (field, value)
            except TypeError:  # unhashable, so it cannot be looked up by value
                pass

        self._unindexed_positions[field][position] = None
        return (field, _UNINDEXED)

    def _leave(self, position: int) -> None:
        for field, value in self._indexed_values.pop(position):
            if value is _UNINDEXED:
                del self._unindexed_positions[field][position]
                continue

            positions = self._positions_by_value[field][value]
            del positions[position]

            if not positions:
                del self._positions_by_value[field][value]

    def _positions_holding(self, field: str, value: Any) -> Optional[Mapping[int, None]]:
        """The positions whose ``field`` may equal ``value``; None if not indexed."""
        if field not in self._positions_by_value:
            return None

        try:
            positions = self._positions_by_value[field].get(value, _NO_POSITIONS)
        except TypeError:
            return None

        unindexed = self._unindexed_positions[field]
        return {**positions, **unindexed} if unindexed else positions

    def _candidates(self, filters: Where) -> Optional[Mapping[int, None]]:
        """The positions that can match, or None when no index narrows the filter.

        Conditions that must all hold are intersected starting from the one with
        the fewest positions, so the work follows the most selective condition.
        """
        if not filters:
            return None

        required: list[Mapping[int, None]] = []

        def any_of(alternatives: Sequence[Optional[Mapping[int, None]]]) -> None:
            if all(alternative is not None for alternative in alternatives):
                merged: dict[int, None] = {}
                for alternative in alternatives:
                    merged.update(cast(Mapping[int, None], alternative))
                required.append(merged)

        if next(iter(filters)) in ("$and", "$or"):
            for operator, operands in cast(Mapping[str, Sequence[Where]], filters).items():
                narrowed = [self._candidates(operand) for operand in operands]
                if operator == "$and":
                    required.extend(n for n in narrowed if n is not None)
                else:
                    any_of(narrowed)
        else:
            for field, conditions in cast(Mapping[str, Mapping[str, Any]], filters).items():
                for operator, value in conditions.items():
                    if operator == "$eq":
                        positions = self._positions_holding(field, value)
                        if positions is not None:
                            required.append(positions)
                    elif operator == "$in":
                        any_of([self._positions_holding(field, item) for item in value])

        if not required:
            return None

        required.sort(key=len)
        return {
            position: None
            for position in required[0]
            if all(position in others for others in required[1:])
        }

    def matching(self, filters: Where) -> list[int]:
        """Positions of the documents matching ``filters``, in insertion order."""
        candidates = self._candidates(filters)
        positions = self._documents if candidates is None else sorted(candidates)
        self.examined_document_count += len(positions)
        return [
            position
            for position in positions
            if matches_filters(filters, self._documents[position])
        ]


_UNINDEXED = object()


class TransientDocumentCollection(DocumentCollection[TDocument]):
    def __init__(
        self,
        name: str,
        schema: type[TDocument],
        data: Optional[Sequence[TDocument]] = None,
    ) -> None:
        self._name = name
        self._schema = schema
        self._indexes: _EqualityIndexes[TDocument] = _EqualityIndexes()

        for document in data or []:
            self._indexes.add(document)

    @property
    def _documents(self) -> list[TDocument]:
        return self._indexes.all()

    @property
    def examined_document_count(self) -> int:
        """How many documents queries have evaluated against a filter so far.

        A lookup on an indexed field counts only the documents holding the
        value; any other query counts every document in the collection. Tests
        and diagnostics use it to tell an indexed lookup from a full scan."""
        return self._indexes.examined_document_count

    def reset_examined_document_count(self) -> None:
        self._indexes.examined_document_count = 0

    @override
    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        # First, filter documents
        filtered_docs = [self._indexes.get(p) for p in self._indexes.matching(filters)]

        # Sort by creation_utc with id as tiebreaker according to sort_direction
        sort_direction = sort_direction or SortDirection.ASC
        filtered_docs = self._apply_sort(filtered_docs, sort_direction)
        total_count = len(filtered_docs)

        # Apply cursor-based pagination if cursor is provided
        if cursor:
            filtered_docs = self._apply_cursor_filter(filtered_docs, cursor, sort_direction)

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
                    sort_value=str(last_doc.get("creation_utc", "")),
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

    def _apply_field_sort(
        self,
        documents: Sequence[TDocument],
        sort: CollectionSort,
    ) -> list[TDocument]:
        docs = list(documents)

        for field_name, direction in reversed(sort):
            docs.sort(
                key=lambda d: cast(Any, d.get(field_name)),
                reverse=direction == SortDirection.DESC,
            )

        return docs

    def _apply_cursor_filter(
        self,
        documents: list[TDocument],
        cursor: Cursor,
        sort_direction: SortDirection,
    ) -> list[TDocument]:
        cursor_creation_utc = str(cursor.sort_value)
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
        sort: Optional[CollectionSort] = None,
    ) -> Optional[TDocument]:
        matching_documents = [self._indexes.get(p) for p in self._indexes.matching(filters)]

        if sort:
            matching_documents = self._apply_field_sort(matching_documents, sort)

        for doc in matching_documents:
            return doc

        return None

    @override
    async def ensure_indexes(
        self,
        indexes: Sequence[CollectionIndex],
    ) -> None:
        # A lookup is by equality on one field, so the leading field of each
        # index is what can be used; the rest of a compound index is its order.
        for index in indexes:
            if index.fields:
                self._indexes.index_field(index.fields[0][0])

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        self._indexes.add(document)

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        for position in self._indexes.matching(filters):
            updated = cast(TDocument, {**self._indexes.get(position), **params})
            self._indexes.replace(position, updated)

            return UpdateResult(
                acknowledged=True,
                matched_count=1,
                modified_count=1,
                updated_document=updated,
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
        for position in self._indexes.matching(filters):
            document = self._indexes.remove(position)

            return DeleteResult(deleted_count=1, acknowledged=True, deleted_document=document)

        return DeleteResult(
            acknowledged=True,
            deleted_count=0,
            deleted_document=None,
        )
