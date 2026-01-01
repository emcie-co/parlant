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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Awaitable,
    Callable,
    Generic,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    TypedDict,
    cast,
)

from parlant.core.persistence.common import Cursor, ObjectId, SortDirection, Where
from parlant.core.common import Version


class BaseDocument(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    version: Version.String


TDocument = TypeVar("TDocument", bound=BaseDocument)


@dataclass(frozen=True)
class FindResult(Generic[TDocument]):
    items: Sequence[TDocument]
    total_count: int
    has_more: bool
    next_cursor: Cursor | None = None

    def __iter__(self) -> Iterator[TDocument]:
        """Allow iteration over the documents in the result."""
        return iter(self.items)

    def __bool__(self) -> bool:
        return self.total_count > 0

    @classmethod
    def create(
        cls,
        items: Sequence[TDocument],
        total_count: int,
        limit: int,
    ) -> FindResult[TDocument]:
        has_more = len(items) == limit and total_count > limit
        next_cursor = None

        if has_more and items:
            # For cursor-based pagination, always use creation_utc (primary) and id (tiebreaker)
            last_item = items[-1]
            creation_utc = last_item.get("creation_utc")
            item_id = last_item.get("id")

            if creation_utc is not None and item_id is not None:
                next_cursor = Cursor(creation_utc=str(creation_utc), id=ObjectId(str(item_id)))

        return cls(items=items, total_count=total_count, has_more=has_more, next_cursor=next_cursor)


@dataclass(frozen=True)
class InsertResult:
    acknowledged: bool


@dataclass(frozen=True)
class UpdateResult(Generic[TDocument]):
    acknowledged: bool
    matched_count: int
    modified_count: int
    updated_document: Optional[TDocument]


@dataclass(frozen=True)
class DeleteResult(Generic[TDocument]):
    acknowledged: bool
    deleted_count: int
    deleted_document: Optional[TDocument]


async def identity_loader(doc: BaseDocument) -> BaseDocument:
    return doc


def identity_loader_for(
    type_: type[TDocument],
) -> Callable[[BaseDocument], Awaitable[Optional[TDocument]]]:
    async def loader(doc: BaseDocument) -> Optional[TDocument]:
        return cast(TDocument, doc)

    return loader


class DocumentDatabase(ABC):
    @abstractmethod
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> DocumentCollection[TDocument]:
        """
        Creates a new collection with the given name and returns the collection.
        """
        ...

    @abstractmethod
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> DocumentCollection[TDocument]:
        """
        Retrieves an existing collection by its name.
        """
        ...

    @abstractmethod
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> DocumentCollection[TDocument]:
        """
        Retrieves an existing collection by its name or creates a new one if it does not exist.
        """
        ...

    @abstractmethod
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        """
        Deletes a collection by its name.
        """
        ...


class DocumentCollection(ABC, Generic[TDocument]):
    @abstractmethod
    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        """Finds documents with cursor-based pagination. Results are sorted by creation_utc with id as tiebreaker."""
        ...

    @abstractmethod
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        """Returns the first document that matches the query criteria."""
        ...

    @abstractmethod
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        """Inserts a single document into the collection."""
        ...

    @abstractmethod
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        """Updates the first document that matches the query criteria. If upsert is True,
        inserts the document if it does not exist."""
        ...

    @abstractmethod
    async def delete_one(
        self,
        filters: Where,
    ) -> DeleteResult[TDocument]:
        """Deletes the first document that matches the query criteria."""
        ...
