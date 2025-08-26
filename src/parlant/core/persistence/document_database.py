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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Awaitable,
    Callable,
    Generic,
    Optional,
    Sequence,
    TypeVar,
    TypedDict,
    cast,
)

from parlant.core.persistence.common import ObjectId, Where
from parlant.core.common import Version


class BaseDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String


TDocument = TypeVar("TDocument", bound=BaseDocument)


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
    ) -> Sequence[TDocument]:
        """Finds all documents that match the given filters."""
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
