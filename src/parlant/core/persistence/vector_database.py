# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# You may not use this file except in compliance with the License.
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
from typing import Awaitable, Callable, Generic, Optional, Sequence, TypeVar, TypedDict
from typing_extensions import Required

from parlant.core.common import JSONSerializable, Version
from parlant.core.nlp.embedding import Embedder
from parlant.core.persistence.common import ObjectId, Where


class BaseDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]


TDocument = TypeVar("TDocument", bound=BaseDocument)


async def identity_loader(doc: BaseDocument) -> BaseDocument:
    return doc


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


@dataclass(frozen=True)
class SimilarDocumentResult(Generic[TDocument]):
    document: TDocument
    distance: float

    def __hash__(self) -> int:
        return hash(str(self.document))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, SimilarDocumentResult):
            return bool(self.document == value.document)
        return False


class VectorDatabase(ABC):
    @abstractmethod
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> VectorCollection[TDocument]: ...

    @abstractmethod
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> VectorCollection[TDocument]: ...

    @abstractmethod
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> VectorCollection[TDocument]: ...

    @abstractmethod
    async def delete_collection(
        self,
        name: str,
    ) -> None: ...

    @abstractmethod
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None: ...

    @abstractmethod
    async def remove_metadata(
        self,
        key: str,
    ) -> None: ...

    @abstractmethod
    async def read_metadata(
        self,
    ) -> dict[str, JSONSerializable]: ...


class VectorCollection(ABC, Generic[TDocument]):
    @abstractmethod
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]: ...

    @abstractmethod
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]: ...

    @abstractmethod
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult: ...

    @abstractmethod
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]: ...

    @abstractmethod
    async def delete_one(
        self,
        filters: Where,
    ) -> DeleteResult[TDocument]: ...

    @abstractmethod
    async def find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
    ) -> Sequence[SimilarDocumentResult[TDocument]]: ...
