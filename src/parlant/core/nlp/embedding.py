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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from lagom import Container
from typing import Any, Callable, Optional, Sequence, TypedDict, cast
from typing_extensions import override

from parlant.core.common import Version
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
)


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of an embedding operation."""

    vectors: Sequence[Sequence[float]]


class Embedder(ABC):
    """An interface for embedding text into vector representations."""

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult: ...

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def max_tokens(self) -> int: ...

    @property
    @abstractmethod
    def tokenizer(self) -> EstimatingTokenizer: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...


class EmbedderFactory:
    """Factory for creating embedder instances."""

    def __init__(self, container: Container):
        self._container = container

    def create_embedder(self, embedder_type: type[Embedder]) -> Embedder:
        if embedder_type == NoOpEmbedder:
            return NoOpEmbedder()
        else:
            return self._container[embedder_type]


class NoOpEmbedder(Embedder):
    """A no-op embedder that returns zero vectors."""

    def __init__(self) -> None:
        self._tokenizer = ZeroEstimatingTokenizer()

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        return EmbeddingResult(vectors=[[0.0] * self.dimensions for _ in texts])

    @property
    @override
    def id(self) -> str:
        return "no_op"

    @property
    @override
    def max_tokens(self) -> int:
        return 8192  # Arbitrary large number for embedding

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def dimensions(self) -> int:
        return 1536  # Standard embedding dimension


class EmbedderResultDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    vectors: Sequence[Sequence[float]]


class EmbeddingCache(ABC):
    """An interface for caching embedding results."""

    @abstractmethod
    async def get(
        self,
        embedder_type: type[Embedder],
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> Optional[EmbeddingResult]:
        pass

    @abstractmethod
    async def set(
        self,
        embedder_type: type[Embedder],
        texts: list[str],
        vectors: Sequence[Sequence[float]],
        hints: Mapping[str, Any] = {},
    ) -> None:
        pass


EmbeddingCacheProvider = Callable[[], EmbeddingCache]


class BasicEmbeddingCache(EmbeddingCache):
    """A basic embedding cache that uses a document database to store results."""

    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        document_database: DocumentDatabase,
    ):
        self._database = document_database
        self._collections: dict[type[Embedder], DocumentCollection[EmbedderResultDocument]] = {}

    async def _document_loader(self, doc: BaseDocument) -> Optional[EmbedderResultDocument]:
        if doc["version"] == "0.1.0":
            return cast(EmbedderResultDocument, doc)
        return None

    async def _get_or_create_collection(
        self,
        embedder_type: type[Embedder],
    ) -> DocumentCollection[EmbedderResultDocument]:
        if embedder_type not in self._collections:
            collection = await self._database.get_or_create_collection(
                name=embedder_type.__name__,
                schema=EmbedderResultDocument,
                document_loader=self._document_loader,
            )
            self._collections[embedder_type] = collection

        return self._collections[embedder_type]

    def _generate_id(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> str:
        sorted_hints = json.dumps(dict(sorted(hints.items())), sort_keys=True)
        key_content = f"{str(texts)}:{sorted_hints}"
        return hashlib.sha256(key_content.encode()).hexdigest()

    def _serialize_result(
        self,
        id: str,
        vectors: Sequence[Sequence[float]],
    ) -> EmbedderResultDocument:
        return EmbedderResultDocument(
            id=ObjectId(id),
            version=self.VERSION.to_string(),
            vectors=vectors,
        )

    def _deserialize_result(
        self,
        doc: EmbedderResultDocument,
    ) -> EmbeddingResult:
        return EmbeddingResult(vectors=doc["vectors"])

    async def get(
        self,
        embedder_type: type[Embedder],
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> Optional[EmbeddingResult]:
        collection = await self._get_or_create_collection(embedder_type)

        id = self._generate_id(texts, hints)
        doc = await collection.find_one({"id": {"$eq": ObjectId(id)}})

        if doc:
            return self._deserialize_result(doc)

        return None

    async def set(
        self,
        embedder_type: type[Embedder],
        texts: list[str],
        vectors: Sequence[Sequence[float]],
        hints: Mapping[str, Any] = {},
    ) -> None:
        collection = await self._get_or_create_collection(embedder_type)

        id = self._generate_id(texts, hints)
        doc = self._serialize_result(id, vectors)

        await collection.insert_one(doc)


class NullEmbeddingCache(EmbeddingCache):
    """A no-op embedding cache that does nothing."""

    async def get(
        self,
        embedder_type: type[Embedder],
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> Optional[EmbeddingResult]:
        return None

    async def set(
        self,
        embedder_type: type[Embedder],
        texts: list[str],
        vectors: Sequence[Sequence[float]],
        hints: Mapping[str, Any] = {},
    ) -> None:
        pass
