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

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import zlib
from lagom import Container
from typing import Any, Callable, Optional, Sequence, TypedDict, cast
from typing_extensions import override

from parlant.core.async_utils import Stopwatch
from parlant.core.common import Version
from parlant.core.loggers import Logger
from parlant.core.meter import DurationHistogram, Meter
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
)
from parlant.core.tracer import Tracer


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of an embedding operation."""

    vectors: Sequence[Sequence[float]]


@dataclass
class _EmbeddingCacheEntry:
    """An entry in the embedding LRU cache."""

    text_length: int
    checksum: int
    vector: Sequence[float]


_EMBEDDING_CACHE_MAX_SIZE = 1000


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


_EMBED_DURATION_HISTOGRAM: DurationHistogram | None = None


class BaseEmbedder(Embedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter, model_name: str) -> None:
        self.logger = logger
        self.tracer = tracer
        self.meter = meter
        self.model_name = model_name

        # LRU cache: checksum -> cache entry
        self._cache: OrderedDict[int, _EmbeddingCacheEntry] = OrderedDict()
        # Index for fast length-based lookup: length -> set of checksums
        self._cache_length_index: dict[int, set[int]] = {}

        global _EMBED_DURATION_HISTOGRAM
        if _EMBED_DURATION_HISTOGRAM is None:
            _EMBED_DURATION_HISTOGRAM = meter.create_duration_histogram(
                name="embed",
                description="Duration of embedding requests in milliseconds",
            )

    @abstractmethod
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult: ...

    def _compute_checksum(self, text: str) -> int:
        """Compute a fast checksum for the given text."""
        return zlib.crc32(text.encode("utf-8"))

    def _cache_get(self, text: str) -> Sequence[float] | None:
        """Get a cached embedding vector for the given text.

        Uses a two-tier lookup:
        1. Check if any cache entries have the same length (fast)
        2. If so, compute checksum and check for exact match
        """
        text_length = len(text)

        # Fast path: check if any entries have this length
        if text_length not in self._cache_length_index:
            return None

        candidate_checksums = self._cache_length_index[text_length]

        if not candidate_checksums:
            return None

        # Compute checksum only if we have length matches
        checksum = self._compute_checksum(text)

        if checksum not in candidate_checksums:
            return None

        # Cache hit - move to end for LRU
        if entry := self._cache.get(checksum):
            self._cache.move_to_end(checksum)
            return entry.vector

        return None

    def _cache_put(self, text: str, vector: Sequence[float]) -> None:
        """Store an embedding vector in the cache."""
        checksum = self._compute_checksum(text)
        text_length = len(text)

        # If already in cache, just update and move to end
        if checksum in self._cache:
            self._cache[checksum].vector = vector
            self._cache.move_to_end(checksum)
            return

        # Evict oldest entry if at capacity
        if len(self._cache) >= _EMBEDDING_CACHE_MAX_SIZE:
            oldest_checksum, oldest_entry = self._cache.popitem(last=False)
            self._cache_length_index[oldest_entry.text_length].discard(oldest_checksum)
            self._cache_length_index.pop(oldest_entry.text_length, None)

        # Add new entry
        self._cache[checksum] = _EmbeddingCacheEntry(
            text_length=text_length,
            checksum=checksum,
            vector=vector,
        )

        # Update length index
        if text_length not in self._cache_length_index:
            self._cache_length_index[text_length] = set()
        self._cache_length_index[text_length].add(checksum)

    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        assert _EMBED_DURATION_HISTOGRAM is not None

        # Check cache for each text, collect hits and misses
        cached_results: dict[int, Sequence[float]] = {}
        texts_to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                cached_results[i] = cached
            else:
                texts_to_embed.append((i, text))

        # If all texts were cached, return immediately
        if not texts_to_embed:
            return EmbeddingResult(vectors=[cached_results[i] for i in range(len(texts))])

        async with _EMBED_DURATION_HISTOGRAM.measure(
            {
                "class.name": self.__class__.__qualname__,
                "embedding.model.name": self.model_name,
                **({"embedding.tag": hints["tag"]} if "tag" in hints else {}),
            },
        ):
            start = Stopwatch.start()

            try:
                # Only embed texts that weren't in cache
                result = await self.do_embed(
                    [text for _, text in texts_to_embed],
                    hints,
                )
            except Exception:
                self.tracer.add_event(
                    "embed.request_failed",
                    attributes={
                        "class.name": self.__class__.__qualname__,
                        "model.name": self.model_name,
                        "duration": start.elapsed,
                    },
                )
                raise
            else:
                self.tracer.add_event(
                    "embed.request_completed",
                    attributes={
                        "class.name": self.__class__.__qualname__,
                        "model.name": self.model_name,
                        "duration": start.elapsed,
                    },
                )

            # Cache new results and merge with cached results
            for (orig_idx, text), vector in zip(texts_to_embed, result.vectors):
                self._cache_put(text, vector)
                cached_results[orig_idx] = vector

        # Reconstruct results in original order
        return EmbeddingResult(vectors=[cached_results[i] for i in range(len(texts))])


class EmbedderFactory:
    """Factory for creating embedder instances."""

    def __init__(self, container: Container):
        self._container = container

    def create_embedder(self, embedder_type: type[Embedder]) -> Embedder:
        if embedder_type == NullEmbedder:
            return NullEmbedder()
        else:
            return self._container[embedder_type]


class NullEmbedder(Embedder):
    """A null embedder that returns zero vectors."""

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


class EmbedderResultDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    vectors: Sequence[Sequence[float]]


class EmbedderResultDocument(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
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

    VERSION = Version.from_string("0.2.0")

    def __init__(
        self,
        document_database: DocumentDatabase,
    ):
        self._database = document_database
        self._collections: dict[type[Embedder], DocumentCollection[EmbedderResultDocument]] = {}

    async def _document_loader(self, doc: BaseDocument) -> Optional[EmbedderResultDocument]:
        if doc["version"] == "0.1.0":
            d = cast(EmbedderResultDocument_v0_1_0, doc)
            return EmbedderResultDocument(
                id=d["id"],
                creation_utc=datetime.now(timezone.utc).isoformat(),
                version=d["version"],
                vectors=d["vectors"],
            )

        if doc["version"] == "0.2.0":
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
            creation_utc=datetime.now(timezone.utc).isoformat(),
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
