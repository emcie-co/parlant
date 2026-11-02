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
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import zlib
from typing import Any, Awaitable, Callable, Optional, Mapping, Sequence, TypedDict, cast
from typing_extensions import NotRequired, override

from parlant.core.async_utils import Stopwatch
from parlant.core.common import Version
from parlant.core.nlp.common import ModelSize, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.meter import DurationHistogram, Meter
from parlant.core.nlp.tokenization import (
    EstimatingTokenizer,
    ZeroEstimatingTokenizer,
    estimate_tokens,
)
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
)
from parlant.core.tracer import Tracer
from parlant.core.usage_reporter import UsageReporter


class EmbedderHints(TypedDict, total=False):
    model_size: NotRequired[ModelSize]


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of an embedding operation."""

    vectors: Sequence[Sequence[float]]
    usage: UsageInfo = field(
        default_factory=lambda: UsageInfo(input_tokens=0, output_tokens=0, requests=0)
    )


@dataclass
class _EmbeddingCacheEntry:
    """An entry in the embedding LRU cache."""

    text_length: int
    checksum: int
    vector: Sequence[float]
    # The text's input-token count (provider-reported share, or a gpt-5 estimate
    # when the provider reports none) — surfaced as cached_input_tokens on a hit.
    input_tokens: int = 0


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


EmbedderProvider = Callable[[], Awaitable[Embedder]]

# Fraction of an embedder's max_tokens used as the effective chunking budget.
# Tokenizers are estimating by contract, so chunks are sized under the model's
# hard cap to absorb estimation error.
_EMBEDDING_BUDGET_HEADROOM = 0.75


async def chunk_to_embedder_budget(text: str, embedder: Embedder) -> list[str]:
    """Fit ``text`` to ``embedder``'s token budget.

    Returns ``[text]`` verbatim when it fits — the common case, keeping
    embedding-cache keys byte-stable — or whitespace-split chunks each within
    the budget when it doesn't. Callers that score by pooling over query rows
    can treat the chunks as additional rows.
    """
    if not text.strip():
        return []

    budget = max(1, int(embedder.max_tokens * _EMBEDDING_BUDGET_HEADROOM))
    estimate = await embedder.tokenizer.estimate_token_count(text)

    if estimate <= budget:
        return [text]

    words = text.split()
    tokens_per_word = estimate / len(words)
    words_per_chunk = max(int(budget / tokens_per_word), 1)

    return [
        " ".join(words[start : start + words_per_chunk])
        for start in range(0, len(words), words_per_chunk)
    ]


def embedder_id_hash(embedder: Embedder) -> str:
    return hashlib.sha256(embedder.id.encode("utf-8")).hexdigest()[:16]


_EMBED_DURATION_HISTOGRAM: DurationHistogram | None = None


class BaseEmbedder(Embedder):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_name: str,
        usage_reporter: UsageReporter | None = None,
        *,
        provider_name: str,
    ) -> None:
        self.logger = logger
        self.tracer = tracer
        self.meter = meter
        self.model_name = model_name
        self.usage_reporter = usage_reporter
        self._provider_name = provider_name

        # LRU cache: checksum -> cache entry
        self._cache: OrderedDict[int, _EmbeddingCacheEntry] = OrderedDict()
        # Index for fast length-based lookup: length -> set of checksums
        self._cache_length_index: dict[int, set[int]] = {}
        # Single-flight: checksum -> future resolving to (vector, input_tokens) for a
        # text currently being embedded. Concurrent identical embeds await the same
        # future instead of each hitting the provider (kills the recall stampede).
        self._inflight: dict[int, asyncio.Future[tuple[Sequence[float], int]]] = {}

        global _EMBED_DURATION_HISTOGRAM
        if _EMBED_DURATION_HISTOGRAM is None:
            _EMBED_DURATION_HISTOGRAM = meter.create_duration_histogram(
                name="embed",
                description="Duration of embedding requests in milliseconds",
            )

    @property
    def provider_name(self) -> str:
        """The provider this embedder reports usage under (``provider:model``).

        Defaults to the adapter's own provider id; a hand-built instance may be
        given another so that usage reached through a second provider is
        accounted and priced apart.
        """
        return self._provider_name

    @abstractmethod
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        """Embed the texts with the provider.

        Must return exactly one vector per text, in the order of ``texts``.
        """
        ...

    def _compute_checksum(self, text: str) -> int:
        """Compute a fast checksum for the given text."""
        return zlib.crc32(text.encode("utf-8"))

    def _cache_get(self, text: str) -> _EmbeddingCacheEntry | None:
        """Get the cached entry (vector + token count) for the given text.

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
            return entry

        return None

    def _cache_put(self, text: str, vector: Sequence[float], input_tokens: int = 0) -> None:
        """Store an embedding vector (and its token count) in the cache."""
        checksum = self._compute_checksum(text)
        text_length = len(text)

        # If already in cache, just update and move to end
        if checksum in self._cache:
            self._cache[checksum].vector = vector
            self._cache[checksum].input_tokens = input_tokens
            self._cache.move_to_end(checksum)
            return

        # Evict oldest entry if at capacity
        if len(self._cache) >= _EMBEDDING_CACHE_MAX_SIZE:
            oldest_checksum, oldest_entry = self._cache.popitem(last=False)
            checksums = self._cache_length_index[oldest_entry.text_length]
            checksums.discard(oldest_checksum)
            if not checksums:
                del self._cache_length_index[oldest_entry.text_length]

        # Add new entry
        self._cache[checksum] = _EmbeddingCacheEntry(
            text_length=text_length,
            checksum=checksum,
            vector=vector,
            input_tokens=input_tokens,
        )

        # Update length index
        if text_length not in self._cache_length_index:
            self._cache_length_index[text_length] = set()
        self._cache_length_index[text_length].add(checksum)

    @staticmethod
    def _attribute_tokens(texts: Sequence[str], provider_total: int) -> list[int]:
        """Per-text token counts to cache.

        With a provider-reported ``provider_total`` (> 0), distribute it across
        the texts by their gpt-5 estimate proportions so the per-text sum equals
        the authoritative total (single-text batches — the common case — are
        exact). When the provider reports nothing (e.g. Gemini embeddings), fall
        back to the raw gpt-5 estimate per text.
        """
        estimates = [estimate_tokens(text) for text in texts]
        if provider_total <= 0:
            return estimates

        estimate_total = sum(estimates)
        if estimate_total == 0:
            # Degenerate (all-empty) batch: put the whole total on the first text.
            return [provider_total, *([0] * (len(texts) - 1))]

        attributed = [round(provider_total * est / estimate_total) for est in estimates]
        # Correct rounding drift so the per-text sum matches the provider total exactly.
        attributed[0] += provider_total - sum(attributed)
        return attributed

    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        assert _EMBED_DURATION_HISTOGRAM is not None

        # Phase 1 — classify each text, synchronously (no await between reading and
        # registering in-flight, so two concurrent calls can never both lead the same
        # text). A text is either an LRU hit, already in flight (this call coalesces
        # onto it), or new (this call leads it).
        loop = asyncio.get_running_loop()
        vectors: dict[int, Sequence[float]] = {}
        cached_input_tokens = 0  # LRU hits + coalesced awaiters (priced at zero)
        awaiting: list[tuple[int, asyncio.Future[tuple[Sequence[float], int]]]] = []
        leaders: list[tuple[int, str, int, asyncio.Future[tuple[Sequence[float], int]]]] = []

        for i, text in enumerate(texts):
            entry = self._cache_get(text)
            if entry is not None:
                vectors[i] = entry.vector
                cached_input_tokens += entry.input_tokens
                continue
            checksum = self._compute_checksum(text)
            if (existing := self._inflight.get(checksum)) is not None:
                awaiting.append((i, existing))
            else:
                future: asyncio.Future[tuple[Sequence[float], int]] = loop.create_future()
                self._inflight[checksum] = future
                leaders.append((i, text, checksum, future))

        # Phase 2 — embed the texts this call leads (one batched provider request).
        # Resolve each leader's future for any concurrent awaiters; on failure, fail
        # every unresolved future so awaiters don't hang; always drain the in-flight
        # entries so the map holds only currently-embedding texts.
        provider_usage: UsageInfo | None = None
        if leaders:
            leader_texts = [text for _, text, _, _ in leaders]
            leader_checksums = [checksum for _, _, checksum, _ in leaders]
            try:
                async with _EMBED_DURATION_HISTOGRAM.measure(
                    {
                        "class.name": self.__class__.__qualname__,
                        "embedding.model.name": self.model_name,
                        **({"embedding.tag": hints["tag"]} if "tag" in hints else {}),
                    },
                ):
                    start = Stopwatch.start()
                    try:
                        result = await self.do_embed(leader_texts, hints)
                        if len(result.vectors) != len(leader_texts):
                            raise ValueError(
                                f"Embedder '{self.model_name}' returned "
                                f"{len(result.vectors)} vector(s) for {len(leader_texts)} "
                                "text(s); a provider adapter must return exactly one "
                                "vector per text, in order."
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
                    self.tracer.add_event(
                        "embed.request_completed",
                        attributes={
                            "class.name": self.__class__.__qualname__,
                            "model.name": self.model_name,
                            "duration": start.elapsed,
                        },
                    )
                    provider_usage = replace(result.usage, requests=max(result.usage.requests, 1))
                    # Attribute the provider's batch total across the embedded texts and
                    # cache each vector with its token count, so a later hit can report it.
                    miss_tokens = self._attribute_tokens(leader_texts, result.usage.input_tokens)
                    for (idx, text, _checksum, future), vector, tokens in zip(
                        leaders, result.vectors, miss_tokens
                    ):
                        self._cache_put(text, vector, tokens)
                        vectors[idx] = vector
                        if not future.done():
                            future.set_result((vector, tokens))
            except BaseException as exc:
                for _, _, _, future in leaders:
                    if not future.done():
                        future.set_exception(exc)
                raise
            finally:
                for checksum in leader_checksums:
                    self._inflight.pop(checksum, None)

        # Phase 3 — await texts a concurrent call is embedding. These made no provider
        # request of their own, so their tokens count as cached (priced at zero).
        for i, future in awaiting:
            vector, tokens = await future
            vectors[i] = vector
            cached_input_tokens += tokens

        # Phase 4 — assemble usage. Leaders' tokens are freshly-billed input (keeping the
        # provider's cost/extra); LRU hits and coalesced awaiters are cached input tokens.
        if provider_usage is not None:
            usage = replace(
                provider_usage,
                input_tokens=provider_usage.input_tokens + cached_input_tokens,
                cached_input_tokens=provider_usage.cached_input_tokens + cached_input_tokens,
            )
        else:
            # A cache-served call (full LRU hit, or coalesced onto a concurrent embed):
            # no provider call, but it IS a request the system made — count it as one so
            # cache-served calls stay visible in the usage report.
            usage = UsageInfo(
                input_tokens=cached_input_tokens,
                output_tokens=0,
                cached_input_tokens=cached_input_tokens,
                requests=1 if texts else 0,
            )
        if self.usage_reporter is not None and (provider_usage is not None or cached_input_tokens):
            self.usage_reporter.report_usage(
                self.model_name,
                usage,
                provider_name=self.provider_name,
            )

        return EmbeddingResult(
            vectors=[vectors[i] for i in range(len(texts))],
            usage=usage,
        )


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


class EmbedderResultDocument(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    version: Version.String
    vectors: Sequence[Sequence[float]]
    # Per-text input-token counts, surfaced as cached_input_tokens on a hit.
    input_tokens: Sequence[int]


class EmbeddingCache(ABC):
    """An interface for caching embedding results."""

    @abstractmethod
    async def get(
        self,
        embedder: Embedder,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> Optional[EmbeddingResult]:
        pass

    @abstractmethod
    async def set(
        self,
        embedder: Embedder,
        texts: list[str],
        vectors: Sequence[Sequence[float]],
        input_tokens: Sequence[int] = (),
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
        self._collections: dict[str, DocumentCollection[EmbedderResultDocument]] = {}

    async def _document_loader(self, doc: BaseDocument) -> Optional[EmbedderResultDocument]:
        # The embedding cache is a disposable derived artifact — no backward
        # compatibility. Only the current schema is accepted; anything else is
        # dropped and re-embedded on demand.
        if doc["version"] == self.VERSION.to_string():
            return cast(EmbedderResultDocument, doc)

        return None

    async def _get_or_create_collection(
        self,
        embedder: Embedder,
    ) -> DocumentCollection[EmbedderResultDocument]:
        if embedder.id not in self._collections:
            collection = await self._database.get_or_create_collection(
                name=f"embedding_{embedder_id_hash(embedder)}",
                schema=EmbedderResultDocument,
                document_loader=self._document_loader,
            )
            self._collections[embedder.id] = collection

        return self._collections[embedder.id]

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
        input_tokens: Sequence[int],
    ) -> EmbedderResultDocument:
        return EmbedderResultDocument(
            id=ObjectId(id),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            version=self.VERSION.to_string(),
            vectors=vectors,
            input_tokens=input_tokens,
        )

    def _deserialize_result(
        self,
        doc: EmbedderResultDocument,
    ) -> EmbeddingResult:
        # A hit serves the whole cached batch, so all its tokens are cached input
        # tokens (priced at zero). No provider request is made.
        cached_input_tokens = sum(doc.get("input_tokens", []))
        return EmbeddingResult(
            vectors=doc["vectors"],
            usage=UsageInfo(
                input_tokens=cached_input_tokens,
                output_tokens=0,
                cached_input_tokens=cached_input_tokens,
                requests=0,
            ),
        )

    async def get(
        self,
        embedder: Embedder,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> Optional[EmbeddingResult]:
        collection = await self._get_or_create_collection(embedder)

        id = self._generate_id(texts, hints)
        doc = await collection.find_one({"id": {"$eq": ObjectId(id)}})

        if doc:
            return self._deserialize_result(doc)

        return None

    async def set(
        self,
        embedder: Embedder,
        texts: list[str],
        vectors: Sequence[Sequence[float]],
        input_tokens: Sequence[int] = (),
        hints: Mapping[str, Any] = {},
    ) -> None:
        collection = await self._get_or_create_collection(embedder)

        id = self._generate_id(texts, hints)
        doc = self._serialize_result(id, vectors, input_tokens)

        await collection.insert_one(doc)


class NullEmbeddingCache(EmbeddingCache):
    """A no-op embedding cache that does nothing."""

    async def get(
        self,
        embedder: Embedder,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> Optional[EmbeddingResult]:
        return None

    async def set(
        self,
        embedder: Embedder,
        texts: list[str],
        vectors: Sequence[Sequence[float]],
        input_tokens: Sequence[int] = (),
        hints: Mapping[str, Any] = {},
    ) -> None:
        pass
