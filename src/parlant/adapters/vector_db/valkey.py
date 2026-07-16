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

import json
import re
import struct
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Union, cast
from typing_extensions import override, Self, TypeAlias

from glide import (  # type: ignore[import-untyped]
    GlideClient,
    GlideClusterClient,
    GlideClientConfiguration,
    GlideClusterClientConfiguration,
    NodeAddress,
    ServerCredentials,
    ft,
    FtCreateOptions,
    FtSearchOptions,
    FtSearchLimit,
    VectorField,
    VectorFieldAttributesHnsw,
    VectorAlgorithm,
    VectorType,
    TagField,
)
from glide import DistanceMetricType, DataType  # type: ignore[import-untyped]
from glide_shared.exceptions import (  # type: ignore[import-untyped]
    ClosingError,
    ConnectionError as GlideConnectionError,
    RequestError,
    TimeoutError as GlideTimeoutError,
)

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCacheProvider,
)
from parlant.core.persistence.common import Where, ensure_is_total, matches_filters
from parlant.core.persistence.vector_database import (
    BaseDocument,
    BaseVectorCollection,
    DeleteResult,
    InsertResult,
    SimilarDocumentResult,
    UpdateResult,
    VectorDatabase,
    TDocument,
)
from parlant.core.tracer import Tracer


ValkeyClient: TypeAlias = Union[GlideClient, GlideClusterClient]

_METADATA_KEY = "parlant:__metadata__"


def _escape_tag_value(value: str) -> str:
    """Escape special characters in a TAG filter value for Valkey Search."""
    special = r",.<>{}[]\"':;!@#$%^&*()-+=~| "
    result = []
    for ch in value:
        if ch in special:
            result.append("\\")
        result.append(ch)
    return "".join(result)


def _convert_where_to_filter_expr(where: Where) -> str:
    """Convert Parlant's Where grammar to a Valkey Search filter expression."""
    if not where:
        return "*"

    first_key = next(iter(where.keys()))

    if first_key == "$and":
        parts = [_convert_where_to_filter_expr(cast(Where, sub)) for sub in where["$and"]]
        return " ".join(f"({p})" for p in parts)

    if first_key == "$or":
        parts = [_convert_where_to_filter_expr(cast(Where, sub)) for sub in where["$or"]]
        return "(" + " | ".join(f"({p})" for p in parts) + ")"

    # Field-level conditions
    _VALID_FIELD_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    clauses: list[str] = []
    for field_name, field_filter in where.items():
        if not isinstance(field_filter, dict):
            continue
        if not _VALID_FIELD_NAME.match(field_name):
            raise ValueError(f"Invalid field name in filter: {field_name!r}")
        for operator, value in field_filter.items():
            if operator == "$eq":
                if isinstance(value, (int, float)):
                    clauses.append(f"@{field_name}:[{value} {value}]")
                else:
                    clauses.append(f"@{field_name}:{{{_escape_tag_value(str(value))}}}")
            elif operator == "$ne":
                if isinstance(value, (int, float)):
                    clauses.append(f"-@{field_name}:[{value} {value}]")
                else:
                    clauses.append(f"-@{field_name}:{{{_escape_tag_value(str(value))}}}")
            elif operator == "$gt":
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Range operator {operator} requires numeric value, got {type(value).__name__}")
                clauses.append(f"@{field_name}:[({value} +inf]")
            elif operator == "$gte":
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Range operator {operator} requires numeric value, got {type(value).__name__}")
                clauses.append(f"@{field_name}:[{value} +inf]")
            elif operator == "$lt":
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Range operator {operator} requires numeric value, got {type(value).__name__}")
                clauses.append(f"@{field_name}:[-inf ({value}]")
            elif operator == "$lte":
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Range operator {operator} requires numeric value, got {type(value).__name__}")
                clauses.append(f"@{field_name}:[-inf {value}]")
            elif operator == "$in":
                escaped = "|".join(_escape_tag_value(str(v)) for v in value)
                clauses.append(f"@{field_name}:{{{escaped}}}")
            elif operator == "$nin":
                escaped = "|".join(_escape_tag_value(str(v)) for v in value)
                clauses.append(f"-@{field_name}:{{{escaped}}}")
            else:
                raise ValueError(f"Unsupported filter operator: {operator}")

    if not clauses:
        return "*"
    return " ".join(clauses)


def _prefix_for(collection_name: str, embedder_type_name: str) -> str:
    return f"parlant:{collection_name}:{embedder_type_name}:"


def _index_name_for(collection_name: str, embedder_type_name: str) -> str:
    return f"parlant_{collection_name}_{embedder_type_name}"


def _doc_key(prefix: str, doc_id: str) -> str:
    return f"{prefix}{doc_id}"


class ValkeyVectorDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        url: str = "valkey://localhost:6379",
        tls: bool = False,
        password: Optional[str] = None,
        cluster_mode: bool = False,
        embedder_factory: Optional[EmbedderFactory] = None,
        embedding_cache_provider: Optional[EmbeddingCacheProvider] = None,
        request_timeout: int = 5000,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._url = url
        self._tls = tls
        self._password = password
        self._cluster_mode = cluster_mode
        self._embedder_factory = embedder_factory
        self._embedding_cache_provider = embedding_cache_provider
        self._request_timeout = request_timeout
        self._client: Optional[ValkeyClient] = None
        self._collections: dict[str, ValkeyVectorCollection[BaseDocument]] = {}

    async def __aenter__(self) -> Self:
        host, port = self._parse_url(self._url)
        addresses = [NodeAddress(host, port)]
        creds = ServerCredentials(self._password) if self._password else None

        try:
            if self._cluster_mode:
                config = GlideClusterClientConfiguration(
                    addresses=addresses,
                    use_tls=self._tls,
                    request_timeout=self._request_timeout,
                    credentials=creds,
                )
                self._client = await GlideClusterClient.create(config)
            else:
                config_standalone = GlideClientConfiguration(
                    addresses=addresses,
                    use_tls=self._tls,
                    request_timeout=self._request_timeout,
                    credentials=creds,
                )
                self._client = await GlideClient.create(config_standalone)
        except (GlideConnectionError, GlideTimeoutError, ClosingError) as e:
            self._logger.error(f"Failed to connect to Valkey at {host}:{port}: {e}")
            raise RuntimeError(f"Valkey connection failed: {e}") from e
        except RequestError as e:
            self._logger.error(f"Valkey request error during connection: {e}")
            raise RuntimeError(f"Valkey authentication or request error: {e}") from e

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        self._collections.clear()
        if self._client is not None:
            await self._client.close()
            self._client = None

    @staticmethod
    def _parse_url(url: str) -> tuple[str, int]:
        """Parse valkey://host:port or just host:port."""
        url = url.replace("valkey://", "").replace("redis://", "")
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            return host, int(port_str)
        return url, 6379

    @property
    def _connected_client(self) -> ValkeyClient:
        if self._client is None:
            raise RuntimeError("ValkeyVectorDatabase not connected; use 'async with' context.")
        return self._client

    async def _index_exists(self, index_name: str) -> bool:
        existing = await ft.list(self._connected_client)
        names = {i.decode() if isinstance(i, bytes) else str(i) for i in (existing or [])}
        return index_name in names

    async def _create_index(
        self,
        index_name: str,
        prefix: str,
        dimensions: int,
    ) -> None:
        schema = [
            TagField("id", separator="|"),
            TagField("version", separator="|"),
            TagField("checksum", separator="|"),
            VectorField(
                "content_vector",
                VectorAlgorithm.HNSW,
                VectorFieldAttributesHnsw(
                    dimensions=dimensions,
                    distance_metric=DistanceMetricType.COSINE,
                    type=VectorType.FLOAT32,
                ),
            ),
        ]
        await ft.create(
            self._connected_client,
            index_name,
            schema,
            FtCreateOptions(data_type=DataType.HASH, prefixes=[prefix]),
        )

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> ValkeyVectorCollection[TDocument]:
        if self._embedder_factory is None:
            raise RuntimeError("embedder_factory is required for collection operations")
        if self._embedding_cache_provider is None:
            raise RuntimeError("embedding_cache_provider is required for collection operations")

        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedder = self._embedder_factory.create_embedder(embedder_type)
        embedder_type_name = embedder_type.__name__
        index_name = _index_name_for(name, embedder_type_name)
        prefix = _prefix_for(name, embedder_type_name)

        if await self._index_exists(index_name):
            raise ValueError(f'Collection "{name}" already exists.')

        await self._create_index(index_name, prefix, embedder.dimensions)

        collection: ValkeyVectorCollection[TDocument] = ValkeyVectorCollection(
            logger=self._logger,
            tracer=self._tracer,
            client=self._connected_client,
            index_name=index_name,
            prefix=prefix,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )
        self._collections[name] = collection  # type: ignore[assignment]
        return collection

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> ValkeyVectorCollection[TDocument]:
        # TODO: document_loader is accepted for interface conformance but not yet used.
        # Valkey assumes documents are already embedded on insert. Implement loader support
        # if schema migration or re-embedding of existing documents is needed.
        if self._embedder_factory is None:
            raise RuntimeError("embedder_factory is required for collection operations")
        if self._embedding_cache_provider is None:
            raise RuntimeError("embedding_cache_provider is required for collection operations")

        if cached := self._collections.get(name):
            return cast(ValkeyVectorCollection[TDocument], cached)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        embedder_type_name = embedder_type.__name__
        index_name = _index_name_for(name, embedder_type_name)
        prefix = _prefix_for(name, embedder_type_name)

        if not await self._index_exists(index_name):
            raise ValueError(f'Valkey collection "{name}" not found.')

        collection: ValkeyVectorCollection[TDocument] = ValkeyVectorCollection(
            logger=self._logger,
            tracer=self._tracer,
            client=self._connected_client,
            index_name=index_name,
            prefix=prefix,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )
        self._collections[name] = collection  # type: ignore[assignment]
        return collection

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> ValkeyVectorCollection[TDocument]:
        # TODO: document_loader not yet used — see get_collection comment.
        if self._embedder_factory is None:
            raise RuntimeError("embedder_factory is required for collection operations")
        if self._embedding_cache_provider is None:
            raise RuntimeError("embedding_cache_provider is required for collection operations")

        if cached := self._collections.get(name):
            return cast(ValkeyVectorCollection[TDocument], cached)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        embedder_type_name = embedder_type.__name__
        index_name = _index_name_for(name, embedder_type_name)
        prefix = _prefix_for(name, embedder_type_name)

        if not await self._index_exists(index_name):
            await self._create_index(index_name, prefix, embedder.dimensions)

        collection: ValkeyVectorCollection[TDocument] = ValkeyVectorCollection(
            logger=self._logger,
            tracer=self._tracer,
            client=self._connected_client,
            index_name=index_name,
            prefix=prefix,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )
        self._collections[name] = collection  # type: ignore[assignment]
        return collection

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name in self._collections:
            collection = self._collections[name]
            index_name = collection._index_name
            prefix = collection._prefix
        else:
            # Check server state for collections created by other processes
            existing = await ft.list(self._connected_client)
            index_names = [
                i.decode() if isinstance(i, bytes) else str(i) for i in (existing or [])
            ]
            matched = [n for n in index_names if n.startswith(f"parlant_{name}_")]
            if not matched:
                raise ValueError(f'Collection "{name}" not found.')
            index_name = matched[0]
            # Derive prefix from index name: parlant_{name}_{embedder} -> parlant:{name}:{embedder}:
            # Use rsplit to handle collection names containing underscores
            without_prefix = index_name[len("parlant_"):]
            name_part, embedder_part = without_prefix.rsplit("_", 1)
            prefix = f"parlant:{name_part}:{embedder_part}:"

        # Drop the FT index
        await ft.dropindex(self._connected_client, index_name)

        # Delete all document keys with this prefix using SCAN + UNLINK
        cursor = "0"
        while True:
            result = await self._connected_client.custom_command(
                ["SCAN", cursor, "MATCH", f"{prefix}*", "COUNT", "100"]
            )
            if isinstance(result, list) and len(result) == 2:
                cursor_val = result[0]
                keys = result[1]
                cursor = cursor_val.decode() if isinstance(cursor_val, bytes) else str(cursor_val)
                if keys:
                    key_list = [k.decode() if isinstance(k, bytes) else str(k) for k in keys]
                    # Chunk deletes to avoid oversized commands (SCAN COUNT is a hint, not a limit)
                    for i in range(0, len(key_list), 100):
                        batch = key_list[i : i + 100]
                        await self._connected_client.unlink(batch)
                if cursor == "0":
                    break
            else:
                break

        self._collections.pop(name, None)

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        serialized = json.dumps(value)
        await self._connected_client.hset(_METADATA_KEY, {key: serialized})

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        await self._connected_client.hdel(_METADATA_KEY, [key])

    @override
    async def read_metadata(
        self,
    ) -> Mapping[str, JSONSerializable]:
        raw = await self._connected_client.hgetall(_METADATA_KEY)
        if not raw:
            return {}
        result: dict[str, JSONSerializable] = {}
        for k, v in raw.items():
            key_str = k.decode() if isinstance(k, bytes) else str(k)
            val_str = v.decode() if isinstance(v, bytes) else str(v)
            result[key_str] = json.loads(val_str)
        return result


class ValkeyVectorCollection(BaseVectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        client: ValkeyClient,
        index_name: str,
        prefix: str,
        schema: type[TDocument],
        embedder: Embedder,
        embedding_cache_provider: EmbeddingCacheProvider,
    ) -> None:
        super().__init__(tracer)
        self._logger = logger
        self._client = client
        self._index_name = index_name
        self._prefix = prefix
        self._schema = schema
        self._embedder = embedder
        self._embedding_cache_provider = embedding_cache_provider
        # NOTE: ReaderWriterLock is in-process only (asyncio-based). In multi-worker
        # deployments, each process gets its own lock instance. Valkey's atomic commands
        # (HSET, DEL) provide basic safety, but compound read-modify-write operations
        # (e.g., update_one: find → merge → write) are NOT atomic across processes.
        self._lock = ReaderWriterLock()

    def _doc_to_hash_fields(self, document: TDocument) -> dict[str, bytes | str]:
        """Convert a typed document to hash fields for HSET."""
        fields: dict[str, bytes | str] = {}
        for key, value in document.items():
            if value is None:
                continue
            # Preserve numeric types as their string representation (Valkey stores all as strings,
            # but numeric strings enable NUMERIC field indexing and proper round-trip)
            fields[key] = str(value)
        return fields

    def _hash_to_doc(self, raw: Mapping[Any, Any]) -> TDocument:
        """Convert raw hash fields back to a typed document."""
        doc: dict[str, Any] = {}
        for k, v in raw.items():
            key_str = k.decode() if isinstance(k, bytes) else str(k)
            if key_str in ("content_vector", "score"):
                continue  # Skip binary vector field and synthetic KNN score
            val_str = v.decode() if isinstance(v, bytes) else str(v)
            # Attempt numeric coercion for round-trip fidelity
            doc[key_str] = self._coerce_value(val_str)
        return cast(TDocument, doc)

    @staticmethod
    def _coerce_value(val: str) -> Any:
        """Attempt to restore original type from string representation."""
        # Try int first (more specific)
        try:
            int_val = int(val)
            # Only coerce if the string is exactly the int representation
            # (avoids "01" -> 1 or "1.0" -> 1)
            if str(int_val) == val:
                return int_val
        except ValueError:
            pass
        # Try float
        try:
            float_val = float(val)
            if str(float_val) == val:
                return float_val
        except ValueError:
            pass
        return val

    async def _embed_content(self, content: str) -> bytes:
        """Embed content text and return packed vector bytes."""
        cache = self._embedding_cache_provider()
        cached = await cache.get(
            embedder_type=type(self._embedder),
            texts=[content],
        )
        if cached:
            vectors = list(cached.vectors)
        else:
            embed_result = await self._embedder.embed([content])
            vectors = list(embed_result.vectors)
            await cache.set(
                embedder_type=type(self._embedder),
                texts=[content],
                vectors=vectors,
            )

        if not vectors or len(vectors[0]) == 0:
            raise ValueError(f"Empty embedding for: {content[:50]}...")

        return struct.pack(f"{len(vectors[0])}f", *vectors[0])

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        try:
            async with self._lock.reader_lock:
                if self._has_range_operators(filters):
                    return await self._scan_and_filter(filters)
                return await self._search_with_filter(filters)
        except (GlideConnectionError, GlideTimeoutError, ClosingError) as e:
            self._logger.error(f"Valkey connection error during find: {e}")
            raise RuntimeError(f"Valkey unavailable: {e}") from e
        except RequestError as e:
            self._logger.error(f"Valkey request error during find: {e}")
            raise RuntimeError(f"Valkey search failed: {e}") from e

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        try:
            async with self._lock.reader_lock:
                if self._has_range_operators(filters):
                    docs = await self._scan_and_filter(filters)
                    return docs[0] if docs else None
                docs = await self._search_with_filter(filters, limit=1)
                return docs[0] if docs else None
        except (GlideConnectionError, GlideTimeoutError, ClosingError) as e:
            self._logger.error(f"Valkey connection error during find_one: {e}")
            raise RuntimeError(f"Valkey unavailable: {e}") from e
        except RequestError as e:
            self._logger.error(f"Valkey request error during find_one: {e}")
            raise RuntimeError(f"Valkey search failed: {e}") from e

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        content = document["content"]
        vector_bytes = await self._embed_content(content)

        try:
            async with self._lock.writer_lock:
                key = _doc_key(self._prefix, str(document["id"]))
                fields = self._doc_to_hash_fields(document)
                fields["content_vector"] = vector_bytes
                await self._client.hset(key, fields)
        except (GlideConnectionError, GlideTimeoutError, ClosingError) as e:
            self._logger.error(f"Valkey connection error during insert_one: {e}")
            raise RuntimeError(f"Valkey unavailable: {e}") from e
        except RequestError as e:
            self._logger.error(f"Valkey request error during insert_one: {e}")
            raise RuntimeError(f"Valkey write failed: {e}") from e

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        try:
            async with self._lock.writer_lock:
                # Find existing document
                existing = await self._find_one_internal(filters)

                if existing:
                    # Merge
                    updated: dict[str, Any] = {**existing}
                    updated.update(params)

                    content = updated.get("content", "")
                    vector_bytes = await self._embed_content(content)
                    key = _doc_key(self._prefix, str(updated["id"]))
                    fields = self._doc_to_hash_fields(cast(TDocument, updated))
                    fields["content_vector"] = vector_bytes
                    await self._client.hset(key, fields)

                    return UpdateResult(
                        acknowledged=True,
                        matched_count=1,
                        modified_count=1,
                        updated_document=cast(TDocument, updated),
                    )

                elif upsert:
                    ensure_is_total(params, self._schema)

                    content = params["content"]
                    vector_bytes = await self._embed_content(content)
                    key = _doc_key(self._prefix, str(params["id"]))
                    fields = self._doc_to_hash_fields(params)
                    fields["content_vector"] = vector_bytes
                    await self._client.hset(key, fields)

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
        except (GlideConnectionError, GlideTimeoutError, ClosingError) as e:
            self._logger.error(f"Valkey connection error during update_one: {e}")
            raise RuntimeError(f"Valkey unavailable: {e}") from e
        except RequestError as e:
            self._logger.error(f"Valkey request error during update_one: {e}")
            raise RuntimeError(f"Valkey write failed: {e}") from e

    @override
    async def delete_one(
        self,
        filters: Where,
    ) -> DeleteResult[TDocument]:
        try:
            async with self._lock.writer_lock:
                existing = await self._find_one_internal(filters)

                if existing:
                    key = _doc_key(self._prefix, str(existing["id"]))
                    await self._client.unlink([key])

                    return DeleteResult(
                        acknowledged=True,
                        deleted_count=1,
                        deleted_document=cast(TDocument, existing),
                    )

                return DeleteResult(
                    acknowledged=True,
                    deleted_count=0,
                    deleted_document=None,
                )
        except (GlideConnectionError, GlideTimeoutError, ClosingError) as e:
            self._logger.error(f"Valkey connection error during delete_one: {e}")
            raise RuntimeError(f"Valkey unavailable: {e}") from e
        except RequestError as e:
            self._logger.error(f"Valkey request error during delete_one: {e}")
            raise RuntimeError(f"Valkey write failed: {e}") from e

    @override
    async def do_find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
        hints: Mapping[str, Any] = {},  # noqa: B006 — matches base class signature
    ) -> Sequence[SimilarDocumentResult[TDocument]]:
        try:
            async with self._lock.reader_lock:
                embed_result = await self._embedder.embed([query], hints or {})
                query_vectors = list(embed_result.vectors)

                if not query_vectors or len(query_vectors[0]) == 0:
                    self._logger.warning(f"Empty embedding generated for query: {query}")
                    return []

                vector_bytes = struct.pack(f"{len(query_vectors[0])}f", *query_vectors[0])

                filter_expr = _convert_where_to_filter_expr(filters)
                if filter_expr == "*":
                    knn_query = f"*=>[KNN {k} @content_vector $vector AS score]"
                else:
                    knn_query = f"({filter_expr})=>[KNN {k} @content_vector $vector AS score]"

                results = await ft.search(
                    client=self._client,
                    index_name=self._index_name,
                    query=knn_query,
                    options=FtSearchOptions(params={"vector": vector_bytes}),
                )

                # ft.search returns [count: int, docs: dict[bytes, dict[bytes, bytes]]]
                count = results[0]
                if count == 0:
                    return []

                similar: list[SimilarDocumentResult[TDocument]] = []
                for _key, fields in results[1].items():
                    score_raw = fields.get(b"score") or fields.get("score")
                    if score_raw is not None:
                        score_str = (
                            score_raw.decode() if isinstance(score_raw, bytes) else str(score_raw)
                        )
                        distance = float(score_str)
                    else:
                        distance = 0.0

                    doc = self._hash_to_doc(fields)
                    similar.append(SimilarDocumentResult(document=doc, distance=distance))

                return similar
        except (GlideConnectionError, GlideTimeoutError, ClosingError) as e:
            self._logger.error(f"Valkey connection error during similarity search: {e}")
            raise RuntimeError(f"Valkey unavailable: {e}") from e
        except RequestError as e:
            self._logger.error(f"Valkey request error during similarity search: {e}")
            raise RuntimeError(f"Valkey search failed: {e}") from e

    async def _find_one_internal(self, filters: Where) -> Optional[dict[str, Any]]:
        """Internal find_one without lock (caller must hold lock)."""
        docs = await self._search_with_filter(filters, limit=1)
        return dict(docs[0]) if docs else None

    async def _search_with_filter(
        self, filters: Where, limit: Optional[int] = None
    ) -> list[TDocument]:
        """Use FT.SEARCH with the index to find documents matching filters."""
        filter_expr = _convert_where_to_filter_expr(filters)

        options: Optional[FtSearchOptions] = None
        if limit is not None:
            options = FtSearchOptions(limit=FtSearchLimit(offset=0, count=limit))

        results = await ft.search(
            client=self._client,
            index_name=self._index_name,
            query=filter_expr,
            options=options,
        )

        # ft.search returns [count: int, docs: dict[bytes, dict[bytes, bytes]]]
        count = results[0]
        if count == 0:
            return []

        docs: list[TDocument] = []
        for _key, fields in results[1].items():
            doc = self._hash_to_doc(fields)
            docs.append(doc)
        return docs

    @staticmethod
    def _has_range_operators(filters: Where) -> bool:
        """Check if filters contain range operators that TAG fields can't handle."""
        _RANGE_OPS = {"$gt", "$gte", "$lt", "$lte"}
        if not filters:
            return False
        first_key = next(iter(filters.keys()))
        if first_key == "$and":
            return any(
                ValkeyVectorCollection._has_range_operators(cast(Where, sub))
                for sub in filters["$and"]
            )
        if first_key == "$or":
            return any(
                ValkeyVectorCollection._has_range_operators(cast(Where, sub))
                for sub in filters["$or"]
            )
        for field_filter in filters.values():
            if isinstance(field_filter, dict):
                if any(op in _RANGE_OPS for op in field_filter):
                    return True
        return False

    async def _scan_and_filter(self, filters: Where) -> list[TDocument]:
        """Fallback for queries with range operators: scan keys and filter in memory."""
        docs: list[TDocument] = []
        cursor = "0"
        while True:
            result = await self._client.custom_command(
                ["SCAN", cursor, "MATCH", f"{self._prefix}*", "COUNT", "100"]
            )
            if isinstance(result, list) and len(result) == 2:
                cursor_val = result[0]
                keys = result[1]
                cursor = cursor_val.decode() if isinstance(cursor_val, bytes) else str(cursor_val)
                if keys:
                    for key in keys:
                        key_str = key.decode() if isinstance(key, bytes) else str(key)
                        raw = await self._client.hgetall(key_str)
                        if raw:
                            doc = self._hash_to_doc(raw)
                            if matches_filters(filters, doc):
                                docs.append(doc)
                if cursor == "0":
                    break
            else:
                break
        return docs
