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

import hashlib
import json
from typing import Any, Awaitable, Callable, Generic, Mapping, Optional, Sequence, cast
from typing_extensions import override, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCacheProvider,
)
from parlant.core.persistence.common import Where, ensure_is_total
from parlant.core.persistence.vector_database import (
    BaseDocument,
    BaseVectorCollection,
    DeleteResult,
    InsertResult,
    SimilarDocumentResult,
    TDocument,
    UpdateResult,
    VectorDatabase,
)
from parlant.core.tracer import Tracer


def _compute_checksum(content: str) -> str:
    """Compute a SHA256 checksum of the content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _version_to_str(version: Any) -> Optional[str]:
    """Convert version to string for TEXT column storage."""
    if version is None:
        return None
    return str(version)


class _WhereTranslator:
    """Translates MongoDB-style Where filters to PostgreSQL SQL."""

    def __init__(self, param_offset: int = 0) -> None:
        self._params: list[Any] = []
        self._param_counter = param_offset

    @property
    def params(self) -> list[Any]:
        return self._params

    def _next_param(self, value: Any) -> str:
        self._param_counter += 1
        # JSONB text extraction (->>) returns TEXT, so convert all values to strings
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, (int, float)):
            value = str(value)
        self._params.append(value)
        return f"${self._param_counter}"

    def _column_expr(self, field: str) -> str:
        """All document fields are stored in the data JSONB column."""
        return f"data->>'{field}'"

    def render(self, where: Where) -> str:
        if not where:
            return "TRUE"

        first_key = next(iter(where.keys()))

        if first_key == "$and":
            sub_filters = cast(list[Where], where["$and"])
            clauses = [self.render(f) for f in sub_filters]
            return f"({' AND '.join(clauses)})"

        if first_key == "$or":
            sub_filters = cast(list[Where], where["$or"])
            clauses = [self.render(f) for f in sub_filters]
            return f"({' OR '.join(clauses)})"

        parts: list[str] = []
        for field, operators in where.items():
            if field in ("$and", "$or"):
                continue

            if isinstance(operators, dict):
                for op, value in operators.items():
                    col = self._column_expr(field)
                    if op == "$eq":
                        parts.append(f"{col} = {self._next_param(value)}")
                    elif op == "$ne":
                        parts.append(f"{col} != {self._next_param(value)}")
                    elif op == "$gt":
                        parts.append(f"{col} > {self._next_param(value)}")
                    elif op == "$gte":
                        parts.append(f"{col} >= {self._next_param(value)}")
                    elif op == "$lt":
                        parts.append(f"{col} < {self._next_param(value)}")
                    elif op == "$lte":
                        parts.append(f"{col} <= {self._next_param(value)}")
                    elif op == "$in":
                        if not value:
                            parts.append("FALSE")
                        else:
                            placeholders = ", ".join(self._next_param(v) for v in value)
                            parts.append(f"{col} IN ({placeholders})")
                    elif op == "$nin":
                        if not value:
                            parts.append("TRUE")
                        else:
                            placeholders = ", ".join(self._next_param(v) for v in value)
                            parts.append(f"{col} NOT IN ({placeholders})")

        return " AND ".join(parts) if parts else "TRUE"


class PostgresVectorDatabase(VectorDatabase):
    """PostgreSQL + pgvector implementation of VectorDatabase."""

    def __init__(
        self,
        connection_string: str,
        logger: Logger,
        tracer: Tracer,
        *,
        table_prefix: str = "parlant_vec_",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
        embedder_factory: Optional[EmbedderFactory] = None,
        embedding_cache_provider: Optional[EmbeddingCacheProvider] = None,
    ) -> None:
        self._connection_string = connection_string
        self._logger = logger
        self._tracer = tracer
        self._table_prefix = table_prefix
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._embedder_factory = embedder_factory
        self._embedding_cache_provider = embedding_cache_provider

        self._pool: Any = None  # asyncpg.Pool
        self._collections: dict[str, PostgresVectorCollection[BaseDocument]] = {}
        self._initialized_tables: set[str] = set()

    async def __aenter__(self) -> Self:
        import asyncpg  # type: ignore[import-untyped]
        from pgvector.asyncpg import register_vector  # type: ignore[import-untyped]

        async def init_connection(conn: Any) -> None:
            # Register vector type for pgvector
            await register_vector(conn)
            # Register JSON codec for JSONB columns
            # This ensures dicts are serialized to JSON on insert and decoded on read
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )

        self._pool = await asyncpg.create_pool(
            self._connection_string,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
            init=init_connection,
        )

        # Ensure pgvector extension is enabled
        await self._execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create metadata table
        await self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_prefix}metadata (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL
            )
            """
        )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        self._collections.clear()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _execute(
        self,
        sql: str,
        params: Optional[list[Any]] = None,
        fetch: str = "none",
    ) -> Any:
        """Execute SQL with optional parameter binding."""
        assert self._pool is not None, "Pool must be initialized"

        async with self._pool.acquire() as conn:
            if fetch == "one":
                return await conn.fetchrow(sql, *(params or []))
            elif fetch == "all":
                return await conn.fetch(sql, *(params or []))
            else:
                await conn.execute(sql, *(params or []))
                return None

    def _table_name(self, name: str, suffix: str = "") -> str:
        """Generate a table name with prefix and optional suffix."""
        base = f"{self._table_prefix}{name}"
        if suffix:
            base = f"{base}_{suffix}"
        return base

    def _quote_table(self, name: str) -> str:
        """Quote a table name for SQL."""
        return f'"{name}"'

    async def _ensure_tables(
        self,
        name: str,
        embedder_type: type[Embedder],
    ) -> tuple[str, str]:
        """Create tables if they don't exist. Returns (unembedded_table, embedded_table)."""
        assert self._embedder_factory is not None, "Embedder factory must be provided"

        unembedded_table = self._table_name(name, "unembedded")
        embedded_table = self._table_name(name, embedder_type.__name__)

        if unembedded_table in self._initialized_tables:
            return unembedded_table, embedded_table

        # Create unembedded table (source of truth)
        await self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._quote_table(unembedded_table)} (
                id TEXT PRIMARY KEY,
                version TEXT,
                content TEXT NOT NULL,
                checksum TEXT NOT NULL,
                data JSONB NOT NULL
            )
            """
        )

        # Create index on checksum for efficient sync
        await self._execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{unembedded_table}_checksum
            ON {self._quote_table(unembedded_table)} (checksum)
            """
        )

        # Get embedder dimensions
        embedder = self._embedder_factory.create_embedder(embedder_type)
        dimensions = embedder.dimensions

        # Create embedded table with vector column
        await self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._quote_table(embedded_table)} (
                id TEXT PRIMARY KEY,
                version TEXT,
                content TEXT NOT NULL,
                checksum TEXT NOT NULL,
                data JSONB NOT NULL,
                embedding vector({dimensions})
            )
            """
        )

        # Create vector index for cosine similarity search
        # pgvector indexes (HNSW, IVFFlat) are limited to 2000 dimensions
        # For higher dimensions (like OpenAI's 3072-dim embeddings), skip the index
        # and rely on sequential scans. Consider using text-embedding-3-small (1536 dims)
        # for better performance with indexed searches.
        if dimensions <= 2000:
            await self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{embedded_table}_embedding
                ON {self._quote_table(embedded_table)}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        else:
            self._logger.warning(
                f"Skipping vector index for {embedded_table}: "
                f"{dimensions} dimensions exceeds pgvector's 2000 dimension limit. "
                "Consider using a smaller embedding model for better search performance."
            )

        # Create GIN index for JSONB data
        await self._execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{embedded_table}_data
            ON {self._quote_table(embedded_table)} USING GIN (data)
            """
        )

        # Create failed migrations table
        failed_table = self._table_name(name, "failed_migrations")
        await self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._quote_table(failed_table)} (
                id TEXT,
                data JSONB NOT NULL
            )
            """
        )

        self._initialized_tables.add(unembedded_table)
        return unembedded_table, embedded_table

    async def _load_and_sync_documents(
        self,
        name: str,
        unembedded_table: str,
        embedded_table: str,
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        """Load documents from unembedded, migrate them, and sync to embedded."""
        assert self._embedder_factory is not None, "Embedder factory must be provided"

        embedder = self._embedder_factory.create_embedder(embedder_type)
        failed_table = self._table_name(name, "failed_migrations")

        # Get all documents from unembedded table
        rows = await self._execute(
            f"SELECT id, data FROM {self._quote_table(unembedded_table)}",
            fetch="all",
        )

        if not rows:
            return

        for row in rows:
            doc_id = row["id"]
            doc_data = row["data"]

            try:
                prospective_doc = cast(BaseDocument, doc_data)
                loaded_doc = await document_loader(prospective_doc)

                if loaded_doc:
                    if loaded_doc != prospective_doc:
                        # Update the unembedded table with migrated doc
                        content = cast(str, loaded_doc.get("content", ""))
                        checksum = _compute_checksum(content)
                        await self._execute(
                            f"""
                            UPDATE {self._quote_table(unembedded_table)}
                            SET version = $1, content = $2, checksum = $3, data = $4
                            WHERE id = $5
                            """,
                            [
                                _version_to_str(loaded_doc.get("version")),
                                content,
                                checksum,
                                loaded_doc,
                                doc_id,
                            ],
                        )
                else:
                    # Migration failed - store in failed migrations and delete from unembedded
                    self._logger.warning(f'Failed to load document "{doc_id}"')
                    await self._execute(
                        f"""
                        INSERT INTO {self._quote_table(failed_table)} (id, data)
                        VALUES ($1, $2)
                        """,
                        [doc_id, doc_data],
                    )
                    await self._execute(
                        f"DELETE FROM {self._quote_table(unembedded_table)} WHERE id = $1",
                        [doc_id],
                    )

            except Exception as e:
                self._logger.error(f"Failed to load document '{doc_id}': {e}")
                await self._execute(
                    f"""
                    INSERT INTO {self._quote_table(failed_table)} (id, data)
                    VALUES ($1, $2)
                    """,
                    [doc_id, doc_data],
                )

        # Sync embedded table with unembedded
        await self._sync_embedded_table(unembedded_table, embedded_table, embedder)

    async def _sync_embedded_table(
        self,
        unembedded_table: str,
        embedded_table: str,
        embedder: Embedder,
    ) -> None:
        """Sync the embedded table with the unembedded table using checksums."""
        # Get all documents from both tables
        unembedded_rows = await self._execute(
            f"SELECT id, content, checksum, data FROM {self._quote_table(unembedded_table)}",
            fetch="all",
        )
        embedded_rows = await self._execute(
            f"SELECT id, checksum FROM {self._quote_table(embedded_table)}",
            fetch="all",
        )

        unembedded_by_id = {row["id"]: row for row in (unembedded_rows or [])}
        embedded_checksums = {row["id"]: row["checksum"] for row in (embedded_rows or [])}

        # Remove documents from embedded that don't exist in unembedded
        for doc_id in list(embedded_checksums.keys()):
            if doc_id not in unembedded_by_id:
                await self._execute(
                    f"DELETE FROM {self._quote_table(embedded_table)} WHERE id = $1",
                    [doc_id],
                )

        # Update or insert documents
        for doc_id, row in unembedded_by_id.items():
            content = row["content"]
            checksum = row["checksum"]
            data = row["data"]

            # Check if document needs embedding update
            needs_update = doc_id not in embedded_checksums
            needs_reembed = doc_id in embedded_checksums and embedded_checksums[doc_id] != checksum

            if needs_update or needs_reembed:
                # Generate embedding
                if content:
                    embeddings = list((await embedder.embed([content])).vectors)
                    if embeddings and len(embeddings[0]) > 0:
                        embedding = embeddings[0]
                    else:
                        self._logger.warning(f"Empty embedding for document {doc_id}, skipping")
                        continue
                else:
                    self._logger.warning(f"No content for document {doc_id}, skipping")
                    continue

                if needs_update:
                    await self._execute(
                        f"""
                        INSERT INTO {self._quote_table(embedded_table)}
                        (id, version, content, checksum, data, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        [
                            doc_id,
                            _version_to_str(data.get("version")),
                            content,
                            checksum,
                            data,
                            embedding,
                        ],
                    )
                else:
                    await self._execute(
                        f"""
                        UPDATE {self._quote_table(embedded_table)}
                        SET version = $1, content = $2, checksum = $3, data = $4, embedding = $5
                        WHERE id = $6
                        """,
                        [
                            _version_to_str(data.get("version")),
                            content,
                            checksum,
                            data,
                            embedding,
                            doc_id,
                        ],
                    )

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> PostgresVectorCollection[TDocument]:
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )

        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        unembedded_table, embedded_table = await self._ensure_tables(name, embedder_type)
        embedder = self._embedder_factory.create_embedder(embedder_type)

        collection: PostgresVectorCollection[TDocument] = PostgresVectorCollection(
            database=self,
            logger=self._logger,
            tracer=self._tracer,
            name=name,
            schema=schema,
            unembedded_table=unembedded_table,
            embedded_table=embedded_table,
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
    ) -> PostgresVectorCollection[TDocument]:
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )

        if name in self._collections:
            return cast(PostgresVectorCollection[TDocument], self._collections[name])

        unembedded_table = self._table_name(name, "unembedded")

        # Check if collection exists
        result = await self._execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = $1
            )
            """,
            [unembedded_table],
            fetch="one",
        )

        if not result or not result["exists"]:
            raise ValueError(f'Collection "{name}" not found.')

        unembedded_table, embedded_table = await self._ensure_tables(name, embedder_type)
        embedder = self._embedder_factory.create_embedder(embedder_type)

        # Load and sync documents
        await self._load_and_sync_documents(
            name, unembedded_table, embedded_table, embedder_type, document_loader
        )

        collection: PostgresVectorCollection[TDocument] = PostgresVectorCollection(
            database=self,
            logger=self._logger,
            tracer=self._tracer,
            name=name,
            schema=schema,
            unembedded_table=unembedded_table,
            embedded_table=embedded_table,
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
    ) -> PostgresVectorCollection[TDocument]:
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )

        if name in self._collections:
            return cast(PostgresVectorCollection[TDocument], self._collections[name])

        unembedded_table, embedded_table = await self._ensure_tables(name, embedder_type)
        embedder = self._embedder_factory.create_embedder(embedder_type)

        # Load and sync documents
        await self._load_and_sync_documents(
            name, unembedded_table, embedded_table, embedder_type, document_loader
        )

        collection: PostgresVectorCollection[TDocument] = PostgresVectorCollection(
            database=self,
            logger=self._logger,
            tracer=self._tracer,
            name=name,
            schema=schema,
            unembedded_table=unembedded_table,
            embedded_table=embedded_table,
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
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        collection = self._collections[name]
        unembedded_table = collection._unembedded_table
        embedded_table = collection._embedded_table
        failed_table = self._table_name(name, "failed_migrations")

        await self._execute(f"DROP TABLE IF EXISTS {self._quote_table(embedded_table)}")
        await self._execute(f"DROP TABLE IF EXISTS {self._quote_table(unembedded_table)}")
        await self._execute(f"DROP TABLE IF EXISTS {self._quote_table(failed_table)}")

        self._initialized_tables.discard(unembedded_table)
        del self._collections[name]

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        metadata_table = f"{self._table_prefix}metadata"
        await self._execute(
            f"""
            INSERT INTO {self._quote_table(metadata_table)} (key, value)
            VALUES ($1, $2)
            ON CONFLICT (key) DO UPDATE SET value = $2
            """,
            [key, value],
        )

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        metadata_table = f"{self._table_prefix}metadata"
        result = await self._execute(
            f"DELETE FROM {self._quote_table(metadata_table)} WHERE key = $1 RETURNING key",
            [key],
            fetch="one",
        )
        if not result:
            raise ValueError(f'Metadata with key "{key}" not found.')

    @override
    async def read_metadata(
        self,
    ) -> Mapping[str, JSONSerializable]:
        metadata_table = f"{self._table_prefix}metadata"
        rows = await self._execute(
            f"SELECT key, value FROM {self._quote_table(metadata_table)}",
            fetch="all",
        )

        if not rows:
            return {}

        return {row["key"]: row["value"] for row in rows}


class PostgresVectorCollection(Generic[TDocument], BaseVectorCollection[TDocument]):
    """PostgreSQL + pgvector implementation of VectorCollection."""

    def __init__(
        self,
        database: PostgresVectorDatabase,
        logger: Logger,
        tracer: Tracer,
        name: str,
        schema: type[TDocument],
        unembedded_table: str,
        embedded_table: str,
        embedder: Embedder,
        embedding_cache_provider: EmbeddingCacheProvider,
    ) -> None:
        super().__init__(tracer)

        self._database = database
        self._logger = logger
        self._tracer = tracer
        self._name = name
        self._schema = schema
        self._unembedded_table = unembedded_table
        self._embedded_table = embedded_table
        self._embedder = embedder
        self._embedding_cache_provider = embedding_cache_provider

        self._lock = ReaderWriterLock()

    def _quote_table(self, name: str) -> str:
        return f'"{name}"'

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            translator = _WhereTranslator()
            where_clause = translator.render(filters)

            rows = await self._database._execute(
                f"""
                SELECT data FROM {self._quote_table(self._embedded_table)}
                WHERE {where_clause}
                """,
                translator.params,
                fetch="all",
            )

            if not rows:
                return []

            return [cast(TDocument, row["data"]) for row in rows]

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            translator = _WhereTranslator()
            where_clause = translator.render(filters)

            row = await self._database._execute(
                f"""
                SELECT data FROM {self._quote_table(self._embedded_table)}
                WHERE {where_clause}
                LIMIT 1
                """,
                translator.params,
                fetch="one",
            )

            if not row:
                return None

            return cast(TDocument, row["data"])

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        content = cast(str, document.get("content", ""))
        checksum = _compute_checksum(content)

        # Get embedding
        if cached := await self._embedding_cache_provider().get(
            embedder_type=type(self._embedder),
            texts=[content],
        ):
            embeddings = list(cached.vectors)
        else:
            embeddings = list((await self._embedder.embed([content])).vectors)
            await self._embedding_cache_provider().set(
                embedder_type=type(self._embedder),
                texts=[content],
                vectors=embeddings,
            )

        if not embeddings or len(embeddings[0]) == 0:
            raise ValueError(f"Empty embedding generated for document content: {content[:50]}...")

        async with self._lock.writer_lock:
            doc_id = document["id"]

            # Insert into unembedded table
            await self._database._execute(
                f"""
                INSERT INTO {self._quote_table(self._unembedded_table)}
                (id, version, content, checksum, data)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    version = $2, content = $3, checksum = $4, data = $5
                """,
                [
                    doc_id,
                    _version_to_str(document.get("version")),
                    content,
                    checksum,
                    document,
                ],
            )

            # Insert into embedded table
            await self._database._execute(
                f"""
                INSERT INTO {self._quote_table(self._embedded_table)}
                (id, version, content, checksum, data, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    version = $2, content = $3, checksum = $4, data = $5, embedding = $6
                """,
                [
                    doc_id,
                    _version_to_str(document.get("version")),
                    content,
                    checksum,
                    document,
                    embeddings[0],
                ],
            )

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        async with self._lock.writer_lock:
            translator = _WhereTranslator()
            where_clause = translator.render(filters)

            # Find existing document
            row = await self._database._execute(
                f"""
                SELECT data FROM {self._quote_table(self._embedded_table)}
                WHERE {where_clause}
                LIMIT 1
                """,
                translator.params,
                fetch="one",
            )

            if row:
                existing_doc = cast(dict[str, Any], row["data"])
                updated_doc = {**existing_doc, **params}

                content = cast(str, updated_doc.get("content", ""))
                checksum = _compute_checksum(content)

                # Get embedding
                if cached := await self._embedding_cache_provider().get(
                    embedder_type=type(self._embedder),
                    texts=[content],
                ):
                    embeddings = list(cached.vectors)
                else:
                    embeddings = list((await self._embedder.embed([content])).vectors)
                    await self._embedding_cache_provider().set(
                        embedder_type=type(self._embedder),
                        texts=[content],
                        vectors=embeddings,
                    )

                if not embeddings or len(embeddings[0]) == 0:
                    raise ValueError(f"Empty embedding generated for content: {content[:50]}...")

                doc_id = updated_doc["id"]

                # Update unembedded table
                await self._database._execute(
                    f"""
                    UPDATE {self._quote_table(self._unembedded_table)}
                    SET version = $1, content = $2, checksum = $3, data = $4
                    WHERE id = $5
                    """,
                    [
                        _version_to_str(updated_doc.get("version")),
                        content,
                        checksum,
                        updated_doc,
                        doc_id,
                    ],
                )

                # Update embedded table
                await self._database._execute(
                    f"""
                    UPDATE {self._quote_table(self._embedded_table)}
                    SET version = $1, content = $2, checksum = $3, data = $4, embedding = $5
                    WHERE id = $6
                    """,
                    [
                        _version_to_str(updated_doc.get("version")),
                        content,
                        checksum,
                        updated_doc,
                        embeddings[0],
                        doc_id,
                    ],
                )

                return UpdateResult(
                    acknowledged=True,
                    matched_count=1,
                    modified_count=1,
                    updated_document=cast(TDocument, updated_doc),
                )

            elif upsert:
                ensure_is_total(params, self._schema)

                content = cast(str, params.get("content", ""))
                checksum = _compute_checksum(content)

                # Get embedding
                if cached := await self._embedding_cache_provider().get(
                    embedder_type=type(self._embedder),
                    texts=[content],
                ):
                    embeddings = list(cached.vectors)
                else:
                    embeddings = list((await self._embedder.embed([content])).vectors)
                    await self._embedding_cache_provider().set(
                        embedder_type=type(self._embedder),
                        texts=[content],
                        vectors=embeddings,
                    )

                if not embeddings or len(embeddings[0]) == 0:
                    raise ValueError(f"Empty embedding generated for content: {content[:50]}...")

                doc_id = params["id"]

                # Insert into unembedded table
                await self._database._execute(
                    f"""
                    INSERT INTO {self._quote_table(self._unembedded_table)}
                    (id, version, content, checksum, data)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    [
                        doc_id,
                        _version_to_str(params.get("version")),
                        content,
                        checksum,
                        params,
                    ],
                )

                # Insert into embedded table
                await self._database._execute(
                    f"""
                    INSERT INTO {self._quote_table(self._embedded_table)}
                    (id, version, content, checksum, data, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    [
                        doc_id,
                        _version_to_str(params.get("version")),
                        content,
                        checksum,
                        params,
                        embeddings[0],
                    ],
                )

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
        async with self._lock.writer_lock:
            translator = _WhereTranslator()
            where_clause = translator.render(filters)

            # Find existing document
            row = await self._database._execute(
                f"""
                SELECT data FROM {self._quote_table(self._embedded_table)}
                WHERE {where_clause}
                LIMIT 1
                """,
                translator.params,
                fetch="one",
            )

            if not row:
                return DeleteResult(
                    acknowledged=True,
                    deleted_count=0,
                    deleted_document=None,
                )

            deleted_doc = cast(TDocument, row["data"])
            doc_id = deleted_doc["id"]

            # Delete from both tables
            await self._database._execute(
                f"DELETE FROM {self._quote_table(self._unembedded_table)} WHERE id = $1",
                [doc_id],
            )
            await self._database._execute(
                f"DELETE FROM {self._quote_table(self._embedded_table)} WHERE id = $1",
                [doc_id],
            )

            return DeleteResult(
                acknowledged=True,
                deleted_count=1,
                deleted_document=deleted_doc,
            )

    @override
    async def do_find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[SimilarDocumentResult[TDocument]]:
        async with self._lock.reader_lock:
            # Generate query embedding
            query_embeddings = list((await self._embedder.embed([query], hints)).vectors)

            if not query_embeddings or len(query_embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding generated for query: {query}")
                return []

            query_embedding = query_embeddings[0]

            # Build WHERE clause with offset for query embedding ($1) and k ($2)
            translator = _WhereTranslator(param_offset=2)
            where_clause = translator.render(filters)

            # Query with cosine distance
            # pgvector uses <=> for cosine distance (1 - cosine_similarity)
            rows = await self._database._execute(
                f"""
                SELECT data, (embedding <=> $1::vector) AS distance
                FROM {self._quote_table(self._embedded_table)}
                WHERE {where_clause}
                ORDER BY distance ASC
                LIMIT $2
                """,
                [query_embedding, k] + translator.params,
                fetch="all",
            )

            if not rows:
                return []

            self._logger.trace(
                f"Similar documents found\n{json.dumps([r['data'] for r in rows], indent=2)}"
            )

            return [
                SimilarDocumentResult(
                    document=cast(TDocument, row["data"]),
                    distance=float(row["distance"]),
                )
                for row in rows
            ]
