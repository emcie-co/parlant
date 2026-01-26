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

import asyncio
import importlib
import json
import re
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    cast,
)

from typing_extensions import Self

from parlant.core.loggers import Logger
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection, Where, ensure_is_total
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


class PostgresAdapterError(Exception):
    """Raised for recoverable adapter errors."""


_IDENTIFIER_RE = re.compile(r"[^0-9a-z_]")


def _sanitize_identifier(raw: str) -> str:
    """Sanitize identifier for PostgreSQL table/column names."""
    sanitized = _IDENTIFIER_RE.sub("_", raw.lower())
    if not sanitized:
        raise PostgresAdapterError("PostgreSQL identifier cannot be empty")

    if sanitized[0].isdigit():
        return f"_{sanitized}"

    return sanitized


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None

    object_id_type = getattr(ObjectId, "__supertype__", str)
    if isinstance(value, object_id_type):
        return str(value)

    return str(value)


class PostgresDocumentDatabase(DocumentDatabase):
    """PostgreSQL-backed document database using asyncpg."""

    def __init__(
        self,
        connection_string: str,
        logger: Logger,
        *,
        table_prefix: str = "parlant_",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        self._connection_string = connection_string
        self._logger = logger
        self._table_prefix = _sanitize_identifier(table_prefix)
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size

        self._asyncpg_module: Any | None = None
        self._pool: Any | None = None

        self._collections: dict[str, PostgresDocumentCollection[Any]] = {}
        self._initialized: set[str] = set()
        self._init_locks: dict[str, asyncio.Lock] = {}

    async def __aenter__(self) -> Self:
        await self._ensure_pool()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

        return False

    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> PostgresDocumentCollection[TDocument]:
        return await self._get_or_create_initialized_collection(
            name,
            schema,
            document_loader=None,
        )

    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> PostgresDocumentCollection[TDocument]:
        return await self._get_or_create_initialized_collection(
            name,
            schema,
            document_loader=document_loader,
        )

    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> PostgresDocumentCollection[TDocument]:
        return await self.get_collection(name, schema, document_loader)

    async def delete_collection(self, name: str) -> None:
        table = self._table_identifier(name)
        failed_table = self._failed_table_identifier(name)

        await self._execute(f"DROP TABLE IF EXISTS {table}")
        await self._execute(f"DROP TABLE IF EXISTS {failed_table}")
        self._collections.pop(name, None)
        self._initialized.discard(name)

    async def _get_or_create_initialized_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]] | None,
    ) -> PostgresDocumentCollection[TDocument]:
        if name not in self._collections:
            self._collections[name] = PostgresDocumentCollection(
                database=self,
                name=name,
                schema=schema,
                logger=self._logger,
            )

        collection = cast(PostgresDocumentCollection[TDocument], self._collections[name])

        if name in self._initialized:
            return collection

        lock = self._init_locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name in self._initialized:
                return collection

            # Create main table
            create_stmt = f"""
                CREATE TABLE IF NOT EXISTS {collection._table} (
                    id TEXT NOT NULL PRIMARY KEY,
                    version TEXT,
                    creation_utc TEXT,
                    data JSONB NOT NULL
                )
            """
            await self._execute(create_stmt)

            # Create indexes
            await self._execute(
                f"CREATE INDEX IF NOT EXISTS idx_{collection._table_name}_creation_utc "
                f"ON {collection._table} (creation_utc)"
            )
            await self._execute(
                f"CREATE INDEX IF NOT EXISTS idx_{collection._table_name}_cursor "
                f"ON {collection._table} (creation_utc, id)"
            )
            await self._execute(
                f"CREATE INDEX IF NOT EXISTS idx_{collection._table_name}_data "
                f"ON {collection._table} USING GIN (data)"
            )

            # Create failed migrations table
            await self._execute(
                f"""
                CREATE TABLE IF NOT EXISTS {collection._failed_table} (
                    id TEXT,
                    data JSONB NOT NULL
                )
                """
            )

            if document_loader is not None:
                await self._load_documents_with_loader(collection, document_loader)

            self._initialized.add(name)
            return collection

    async def _load_documents_with_loader(
        self,
        collection: PostgresDocumentCollection[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        rows = await self._execute(
            f"SELECT data FROM {collection._table}",
            fetch="all",
        )

        failed: list[BaseDocument] = []
        for row in rows or []:
            doc = collection._row_to_document(row)
            try:
                migrated = await document_loader(doc)
            except Exception as exc:
                self._logger.error(
                    f"Failed to load document '{doc.get('id')}' in collection '{collection._name}': {exc}"
                )
                failed.append(doc)
                continue

            if migrated is None:
                failed.append(doc)
                continue

            if migrated is not doc:
                await collection._replace_document(migrated)

        if failed:
            await collection._persist_failed_documents(failed)
            await collection._delete_documents([doc["id"] for doc in failed if "id" in doc])

    async def _execute(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        fetch: str = "none",
    ) -> Any:
        await self._ensure_pool()
        assert self._pool is not None

        async with self._pool.acquire() as conn:
            if fetch == "all":
                return await conn.fetch(sql, *(params or []))
            elif fetch == "one":
                return await conn.fetchrow(sql, *(params or []))
            else:
                await conn.execute(sql, *(params or []))
                return None

    async def _ensure_pool(self) -> None:
        if self._pool is not None:
            return

        self._import_asyncpg()

        assert self._asyncpg_module is not None
        self._pool = await self._asyncpg_module.create_pool(
            self._connection_string,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
        )

    def _import_asyncpg(self) -> None:
        if self._asyncpg_module is not None:
            return

        try:
            asyncpg_module = importlib.import_module("asyncpg")
        except ImportError as exc:
            raise PostgresAdapterError(
                "PostgreSQL adapter requires asyncpg. Install parlant[postgres]."
            ) from exc

        self._asyncpg_module = asyncpg_module

    def _table_identifier(self, name: str) -> str:
        return f'"{_sanitize_identifier(self._table_prefix + name)}"'

    def _failed_table_identifier(self, name: str) -> str:
        return f'"{_sanitize_identifier(self._table_prefix + name + "_failed_migrations")}"'


class PostgresDocumentCollection(DocumentCollection[TDocument]):
    """PostgreSQL-backed document collection."""

    INDEXED_FIELDS = {
        "id",
        "version",
        "creation_utc",
    }

    def __init__(
        self,
        database: PostgresDocumentDatabase,
        name: str,
        schema: type[TDocument],
        logger: Logger,
    ) -> None:
        self._database = database
        self._name = name
        self._schema = schema
        self._logger = logger

        self._table_name = _sanitize_identifier(database._table_prefix + name)
        self._table = database._table_identifier(name)
        self._failed_table = database._failed_table_identifier(name)

    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        sort_direction = sort_direction or SortDirection.ASC

        translator = _WhereTranslator(self.INDEXED_FIELDS)
        base_clause = translator.render(filters)
        params: list[Any] = list(translator.params)

        # Build cursor clause
        cursor_clause = ""
        if cursor is not None:
            op = "<" if sort_direction == SortDirection.DESC else ">"
            param_offset = len(params)
            cursor_clause = (
                f"(creation_utc {op} ${param_offset + 1} "
                f"OR (creation_utc = ${param_offset + 1} AND id {op} ${param_offset + 2}))"
            )
            params.extend([cursor.creation_utc, str(cursor.id)])

        # Combine clauses
        where_clause = ""
        if base_clause and cursor_clause:
            where_clause = f"WHERE {base_clause} AND {cursor_clause}"
        elif base_clause:
            where_clause = f"WHERE {base_clause}"
        elif cursor_clause:
            where_clause = f"WHERE {cursor_clause}"

        # Order
        order_direction = "DESC" if sort_direction == SortDirection.DESC else "ASC"
        order_by = f"ORDER BY creation_utc {order_direction}, id {order_direction}"

        # Limit
        query_limit = (limit + 1) if limit else None
        limit_clause = f"LIMIT {query_limit}" if query_limit else ""

        sql = f"SELECT data FROM {self._table} {where_clause} {order_by} {limit_clause}"

        rows = await self._database._execute(sql, params or None, fetch="all")
        documents = [cast(TDocument, self._row_to_document(row)) for row in rows or []]

        total_count = len(documents)
        has_more = False
        next_cursor = None

        if limit and len(documents) > limit:
            has_more = True
            documents = documents[:limit]

            if documents:
                last_doc = documents[-1]
                creation_utc = last_doc.get("creation_utc")
                identifier = last_doc.get("id")

                if creation_utc is not None and identifier is not None:
                    next_cursor = Cursor(
                        creation_utc=str(creation_utc),
                        id=ObjectId(str(identifier)),
                    )

        return FindResult(
            items=documents,
            total_count=total_count,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    async def find_one(self, filters: Where) -> Optional[TDocument]:
        translator = _WhereTranslator(self.INDEXED_FIELDS)
        clause = translator.render(filters)
        params = list(translator.params)

        where_clause = f"WHERE {clause}" if clause else ""
        sql = f"SELECT data FROM {self._table} {where_clause} LIMIT 1"

        row = await self._database._execute(sql, params or None, fetch="one")
        if not row:
            return None

        return cast(TDocument, self._row_to_document(row))

    async def insert_one(self, document: TDocument) -> InsertResult:
        ensure_is_total(document, self._schema)

        params = self._serialize_document(document)
        sql = f"""
            INSERT INTO {self._table} (id, version, creation_utc, data)
            VALUES ($1, $2, $3, $4)
        """

        await self._database._execute(
            sql,
            [params["id"], params["version"], params["creation_utc"], params["data"]],
        )
        return InsertResult(acknowledged=True)

    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        existing = await self.find_one(filters)

        if existing:
            updated_document = cast(TDocument, {**existing, **params})
            await self._replace_document(updated_document)
            return UpdateResult(
                True,
                matched_count=1,
                modified_count=1,
                updated_document=updated_document,
            )

        if upsert:
            await self.insert_one(params)
            return UpdateResult(True, matched_count=0, modified_count=0, updated_document=params)

        return UpdateResult(True, matched_count=0, modified_count=0, updated_document=None)

    async def delete_one(self, filters: Where) -> DeleteResult[TDocument]:
        existing = await self.find_one(filters)
        if not existing:
            return DeleteResult(True, deleted_count=0, deleted_document=None)

        identifier = existing.get("id")
        if identifier is None:
            return DeleteResult(True, deleted_count=0, deleted_document=None)

        await self._delete_documents([identifier])

        return DeleteResult(True, deleted_count=1, deleted_document=existing)

    def _row_to_document(self, row: Any) -> BaseDocument:
        """Extract document from PostgreSQL row."""
        if isinstance(row, Mapping):
            data = row.get("data")
        else:
            # asyncpg Record - access by index or key
            data = row[0] if len(row) == 1 else row["data"]

        if isinstance(data, str):
            return cast(BaseDocument, json.loads(data))

        # asyncpg returns JSONB as dict directly
        return cast(BaseDocument, data)

    async def _replace_document(self, document: TDocument) -> None:
        params = self._serialize_document(document)
        sql = f"""
            UPDATE {self._table}
            SET version = $1, creation_utc = $2, data = $3
            WHERE id = $4
        """
        await self._database._execute(
            sql,
            [params["version"], params["creation_utc"], params["data"], params["id"]],
        )

    async def _delete_documents(self, identifiers: Sequence[Any]) -> None:
        if not identifiers:
            return

        placeholders = ", ".join(f"${i + 1}" for i in range(len(identifiers)))
        params = [_stringify(value) for value in identifiers]
        sql = f"DELETE FROM {self._table} WHERE id IN ({placeholders})"
        await self._database._execute(sql, params)

    async def _persist_failed_documents(self, documents: Sequence[BaseDocument]) -> None:
        if not documents:
            return

        for doc in documents:
            sql = f"""
                INSERT INTO {self._failed_table} (id, data)
                VALUES ($1, $2)
            """
            await self._database._execute(
                sql,
                [_stringify(doc.get("id")), json.dumps(doc, ensure_ascii=False)],
            )

    def _serialize_document(self, document: TDocument) -> MutableMapping[str, Any]:
        return {
            "id": _stringify(document["id"]),
            "version": _stringify(document.get("version")),
            "creation_utc": _stringify(document.get("creation_utc")),
            "data": json.dumps(document, ensure_ascii=False),
        }


class _WhereTranslator:
    """Translates MongoDB-style Where filters to PostgreSQL SQL."""

    def __init__(self, indexed_fields: set[str]) -> None:
        self._indexed_fields = indexed_fields
        self._params: list[Any] = []
        self._counter = 0

    @property
    def params(self) -> Sequence[Any]:
        return self._params

    def render(self, filters: Where) -> str:
        return self._render(filters)

    def _render(self, filters: Where) -> str:
        if not filters:
            return ""

        if isinstance(filters, Mapping):
            fragments: list[str] = []
            for key, value in filters.items():
                if key == "$and":
                    parts = [self._render(part) for part in cast(Sequence[Where], value)]
                    parts = [part for part in parts if part]
                    if parts:
                        fragments.append("(" + " AND ".join(parts) + ")")
                elif key == "$or":
                    parts = [self._render(part) for part in cast(Sequence[Where], value)]
                    parts = [part for part in parts if part]
                    if parts:
                        fragments.append("(" + " OR ".join(parts) + ")")
                else:
                    frag = self._render_field(key, value)
                    if frag:
                        fragments.append(frag)

            return " AND ".join(part for part in fragments if part)

        raise PostgresAdapterError("Unsupported filter format for PostgreSQL adapter")

    def _render_field(self, field: str, condition: Any) -> str:
        if not isinstance(condition, Mapping):
            return self._equality_clause(field, condition)

        clauses: list[str] = []
        for operator, operand in condition.items():
            if operator == "$eq":
                clauses.append(self._equality_clause(field, operand))
            elif operator in {"$gt", "$gte", "$lt", "$lte", "$ne"}:
                clauses.append(self._comparison_clause(field, operator, operand))
            elif operator == "$in":
                clauses.append(self._membership_clause(field, operand, negate=False))
            elif operator == "$nin":
                clauses.append(self._membership_clause(field, operand, negate=True))
            else:
                raise PostgresAdapterError(
                    f"Unsupported operator '{operator}' in PostgreSQL filter"
                )

        return " AND ".join(clauses)

    def _column_expr(self, field: str) -> str:
        """Returns PostgreSQL column expression."""
        if field in self._indexed_fields:
            return field
        # JSONB text extraction for non-indexed fields
        return f"data->>'{field}'"

    def _equality_clause(self, field: str, operand: Any) -> str:
        placeholder = self._add_param(operand)
        column = self._column_expr(field)
        return f"{column} = {placeholder}"

    def _comparison_clause(self, field: str, operator: str, operand: Any) -> str:
        sql_operator = {
            "$gt": ">",
            "$gte": ">=",
            "$lt": "<",
            "$lte": "<=",
            "$ne": "!=",
        }[operator]

        placeholder = self._add_param(operand)
        column = self._column_expr(field)
        return f"{column} {sql_operator} {placeholder}"

    def _membership_clause(self, field: str, operand: Any, *, negate: bool) -> str:
        values = list(operand or [])
        if not values:
            return "TRUE" if negate else "FALSE"

        column = self._column_expr(field)
        placeholders = [self._add_param(v) for v in values]

        operator = "NOT IN" if negate else "IN"
        return f"{column} {operator} ({', '.join(placeholders)})"

    def _add_param(self, value: Any) -> str:
        self._counter += 1
        object_id_type = getattr(ObjectId, "__supertype__", str)
        if isinstance(value, object_id_type):
            value = str(value)
        elif isinstance(value, bool):
            # JSONB text extraction (->>) returns "true"/"false" as strings
            value = str(value).lower()
        elif isinstance(value, (int, float)):
            # All indexed columns are TEXT and JSONB ->> returns TEXT
            value = str(value)
        self._params.append(value)
        return f"${self._counter}"


__all__ = [
    "PostgresAdapterError",
    "PostgresDocumentCollection",
    "PostgresDocumentDatabase",
]
