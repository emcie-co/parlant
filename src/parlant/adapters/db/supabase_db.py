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

import asyncio
import importlib
import json
import os
import re
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
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


class SupabaseAdapterError(Exception):
    """Raised for recoverable adapter errors."""


_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]")


def _sanitize_identifier(raw: str) -> str:
    """Sanitize identifier for PostgreSQL table/column names."""
    sanitized = _IDENTIFIER_RE.sub("_", raw).lower()
    if not sanitized:
        raise SupabaseAdapterError("Supabase identifier cannot be empty")

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


def _load_connection_params_from_env() -> dict[str, Any]:
    env = os.environ
    required = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
    ]

    missing = [key for key in required if not env.get(key)]
    if missing:
        raise SupabaseAdapterError(
            "Missing Supabase configuration. Set the following environment variables: "
            + ", ".join(missing)
        )

    params: dict[str, Any] = {
        "url": env["SUPABASE_URL"],
        "key": env["SUPABASE_KEY"],
    }

    # Optional: schema name (defaults to 'public')
    if env.get("SUPABASE_SCHEMA"):
        params["schema"] = env["SUPABASE_SCHEMA"]
    else:
        params["schema"] = "public"

    return params


FetchMode = Literal["none", "all", "one"]


class SupabaseDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        logger: Logger,
        connection_params: Mapping[str, Any] | None = None,
        *,
        table_prefix: str | None = None,
        client_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self._logger = logger
        self._connection_params = (
            dict(connection_params)
            if connection_params is not None
            else _load_connection_params_from_env()
        )
        self._table_prefix = _sanitize_identifier(table_prefix) if table_prefix else "parlant_"
        self._client_factory = client_factory

        self._supabase_module: Any | None = None
        self._postgrest_module: Any | None = None
        self._client: Any | None = None
        self._pg_connection: Any | None = None

        self._collections: dict[str, SupabaseDocumentCollection[Any]] = {}

        self._connection_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        await self._ensure_connection()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool:
        # Close PostgreSQL connection if it exists
        if hasattr(self, "_pg_connection") and self._pg_connection is not None:
            await self._pg_connection.close()
            self._pg_connection = None
        
        # Supabase client doesn't need explicit closing, but we can clean up
        if self._client is not None:
            self._client = None

        return False

    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> SupabaseDocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
        await collection.ensure_table()
        return collection

    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> SupabaseDocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
        await collection.ensure_table()
        await collection.load_existing_documents(document_loader)
        return collection

    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> SupabaseDocumentCollection[TDocument]:
        return await self.get_collection(name, schema, document_loader)

    async def delete_collection(self, name: str) -> None:
        table = self._table_identifier(name)
        failed_table = self._failed_table_identifier(name)
        
        await self._execute_sql(f"DROP TABLE IF EXISTS {failed_table} CASCADE")
        await self._execute_sql(f"DROP TABLE IF EXISTS {table} CASCADE")
        self._collections.pop(name, None)

    async def _get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> SupabaseDocumentCollection[TDocument]:
        if name not in self._collections:
            self._collections[name] = SupabaseDocumentCollection(
                database=self,
                name=name,
                schema=schema,
                logger=self._logger,
            )

        return cast(SupabaseDocumentCollection[TDocument], self._collections[name])

    async def _execute_sql(self, sql: str) -> None:
        """Execute raw SQL using direct PostgreSQL connection."""
        await self._ensure_connection()
        
        async with self._operation_lock:
            # Extract connection string from Supabase URL
            # Supabase URL format: https://<project-ref>.supabase.co
            # We need to construct PostgreSQL connection string
            if self._pg_connection is None:
                # Use asyncpg for async PostgreSQL access
                try:
                    import asyncpg
                except ImportError:
                    raise SupabaseAdapterError(
                        "Supabase adapter requires asyncpg for table creation. Install asyncpg."
                    )
                
                # Parse Supabase URL to get connection details
                url = self._connection_params["url"]
                # Extract project ref from URL
                # Format: https://<project-ref>.supabase.co
                import re
                match = re.search(r"https://([^.]+)\.supabase\.co", url)
                if not match:
                    raise SupabaseAdapterError("Invalid Supabase URL format")
                
                project_ref = match.group(1)
                
                # Get database password from env or connection params
                db_password = os.environ.get("SUPABASE_DB_PASSWORD") or self._connection_params.get("db_password")
                if not db_password:
                    raise SupabaseAdapterError(
                        "SUPABASE_DB_PASSWORD is required for table creation. "
                        "Get it from your Supabase project settings."
                    )
                
                # Construct connection string
                # Default port is 5432, database name is usually 'postgres'
                db_host = f"{project_ref}.supabase.co"
                db_user = os.environ.get("SUPABASE_DB_USER", "postgres")
                db_name = os.environ.get("SUPABASE_DB_NAME", "postgres")
                db_port = int(os.environ.get("SUPABASE_DB_PORT", "5432"))
                
                self._pg_connection = await asyncpg.connect(
                    host=db_host,
                    port=db_port,
                    user=db_user,
                    password=db_password,
                    database=db_name,
                    ssl="require",
                )
            
            await self._pg_connection.execute(sql)

    async def _ensure_connection(self) -> None:
        if self._client is not None:
            return

        async with self._connection_lock:
            if self._client is not None:
                return

            self._import_client()

            if self._client_factory is not None:
                self._client = self._client_factory(self._connection_params)
            else:
                assert self._supabase_module is not None
                self._client = await asyncio.to_thread(
                    self._supabase_module.create_client,
                    self._connection_params["url"],
                    self._connection_params["key"],
                )

    def _import_client(self) -> None:
        if self._supabase_module is not None:
            return

        try:
            supabase_module = importlib.import_module("supabase")
        except ImportError as exc:
            raise SupabaseAdapterError(
                "Supabase adapter requires supabase-py. Install parlant[supabase] or pip install supabase."
            ) from exc

        self._supabase_module = supabase_module

    def _table_identifier(self, name: str) -> str:
        return _sanitize_identifier(self._table_prefix + name)

    def _failed_table_identifier(self, name: str) -> str:
        return _sanitize_identifier(self._table_prefix + name + "_failed_migrations")


class SupabaseDocumentCollection(DocumentCollection[TDocument]):
    INDEXED_FIELDS = {
        "id",
        "version",
        "creation_utc",
        "session_id",
        "customer_id",
        "agent_id",
    }

    def __init__(
        self,
        database: SupabaseDocumentDatabase,
        name: str,
        schema: type[TDocument],
        logger: Logger,
    ) -> None:
        self._database = database
        self._name = name
        self._schema = schema
        self._logger = logger

        self._table = self._database._table_identifier(name)
        self._failed_table = self._database._failed_table_identifier(name)

        self._table_ready = False
        self._loader_done = False
        self._table_lock = asyncio.Lock()
        self._loader_lock = asyncio.Lock()

    async def ensure_table(self) -> None:
        if self._table_ready:
            return

        async with self._table_lock:
            if self._table_ready:
                return

            # Create main table
            create_stmt = f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT NOT NULL PRIMARY KEY,
                    version TEXT,
                    creation_utc TEXT,
                    session_id TEXT,
                    customer_id TEXT,
                    agent_id TEXT,
                    data JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
            """

            # Create indexes for commonly queried fields
            index_stmt = f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_creation_utc 
                ON {self._table}(creation_utc);
                
                CREATE INDEX IF NOT EXISTS idx_{self._table}_session_id 
                ON {self._table}(session_id) WHERE session_id IS NOT NULL;
                
                CREATE INDEX IF NOT EXISTS idx_{self._table}_customer_id 
                ON {self._table}(customer_id) WHERE customer_id IS NOT NULL;
                
                CREATE INDEX IF NOT EXISTS idx_{self._table}_agent_id 
                ON {self._table}(agent_id) WHERE agent_id IS NOT NULL;
                
                CREATE INDEX IF NOT EXISTS idx_{self._table}_data_gin 
                ON {self._table} USING GIN (data);
            """

            # Create failed migrations table
            failed_table_stmt = f"""
                CREATE TABLE IF NOT EXISTS {self._failed_table} (
                    id TEXT,
                    data JSONB DEFAULT '{{}}'::jsonb
                )
            """

            try:
                await self._database._execute_sql(create_stmt)
                await self._database._execute_sql(index_stmt)
                await self._database._execute_sql(failed_table_stmt)
            except Exception as exc:
                self._logger.error(
                    f"Failed to create table {self._table}: {exc}. "
                    "Ensure SUPABASE_DB_PASSWORD is set correctly."
                )
                raise SupabaseAdapterError(
                    f"Failed to create table: {exc}"
                ) from exc

            self._table_ready = True

    async def load_existing_documents(
        self,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        if self._loader_done:
            return

        async with self._loader_lock:
            if self._loader_done:
                return

            await self.ensure_table()

            # Fetch all documents
            response = await asyncio.to_thread(
                lambda: self._database._client.table(self._table)
                .select("*")
                .execute()
            )

            failed: list[BaseDocument] = []
            for row in response.data or []:
                doc = self._row_to_document(row)
                try:
                    migrated = await document_loader(doc)
                except Exception as exc:
                    self._logger.error(
                        f"Failed to load document '{doc.get('id')}' in collection '{self._name}': {exc}"
                    )
                    failed.append(doc)
                    continue

                if migrated is None:
                    failed.append(doc)
                    continue

                if migrated is not doc:
                    await self._replace_document(migrated)

            if failed:
                await self._persist_failed_documents(failed)
                await self._delete_documents([doc["id"] for doc in failed if "id" in doc])

            self._loader_done = True

    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        await self.ensure_table()

        sort_direction = sort_direction or SortDirection.ASC

        # Build query using Supabase PostgREST
        query = self._database._client.table(self._table)

        # Apply filters
        query = self._apply_filters(query, filters)

        # Apply cursor filtering - simplified approach
        if cursor is not None:
            if sort_direction == SortDirection.DESC:
                # For DESC: get records where creation_utc < cursor OR (creation_utc == cursor AND id < cursor.id)
                query = query.or_(
                    f"creation_utc.lt.{cursor.creation_utc}",
                    f"and(creation_utc.eq.{cursor.creation_utc},id.lt.{cursor.id})",
                )
            else:
                # For ASC: get records where creation_utc > cursor OR (creation_utc == cursor AND id > cursor.id)
                query = query.or_(
                    f"creation_utc.gt.{cursor.creation_utc}",
                    f"and(creation_utc.eq.{cursor.creation_utc},id.gt.{cursor.id})",
                )

        # Apply sorting
        order_direction = "desc" if sort_direction == SortDirection.DESC else "asc"
        query = query.order("creation_utc", desc=(order_direction == "desc"))
        query = query.order("id", desc=(order_direction == "desc"))

        # Apply limit (fetch one extra to check for more)
        query_limit = (limit + 1) if limit else None
        if query_limit:
            query = query.limit(query_limit)

        # Execute query - select all columns, we'll extract data
        response = await asyncio.to_thread(lambda: query.select("*").execute())
        rows = response.data or []

        documents = [cast(TDocument, self._row_to_document(row)) for row in rows]

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

    def _apply_filters(self, query: Any, filters: Where) -> Any:
        """Apply filters to Supabase query builder."""
        if not filters:
            return query

        if isinstance(filters, Mapping):
            for key, value in filters.items():
                if key == "$and":
                    # For AND, chain filters (PostgREST applies them with AND by default)
                    for part in cast(Sequence[Where], value):
                        query = self._apply_filters(query, part)
                elif key == "$or":
                    # PostgREST supports OR using or_() method with filter strings
                    # We'll need to build filter strings for each OR condition
                    or_parts = []
                    for part in cast(Sequence[Where], value):
                        or_filter = self._build_filter_string(part)
                        if or_filter:
                            or_parts.append(or_filter)
                    if or_parts:
                        # Combine OR conditions
                        query = query.or_(",".join(or_parts))
                else:
                    query = self._apply_field_filter(query, key, value)

        return query

    def _build_filter_string(self, filters: Where) -> str:
        """Build a filter string for PostgREST OR conditions."""
        if not filters or not isinstance(filters, Mapping):
            return ""
        
        parts = []
        for key, value in filters.items():
            if key in ("$and", "$or"):
                continue
            if not isinstance(value, Mapping):
                parts.append(f"{key}.eq.{value}")
            else:
                for op, operand in value.items():
                    op_map = {
                        "$eq": "eq",
                        "$ne": "neq",
                        "$gt": "gt",
                        "$gte": "gte",
                        "$lt": "lt",
                        "$lte": "lte",
                    }
                    if op in op_map:
                        parts.append(f"{key}.{op_map[op]}.{operand}")
                    elif op == "$in":
                        parts.append(f"{key}.in.({','.join(map(str, operand))})")
        
        return ",".join(parts)

    def _apply_field_filter(self, query: Any, field: str, condition: Any) -> Any:
        """Apply a single field filter."""
        if not isinstance(condition, Mapping):
            # Simple equality
            if field in self.INDEXED_FIELDS:
                return query.eq(field, condition)
            else:
                # JSONB field access using PostgREST JSON operators
                return query.eq(f"data->{field}", json.dumps(condition))

        # Handle operators
        for operator, operand in condition.items():
            if operator == "$eq":
                if field in self.INDEXED_FIELDS:
                    return query.eq(field, operand)
                else:
                    return query.eq(f"data->{field}", json.dumps(operand))
            elif operator == "$ne":
                if field in self.INDEXED_FIELDS:
                    return query.neq(field, operand)
                else:
                    return query.neq(f"data->{field}", json.dumps(operand))
            elif operator == "$gt":
                if field in self.INDEXED_FIELDS:
                    return query.gt(field, operand)
                else:
                    return query.gt(f"data->{field}", json.dumps(operand))
            elif operator == "$gte":
                if field in self.INDEXED_FIELDS:
                    return query.gte(field, operand)
                else:
                    return query.gte(f"data->{field}", json.dumps(operand))
            elif operator == "$lt":
                if field in self.INDEXED_FIELDS:
                    return query.lt(field, operand)
                else:
                    return query.lt(f"data->{field}", json.dumps(operand))
            elif operator == "$lte":
                if field in self.INDEXED_FIELDS:
                    return query.lte(field, operand)
                else:
                    return query.lte(f"data->{field}", json.dumps(operand))
            elif operator == "$in":
                if field in self.INDEXED_FIELDS:
                    return query.in_(field, operand)
                else:
                    # For JSONB, convert to array format
                    return query.in_(f"data->{field}", [json.dumps(v) for v in operand])
            elif operator == "$nin":
                # PostgREST doesn't have direct NOT IN, skip for now
                # Could be implemented with multiple .neq() calls
                pass

        return query

    async def find_one(self, filters: Where) -> Optional[TDocument]:
        await self.ensure_table()

        query = self._database._client.table(self._table)
        query = self._apply_filters(query, filters)
        query = query.limit(1)

        response = await asyncio.to_thread(
            lambda: query.select("*").execute()
        )

        if not response.data:
            return None

        return cast(TDocument, self._row_to_document(response.data[0]))

    async def insert_one(self, document: TDocument) -> InsertResult:
        await self.ensure_table()
        ensure_is_total(document, self._schema)

        params = self._serialize_document(document)
        
        response = await asyncio.to_thread(
            lambda: self._database._client.table(self._table)
            .insert(params)
            .execute()
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
        """Extract document from Supabase row."""
        if isinstance(row, Mapping):
            # If row has 'data' key, use it (JSONB column)
            if "data" in row:
                data = row["data"]
                # If data is already a dict, use it directly
                if isinstance(data, dict):
                    return cast(BaseDocument, data)
                # If it's a string, parse it
                if isinstance(data, str):
                    return cast(BaseDocument, json.loads(data))
            # Otherwise, treat the whole row as the document
            return cast(BaseDocument, row)
        
        # If row is not a mapping, try to parse as JSON string
        if isinstance(row, str):
            return cast(BaseDocument, json.loads(row))
        
        # Fallback: treat as document directly
        return cast(BaseDocument, row)

    async def _replace_document(self, document: TDocument) -> None:
        params = self._serialize_document(document)
        doc_id = params["id"]
        
        # Remove id from update params
        update_params = {k: v for k, v in params.items() if k != "id"}
        
        await asyncio.to_thread(
            lambda: self._database._client.table(self._table)
            .update(update_params)
            .eq("id", doc_id)
            .execute()
        )

    async def _delete_documents(self, identifiers: Sequence[Any]) -> None:
        if not identifiers:
            return

        # Supabase supports deleting by multiple IDs
        for identifier in identifiers:
            await asyncio.to_thread(
                lambda id=identifier: self._database._client.table(self._table)
                .delete()
                .eq("id", _stringify(id))
                .execute()
            )

    async def _persist_failed_documents(self, documents: Sequence[BaseDocument]) -> None:
        if not documents:
            return

        for doc in documents:
            params = {
                "id": _stringify(doc.get("id")),
                "data": doc,
            }

            await asyncio.to_thread(
                lambda p=params: self._database._client.table(self._failed_table)
                .insert(p)
                .execute()
            )

    def _serialize_document(self, document: TDocument) -> MutableMapping[str, Any]:
        return {
            "id": _stringify(document["id"]),
            "version": document.get("version"),
            "creation_utc": document.get("creation_utc"),
            "session_id": _stringify(document.get("session_id")),
            "customer_id": _stringify(document.get("customer_id")),
            "agent_id": _stringify(document.get("agent_id")),
            "data": document,
        }


__all__ = [
    "SupabaseAdapterError",
    "SupabaseDocumentCollection",
    "SupabaseDocumentDatabase",
]

