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
import time
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
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection, Where
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


# Retry configuration for connection errors
_MAX_RETRIES = 5  # Increased retries for timeout errors
_RETRY_DELAY = 1.0  # seconds (increased initial delay)
_RETRY_BACKOFF = 1.5  # multiplier (reduced for faster retries)


def _retry_on_connection_error(func: Callable[[], Any], max_retries: int = _MAX_RETRIES) -> Any:
    """
    Retry a function call on connection errors.

    Handles httpx.RemoteProtocolError and connection-related errors
    that can occur with HTTP/2 connections.
    """
    last_exception = None

    # Import exception types for proper checking
    try:
        import httpx
        import httpcore

        _HTTPX_AVAILABLE = True
    except ImportError:
        _HTTPX_AVAILABLE = False

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            # Check if it's a connection error
            error_str = str(exc)
            error_type = type(exc).__name__
            error_module = type(exc).__module__

            # Check for specific httpx/httpcore exceptions
            is_connection_error = False

            if _HTTPX_AVAILABLE:
                # Check for httpx exceptions
                if isinstance(
                    exc,
                    (
                        httpx.RemoteProtocolError,
                        httpx.ConnectError,
                        httpx.NetworkError,
                        httpx.ReadTimeout,
                        httpx.WriteTimeout,
                        httpx.ConnectTimeout,
                        httpx.PoolTimeout,
                        httpx.LocalProtocolError,
                    ),
                ):
                    is_connection_error = True
                # Check for httpcore exceptions
                elif hasattr(httpcore, "RemoteProtocolError") and isinstance(
                    exc, httpcore.RemoteProtocolError
                ):
                    is_connection_error = True
                elif hasattr(httpcore, "ConnectError") and isinstance(exc, httpcore.ConnectError):
                    is_connection_error = True
                elif hasattr(httpcore, "ReadTimeout") and isinstance(exc, httpcore.ReadTimeout):
                    is_connection_error = True
                elif hasattr(httpcore, "WriteTimeout") and isinstance(exc, httpcore.WriteTimeout):
                    is_connection_error = True
                elif hasattr(httpcore, "LocalProtocolError") and isinstance(
                    exc, httpcore.LocalProtocolError
                ):
                    is_connection_error = True

            # Also check error message for connection-related terms
            if not is_connection_error:
                is_connection_error = (
                    "ConnectionTerminated" in error_str
                    or "Server disconnected" in error_str
                    or "timed out" in error_str.lower()
                    or "TimeoutError" in error_type
                    or "ReadTimeout" in error_type
                    or "WriteTimeout" in error_type
                    or "ConnectTimeout" in error_type
                    or "PoolTimeout" in error_type
                    or "RemoteProtocolError" in error_type
                    or "LocalProtocolError" in error_type
                    or "ConnectionError" in error_type
                    or "ConnectError" in error_type
                    or "StreamIDTooLow" in error_str
                    or "KeyError" in error_type
                    or "httpcore" in error_module
                )

            if is_connection_error and attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = _RETRY_DELAY * (_RETRY_BACKOFF**attempt)
                time.sleep(delay)
                last_exception = exc
                continue
            else:
                # Either not a connection error or last attempt
                raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


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

        self._collections: dict[str, SupabaseDocumentCollection[Any]] = {}

        self._connection_lock = asyncio.Lock()

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
        return collection

    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> SupabaseDocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
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
        """Remove collection from cache. Tables must be dropped manually via SQL."""
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

                # Configure client with HTTP/1.1 to avoid HTTP/2 connection issues
                # The Supabase client uses httpx internally, and we need to configure it
                # to disable HTTP/2 which can cause connection termination errors
                client_created = False

                # Try method 1: Use ClientOptions (supabase-py >= 2.0.0)
                try:
                    from supabase.lib.client_options import ClientOptions
                    import httpx

                    # Create httpx client with HTTP/1.1 only and longer timeouts
                    http_client = httpx.Client(
                        http2=False,  # Disable HTTP/2 to avoid connection issues
                        timeout=httpx.Timeout(
                            connect=30.0,  # Connection timeout
                            read=60.0,  # Read timeout (increased for slow queries)
                            write=30.0,  # Write timeout
                            pool=30.0,  # Pool timeout
                        ),
                        limits=httpx.Limits(
                            max_keepalive_connections=10,
                            max_connections=20,
                            keepalive_expiry=30.0,
                        ),
                    )

                    # Configure client options
                    options = ClientOptions(
                        http_client=http_client,
                    )

                    self._client = await asyncio.to_thread(
                        self._supabase_module.create_client,
                        self._connection_params["url"],
                        self._connection_params["key"],
                        options=options,
                    )
                    client_created = True
                except (ImportError, TypeError, AttributeError):
                    # ClientOptions not available or not supported
                    pass

                # Try method 2: Use environment variable to force HTTP/1.1 (fallback)
                if not client_created:
                    try:
                        import httpx
                        import os

                        # Set environment variable to disable HTTP/2 globally for httpx
                        # This is a workaround if ClientOptions doesn't work
                        os.environ.setdefault("HTTPX_DISABLE_HTTP2", "1")

                        # Create httpx client with HTTP/1.1 only
                        http_client = httpx.Client(
                            http2=False,
                            timeout=httpx.Timeout(
                                connect=30.0,
                                read=60.0,
                                write=30.0,
                                pool=30.0,
                            ),
                            limits=httpx.Limits(
                                max_keepalive_connections=10,
                                max_connections=20,
                                keepalive_expiry=30.0,
                            ),
                        )

                        # Try to create client with custom http_client if the library supports it
                        # Some versions of supabase-py allow passing http_client directly
                        try:
                            self._client = await asyncio.to_thread(
                                self._supabase_module.create_client,
                                self._connection_params["url"],
                                self._connection_params["key"],
                                http_client=http_client,
                            )
                            client_created = True
                        except TypeError:
                            # http_client parameter not supported, try with options
                            try:
                                # Try creating with a custom session factory
                                from supabase.lib.client_options import ClientOptions

                                options = ClientOptions(http_client=http_client)
                                self._client = await asyncio.to_thread(
                                    self._supabase_module.create_client,
                                    self._connection_params["url"],
                                    self._connection_params["key"],
                                    options=options,
                                )
                                client_created = True
                            except Exception:
                                # Fall through to default client creation
                                pass

                        if not client_created:
                            # Create default client - retry logic will handle issues
                            self._client = await asyncio.to_thread(
                                self._supabase_module.create_client,
                                self._connection_params["url"],
                                self._connection_params["key"],
                            )
                            client_created = True

                    except Exception as exc:
                        self._logger.warning(
                            f"Could not configure Supabase client with HTTP/1.1: {exc}. "
                            "Using default client with retry logic."
                        )

                # Method 3: Last resort - create client without custom config
                # Retry logic will handle connection errors
                if not client_created:
                    # Try to force HTTP/1.1 via environment variable
                    import os

                    original_env = os.environ.get("HTTPX_DISABLE_HTTP2")
                    try:
                        os.environ["HTTPX_DISABLE_HTTP2"] = "1"
                        self._client = await asyncio.to_thread(
                            self._supabase_module.create_client,
                            self._connection_params["url"],
                            self._connection_params["key"],
                        )
                    finally:
                        # Restore original environment variable if it existed
                        if original_env is not None:
                            os.environ["HTTPX_DISABLE_HTTP2"] = original_env
                        elif "HTTPX_DISABLE_HTTP2" in os.environ:
                            del os.environ["HTTPX_DISABLE_HTTP2"]

    def _import_client(self) -> None:
        if self._supabase_module is not None:
            return

        # Set environment variable to disable HTTP/2 before importing supabase
        # This ensures httpx uses HTTP/1.1 by default
        import os

        original_env = os.environ.get("HTTPX_DISABLE_HTTP2")
        try:
            os.environ["HTTPX_DISABLE_HTTP2"] = "1"

            try:
                supabase_module = importlib.import_module("supabase")
            except ImportError as exc:
                raise SupabaseAdapterError(
                    "Supabase adapter requires supabase-py. Install parlant[supabase] or pip install supabase."
                ) from exc

            self._supabase_module = supabase_module
        finally:
            # Restore original environment variable if it existed
            if original_env is not None:
                os.environ["HTTPX_DISABLE_HTTP2"] = original_env
            elif "HTTPX_DISABLE_HTTP2" in os.environ:
                # Keep it set since we want HTTP/1.1
                pass

    def _table_identifier(self, name: str) -> str:
        # If prefix already ends with the collection name, use prefix as-is
        # This handles cases where table_prefix is already the full table name
        # e.g., prefix="parlant_sessions_", name="sessions" -> use "parlant_sessions_"
        prefix_normalized = self._table_prefix.lower().rstrip("_")
        name_normalized = name.lower()

        # Check if prefix already ends with the collection name
        if prefix_normalized.endswith(name_normalized):
            # Use prefix as-is (preserving trailing underscore if present)
            return _sanitize_identifier(self._table_prefix)

        # Otherwise, append the collection name to the prefix
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

        self._loader_done = False
        self._loader_lock = asyncio.Lock()

    async def load_existing_documents(
        self,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        if self._loader_done:
            return

        async with self._loader_lock:
            if self._loader_done:
                return

            await self._database._ensure_connection()
            assert self._database._client is not None

            # Fetch all documents
            client = self._database._client
            table_name = self._table
            response = await asyncio.to_thread(
                lambda: _retry_on_connection_error(
                    lambda: client.table(table_name).select("*").execute()
                )
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
        await self._database._ensure_connection()
        assert self._database._client is not None

        sort_direction = sort_direction or SortDirection.ASC

        # Build query using Supabase PostgREST
        # Start with select("*") to get all columns
        query = self._database._client.table(self._table).select("*")

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

        # Execute query with retry logic
        response = await asyncio.to_thread(
            lambda: _retry_on_connection_error(lambda: query.execute())
        )
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
                    # For OR conditions, we need to handle them carefully
                    # PostgREST's or_() method can be tricky with JSONB fields
                    # So we'll build OR conditions using filter strings, but handle JSONB properly
                    or_conditions = cast(Sequence[Where], value)

                    # Check if any OR condition involves non-indexed fields
                    has_jsonb_fields = False
                    for condition in or_conditions:
                        if isinstance(condition, Mapping):
                            for field_name in condition.keys():
                                if (
                                    field_name not in ("$and", "$or")
                                    and field_name not in self.INDEXED_FIELDS
                                ):
                                    has_jsonb_fields = True
                                    break
                        if has_jsonb_fields:
                            break

                    if has_jsonb_fields:
                        # For JSONB fields, use a different approach
                        # Build filter strings with proper JSONB syntax
                        or_parts = []
                        for part in or_conditions:
                            or_filter = self._build_filter_string(part)
                            if or_filter:
                                or_parts.append(or_filter)
                        if or_parts:
                            # Combine OR conditions
                            query = query.or_(",".join(or_parts))
                    else:
                        # For indexed fields only, use filter strings
                        or_parts = []
                        for part in or_conditions:
                            or_filter = self._build_filter_string(part)
                            if or_filter:
                                or_parts.append(or_filter)
                        if or_parts:
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

            # Determine if field is indexed (top-level column) or in JSONB data
            if key in self.INDEXED_FIELDS:
                field_name = key
            else:
                # For JSONB fields in OR filter strings, use PostgREST JSONB operator syntax
                # PostgREST format: data->>field_name for text extraction (no quotes around field name)
                # This allows filtering JSONB fields in OR conditions
                # Format: data->>tag_id.eq.value
                field_name = f"data->>{key}"

            if not isinstance(value, Mapping):
                # Simple equality
                if key in self.INDEXED_FIELDS:
                    parts.append(f"{field_name}.eq.{value}")
                else:
                    # For JSONB text extraction, compare as string
                    # The value should be converted to string for comparison
                    str_value = str(value) if not isinstance(value, str) else value
                    parts.append(f"{field_name}.eq.{str_value}")
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
                        if key in self.INDEXED_FIELDS:
                            parts.append(f"{field_name}.{op_map[op]}.{operand}")
                        else:
                            # For JSONB text extraction, convert operand to string
                            str_operand = str(operand) if not isinstance(operand, str) else operand
                            parts.append(f"{field_name}.{op_map[op]}.{str_operand}")
                    elif op == "$in":
                        if key in self.INDEXED_FIELDS:
                            parts.append(f"{field_name}.in.({','.join(map(str, operand))})")
                        else:
                            # For JSONB fields with $in, convert each value to string
                            str_values = [str(v) if not isinstance(v, str) else v for v in operand]
                            parts.append(f"{field_name}.in.({','.join(str_values)})")

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
        await self._database._ensure_connection()
        assert self._database._client is not None

        query = self._database._client.table(self._table).select("*")
        query = self._apply_filters(query, filters)
        query = query.limit(1)

        response = await asyncio.to_thread(
            lambda: _retry_on_connection_error(lambda: query.execute())
        )

        if not response.data or len(response.data) == 0:
            return None

        return cast(TDocument, self._row_to_document(response.data[0]))

    async def insert_one(self, document: TDocument) -> InsertResult:
        await self._database._ensure_connection()
        assert self._database._client is not None

        params = self._serialize_document(document)

        client = self._database._client
        table_name = self._table
        await asyncio.to_thread(
            lambda: _retry_on_connection_error(
                lambda: client.table(table_name).insert(params).execute()
            )
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
        await self._database._ensure_connection()
        assert self._database._client is not None

        params = self._serialize_document(document)
        doc_id = params["id"]

        # Remove id from update params
        update_params = {k: v for k, v in params.items() if k != "id"}

        client = self._database._client
        table_name = self._table
        await asyncio.to_thread(
            lambda: _retry_on_connection_error(
                lambda: client.table(table_name).update(update_params).eq("id", doc_id).execute()
            )
        )

    async def _delete_documents(self, identifiers: Sequence[Any]) -> None:
        if not identifiers:
            return

        await self._database._ensure_connection()
        assert self._database._client is not None

        # Supabase supports deleting by multiple IDs
        client = self._database._client
        table_name = self._table
        for identifier in identifiers:
            id_str = _stringify(identifier)
            if id_str is None:
                continue
            # Capture id_str in closure to avoid lambda default parameter issues
            id_val = id_str
            await asyncio.to_thread(
                lambda: _retry_on_connection_error(
                    lambda: client.table(table_name).delete().eq("id", id_val).execute()
                )
            )

    async def _persist_failed_documents(self, documents: Sequence[BaseDocument]) -> None:
        if not documents:
            return

        await self._database._ensure_connection()
        assert self._database._client is not None

        client = self._database._client
        table_name = self._failed_table
        for doc in documents:
            params = {
                "id": _stringify(doc.get("id")),
                "data": doc,
            }

            # Use a function to avoid lambda closure issues
            def insert_doc(p: dict[str, Any] = params) -> Any:
                return _retry_on_connection_error(
                    lambda: client.table(table_name).insert(p).execute()
                )

            await asyncio.to_thread(insert_doc)

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
