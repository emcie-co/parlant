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

# Maintainer: Tao Tang <ttan@habitus.dk>

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


class SnowflakeAdapterError(Exception):
    """Raised for recoverable adapter errors."""


_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]")


def _sanitize_identifier(raw: str) -> str:
    sanitized = _IDENTIFIER_RE.sub("_", raw).upper()
    if not sanitized:
        raise SnowflakeAdapterError("Snowflake identifier cannot be empty")

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
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]

    missing = [key for key in required if not env.get(key)]
    if missing:
        raise SnowflakeAdapterError(
            "Missing Snowflake configuration. Set the following environment variables: "
            + ", ".join(missing)
        )

    params: dict[str, Any] = {
        "account": env["SNOWFLAKE_ACCOUNT"],
        "user": env["SNOWFLAKE_USER"],
        "warehouse": env["SNOWFLAKE_WAREHOUSE"],
        "database": env["SNOWFLAKE_DATABASE"],
        "schema": env["SNOWFLAKE_SCHEMA"],
    }

    if env.get("SNOWFLAKE_ROLE"):
        params["role"] = env["SNOWFLAKE_ROLE"]

    token = env.get("SNOWFLAKE_TOKEN")
    password = env.get("SNOWFLAKE_PASSWORD")

    if token:
        params["authenticator"] = "oauth"
        params["token"] = token
    elif password:
        params["authenticator"] = env.get("SNOWFLAKE_AUTHENTICATOR", "snowflake")
        params["password"] = password
    else:
        raise SnowflakeAdapterError(
            "Provide either SNOWFLAKE_PASSWORD or SNOWFLAKE_TOKEN for authentication"
        )

    return params


FetchMode = Literal["none", "all", "one"]


class SnowflakeDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        logger: Logger,
        connection_params: Mapping[str, Any] | None = None,
        *,
        table_prefix: str | None = None,
        connection_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self._logger = logger
        self._connection_params = (
            dict(connection_params)
            if connection_params is not None
            else _load_connection_params_from_env()
        )
        self._table_prefix = _sanitize_identifier(table_prefix) if table_prefix else "PARLANT_"
        self._connection_factory = connection_factory

        self._connector_module: Any | None = None
        self._snowflake_error: type[BaseException] | None = None
        self._dict_cursor_cls: Any | None = None
        self._connection: Any | None = None

        self._collections: dict[str, SnowflakeDocumentCollection[Any]] = {}

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
        if self._connection is not None:
            await asyncio.to_thread(self._connection.close)
            self._connection = None

        return False

    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> SnowflakeDocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
        await collection.ensure_table()
        return collection

    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> SnowflakeDocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
        await collection.ensure_table()
        await collection.load_existing_documents(document_loader)
        return collection

    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> SnowflakeDocumentCollection[TDocument]:
        return await self.get_collection(name, schema, document_loader)

    async def delete_collection(self, name: str) -> None:
        table = self._table_identifier(name)
        failed_table = self._failed_table_identifier(name)
        await self._execute(f"DROP TABLE IF EXISTS {table}")
        await self._execute(f"DROP TABLE IF EXISTS {failed_table}")
        self._collections.pop(name, None)

    async def _get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> SnowflakeDocumentCollection[TDocument]:
        if name not in self._collections:
            self._collections[name] = SnowflakeDocumentCollection(
                database=self,
                name=name,
                schema=schema,
                logger=self._logger,
            )

        return cast(SnowflakeDocumentCollection[TDocument], self._collections[name])

    async def _execute(
        self,
        sql: str,
        params: Mapping[str, Any] | Sequence[Any] | None = None,
        *,
        fetch: FetchMode = "none",
    ) -> Any:
        await self._ensure_connection()

        async with self._operation_lock:
            return await asyncio.to_thread(self._run_query, sql, params, fetch)

    def _run_query(
        self,
        sql: str,
        params: Mapping[str, Any] | Sequence[Any] | None,
        fetch: FetchMode,
    ) -> Any:
        assert self._connection is not None
        cursor = (
            self._connection.cursor(self._dict_cursor_cls)
            if self._dict_cursor_cls is not None
            else self._connection.cursor()
        )

        try:
            cursor.execute(sql, params)
            if fetch == "all":
                return cursor.fetchall()
            if fetch == "one":
                return cursor.fetchone()
            return None
        except Exception as exc:  # pragma: no cover - wrapped below
            if self._snowflake_error and isinstance(exc, self._snowflake_error):
                raise SnowflakeAdapterError(f"Snowflake query failed: {exc}") from exc
            raise
        finally:
            cursor.close()

    async def _ensure_connection(self) -> None:
        if self._connection is not None:
            return

        async with self._connection_lock:
            if self._connection is not None:
                return

            self._import_connector()

            if self._connection_factory is not None:
                self._connection = self._connection_factory(self._connection_params)
            else:
                assert self._connector_module is not None
                self._connection = await asyncio.to_thread(
                    self._connector_module.connect,
                    **self._connection_params,
                )

    def _import_connector(self) -> None:
        if self._connector_module is not None:
            return

        try:
            connector_module = importlib.import_module("snowflake.connector")
        except ImportError as exc:  # pragma: no cover - exercised when dependency missing
            raise SnowflakeAdapterError(
                "Snowflake adapter requires snowflake-connector-python. Install parlant[snowflake]."
            ) from exc

        self._connector_module = connector_module
        self._dict_cursor_cls = getattr(connector_module, "DictCursor", None)

        try:
            errors_module = importlib.import_module("snowflake.connector.errors")
            self._snowflake_error = getattr(errors_module, "Error", None)
        except ImportError:
            self._snowflake_error = None

    def _table_identifier(self, name: str) -> str:
        return f'"{_sanitize_identifier(self._table_prefix + name)}"'

    def _failed_table_identifier(self, name: str) -> str:
        return f'"{_sanitize_identifier(self._table_prefix + name + "_failed_migrations")}"'


class SnowflakeDocumentCollection(DocumentCollection[TDocument]):
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
        database: SnowflakeDocumentDatabase,
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

            create_stmt = f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    ID STRING NOT NULL,
                    VERSION STRING,
                    CREATION_UTC STRING,
                    SESSION_ID STRING,
                    CUSTOMER_ID STRING,
                    AGENT_ID STRING,
                    DATA VARIANT,
                    PRIMARY KEY (ID)
                )
            """

            await self._database._execute(create_stmt)
            await self._database._execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._failed_table} (
                    ID STRING,
                    DATA VARIANT
                )
                """
            )

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

            rows = await self._database._execute(
                f"SELECT DATA FROM {self._table}",
                fetch="all",
            )

            failed: list[BaseDocument] = []
            for row in rows or []:
                doc = self._row_to_document(row)
                try:
                    migrated = await document_loader(doc)
                except Exception as exc:  # pragma: no cover - defensive logging
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

        base_clause, base_params = _build_where_clause(filters, self.INDEXED_FIELDS)
        params: dict[str, Any] = dict(base_params)

        cursor_clause, cursor_params = _build_cursor_clause(cursor, sort_direction)
        clause = base_clause
        if cursor_clause:
            clause = f"{clause} AND {cursor_clause}" if clause else f"WHERE {cursor_clause}"
            params.update(cursor_params)

        order_direction = "DESC" if sort_direction == SortDirection.DESC else "ASC"
        order_by = f"ORDER BY CREATION_UTC {order_direction}, ID {order_direction}"

        query_limit = (limit + 1) if limit else None
        limit_sql = f" LIMIT {query_limit}" if query_limit else ""

        sql = f"SELECT DATA FROM {self._table}"
        if clause:
            sql += f" {clause}"
        sql += f" {order_by}{limit_sql}"

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
        await self.ensure_table()
        clause, params = _build_where_clause(filters, self.INDEXED_FIELDS)
        sql = f"SELECT DATA FROM {self._table} {clause} LIMIT 1"
        row = await self._database._execute(sql, params, fetch="one")
        if not row:
            return None

        return cast(TDocument, self._row_to_document(row))

    async def insert_one(self, document: TDocument) -> InsertResult:
        await self.ensure_table()
        ensure_is_total(document, self._schema)

        params = self._serialize_document(document)
        sql = f"""
            INSERT INTO {self._table}
            (ID, VERSION, CREATION_UTC, SESSION_ID, CUSTOMER_ID, AGENT_ID, DATA)
            SELECT
                V.ID,
                V.VERSION,
                V.CREATION_UTC,
                V.SESSION_ID,
                V.CUSTOMER_ID,
                V.AGENT_ID,
                PARSE_JSON(V.DATA_RAW)
            FROM VALUES (
                %(id)s,
                %(version)s,
                %(creation_utc)s,
                %(session_id)s,
                %(customer_id)s,
                %(agent_id)s,
                %(data)s
            ) AS V(ID, VERSION, CREATION_UTC, SESSION_ID, CUSTOMER_ID, AGENT_ID, DATA_RAW)
        """

        await self._database._execute(sql, params)
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
        if isinstance(row, Mapping):
            data = row.get("DATA")
        else:
            data = row[0]

        if isinstance(data, str):
            return cast(BaseDocument, json.loads(data))

        return cast(BaseDocument, data)

    async def _replace_document(self, document: TDocument) -> None:
        params = self._serialize_document(document)
        sql = f"""
            UPDATE {self._table}
            SET VERSION=%(version)s,
                CREATION_UTC=%(creation_utc)s,
                SESSION_ID=%(session_id)s,
                CUSTOMER_ID=%(customer_id)s,
                AGENT_ID=%(agent_id)s,
                DATA=PARSE_JSON(%(data)s)
            WHERE ID=%(id)s
        """
        await self._database._execute(sql, params)

    async def _delete_documents(self, identifiers: Sequence[Any]) -> None:
        if not identifiers:
            return

        placeholders = ", ".join(f"%(id_{i})s" for i in range(len(identifiers)))
        params = {f"id_{i}": _stringify(value) for i, value in enumerate(identifiers)}
        sql = f"DELETE FROM {self._table} WHERE ID IN ({placeholders})"
        await self._database._execute(sql, params)

    async def _persist_failed_documents(self, documents: Sequence[BaseDocument]) -> None:
        if not documents:
            return

        for doc in documents:
            params = {
                "id": _stringify(doc.get("id")),
                "data": json.dumps(doc, ensure_ascii=False),
            }

            sql = f"""
                INSERT INTO {self._failed_table} (ID, DATA)
                SELECT
                    V.ID,
                    PARSE_JSON(V.DATA_RAW)
                FROM VALUES (%(id)s, %(data)s) AS V(ID, DATA_RAW)
            """
            await self._database._execute(sql, params)

    def _serialize_document(self, document: TDocument) -> MutableMapping[str, Any]:
        return {
            "id": _stringify(document["id"]),
            "version": document.get("version"),
            "creation_utc": document.get("creation_utc"),
            "session_id": _stringify(document.get("session_id")),
            "customer_id": _stringify(document.get("customer_id")),
            "agent_id": _stringify(document.get("agent_id")),
            "data": json.dumps(document, ensure_ascii=False),
        }


def _build_where_clause(filters: Where, indexed_fields: set[str]) -> tuple[str, Mapping[str, Any]]:
    if not filters:
        return "", {}

    translator = _WhereTranslator(indexed_fields)
    clause = translator.render(filters)
    if not clause:
        return "", {}

    return f"WHERE {clause}", translator.params


def _build_cursor_clause(
    cursor: Cursor | None,
    sort_direction: SortDirection,
) -> tuple[str, Mapping[str, Any]]:
    if cursor is None:
        return "", {}

    creation_operator = "<" if sort_direction == SortDirection.DESC else ">"
    id_operator = "<" if sort_direction == SortDirection.DESC else ">"

    clause = (
        f"(CREATION_UTC {creation_operator} %(cursor_creation)s "
        f"OR (CREATION_UTC = %(cursor_creation)s AND ID {id_operator} %(cursor_id)s))"
    )

    params = {
        "cursor_creation": cursor.creation_utc,
        "cursor_id": str(cursor.id),
    }

    return clause, params


class _WhereTranslator:
    def __init__(self, indexed_fields: set[str]) -> None:
        self._indexed_fields = indexed_fields
        self._params: dict[str, Any] = {}
        self._counter = 0

    @property
    def params(self) -> Mapping[str, Any]:
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
                    fragments.append(self._render_field(key, value))

            return " AND ".join(part for part in fragments if part)

        raise SnowflakeAdapterError("Unsupported filter format for Snowflake adapter")

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
                raise SnowflakeAdapterError(
                    f"Unsupported operator '{operator}' in Snowflake filter"
                )

        return " AND ".join(clauses)

    def _membership_clause(self, field: str, operand: Any, *, negate: bool) -> str:
        values = list(operand or [])
        if not values:
            return "1=1" if negate else "1=0"

        column, needs_variant = self._column_expr(field)
        placeholders: list[str] = []
        for value in values:
            name = self._add_param(value)
            placeholders.append(self._wrap_value(name, needs_variant))

        operator = "NOT IN" if negate else "IN"
        return f"{column} {operator} (" + ", ".join(placeholders) + ")"

    def _equality_clause(self, field: str, operand: Any) -> str:
        name = self._add_param(operand)
        column, needs_variant = self._column_expr(field)
        return f"{column} = {self._wrap_value(name, needs_variant)}"

    def _column_expr(self, field: str) -> tuple[str, bool]:
        sanitized = _sanitize_identifier(field)
        if field in self._indexed_fields:
            return f'"{sanitized}"', False

        json_path = json.dumps(field)
        return f"DATA:{json_path}", True

    def _wrap_value(self, placeholder: str, needs_variant: bool) -> str:
        return f"TO_VARIANT({placeholder})" if needs_variant else placeholder

    def _comparison_clause(self, field: str, operator: str, operand: Any) -> str:
        sql_operator = {
            "$gt": ">",
            "$gte": ">=",
            "$lt": "<",
            "$lte": "<=",
            "$ne": "!=",
        }[operator]

        name = self._add_param(operand)
        column, needs_variant = self._column_expr(field)
        return f"{column} {sql_operator} {self._wrap_value(name, needs_variant)}"

    def _add_param(self, value: Any) -> str:
        name = f"param_{self._counter}"
        self._counter += 1
        object_id_type = getattr(ObjectId, "__supertype__", str)
        if isinstance(value, object_id_type):
            value = str(value)
        self._params[name] = value
        return f"%({name})s"


__all__ = [
    "SnowflakeAdapterError",
    "SnowflakeDocumentCollection",
    "SnowflakeDocumentDatabase",
]
