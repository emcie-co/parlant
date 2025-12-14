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
import json
import os
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

import httpx

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


class PocketBaseAdapterError(Exception):
    """Raised for recoverable adapter errors."""


_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def _sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to PocketBase requirements."""
    # PocketBase collection names must be alphanumeric + underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not sanitized:
        raise PocketBaseAdapterError("PocketBase collection name cannot be empty")
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized.lower()


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
        "POCKETBASE_URL",
    ]

    missing = [key for key in required if not env.get(key)]
    if missing:
        raise PocketBaseAdapterError(
            "Missing PocketBase configuration. Set the following environment variables: "
            + ", ".join(missing)
        )

    params: dict[str, Any] = {
        "url": env["POCKETBASE_URL"].rstrip("/"),
    }

    # Optional authentication
    admin_email = env.get("POCKETBASE_ADMIN_EMAIL")
    admin_password = env.get("POCKETBASE_ADMIN_PASSWORD")
    admin_token = env.get("POCKETBASE_ADMIN_TOKEN")

    if admin_token:
        params["admin_token"] = admin_token
    elif admin_email and admin_password:
        params["admin_email"] = admin_email
        params["admin_password"] = admin_password
    # If neither is provided, we'll work without admin auth (public collections only)

    return params


class PocketBaseDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        logger: Logger,
        connection_params: Mapping[str, Any] | None = None,
        *,
        collection_prefix: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._logger = logger
        self._connection_params = (
            dict(connection_params)
            if connection_params is not None
            else _load_connection_params_from_env()
        )
        self._collection_prefix = (
            _sanitize_collection_name(collection_prefix) if collection_prefix else "parlant_"
        )
        self._http_client = http_client

        self._base_url = self._connection_params["url"]
        self._admin_token: str | None = self._connection_params.get("admin_token")
        self._admin_email: str | None = self._connection_params.get("admin_email")
        self._admin_password: str | None = self._connection_params.get("admin_password")

        self._collections: dict[str, PocketBaseDocumentCollection[Any]] = {}
        self._client: httpx.AsyncClient | None = None
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
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        return False

    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> PocketBaseDocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
        await collection.ensure_collection()
        return collection

    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> PocketBaseDocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
        await collection.ensure_collection()
        await collection.load_existing_documents(document_loader)
        return collection

    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> PocketBaseDocumentCollection[TDocument]:
        return await self.get_collection(name, schema, document_loader)

    async def delete_collection(self, name: str) -> None:
        await self._ensure_connection()
        collection_name = self._collection_identifier(name)

        async with self._operation_lock:
            assert self._client is not None
            try:
                # Delete the collection via PocketBase API
                response = await self._client.delete(
                    f"{self._base_url}/api/collections/{collection_name}",
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    # Collection doesn't exist, that's fine
                    pass
                else:
                    raise PocketBaseAdapterError(
                        f"Failed to delete PocketBase collection '{collection_name}': {exc}"
                    ) from exc

        self._collections.pop(name, None)

    async def _get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> PocketBaseDocumentCollection[TDocument]:
        if name not in self._collections:
            self._collections[name] = PocketBaseDocumentCollection(
                database=self,
                name=name,
                schema=schema,
                logger=self._logger,
            )

        return cast(PocketBaseDocumentCollection[TDocument], self._collections[name])

    async def _ensure_connection(self) -> None:
        if self._client is not None:
            return

        async with self._connection_lock:
            if self._client is not None:
                return

            if self._http_client is not None:
                self._client = self._http_client
            else:
                self._client = httpx.AsyncClient(timeout=30.0)

            # Authenticate if credentials provided
            if not self._admin_token and self._admin_email and self._admin_password:
                await self._authenticate()

    async def _authenticate(self) -> None:
        """Authenticate with PocketBase admin API."""
        assert self._client is not None
        try:
            response = await self._client.post(
                f"{self._base_url}/api/admins/auth-with-password",
                json={
                    "identity": self._admin_email,
                    "password": self._admin_password,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            self._admin_token = data.get("token")
        except httpx.HTTPStatusError as exc:
            raise PocketBaseAdapterError(f"Failed to authenticate with PocketBase: {exc}") from exc

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers with authentication if available."""
        headers = {"Content-Type": "application/json"}
        if self._admin_token:
            headers["Authorization"] = f"Bearer {self._admin_token}"
        return headers

    def _collection_identifier(self, name: str) -> str:
        return _sanitize_collection_name(self._collection_prefix + name)


class PocketBaseDocumentCollection(DocumentCollection[TDocument]):
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
        database: PocketBaseDocumentDatabase,
        name: str,
        schema: type[TDocument],
        logger: Logger,
    ) -> None:
        self._database = database
        self._name = name
        self._schema = schema
        self._logger = logger

        self._collection_name = self._database._collection_identifier(name)
        self._collection_ready = False
        self._loader_done = False
        self._collection_lock = asyncio.Lock()
        self._loader_lock = asyncio.Lock()

    async def ensure_collection(self) -> None:
        if self._collection_ready:
            return

        async with self._collection_lock:
            if self._collection_ready:
                return

            await self._database._ensure_connection()
            assert self._database._client is not None

            # Check if collection exists
            try:
                async with self._database._operation_lock:
                    response = await self._database._client.get(
                        f"{self._database._base_url}/api/collections/{self._collection_name}",
                        headers=self._database._get_headers(),
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    if response.status_code == 200:
                        self._collection_ready = True
                        return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    # Collection doesn't exist, will create it
                    pass
                else:
                    raise

            # Create collection if it doesn't exist
            collection_schema = {
                "name": self._collection_name,
                "type": "base",
                "schema": [
                    {
                        "name": "id",
                        "type": "text",
                        "required": True,
                        "primary": True,
                    },
                    {
                        "name": "version",
                        "type": "text",
                        "required": False,
                    },
                    {
                        "name": "creation_utc",
                        "type": "text",
                        "required": False,
                    },
                    {
                        "name": "session_id",
                        "type": "text",
                        "required": False,
                    },
                    {
                        "name": "customer_id",
                        "type": "text",
                        "required": False,
                    },
                    {
                        "name": "agent_id",
                        "type": "text",
                        "required": False,
                    },
                    {
                        "name": "data",
                        "type": "json",
                        "required": False,
                    },
                ],
            }

            try:
                async with self._database._operation_lock:
                    response = await self._database._client.post(
                        f"{self._database._base_url}/api/collections",
                        json=collection_schema,
                        headers=self._database._get_headers(),
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    self._collection_ready = True
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    # Collection might already exist, try to get it
                    try:
                        response = await self._database._client.get(
                            f"{self._database._base_url}/api/collections/{self._collection_name}",
                            headers=self._database._get_headers(),
                            timeout=30.0,
                        )
                        if response.status_code == 200:
                            self._collection_ready = True
                            return
                    except httpx.HTTPStatusError:
                        pass
                raise PocketBaseAdapterError(
                    f"Failed to create PocketBase collection '{self._collection_name}': {exc}"
                ) from exc

    async def load_existing_documents(
        self,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        if self._loader_done:
            return

        async with self._loader_lock:
            if self._loader_done:
                return

            await self.ensure_collection()

            # Fetch all documents
            all_docs: list[BaseDocument] = []
            page = 1
            per_page = 500

            while True:
                async with self._database._operation_lock:
                    assert self._database._client is not None
                    response = await self._database._client.get(
                        f"{self._database._base_url}/api/collections/{self._collection_name}/records",
                        params={"page": page, "perPage": per_page},
                        headers=self._database._get_headers(),
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    doc = self._record_to_document(item)
                    all_docs.append(doc)

                if len(items) < per_page:
                    break
                page += 1

            failed: list[BaseDocument] = []
            for doc in all_docs:
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

            # Delete failed documents
            if failed:
                failed_ids = [doc["id"] for doc in failed if "id" in doc]
                for doc_id in failed_ids:
                    await self._delete_record(doc_id)

            self._loader_done = True

    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        await self.ensure_collection()

        sort_direction = sort_direction or SortDirection.ASC
        sort_field = "-creation_utc" if sort_direction == SortDirection.DESC else "creation_utc"

        # Build PocketBase filter
        pb_filter = _build_pocketbase_filter(filters, self.INDEXED_FIELDS)

        # Add cursor filter if provided
        if cursor:
            cursor_filter = _build_cursor_filter(cursor, sort_direction)
            if pb_filter:
                pb_filter = f"({pb_filter}) && ({cursor_filter})"
            else:
                pb_filter = cursor_filter

        # Calculate page size
        query_limit = (limit + 1) if limit else None

        async with self._database._operation_lock:
            assert self._database._client is not None
            params: dict[str, Any] = {
                "sort": sort_field,
                "perPage": query_limit or 500,
            }
            if pb_filter:
                params["filter"] = pb_filter

            response = await self._database._client.get(
                f"{self._database._base_url}/api/collections/{self._collection_name}/records",
                params=params,
                headers=self._database._get_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        items = data.get("items", [])
        documents = [cast(TDocument, self._record_to_document(item)) for item in items]

        total_count = data.get("totalItems", len(documents))
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
        await self.ensure_collection()
        pb_filter = _build_pocketbase_filter(filters, self.INDEXED_FIELDS)

        async with self._database._operation_lock:
            assert self._database._client is not None
            params: dict[str, Any] = {"perPage": 1}
            if pb_filter:
                params["filter"] = pb_filter

            response = await self._database._client.get(
                f"{self._database._base_url}/api/collections/{self._collection_name}/records",
                params=params,
                headers=self._database._get_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        items = data.get("items", [])
        if not items:
            return None

        return cast(TDocument, self._record_to_document(items[0]))

    async def insert_one(self, document: TDocument) -> InsertResult:
        await self.ensure_collection()
        ensure_is_total(document, self._schema)

        record = self._serialize_document(document)

        async with self._database._operation_lock:
            assert self._database._client is not None
            response = await self._database._client.post(
                f"{self._database._base_url}/api/collections/{self._collection_name}/records",
                json=record,
                headers=self._database._get_headers(),
                timeout=30.0,
            )
            response.raise_for_status()

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

        await self._delete_record(identifier)

        return DeleteResult(True, deleted_count=1, deleted_document=existing)

    def _record_to_document(self, record: dict[str, Any]) -> BaseDocument:
        """Convert PocketBase record to document."""
        # If data field exists, use it; otherwise reconstruct from record
        if "data" in record and record["data"]:
            if isinstance(record["data"], str):
                return cast(BaseDocument, json.loads(record["data"]))
            return cast(BaseDocument, record["data"])

        # Reconstruct document from record fields
        doc: dict[str, Any] = {}
        if "id" in record:
            doc["id"] = ObjectId(record["id"])
        if "version" in record:
            doc["version"] = record["version"]
        if "creation_utc" in record:
            doc["creation_utc"] = record["creation_utc"]
        if "session_id" in record:
            doc["session_id"] = record["session_id"]
        if "customer_id" in record:
            doc["customer_id"] = record["customer_id"]
        if "agent_id" in record:
            doc["agent_id"] = record["agent_id"]

        # Merge any additional fields from data if it's a dict
        if "data" in record and isinstance(record["data"], dict):
            doc.update(record["data"])

        return cast(BaseDocument, doc)

    async def _replace_document(self, document: TDocument) -> None:
        """Replace an existing document."""
        record = self._serialize_document(document)
        doc_id = _stringify(document.get("id"))

        async with self._database._operation_lock:
            assert self._database._client is not None
            response = await self._database._client.patch(
                f"{self._database._base_url}/api/collections/{self._collection_name}/records/{doc_id}",
                json=record,
                headers=self._database._get_headers(),
                timeout=30.0,
            )
            response.raise_for_status()

    async def _delete_record(self, identifier: Any) -> None:
        """Delete a record by ID."""
        doc_id = _stringify(identifier)
        if not doc_id:
            return

        async with self._database._operation_lock:
            assert self._database._client is not None
            try:
                response = await self._database._client.delete(
                    f"{self._database._base_url}/api/collections/{self._collection_name}/records/{doc_id}",
                    headers=self._database._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise PocketBaseAdapterError(
                        f"Failed to delete PocketBase record '{doc_id}': {exc}"
                    ) from exc

    def _serialize_document(self, document: TDocument) -> MutableMapping[str, Any]:
        """Serialize document to PocketBase record format."""
        return {
            "id": _stringify(document.get("id")),
            "version": document.get("version"),
            "creation_utc": document.get("creation_utc"),
            "session_id": _stringify(document.get("session_id")),
            "customer_id": _stringify(document.get("customer_id")),
            "agent_id": _stringify(document.get("agent_id")),
            "data": document,
        }


def _build_pocketbase_filter(filters: Where, indexed_fields: set[str]) -> str:
    """Build PocketBase filter string from Where clause."""
    if not filters:
        return ""

    translator = _PocketBaseFilterTranslator(indexed_fields)
    return translator.render(filters)


def _build_cursor_filter(
    cursor: Cursor,
    sort_direction: SortDirection,
) -> str:
    """Build cursor filter for pagination."""
    if sort_direction == SortDirection.DESC:
        creation_op = "<"
        id_op = "<"
    else:
        creation_op = ">"
        id_op = ">"

    # Escape values for PocketBase filter
    creation_utc_escaped = cursor.creation_utc.replace('"', '\\"')
    cursor_id_escaped = str(cursor.id).replace('"', '\\"')

    return (
        f'(creation_utc {creation_op} "{creation_utc_escaped}" || '
        f'(creation_utc = "{creation_utc_escaped}" && id {id_op} "{cursor_id_escaped}"))'
    )


class _PocketBaseFilterTranslator:
    """Translates Where filters to PocketBase filter syntax."""

    def __init__(self, indexed_fields: set[str]) -> None:
        self._indexed_fields = indexed_fields

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
                        fragments.append("(" + " && ".join(parts) + ")")
                elif key == "$or":
                    parts = [self._render(part) for part in cast(Sequence[Where], value)]
                    parts = [part for part in parts if part]
                    if parts:
                        fragments.append("(" + " || ".join(parts) + ")")
                else:
                    fragments.append(self._render_field(key, value))

            return " && ".join(part for part in fragments if part)

        raise PocketBaseAdapterError("Unsupported filter format for PocketBase adapter")

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
                raise PocketBaseAdapterError(
                    f"Unsupported operator '{operator}' in PocketBase filter"
                )

        return " && ".join(clauses)

    def _membership_clause(self, field: str, operand: Any, *, negate: bool) -> str:
        values = list(operand or [])
        if not values:
            return "id != ''" if negate else "id = ''"  # Always false/true

        field_expr = self._field_expr(field)

        # For single value, use equality
        if len(values) == 1:
            operator = "!=" if negate else "="
            value_expr = self._escape_value(values[0])
            return f"{field_expr} {operator} {value_expr}"

        # For multiple values, use OR conditions (more reliable than regex)
        conditions = []
        for value in values:
            value_expr = self._escape_value(value)
            if negate:
                conditions.append(f"{field_expr} != {value_expr}")
            else:
                conditions.append(f"{field_expr} = {value_expr}")

        if negate:
            # All values must not match (AND)
            return "(" + " && ".join(conditions) + ")"
        else:
            # At least one value must match (OR)
            return "(" + " || ".join(conditions) + ")"

    def _equality_clause(self, field: str, operand: Any) -> str:
        field_expr = self._field_expr(field)
        value_expr = self._escape_value(operand)
        return f"{field_expr} = {value_expr}"

    def _field_expr(self, field: str) -> str:
        """Get field expression for PocketBase filter."""
        if field in self._indexed_fields:
            return field
        # For nested fields in data, use JSON path
        return f'data."{field}"'

    def _escape_value(self, value: Any) -> str:
        """Escape value for PocketBase filter."""
        object_id_type = getattr(ObjectId, "__supertype__", str)
        if isinstance(value, object_id_type):
            value = str(value)

        if isinstance(value, str):
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif value is None:
            return "null"
        else:
            return str(value)

    def _comparison_clause(self, field: str, operator: str, operand: Any) -> str:
        pb_operator = {
            "$gt": ">",
            "$gte": ">=",
            "$lt": "<",
            "$lte": "<=",
            "$ne": "!=",
        }[operator]

        field_expr = self._field_expr(field)
        value_expr = self._escape_value(operand)
        return f"{field_expr} {pb_operator} {value_expr}"


__all__ = [
    "PocketBaseAdapterError",
    "PocketBaseDocumentCollection",
    "PocketBaseDocumentDatabase",
]
