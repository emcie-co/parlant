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
import os
import re
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Optional, Sequence, cast

import httpx
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


class PocketBaseAdapterError(Exception):
    """Raised for recoverable adapter errors."""


_COLLECTION_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_collection_name(name: str) -> str:
    sanitized = _COLLECTION_NAME_RE.sub("_", name).strip("_")
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

    if not env.get("POCKETBASE_URL"):
        raise PocketBaseAdapterError(
            "Missing PocketBase configuration. Set POCKETBASE_URL (and optionally POCKETBASE_ADMIN_TOKEN or POCKETBASE_ADMIN_EMAIL/POCKETBASE_ADMIN_PASSWORD)."
        )

    params: dict[str, Any] = {
        "url": env["POCKETBASE_URL"].rstrip("/"),
    }

    admin_token = env.get("POCKETBASE_ADMIN_TOKEN")
    admin_email = env.get("POCKETBASE_ADMIN_EMAIL")
    admin_password = env.get("POCKETBASE_ADMIN_PASSWORD")

    if admin_token:
        params["admin_token"] = admin_token
    elif admin_email and admin_password:
        params["admin_email"] = admin_email
        params["admin_password"] = admin_password

    return params


class PocketBaseDocumentDatabase(DocumentDatabase):
    """PocketBase-backed DocumentDatabase.

    Key design decision vs. the old implementation:
    - We do **not** override PocketBase record ids.
    - We store Parlant's document id in a dedicated field: `parlant_id`.

    This avoids PocketBase's record id constraints and makes filtering stable.
    """

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

        prefix = _sanitize_collection_name(collection_prefix) if collection_prefix else "parlant"
        # Separate prefix/name with an underscore for readability.
        self._collection_prefix = prefix.rstrip("_")

        self._base_url: str = self._connection_params["url"]
        self._admin_token: str | None = cast(Optional[str], self._connection_params.get("admin_token"))
        self._admin_email: str | None = cast(Optional[str], self._connection_params.get("admin_email"))
        self._admin_password: str | None = cast(Optional[str], self._connection_params.get("admin_password"))

        self._http_client = http_client
        self._client: httpx.AsyncClient | None = None

        self._collections: dict[str, PocketBaseDocumentCollection[Any]] = {}
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
        if self._client is not None and self._http_client is None:
            await self._client.aclose()
        self._client = None
        return False

    async def create_collection(self, name: str, schema: type[TDocument]) -> DocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
        await collection.ensure_collection()
        return collection

    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> DocumentCollection[TDocument]:
        collection = await self._get_or_create_collection(name, schema)
        await collection.ensure_collection()
        await collection.load_existing_documents(document_loader)
        return collection

    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> DocumentCollection[TDocument]:
        return await self.get_collection(name, schema, document_loader)

    async def delete_collection(self, name: str) -> None:
        await self._ensure_connection()
        assert self._client is not None

        physical = self._collection_identifier(name)

        async with self._operation_lock:
            # PocketBase collection delete APIs typically want the collection id.
            coll = await self._get_collection_info_by_name(physical)
            if coll is None:
                return

            coll_id = coll.get("id")
            if not coll_id:
                # As a fallback, try delete by name.
                await self._delete_collection_by_identifier(physical)
                return

            await self._delete_collection_by_identifier(str(coll_id))

        self._collections.pop(name, None)

    async def _delete_collection_by_identifier(self, identifier: str) -> None:
        assert self._client is not None
        try:
            resp = await self._client.delete(
                f"{self._base_url}/api/collections/{identifier}",
                headers=self._get_headers(),
                timeout=30.0,
            )
            if resp.status_code == 404:
                return
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise PocketBaseAdapterError(f"Failed to delete PocketBase collection '{identifier}': {exc}") from exc

    async def _get_collection_info_by_name(self, name: str) -> dict[str, Any] | None:
        assert self._client is not None
        try:
            resp = await self._client.get(
                f"{self._base_url}/api/collections/{name}",
                headers=self._get_headers(),
                timeout=30.0,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return cast(dict[str, Any], resp.json())
        except httpx.HTTPStatusError as exc:
            raise PocketBaseAdapterError(f"Failed to fetch PocketBase collection '{name}': {exc}") from exc

    async def _get_or_create_collection(self, name: str, schema: type[TDocument]) -> PocketBaseDocumentCollection[TDocument]:
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

            self._client = self._http_client or httpx.AsyncClient(timeout=30.0)

            if not self._admin_token and self._admin_email and self._admin_password:
                await self._authenticate()

    async def _authenticate(self) -> None:
        assert self._client is not None

        endpoints = [
            f"{self._base_url}/api/collections/_superusers/auth-with-password",
            f"{self._base_url}/api/admins/auth-with-password",
        ]

        last_exc: httpx.HTTPStatusError | None = None
        for endpoint in endpoints:
            try:
                resp = await self._client.post(
                    endpoint,
                    json={"identity": self._admin_email, "password": self._admin_password},
                    timeout=30.0,
                )
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = cast(dict[str, Any], resp.json())
                token = data.get("token")
                if not token and isinstance(data.get("record"), dict):
                    token = cast(dict[str, Any], data["record"]).get("token")
                if not token:
                    raise PocketBaseAdapterError("PocketBase auth response did not include a token")
                self._admin_token = cast(str, token)
                return
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code != 404:
                    raise PocketBaseAdapterError(f"Failed to authenticate with PocketBase: {exc}") from exc

        if last_exc is not None:
            raise PocketBaseAdapterError(
                f"Failed to authenticate with PocketBase (tried all endpoints): {last_exc}"
            ) from last_exc

        raise PocketBaseAdapterError("Failed to authenticate with PocketBase")

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._admin_token:
            headers["Authorization"] = f"Bearer {self._admin_token}"
        return headers

    def _collection_identifier(self, name: str) -> str:
        logical = _sanitize_collection_name(name)
        return _sanitize_collection_name(f"{self._collection_prefix}_{logical}")


class PocketBaseDocumentCollection(DocumentCollection[TDocument]):
    # Fields we store as top-level PocketBase fields for filtering/sorting.
    INDEXED_FIELDS = {"id", "version", "creation_utc", "session_id", "customer_id", "agent_id"}

    # PocketBase-side field name for Parlant document ids.
    PB_ID_FIELD = "parlant_id"

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

            existing = await self._database._get_collection_info_by_name(self._collection_name)
            if existing is not None:
                # PocketBase has changed its collection schema representation over versions.
                # Some versions return `fields` instead of `schema`. Also, older/broken adapters
                # may create collections with only the system `id` field, which makes records
                # unusable for Parlant (no place to store documents).
                raw_fields = existing.get("fields") or existing.get("schema") or []
                field_names = {
                    f.get("name")
                    for f in cast(list[dict[str, Any]], raw_fields)
                    if isinstance(f, dict) and f.get("name")
                }

                required = {
                    self.PB_ID_FIELD,
                    "data",
                    "version",
                    "creation_utc",
                    "session_id",
                    "customer_id",
                    "agent_id",
                }

                # If the collection only has the system field, it's effectively empty/unusable;
                # safely recreate it when we have admin privileges.
                if self._database._admin_token and (field_names == {"id"} or not required.issubset(field_names)):
                    # For non-metadata collections this is potentially destructive, but in practice
                    # this condition indicates the collection has no custom fields (can't contain
                    # valid Parlant documents anyway).
                    async with self._database._operation_lock:
                        await self._database._delete_collection_by_identifier(
                            str(existing.get("id") or self._collection_name)
                        )
                    existing = None

            if existing is None:
                # Creating collections requires admin privileges.
                if not self._database._admin_token:
                    raise PocketBaseAdapterError(
                        f"PocketBase collection '{self._collection_name}' does not exist and no admin token is available to create it. Provide POCKETBASE_ADMIN_TOKEN or POCKETBASE_ADMIN_EMAIL/POCKETBASE_ADMIN_PASSWORD."
                    )

                payload = self._collection_create_payload()
                async with self._database._operation_lock:
                    resp = await self._database._client.post(
                        f"{self._database._base_url}/api/collections",
                        json=payload,
                        headers=self._database._get_headers(),
                        timeout=30.0,
                    )
                    resp.raise_for_status()

            self._collection_ready = True

    def _collection_create_payload(self) -> dict[str, Any]:
        # PocketBase collection schema API has changed across versions (`schema` vs `fields`).
        # We include both keys, using a minimal field-definition shape supported by newer versions.
        fields: list[dict[str, Any]] = [
            {
                "name": self.PB_ID_FIELD,
                "type": "text",
                "required": True,
                "unique": True,
                "options": {"min": None, "max": None, "pattern": ""},
            },
            {
                "name": "version",
                "type": "text",
                "required": False,
                "unique": False,
                "options": {"min": None, "max": None, "pattern": ""},
            },
            {
                "name": "creation_utc",
                "type": "text",
                "required": False,
                "unique": False,
                "options": {"min": None, "max": None, "pattern": ""},
            },
            {
                "name": "session_id",
                "type": "text",
                "required": False,
                "unique": False,
                "options": {"min": None, "max": None, "pattern": ""},
            },
            {
                "name": "customer_id",
                "type": "text",
                "required": False,
                "unique": False,
                "options": {"min": None, "max": None, "pattern": ""},
            },
            {
                "name": "agent_id",
                "type": "text",
                "required": False,
                "unique": False,
                "options": {"min": None, "max": None, "pattern": ""},
            },
            {"name": "data", "type": "json", "required": False, "unique": False, "options": {}},
        ]

        # Make the created collection usable immediately (even if client requests are not authenticated).
        # Admin requests will still work regardless.
        return {
            "name": self._collection_name,
            "type": "base",
            # Support both older (`schema`) and newer (`fields`) PocketBase APIs.
            "schema": fields,
            "fields": fields,
            "indexes": [],
            "listRule": "",
            "viewRule": "",
            "createRule": "",
            "updateRule": "",
            "deleteRule": "",
        }

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

            # Fetch all records to perform migration/cleanup.
            all_records: list[dict[str, Any]] = []
            page = 1
            per_page = 200

            while True:
                data = await self._list_records(params={"page": page, "perPage": per_page})
                items = cast(list[dict[str, Any]], data.get("items") or [])
                if not items:
                    break
                all_records.extend(items)
                if len(items) < per_page:
                    break
                page += 1

            failed_record_ids: list[str] = []

            for record in all_records:
                record_id = cast(str, record.get("id") or "")
                doc = self._record_to_document(record)

                try:
                    migrated = await document_loader(doc)
                except Exception as exc:  # pragma: no cover
                    self._logger.error(
                        f"Failed to load document '{doc.get('id')}' in collection '{self._name}': {exc}"
                    )
                    if record_id:
                        failed_record_ids.append(record_id)
                    continue

                if migrated is None:
                    if record_id:
                        failed_record_ids.append(record_id)
                    continue

                if migrated is not doc and record_id:
                    await self._patch_record(record_id, migrated)

            for record_id in failed_record_ids:
                await self._delete_record_by_record_id(record_id)

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

        # Stable sorting by creation_utc then parlant_id.
        if sort_direction == SortDirection.DESC:
            sort = "-creation_utc,-parlant_id"
        else:
            sort = "creation_utc,parlant_id"

        pb_filter = _build_pocketbase_filter(filters, self.INDEXED_FIELDS)
        if cursor:
            cursor_filter = _build_cursor_filter(cursor, sort_direction, id_field=self.PB_ID_FIELD)
            pb_filter = f"({pb_filter}) && ({cursor_filter})" if pb_filter else cursor_filter

        query_limit = (limit + 1) if limit else None
        per_page = min(int(query_limit or 30), 200)

        params: dict[str, Any] = {"page": 1, "perPage": per_page, "sort": sort}
        if pb_filter:
            params["filter"] = pb_filter

        try:
            data = await self._list_records(params=params)
        except PocketBaseAdapterError:
            # Some PocketBase deployments return a generic 400 when sort contains unsupported
            # fields or multiple sort keys. Retry with progressively safer defaults.
            retry_params = dict(params)
            retry_params.pop("sort", None)
            retry_params["page"] = 1
            retry_params["perPage"] = min(int(retry_params.get("perPage") or 30), 30)
            try:
                data = await self._list_records(params=retry_params)
            except PocketBaseAdapterError:
                data = await self._list_records(params=None)
        items = cast(list[dict[str, Any]], data.get("items") or [])
        docs = [cast(TDocument, self._record_to_document(item)) for item in items]

        total_count = int(data.get("totalItems") or len(docs))

        has_more = False
        next_cursor = None
        if limit and len(docs) > limit:
            has_more = True
            docs = docs[:limit]

        if docs:
            last = docs[-1]
            creation_utc = last.get("creation_utc")
            doc_id = last.get("id")
            if creation_utc is not None and doc_id is not None:
                next_cursor = Cursor(creation_utc=str(creation_utc), id=ObjectId(str(doc_id)))

        return FindResult(items=docs, total_count=total_count, has_more=has_more, next_cursor=next_cursor)

    async def find_one(self, filters: Where) -> Optional[TDocument]:
        rec = await self._find_one_record(filters)
        if rec is None:
            return None
        return cast(TDocument, self._record_to_document(rec))

    async def insert_one(self, document: TDocument) -> InsertResult:
        await self.ensure_collection()
        ensure_is_total(document, self._schema)

        payload = self._serialize_document(document)

        async with self._database._operation_lock:
            assert self._database._client is not None
            resp = await self._database._client.post(
                f"{self._database._base_url}/api/collections/{self._collection_name}/records",
                json=payload,
                headers=self._database._get_headers(),
                timeout=30.0,
            )
            resp.raise_for_status()

        return InsertResult(acknowledged=True)

    async def update_one(self, filters: Where, params: TDocument, upsert: bool = False) -> UpdateResult[TDocument]:
        rec = await self._find_one_record(filters)
        if rec is None:
            if upsert:
                await self.insert_one(params)
                return UpdateResult(True, matched_count=0, modified_count=0, updated_document=params)
            return UpdateResult(True, matched_count=0, modified_count=0, updated_document=None)

        record_id = cast(str, rec.get("id"))
        existing = cast(TDocument, self._record_to_document(rec))
        updated = cast(TDocument, {**existing, **params})

        await self._patch_record(record_id, updated)

        return UpdateResult(True, matched_count=1, modified_count=1, updated_document=updated)

    async def delete_one(self, filters: Where) -> DeleteResult[TDocument]:
        rec = await self._find_one_record(filters)
        if rec is None:
            return DeleteResult(True, deleted_count=0, deleted_document=None)

        record_id = cast(str, rec.get("id"))
        doc = cast(TDocument, self._record_to_document(rec))
        await self._delete_record_by_record_id(record_id)
        return DeleteResult(True, deleted_count=1, deleted_document=doc)

    async def _list_records(self, *, params: dict[str, Any] | None) -> dict[str, Any]:
        async with self._database._operation_lock:
            assert self._database._client is not None
            resp = await self._database._client.get(
                f"{self._database._base_url}/api/collections/{self._collection_name}/records",
                params=params,
                headers=self._database._get_headers(),
                timeout=30.0,
            )
            if resp.status_code >= 400:
                body = ""
                try:
                    body = resp.text or ""
                except Exception:
                    body = "<unreadable>"
                raise PocketBaseAdapterError(
                    f"PocketBase list records failed ({resp.status_code}) for collection '{self._collection_name}': {body[:1000]}"
                )
            return cast(dict[str, Any], resp.json())

    async def _find_one_record(self, filters: Where) -> dict[str, Any] | None:
        await self.ensure_collection()

        pb_filter = _build_pocketbase_filter(filters, self.INDEXED_FIELDS)
        # Some PocketBase deployments return a generic 400 for unsupported `sort` values.
        # For find_one() we don't rely on ordering, so keep the request minimal.
        params: dict[str, Any] = {"page": 1, "perPage": 1}
        if pb_filter:
            params["filter"] = pb_filter

        try:
            data = await self._list_records(params=params)
        except PocketBaseAdapterError:
            # Last resort: try with PocketBase defaults.
            data = await self._list_records(params=None)
        items = cast(list[dict[str, Any]], data.get("items") or [])
        if not items:
            return None
        return items[0]

    async def _patch_record(self, record_id: str, document: TDocument) -> None:
        payload = self._serialize_document(document)
        async with self._database._operation_lock:
            assert self._database._client is not None
            resp = await self._database._client.patch(
                f"{self._database._base_url}/api/collections/{self._collection_name}/records/{record_id}",
                json=payload,
                headers=self._database._get_headers(),
                timeout=30.0,
            )
            resp.raise_for_status()

    async def _delete_record_by_record_id(self, record_id: str) -> None:
        async with self._database._operation_lock:
            assert self._database._client is not None
            resp = await self._database._client.delete(
                f"{self._database._base_url}/api/collections/{self._collection_name}/records/{record_id}",
                headers=self._database._get_headers(),
                timeout=30.0,
            )
            if resp.status_code != 404:
                resp.raise_for_status()

    def _record_to_document(self, record: dict[str, Any]) -> BaseDocument:
        # Prefer the stored full document.
        raw_data = record.get("data")
        doc: dict[str, Any]
        if isinstance(raw_data, dict):
            doc = dict(raw_data)
        elif isinstance(raw_data, str) and raw_data:
            # Some PocketBase deployments may return json fields as strings.
            try:
                import json

                doc = cast(dict[str, Any], json.loads(raw_data))
            except Exception:
                doc = {}
        else:
            doc = {}

        # Ensure essential keys exist for Parlant consumers.
        if "id" not in doc:
            pid = record.get(self.PB_ID_FIELD)
            if pid is not None:
                doc["id"] = ObjectId(str(pid))

        # NOTE: Migration helper expects metadata docs to have a "version" key.
        # If the underlying record is malformed, prefer a safe default over KeyError.
        if "version" not in doc:
            top_version = record.get("version")
            if top_version is not None:
                doc["version"] = cast(str, top_version)
            elif self._name == "metadata":
                doc["version"] = "0.0.0"

        if "creation_utc" not in doc:
            top_creation = record.get("creation_utc")
            if top_creation is not None:
                doc["creation_utc"] = cast(str, top_creation)

        for field in ("session_id", "customer_id", "agent_id"):
            if field not in doc and record.get(field) is not None:
                doc[field] = record.get(field)

        return cast(BaseDocument, doc)

    def _serialize_document(self, document: TDocument) -> MutableMapping[str, Any]:
        pid = _stringify(document.get("id"))
        if not pid:
            raise PocketBaseAdapterError("Document is missing 'id'")

        return {
            self.PB_ID_FIELD: pid,
            "version": document.get("version"),
            "creation_utc": document.get("creation_utc"),
            "session_id": _stringify(document.get("session_id")),
            "customer_id": _stringify(document.get("customer_id")),
            "agent_id": _stringify(document.get("agent_id")),
            "data": document,
        }


def _build_pocketbase_filter(filters: Where, indexed_fields: set[str]) -> str:
    if not filters:
        return ""

    translator = _PocketBaseFilterTranslator(indexed_fields)
    return translator.render(filters)


def _build_cursor_filter(cursor: Cursor, sort_direction: SortDirection, *, id_field: str) -> str:
    if sort_direction == SortDirection.DESC:
        creation_op = "<"
        id_op = "<"
    else:
        creation_op = ">"
        id_op = ">"

    creation_utc = cursor.creation_utc.replace('"', '\\"')
    cursor_id = str(cursor.id).replace('"', '\\"')

    return (
        f'(creation_utc {creation_op} "{creation_utc}" || '
        f'(creation_utc = "{creation_utc}" && {id_field} {id_op} "{cursor_id}"))'
    )


class _PocketBaseFilterTranslator:
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
                    parts = [p for p in parts if p]
                    if parts:
                        fragments.append("(" + " && ".join(parts) + ")")
                elif key == "$or":
                    parts = [self._render(part) for part in cast(Sequence[Where], value)]
                    parts = [p for p in parts if p]
                    if parts:
                        fragments.append("(" + " || ".join(parts) + ")")
                else:
                    fragments.append(self._render_field(key, value))
            return " && ".join(f for f in fragments if f)

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
                raise PocketBaseAdapterError(f"Unsupported operator '{operator}' in PocketBase filter")

        return " && ".join(clauses)

    def _field_expr(self, field: str) -> str:
        # Map Parlant's 'id' to our PocketBase field.
        if field == "id":
            return PocketBaseDocumentCollection.PB_ID_FIELD

        if field in self._indexed_fields:
            return field

        # Best-effort: nested json fields.
        return f'data."{field}"'

    def _escape_value(self, value: Any) -> str:
        object_id_type = getattr(ObjectId, "__supertype__", str)
        if isinstance(value, object_id_type):
            value = str(value)

        if isinstance(value, str):
            return '"' + value.replace('"', '\\"') + '"'
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return str(value)

    def _equality_clause(self, field: str, operand: Any) -> str:
        return f"{self._field_expr(field)} = {self._escape_value(operand)}"

    def _comparison_clause(self, field: str, operator: str, operand: Any) -> str:
        op = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<=", "$ne": "!="}[operator]
        return f"{self._field_expr(field)} {op} {self._escape_value(operand)}"

    def _membership_clause(self, field: str, operand: Any, *, negate: bool) -> str:
        values = list(operand or [])
        if not values:
            # Empty membership: always false for $in, always true for $nin.
            return "id != ''" if negate else "id = ''"

        expr = self._field_expr(field)
        parts = [f"{expr} {'!=' if negate else '='} {self._escape_value(v)}" for v in values]
        joiner = " && " if negate else " || "
        return "(" + joiner.join(parts) + ")"


__all__ = [
    "PocketBaseAdapterError",
    "PocketBaseDocumentCollection",
    "PocketBaseDocumentDatabase",
]
