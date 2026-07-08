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

from pathlib import Path
import json
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, cast

import aiosqlite
from typing_extensions import Self, override

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.loggers import Logger
from parlant.core.persistence.common import (
    Cursor,
    ObjectId,
    SortDirection,
    Where,
    ensure_is_total,
    matches_filters,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    CollectionIndex,
    CollectionSort,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
    FindResult,
    InsertResult,
    TDocument,
    UpdateResult,
)


class SQLiteDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        logger: Logger,
        file_path: Path,
    ) -> None:
        self._logger = logger
        self.file_path = file_path
        self._connection: aiosqlite.Connection | None = None
        self._collections: dict[str, SQLiteDocumentCollection[BaseDocument]] = {}

    async def __aenter__(self) -> Self:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.file_path)
        self._connection.row_factory = aiosqlite.Row
        await self._initialize()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        if self._connection:
            await self._connection.close()
            self._connection = None
        return False

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("SQLiteDocumentDatabase must be entered before use")
        return self._connection

    async def _initialize(self) -> None:
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                collection_name TEXT NOT NULL,
                id TEXT NOT NULL,
                creation_utc TEXT,
                document_json TEXT NOT NULL,
                PRIMARY KEY (collection_name, id)
            )
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_collection_creation_idx
            ON documents (collection_name, creation_utc, id)
            """
        )
        await self._db.commit()

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> SQLiteDocumentCollection[TDocument]:
        await self._db.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (name,))
        await self._db.execute("DELETE FROM documents WHERE collection_name = ?", (name,))
        await self._db.commit()

        collection = SQLiteDocumentCollection(
            database=self,
            name=name,
            schema=schema,
        )
        self._collections[name] = cast(SQLiteDocumentCollection[BaseDocument], collection)
        return collection

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> SQLiteDocumentCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(SQLiteDocumentCollection[TDocument], collection)

        async with self._db.execute("SELECT 1 FROM collections WHERE name = ?", (name,)) as cursor:
            if await cursor.fetchone() is None:
                raise ValueError(f'Collection "{name}" does not exist')

        collection = SQLiteDocumentCollection(
            database=self,
            name=name,
            schema=schema,
            document_loader=document_loader,
        )
        self._collections[name] = cast(SQLiteDocumentCollection[BaseDocument], collection)
        return collection

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> SQLiteDocumentCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(SQLiteDocumentCollection[TDocument], collection)

        await self._db.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (name,))
        await self._db.commit()

        collection = SQLiteDocumentCollection(
            database=self,
            name=name,
            schema=schema,
            document_loader=document_loader,
        )
        self._collections[name] = cast(SQLiteDocumentCollection[BaseDocument], collection)
        return collection

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        async with self._db.execute("SELECT 1 FROM collections WHERE name = ?", (name,)) as cursor:
            if await cursor.fetchone() is None:
                raise ValueError(f'Collection "{name}" does not exist')

        await self._db.execute("DELETE FROM documents WHERE collection_name = ?", (name,))
        await self._db.execute("DELETE FROM collections WHERE name = ?", (name,))
        await self._db.commit()
        self._collections.pop(name, None)


class SQLiteDocumentCollection(DocumentCollection[TDocument]):
    def __init__(
        self,
        database: SQLiteDocumentDatabase,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]] | None = None,
    ) -> None:
        self._database = database
        self._name = name
        self._schema = schema
        self._document_loader = document_loader or self._identity_loader
        self._lock = ReaderWriterLock()

    async def _identity_loader(self, doc: BaseDocument) -> TDocument:
        return cast(TDocument, doc)

    def _serialize_document(self, document: TDocument) -> str:
        return json.dumps(document, ensure_ascii=False, separators=(",", ":"))

    async def _deserialize_document(self, document_json: str) -> TDocument | None:
        document = cast(BaseDocument, json.loads(document_json))
        return await self._document_loader(document)

    async def _load_all_documents(self) -> list[TDocument]:
        async with self._database._db.execute(
            """
            SELECT document_json
            FROM documents
            WHERE collection_name = ?
            """,
            (self._name,),
        ) as cursor:
            rows = await cursor.fetchall()

        documents: list[TDocument] = []
        for row in rows:
            if document := await self._deserialize_document(str(row["document_json"])):
                documents.append(document)

        return documents

    async def _find_one_unlocked(
        self,
        filters: Where,
        sort: Optional[CollectionSort] = None,
    ) -> Optional[TDocument]:
        if sort is None and (document_id := self._extract_id_eq_filter(filters)):
            async with self._database._db.execute(
                """
                SELECT document_json
                FROM documents
                WHERE collection_name = ? AND id = ?
                """,
                (self._name, str(document_id)),
            ) as cursor:
                if row := await cursor.fetchone():
                    return await self._deserialize_document(str(row["document_json"]))
                return None

        matching_documents = [
            doc for doc in await self._load_all_documents() if matches_filters(filters, doc)
        ]

        if sort:
            matching_documents = self._apply_field_sort(matching_documents, sort)

        for doc in matching_documents:
            return doc

        return None

    def _apply_sort(
        self,
        documents: list[TDocument],
        sort_direction: SortDirection,
    ) -> list[TDocument]:
        docs = list(documents)
        reverse_order = sort_direction == SortDirection.DESC
        docs.sort(
            key=lambda d: (
                d.get("creation_utc") or "",
                d.get("id") or "",
            ),
            reverse=reverse_order,
        )
        return docs

    def _apply_field_sort(
        self,
        documents: Sequence[TDocument],
        sort: CollectionSort,
    ) -> list[TDocument]:
        docs = list(documents)

        for field_name, direction in reversed(sort):
            docs.sort(
                key=lambda d: cast(Any, d.get(field_name)),
                reverse=direction == SortDirection.DESC,
            )

        return docs

    def _apply_cursor_filter(
        self,
        documents: list[TDocument],
        cursor: Cursor,
        sort_direction: SortDirection,
    ) -> list[TDocument]:
        result = []

        for doc in documents:
            doc_creation_utc = str(doc.get("creation_utc", ""))
            doc_id = str(doc.get("id", ""))

            if sort_direction == SortDirection.DESC:
                if doc_creation_utc < cursor.creation_utc or (
                    doc_creation_utc == cursor.creation_utc and doc_id < cursor.id
                ):
                    result.append(doc)
            else:
                if doc_creation_utc > cursor.creation_utc or (
                    doc_creation_utc == cursor.creation_utc and doc_id > cursor.id
                ):
                    result.append(doc)

        return result

    def _extract_id_eq_filter(self, filters: Where) -> ObjectId | None:
        if set(filters.keys()) != {"id"}:
            return None

        maybe_id_filter = filters.get("id")
        if not isinstance(maybe_id_filter, Mapping):
            return None

        id_filter = cast(Mapping[str, Any], maybe_id_filter)
        if set(id_filter.keys()) != {"$eq"}:
            return None

        return ObjectId(str(id_filter["$eq"]))

    @override
    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        async with self._lock.reader_lock:
            documents = await self._load_all_documents()
            filtered_docs = [doc for doc in documents if matches_filters(filters, doc)]

            sort_direction = sort_direction or SortDirection.ASC
            filtered_docs = self._apply_sort(filtered_docs, sort_direction)

            if cursor:
                filtered_docs = self._apply_cursor_filter(filtered_docs, cursor, sort_direction)

            total_count = len(filtered_docs)
            has_more = False
            next_cursor = None

            if limit is not None and len(filtered_docs) > limit:
                has_more = True
                result_docs = filtered_docs[:limit]
                if result_docs:
                    last_doc = result_docs[-1]
                    next_cursor = Cursor(
                        creation_utc=str(last_doc.get("creation_utc", "")),
                        id=ObjectId(str(last_doc.get("id", ""))),
                    )
            else:
                result_docs = filtered_docs

            return FindResult(
                items=result_docs,
                total_count=total_count,
                has_more=has_more,
                next_cursor=next_cursor,
            )

    @override
    async def find_one(
        self,
        filters: Where,
        sort: Optional[CollectionSort] = None,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            return await self._find_one_unlocked(filters, sort)

    @override
    async def ensure_indexes(
        self,
        indexes: Sequence[CollectionIndex],
    ) -> None:
        return None

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        async with self._lock.writer_lock:
            await self._database._db.execute(
                """
                INSERT OR REPLACE INTO documents
                    (collection_name, id, creation_utc, document_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._name,
                    str(document["id"]),
                    str(document.get("creation_utc", "")),
                    self._serialize_document(document),
                ),
            )
            await self._database._db.commit()

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        async with self._lock.writer_lock:
            document = await self._find_one_unlocked(filters)
            if document:
                updated_document = cast(TDocument, {**document, **params})
                await self._database._db.execute(
                    """
                    INSERT OR REPLACE INTO documents
                        (collection_name, id, creation_utc, document_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        self._name,
                        str(updated_document["id"]),
                        str(updated_document.get("creation_utc", "")),
                        self._serialize_document(updated_document),
                    ),
                )
                await self._database._db.commit()

                return UpdateResult(
                    acknowledged=True,
                    matched_count=1,
                    modified_count=1,
                    updated_document=updated_document,
                )

        if upsert:
            await self.insert_one(params)
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
            document = await self._find_one_unlocked(filters)
            if document:
                await self._database._db.execute(
                    """
                    DELETE FROM documents
                    WHERE collection_name = ? AND id = ?
                    """,
                    (self._name, str(document["id"])),
                )
                await self._database._db.commit()

                return DeleteResult(
                    acknowledged=True,
                    deleted_count=1,
                    deleted_document=document,
                )

        return DeleteResult(
            acknowledged=True,
            deleted_count=0,
            deleted_document=None,
        )
