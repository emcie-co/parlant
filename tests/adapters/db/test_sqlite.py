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

from pathlib import Path
from typing import TypedDict

from parlant.adapters.db.sqlite import SQLiteDocumentDatabase
from parlant.core.common import Version
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import BasicEmbeddingCache, NullEmbedder
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import identity_loader_for


class _TestLogger:
    def info(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass

    def debug(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass


class _DummyDocument(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    version: Version.String
    name: str


def _logger() -> Logger:
    return _TestLogger()  # type: ignore[return-value]


async def test_that_sqlite_document_database_persists_documents(tmp_path: Path) -> None:
    db_path = tmp_path / "documents.sqlite"

    async with SQLiteDocumentDatabase(_logger(), db_path) as db:
        collection = await db.get_or_create_collection(
            name="documents",
            schema=_DummyDocument,
            document_loader=identity_loader_for(_DummyDocument),
        )

        await collection.insert_one(
            _DummyDocument(
                id=ObjectId("doc-1"),
                creation_utc="2026-01-01T00:00:00Z",
                version=Version.String("1.0.0"),
                name="First",
            )
        )

    async with SQLiteDocumentDatabase(_logger(), db_path) as db:
        collection = await db.get_collection(
            name="documents",
            schema=_DummyDocument,
            document_loader=identity_loader_for(_DummyDocument),
        )

        document = await collection.find_one({"id": {"$eq": ObjectId("doc-1")}})

    assert document
    assert document["name"] == "First"


async def test_that_basic_embedding_cache_can_use_sqlite_document_database(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "embeddings.sqlite"

    async with SQLiteDocumentDatabase(_logger(), db_path) as db:
        cache = BasicEmbeddingCache(db)

        await cache.set(
            NullEmbedder,
            ["hello", "world"],
            [[0.1, 0.2], [0.3, 0.4]],
            hints={"scope": "test"},
        )

    async with SQLiteDocumentDatabase(_logger(), db_path) as db:
        cache = BasicEmbeddingCache(db)

        result = await cache.get(
            NullEmbedder,
            ["hello", "world"],
            hints={"scope": "test"},
        )

    assert result
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
