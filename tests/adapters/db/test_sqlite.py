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

import asyncio
from pathlib import Path
from typing import TypedDict

from parlant.adapters.db.sqlite import SQLiteDocumentDatabase
from parlant.core.common import Version
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import BasicEmbeddingCache, NullEmbedder
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import BaseDocument, identity_loader_for


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


class _CASDocument(BaseDocument):
    name: str
    metadata_revision: int


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


async def test_that_sqlite_pagination_returns_exact_total(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "pagination.sqlite"

    async with SQLiteDocumentDatabase(_logger(), db_path) as db:
        collection = await db.get_or_create_collection(
            name="documents",
            schema=_DummyDocument,
            document_loader=identity_loader_for(_DummyDocument),
        )
        for index in range(5):
            await collection.insert_one(
                _DummyDocument(
                    id=ObjectId(f"doc-{index}"),
                    creation_utc=f"2026-01-01T00:00:0{index}Z",
                    version=Version.String("1.0.0"),
                    name=f"Document {index}",
                )
            )

        first_page = await collection.find({}, limit=3)
        assert first_page.total_count == 5
        assert first_page.next_cursor is not None

        second_page = await collection.find(
            {},
            limit=3,
            cursor=first_page.next_cursor,
        )
        assert len(second_page.items) == 2
        assert second_page.total_count == 5


async def test_that_basic_embedding_cache_can_use_sqlite_document_database(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "embeddings.sqlite"

    async with SQLiteDocumentDatabase(_logger(), db_path) as db:
        cache = BasicEmbeddingCache(db)
        embedder = NullEmbedder()

        await cache.set(
            embedder,
            ["hello", "world"],
            [[0.1, 0.2], [0.3, 0.4]],
            input_tokens=[3, 4],
            hints={"scope": "test"},
        )

    async with SQLiteDocumentDatabase(_logger(), db_path) as db:
        cache = BasicEmbeddingCache(db)
        embedder = NullEmbedder()

        result = await cache.get(
            embedder,
            ["hello", "world"],
            hints={"scope": "test"},
        )

    assert result
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    # A hit serves the whole cached batch: its tokens come back as cached input
    # tokens (priced at zero), with no provider request.
    assert result.usage.input_tokens == 7
    assert result.usage.cached_input_tokens == 7
    assert result.usage.requests == 0


async def test_that_sqlite_metadata_revision_compare_and_swap_has_one_winner(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata-cas.sqlite"

    async with SQLiteDocumentDatabase(_logger(), db_path) as first_db:
        first = await first_db.get_or_create_collection(
            name="documents",
            schema=_CASDocument,
            document_loader=identity_loader_for(_CASDocument),
        )
        await first.insert_one(
            _CASDocument(
                id=ObjectId("doc-1"),
                creation_utc="2026-01-01T00:00:00Z",
                version=Version.String("1.0.0"),
                name="Initial",
                metadata_revision=0,
            )
        )

        async with SQLiteDocumentDatabase(_logger(), db_path) as second_db:
            second = await second_db.get_collection(
                name="documents",
                schema=_CASDocument,
                document_loader=identity_loader_for(_CASDocument),
            )

            results = await asyncio.gather(
                first.update_one(
                    {
                        "id": {"$eq": ObjectId("doc-1")},
                        "metadata_revision": {"$eq": 0},
                    },
                    {"name": "First", "metadata_revision": 1},
                ),
                second.update_one(
                    {
                        "id": {"$eq": ObjectId("doc-1")},
                        "metadata_revision": {"$eq": 0},
                    },
                    {"name": "Second", "metadata_revision": 1},
                ),
            )

            assert sorted(result.matched_count for result in results) == [0, 1]
            stored = await first.find_one({"id": {"$eq": ObjectId("doc-1")}})
            assert stored is not None
            assert stored["metadata_revision"] == 1
            assert stored["name"] in {"First", "Second"}
