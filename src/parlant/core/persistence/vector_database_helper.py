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

import heapq
from typing import Awaitable, Callable, Generic, Mapping, Optional, Sequence, TypeVar, cast
from typing_extensions import Self
from parlant.core.common import Version
from parlant.core.nlp.embedding import Embedder
from parlant.core.persistence.common import MigrationRequired, ServerOutdated, VersionedStore
from parlant.core.persistence.vector_database import BaseDocument, TDocument, VectorDatabase


async def query_chunks(query: str, embedder: Embedder) -> list[str]:
    max_length = embedder.max_tokens // 5
    total_token_count = await embedder.tokenizer.estimate_token_count(query)

    words = query.split()
    total_word_count = len(words)

    tokens_per_word = total_token_count / total_word_count

    words_per_chunk = max(int(max_length / tokens_per_word), 1)

    chunks = []
    for i in range(0, total_word_count, words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return [text if await embedder.tokenizer.estimate_token_count(text) else "" for text in chunks]


T = TypeVar("T")


def calculate_min_vectors_for_max_item_count(
    items: Sequence[T],
    count_item_vectors: Callable[[T], int],
    max_items_to_return: int,
) -> int:
    # Vector databases return the top `top_k` vectors globally
    # — not grouped by item (which may have multiple vectors).
    #
    # So if we set `top_k = max_documents`, it's likely that the results will include
    # fewer than `max_documents` unique items, since a single item may have multiple vectors.
    #
    # To guarantee that we retrieve (up to) `max_documents` unique items, we could:
    # 1. Count how many vectors each item has (from the available items).
    # 2. Sum the vector counts.
    # 3. Filter duplicates by item.
    # 4. Sort by distance.
    # 5. Select the top `max_items` distinct items.
    #
    # To optimize this process, we would like to estimate the minimum number of items (`top_k`)
    # needed to ensure that at least `max_items` unique items are likely to be represented.
    #
    # We do this by:
    # 1. Counting how many vectors (e.g., content entries) each item has.
    # 2. Selecting the top `max_items` items with the most vectors (`heapq.nlargest`).
    # 3. Summing their vector counts to compute `top_k`.
    #
    # Example:
    # - 3 items with 10, 15, and 20 vectors → max_items = 2
    # - Top two items have 20 and 15 → top_k = 35
    # - Instead of fetching all 45, we fetch just 35 — enough to likely include the most relevant items.
    return sum(
        heapq.nlargest(
            max_items_to_return,
            [count_item_vectors(i) for i in items],
        )
    )


class VectorDocumentStoreMigrationHelper:
    def __init__(
        self,
        store: VersionedStore,
        database: VectorDatabase,
        allow_migration: bool,
    ):
        self._store_name = store.__class__.__name__
        self._runtime_store_version = store.VERSION.to_string()
        self._database = database
        self._allow_migration = allow_migration

    @staticmethod
    def get_store_version_key(store_name: str) -> str:
        return f"{store_name}_version"

    async def __aenter__(self) -> Self:
        migration_required = await self._is_migration_required(
            self._database,
            self._runtime_store_version,
        )

        if migration_required and not self._allow_migration:
            raise MigrationRequired(f"Migration required for {self._store_name}.")

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        if exc_type is None:
            await self._update_metadata_version(
                self._database,
                self._runtime_store_version,
            )

        return False

    async def _is_migration_required(
        self,
        database: VectorDatabase,
        runtime_store_version: Version.String,
    ) -> bool:
        metadata = await database.read_metadata()
        key = self.get_store_version_key(self._store_name)
        if key in metadata:
            if Version.from_string(cast(str, metadata[key])) > Version.from_string(
                runtime_store_version
            ):
                raise ServerOutdated

            return metadata[key] != runtime_store_version
        else:
            await database.upsert_metadata(key, runtime_store_version)
            return False  # No migration is required for a new store

    async def _update_metadata_version(
        self,
        database: VectorDatabase,
        runtime_store_version: Version.String,
    ) -> None:
        await database.upsert_metadata("version", runtime_store_version)


class VectorDocumentMigrationHelper(Generic[TDocument]):
    def __init__(
        self,
        versioned_store: VersionedStore,
        converters: Mapping[str, Callable[[BaseDocument], Awaitable[Optional[BaseDocument]]]],
    ) -> None:
        self.target_version = versioned_store.VERSION.to_string()
        self.converters = converters

    async def migrate(self, doc: BaseDocument) -> Optional[TDocument]:
        while doc["version"] != self.target_version:
            if converted_doc := await self.converters[doc["version"]](doc):
                doc = converted_doc
            else:
                return None

        return cast(TDocument, doc)
