# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# You may not use this file except in compliance with the License.
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
from typing import Any, Generic, Mapping, Optional, Sequence, cast
import numpy as np
from typing_extensions import override, Self

import nano_vectordb  # type: ignore

from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.logging import Logger
from parlant.core.persistence.common import ensure_is_total, matches_filters, Where
from parlant.core.persistence.vector_database import (
    BaseDocument,
    DeleteResult,
    InsertResult,
    SimilarDocumentResult,
    UpdateResult,
    VectorCollection,
    VectorDatabase,
    TDocument,
)


class TransientVectorDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        embedder_factory: EmbedderFactory,
        embedder_type: type[Embedder],
    ) -> None:
        self._logger = logger
        self._embedder_factory = embedder_factory
        self._embedder_type = embedder_type

        self._database: nano_vectordb.MultiTenantNanoVDB
        self._collection_name_to_tenant_id: dict[str, str] = {}
        self._collections: dict[str, TransientVectorCollection[BaseDocument]] = {}

    async def __aenter__(self) -> Self:
        self._embedder = self._embedder_factory.create_embedder(self._embedder_type)
        self._database = nano_vectordb.MultiTenantNanoVDB(self._embedder.dimensions)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> TransientVectorCollection[TDocument]:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        tenant_id = self._database.create_tenant()
        self._collection_name_to_tenant_id[name] = tenant_id

        self._collections[name] = TransientVectorCollection(
            self._logger,
            nano_db=self._database.get_tenant(tenant_id),
            name=name,
            schema=schema,
            embedder=self._embedder_factory.create_embedder(embedder_type),
        )

        return cast(TransientVectorCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
    ) -> TransientVectorCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(TransientVectorCollection[TDocument], collection)

        raise ValueError(f'Transient collection "{name}" not found.')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> TransientVectorCollection[TDocument]:
        if collection := self._collections.get(name):
            assert schema == collection._schema
            return cast(TransientVectorCollection[TDocument], collection)

        tenant_id = self._database.create_tenant()
        self._collection_name_to_tenant_id[name] = tenant_id

        self._collections[name] = TransientVectorCollection(
            self._logger,
            nano_db=self._database.get_tenant(tenant_id),
            name=name,
            schema=schema,
            embedder=self._embedder_factory.create_embedder(embedder_type),
        )

        return cast(TransientVectorCollection[TDocument], self._collections[name])

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')
        self._database.delete_tenant(self._collection_name_to_tenant_id[name])
        del self._collections[name]


class TransientVectorCollection(Generic[TDocument], VectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        nano_db: nano_vectordb.NanoVectorDB,
        name: str,
        schema: type[TDocument],
        embedder: Embedder,
    ) -> None:
        self._logger = logger
        self._name = name
        self._schema = schema
        self._embedder = embedder

        self._lock = asyncio.Lock()
        self._nano_db = nano_db
        self._documents: list[TDocument] = []

    @staticmethod
    def _build_filter_lambda(
        filters: Where,
    ) -> nano_vectordb.dbs.ConditionLambda:
        def filter_lambda(candidate: Mapping[str, Any]) -> bool:
            return matches_filters(filters, candidate)

        return filter_lambda

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        result = []
        for doc in filter(
            lambda d: matches_filters(filters, d),
            self._documents,
        ):
            result.append(doc)

        return result

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        for doc in self._documents:
            if matches_filters(filters, doc):
                return doc

        return None

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        embeddings = list((await self._embedder.embed([document["content"]])).vectors)
        vector = np.array(embeddings[0], dtype=np.float32)

        data = {**document, "__id__": document["id"], "__vector__": vector}

        async with self._lock:
            self._nano_db.upsert([data])
            self._documents.append(document)

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        async with self._lock:
            for i, doc in enumerate(self._documents):
                if matches_filters(filters, doc):
                    if "content" in params:
                        embeddings = list((await self._embedder.embed([params["content"]])).vectors)
                    else:
                        embeddings = list((await self._embedder.embed([doc["content"]])).vectors)

                    vector = np.array(embeddings[0], dtype=np.float32)
                    data = {**params, "__id__": doc["id"], "__vector__": vector}

                    self._nano_db.upsert([data])
                    self._documents[i] = cast(TDocument, {**self._documents[i], **params})

                    return UpdateResult(
                        acknowledged=True,
                        matched_count=1,
                        modified_count=1,
                        updated_document=self._documents[i],
                    )

            if upsert:
                ensure_is_total(params, self._schema)
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
        for i, d in enumerate(self._documents):
            if matches_filters(filters, d):
                document = self._documents.pop(i)

                self._nano_db.delete([d["id"]])

                return DeleteResult(deleted_count=1, acknowledged=True, deleted_document=document)

        return DeleteResult(
            acknowledged=True,
            deleted_count=0,
            deleted_document=None,
        )

    async def find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
    ) -> Sequence[SimilarDocumentResult[TDocument]]:
        if not self._documents:
            return []

        query_embeddings = list((await self._embedder.embed([query])).vectors)
        vector = np.array(query_embeddings[0], dtype=np.float32)

        keys_to_exclude = {"__id__", "__metrics__"}

        docs = [
            {key: value for key, value in d.items() if key not in keys_to_exclude}
            for d in self._nano_db.query(
                query=vector,
                top_k=k,
                filter_lambda=self._build_filter_lambda(filters),
            )
        ]

        self._logger.debug(f"Similar documents found\n{json.dumps(docs, indent=2)}")

        return [
            SimilarDocumentResult(
                document=cast(TDocument, d),
                distance=i,
            )
            for i, d in enumerate(docs)
        ]
