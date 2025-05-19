# Copyright 2024 Emcie Co Ltd.
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

import json
import uuid
from typing import Awaitable, Callable, Generic, Optional, Sequence, cast

from qdrant_client import QdrantClient, grpc, models
from typing_extensions import Self, override

from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.common import Where, ensure_is_total
from parlant.core.persistence.vector_database import (
    BaseDocument,
    DeleteResult,
    InsertResult,
    SimilarDocumentResult,
    TDocument,
    UpdateResult,
    VectorCollection,
    VectorDatabase,
)

METADATA_POINT_ID = "11111111-1111-1111-1111-111111111111"
METADATA_COLLECTION_NAME = "_PARLANT_METADATA"


class QdrantCollection(Generic[TDocument], VectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        client: QdrantClient,
        name: str,
        schema: type[TDocument],
        embedder: Embedder,
        version: int,
    ) -> None:
        self._logger = logger
        self._client = client
        self._name = name
        self._schema = schema
        self._embedder = embedder
        self._version = version
        self._vector_size = embedder.dimensions

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        if not self._client.collection_exists(collection_name=self._name):
            self._client.create_collection(
                collection_name=self._name,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def _build_condition(self, operator, key, value):
        # See https://qdrant.tech/documentation/concepts/filtering/#range
        if operator == "$eq":
            return models.FieldCondition(key=key, match=models.MatchValue(value=value))
        elif operator == "$lt":
            return models.FieldCondition(key=key, range=models.Range(lt=value))
        elif operator == "$lte":
            return models.FieldCondition(key=key, range=models.Range(lte=value))
        elif operator == "$gt":
            return models.FieldCondition(key=key, range=models.Range(gt=value))
        elif operator == "$gte":
            return models.FieldCondition(key=key, range=models.Range(gte=value))
        elif operator == "$ne":
            return models.FieldCondition(key=key, match=models.MatchExcept(**{"except": [value]}))
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def _translate_where(self, filters: Where) -> models.Filter:
        if not filters:
            return None

        conditions = []
        for field, value in filters.items():
            if isinstance(value, list):
                if field == "$and":
                    return models.Filter(must=[self._translate_where(child) for child in value])
                elif field == "$or":
                    return models.Filter(should=[self._translate_where(child) for child in value])
            elif isinstance(value, dict):
                for op, op_value in value.items():
                    conditions.append(self._build_condition(op, field, op_value))
        return models.Filter(must=conditions)

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        SCROLL_SIZE = 64

        points = []
        next_offset = None
        stop_scrolling = False
        while not stop_scrolling:
            results, next_offset = self._client.scroll(
                collection_name=self._name,
                scroll_filter=self._translate_where(filters),
                limit=SCROLL_SIZE,
                offset=next_offset,
                with_payload=True,
            )
            stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId)
                and next_offset.num == 0
                and next_offset.uuid == ""
            )
            points.extend(results)

        return [cast(TDocument, point.payload) for point in points]

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        results, _next_offset = self._client.scroll(
            collection_name=self._name,
            scroll_filter=self._translate_where(filters),
            with_payload=True,
            with_vectors=False,
            limit=1,
        )

        if not results:
            return None

        return cast(TDocument, results[0].payload)

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        embeddings = list((await self._embedder.embed([document["content"]])).vectors)[0]

        self._client.upsert(
            collection_name=self._name,
            points=[
                models.PointStruct(
                    id=format_point_id(document["id"]), vector=embeddings, payload=document
                )
            ],
        )

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        existing_doc = await self.find_one(filters)

        if existing_doc:
            updated_document = {**existing_doc, **params}

            if "content" in params:
                embeddings = list((await self._embedder.embed([params["content"]])).vectors)[0]
            else:
                embeddings = list((await self._embedder.embed([existing_doc["content"]])).vectors)[
                    0
                ]

            self._client.upsert(
                collection_name=self._name,
                points=[
                    models.PointStruct(
                        id=format_point_id(updated_document["id"]),
                        vector=embeddings,
                        payload=updated_document,
                    )
                ],
            )

            return UpdateResult(
                acknowledged=True,
                matched_count=1,
                modified_count=1,
                updated_document=cast(TDocument, updated_document),
            )

        elif upsert:
            ensure_is_total(params, self._schema)

            embeddings = list((await self._embedder.embed([params["content"]])).vectors)[0]

            self._client.upsert(
                collection_name=self._name,
                points=[
                    models.PointStruct(
                        id=format_point_id(params["id"]), vector=embeddings, payload=params
                    )
                ],
            )

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
        doc_to_delete = await self.find_one(filters)

        if doc_to_delete:
            self._client.delete(collection_name=self._name, points_selector=[doc_to_delete["id"]])

            return DeleteResult(
                deleted_count=1,
                acknowledged=True,
                deleted_document=doc_to_delete,
            )

        return DeleteResult(
            acknowledged=True,
            deleted_count=0,
            deleted_document=None,
        )

    @override
    async def find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
    ) -> Sequence[SimilarDocumentResult[TDocument]]:
        query_embeddings = list((await self._embedder.embed([query])).vectors)[0]

        qdrant_filter = self._translate_where(filters)

        search_results = self._client.query_points(
            collection_name=self._name,
            query=query_embeddings,
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True,
        ).points

        if not search_results:
            return []

        self._logger.debug(
            f"Similar documents found\n{json.dumps([sr.payload for sr in search_results], indent=2)}"
        )

        return [
            SimilarDocumentResult(
                document=cast(TDocument, result.payload),
                distance=1.0 - result.score,
            )
            for result in search_results
            if result.id != 0
        ]


class QdrantDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        client: QdrantClient,
        embedder_factory: EmbedderFactory,
    ) -> None:
        self._logger = logger
        self._client = client
        self._embedder_factory = embedder_factory
        self._collections: dict[str, QdrantCollection[BaseDocument]] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        self._client.close()

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> QdrantCollection[TDocument]:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedder = self._embedder_factory.create_embedder(embedder_type)

        self._collections[name] = QdrantCollection(
            logger=self._logger,
            client=self._client,
            name=name,
            schema=schema,
            embedder=embedder,
            version=1,
        )

        return cast(QdrantCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> QdrantCollection[TDocument]:
        if not self._collections.get(name) and not self._client.collection_exists(name):
            raise ValueError(f'Qdrant collection "{name}" not found.')

        # The check above ensures we're only getting the collection
        # and not creating it.
        return await self.get_or_create_collection(
            name=name, schema=schema, embedder_type=embedder_type, document_loader=document_loader
        )

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> QdrantCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(QdrantCollection[TDocument], collection)

        return await self.create_collection(name=name, schema=schema, embedder_type=embedder_type)

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        self._client.delete_collection(collection_name=name)
        del self._collections[name]

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        if not self._client.collection_exists(collection_name=METADATA_COLLECTION_NAME):
            self._client.create_collection(
                collection_name=METADATA_COLLECTION_NAME,
                vectors_config={},
            )

        points = self._client.retrieve(
            collection_name=METADATA_COLLECTION_NAME, ids=[METADATA_POINT_ID], with_payload=True
        )

        document = {}
        if points:
            document = points[0].payload
        document[key] = value

        self._client.upsert(
            collection_name=METADATA_COLLECTION_NAME,
            points=[models.PointStruct(id=METADATA_POINT_ID, vector={}, payload=document)],
        )

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        if not self._client.collection_exists(METADATA_COLLECTION_NAME):
            raise ValueError("Metadata collection not found.")

        points = self._client.retrieve(
            collection_name=METADATA_COLLECTION_NAME, ids=[METADATA_POINT_ID], with_payload=True
        )
        if not points:
            raise ValueError(f'Metadata with key "{key}" not found.')

        document = points[0].payload
        if key not in document:
            raise ValueError(f'Metadata with key "{key}" not found.')

        document.pop(key)
        self._client.upsert(
            collection_name=METADATA_COLLECTION_NAME,
            points=[models.PointStruct(id=METADATA_POINT_ID, vector={}, payload=document)],
        )

    @override
    async def read_metadata(
        self,
    ) -> dict[str, JSONSerializable]:
        if not self._client.collection_exists(METADATA_COLLECTION_NAME):
            return {}

        points = self._client.retrieve(
            collection_name=METADATA_COLLECTION_NAME, ids=[METADATA_POINT_ID], with_payload=True
        )
        if not points:
            return {}
        return points[0].payload


def is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(val)
        return True
    except Exception:
        return False


def format_point_id(_id) -> str:
    """
    Converts any string into a UUID string based on a seed.

    Qdrant accepts UUID strings and unsigned integers as point ID.
    We use a seed to convert each string into a UUID string deterministically.
    This allows us to overwrite the same point with the original ID.
    """
    # If already a valid UUID, return as-is.
    if is_valid_uuid(str(_id)):
        return str(_id)

    return uuid.uuid5(uuid.NAMESPACE_DNS, str(_id)).hex
