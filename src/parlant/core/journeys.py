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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Awaitable, Callable, NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self, Required

from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.common import md5_checksum
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id, to_json_dict
from parlant.core.guidelines import GuidelineId
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.common import (
    ObjectId,
    Where,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import (
    DocumentStoreMigrationHelper,
)
from parlant.core.persistence.vector_database import VectorCollection, VectorDatabase
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentStoreMigrationHelper,
    query_chunks,
)
from parlant.core.tags import TagId

JourneyId = NewType("JourneyId", str)


@dataclass(frozen=True)
class Journey:
    id: JourneyId
    creation_utc: datetime
    conditions: Sequence[GuidelineId]
    title: str
    description: str
    tags: Sequence[TagId]

    def __hash__(self) -> int:
        return hash(self.id)


class JourneyUpdateParams(TypedDict, total=False):
    title: str
    description: str


class JourneyStore(ABC):
    @abstractmethod
    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: Sequence[GuidelineId],
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Journey: ...

    @abstractmethod
    async def list_journeys(
        self,
        tags: Optional[Sequence[TagId]] = None,
        condition: Optional[GuidelineId] = None,
    ) -> Sequence[Journey]: ...

    @abstractmethod
    async def read_journey(
        self,
        journey_id: JourneyId,
    ) -> Journey: ...

    @abstractmethod
    async def update_journey(
        self,
        journey_id: JourneyId,
        params: JourneyUpdateParams,
    ) -> Journey: ...

    @abstractmethod
    async def delete_journey(
        self,
        journey_id: JourneyId,
    ) -> None: ...

    @abstractmethod
    async def add_condition(
        self,
        journey_id: JourneyId,
        condition: GuidelineId,
    ) -> bool: ...

    @abstractmethod
    async def remove_condition(
        self,
        journey_id: JourneyId,
        condition: GuidelineId,
    ) -> bool: ...

    @abstractmethod
    async def upsert_tag(
        self,
        journey_id: JourneyId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        journey_id: JourneyId,
        tag_id: TagId,
    ) -> None: ...

    @abstractmethod
    async def find_relevant_journeys(
        self,
        query: str,
        available_journeys: Sequence[Journey],
        max_journeys: int = 5,
    ) -> Sequence[Journey]: ...


class JourneyDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    title: str
    description: str


class _JourneyDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    content: str
    checksum: Required[str]
    title: str
    description: str


class JourneyConditionAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    journey_id: JourneyId
    condition: GuidelineId


class JourneyTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    journey_id: JourneyId
    tag_id: TagId


class JourneyVectorStore(JourneyStore):
    VERSION = Version.from_string("0.2.0")

    def __init__(
        self,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
        allow_migration: bool = True,
    ):
        self._vector_db = vector_db
        self._document_db = document_db
        self._allow_migration = allow_migration
        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._embedder: Embedder
        self._lock = ReaderWriterLock()
        self._journeys_collection: VectorCollection[_JourneyDocument]
        self._tag_association_collection: DocumentCollection[JourneyTagAssociationDocument]
        self._condition_association_collection: DocumentCollection[
            JourneyConditionAssociationDocument
        ]

    async def _document_loader(self, doc: BaseDocument) -> Optional[_JourneyDocument]:
        if doc["version"] == "0.1.0":
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )
        if doc["version"] == "0.2.0":
            return cast(_JourneyDocument, doc)
        return None

    async def _tag_association_loader(
        self, doc: BaseDocument
    ) -> Optional[JourneyTagAssociationDocument]:
        if doc["version"] == "0.1.0":
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )
        if doc["version"] == "0.2.0":
            return cast(JourneyTagAssociationDocument, doc)
        return None

    async def _condition_association_loader(
        self, doc: BaseDocument
    ) -> Optional[JourneyConditionAssociationDocument]:
        if doc["version"] == "0.1.0":
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )
        if doc["version"] == "0.2.0":
            return cast(JourneyConditionAssociationDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()
        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._journeys_collection = await self._vector_db.get_or_create_collection(
                name="journeys",
                schema=_JourneyDocument,
                embedder_type=embedder_type,
                document_loader=self._document_loader,
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._document_db,
            allow_migration=self._allow_migration,
        ):
            self._tag_association_collection = await self._document_db.get_or_create_collection(
                name="journey_tags",
                schema=JourneyTagAssociationDocument,
                document_loader=self._tag_association_loader,
            )

            self._condition_association_collection = (
                await self._document_db.get_or_create_collection(
                    name="journey_conditions",
                    schema=JourneyConditionAssociationDocument,
                    document_loader=self._condition_association_loader,
                )
            )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    def _serialize_journey(
        self,
        journey: Journey,
        content: str,
    ) -> _JourneyDocument:
        return _JourneyDocument(
            id=ObjectId(journey.id),
            version=self.VERSION.to_string(),
            content=content,
            checksum=md5_checksum(content),
            creation_utc=journey.creation_utc.isoformat(),
            title=journey.title,
            description=journey.description,
        )

    @staticmethod
    def assemble_content(
        title: str,
        description: str,
        conditions: Sequence[GuidelineId],
    ) -> str:
        return f"{title}\n{description}\nConditions: {', '.join(conditions)}"

    async def _deserialize_journey(self, doc: _JourneyDocument) -> Journey:
        tags = [
            d["tag_id"]
            for d in await self._tag_association_collection.find({"journey_id": {"$eq": doc["id"]}})
        ]

        conditions = [
            d["condition"]
            for d in await self._condition_association_collection.find(
                {"journey_id": {"$eq": doc["id"]}}
            )
        ]

        return Journey(
            id=JourneyId(doc["id"]),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
            conditions=conditions,
            title=doc["title"],
            description=doc["description"],
            tags=tags,
        )

    @override
    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: Sequence[GuidelineId],
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Journey:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            journey = Journey(
                id=JourneyId(generate_id()),
                creation_utc=creation_utc,
                conditions=conditions,
                title=title,
                description=description,
                tags=tags or [],
            )

            content = self.assemble_content(
                title=title, description=description, conditions=conditions
            )

            await self._journeys_collection.insert_one(
                document=self._serialize_journey(journey, content)
            )

            for tag in tags or []:
                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(generate_id()),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "journey_id": journey.id,
                        "tag_id": tag,
                    }
                )

            for condition in conditions:
                await self._condition_association_collection.insert_one(
                    document={
                        "id": ObjectId(generate_id()),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "journey_id": journey.id,
                        "condition": condition,
                    }
                )

        return journey

    @override
    async def read_journey(self, journey_id: JourneyId) -> Journey:
        async with self._lock.reader_lock:
            doc = await self._journeys_collection.find_one({"id": {"$eq": journey_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(journey_id))

        return await self._deserialize_journey(doc)

    @override
    async def update_journey(
        self,
        journey_id: JourneyId,
        params: JourneyUpdateParams,
    ) -> Journey:
        async with self._lock.writer_lock:
            doc = await self._journeys_collection.find_one({"id": {"$eq": journey_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(journey_id))

            updated = {**doc, **params}

            conditions = await self._condition_association_collection.find(
                filters={"journey_id": {"$eq": journey_id}}
            )

            content = self.assemble_content(
                title=cast(str, updated["title"]),
                description=cast(str, updated["description"]),
                conditions=[c["condition"] for c in conditions],
            )

            updated["content"] = content
            updated["checksum"] = md5_checksum(content)

            result = await self._journeys_collection.update_one(
                filters={"id": {"$eq": journey_id}},
                params=cast(_JourneyDocument, to_json_dict(updated)),
            )

        assert result.updated_document

        return await self._deserialize_journey(result.updated_document)

    @override
    async def list_journeys(
        self,
        tags: Optional[Sequence[TagId]] = None,
        condition: Optional[GuidelineId] = None,
    ) -> Sequence[Journey]:
        filters: Where = {}
        tag_journey_ids: set[JourneyId] = set()
        condition_journey_ids: set[JourneyId] = set()

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    journey_ids = {
                        doc["journey_id"]
                        for doc in await self._tag_association_collection.find(filters={})
                    }
                    filters = (
                        {"$and": [{"id": {"$ne": id}} for id in journey_ids]} if journey_ids else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )
                    tag_journey_ids = {assoc["journey_id"] for assoc in tag_associations}

                    if not tag_journey_ids:
                        return []

            if condition is not None:
                condition_journey_ids = {
                    c_doc["journey_id"]
                    for c_doc in await self._condition_association_collection.find(
                        filters={"condition": {"$eq": condition}}
                    )
                }

            if tag_journey_ids and condition_journey_ids:
                filters = {
                    "$or": [
                        {"id": {"$eq": id}}
                        for id in tag_journey_ids.intersection(condition_journey_ids)
                    ]
                }
            elif tag_journey_ids:
                filters = {"$or": [{"id": {"$eq": id}} for id in tag_journey_ids]}
            elif condition_journey_ids:
                filters = {"$or": [{"id": {"$eq": id}} for id in condition_journey_ids]}

            return [
                await self._deserialize_journey(d)
                for d in await self._journeys_collection.find(filters=filters)
            ]

    @override
    async def delete_journey(
        self,
        journey_id: JourneyId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._journeys_collection.delete_one({"id": {"$eq": journey_id}})

            for c_doc in await self._condition_association_collection.find(
                filters={
                    "journey_id": {"$eq": journey_id},
                }
            ):
                await self._condition_association_collection.delete_one(
                    filters={"id": {"$eq": c_doc["id"]}}
                )

            for t_doc in await self._tag_association_collection.find(
                filters={
                    "journey_id": {"$eq": journey_id},
                }
            ):
                await self._tag_association_collection.delete_one(
                    filters={"id": {"$eq": t_doc["id"]}}
                )

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(journey_id))

    @override
    async def add_condition(
        self,
        journey_id: JourneyId,
        condition: GuidelineId,
    ) -> bool:
        async with self._lock.writer_lock:
            journey = await self.read_journey(journey_id)

            if condition in journey.conditions:
                return False

            await self._condition_association_collection.insert_one(
                document={
                    "id": ObjectId(generate_id()),
                    "version": self.VERSION.to_string(),
                    "creation_utc": datetime.now(timezone.utc).isoformat(),
                    "journey_id": journey_id,
                    "condition": condition,
                }
            )

            return True

    @override
    async def remove_condition(
        self,
        journey_id: JourneyId,
        condition: GuidelineId,
    ) -> bool:
        async with self._lock.writer_lock:
            await self._condition_association_collection.delete_one(
                filters={
                    "journey_id": {"$eq": journey_id},
                    "condition": {"$eq": condition},
                }
            )

            return True

    @override
    async def upsert_tag(
        self,
        journey_id: JourneyId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            journey = await self.read_journey(journey_id)

            if tag_id in journey.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_document: JourneyTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "journey_id": journey_id,
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            journey_document = await self._journeys_collection.find_one({"id": {"$eq": journey_id}})

        if not journey_document:
            raise ItemNotFoundError(item_id=UniqueId(journey_id))

        return True

    @override
    async def remove_tag(
        self,
        journey_id: JourneyId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "journey_id": {"$eq": journey_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            journey_document = await self._journeys_collection.find_one({"id": {"$eq": journey_id}})

        if not journey_document:
            raise ItemNotFoundError(item_id=UniqueId(journey_id))

    @override
    async def find_relevant_journeys(
        self,
        query: str,
        available_journeys: Sequence[Journey],
        max_journeys: int = 5,
    ) -> Sequence[Journey]:
        if not available_journeys:
            return []

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)
            filters: Where = {"id": {"$in": [str(j.id) for j in available_journeys]}}

            tasks = [
                self._journeys_collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=max_journeys,
                )
                for q in queries
            ]

        all_results = chain.from_iterable(await safe_gather(*tasks))
        unique_results = list(set(all_results))
        top_results = sorted(unique_results, key=lambda r: r.distance)[:max_journeys]

        return [await self._deserialize_journey(r.document) for r in top_results]
