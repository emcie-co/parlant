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
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id, to_json_dict
from parlant.core.guidelines import GuidelineId
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


class _JourneyDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    title: str
    description: str


class _JourneyConditionAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    journey_id: JourneyId
    condition: GuidelineId


class _JourneyTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    journey_id: JourneyId
    tag_id: TagId


class JourneyDocumentStore(JourneyStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase, allow_migration: bool = False):
        self._database = database
        self._journeys_collection: DocumentCollection[_JourneyDocument]
        self._condition_association_collection: DocumentCollection[
            _JourneyConditionAssociationDocument
        ]
        self._tag_association_collection: DocumentCollection[_JourneyTagAssociationDocument]
        self._allow_migration = allow_migration

        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_JourneyDocument]:
        if doc["version"] == "0.1.0":
            return cast(_JourneyDocument, doc)
        return None

    async def _condition_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_JourneyConditionAssociationDocument]:
        if doc["version"] == "0.1.0":
            return cast(_JourneyConditionAssociationDocument, doc)
        return None

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_JourneyTagAssociationDocument]:
        if doc["version"] == "0.1.0":
            return cast(_JourneyTagAssociationDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._journeys_collection = await self._database.get_or_create_collection(
                name="journeys",
                schema=_JourneyDocument,
                document_loader=self._document_loader,
            )

            self._condition_association_collection = await self._database.get_or_create_collection(
                name="journey_conditions",
                schema=_JourneyConditionAssociationDocument,
                document_loader=self._condition_document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name="journey_tags",
                schema=_JourneyTagAssociationDocument,
                document_loader=self._association_document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    def _serialize_journey(self, journey: Journey) -> _JourneyDocument:
        return _JourneyDocument(
            id=ObjectId(journey.id),
            version=self.VERSION.to_string(),
            creation_utc=journey.creation_utc.isoformat(),
            title=journey.title,
            description=journey.description,
        )

    async def _deserialize_journey(self, journey_document: _JourneyDocument) -> Journey:
        conditions = [
            d["condition"]
            for d in await self._condition_association_collection.find(
                {"journey_id": {"$eq": journey_document["id"]}}
            )
        ]

        tags = [
            d["tag_id"]
            for d in await self._tag_association_collection.find(
                {"journey_id": {"$eq": journey_document["id"]}}
            )
        ]

        return Journey(
            id=JourneyId(journey_document["id"]),
            creation_utc=datetime.fromisoformat(journey_document["creation_utc"]),
            conditions=conditions,
            title=journey_document["title"],
            description=journey_document["description"],
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

            for condition in conditions:
                await self._condition_association_collection.insert_one(
                    document={
                        "id": ObjectId(generate_id()),
                        "version": self.VERSION.to_string(),
                        "creation_utc": datetime.now(timezone.utc).isoformat(),
                        "journey_id": journey.id,
                        "condition": condition,
                    }
                )

            await self._journeys_collection.insert_one(
                document=self._serialize_journey(journey=journey)
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

        return journey

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
    async def read_journey(self, journey_id: JourneyId) -> Journey:
        async with self._lock.reader_lock:
            journey_document = await self._journeys_collection.find_one(
                filters={
                    "id": {"$eq": journey_id},
                }
            )

        if not journey_document:
            raise ItemNotFoundError(item_id=UniqueId(journey_id))

        return await self._deserialize_journey(journey_document=journey_document)

    @override
    async def update_journey(
        self,
        journey_id: JourneyId,
        params: JourneyUpdateParams,
    ) -> Journey:
        async with self._lock.writer_lock:
            journey_document = await self._journeys_collection.find_one(
                filters={
                    "id": {"$eq": journey_id},
                }
            )

            if not journey_document:
                raise ItemNotFoundError(item_id=UniqueId(journey_id))

            result = await self._journeys_collection.update_one(
                filters={"id": {"$eq": journey_id}},
                params=cast(_JourneyDocument, to_json_dict(params)),
            )

        assert result.updated_document

        return await self._deserialize_journey(journey_document=result.updated_document)

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

            association_document: _JourneyTagAssociationDocument = {
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
