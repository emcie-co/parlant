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

from typing import NewType, Optional, Sequence
from typing_extensions import override, TypedDict, Self
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import DocumentDatabase, DocumentCollection

GuidelineId = NewType("GuidelineId", str)


@dataclass(frozen=True)
class GuidelineContent:
    condition: str
    action: str


@dataclass(frozen=True)
class Guideline:
    id: GuidelineId
    creation_utc: datetime
    content: GuidelineContent

    def __str__(self) -> str:
        return f"When {self.content.condition}, then {self.content.action}"


class GuidelineUpdateParams(TypedDict, total=False):
    guideline_set: str
    condition: str
    action: str


class GuidelineStore(ABC):
    @abstractmethod
    async def create_guideline(
        self,
        guideline_set: str,
        condition: str,
        action: str,
        creation_utc: Optional[datetime] = None,
    ) -> Guideline: ...

    @abstractmethod
    async def list_guidelines(
        self,
        guideline_set: str,
    ) -> Sequence[Guideline]: ...

    @abstractmethod
    async def read_guideline(
        self,
        guideline_set: str,
        guideline_id: GuidelineId,
    ) -> Guideline: ...

    @abstractmethod
    async def delete_guideline(
        self,
        guideline_set: str,
        guideline_id: GuidelineId,
    ) -> None: ...

    @abstractmethod
    async def update_guideline(
        self,
        guideline_id: GuidelineId,
        params: GuidelineUpdateParams,
    ) -> Guideline: ...

    @abstractmethod
    async def find_guideline(
        self,
        guideline_set: str,
        guideline_content: GuidelineContent,
    ) -> Guideline: ...


class _GuidelineDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    guideline_set: str
    condition: str
    action: str


class GuidelineDocumentStore(GuidelineStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase):
        self._database = database
        self._collection: DocumentCollection[_GuidelineDocument]

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._collection = await self._database.get_or_create_collection(
            name="guidelines",
            schema=_GuidelineDocument,
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize(
        self,
        guideline: Guideline,
        guideline_set: str,
    ) -> _GuidelineDocument:
        return _GuidelineDocument(
            id=ObjectId(guideline.id),
            version=self.VERSION.to_string(),
            creation_utc=guideline.creation_utc.isoformat(),
            guideline_set=guideline_set,
            condition=guideline.content.condition,
            action=guideline.content.action,
        )

    def _deserialize(
        self,
        guideline_document: _GuidelineDocument,
    ) -> Guideline:
        return Guideline(
            id=GuidelineId(guideline_document["id"]),
            creation_utc=datetime.fromisoformat(guideline_document["creation_utc"]),
            content=GuidelineContent(
                condition=guideline_document["condition"], action=guideline_document["action"]
            ),
        )

    @override
    async def create_guideline(
        self,
        guideline_set: str,
        condition: str,
        action: str,
        creation_utc: Optional[datetime] = None,
    ) -> Guideline:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            guideline = Guideline(
                id=GuidelineId(generate_id()),
                creation_utc=creation_utc,
                content=GuidelineContent(
                    condition=condition,
                    action=action,
                ),
            )

            await self._collection.insert_one(
                document=self._serialize(
                    guideline=guideline,
                    guideline_set=guideline_set,
                )
            )

        return guideline

    @override
    async def list_guidelines(
        self,
        guideline_set: str,
    ) -> Sequence[Guideline]:
        async with self._lock.reader_lock:
            return [
                self._deserialize(d)
                for d in await self._collection.find(
                    filters={"guideline_set": {"$eq": guideline_set}}
                )
            ]

    @override
    async def read_guideline(
        self,
        guideline_set: str,
        guideline_id: GuidelineId,
    ) -> Guideline:
        async with self._lock.reader_lock:
            guideline_document = await self._collection.find_one(
                filters={
                    "guideline_set": {"$eq": guideline_set},
                    "id": {"$eq": guideline_id},
                }
            )

        if not guideline_document:
            raise ItemNotFoundError(
                item_id=UniqueId(guideline_id), message=f"with guideline_set '{guideline_set}'"
            )

        return self._deserialize(guideline_document)

    @override
    async def delete_guideline(
        self,
        guideline_set: str,
        guideline_id: GuidelineId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one(
                filters={
                    "guideline_set": {"$eq": guideline_set},
                    "id": {"$eq": guideline_id},
                }
            )

        if not result.deleted_document:
            raise ItemNotFoundError(
                item_id=UniqueId(guideline_id), message=f"with guideline_set '{guideline_set}'"
            )

    @override
    async def update_guideline(
        self,
        guideline_id: GuidelineId,
        params: GuidelineUpdateParams,
    ) -> Guideline:
        async with self._lock.writer_lock:
            guideline_document = _GuidelineDocument(
                {
                    **(
                        {"guideline_set": params["guideline_set"]}
                        if "guideline_set" in params
                        else {}
                    ),
                    **({"condition": params["condition"]} if "condition" in params else {}),
                    **({"action": params["action"]} if "action" in params else {}),
                }
            )

            result = await self._collection.update_one(
                filters={"id": {"$eq": guideline_id}},
                params=guideline_document,
            )

        assert result.updated_document

        return self._deserialize(result.updated_document)

    @override
    async def find_guideline(
        self,
        guideline_set: str,
        guideline_content: GuidelineContent,
    ) -> Guideline:
        async with self._lock.reader_lock:
            guideline_document = await self._collection.find_one(
                filters={
                    "guideline_set": {"$eq": guideline_set},
                    "condition": {"$eq": guideline_content.condition},
                    "action": {"$eq": guideline_content.action},
                }
            )

        if not guideline_document:
            raise ItemNotFoundError(
                item_id=UniqueId(f"{guideline_content.condition}{guideline_content.action}"),
                message=f"with guideline_set '{guideline_set}'",
            )

        return self._deserialize(guideline_document)
