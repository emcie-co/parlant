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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NewType, Optional, Sequence, TypedDict
from typing_extensions import Self, override

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import (
    SCHEMA_VERSION_UNKNOWN,
    ItemNotFoundError,
    SchemaVersion,
    UniqueId,
    Version,
    generate_id,
    GuidelineId,
)
from parlant.core.documents.guideline_tools_associations import _GuidelineToolAssociationDocument_v1
from parlant.core.persistence.common import VersionedStore, ObjectId
from parlant.core.persistence.document_database import DocumentCollection, DocumentDatabase
from parlant.core.tools import ToolId

GuidelineToolAssociationId = NewType("GuidelineToolAssociationId", str)


@dataclass(frozen=True)
class GuidelineToolAssociation:
    id: GuidelineToolAssociationId
    creation_utc: datetime
    guideline_id: GuidelineId
    tool_id: ToolId

    def __hash__(self) -> int:
        return hash(self.id)


class GuidelineToolAssociationStore(ABC):
    @abstractmethod
    async def create_association(
        self,
        guideline_id: GuidelineId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> GuidelineToolAssociation: ...

    @abstractmethod
    async def read_association(
        self,
        association_id: GuidelineToolAssociationId,
    ) -> GuidelineToolAssociation: ...

    @abstractmethod
    async def delete_association(
        self,
        association_id: GuidelineToolAssociationId,
    ) -> None: ...

    @abstractmethod
    async def list_associations(self) -> Sequence[GuidelineToolAssociation]: ...


class _GuidelineToolAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    guideline_id: GuidelineId
    tool_id: str


class GuidelineToolAssociationDocumentStore(GuidelineToolAssociationStore, VersionedStore):
    VERSION = SchemaVersion(1)

    def __init__(self, database: DocumentDatabase):
        if database.version == SCHEMA_VERSION_UNKNOWN:
            database.version = self.VERSION
        self._database = database
        self._collection: DocumentCollection[_GuidelineToolAssociationDocument_v1]

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._collection = await self._database.get_or_create_collection(
            name="associations", schema=_GuidelineToolAssociationDocument_v1
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
        association: GuidelineToolAssociation,
    ) -> _GuidelineToolAssociationDocument_v1:
        return _GuidelineToolAssociationDocument_v1(
            id=ObjectId(association.id),
            creation_utc=association.creation_utc.isoformat(),
            guideline_id=association.guideline_id,
            tool_id=association.tool_id.to_string(),
        )

    def _deserialize(
        self,
        association_document: _GuidelineToolAssociationDocument_v1,
    ) -> GuidelineToolAssociation:
        return GuidelineToolAssociation(
            id=GuidelineToolAssociationId(association_document["id"]),
            creation_utc=datetime.fromisoformat(association_document["creation_utc"]),
            guideline_id=association_document["guideline_id"],
            tool_id=ToolId.from_string(association_document["tool_id"]),
        )

    @property
    @override
    def database_version(self) -> SchemaVersion:
        return self._database.version

    @property
    @override
    def database_name(self) -> str:
        if "file_path" in self._database.__dict__ and isinstance(
            db_path := getattr(self._database, "file_path"), Path
        ):
            return db_path.name
        return "unknown"

    @override
    async def create_association(
        self,
        guideline_id: GuidelineId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> GuidelineToolAssociation:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            association = GuidelineToolAssociation(
                id=GuidelineToolAssociationId(generate_id()),
                creation_utc=creation_utc,
                guideline_id=guideline_id,
                tool_id=tool_id,
            )

            await self._collection.insert_one(document=self._serialize(association))

        return association

    @override
    async def read_association(
        self,
        association_id: GuidelineToolAssociationId,
    ) -> GuidelineToolAssociation:
        async with self._lock.reader_lock:
            guideline_tool_association_document = await self._collection.find_one(
                filters={"id": {"$eq": association_id}}
            )

        if not guideline_tool_association_document:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

        return self._deserialize(guideline_tool_association_document)

    @override
    async def delete_association(self, association_id: GuidelineToolAssociationId) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one(filters={"id": {"$eq": association_id}})

        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

    @override
    async def list_associations(self) -> Sequence[GuidelineToolAssociation]:
        async with self._lock.reader_lock:
            return [self._deserialize(d) for d in await self._collection.find(filters={})]
