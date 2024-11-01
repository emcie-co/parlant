from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence, TypedDict

from emcie.server.core.common import ItemNotFoundError, Version, generate_id, UniqueId
from emcie.server.core.guidelines import GuidelineId
from emcie.server.core.persistence.document_database import (
    DocumentDatabase,
    ObjectId,
)
from emcie.server.core.tools import ToolId

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
    ) -> GuidelineToolAssociation: ...

    @abstractmethod
    async def list_associations(self) -> Sequence[GuidelineToolAssociation]: ...


class _GuidelineToolAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    guideline_id: GuidelineId
    tool_id: str


class GuidelineToolAssociationDocumentStore(GuidelineToolAssociationStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase):
        self._collection = database.get_or_create_collection(
            name="associations", schema=_GuidelineToolAssociationDocument
        )

    def _serialize(
        self,
        association: GuidelineToolAssociation,
    ) -> _GuidelineToolAssociationDocument:
        return _GuidelineToolAssociationDocument(
            id=ObjectId(association.id),
            version=self.VERSION.to_string(),
            creation_utc=association.creation_utc.isoformat(),
            guideline_id=association.guideline_id,
            tool_id=association.tool_id.to_string(),
        )

    def _deserialize(
        self,
        association_document: _GuidelineToolAssociationDocument,
    ) -> GuidelineToolAssociation:
        return GuidelineToolAssociation(
            id=GuidelineToolAssociationId(association_document["id"]),
            creation_utc=datetime.fromisoformat(association_document["creation_utc"]),
            guideline_id=association_document["guideline_id"],
            tool_id=ToolId.from_string(association_document["tool_id"]),
        )

    async def create_association(
        self,
        guideline_id: GuidelineId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> GuidelineToolAssociation:
        creation_utc = creation_utc or datetime.now(timezone.utc)

        association = GuidelineToolAssociation(
            id=GuidelineToolAssociationId(generate_id()),
            creation_utc=creation_utc,
            guideline_id=guideline_id,
            tool_id=tool_id,
        )

        await self._collection.insert_one(document=self._serialize(association))

        return association

    async def read_association(
        self,
        association_id: GuidelineToolAssociationId,
    ) -> GuidelineToolAssociation:
        guideline_tool_association_document = await self._collection.find_one(
            filters={"id": {"$eq": association_id}}
        )

        if not guideline_tool_association_document:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

        return self._deserialize(guideline_tool_association_document)

    async def delete_association(
        self, association_id: GuidelineToolAssociationId
    ) -> GuidelineToolAssociation:
        result = await self._collection.delete_one(filters={"id": {"$eq": association_id}})

        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

        return self._deserialize(result.deleted_document)

    async def list_associations(self) -> Sequence[GuidelineToolAssociation]:
        return [self._deserialize(d) for d in await self._collection.find(filters={})]
