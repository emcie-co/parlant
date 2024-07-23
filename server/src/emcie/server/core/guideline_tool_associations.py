from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core import common
from emcie.server.core.guidelines import GuidelineId
from emcie.server.core.tools import ToolId
from emcie.server.core.persistence.document_database import DocumentDatabase

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
    async def list_associations(self) -> Sequence[GuidelineToolAssociation]: ...


class GuidelineToolAssociationDocumentStore(GuidelineToolAssociationStore):
    class GuidelineToolAssociationDocument(DefaultBaseModel):
        id: GuidelineToolAssociationId
        creation_utc: datetime
        guideline_id: GuidelineId
        tool_id: ToolId

    def __init__(self, database: DocumentDatabase):
        self._collection = database.get_or_create_collection(
            name="associations", schema=self.GuidelineToolAssociationDocument
        )

    async def create_association(
        self,
        guideline_id: GuidelineId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> GuidelineToolAssociation:
        creation_utc = creation_utc or datetime.now(timezone.utc)
        association_id = await self._collection.insert_one(
            document={
                "id": common.generate_id(),
                "creation_utc": creation_utc,
                "guideline_id": guideline_id,
                "tool_id": tool_id,
            },
        )

        return GuidelineToolAssociation(
            id=GuidelineToolAssociationId(association_id),
            creation_utc=creation_utc,
            guideline_id=guideline_id,
            tool_id=tool_id,
        )

    async def list_associations(self) -> Sequence[GuidelineToolAssociation]:
        return [
            GuidelineToolAssociation(
                id=GuidelineToolAssociationId(d["id"]),
                creation_utc=d["creation_utc"],
                guideline_id=d["guideline_id"],
                tool_id=d["tool_id"],
            )
            for d in await self._collection.find(filters={})
        ]
