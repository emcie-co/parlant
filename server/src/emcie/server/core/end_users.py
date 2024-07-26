from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core import common
from emcie.server.core.persistence.document_database import DocumentDatabase

EndUserId = NewType("EndUserId", str)


@dataclass(frozen=True)
class EndUser:
    id: EndUserId
    creation_utc: datetime
    name: str
    email: str


class EndUserStore(ABC):
    @abstractmethod
    async def create_end_user(
        self,
        name: str,
        email: str,
        creation_utc: Optional[datetime] = None,
    ) -> EndUser: ...

    @abstractmethod
    async def read_end_user(
        self,
        end_user_id: EndUserId,
    ) -> EndUser: ...


class EndUserDocumentStore(EndUserStore):
    class EndUserDocument(DefaultBaseModel):
        id: EndUserId
        creation_utc: datetime
        name: str
        email: str

    def __init__(
        self,
        database: DocumentDatabase,
    ) -> None:
        self._collection = database.get_or_create_collection(
            name="end_users",
            schema=self.EndUserDocument,
        )

    async def create_end_user(
        self,
        name: str,
        email: str,
        creation_utc: Optional[datetime] = None,
    ) -> EndUser:
        creation_utc = creation_utc or datetime.now(timezone.utc)
        end_user_id = await self._collection.insert_one(
            document={
                "id": common.generate_id(),
                "name": name,
                "email": email,
                "creation_utc": creation_utc,
            },
        )

        return EndUser(
            id=EndUserId(end_user_id),
            name=name,
            email=email,
            creation_utc=creation_utc,
        )

    async def read_end_user(
        self,
        end_user_id: EndUserId,
    ) -> EndUser:
        end_user_document = await self._collection.find_one(filters={"id": {"$eq": end_user_id}})

        return EndUser(
            id=EndUserId(end_user_document["id"]),
            name=end_user_document["name"],
            email=end_user_document["email"],
            creation_utc=end_user_document["creation_utc"],
        )
