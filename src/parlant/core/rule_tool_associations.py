# Copyright 2026 Emcie Co Ltd.
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
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.common import ItemNotFoundError, try_or_none, Version, IdGenerator, UniqueId
from parlant.core.rules import RuleId
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.tools import ToolId

RuleToolAssociationId = NewType("RuleToolAssociationId", str)


@dataclass(frozen=True)
class RuleToolAssociation:
    id: RuleToolAssociationId
    creation_utc: datetime
    rule_id: RuleId
    tool_id: ToolId

    def __hash__(self) -> int:
        return hash(self.id)


class RuleToolAssociationStore(ABC):
    @abstractmethod
    async def create_association(
        self,
        rule_id: RuleId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> RuleToolAssociation: ...

    @abstractmethod
    async def read_association(
        self,
        association_id: RuleToolAssociationId,
    ) -> RuleToolAssociation: ...

    @abstractmethod
    async def delete_association(
        self,
        association_id: RuleToolAssociationId,
    ) -> None: ...

    @abstractmethod
    async def list_associations(self) -> Sequence[RuleToolAssociation]: ...


class _RuleToolAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    rule_id: RuleId
    tool_id: str


class RuleToolAssociationDocumentStore(RuleToolAssociationStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ) -> None:
        self._id_generator = id_generator

        self._database = database
        self._collection: DocumentCollection[_RuleToolAssociationDocument]

        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._lock = ReaderWriterLock()

    async def _document_loader(
        self,
        doc: BaseDocument,
    ) -> Optional[_RuleToolAssociationDocument]:
        if Version.from_string(doc["version"]) >= Version.from_string("0.1.0"):
            return cast(_RuleToolAssociationDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_associations"
                if self._collections_prefix
                else "associations",
                schema=_RuleToolAssociationDocument,
                document_loader=self._document_loader,
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
        association: RuleToolAssociation,
    ) -> _RuleToolAssociationDocument:
        return _RuleToolAssociationDocument(
            id=ObjectId(association.id),
            version=self.VERSION.to_string(),
            creation_utc=association.creation_utc.isoformat(),
            rule_id=association.rule_id,
            tool_id=association.tool_id.to_string(),
        )

    def _deserialize(
        self,
        association_document: _RuleToolAssociationDocument,
    ) -> RuleToolAssociation:
        return RuleToolAssociation(
            id=RuleToolAssociationId(association_document["id"]),
            creation_utc=datetime.fromisoformat(association_document["creation_utc"]),
            rule_id=association_document["rule_id"],
            tool_id=ToolId.from_string(association_document["tool_id"]),
        )

    @override
    async def create_association(
        self,
        rule_id: RuleId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> RuleToolAssociation:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = f"{rule_id}{tool_id}"

            association = RuleToolAssociation(
                id=RuleToolAssociationId(self._id_generator.generate(association_checksum)),
                creation_utc=creation_utc,
                rule_id=rule_id,
                tool_id=tool_id,
            )

            await self._collection.insert_one(document=self._serialize(association))

        return association

    @override
    async def read_association(
        self,
        association_id: RuleToolAssociationId,
    ) -> RuleToolAssociation:
        async with self._lock.reader_lock:
            rule_tool_association_document = await self._collection.find_one(
                filters={"id": {"$eq": association_id}}
            )

        if not rule_tool_association_document:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

        return self._deserialize(rule_tool_association_document)

    @override
    async def delete_association(self, association_id: RuleToolAssociationId) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one(filters={"id": {"$eq": association_id}})

        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

    @override
    async def list_associations(self) -> Sequence[RuleToolAssociation]:
        async with self._lock.reader_lock:
            return [self._deserialize(d) for d in await self._collection.find(filters={})]


class CompositeRuleToolAssociationStore(RuleToolAssociationStore):
    def __init__(
        self,
        writable_store: RuleToolAssociationStore,
        readable_stores: Sequence[RuleToolAssociationStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[RuleToolAssociationStore] = [
            writable_store,
            *readable_stores,
        ]

    @override
    async def create_association(
        self,
        rule_id: RuleId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> RuleToolAssociation:
        return await self._writable_store.create_association(rule_id, tool_id, creation_utc)

    @override
    async def read_association(
        self,
        association_id: RuleToolAssociationId,
    ) -> RuleToolAssociation:
        results = await safe_gather(
            *[try_or_none(store.read_association(association_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(association_id))
        return result

    @override
    async def delete_association(
        self,
        association_id: RuleToolAssociationId,
    ) -> None:
        return await self._writable_store.delete_association(association_id)

    @override
    async def list_associations(self) -> Sequence[RuleToolAssociation]:
        results = await safe_gather(*[store.list_associations() for store in self._all_stores])
        return list(chain.from_iterable(results))
