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
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import (
    ItemNotFoundError,
    UniqueId,
    Version,
    IdGenerator,
    md5_checksum,
    to_json_dict,
)
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.tags import TagId

PlaybookId = NewType("PlaybookId", str)
DisabledRuleRef = NewType("DisabledRuleRef", str)


class PlaybookUpdateParams(TypedDict, total=False):
    name: str
    description: Optional[str]
    parent_id: Optional[PlaybookId]


@dataclass(frozen=True)
class Playbook:
    id: PlaybookId
    name: str
    description: Optional[str]
    parent_id: Optional[PlaybookId]
    disabled_rules: Sequence[DisabledRuleRef]
    tags: Sequence[TagId]
    creation_utc: datetime


class PlaybookStore(ABC):
    @abstractmethod
    async def create_playbook(
        self,
        name: str,
        description: Optional[str] = None,
        parent_id: Optional[PlaybookId] = None,
        tags: Optional[Sequence[TagId]] = None,
        id: Optional[PlaybookId] = None,
        creation_utc: Optional[datetime] = None,
    ) -> Playbook: ...

    @abstractmethod
    async def list_playbooks(self) -> Sequence[Playbook]: ...

    @abstractmethod
    async def read_playbook(
        self,
        playbook_id: PlaybookId,
    ) -> Playbook: ...

    @abstractmethod
    async def update_playbook(
        self,
        playbook_id: PlaybookId,
        params: PlaybookUpdateParams,
    ) -> Playbook: ...

    @abstractmethod
    async def delete_playbook(
        self,
        playbook_id: PlaybookId,
    ) -> None: ...

    @abstractmethod
    async def upsert_tag(
        self,
        playbook_id: PlaybookId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        playbook_id: PlaybookId,
        tag_id: TagId,
    ) -> None: ...

    @abstractmethod
    async def add_disabled_rule(
        self,
        playbook_id: PlaybookId,
        rule_ref: DisabledRuleRef,
    ) -> bool: ...

    @abstractmethod
    async def remove_disabled_rule(
        self,
        playbook_id: PlaybookId,
        rule_ref: DisabledRuleRef,
    ) -> None: ...


class _PlaybookDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    name: str
    description: Optional[str]
    parent_id: Optional[str]


class _PlaybookTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    playbook_id: PlaybookId
    tag_id: TagId


class _PlaybookDisabledRuleDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    playbook_id: PlaybookId
    rule_ref: DisabledRuleRef


class PlaybookDocumentStore(PlaybookStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ):
        self._id_generator = id_generator
        self._database = database
        self._playbooks_collection: DocumentCollection[_PlaybookDocument]
        self._tag_association_collection: DocumentCollection[_PlaybookTagAssociationDocument]
        self._disabled_rules_collection: DocumentCollection[_PlaybookDisabledRuleDocument]
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_PlaybookDocument]:
        doc = cast(_PlaybookDocument, doc)

        if doc["version"] == "0.1.0":
            return doc

        return None

    async def _tag_association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_PlaybookTagAssociationDocument]:
        doc = cast(_PlaybookTagAssociationDocument, doc)

        if doc["version"] == "0.1.0":
            return doc

        return None

    async def _disabled_rule_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_PlaybookDisabledRuleDocument]:
        doc = cast(_PlaybookDisabledRuleDocument, doc)

        if doc["version"] == "0.1.0":
            return doc

        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._playbooks_collection = await self._database.get_or_create_collection(
                name="playbooks",
                schema=_PlaybookDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name="playbook_tags",
                schema=_PlaybookTagAssociationDocument,
                document_loader=self._tag_association_document_loader,
            )

            self._disabled_rules_collection = await self._database.get_or_create_collection(
                name="playbook_disabled_rules",
                schema=_PlaybookDisabledRuleDocument,
                document_loader=self._disabled_rule_document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    def _serialize_playbook(self, playbook: Playbook) -> _PlaybookDocument:
        return _PlaybookDocument(
            id=ObjectId(playbook.id),
            version=self.VERSION.to_string(),
            creation_utc=playbook.creation_utc.isoformat(),
            name=playbook.name,
            description=playbook.description,
            parent_id=playbook.parent_id,
        )

    async def _deserialize_playbook(self, playbook_document: _PlaybookDocument) -> Playbook:
        tags = [
            d["tag_id"]
            for d in await self._tag_association_collection.find(
                {"playbook_id": {"$eq": playbook_document["id"]}}
            )
        ]

        disabled_rules = [
            d["rule_ref"]
            for d in await self._disabled_rules_collection.find(
                {"playbook_id": {"$eq": playbook_document["id"]}}
            )
        ]

        parent_id_str = playbook_document.get("parent_id")

        return Playbook(
            id=PlaybookId(playbook_document["id"]),
            creation_utc=datetime.fromisoformat(playbook_document["creation_utc"]),
            name=playbook_document["name"],
            description=playbook_document.get("description"),
            parent_id=PlaybookId(parent_id_str) if parent_id_str else None,
            tags=tags,
            disabled_rules=disabled_rules,
        )

    @override
    async def create_playbook(
        self,
        name: str,
        description: Optional[str] = None,
        parent_id: Optional[PlaybookId] = None,
        tags: Optional[Sequence[TagId]] = None,
        id: Optional[PlaybookId] = None,
        creation_utc: Optional[datetime] = None,
    ) -> Playbook:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            if id is not None:
                playbook_id = id

                existing = await self._playbooks_collection.find_one(
                    filters={"id": {"$eq": playbook_id}}
                )
                if existing:
                    raise ValueError(f"Playbook with id '{playbook_id}' already exists")
            else:
                playbook_checksum = md5_checksum(f"{name}{description}{parent_id}{tags}")
                playbook_id = PlaybookId(self._id_generator.generate(playbook_checksum))

            playbook = Playbook(
                id=playbook_id,
                name=name,
                description=description,
                parent_id=parent_id,
                disabled_rules=[],
                tags=tags or [],
                creation_utc=creation_utc,
            )

            await self._playbooks_collection.insert_one(
                document=self._serialize_playbook(playbook=playbook)
            )

            for tag_id in tags or []:
                tag_checksum = md5_checksum(f"{playbook.id}{tag_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "playbook_id": playbook.id,
                        "tag_id": tag_id,
                    }
                )

        return playbook

    @override
    async def list_playbooks(self) -> Sequence[Playbook]:
        async with self._lock.reader_lock:
            return [
                await self._deserialize_playbook(d)
                for d in await self._playbooks_collection.find(filters={})
            ]

    @override
    async def read_playbook(self, playbook_id: PlaybookId) -> Playbook:
        async with self._lock.reader_lock:
            playbook_document = await self._playbooks_collection.find_one(
                filters={"id": {"$eq": playbook_id}}
            )

        if not playbook_document:
            raise ItemNotFoundError(item_id=UniqueId(playbook_id))

        return await self._deserialize_playbook(playbook_document=playbook_document)

    @override
    async def update_playbook(
        self,
        playbook_id: PlaybookId,
        params: PlaybookUpdateParams,
    ) -> Playbook:
        async with self._lock.writer_lock:
            playbook_document = await self._playbooks_collection.find_one(
                filters={"id": {"$eq": playbook_id}}
            )

            if not playbook_document:
                raise ItemNotFoundError(item_id=UniqueId(playbook_id))

            result = await self._playbooks_collection.update_one(
                filters={"id": {"$eq": playbook_id}},
                params=cast(_PlaybookDocument, to_json_dict(params)),
            )

        assert result.updated_document

        return await self._deserialize_playbook(playbook_document=result.updated_document)

    @override
    async def delete_playbook(
        self,
        playbook_id: PlaybookId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._playbooks_collection.delete_one({"id": {"$eq": playbook_id}})

            for tag_doc in await self._tag_association_collection.find(
                filters={"playbook_id": {"$eq": playbook_id}}
            ):
                await self._tag_association_collection.delete_one(
                    filters={"id": {"$eq": tag_doc["id"]}}
                )

            for rule_doc in await self._disabled_rules_collection.find(
                filters={"playbook_id": {"$eq": playbook_id}}
            ):
                await self._disabled_rules_collection.delete_one(
                    filters={"id": {"$eq": rule_doc["id"]}}
                )

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(playbook_id))

    @override
    async def upsert_tag(
        self,
        playbook_id: PlaybookId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            playbook = await self.read_playbook(playbook_id)

            if tag_id in playbook.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = md5_checksum(f"{playbook_id}{tag_id}")

            association_document: _PlaybookTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "playbook_id": playbook_id,
                "tag_id": tag_id,
            }

            await self._tag_association_collection.insert_one(document=association_document)

            playbook_document = await self._playbooks_collection.find_one(
                {"id": {"$eq": playbook_id}}
            )

        if not playbook_document:
            raise ItemNotFoundError(item_id=UniqueId(playbook_id))

        return True

    @override
    async def remove_tag(
        self,
        playbook_id: PlaybookId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "playbook_id": {"$eq": playbook_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            playbook_document = await self._playbooks_collection.find_one(
                {"id": {"$eq": playbook_id}}
            )

        if not playbook_document:
            raise ItemNotFoundError(item_id=UniqueId(playbook_id))

    @override
    async def add_disabled_rule(
        self,
        playbook_id: PlaybookId,
        rule_ref: DisabledRuleRef,
    ) -> bool:
        async with self._lock.writer_lock:
            playbook = await self.read_playbook(playbook_id)

            if rule_ref in playbook.disabled_rules:
                return False

            creation_utc = datetime.now(timezone.utc)

            rule_checksum = md5_checksum(f"{playbook_id}{rule_ref}")

            rule_document: _PlaybookDisabledRuleDocument = {
                "id": ObjectId(self._id_generator.generate(rule_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "playbook_id": playbook_id,
                "rule_ref": rule_ref,
            }

            await self._disabled_rules_collection.insert_one(document=rule_document)

        return True

    @override
    async def remove_disabled_rule(
        self,
        playbook_id: PlaybookId,
        rule_ref: DisabledRuleRef,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._disabled_rules_collection.delete_one(
                {
                    "playbook_id": {"$eq": playbook_id},
                    "rule_ref": {"$eq": rule_ref},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(rule_ref))

            playbook_document = await self._playbooks_collection.find_one(
                {"id": {"$eq": playbook_id}}
            )

        if not playbook_document:
            raise ItemNotFoundError(item_id=UniqueId(playbook_id))
