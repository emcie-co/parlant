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
from enum import Enum
from functools import total_ordering
from itertools import chain
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.common import (
    ItemNotFoundError,
    try_or_none,
    UniqueId,
    Version,
    IdGenerator,
    xxh3_checksum,
    to_json_dict,
)
from parlant.core.persistence.common import (
    ObjectId,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
    group_ids_by_entity,
)
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.groups import GroupId

AgentId = NewType("AgentId", str)


class MessageOutputMode(Enum):
    """Defines how the agent outputs messages."""

    BLOCK = "block"
    """Full message is sent at once (default behavior)."""

    STREAM = "stream"
    """Message is streamed token by token."""

    VOICE = "voice"
    """Message is streamed as speakable segments to the in-process session
    tap for real-time (voice) delivery, and persisted coarsely."""


@total_ordering
class Effort(Enum):
    """Defines how much effort the agent invests in processing."""

    MIN = "min"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"

    @property
    def level(self) -> int:
        return _EFFORT_ORDER[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Effort):
            return NotImplemented

        return self.level < other.level


_EFFORT_ORDER: dict[Effort, int] = {
    Effort.MIN: 0,
    Effort.LOW: 1,
    Effort.MEDIUM: 2,
    Effort.HIGH: 3,
    Effort.MAX: 4,
}


class Moderation(Enum):
    """How the agent applies the optional moderation checkpoints
    (the "moderation.customer" and "moderation.agent" model routes)."""

    AUTO = "auto"
    """Each checkpoint runs iff its route is configured; unconfigured
    checkpoints are skipped silently."""

    ON = "on"
    """Moderation is required: both routes must be configured (routing a
    checkpoint to the "null" provider counts as deliberately waiving it).
    Validated at startup and on agent creation/update."""

    OFF = "off"
    """No moderation, regardless of configured routes."""


class AgentUpdateParams(TypedDict, total=False):
    name: str
    instructions: Optional[str]
    max_steps: int
    message_output_mode: MessageOutputMode
    engine: str
    effort: Effort
    moderation: Moderation
    message_review: bool
    tool_review: bool


@dataclass(frozen=True)
class Agent:
    id: AgentId
    name: str
    instructions: Optional[str]
    creation_utc: datetime
    modified_utc: datetime
    max_steps: int
    groups: Sequence[GroupId]
    engine: str
    message_output_mode: MessageOutputMode = MessageOutputMode.BLOCK
    effort: Effort = Effort.MEDIUM
    moderation: Moderation = Moderation.AUTO
    # Review defaults for all of this agent's turns; matched rules can
    # additionally request either review but never veto these.
    message_review: bool = False
    tool_review: bool = False


class AgentStore(ABC):
    @abstractmethod
    async def create_agent(
        self,
        name: str,
        instructions: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_steps: Optional[int] = None,
        message_output_mode: Optional[MessageOutputMode] = None,
        engine: Optional[str] = None,
        effort: Optional[Effort] = None,
        moderation: Optional[Moderation] = None,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[AgentId] = None,
        message_review: bool = False,
        tool_review: bool = False,
    ) -> Agent: ...

    @abstractmethod
    async def list_agents(
        self,
    ) -> Sequence[Agent]: ...

    @abstractmethod
    async def read_agent(
        self,
        agent_id: AgentId,
    ) -> Agent: ...

    @abstractmethod
    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
    ) -> Agent: ...

    @abstractmethod
    async def delete_agent(
        self,
        agent_id: AgentId,
    ) -> None: ...

    @abstractmethod
    async def upsert_group(
        self,
        agent_id: AgentId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_group(
        self,
        agent_id: AgentId,
        group_id: GroupId,
    ) -> None: ...


class _AgentDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    name: str
    instructions: Optional[str]
    max_steps: int
    message_output_mode: str
    engine: str
    effort: str
    moderation: str
    message_review: bool
    tool_review: bool


class _AgentTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    agent_id: AgentId
    group_id: GroupId


class AgentDocumentStore(AgentStore):
    VERSION = Version.from_string("0.6.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ):
        self._id_generator = id_generator

        self._database = database

        self._agents_collection: DocumentCollection[_AgentDocument]
        self._tag_association_collection: DocumentCollection[_AgentTagAssociationDocument]
        self._collections_prefix = collections_prefix

        self._allow_migration = allow_migration

        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_AgentDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[_AgentDocument](self, {}).migrate(doc)

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_AgentTagAssociationDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[_AgentTagAssociationDocument](self, {}).migrate(doc)

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._agents_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_agents" if self._collections_prefix else "agents",
                schema=_AgentDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_agent_groups"
                if self._collections_prefix
                else "agent_groups",
                schema=_AgentTagAssociationDocument,
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

    def _serialize_agent(self, agent: Agent) -> _AgentDocument:
        return _AgentDocument(
            id=ObjectId(agent.id),
            version=self.VERSION.to_string(),
            creation_utc=agent.creation_utc.isoformat(),
            last_modified=agent.modified_utc.isoformat(),
            name=agent.name,
            instructions=agent.instructions,
            max_steps=agent.max_steps,
            message_output_mode=agent.message_output_mode.value,
            engine=agent.engine,
            effort=agent.effort.value,
            moderation=agent.moderation.value,
            message_review=agent.message_review,
            tool_review=agent.tool_review,
        )

    async def _deserialize_agent(self, agent_document: _AgentDocument) -> Agent:
        groups = [
            d["group_id"]
            for d in await self._tag_association_collection.find(
                {"agent_id": {"$eq": agent_document["id"]}}
            )
        ]

        return self._to_agent(agent_document, groups)

    async def _deserialize_many(
        self,
        documents: Sequence[_AgentDocument],
    ) -> Sequence[Agent]:
        """Reconstruct a whole page of entities with one association query."""
        group_ids = await group_ids_by_entity(
            self._tag_association_collection,
            entity_field="agent_id",
            value_field="group_id",
            entity_ids=[d["id"] for d in documents],
        )

        return [self._to_agent(d, group_ids[d["id"]]) for d in documents]

    def _to_agent(
        self,
        agent_document: _AgentDocument,
        group_ids: Sequence[str],
    ) -> Agent:
        return Agent(
            id=AgentId(agent_document["id"]),
            creation_utc=datetime.fromisoformat(agent_document["creation_utc"]),
            modified_utc=datetime.fromisoformat(agent_document["last_modified"]),
            name=agent_document["name"],
            instructions=agent_document["instructions"],
            max_steps=agent_document["max_steps"],
            groups=[GroupId(group_id) for group_id in group_ids],
            message_output_mode=MessageOutputMode(
                agent_document.get("message_output_mode", "block")
            ),
            engine=agent_document.get("engine", "compass"),
            effort=Effort(agent_document.get("effort", Effort.MEDIUM.value)),
            moderation=Moderation(agent_document.get("moderation", Moderation.AUTO.value)),
            message_review=agent_document.get("message_review", False),
            tool_review=agent_document.get("tool_review", False),
        )

    @override
    async def create_agent(
        self,
        name: str,
        instructions: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_steps: Optional[int] = None,
        message_output_mode: Optional[MessageOutputMode] = None,
        engine: Optional[str] = None,
        effort: Optional[Effort] = None,
        moderation: Optional[Moderation] = None,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[AgentId] = None,
        message_review: bool = False,
        tool_review: bool = False,
    ) -> Agent:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            max_steps = max_steps or 15

            # Use provided ID or generate one
            if id is not None:
                agent_id = id

                # Check if agent with this ID already exists
                existing = await self._agents_collection.find_one(filters={"id": {"$eq": agent_id}})
                if existing:
                    raise ValueError(f"Agent with id '{agent_id}' already exists")
            else:
                agent_checksum = xxh3_checksum(f"{name}{instructions}{max_steps}{groups}")
                agent_id = AgentId(self._id_generator.generate(agent_checksum))

            agent = Agent(
                id=agent_id,
                name=name,
                instructions=instructions,
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                max_steps=max_steps,
                groups=groups or [],
                message_output_mode=message_output_mode or MessageOutputMode.BLOCK,
                engine=engine or "compass",
                effort=effort or Effort.MEDIUM,
                moderation=moderation or Moderation.AUTO,
                message_review=message_review,
                tool_review=tool_review,
            )

            await self._agents_collection.insert_one(document=self._serialize_agent(agent=agent))

            for group_id in groups or []:
                tag_checksum = xxh3_checksum(f"{agent.id}{group_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "agent_id": agent.id,
                        "group_id": group_id,
                    }
                )

        return agent

    @override
    async def list_agents(
        self,
    ) -> Sequence[Agent]:
        async with self._lock.reader_lock:
            return await self._deserialize_many(
                list(await self._agents_collection.find(filters={}))
            )

    @override
    async def read_agent(self, agent_id: AgentId) -> Agent:
        async with self._lock.reader_lock:
            agent_document = await self._agents_collection.find_one(
                filters={
                    "id": {"$eq": agent_id},
                }
            )

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))

        return await self._deserialize_agent(agent_document=agent_document)

    @override
    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
    ) -> Agent:
        async with self._lock.writer_lock:
            agent_document = await self._agents_collection.find_one(
                filters={
                    "id": {"$eq": agent_id},
                }
            )

            if not agent_document:
                raise ItemNotFoundError(item_id=UniqueId(agent_id))

            update_payload = cast(_AgentDocument, to_json_dict(params))
            update_payload["last_modified"] = datetime.now(timezone.utc).isoformat()

            result = await self._agents_collection.update_one(
                filters={"id": {"$eq": agent_id}},
                params=update_payload,
            )

        assert result.updated_document

        return await self._deserialize_agent(agent_document=result.updated_document)

    @override
    async def delete_agent(
        self,
        agent_id: AgentId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._agents_collection.delete_one({"id": {"$eq": agent_id}})

            for doc in await self._tag_association_collection.find(
                filters={
                    "agent_id": {"$eq": agent_id},
                }
            ):
                await self._tag_association_collection.delete_one(
                    filters={"id": {"$eq": doc["id"]}}
                )

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))

    @override
    async def upsert_group(
        self,
        agent_id: AgentId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            agent = await self.read_agent(agent_id)

            if group_id in agent.groups:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = xxh3_checksum(f"{agent_id}{group_id}")

            association_document: _AgentTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "agent_id": agent_id,
                "group_id": group_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            agent_document = await self._agents_collection.find_one({"id": {"$eq": agent_id}})

            if agent_document:
                await self._agents_collection.update_one(
                    filters={"id": {"$eq": agent_id}},
                    params={"last_modified": datetime.now(timezone.utc).isoformat()},
                )

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))

        return True

    @override
    async def remove_group(
        self,
        agent_id: AgentId,
        group_id: GroupId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "agent_id": {"$eq": agent_id},
                    "group_id": {"$eq": group_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(group_id))

            agent_document = await self._agents_collection.find_one({"id": {"$eq": agent_id}})

            if agent_document:
                await self._agents_collection.update_one(
                    filters={"id": {"$eq": agent_id}},
                    params={"last_modified": datetime.now(timezone.utc).isoformat()},
                )

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))


class CompositeAgentStore(AgentStore):
    def __init__(
        self,
        writable_store: AgentStore,
        readable_stores: Sequence[AgentStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[AgentStore] = [writable_store, *readable_stores]

    @override
    async def create_agent(
        self,
        name: str,
        instructions: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_steps: Optional[int] = None,
        message_output_mode: Optional[MessageOutputMode] = None,
        engine: Optional[str] = None,
        effort: Optional[Effort] = None,
        moderation: Optional[Moderation] = None,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[AgentId] = None,
        message_review: bool = False,
        tool_review: bool = False,
    ) -> Agent:
        return await self._writable_store.create_agent(
            name=name,
            instructions=instructions,
            creation_utc=creation_utc,
            max_steps=max_steps,
            message_output_mode=message_output_mode,
            engine=engine,
            effort=effort,
            moderation=moderation,
            groups=groups,
            id=id,
            message_review=message_review,
            tool_review=tool_review,
        )

    @override
    async def list_agents(
        self,
    ) -> Sequence[Agent]:
        results = await safe_gather(*[store.list_agents() for store in self._all_stores])
        return list(chain.from_iterable(results))

    @override
    async def read_agent(
        self,
        agent_id: AgentId,
    ) -> Agent:
        results = await safe_gather(
            *[try_or_none(store.read_agent(agent_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))
        return result

    @override
    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
    ) -> Agent:
        return await self._writable_store.update_agent(agent_id, params)

    @override
    async def delete_agent(
        self,
        agent_id: AgentId,
    ) -> None:
        return await self._writable_store.delete_agent(agent_id)

    @override
    async def upsert_group(
        self,
        agent_id: AgentId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        return await self._writable_store.upsert_group(agent_id, group_id, creation_utc)

    @override
    async def remove_group(
        self,
        agent_id: AgentId,
        group_id: GroupId,
    ) -> None:
        return await self._writable_store.remove_group(agent_id, group_id)
