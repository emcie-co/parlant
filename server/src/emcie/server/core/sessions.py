from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, NewType, Optional, Sequence, TypeAlias, TypedDict, cast

from emcie.server.async_utils import Timeout
from emcie.server.core.common import ItemNotFoundError, JSONSerializable, UniqueId, generate_id
from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.agents import AgentId
from emcie.server.core.end_users import EndUserId
from emcie.server.core.persistence.common import NoMatchingDocumentsError, Where
from emcie.server.core.persistence.document_database import DocumentDatabase

SessionId = NewType("SessionId", str)
EventId = NewType("EventId", str)


EventSource: TypeAlias = Literal["client", "server"]
EventKind: TypeAlias = Literal["message", "tool", "custom"]


@dataclass(frozen=True)
class Event:
    id: EventId
    source: EventSource
    kind: EventKind
    creation_utc: datetime
    offset: int
    correlation_id: str
    data: JSONSerializable


class MessageEventData(TypedDict):
    message: str


class ToolResult(TypedDict):
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable]


class ToolCall(TypedDict):
    tool_name: str
    arguments: Mapping[str, JSONSerializable]
    result: ToolResult


class ToolEventData(TypedDict):
    tool_calls: list[ToolCall]


ConsumerId = Literal["client"]
"""In the future we may support multiple consumer IDs"""


@dataclass(frozen=True)
class Session:
    id: SessionId
    end_user_id: EndUserId
    agent_id: AgentId
    title: Optional[str]
    consumption_offsets: dict[ConsumerId, int]


class SessionStore(ABC):
    @abstractmethod
    async def create_session(
        self,
        end_user_id: EndUserId,
        agent_id: AgentId,
        title: Optional[str] = None,
    ) -> Session: ...

    @abstractmethod
    async def read_session(
        self,
        session_id: SessionId,
    ) -> Session: ...

    @abstractmethod
    async def delete_session(
        self,
        session_id: SessionId,
    ) -> None: ...

    @abstractmethod
    async def update_consumption_offset(
        self,
        session_id: SessionId,
        consumer_id: ConsumerId,
        new_offset: int,
    ) -> None: ...

    @abstractmethod
    async def create_event(
        self,
        session_id: SessionId,
        source: EventSource,
        kind: EventKind,
        correlation_id: str,
        data: JSONSerializable,
        creation_utc: Optional[datetime] = None,
    ) -> Event: ...

    @abstractmethod
    async def delete_event(
        self,
        event_id: EventId,
    ) -> None: ...

    @abstractmethod
    async def list_events(
        self,
        session_id: SessionId,
        source: Optional[EventSource] = None,
        min_offset: Optional[int] = None,
    ) -> Sequence[Event]: ...

    @abstractmethod
    async def list_sessions(
        self,
        agent_id: Optional[AgentId],
    ) -> Sequence[Session]: ...


class SessionDocumentStore(SessionStore):
    class SessionDocument(DefaultBaseModel):
        id: SessionId
        end_user_id: EndUserId
        agent_id: AgentId
        title: Optional[str] = None
        consumption_offsets: dict[ConsumerId, int]

    class EventDocument(DefaultBaseModel):
        id: EventId
        creation_utc: datetime
        session_id: SessionId
        source: EventSource
        kind: str
        offset: int
        correlation_id: str
        data: Any

    def __init__(self, database: DocumentDatabase):
        self._session_collection = database.get_or_create_collection(
            name="sessions",
            schema=self.SessionDocument,
        )
        self._event_collection = database.get_or_create_collection(
            name="events",
            schema=self.EventDocument,
        )

    async def create_session(
        self,
        end_user_id: EndUserId,
        agent_id: AgentId,
        title: Optional[str] = None,
    ) -> Session:
        consumption_offsets: dict[ConsumerId, int] = {"client": 0}

        document = {
            "id": generate_id(),
            "end_user_id": end_user_id,
            "agent_id": agent_id,
            "consumption_offsets": consumption_offsets,
        }

        if title:
            document["title"] = title

        session_id = await self._session_collection.insert_one(document=document)

        return Session(
            id=SessionId(session_id),
            end_user_id=end_user_id,
            agent_id=agent_id,
            consumption_offsets=consumption_offsets,
            title=title,
        )

    async def delete_session(
        self,
        session_id: SessionId,
    ) -> None:
        events_to_delete = await self.list_events(session_id=session_id)
        asyncio.gather(*iter(self.delete_event(event_id=e.id) for e in events_to_delete))

        await self._session_collection.delete_one({"id": {"$eq": session_id}})

    async def read_session(
        self,
        session_id: SessionId,
    ) -> Session:
        try:
            session_document = await self._session_collection.find_one(
                filters={"id": {"$eq": session_id}}
            )
        except NoMatchingDocumentsError:
            raise ItemNotFoundError(item_id=UniqueId(session_id))

        return Session(
            id=session_document["id"],
            end_user_id=session_document["end_user_id"],
            agent_id=session_document["agent_id"],
            consumption_offsets=session_document["consumption_offsets"],
            title=session_document.get("title"),
        )

    async def update_session(
        self,
        session_id: SessionId,
        updated_session: Session,
    ) -> None:
        await self._session_collection.update_one(
            filters={"id": {"$eq": session_id}},
            updated_document=updated_session.__dict__,
        )

    async def update_consumption_offset(
        self,
        session_id: SessionId,
        consumer_id: ConsumerId,
        new_offset: int,
    ) -> None:
        session = await self.read_session(session_id)
        session.consumption_offsets[consumer_id] = new_offset
        await self.update_session(session_id, session)

    async def create_event(
        self,
        session_id: SessionId,
        source: EventSource,
        kind: EventKind,
        correlation_id: str,
        data: JSONSerializable,
        creation_utc: Optional[datetime] = None,
    ) -> Event:
        try:
            await self._session_collection.find_one(filters={"id": {"$eq": session_id}})
        except NoMatchingDocumentsError:
            raise ItemNotFoundError(item_id=UniqueId(session_id))

        session_events = await self.list_events(session_id)
        creation_utc = creation_utc or datetime.now(timezone.utc)
        offset = len(list(session_events))

        event_id = await self._event_collection.insert_one(
            document={
                "id": generate_id(),
                "session_id": session_id,
                "source": source,
                "kind": kind,
                "offset": offset,
                "creation_utc": creation_utc,
                "correlation_id": correlation_id,
                "data": data,
            },
        )

        return Event(
            id=EventId(event_id),
            source=source,
            kind=kind,
            offset=offset,
            creation_utc=creation_utc,
            correlation_id=correlation_id,
            data=data,
        )

    async def delete_event(
        self,
        event_id: EventId,
    ) -> None:
        try:
            await self._event_collection.delete_one(filters={"id": {"$eq": event_id}})
        except NoMatchingDocumentsError:
            raise ItemNotFoundError(item_id=UniqueId(event_id))

    async def list_events(
        self,
        session_id: SessionId,
        source: Optional[EventSource] = None,
        min_offset: Optional[int] = None,
    ) -> Sequence[Event]:
        try:
            await self._session_collection.find_one(filters={"id": {"$eq": session_id}})
        except NoMatchingDocumentsError:
            raise ItemNotFoundError(item_id=UniqueId(session_id))

        return [
            Event(
                id=EventId(d["id"]),
                source=d["source"],
                kind=d["kind"],
                offset=d["offset"],
                creation_utc=d["creation_utc"],
                correlation_id=d["correlation_id"],
                data=d["data"],
            )
            for d in await self._event_collection.find(
                filters=cast(
                    Where,
                    {
                        "session_id": {"$eq": session_id},
                        **({"source": {"$eq": source}} if source else {}),
                        **({"offset": {"$gte": min_offset}} if min_offset else {}),
                    },
                )
            )
        ]

    async def list_sessions(
        self,
        agent_id: Optional[AgentId] = None,
    ) -> Sequence[Session]:
        return [
            Session(
                id=SessionId(s["id"]),
                end_user_id=EndUserId(s["end_user_id"]),
                agent_id=AgentId(s["agent_id"]),
                consumption_offsets=s["consumption_offsets"],
                title=s["title"],
            )
            for s in await self._session_collection.find(
                filters={"agent_id": {"$eq": agent_id}} if agent_id else {}
            )
        ]


class SessionListener(ABC):
    @abstractmethod
    async def wait_for_events(
        self,
        session_id: SessionId,
        min_offset: int,
        timeout: Timeout,
    ) -> bool: ...


class PollingSessionListener(SessionListener):
    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store

    async def wait_for_events(
        self,
        session_id: SessionId,
        min_offset: int,
        timeout: Timeout,
        # TODO: allow filtering based on type (e.g. to filter out tool events)
    ) -> bool:
        while True:
            events = list(
                await self._session_store.list_events(
                    session_id,
                    min_offset=min_offset,
                )
            )

            if events:
                return True
            elif timeout.expired():
                return False
            else:
                await timeout.wait_up_to(1)
