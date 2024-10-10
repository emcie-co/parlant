from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Literal,
    Mapping,
    NewType,
    NotRequired,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
    cast,
)

from emcie.server.core.async_utils import Timeout
from emcie.server.core.common import (
    ItemNotFoundError,
    JSONSerializable,
    UniqueId,
    Version,
    generate_id,
)
from emcie.server.core.agents import AgentId
from emcie.server.core.end_users import EndUserId
from emcie.server.core.persistence.document_database import (
    DocumentDatabase,
    ObjectId,
    Where,
)

SessionId = NewType("SessionId", str)
EventId = NewType("EventId", str)


EventSource: TypeAlias = Literal["client", "server"]
EventKind: TypeAlias = Literal["message", "tool", "status", "custom"]


@dataclass(frozen=True)
class Event:
    id: EventId
    source: EventSource
    kind: EventKind
    creation_utc: datetime
    offset: int
    correlation_id: str
    is_skipped: bool
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


SessionStatus: TypeAlias = Literal[
    "acknowledged",
    "cancelled",
    "processing",
    "ready",
    "typing",
    "error",
]


class StatusEventData(TypedDict):
    acknowledged_offset: NotRequired[int]
    status: SessionStatus
    data: JSONSerializable


ConsumerId: TypeAlias = Literal["client"]
"""In the future we may support multiple consumer IDs"""


@dataclass(frozen=True)
class Session:
    id: SessionId
    creation_utc: datetime
    end_user_id: EndUserId
    agent_id: AgentId
    title: Optional[str]
    consumption_offsets: dict[ConsumerId, int]


class SessionUpdateParams(TypedDict, total=False):
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
        creation_utc: Optional[datetime] = None,
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
    ) -> Optional[SessionId]: ...

    @abstractmethod
    async def update_session(
        self,
        session_id: SessionId,
        params: SessionUpdateParams,
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
        kinds: Sequence[EventKind] = [],
        min_offset: Optional[int] = None,
        include_skipped: Optional[bool] = False,
    ) -> Sequence[Event]: ...

    @abstractmethod
    async def list_sessions(
        self,
        agent_id: Optional[AgentId],
    ) -> Sequence[Session]: ...


class _SessionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    end_user_id: EndUserId
    agent_id: AgentId
    title: Optional[str]
    consumption_offsets: dict[ConsumerId, int]


class _EventDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    session_id: SessionId
    source: EventSource
    kind: EventKind
    offset: int
    correlation_id: str
    data: JSONSerializable
    is_skipped: bool


class SessionDocumentStore(SessionStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase):
        self._session_collection = database.get_or_create_collection(
            name="sessions",
            schema=_SessionDocument,
        )
        self._event_collection = database.get_or_create_collection(
            name="events",
            schema=_EventDocument,
        )

    def _serialize_session(
        self,
        session: Session,
    ) -> _SessionDocument:
        return _SessionDocument(
            id=ObjectId(session.id),
            version=self.VERSION.to_string(),
            creation_utc=session.creation_utc.isoformat(),
            end_user_id=session.end_user_id,
            agent_id=session.agent_id,
            title=session.title if session.title else None,
            consumption_offsets=session.consumption_offsets,
        )

    def _deserialize_session(
        self,
        session_document: _SessionDocument,
    ) -> Session:
        return Session(
            id=SessionId(session_document["id"]),
            creation_utc=datetime.fromisoformat(session_document["creation_utc"]),
            end_user_id=session_document["end_user_id"],
            agent_id=session_document["agent_id"],
            title=session_document["title"],
            consumption_offsets=session_document["consumption_offsets"],
        )

    def _serialize_event(
        self,
        event: Event,
        session_id: SessionId,
    ) -> _EventDocument:
        return _EventDocument(
            id=ObjectId(event.id),
            version=self.VERSION.to_string(),
            creation_utc=event.creation_utc.isoformat(),
            session_id=session_id,
            source=event.source,
            kind=event.kind,
            offset=event.offset,
            correlation_id=event.correlation_id,
            data=event.data,
            is_skipped=event.is_skipped,
        )

    def _deserialize_event(
        self,
        event_document: _EventDocument,
    ) -> Event:
        return Event(
            id=EventId(event_document["id"]),
            creation_utc=datetime.fromisoformat(event_document["creation_utc"]),
            source=event_document["source"],
            kind=event_document["kind"],
            offset=event_document["offset"],
            correlation_id=event_document["correlation_id"],
            data=event_document["data"],
            is_skipped=event_document["is_skipped"],
        )

    async def create_session(
        self,
        end_user_id: EndUserId,
        agent_id: AgentId,
        creation_utc: Optional[datetime] = None,
        title: Optional[str] = None,
    ) -> Session:
        creation_utc = creation_utc or datetime.now(timezone.utc)

        consumption_offsets: dict[ConsumerId, int] = {"client": 0}

        session = Session(
            id=SessionId(generate_id()),
            creation_utc=creation_utc,
            end_user_id=end_user_id,
            agent_id=agent_id,
            consumption_offsets=consumption_offsets,
            title=title,
        )

        await self._session_collection.insert_one(document=self._serialize_session(session))

        return session

    async def delete_session(
        self,
        session_id: SessionId,
    ) -> Optional[SessionId]:
        events_to_delete = await self.list_events(session_id=session_id)
        asyncio.gather(*iter(self.delete_event(event_id=e.id) for e in events_to_delete))

        result = await self._session_collection.delete_one({"id": {"$eq": session_id}})

        return session_id if result.deleted_count else None

    async def read_session(
        self,
        session_id: SessionId,
    ) -> Session:
        session_document = await self._session_collection.find_one(
            filters={"id": {"$eq": session_id}}
        )
        if not session_document:
            raise ItemNotFoundError(item_id=UniqueId(session_id))

        return self._deserialize_session(session_document)

    async def update_session(
        self,
        session_id: SessionId,
        params: SessionUpdateParams,
    ) -> None:
        await self._session_collection.update_one(
            filters={"id": {"$eq": session_id}},
            params=cast(_SessionDocument, params),
        )

    async def create_event(
        self,
        session_id: SessionId,
        source: EventSource,
        kind: EventKind,
        correlation_id: str,
        data: JSONSerializable,
        creation_utc: Optional[datetime] = None,
    ) -> Event:
        if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
            raise ItemNotFoundError(item_id=UniqueId(session_id))

        session_events = await self.list_events(session_id)
        creation_utc = creation_utc or datetime.now(timezone.utc)
        offset = len(list(session_events))

        event = Event(
            id=EventId(generate_id()),
            source=source,
            kind=kind,
            offset=offset,
            creation_utc=creation_utc,
            correlation_id=correlation_id,
            data=data,
            is_skipped=False,
        )

        await self._event_collection.insert_one(document=self._serialize_event(event, session_id))

        return event

    async def delete_event(
        self,
        event_id: EventId,
    ) -> None:
        result = await self._event_collection.delete_one(filters={"id": {"$eq": event_id}})
        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(event_id))

    async def list_events(
        self,
        session_id: SessionId,
        source: Optional[EventSource] = None,
        kinds: Sequence[EventKind] = [],
        min_offset: Optional[int] = None,
        include_skipped: Optional[bool] = None,
    ) -> Sequence[Event]:
        if not await self._session_collection.find_one(
            cast(
                Where,
                {
                    "id": {"$eq": session_id},
                    **({"is_skipped": {"$eq": include_skipped}} if include_skipped else {}),
                },
            )
        ):
            raise ItemNotFoundError(item_id=UniqueId(session_id))

        if kinds:
            event_documents = await self._event_collection.find(
                {
                    "$or": [
                        cast(
                            Where,
                            {
                                "kind": {"$eq": k},
                                "session_id": {"$eq": session_id},
                                **({"source": {"$eq": source}} if source else {}),
                                **({"offset": {"$gte": min_offset}} if min_offset else {}),
                            },
                        )
                        for k in kinds
                    ],
                }
            )
        else:
            event_documents = await self._event_collection.find(
                cast(
                    Where,
                    {
                        "session_id": {"$eq": session_id},
                        **({"source": {"$eq": source}} if source else {}),
                        **({"offset": {"$gte": min_offset}} if min_offset else {}),
                    },
                )
            )

        return [self._deserialize_event(d) for d in event_documents]

    async def list_sessions(
        self,
        agent_id: Optional[AgentId] = None,
    ) -> Sequence[Session]:
        return [
            self._deserialize_session(d)
            for d in await self._session_collection.find(
                filters={"agent_id": {"$eq": agent_id}} if agent_id else {}
            )
        ]


class SessionListener(ABC):
    @abstractmethod
    async def wait_for_events(
        self,
        session_id: SessionId,
        min_offset: int,
        kinds: Sequence[EventKind],
        source: Optional[EventSource] = None,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool: ...


class PollingSessionListener(SessionListener):
    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store

    async def wait_for_events(
        self,
        session_id: SessionId,
        min_offset: int,
        kinds: Sequence[EventKind],
        source: Optional[EventSource] = None,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        while True:
            events = await self._session_store.list_events(
                session_id,
                min_offset=min_offset,
                source=source,
                kinds=kinds,
                include_skipped=False,
            )

            if events:
                return True
            elif timeout.expired():
                return False
            else:
                await timeout.wait_up_to(1)
