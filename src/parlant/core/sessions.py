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

from parlant.core.async_utils import Timeout
from parlant.core.common import (
    ItemNotFoundError,
    JSONSerializable,
    UniqueId,
    Version,
    generate_id,
)
from parlant.core.agents import AgentId
from parlant.core.context_variables import ContextVariableId
from parlant.core.end_users import EndUserId
from parlant.core.guidelines import GuidelineId
from parlant.core.persistence.document_database import (
    DocumentDatabase,
    ObjectId,
    Where,
)
from parlant.core.glossary import TermId

SessionId = NewType("SessionId", str)

EventId = NewType("EventId", str)
EventSource: TypeAlias = Literal[
    "end_user",
    "end_user_ui",
    "human_agent",
    "human_agent_on_behalf_of_ai_agent",
    "ai_agent",
]
EventKind: TypeAlias = Literal["message", "tool", "status", "custom"]


@dataclass(frozen=True)
class Event:
    id: EventId
    source: EventSource
    kind: EventKind
    creation_utc: datetime
    offset: int
    correlation_id: str
    data: JSONSerializable
    deleted: bool

    def is_from_client(self) -> bool:
        return self.source in list[EventSource](
            [
                "end_user",
                "end_user_ui",
            ]
        )

    def is_from_server(self) -> bool:
        return self.source in list[EventSource](
            [
                "human_agent",
                "human_agent_on_behalf_of_ai_agent",
                "ai_agent",
            ]
        )


class Participant(TypedDict):
    id: NotRequired[AgentId | EndUserId | None]
    display_name: str


class MessageEventData(TypedDict):
    message: str
    participant: Participant
    flagged: NotRequired[bool]
    tags: NotRequired[list[str]]


class ControlOptions(TypedDict, total=False):
    mode: SessionMode


class ToolResult(TypedDict):
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable]
    control: ControlOptions


class ToolCall(TypedDict):
    tool_id: str
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


class GuidelineProposition(TypedDict):
    guideline_id: GuidelineId
    predicate: str
    action: str
    score: int
    rationale: str


class Term(TypedDict):
    id: TermId
    name: str
    description: str
    synonyms: list[str]


class ContextVariable(TypedDict):
    id: ContextVariableId
    name: str
    description: Optional[str]
    key: str
    value: JSONSerializable


@dataclass(frozen=True)
class PreparationIteration:
    guideline_propositions: Sequence[GuidelineProposition]
    tool_calls: Sequence[ToolCall]
    terms: Sequence[Term]
    context_variables: Sequence[ContextVariable]


@dataclass(frozen=True)
class Inspection:
    preparation_iterations: Sequence[PreparationIteration]


ConsumerId: TypeAlias = Literal["client"]
"""In the future we may support multiple consumer IDs"""

SessionMode: TypeAlias = Literal["auto", "manual"]


@dataclass(frozen=True)
class Session:
    id: SessionId
    creation_utc: datetime
    end_user_id: EndUserId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]


class SessionUpdateParams(TypedDict, total=False):
    end_user_id: EndUserId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]


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
    async def list_sessions(
        self,
        agent_id: Optional[AgentId] = None,
        end_user_id: Optional[EndUserId] = None,
    ) -> Sequence[Session]: ...

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
    async def read_event(
        self,
        session_id: SessionId,
        event_id: EventId,
    ) -> Event: ...

    @abstractmethod
    async def delete_event(
        self,
        event_id: EventId,
    ) -> EventId: ...

    @abstractmethod
    async def list_events(
        self,
        session_id: SessionId,
        source: Optional[EventSource] = None,
        correlation_id: Optional[str] = None,
        kinds: Sequence[EventKind] = [],
        min_offset: Optional[int] = None,
        exclude_deleted: bool = True,
    ) -> Sequence[Event]: ...

    @abstractmethod
    async def create_inspection(
        self,
        session_id: SessionId,
        correlation_id: str,
        preparation_iterations: Sequence[PreparationIteration],
    ) -> Inspection: ...

    @abstractmethod
    async def read_inspection(
        self,
        session_id: SessionId,
        correlation_id: str,
    ) -> Inspection: ...


class _SessionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    end_user_id: EndUserId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]


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
    deleted: bool


class _PreparationIterationDocument(TypedDict):
    guideline_propositions: Sequence[GuidelineProposition]
    tool_calls: Sequence[ToolCall]
    terms: Sequence[Term]
    context_variables: Sequence[ContextVariable]


class _MessageInspectionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    session_id: SessionId
    correlation_id: str
    preparation_iterations: Sequence[_PreparationIterationDocument]


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
        self._message_inspection_collection = database.get_or_create_collection(
            name="message_inspections",
            schema=_MessageInspectionDocument,
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
            mode=session.mode,
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
            mode=session_document["mode"],
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
            deleted=event.deleted,
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
            deleted=event_document["deleted"],
        )

    def _serialize_message_inspection(
        self,
        message_inspection: Inspection,
        session_id: SessionId,
        correlation_id: str,
    ) -> _MessageInspectionDocument:
        return _MessageInspectionDocument(
            id=ObjectId(generate_id()),
            version=self.VERSION.to_string(),
            session_id=session_id,
            correlation_id=correlation_id,
            preparation_iterations=[
                {
                    "guideline_propositions": i.guideline_propositions,
                    "tool_calls": i.tool_calls,
                    "terms": i.terms,
                    "context_variables": i.context_variables,
                }
                for i in message_inspection.preparation_iterations
            ],
        )

    def _deserialize_message_inspection(
        self,
        message_inspection_document: _MessageInspectionDocument,
    ) -> Inspection:
        return Inspection(
            preparation_iterations=[
                PreparationIteration(
                    guideline_propositions=i["guideline_propositions"],
                    tool_calls=i["tool_calls"],
                    terms=i["terms"],
                    context_variables=i["context_variables"],
                )
                for i in message_inspection_document["preparation_iterations"]
            ],
        )

    async def create_session(
        self,
        end_user_id: EndUserId,
        agent_id: AgentId,
        creation_utc: Optional[datetime] = None,
        title: Optional[str] = None,
        mode: Optional[SessionMode] = None,
    ) -> Session:
        creation_utc = creation_utc or datetime.now(timezone.utc)

        consumption_offsets: dict[ConsumerId, int] = {"client": 0}

        session = Session(
            id=SessionId(generate_id()),
            creation_utc=creation_utc,
            end_user_id=end_user_id,
            agent_id=agent_id,
            mode=mode or "auto",
            consumption_offsets=consumption_offsets,
            title=title,
        )

        await self._session_collection.insert_one(document=self._serialize_session(session))

        return session

    async def delete_session(
        self,
        session_id: SessionId,
    ) -> Optional[SessionId]:
        events = await self._event_collection.find(filters={"session_id": {"$eq": session_id}})
        asyncio.gather(
            *(self._event_collection.delete_one(filters={"id": {"$eq": e["id"]}}) for e in events)
        )

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
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

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

    async def list_sessions(
        self,
        agent_id: Optional[AgentId] = None,
        end_user_id: Optional[EndUserId] = None,
    ) -> Sequence[Session]:
        filters = {
            **({"agent_id": {"$eq": agent_id}} if agent_id else {}),
            **({"end_user_id": {"$eq": end_user_id}} if end_user_id else {}),
        }

        return [
            self._deserialize_session(d)
            for d in await self._session_collection.find(filters=cast(Where, filters))
        ]

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
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        session_events = await self.list_events(
            session_id
        )  # FIXME: we need a more efficient way to do this
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
            deleted=False,
        )

        await self._event_collection.insert_one(document=self._serialize_event(event, session_id))

        return event

    async def read_event(
        self,
        session_id: SessionId,
        event_id: EventId,
    ) -> Event:
        if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        if event_document := await self._event_collection.find_one(
            filters={"id": {"$eq": event_id}}
        ):
            return self._deserialize_event(event_document)

        raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")

    async def delete_event(
        self,
        event_id: EventId,
    ) -> EventId:
        result = await self._event_collection.update_one(
            filters={"id": {"$eq": event_id}},
            params=cast(_EventDocument, {"deleted": True}),
        )

        if result.matched_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")

        return event_id

    async def list_events(
        self,
        session_id: SessionId,
        source: Optional[EventSource] = None,
        correlation_id: Optional[str] = None,
        kinds: Sequence[EventKind] = [],
        min_offset: Optional[int] = None,
        exclude_deleted: bool = True,
    ) -> Sequence[Event]:
        if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        base_filters = {
            "session_id": {"$eq": session_id},
            **({"source": {"$eq": source}} if source else {}),
            **({"offset": {"$gte": min_offset}} if min_offset else {}),
            **({"correlation_id": {"$eq": correlation_id}} if correlation_id else {}),
            **({"deleted": {"$eq": False}} if exclude_deleted else {}),
        }

        if kinds:
            event_documents = await self._event_collection.find(
                cast(Where, {"$or": [{**base_filters, "kind": {"$eq": k}} for k in kinds]})
            )
        else:
            event_documents = await self._event_collection.find(
                cast(
                    Where,
                    base_filters,
                )
            )

        return [self._deserialize_event(d) for d in event_documents]

    async def create_inspection(
        self,
        session_id: SessionId,
        correlation_id: str,
        preparation_iterations: Sequence[PreparationIteration],
    ) -> Inspection:
        if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        message_inspection = Inspection(
            preparation_iterations=preparation_iterations,
        )

        await self._message_inspection_collection.insert_one(
            document=self._serialize_message_inspection(
                message_inspection,
                session_id,
                correlation_id,
            )
        )

        return message_inspection

    async def read_inspection(
        self,
        session_id: SessionId,
        correlation_id: str,
    ) -> Inspection:
        if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        if not await self._event_collection.find_one(
            filters={
                "correlation_id": {"$eq": correlation_id},
                "kind": {"$eq": "message"},
            }
        ):
            raise ItemNotFoundError(
                item_id=UniqueId(correlation_id), message="Message event not found"
            )

        if message_inspection_document := await self._message_inspection_collection.find_one(
            filters={"correlation_id": {"$eq": correlation_id}}
        ):
            return self._deserialize_message_inspection(message_inspection_document)

        raise ItemNotFoundError(
            item_id=UniqueId(correlation_id), message="Message inspection not found"
        )


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
            )

            if events:
                return True
            elif timeout.expired():
                return False
            else:
                await timeout.wait_up_to(1)
