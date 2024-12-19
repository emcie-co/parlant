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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Literal,
    Mapping,
    NewType,
    Optional,
    Sequence,
    TypeAlias,
    cast,
)
from typing_extensions import override, TypedDict, NotRequired, Self

from parlant.core import async_utils
from parlant.core.async_utils import ReaderWriterLock, Timeout
from parlant.core.common import (
    ItemNotFoundError,
    JSONSerializable,
    UniqueId,
    Version,
    generate_id,
)
from parlant.core.agents import AgentId
from parlant.core.context_variables import ContextVariableId
from parlant.core.customers import CustomerId
from parlant.core.guidelines import GuidelineId
from parlant.core.nlp.generation import GenerationInfo, UsageInfo
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import DocumentDatabase, DocumentCollection
from parlant.core.glossary import TermId

SessionId = NewType("SessionId", str)

EventId = NewType("EventId", str)
EventSource: TypeAlias = Literal[
    "customer",
    "customer_ui",
    "human_agent",
    "human_agent_on_behalf_of_ai_agent",
    "ai_agent",
    "system",
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
                "customer",
                "customer_ui",
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
    id: NotRequired[AgentId | CustomerId | None]
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
    condition: str
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
class MessageGenerationInspection:
    generation: GenerationInfo
    messages: Sequence[Optional[str]]


@dataclass(frozen=True)
class GuidelinePropositionInspection:
    total_duration: float
    batches: Sequence[GenerationInfo]


@dataclass(frozen=True)
class PreparationIterationGenerations:
    guideline_proposition: GuidelinePropositionInspection
    tool_calls: Sequence[GenerationInfo]


@dataclass(frozen=True)
class PreparationIteration:
    guideline_propositions: Sequence[GuidelineProposition]
    tool_calls: Sequence[ToolCall]
    terms: Sequence[Term]
    context_variables: Sequence[ContextVariable]
    generations: PreparationIterationGenerations


@dataclass(frozen=True)
class Inspection:
    message_generations: Sequence[MessageGenerationInspection]
    preparation_iterations: Sequence[PreparationIteration]


ConsumerId: TypeAlias = Literal["client"]
"""In the future we may support multiple consumer IDs"""

SessionMode: TypeAlias = Literal["auto", "manual"]


@dataclass(frozen=True)
class Session:
    id: SessionId
    creation_utc: datetime
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]


class SessionUpdateParams(TypedDict, total=False):
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]


class SessionStore(ABC):
    @abstractmethod
    async def create_session(
        self,
        customer_id: CustomerId,
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
    ) -> None: ...

    @abstractmethod
    async def update_session(
        self,
        session_id: SessionId,
        params: SessionUpdateParams,
    ) -> Session: ...

    @abstractmethod
    async def list_sessions(
        self,
        agent_id: Optional[AgentId] = None,
        customer_id: Optional[CustomerId] = None,
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
    ) -> None: ...

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
        message_generations: Sequence[MessageGenerationInspection],
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
    customer_id: CustomerId
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


class _UsageInfoDocument(TypedDict):
    input_tokens: int
    output_tokens: int
    extra: Optional[Mapping[str, int]]


class _GenerationInfoDocument(TypedDict):
    schema_name: str
    model: str
    duration: float
    usage: _UsageInfoDocument


class _GuidelinePropositionInspectionDocument(TypedDict):
    total_duration: float
    batches: Sequence[_GenerationInfoDocument]


class _PreparationIterationGenerationsDocument(TypedDict):
    guideline_proposition: _GuidelinePropositionInspectionDocument
    tool_calls: Sequence[_GenerationInfoDocument]


class _MessageGenerationInspectionDocument(TypedDict):
    generation: _GenerationInfoDocument
    messages: Sequence[Optional[str]]


class _PreparationIterationDocument(TypedDict):
    guideline_propositions: Sequence[GuidelineProposition]
    tool_calls: Sequence[ToolCall]
    terms: Sequence[Term]
    context_variables: Sequence[ContextVariable]
    generations: _PreparationIterationGenerationsDocument


class _InspectionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    session_id: SessionId
    correlation_id: str
    message_generations: Sequence[_MessageGenerationInspectionDocument]
    preparation_iterations: Sequence[_PreparationIterationDocument]


class SessionDocumentStore(SessionStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase):
        self._database = database
        self._session_collection: DocumentCollection[_SessionDocument]
        self._event_collection: DocumentCollection[_EventDocument]
        self._inspection_collection: DocumentCollection[_InspectionDocument]

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._session_collection = await self._database.get_or_create_collection(
            name="sessions",
            schema=_SessionDocument,
        )
        self._event_collection = await self._database.get_or_create_collection(
            name="events",
            schema=_EventDocument,
        )
        self._inspection_collection = await self._database.get_or_create_collection(
            name="inspections",
            schema=_InspectionDocument,
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize_session(
        self,
        session: Session,
    ) -> _SessionDocument:
        return _SessionDocument(
            id=ObjectId(session.id),
            version=self.VERSION.to_string(),
            creation_utc=session.creation_utc.isoformat(),
            customer_id=session.customer_id,
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
            customer_id=session_document["customer_id"],
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

    def _serialize_inspection(
        self,
        inspection: Inspection,
        session_id: SessionId,
        correlation_id: str,
    ) -> _InspectionDocument:
        def serialize_generation_info(generation: GenerationInfo) -> _GenerationInfoDocument:
            return _GenerationInfoDocument(
                schema_name=generation.schema_name,
                model=generation.model,
                duration=generation.duration,
                usage=_UsageInfoDocument(
                    input_tokens=generation.usage.input_tokens,
                    output_tokens=generation.usage.output_tokens,
                    extra=generation.usage.extra,
                ),
            )

        return _InspectionDocument(
            id=ObjectId(generate_id()),
            version=self.VERSION.to_string(),
            session_id=session_id,
            correlation_id=correlation_id,
            message_generations=[
                _MessageGenerationInspectionDocument(
                    generation=serialize_generation_info(m.generation), messages=m.messages
                )
                for m in inspection.message_generations
            ],
            preparation_iterations=[
                {
                    "guideline_propositions": i.guideline_propositions,
                    "tool_calls": i.tool_calls,
                    "terms": i.terms,
                    "context_variables": i.context_variables,
                    "generations": _PreparationIterationGenerationsDocument(
                        guideline_proposition=_GuidelinePropositionInspectionDocument(
                            total_duration=i.generations.guideline_proposition.total_duration,
                            batches=[
                                serialize_generation_info(g)
                                for g in i.generations.guideline_proposition.batches
                            ],
                        ),
                        tool_calls=[serialize_generation_info(g) for g in i.generations.tool_calls],
                    ),
                }
                for i in inspection.preparation_iterations
            ],
        )

    def _deserialize_message_inspection(
        self,
        inspection_document: _InspectionDocument,
    ) -> Inspection:
        def deserialize_generation_info(
            generation_document: _GenerationInfoDocument,
        ) -> GenerationInfo:
            return GenerationInfo(
                schema_name=generation_document["schema_name"],
                model=generation_document["model"],
                duration=generation_document["duration"],
                usage=UsageInfo(
                    input_tokens=generation_document["usage"]["input_tokens"],
                    output_tokens=generation_document["usage"]["output_tokens"],
                    extra=generation_document["usage"]["extra"],
                ),
            )

        return Inspection(
            message_generations=[
                MessageGenerationInspection(
                    generation=deserialize_generation_info(m["generation"]), messages=m["messages"]
                )
                for m in inspection_document["message_generations"]
            ],
            preparation_iterations=[
                PreparationIteration(
                    guideline_propositions=i["guideline_propositions"],
                    tool_calls=i["tool_calls"],
                    terms=i["terms"],
                    context_variables=i["context_variables"],
                    generations=PreparationIterationGenerations(
                        guideline_proposition=GuidelinePropositionInspection(
                            total_duration=i["generations"]["guideline_proposition"][
                                "total_duration"
                            ],
                            batches=[
                                deserialize_generation_info(g)
                                for g in i["generations"]["guideline_proposition"]["batches"]
                            ],
                        ),
                        tool_calls=[
                            deserialize_generation_info(g) for g in i["generations"]["tool_calls"]
                        ],
                    ),
                )
                for i in inspection_document["preparation_iterations"]
            ],
        )

    @override
    async def create_session(
        self,
        customer_id: CustomerId,
        agent_id: AgentId,
        creation_utc: Optional[datetime] = None,
        title: Optional[str] = None,
        mode: Optional[SessionMode] = None,
    ) -> Session:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            consumption_offsets: dict[ConsumerId, int] = {"client": 0}

            session = Session(
                id=SessionId(generate_id()),
                creation_utc=creation_utc,
                customer_id=customer_id,
                agent_id=agent_id,
                mode=mode or "auto",
                consumption_offsets=consumption_offsets,
                title=title,
            )

            await self._session_collection.insert_one(document=self._serialize_session(session))

        return session

    @override
    async def delete_session(
        self,
        session_id: SessionId,
    ) -> None:
        async with self._lock.writer_lock:
            events = await self._event_collection.find(filters={"session_id": {"$eq": session_id}})
            await async_utils.safe_gather(
                *(
                    self._event_collection.delete_one(filters={"id": {"$eq": e["id"]}})
                    for e in events
                )
            )

            await self._session_collection.delete_one({"id": {"$eq": session_id}})

    @override
    async def read_session(
        self,
        session_id: SessionId,
    ) -> Session:
        async with self._lock.reader_lock:
            session_document = await self._session_collection.find_one(
                filters={"id": {"$eq": session_id}}
            )

        if not session_document:
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        return self._deserialize_session(session_document)

    @override
    async def update_session(
        self,
        session_id: SessionId,
        params: SessionUpdateParams,
    ) -> Session:
        async with self._lock.writer_lock:
            session_document = await self._session_collection.find_one(
                filters={"id": {"$eq": session_id}}
            )

            if not session_document:
                raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

            result = await self._session_collection.update_one(
                filters={"id": {"$eq": session_id}},
                params=cast(_SessionDocument, params),
            )

        assert result.updated_document

        return self._deserialize_session(session_document=result.updated_document)

    @override
    async def list_sessions(
        self,
        agent_id: Optional[AgentId] = None,
        customer_id: Optional[CustomerId] = None,
    ) -> Sequence[Session]:
        async with self._lock.reader_lock:
            filters = {
                **({"agent_id": {"$eq": agent_id}} if agent_id else {}),
                **({"customer_id": {"$eq": customer_id}} if customer_id else {}),
            }

            return [
                self._deserialize_session(d)
                for d in await self._session_collection.find(filters=cast(Where, filters))
            ]

    @override
    async def create_event(
        self,
        session_id: SessionId,
        source: EventSource,
        kind: EventKind,
        correlation_id: str,
        data: JSONSerializable,
        creation_utc: Optional[datetime] = None,
    ) -> Event:
        async with self._lock.writer_lock:
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

            await self._event_collection.insert_one(
                document=self._serialize_event(event, session_id)
            )

        return event

    @override
    async def read_event(
        self,
        session_id: SessionId,
        event_id: EventId,
    ) -> Event:
        async with self._lock.reader_lock:
            if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
                raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

            if event_document := await self._event_collection.find_one(
                filters={"id": {"$eq": event_id}}
            ):
                return self._deserialize_event(event_document)

        raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")

    @override
    async def delete_event(
        self,
        event_id: EventId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._event_collection.update_one(
                filters={"id": {"$eq": event_id}},
                params=cast(_EventDocument, {"deleted": True}),
            )

        if result.matched_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")

    @override
    async def list_events(
        self,
        session_id: SessionId,
        source: Optional[EventSource] = None,
        correlation_id: Optional[str] = None,
        kinds: Sequence[EventKind] = [],
        min_offset: Optional[int] = None,
        exclude_deleted: bool = True,
    ) -> Sequence[Event]:
        async with self._lock.reader_lock:
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

    @override
    async def create_inspection(
        self,
        session_id: SessionId,
        correlation_id: str,
        message_generations: Sequence[MessageGenerationInspection],
        preparation_iterations: Sequence[PreparationIteration],
    ) -> Inspection:
        async with self._lock.writer_lock:
            if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
                raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

            inspection = Inspection(
                message_generations=message_generations,
                preparation_iterations=preparation_iterations,
            )

            await self._inspection_collection.insert_one(
                document=self._serialize_inspection(
                    inspection,
                    session_id,
                    correlation_id,
                )
            )

        return inspection

    @override
    async def read_inspection(
        self,
        session_id: SessionId,
        correlation_id: str,
    ) -> Inspection:
        async with self._lock.reader_lock:
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

            if inspection_document := await self._inspection_collection.find_one(
                filters={"correlation_id": {"$eq": correlation_id}}
            ):
                return self._deserialize_message_inspection(inspection_document)

        raise ItemNotFoundError(
            item_id=UniqueId(correlation_id), message="Message inspection not found"
        )


class SessionListener(ABC):
    @abstractmethod
    async def wait_for_events(
        self,
        session_id: SessionId,
        kinds: Sequence[EventKind] = [],
        min_offset: Optional[int] = None,
        source: Optional[EventSource] = None,
        correlation_id: Optional[str] = None,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool: ...


class PollingSessionListener(SessionListener):
    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store

    @override
    async def wait_for_events(
        self,
        session_id: SessionId,
        kinds: Sequence[EventKind] = [],
        min_offset: Optional[int] = None,
        source: Optional[EventSource] = None,
        correlation_id: Optional[str] = None,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        # Trigger exception if not found
        _ = await self._session_store.read_session(session_id)

        while True:
            events = await self._session_store.list_events(
                session_id,
                min_offset=min_offset,
                source=source,
                kinds=kinds,
                correlation_id=correlation_id,
            )

            if events:
                return True
            elif timeout.expired():
                return False
            else:
                await timeout.wait_up_to(1)
