# Copyright 2025 Emcie Co Ltd.
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
from enum import Enum
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
from parlant.core.journeys import JourneyId
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.persistence.common import (
    ObjectId,
    Where,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.glossary import TermId
from parlant.core.canned_responses import CannedResponseId
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)

SessionId = NewType("SessionId", str)

EventId = NewType("EventId", str)


class EventSource(Enum):
    """The source of an event in a session."""

    CUSTOMER = "customer"
    """Represents an event from the customer, such as a message or action."""

    CUSTOMER_UI = "customer_ui"
    """Represents an event from the customer UI, such as a page navigation or button click."""

    HUMAN_AGENT = "human_agent"
    """Represents an event from a human agent, such as a status update, message or action."""

    HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT = "human_agent_on_behalf_of_ai_agent"
    """Represents an event from a human agent acting on behalf of an AI agent, such as a status update, message or action."""

    AI_AGENT = "ai_agent"
    """Represents an event from an AI agent, such as a status update, message or action."""

    SYSTEM = "system"
    """Represents an event from the system, such as a tool execution."""


class EventKind(Enum):
    """The kind of event in a session."""

    MESSAGE = "message"
    """Represents a message event, such as a message sent by the customer or AI agent."""

    TOOL = "tool"
    """Represents a tool event, such as a tool result or tool error."""

    STATUS = "status"
    """Represents a status event, such as a 'typing', 'thinking', etc."""

    CUSTOM = "custom"
    """Represents a custom event, used in custom frontends."""


@dataclass(frozen=True)
class Event:
    """Represents an event in a session."""

    id: EventId
    source: EventSource
    kind: EventKind
    creation_utc: datetime
    offset: int
    correlation_id: str
    data: JSONSerializable
    deleted: bool

    def is_from_client(self) -> bool:
        return self.source in [
            EventSource.CUSTOMER,
            EventSource.CUSTOMER_UI,
        ]

    def is_from_server(self) -> bool:
        return self.source in [
            EventSource.HUMAN_AGENT,
            EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
            EventSource.AI_AGENT,
        ]


class Participant(TypedDict):
    """Represents a participant in a session, such as a customer or AI agent."""

    id: NotRequired[AgentId | CustomerId | None]
    display_name: str


class MessageEventData(TypedDict):
    """Data for a message event in a session."""

    message: str
    participant: Participant
    flagged: NotRequired[bool]
    tags: NotRequired[Sequence[str]]
    draft: NotRequired[str]
    canned_responses: NotRequired[Sequence[tuple[CannedResponseId, str]]]


class ControlOptions(TypedDict, total=False):
    """Options for controlling the behavior of a tool result."""

    mode: SessionMode
    lifespan: LifeSpan


class ToolResult(TypedDict):
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable]
    control: ControlOptions
    canned_responses: Sequence[str]
    canned_response_fields: Mapping[str, JSONSerializable]


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
    status: SessionStatus
    data: JSONSerializable


class GuidelineMatch(TypedDict):
    guideline_id: GuidelineId
    condition: str
    action: Optional[str]
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
    generations: Mapping[str, GenerationInfo]
    messages: Sequence[Optional[str]]


@dataclass(frozen=True)
class GuidelineMatchingInspection:
    total_duration: float
    batches: Sequence[GenerationInfo]


@dataclass(frozen=True)
class PreparationIterationGenerations:
    guideline_matching: GuidelineMatchingInspection
    tool_calls: Sequence[GenerationInfo]


@dataclass(frozen=True)
class PreparationIteration:
    guideline_matches: Sequence[GuidelineMatch]
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
"""The mode of the session, either 'auto' for automatic handling or 'manual' for manual handling by a human agent."""

LifeSpan: TypeAlias = Literal["response", "session"]
"""The lifespan of a tool result, either 'response' for just the current response or 'session' for the entire session."""


@dataclass(frozen=True)
class AgentState:
    correlation_id: str
    applied_guideline_ids: Sequence[GuidelineId]
    journey_paths: Mapping[JourneyId, Sequence[Optional[GuidelineId]]]


@dataclass(frozen=True)
class Session:
    id: SessionId
    creation_utc: datetime
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]
    agent_states: Sequence[AgentState]


class SessionUpdateParams(TypedDict, total=False):
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]
    agent_states: Sequence[AgentState]


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


class _SessionDocument_v0_4_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]


class _AgentStateDocument(TypedDict):
    correlation_id: str
    applied_guideline_ids: Sequence[GuidelineId]
    journey_paths: Mapping[JourneyId, Sequence[Optional[GuidelineId]]]


class _SessionDocument_v0_5_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]
    agent_state: _AgentStateDocument


class _SessionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]
    agent_states: Sequence[_AgentStateDocument]


class _EventDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    session_id: SessionId
    source: str
    kind: str
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


class _GuidelineMatchInspectionDocument(TypedDict):
    total_duration: float
    batches: Sequence[_GenerationInfoDocument]


class _PreparationIterationGenerationsDocument_v0_2_0(TypedDict):
    guideline_proposition: _GuidelineMatchInspectionDocument
    tool_calls: Sequence[_GenerationInfoDocument]


class _PreparationIterationGenerationsDocument(TypedDict):
    guideline_match: _GuidelineMatchInspectionDocument
    tool_calls: Sequence[_GenerationInfoDocument]


class _MessageGenerationInspectionDocument_v0_1_0(TypedDict):
    generation: _GenerationInfoDocument
    messages: Sequence[Optional[MessageEventData]]


class _MessageGenerationInspectionDocument_v0_2_0(TypedDict):
    generation: _GenerationInfoDocument
    messages: Sequence[Optional[str]]


class _MessageGenerationInspectionDocument(TypedDict):
    generations: Sequence[_GenerationInfoDocument]
    generation_names: Sequence[str]
    messages: Sequence[Optional[str]]


class _PreparationIterationDocument_v0_2_0(TypedDict):
    guideline_propositions: Sequence[GuidelineMatch]
    tool_calls: Sequence[ToolCall]
    terms: Sequence[Term]
    context_variables: Sequence[ContextVariable]
    generations: _PreparationIterationGenerationsDocument_v0_2_0


_PreparationIterationDocument_v0_1_0: TypeAlias = _PreparationIterationDocument_v0_2_0


class _PreparationIterationDocument(TypedDict):
    guideline_matches: Sequence[GuidelineMatch]
    tool_calls: Sequence[ToolCall]
    terms: Sequence[Term]
    context_variables: Sequence[ContextVariable]
    generations: _PreparationIterationGenerationsDocument


class _InspectionDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    session_id: SessionId
    correlation_id: str
    message_generations: Sequence[_MessageGenerationInspectionDocument_v0_1_0]
    preparation_iterations: Sequence[_PreparationIterationDocument_v0_1_0]


class _InspectionDocument_v0_2_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    session_id: SessionId
    correlation_id: str
    message_generations: Sequence[_MessageGenerationInspectionDocument_v0_2_0]
    preparation_iterations: Sequence[_PreparationIterationDocument_v0_2_0]


class _InspectionDocument_v0_3_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    session_id: SessionId
    correlation_id: str
    message_generations: Sequence[_MessageGenerationInspectionDocument_v0_2_0]
    preparation_iterations: Sequence[_PreparationIterationDocument]


class _InspectionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    session_id: SessionId
    correlation_id: str
    message_generations: Sequence[_MessageGenerationInspectionDocument]
    preparation_iterations: Sequence[_PreparationIterationDocument]


class _MessageEventData_v0_5_0(TypedDict):
    message: str
    participant: Participant
    flagged: NotRequired[bool]
    tags: NotRequired[Sequence[str]]
    draft: NotRequired[str]
    utterances: NotRequired[Sequence[tuple[CannedResponseId, str]]]


class _ToolResult_v0_5_0(TypedDict):
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable]
    control: ControlOptions
    utterances: Sequence[str]
    utterance_fields: Mapping[str, JSONSerializable]


class _ToolCall_v0_5_0(TypedDict):
    tool_id: str
    arguments: Mapping[str, JSONSerializable]
    result: _ToolResult_v0_5_0


class _ToolEventData_v0_5_0(TypedDict):
    tool_calls: list[_ToolCall_v0_5_0]


class SessionDocumentStore(SessionStore):
    VERSION = Version.from_string("0.6.0")

    def __init__(self, database: DocumentDatabase, allow_migration: bool = False):
        self._database = database
        self._session_collection: DocumentCollection[_SessionDocument]
        self._event_collection: DocumentCollection[_EventDocument]
        self._inspection_collection: DocumentCollection[_InspectionDocument]
        self._allow_migration = allow_migration

        self._lock = ReaderWriterLock()

    async def _session_document_loader(self, doc: BaseDocument) -> Optional[_SessionDocument]:
        async def v0_1_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_SessionDocument_v0_4_0, doc)

            return _SessionDocument_v0_4_0(
                id=doc["id"],
                version=Version.String("0.4.0"),
                creation_utc=doc["creation_utc"],
                customer_id=doc["customer_id"],
                agent_id=doc["agent_id"],
                mode=doc["mode"],
                title=doc["title"],
                consumption_offsets=doc["consumption_offsets"],
            )

        async def v0_4_0_to_v0_5_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_SessionDocument_v0_4_0, doc)

            return _SessionDocument_v0_5_0(
                id=doc["id"],
                version=Version.String("0.5.0"),
                creation_utc=doc["creation_utc"],
                customer_id=doc["customer_id"],
                agent_id=doc["agent_id"],
                mode=doc["mode"],
                title=doc["title"],
                consumption_offsets=doc["consumption_offsets"],
                agent_state=_AgentStateDocument(
                    applied_guideline_ids=[],
                    journey_paths={},
                    correlation_id="N/A",
                ),
            )

        async def v0_5_0_to_v0_6_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_SessionDocument_v0_5_0, doc)

            return _SessionDocument(
                id=doc["id"],
                version=Version.String("0.6.0"),
                creation_utc=doc["creation_utc"],
                customer_id=doc["customer_id"],
                agent_id=doc["agent_id"],
                mode=doc["mode"],
                title=doc["title"],
                consumption_offsets=doc["consumption_offsets"],
                agent_states=[],
            )

        return await DocumentMigrationHelper[_SessionDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_4_0,
                "0.2.0": v0_1_0_to_v0_4_0,
                "0.3.0": v0_1_0_to_v0_4_0,
                "0.4.0": v0_4_0_to_v0_5_0,
                "0.5.0": v0_5_0_to_v0_6_0,
            },
        ).migrate(doc)

    async def _event_document_loader(self, doc: BaseDocument) -> Optional[_EventDocument]:
        async def v0_1_0_to_v0_5_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_EventDocument, doc)

            return _EventDocument(
                id=doc["id"],
                version=Version.String("0.5.0"),
                creation_utc=doc["creation_utc"],
                session_id=doc["session_id"],
                source=doc["source"],
                kind=doc["kind"],
                offset=doc["offset"],
                correlation_id=doc["correlation_id"],
                data=doc["data"],
                deleted=doc["deleted"],
            )

        async def v0_5_0_to_v0_6_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_EventDocument, doc)

            if doc["kind"] == "message":
                doc_data = cast(_MessageEventData_v0_5_0, doc["data"])

                data = cast(
                    JSONSerializable,
                    MessageEventData(
                        message=doc_data["message"],
                        participant=doc_data["participant"],
                        flagged=doc_data.get("flagged", False),
                        tags=doc_data.get("tags", []),
                        draft=doc_data.get("draft", ""),
                        canned_responses=doc_data.get("utterances", []),
                    ),
                )

            elif doc["kind"] == "tool":
                t_data = cast(_ToolEventData_v0_5_0, doc["data"])

                data = cast(
                    JSONSerializable,
                    ToolEventData(
                        tool_calls=[
                            ToolCall(
                                tool_id=tc["tool_id"],
                                arguments=tc["arguments"],
                                result=ToolResult(
                                    data=tc["result"]["data"],
                                    metadata=tc["result"]["metadata"],
                                    control=tc["result"]["control"],
                                    canned_responses=tc["result"].get("utterances", []),
                                    canned_response_fields=tc["result"].get("utterance_fields", {}),
                                ),
                            )
                            for tc in t_data["tool_calls"]
                        ]
                    ),
                )
            else:
                data = doc["data"]

            return _EventDocument(
                id=doc["id"],
                version=Version.String("0.6.0"),
                creation_utc=doc["creation_utc"],
                session_id=doc["session_id"],
                source=doc["source"],
                kind=doc["kind"],
                offset=doc["offset"],
                correlation_id=doc["correlation_id"],
                data=data,
                deleted=doc["deleted"],
            )

        return await DocumentMigrationHelper[_EventDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_5_0,
                "0.2.0": v0_1_0_to_v0_5_0,
                "0.3.0": v0_1_0_to_v0_5_0,
                "0.4.0": v0_1_0_to_v0_5_0,
                "0.5.0": v0_5_0_to_v0_6_0,
            },
        ).migrate(doc)

    async def _inspection_document_loader(self, doc: BaseDocument) -> Optional[_InspectionDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_InspectionDocument_v0_1_0, doc)

            return _InspectionDocument_v0_2_0(
                id=doc["id"],
                version=Version.String("0.2.0"),
                session_id=doc["session_id"],
                correlation_id=doc["correlation_id"],
                message_generations=[
                    _MessageGenerationInspectionDocument_v0_2_0(
                        generation=mg["generation"],
                        messages=[
                            m if isinstance(m, str) else m["message"] if m else None
                            for m in mg["messages"]
                        ],
                    )
                    for mg in doc["message_generations"]
                ],
                preparation_iterations=doc["preparation_iterations"],
            )

        async def v0_2_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_InspectionDocument_v0_2_0, doc)
            return _InspectionDocument_v0_3_0(
                id=doc["id"],
                version=Version.String("0.3.0"),
                session_id=doc["session_id"],
                correlation_id=doc["correlation_id"],
                message_generations=[
                    _MessageGenerationInspectionDocument_v0_2_0(
                        generation=mg["generation"],
                        messages=[
                            m if isinstance(m, str) else m["message"] if m else None
                            for m in mg["messages"]
                        ],
                    )
                    for mg in doc["message_generations"]
                ],
                preparation_iterations=[
                    _PreparationIterationDocument(
                        guideline_matches=i["guideline_propositions"],
                        tool_calls=i["tool_calls"],
                        terms=i["terms"],
                        context_variables=i["context_variables"],
                        generations=_PreparationIterationGenerationsDocument(
                            guideline_match=_GuidelineMatchInspectionDocument(
                                total_duration=i["generations"]["guideline_proposition"][
                                    "total_duration"
                                ],
                                batches=[
                                    _GenerationInfoDocument(
                                        schema_name=g["schema_name"],
                                        model=g["model"],
                                        duration=g["duration"],
                                        usage=_UsageInfoDocument(
                                            input_tokens=g["usage"]["input_tokens"],
                                            output_tokens=g["usage"]["output_tokens"],
                                            extra={
                                                k: v if v else 0
                                                for k, v in g["usage"]["extra"].items()  # type: ignore  # fix bug where values were None
                                            },
                                        ),
                                    )
                                    for g in i["generations"]["guideline_proposition"]["batches"]
                                ],
                            ),
                            tool_calls=i["generations"]["tool_calls"],
                        ),
                    )
                    for i in doc["preparation_iterations"]
                ],
            )

        async def v0_3_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_InspectionDocument_v0_3_0, doc)

            return _InspectionDocument(
                id=doc["id"],
                version=Version.String("0.4.0"),
                session_id=doc["session_id"],
                correlation_id=doc["correlation_id"],
                message_generations=[
                    _MessageGenerationInspectionDocument(
                        generations=[mg["generation"]],
                        generation_names=[
                            "selection"
                            if mg["generation"]["schema_name"]
                            in ["UtteranceCompositionSchema", "UtteranceSelectionSchema"]
                            else "message_generation"
                        ],
                        messages=mg["messages"],
                    )
                    for mg in doc["message_generations"]
                ],
                preparation_iterations=doc["preparation_iterations"],
            )

        async def v0_4_0_to_v0_6_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_InspectionDocument, doc)
            return _InspectionDocument(
                id=doc["id"],
                version=Version.String("0.6.0"),
                session_id=doc["session_id"],
                correlation_id=doc["correlation_id"],
                message_generations=doc["message_generations"],
                preparation_iterations=doc["preparation_iterations"],
            )

        return await DocumentMigrationHelper[_InspectionDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
                "0.3.0": v0_3_0_to_v0_4_0,
                "0.4.0": v0_4_0_to_v0_6_0,
            },
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._session_collection = await self._database.get_or_create_collection(
                name="sessions",
                schema=_SessionDocument,
                document_loader=self._session_document_loader,
            )
            self._event_collection = await self._database.get_or_create_collection(
                name="events",
                schema=_EventDocument,
                document_loader=self._event_document_loader,
            )
            self._inspection_collection = await self._database.get_or_create_collection(
                name="inspections",
                schema=_InspectionDocument,
                document_loader=self._inspection_document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize_session_update_params(self, params: SessionUpdateParams) -> _SessionDocument:
        doc_params: _SessionDocument = {}

        if "customer_id" in params:
            doc_params["customer_id"] = params["customer_id"]
        if "agent_id" in params:
            doc_params["agent_id"] = params["agent_id"]
        if "mode" in params:
            doc_params["mode"] = params["mode"]
        if "title" in params:
            doc_params["title"] = params["title"]
        if "consumption_offsets" in params:
            doc_params["consumption_offsets"] = params["consumption_offsets"]
        if "agent_states" in params:
            doc_params["agent_states"] = [
                _AgentStateDocument(
                    correlation_id=s.correlation_id,
                    applied_guideline_ids=s.applied_guideline_ids,
                    journey_paths=s.journey_paths,
                )
                for s in params["agent_states"]
            ]

        return doc_params

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
            agent_states=[
                _AgentStateDocument(
                    correlation_id=s.correlation_id,
                    applied_guideline_ids=s.applied_guideline_ids,
                    journey_paths=s.journey_paths,
                )
                for s in session.agent_states
            ],
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
            agent_states=[
                AgentState(
                    correlation_id=s["correlation_id"],
                    applied_guideline_ids=s["applied_guideline_ids"],
                    journey_paths=s["journey_paths"],
                )
                for s in session_document["agent_states"]
            ],
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
            source=event.source.value,
            kind=event.kind.value,
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
            source=EventSource(event_document["source"]),
            kind=EventKind(event_document["kind"]),
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
                    generations=[
                        serialize_generation_info(generation_info)
                        for generation_info in m.generations.values()
                    ],
                    generation_names=list(m.generations.keys()),
                    messages=m.messages,
                )
                for m in inspection.message_generations
            ],
            preparation_iterations=[
                {
                    "guideline_matches": i.guideline_matches,
                    "tool_calls": i.tool_calls,
                    "terms": i.terms,
                    "context_variables": i.context_variables,
                    "generations": _PreparationIterationGenerationsDocument(
                        guideline_match=_GuidelineMatchInspectionDocument(
                            total_duration=i.generations.guideline_matching.total_duration,
                            batches=[
                                serialize_generation_info(g)
                                for g in i.generations.guideline_matching.batches
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
                    generations={
                        m["generation_names"][i]: deserialize_generation_info(m["generations"][i])
                        for i in range(len(m["generation_names"]))
                    },
                    messages=m["messages"],
                )
                for m in inspection_document["message_generations"]
            ],
            preparation_iterations=[
                PreparationIteration(
                    guideline_matches=i["guideline_matches"],
                    tool_calls=i["tool_calls"],
                    terms=i["terms"],
                    context_variables=i["context_variables"],
                    generations=PreparationIterationGenerations(
                        guideline_matching=GuidelineMatchingInspection(
                            total_duration=i["generations"]["guideline_match"]["total_duration"],
                            batches=[
                                deserialize_generation_info(g)
                                for g in i["generations"]["guideline_match"]["batches"]
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
                agent_states=[],
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
                params=self._serialize_session_update_params(params),
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
                params={"deleted": True},
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
                **({"source": {"$eq": source.value}} if source else {}),
                **({"offset": {"$gte": min_offset}} if min_offset else {}),
                **({"correlation_id": {"$eq": correlation_id}} if correlation_id else {}),
                **({"deleted": {"$eq": False}} if exclude_deleted else {}),
            }

            if kinds:
                event_documents = await self._event_collection.find(
                    cast(
                        Where,
                        {"$or": [{**base_filters, "kind": {"$eq": k.value}} for k in kinds]},
                    )
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
                await timeout.wait_up_to(0.25)
