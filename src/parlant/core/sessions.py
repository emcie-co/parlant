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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from dataclasses import field
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Set,
    TypeAlias,
    cast,
)
from typing_extensions import override, TypedDict, NotRequired, Self

from itertools import chain
from parlant.core import async_utils
from parlant.core.async_utils import ReaderWriterLock, Timeout, safe_gather
from parlant.core.common import (
    ItemNotFoundError,
    try_or_none,
    JSONSerializable,
    UniqueId,
    Version,
    generate_id,
)
from parlant.core.agents import AgentId
from parlant.core.context_variables import ContextVariableId
from parlant.core.customers import CustomerId
from parlant.core.store_provider import StoreProvider, StoreProviderHints
from parlant.core.rules import RuleId
from parlant.core.tools import TransientRule as TransientRule
from parlant.core import metadata as metadata_utils
from parlant.core.persistence.common import (
    ObjectId,
    Where,
)
from parlant.core.persistence.common import (
    Cursor,
    SortDirection,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    CollectionIndex,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.glossary import TermId
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)

SessionId = NewType("SessionId", str)

EventId = NewType("EventId", str)


class EventSource(Enum):
    """The source of an event in a session."""

    CUSTOMER = "user"
    """Represents an event from the customer, such as a message or action."""

    CUSTOMER_UI = "user_ui"
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
    modified_utc: datetime
    offset: int
    trace_id: str
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable]
    deleted: bool
    metadata_revision: int = 0

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


class AttachmentRef(TypedDict):
    """A reference to a file that lives in the integrator's own storage.

    Parlant never stores the bytes; it only carries this reference through the
    session so tools can act on the file (usually by forwarding its URL)."""

    url: str
    media_type: NotRequired[str]
    caption: NotRequired[str]
    filename: NotRequired[str]
    metadata: NotRequired[Mapping[str, JSONSerializable]]


class MessageEventData(TypedDict):
    """Data for a message event in a session."""

    message: str
    participant: Participant
    flagged: NotRequired[bool]
    groups: NotRequired[Sequence[str]]
    draft: NotRequired[str]
    chunks: NotRequired[list[str | None]]
    attachments: NotRequired[Sequence[AttachmentRef]]


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
    rules: NotRequired[Sequence[TransientRule]]


class ToolCall(TypedDict):
    tool_id: str
    rationale: str
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
    data: NotRequired[JSONSerializable]
    message: NotRequired[str]
    chunks: NotRequired[list[str | None]]


class RuleMatch(TypedDict):
    rule_id: RuleId
    condition: str
    action: str | None
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
    description: str | None
    key: str
    value: JSONSerializable


SessionMode: TypeAlias = Literal["auto", "manual"]
"""The mode of the session, either 'auto' for automatic handling or 'manual' for manual handling by a human agent."""

LifeSpan: TypeAlias = Literal["response", "session", "auto"]
"""The lifespan of a tool result, either 'response' for just the current response, 'session' for the entire session, or 'auto' for automatically determined based on token count."""


@dataclass(frozen=True, init=False)
class AgentState:
    trace_id: str
    applied_rule_ids: Sequence[RuleId]

    def __init__(
        self,
        trace_id: str,
        applied_rule_ids: Sequence[RuleId] | None = None,
        applied_guideline_ids: Sequence[RuleId] | None = None,
    ) -> None:
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(
            self,
            "applied_rule_ids",
            applied_rule_ids if applied_rule_ids is not None else applied_guideline_ids or [],
        )

    @property
    def applied_guideline_ids(self) -> Sequence[RuleId]:
        return self.applied_rule_ids


@dataclass(frozen=True)
class Session:
    id: SessionId
    creation_utc: datetime
    modified_utc: datetime
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: str | None
    agent_states: Sequence[AgentState]
    metadata: Mapping[str, JSONSerializable]
    labels: Set[str] = field(default_factory=set)
    metadata_revision: int = 0


class MetadataRevisionConflict(RuntimeError):
    """Signals that metadata changed after a caller loaded its base revision."""

    def __init__(self, expected_revision: int) -> None:
        super().__init__(f"Metadata revision {expected_revision} is no longer current")
        self.expected_revision = expected_revision


class SessionUpdateParams(TypedDict, total=False):
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: str | None
    agent_states: Sequence[AgentState]
    metadata: Mapping[str, JSONSerializable]


class EventUpdateParams(TypedDict, total=False):
    metadata: Mapping[str, JSONSerializable]
    data: JSONSerializable


def censored_message_notice(data: "MessageEventData") -> str | None:
    """The model-visible stand-in for a moderation-flagged message, or None
    when the message isn't flagged.

    A flagged message's original text must never reach a prompt (the engine
    would otherwise act on content moderation rejected), but the fact of the
    censorship — and its reasons — must: rules can condition on a message
    having been censored, and the agent may need to explain why."""
    if not data.get("flagged"):
        return None

    reasons = ", ".join(str(tag) for tag in data.get("groups", []))
    return (
        f"<This message was censored by content moderation (reasons: {reasons})>"
        if reasons
        else "<This message was censored by content moderation>"
    )


def delivered_message_text(
    data: "MessageEventData",
    metadata: Mapping[str, JSONSerializable] | None,
) -> str:
    """The message text a history consumer should treat as what was said.

    A moderation-flagged message is replaced wholesale by its censor notice
    (see :func:`censored_message_notice`) — the flagged text was never
    "said" as far as any prompt is concerned.

    Voice delivery may annotate a message event with the portion that was
    actually played before an interruption
    (``metadata["__parlant__"]["voice"]["spoken_text"]``,
    empty when nothing was heard). Prompt-building code must reason from the
    delivered text, never from words the customer never heard; absent the
    annotation, the message text is the delivered truth."""
    if (notice := censored_message_notice(data)) is not None:
        return notice

    voice = metadata_utils.internal_value(metadata, ("voice",), {})

    if isinstance(voice, Mapping):
        spoken = voice.get("spoken_text")
        if isinstance(spoken, str):
            return spoken

    return data["message"]


@dataclass(frozen=True)
class SessionListing:
    items: Sequence[Session]
    total_count: int
    has_more: bool
    next_cursor: Cursor | None = None

    def __iter__(self) -> Iterator[Session]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


class SessionStore(ABC):
    @abstractmethod
    async def create_session(
        self,
        customer_id: CustomerId,
        agent_id: AgentId,
        creation_utc: datetime | None = None,
        title: str | None = None,
        mode: SessionMode | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        labels: Optional[Set[str]] = None,
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
        agent_id: AgentId | None = None,
        customer_id: CustomerId | None = None,
        limit: int | None = None,
        cursor: Cursor | None = None,
        sort_direction: SortDirection | None = None,
        labels: Optional[Set[str]] = None,
        min_modified_utc: datetime | None = None,
    ) -> SessionListing: ...

    @abstractmethod
    async def set_metadata(
        self,
        session_id: SessionId,
        key: str,
        value: JSONSerializable,
    ) -> Session: ...

    @abstractmethod
    async def unset_metadata(
        self,
        session_id: SessionId,
        key: str,
    ) -> Session: ...

    async def commit_session_metadata(
        self,
        session_id: SessionId,
        expected_revision: int,
        metadata: Mapping[str, JSONSerializable],
    ) -> Session:
        """Conditionally replace metadata at one exact metadata revision.

        Compass-capable custom stores override this internal capability. It is
        deliberately non-abstract so unrelated custom stores and test doubles
        remain constructible.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support atomic session metadata commits"
        )

    async def patch_event_metadata(
        self,
        session_id: SessionId,
        event_id: EventId,
        expected_revision: int,
        set_values: Mapping[str, JSONSerializable],
        unset_keys: Sequence[str] = (),
    ) -> Event:
        """Conditionally apply top-level event metadata operations."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support atomic event metadata patches"
        )

    @abstractmethod
    async def upsert_labels(
        self,
        session_id: SessionId,
        labels: Set[str],
    ) -> Session: ...

    @abstractmethod
    async def remove_labels(
        self,
        session_id: SessionId,
        labels: Set[str],
    ) -> Session: ...

    @abstractmethod
    async def create_event(
        self,
        session_id: SessionId,
        source: EventSource,
        kind: EventKind,
        trace_id: str,
        data: JSONSerializable,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: datetime | None = None,
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
        source: EventSource | None = None,
        trace_id: str | None = None,
        kinds: Sequence[EventKind] = [],
        min_offset: int | None = None,
        exclude_deleted: bool = True,
    ) -> Sequence[Event]: ...

    @abstractmethod
    async def update_event(
        self,
        session_id: SessionId,
        event_id: EventId,
        params: EventUpdateParams,
    ) -> Event: ...


async def patch_internal_event_metadata(
    store: SessionStore,
    session_id: SessionId,
    event_id: EventId,
    path: Sequence[str],
    value: JSONSerializable,
    *,
    max_attempts: int = 8,
) -> Event:
    """Atomically set one Parlant-owned event metadata path.

    Each retry rebuilds the reserved root from the latest event revision, so a
    concurrent public annotation or independent internal voice update survives.
    """

    for _ in range(max_attempts):
        event = await store.read_event(session_id, event_id)
        updated = metadata_utils.set_internal_value(event.metadata, path, value)
        try:
            return await store.patch_event_metadata(
                session_id,
                event_id,
                event.metadata_revision,
                {
                    metadata_utils.PARLANT_METADATA_KEY: updated[
                        metadata_utils.PARLANT_METADATA_KEY
                    ],
                },
            )
        except MetadataRevisionConflict:
            continue

    raise MetadataRevisionConflict(event.metadata_revision)


async def patch_internal_session_metadata(
    store: SessionStore,
    session_id: SessionId,
    path: Sequence[str],
    value: JSONSerializable,
    *,
    max_attempts: int = 8,
) -> Session:
    """Atomically set one Parlant-owned session metadata path."""

    for _ in range(max_attempts):
        session = await store.read_session(session_id)
        updated = metadata_utils.set_internal_value(session.metadata, path, value)
        try:
            return await store.commit_session_metadata(
                session_id,
                session.metadata_revision,
                updated,
            )
        except MetadataRevisionConflict:
            continue

    raise MetadataRevisionConflict(session.metadata_revision)


async def patch_public_session_metadata(
    store: SessionStore,
    session_id: SessionId,
    set_values: Mapping[str, JSONSerializable],
    unset_keys: Sequence[str] = (),
    *,
    max_attempts: int = 8,
) -> Session:
    """Atomically apply one public session-metadata update request.

    The complete requested mutation is replayed after a revision conflict, so
    an API request that sets or removes several keys still performs one
    conditional write on the uncontended path and preserves concurrent
    Parlant-owned changes.
    """

    metadata_utils.ensure_public_metadata(set_values)
    for key in unset_keys:
        metadata_utils.ensure_public_metadata_key(key)

    for _ in range(max_attempts):
        session = await store.read_session(session_id)
        updated = dict(session.metadata)
        updated.update(set_values)
        for key in unset_keys:
            updated.pop(key, None)

        if updated == dict(session.metadata):
            return session

        try:
            return await store.commit_session_metadata(
                session_id,
                session.metadata_revision,
                updated,
            )
        except MetadataRevisionConflict:
            continue

    raise MetadataRevisionConflict(session.metadata_revision)


class _AgentStateDocument(TypedDict):
    trace_id: str
    applied_rule_ids: Sequence[RuleId]


class _SessionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    modified_utc: str
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: str | None
    agent_states: Sequence[_AgentStateDocument]
    metadata: Mapping[str, JSONSerializable]
    metadata_revision: int
    labels: Sequence[str]


class _EventDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    modified_utc: str
    session_id: SessionId
    source: str
    kind: str
    offset: int
    trace_id: str
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable] | None
    metadata_revision: int
    deleted: bool


class _UsageInfoDocument(TypedDict):
    input_tokens: int
    output_tokens: int
    extra: Mapping[str, int | float | str] | None


class _GenerationInfoDocument(TypedDict):
    schema_name: str
    model: str
    duration: float
    usage: _UsageInfoDocument


class _RuleMatchInspectionDocument(TypedDict):
    total_duration: float
    batches: Sequence[_GenerationInfoDocument]


class _PreparationIterationGenerationsDocument(TypedDict):
    rule_match: _RuleMatchInspectionDocument
    tool_calls: Sequence[_GenerationInfoDocument]


class _MessageGenerationInspectionDocument(TypedDict):
    generations: Sequence[_GenerationInfoDocument]
    generation_names: Sequence[str]
    messages: Sequence[str | None]


class _PreparationIterationDocument(TypedDict):
    rule_matches: Sequence[RuleMatch]
    tool_calls: Sequence[ToolCall]
    terms: Sequence[Term]
    context_variables: Sequence[ContextVariable]
    generations: _PreparationIterationGenerationsDocument


class _InspectionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    session_id: SessionId
    trace_id: str
    message_generations: Sequence[_MessageGenerationInspectionDocument]
    preparation_iterations: Sequence[_PreparationIterationDocument]


class SessionDocumentStore(SessionStore):
    VERSION = Version.from_string("0.11.0")

    def __init__(
        self,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ):
        self._database = database
        self._session_collection: DocumentCollection[_SessionDocument]
        self._event_collection: DocumentCollection[_EventDocument]
        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix

        self._lock = ReaderWriterLock()

    async def _session_document_loader(self, doc: BaseDocument) -> _SessionDocument | None:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[_SessionDocument](
            self,
            {},
        ).migrate(doc)

    async def _event_document_loader(self, doc: BaseDocument) -> _EventDocument | None:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[_EventDocument](
            self,
            {},
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._session_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_sessions"
                if self._collections_prefix
                else "sessions",
                schema=_SessionDocument,
                document_loader=self._session_document_loader,
            )
            self._event_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_events" if self._collections_prefix else "events",
                schema=_EventDocument,
                document_loader=self._event_document_loader,
            )
            await self._session_collection.ensure_indexes(
                [
                    CollectionIndex(fields=(("id", SortDirection.ASC),)),
                    CollectionIndex(
                        fields=(
                            ("creation_utc", SortDirection.ASC),
                            ("id", SortDirection.ASC),
                        )
                    ),
                    CollectionIndex(
                        fields=(
                            ("modified_utc", SortDirection.ASC),
                            ("id", SortDirection.ASC),
                        )
                    ),
                    CollectionIndex(
                        fields=(
                            ("agent_id", SortDirection.ASC),
                            ("creation_utc", SortDirection.ASC),
                            ("id", SortDirection.ASC),
                        )
                    ),
                    CollectionIndex(
                        fields=(
                            ("customer_id", SortDirection.ASC),
                            ("creation_utc", SortDirection.ASC),
                            ("id", SortDirection.ASC),
                        )
                    ),
                ]
            )
            await self._event_collection.ensure_indexes(
                [
                    CollectionIndex(fields=(("id", SortDirection.ASC),)),
                    CollectionIndex(
                        fields=(
                            ("session_id", SortDirection.ASC),
                            ("offset", SortDirection.ASC),
                        )
                    ),
                    CollectionIndex(
                        fields=(
                            ("modified_utc", SortDirection.ASC),
                            ("session_id", SortDirection.ASC),
                        )
                    ),
                    CollectionIndex(
                        fields=(
                            ("session_id", SortDirection.ASC),
                            ("deleted", SortDirection.ASC),
                            ("offset", SortDirection.ASC),
                        )
                    ),
                ]
            )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
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
        if "agent_states" in params:
            doc_params["agent_states"] = [
                _AgentStateDocument(
                    trace_id=s.trace_id,
                    applied_rule_ids=s.applied_rule_ids,
                )
                for s in params["agent_states"]
            ]
        if "metadata" in params:
            doc_params["metadata"] = params["metadata"]

        return doc_params

    def _serialize_session(
        self,
        session: Session,
    ) -> _SessionDocument:
        return _SessionDocument(
            id=ObjectId(session.id),
            version=self.VERSION.to_string(),
            creation_utc=session.creation_utc.isoformat(),
            modified_utc=session.modified_utc.isoformat(),
            customer_id=session.customer_id,
            agent_id=session.agent_id,
            mode=session.mode,
            title=session.title if session.title else None,
            agent_states=[
                _AgentStateDocument(
                    trace_id=s.trace_id,
                    applied_rule_ids=s.applied_rule_ids,
                )
                for s in session.agent_states
            ],
            metadata=session.metadata,
            metadata_revision=session.metadata_revision,
            labels=list(session.labels),
        )

    def _deserialize_session(
        self,
        session_document: _SessionDocument,
    ) -> Session:
        return Session(
            id=SessionId(session_document["id"]),
            creation_utc=datetime.fromisoformat(session_document["creation_utc"]),
            modified_utc=datetime.fromisoformat(session_document["modified_utc"]),
            customer_id=session_document["customer_id"],
            agent_id=session_document["agent_id"],
            mode=session_document["mode"],
            title=session_document["title"],
            agent_states=[
                AgentState(
                    trace_id=s["trace_id"],
                    applied_rule_ids=s["applied_rule_ids"],
                )
                for s in session_document["agent_states"]
            ],
            metadata=session_document.get("metadata", {}),
            labels=set(session_document.get("labels", [])),
            metadata_revision=session_document.get("metadata_revision", 0),
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
            modified_utc=event.modified_utc.isoformat(),
            session_id=session_id,
            source=event.source.value,
            kind=event.kind.value,
            offset=event.offset,
            trace_id=event.trace_id,
            data=event.data,
            metadata=event.metadata if event.metadata else None,
            metadata_revision=event.metadata_revision,
            deleted=event.deleted,
        )

    def _deserialize_event(
        self,
        event_document: _EventDocument,
    ) -> Event:
        return Event(
            id=EventId(event_document["id"]),
            creation_utc=datetime.fromisoformat(event_document["creation_utc"]),
            modified_utc=datetime.fromisoformat(event_document["modified_utc"]),
            source=EventSource(event_document["source"]),
            kind=EventKind(event_document["kind"]),
            offset=event_document["offset"],
            trace_id=event_document["trace_id"],
            data=event_document["data"],
            metadata=cast(Mapping[str, JSONSerializable], event_document["metadata"] or {}),
            deleted=event_document["deleted"],
            metadata_revision=event_document.get("metadata_revision", 0),
        )

    @override
    async def create_session(
        self,
        customer_id: CustomerId,
        agent_id: AgentId,
        creation_utc: datetime | None = None,
        title: str | None = None,
        mode: SessionMode | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        labels: Optional[Set[str]] = None,
    ) -> Session:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            session = Session(
                id=SessionId(generate_id()),
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                customer_id=customer_id,
                agent_id=agent_id,
                mode=mode or "auto",
                title=title,
                agent_states=[],
                metadata=metadata,
                labels=labels or set(),
                metadata_revision=0,
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

            serialized_params = self._serialize_session_update_params(params)
            filters: dict[str, Any] = {"id": {"$eq": session_id}}
            if "metadata" in params:
                expected_revision = session_document.get("metadata_revision", 0)
                filters["metadata_revision"] = {"$eq": expected_revision}
                serialized_params["metadata_revision"] = expected_revision + 1

            result = await self._session_collection.update_one(
                filters=cast(Where, filters),
                params={
                    **serialized_params,
                    "modified_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

            if result.matched_count == 0:
                raise MetadataRevisionConflict(session_document.get("metadata_revision", 0))

            updated_document = result.updated_document or await self._session_collection.find_one(
                {"id": {"$eq": session_id}}
            )

        assert updated_document
        return self._deserialize_session(session_document=updated_document)

    async def _commit_session_metadata_unlocked(
        self,
        session_id: SessionId,
        expected_revision: int,
        metadata: Mapping[str, JSONSerializable],
    ) -> Session:
        """Apply a conditional metadata replacement while the store writer lock is held."""

        result = await self._session_collection.update_one(
            filters={
                "id": {"$eq": session_id},
                "metadata_revision": {"$eq": expected_revision},
            },
            params={
                "metadata": metadata,
                "metadata_revision": expected_revision + 1,
                "modified_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        if result.matched_count == 0:
            if not await self._session_collection.find_one({"id": {"$eq": session_id}}):
                raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")
            raise MetadataRevisionConflict(expected_revision)

        updated_document = result.updated_document or await self._session_collection.find_one(
            {"id": {"$eq": session_id}}
        )
        assert updated_document
        return self._deserialize_session(updated_document)

    @override
    async def commit_session_metadata(
        self,
        session_id: SessionId,
        expected_revision: int,
        metadata: Mapping[str, JSONSerializable],
    ) -> Session:
        async with self._lock.writer_lock:
            return await self._commit_session_metadata_unlocked(
                session_id,
                expected_revision,
                metadata,
            )

    @override
    async def list_sessions(
        self,
        agent_id: AgentId | None = None,
        customer_id: CustomerId | None = None,
        limit: int | None = None,
        cursor: Cursor | None = None,
        sort_direction: SortDirection | None = None,
        labels: Optional[Set[str]] = None,
        min_modified_utc: datetime | None = None,
    ) -> SessionListing:
        async with self._lock.reader_lock:
            filters = {
                **({"agent_id": {"$eq": agent_id}} if agent_id else {}),
                **({"customer_id": {"$eq": customer_id}} if customer_id else {}),
            }

            if min_modified_utc:
                min_modified_utc_str = min_modified_utc.isoformat()

                event_result = await self._event_collection.find(
                    filters={"modified_utc": {"$gte": min_modified_utc_str}},
                )

                event_session_ids = {d["session_id"] for d in event_result.items}
                latest_event_modified_utc_by_session_id: dict[SessionId, datetime] = {}

                for event_document in event_result.items:
                    event_modified_utc = datetime.fromisoformat(event_document["modified_utc"])
                    session_id = event_document["session_id"]
                    latest_event_modified_utc_by_session_id[session_id] = max(
                        latest_event_modified_utc_by_session_id.get(session_id, event_modified_utc),
                        event_modified_utc,
                    )

                modified_filters: list[Where] = [
                    {"modified_utc": {"$gte": min_modified_utc_str}},
                ]
                if event_session_ids:
                    modified_filters.append(
                        {
                            "id": {
                                "$in": [ObjectId(session_id) for session_id in event_session_ids],
                            }
                        }
                    )

                session_filters: Where = (
                    cast(Where, {"$and": [filters, {"$or": modified_filters}]})
                    if filters
                    else cast(Where, {"$or": modified_filters})
                )

                result = await self._session_collection.find(
                    filters=session_filters,
                    sort_direction=sort_direction,
                )

                matching_session_documents = list(result.items)

                if labels:
                    matching_session_documents = [
                        d
                        for d in matching_session_documents
                        if labels.issubset(set(d.get("labels", [])))
                    ]

                sessions = []
                for session_document in matching_session_documents:
                    session = self._deserialize_session(session_document)
                    latest_event_modified_utc = latest_event_modified_utc_by_session_id.get(
                        session.id,
                    )
                    if latest_event_modified_utc:
                        session = replace(
                            session,
                            modified_utc=max(session.modified_utc, latest_event_modified_utc),
                        )
                    sessions.append(session)

                sessions.sort(
                    key=lambda s: (s.modified_utc.isoformat(), str(s.id)),
                    reverse=sort_direction == SortDirection.DESC,
                )

                total_count = len(sessions)

                if cursor:
                    if sort_direction == SortDirection.DESC:
                        sessions = [
                            s
                            for s in sessions
                            if (s.modified_utc.isoformat(), str(s.id))
                            < (cursor.sort_value, str(cursor.id))
                        ]
                    else:
                        sessions = [
                            s
                            for s in sessions
                            if (s.modified_utc.isoformat(), str(s.id))
                            > (cursor.sort_value, str(cursor.id))
                        ]

                paginated_sessions = sessions[:limit] if limit else sessions
                has_more = bool(limit and len(sessions) > limit)
                next_cursor = (
                    Cursor(
                        sort_value=paginated_sessions[-1].modified_utc.isoformat(),
                        id=ObjectId(paginated_sessions[-1].id),
                    )
                    if has_more and paginated_sessions
                    else None
                )

                return SessionListing(
                    items=paginated_sessions,
                    total_count=total_count,
                    has_more=has_more,
                    next_cursor=next_cursor,
                )

            result = await self._session_collection.find(
                filters=cast(Where, filters),
                limit=limit,
                cursor=cursor,
                sort_direction=sort_direction,
            )

            # Filter by labels if specified
            if labels:
                items = [
                    self._deserialize_session(d)
                    for d in result.items
                    if labels.issubset(set(d.get("labels", [])))
                ]
            else:
                items = [self._deserialize_session(d) for d in result.items]

            return SessionListing(
                items=items,
                total_count=len(items) if labels else result.total_count,
                has_more=result.has_more if not labels else False,
                next_cursor=result.next_cursor if not labels else None,
            )

    @override
    async def set_metadata(
        self,
        session_id: SessionId,
        key: str,
        value: JSONSerializable,
    ) -> Session:
        metadata_utils.ensure_public_metadata_key(key)
        async with self._lock.writer_lock:
            for _ in range(8):
                session_document = await self._session_collection.find_one(
                    {"id": {"$eq": session_id}}
                )
                if not session_document:
                    raise ItemNotFoundError(item_id=UniqueId(session_id))

                revision = session_document.get("metadata_revision", 0)
                updated_metadata = {**session_document.get("metadata", {}), key: value}
                try:
                    return await self._commit_session_metadata_unlocked(
                        session_id,
                        revision,
                        updated_metadata,
                    )
                except MetadataRevisionConflict:
                    continue

        raise MetadataRevisionConflict(revision)

    @override
    async def unset_metadata(
        self,
        session_id: SessionId,
        key: str,
    ) -> Session:
        metadata_utils.ensure_public_metadata_key(key)
        async with self._lock.writer_lock:
            for _ in range(8):
                session_document = await self._session_collection.find_one(
                    {"id": {"$eq": session_id}}
                )
                if not session_document:
                    raise ItemNotFoundError(item_id=UniqueId(session_id))

                revision = session_document.get("metadata_revision", 0)
                updated_metadata = {
                    k: v for k, v in session_document.get("metadata", {}).items() if k != key
                }
                try:
                    return await self._commit_session_metadata_unlocked(
                        session_id,
                        revision,
                        updated_metadata,
                    )
                except MetadataRevisionConflict:
                    continue

        raise MetadataRevisionConflict(revision)

    @override
    async def upsert_labels(
        self,
        session_id: SessionId,
        labels: Set[str],
    ) -> Session:
        async with self._lock.writer_lock:
            session_document = await self._session_collection.find_one({"id": {"$eq": session_id}})

            if not session_document:
                raise ItemNotFoundError(item_id=UniqueId(session_id))

            existing_labels = set(session_document.get("labels", []))
            updated_labels = list(existing_labels | labels)

            result = await self._session_collection.update_one(
                filters={"id": {"$eq": session_id}},
                params={
                    "labels": updated_labels,
                    "modified_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return self._deserialize_session(session_document=result.updated_document)

    @override
    async def remove_labels(
        self,
        session_id: SessionId,
        labels: Set[str],
    ) -> Session:
        async with self._lock.writer_lock:
            session_document = await self._session_collection.find_one({"id": {"$eq": session_id}})

            if not session_document:
                raise ItemNotFoundError(item_id=UniqueId(session_id))

            existing_labels = set(session_document.get("labels", []))
            updated_labels = list(existing_labels - labels)

            result = await self._session_collection.update_one(
                filters={"id": {"$eq": session_id}},
                params={
                    "labels": updated_labels,
                    "modified_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return self._deserialize_session(session_document=result.updated_document)

    @override
    async def create_event(
        self,
        session_id: SessionId,
        source: EventSource,
        kind: EventKind,
        trace_id: str,
        data: JSONSerializable,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: datetime | None = None,
    ) -> Event:
        async with self._lock.writer_lock:
            if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
                raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

            creation_utc = creation_utc or datetime.now(timezone.utc)
            latest_event = await self._event_collection.find_one(
                filters={"session_id": {"$eq": session_id}},
                sort=(("offset", SortDirection.DESC),),
            )
            offset = latest_event["offset"] + 1 if latest_event else 0

            event = Event(
                id=EventId(generate_id()),
                source=source,
                kind=kind,
                offset=offset,
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                trace_id=trace_id,
                data=data,
                metadata=metadata,
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

            # Scope the lookup to the session: an event id alone must not reach
            # an event belonging to a different session (cross-session read).
            if event_document := await self._event_collection.find_one(
                filters={
                    "id": {"$eq": event_id},
                    "session_id": {"$eq": session_id},
                }
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
                params={
                    "deleted": True,
                    "modified_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        if result.matched_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")

    @override
    async def list_events(
        self,
        session_id: SessionId,
        source: EventSource | None = None,
        trace_id: str | None = None,
        kinds: Sequence[EventKind] = [],
        min_offset: int | None = None,
        exclude_deleted: bool = True,
    ) -> Sequence[Event]:
        async with self._lock.reader_lock:
            if not await self._session_collection.find_one(filters={"id": {"$eq": session_id}}):
                raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

            base_filters = {
                "session_id": {"$eq": session_id},
                **({"source": {"$eq": source.value}} if source else {}),
                **({"offset": {"$gte": min_offset}} if min_offset else {}),
                **({"trace_id": {"$eq": trace_id}} if trace_id else {}),
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
    async def update_event(
        self,
        session_id: SessionId,
        event_id: EventId,
        params: EventUpdateParams,
    ) -> Event:
        async with self._lock.writer_lock:
            event_document = await self._event_collection.find_one(
                filters={
                    "id": {"$eq": ObjectId(event_id)},
                    "session_id": {"$eq": session_id},
                    "deleted": {"$ne": True},
                }
            )

            if not event_document:
                raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")

            update_params: _EventDocument = {}
            if "metadata" in params:
                update_params["metadata"] = params["metadata"] if params["metadata"] else None
                update_params["metadata_revision"] = event_document.get("metadata_revision", 0) + 1
            if "data" in params:
                update_params["data"] = params["data"]

            if not update_params:
                return self._deserialize_event(event_document)

            update_params["modified_utc"] = datetime.now(timezone.utc).isoformat()

            event_filters: dict[str, Any] = {
                "id": {"$eq": ObjectId(event_id)},
                "session_id": {"$eq": session_id},
            }
            if "metadata" in params:
                event_filters["metadata_revision"] = {
                    "$eq": event_document.get("metadata_revision", 0)
                }

            result = await self._event_collection.update_one(
                filters=cast(Where, event_filters),
                params=update_params,
            )

        if result.matched_count == 0:
            raise MetadataRevisionConflict(event_document.get("metadata_revision", 0))

        updated_document = result.updated_document or await self._event_collection.find_one(
            {
                "id": {"$eq": ObjectId(event_id)},
                "session_id": {"$eq": session_id},
            }
        )
        assert updated_document

        return self._deserialize_event(updated_document)

    @override
    async def patch_event_metadata(
        self,
        session_id: SessionId,
        event_id: EventId,
        expected_revision: int,
        set_values: Mapping[str, JSONSerializable],
        unset_keys: Sequence[str] = (),
    ) -> Event:
        async with self._lock.writer_lock:
            event_document = await self._event_collection.find_one(
                filters={
                    "id": {"$eq": ObjectId(event_id)},
                    "session_id": {"$eq": session_id},
                    "deleted": {"$ne": True},
                }
            )
            if not event_document:
                raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")
            if event_document.get("metadata_revision", 0) != expected_revision:
                raise MetadataRevisionConflict(expected_revision)

            metadata = dict(event_document.get("metadata") or {})
            metadata.update(set_values)
            for key in unset_keys:
                metadata.pop(key, None)

            result = await self._event_collection.update_one(
                filters={
                    "id": {"$eq": ObjectId(event_id)},
                    "session_id": {"$eq": session_id},
                    "metadata_revision": {"$eq": expected_revision},
                },
                params={
                    "metadata": metadata or None,
                    "metadata_revision": expected_revision + 1,
                    "modified_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if result.matched_count == 0:
                raise MetadataRevisionConflict(expected_revision)

            updated_document = result.updated_document or await self._event_collection.find_one(
                {
                    "id": {"$eq": ObjectId(event_id)},
                    "session_id": {"$eq": session_id},
                }
            )

        assert updated_document
        return self._deserialize_event(updated_document)


class SessionListener(ABC):
    @abstractmethod
    async def wait_for_more_events(
        self,
        session_id: SessionId,
        kinds: Sequence[EventKind] = [],
        min_offset: int | None = None,
        source: EventSource | None = None,
        trace_id: str | None = None,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        """Wait for new events to arrive in the session.

        Returns True if new events arrived, False if timeout expired.
        """
        ...

    @abstractmethod
    async def wait_for_event_completion(
        self,
        session_id: SessionId,
        event_id: EventId,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        """Wait for a streaming event to complete (chunks ends with None).

        Returns True if the event completed, False if timeout expired.
        For non-streaming events (no chunks property), returns True immediately.
        """
        ...

    @abstractmethod
    async def wait_for_new_streaming_chunks(
        self,
        session_id: SessionId,
        event_id: EventId,
        last_known_chunk_count: int,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        """Wait for new streaming chunks to arrive or for the event to complete.

        Returns True when len(chunks) > last_known_chunk_count or event is complete.
        Returns False on timeout.
        """
        ...


class PollingSessionListener(SessionListener):
    def __init__(self, store_provider_factory: Callable[[], StoreProvider]) -> None:
        self._store_provider_factory = store_provider_factory

    @property
    def _session_store(self) -> SessionStore:
        return self._store_provider_factory().get_store(
            SessionStore, StoreProviderHints(call_site="engine")
        )

    @override
    async def wait_for_more_events(
        self,
        session_id: SessionId,
        kinds: Sequence[EventKind] = [],
        min_offset: int | None = None,
        source: EventSource | None = None,
        trace_id: str | None = None,
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
                trace_id=trace_id,
            )

            if events:
                return True
            elif timeout.expired():
                return False
            else:
                await timeout.wait_up_to(0.25)

    @override
    async def wait_for_event_completion(
        self,
        session_id: SessionId,
        event_id: EventId,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        # Trigger exception if not found
        _ = await self._session_store.read_session(session_id)

        while True:
            event = await self._session_store.read_event(session_id, event_id)

            # Check if the event has chunks property
            data = cast(dict[str, object], event.data)
            if "chunks" in data:
                chunks = cast(list[str | None], data["chunks"])
                # Check if the last chunk is None (completion signal)
                if chunks and chunks[-1] is None:
                    return True
            else:
                # Non-streaming event, return immediately
                return True

            if timeout.expired():
                return False
            else:
                await timeout.wait_up_to(0.1)

    @override
    async def wait_for_new_streaming_chunks(
        self,
        session_id: SessionId,
        event_id: EventId,
        last_known_chunk_count: int,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        # Trigger exception if not found
        _ = await self._session_store.read_session(session_id)

        while True:
            event = await self._session_store.read_event(session_id, event_id)

            data = cast(dict[str, object], event.data)
            if "chunks" in data:
                chunks = cast(list[str | None], data["chunks"])
                if len(chunks) > last_known_chunk_count:
                    return True
                if chunks and chunks[-1] is None:
                    return True
            else:
                return True

            if timeout.expired():
                return False
            else:
                await timeout.wait_up_to(0.1)


class CompositeSessionStore(SessionStore):
    def __init__(
        self,
        writable_store: SessionStore,
        readable_stores: Sequence[SessionStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[SessionStore] = [writable_store, *readable_stores]

    @override
    async def create_session(
        self,
        customer_id: CustomerId,
        agent_id: AgentId,
        creation_utc: datetime | None = None,
        title: str | None = None,
        mode: SessionMode | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        labels: Optional[Set[str]] = None,
    ) -> Session:
        return await self._writable_store.create_session(
            customer_id=customer_id,
            agent_id=agent_id,
            creation_utc=creation_utc,
            title=title,
            mode=mode,
            metadata=metadata,
            labels=labels,
        )

    @override
    async def read_session(
        self,
        session_id: SessionId,
    ) -> Session:
        results = await safe_gather(
            *[try_or_none(store.read_session(session_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(session_id))
        return result

    @override
    async def delete_session(
        self,
        session_id: SessionId,
    ) -> None:
        return await self._writable_store.delete_session(session_id)

    @override
    async def update_session(
        self,
        session_id: SessionId,
        params: SessionUpdateParams,
    ) -> Session:
        return await self._writable_store.update_session(session_id, params)

    @override
    async def list_sessions(
        self,
        agent_id: AgentId | None = None,
        customer_id: CustomerId | None = None,
        limit: int | None = None,
        cursor: Cursor | None = None,
        sort_direction: SortDirection | None = None,
        labels: Optional[Set[str]] = None,
        min_modified_utc: datetime | None = None,
    ) -> SessionListing:
        results = await safe_gather(
            *[
                store.list_sessions(
                    agent_id=agent_id,
                    customer_id=customer_id,
                    limit=limit,
                    cursor=cursor,
                    sort_direction=sort_direction,
                    labels=labels,
                    min_modified_utc=min_modified_utc,
                )
                for store in self._all_stores
            ]
        )
        all_items = list(chain.from_iterable(r.items for r in results))
        return SessionListing(
            items=all_items,
            total_count=sum(r.total_count for r in results),
            has_more=any(r.has_more for r in results),
        )

    @override
    async def set_metadata(
        self,
        session_id: SessionId,
        key: str,
        value: JSONSerializable,
    ) -> Session:
        return await self._writable_store.set_metadata(session_id, key, value)

    @override
    async def unset_metadata(
        self,
        session_id: SessionId,
        key: str,
    ) -> Session:
        return await self._writable_store.unset_metadata(session_id, key)

    @override
    async def commit_session_metadata(
        self,
        session_id: SessionId,
        expected_revision: int,
        metadata: Mapping[str, JSONSerializable],
    ) -> Session:
        return await self._writable_store.commit_session_metadata(
            session_id,
            expected_revision,
            metadata,
        )

    @override
    async def patch_event_metadata(
        self,
        session_id: SessionId,
        event_id: EventId,
        expected_revision: int,
        set_values: Mapping[str, JSONSerializable],
        unset_keys: Sequence[str] = (),
    ) -> Event:
        return await self._writable_store.patch_event_metadata(
            session_id,
            event_id,
            expected_revision,
            set_values,
            unset_keys,
        )

    @override
    async def upsert_labels(
        self,
        session_id: SessionId,
        labels: Set[str],
    ) -> Session:
        return await self._writable_store.upsert_labels(session_id, labels)

    @override
    async def remove_labels(
        self,
        session_id: SessionId,
        labels: Set[str],
    ) -> Session:
        return await self._writable_store.remove_labels(session_id, labels)

    @override
    async def create_event(
        self,
        session_id: SessionId,
        source: EventSource,
        kind: EventKind,
        trace_id: str,
        data: JSONSerializable,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: datetime | None = None,
    ) -> Event:
        try:
            return await self._writable_store.create_event(
                session_id=session_id,
                source=source,
                kind=kind,
                trace_id=trace_id,
                data=data,
                metadata=metadata,
                creation_utc=creation_utc,
            )
        except Exception as e:
            raise e

    @override
    async def read_event(
        self,
        session_id: SessionId,
        event_id: EventId,
    ) -> Event:
        results = await safe_gather(
            *[try_or_none(store.read_event(session_id, event_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(event_id))
        return result

    @override
    async def delete_event(
        self,
        event_id: EventId,
    ) -> None:
        return await self._writable_store.delete_event(event_id)

    @override
    async def list_events(
        self,
        session_id: SessionId,
        source: EventSource | None = None,
        trace_id: str | None = None,
        kinds: Sequence[EventKind] = [],
        min_offset: int | None = None,
        exclude_deleted: bool = True,
    ) -> Sequence[Event]:
        for store in self._all_stores:
            result = await try_or_none(
                store.list_events(
                    session_id=session_id,
                    source=source,
                    trace_id=trace_id,
                    kinds=kinds,
                    min_offset=min_offset,
                    exclude_deleted=exclude_deleted,
                )
            )

            if result is not None:
                return result

        raise ItemNotFoundError(item_id=UniqueId(session_id))

    @override
    async def update_event(
        self,
        session_id: SessionId,
        event_id: EventId,
        params: EventUpdateParams,
    ) -> Event:
        return await self._writable_store.update_event(session_id, event_id, params)
