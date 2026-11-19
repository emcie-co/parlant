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

import asyncio
import os

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
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
from parlant.core.loggers import Logger, ScopedLogger
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
    WhereExpression,
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
    FindResult,
    identity_loader_for,
)
from parlant.core.glossary import TermId
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)

SessionLifecycle = Literal["provisional", "established"]
ObservabilityKind = Literal["usage", "session_deleted"]


def _boolean_setting(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false; restart the server after changing it")
    return normalized == "true"


@dataclass(frozen=True)
class SessionCleanupSettings:
    enabled: bool
    include_greetings: bool

    @classmethod
    def from_environment(cls) -> "SessionCleanupSettings":
        return cls(
            enabled=_boolean_setting("PARLANT_EMPTY_SESSION_CLEANUP_ENABLED", True),
            include_greetings=_boolean_setting(
                "PARLANT_EMPTY_SESSION_CLEANUP_INCLUDE_GREETINGS", False
            ),
        )


@dataclass(frozen=True)
class SessionObservabilityRecord:
    id: str
    revision: int
    kind: ObservabilityKind
    session_id: str
    agent_id: str
    trace_id: str | None
    occurred_at: datetime
    session_creation_utc: datetime
    is_simulation: bool
    customer_represented_by_an_agent: bool
    payload: Mapping[str, JSONSerializable]


@dataclass(frozen=True)
class SessionObservabilityPage:
    records: Sequence[SessionObservabilityRecord]
    has_more: bool = False
    next_cursor: Cursor | None = None


class SessionObservabilityDocument(BaseDocument, total=False):
    revision: int
    kind: ObservabilityKind
    session_id: str
    agent_id: str
    trace_id: str | None
    occurred_at: str
    session_creation_utc: str
    is_simulation: bool
    customer_represented_by_an_agent: bool
    payload: Mapping[str, JSONSerializable]
    fingerprint: str
    pending: bool


def observability_limit(value: object = 100) -> int:
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    return value


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
    costs: NotRequired[Mapping[str, str]]


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


# Custom event types only Parlant itself emits: the sign-in request a tool
# raises, the sign-in that completes it, and a signed-in customer allowing
# more. Nothing else may emit them, so a customer's app can trust them and a
# client can't fake one.
RESERVED_CUSTOM_EVENT_TYPES = frozenset(
    {"authorization_required", "customer.signed_in", "customer.authorized"}
)


def ensure_custom_event_type_is_not_reserved(data: JSONSerializable) -> None:
    if isinstance(data, Mapping) and data.get("type") in RESERVED_CUSTOM_EVENT_TYPES:
        raise ValueError(
            f"Custom event type '{data['type']}' is reserved for events Parlant emits itself"
        )


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
    customer_represented_by_an_agent: bool = False
    """Whether an AI agent speaks for the customer in this session, such as
    their personal assistant. Fixed when the session is created."""
    lifecycle: SessionLifecycle | None = None


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
    async def list_observability_records(
        self, *, limit: int = 100, cursor: Cursor | None = None
    ) -> SessionObservabilityPage:
        """Optional durable runtime feed. Unsupported stores expose no records."""
        return SessionObservabilityPage(records=[])

    async def acknowledge_observability_records(
        self, acknowledgements: Sequence[tuple[str, int]]
    ) -> int:
        """Acknowledge exact revisions only after downstream durable persistence."""
        return 0

    async def list_empty_session_candidates(
        self,
        *,
        created_before: datetime,
        limit: int,
        cursor: Cursor | None = None,
    ) -> SessionListing:
        """Optional runtime cleanup capability; unsupported stores retain sessions."""
        return SessionListing(items=[], total_count=0, has_more=False)

    async def delete_session_if_empty(
        self,
        session_id: SessionId,
        *,
        created_before: datetime,
        include_greetings: bool,
    ) -> bool:
        """Recheck and delete under the same guard as event writes, or safely decline."""
        return False

    async def pin_session(self, session_id: SessionId) -> Callable[[], None]:
        """Protect accepted live work until the returned idempotent release is called."""
        return lambda: None

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
        customer_represented_by_an_agent: bool = False,
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
    customer_represented_by_an_agent: bool
    cleanup_established: bool
    cleanup_has_greeting: bool
    observability_dirty: bool


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
        self._observability_collection: DocumentCollection[SessionObservabilityDocument]
        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix

        self._lock = ReaderWriterLock()
        self._cleanup_pins: dict[SessionId, int] = {}
        self._cleanup_settings = SessionCleanupSettings.from_environment()

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
            self._observability_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_session_observability"
                if self._collections_prefix
                else "session_observability",
                schema=SessionObservabilityDocument,
                document_loader=identity_loader_for(SessionObservabilityDocument),
            )
            await self._observability_collection.ensure_indexes(
                [
                    CollectionIndex(fields=(("id", SortDirection.ASC),), unique=True),
                    CollectionIndex(
                        fields=(
                            ("pending", SortDirection.ASC),
                            ("creation_utc", SortDirection.ASC),
                            ("id", SortDirection.ASC),
                        )
                    ),
                    CollectionIndex(fields=(("session_id", SortDirection.ASC),)),
                ]
            )
            await self._session_collection.ensure_indexes(
                [
                    CollectionIndex(
                        fields=(
                            ("cleanup_established", SortDirection.ASC),
                            ("creation_utc", SortDirection.ASC),
                            ("id", SortDirection.ASC),
                        )
                    ),
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
            creation_utc=session.creation_utc.astimezone(timezone.utc).isoformat(),
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
            customer_represented_by_an_agent=session.customer_represented_by_an_agent,
            cleanup_established=False,
            cleanup_has_greeting=False,
            observability_dirty=False,
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
            customer_represented_by_an_agent=session_document.get(
                "customer_represented_by_an_agent", False
            ),
            lifecycle=(
                (
                    "established"
                    if session_document["cleanup_established"]
                    or (
                        session_document.get("cleanup_has_greeting")
                        and not self._cleanup_settings.include_greetings
                    )
                    else "provisional"
                )
                if "cleanup_established" in session_document
                else None
            ),
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
        customer_represented_by_an_agent: bool = False,
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
                customer_represented_by_an_agent=customer_represented_by_an_agent,
                lifecycle="provisional",
            )

            await self._session_collection.insert_one(document=self._serialize_session(session))

        return session

    @override
    async def delete_session(
        self,
        session_id: SessionId,
    ) -> None:
        async with self._lock.writer_lock:
            # Explicit deletion follows the existing privacy path. Automatic
            # cleanup uses the internal helper and retains its accounting.
            records = await self._observability_collection.find({"session_id": {"$eq": session_id}})
            for record in records:
                await self._observability_collection.delete_one({"id": {"$eq": record["id"]}})
            await self._delete_session_unlocked(session_id)

    @staticmethod
    def _usage_records(metadata: Mapping[str, JSONSerializable]) -> Mapping[str, JSONSerializable]:
        records = metadata_utils.internal_value(metadata, ("usage", "records"), {})
        if not isinstance(records, Mapping):
            raise ValueError("Session usage records must be a mapping")
        return records

    async def _write_observability_unlocked(
        self,
        session: _SessionDocument,
        kind: Literal["usage", "session_deleted"],
        trace_id: str | None,
        occurred_at: datetime,
        payload: Mapping[str, JSONSerializable],
    ) -> None:
        identity = json.dumps([kind, session["id"], trace_id], separators=(",", ":"))
        record_id = ObjectId(hashlib.sha256(identity.encode()).hexdigest())
        previous = await self._observability_collection.find_one({"id": {"$eq": record_id}})
        normalized = deepcopy(dict(payload))
        if previous is not None:
            occurred_at = datetime.fromisoformat(previous["occurred_at"])
        if kind == "usage":
            normalized["timestamp_utc"] = occurred_at.astimezone(timezone.utc).isoformat()
        fingerprint = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if previous is not None and previous["fingerprint"] == fingerprint:
            return
        dashboard_metadata = session["metadata"].get("__dashboard__")
        is_simulation = (
            isinstance(dashboard_metadata, Mapping)
            and dashboard_metadata.get("isSimulation") is True
        )
        document = SessionObservabilityDocument(
            id=record_id,
            version=self.VERSION.to_string(),
            creation_utc=(
                previous["creation_utc"]
                if previous and previous["pending"]
                else datetime.now(timezone.utc).isoformat()
            ),
            revision=previous["revision"] + 1 if previous else 1,
            kind=kind,
            session_id=session["id"],
            agent_id=session["agent_id"],
            trace_id=trace_id,
            occurred_at=occurred_at.astimezone(timezone.utc).isoformat(),
            session_creation_utc=session["creation_utc"],
            is_simulation=is_simulation,
            customer_represented_by_an_agent=(
                session.get("customer_represented_by_an_agent", False) is True
            ),
            payload=normalized,
            fingerprint=fingerprint,
            pending=True,
        )
        await self._observability_collection.update_one(
            {"id": {"$eq": record_id}}, document, upsert=True
        )

    async def _capture_usage_unlocked(
        self, session: _SessionDocument, records: Mapping[str, JSONSerializable]
    ) -> None:
        for trace_id, record in records.items():
            if not isinstance(record, Mapping) or not isinstance(record.get("timestamp_utc"), str):
                raise ValueError("Session usage record must include timestamp_utc")
            occurred_at = datetime.fromisoformat(str(record["timestamp_utc"]))
            if occurred_at.tzinfo is None:
                raise ValueError("Session usage timestamp_utc must include a timezone")
            await self._write_observability_unlocked(
                session, "usage", trace_id, occurred_at, record
            )

    @override
    async def list_observability_records(
        self, *, limit: int = 100, cursor: Cursor | None = None
    ) -> SessionObservabilityPage:
        limit = observability_limit(limit)
        async with self._lock.reader_lock:
            page = await self._observability_collection.find(
                {"pending": {"$eq": True}},
                limit=limit,
                cursor=cursor,
                sort_direction=SortDirection.ASC,
            )
            records = []
            for doc in page:
                if doc["kind"] == "session_deleted" and await self._session_collection.find_one(
                    {"id": {"$eq": doc["session_id"]}}
                ):
                    # Prepared deletion, not a confirmed removal. Advance the
                    # scanned cursor anyway so it cannot starve later records.
                    continue
                records.append(
                    SessionObservabilityRecord(
                        id=doc["id"],
                        revision=doc["revision"],
                        kind=doc["kind"],
                        session_id=doc["session_id"],
                        agent_id=doc["agent_id"],
                        trace_id=doc["trace_id"],
                        occurred_at=datetime.fromisoformat(doc["occurred_at"]),
                        session_creation_utc=datetime.fromisoformat(doc["session_creation_utc"]),
                        is_simulation=doc["is_simulation"],
                        customer_represented_by_an_agent=doc["customer_represented_by_an_agent"],
                        payload=deepcopy(doc["payload"]),
                    )
                )
        return SessionObservabilityPage(
            records=records, has_more=page.has_more, next_cursor=page.next_cursor
        )

    @override
    async def acknowledge_observability_records(
        self, acknowledgements: Sequence[tuple[str, int]]
    ) -> int:
        if len(acknowledgements) > 100:
            raise ValueError("records must contain at most 100 acknowledgements")
        for record_id, revision in acknowledgements:
            if not record_id or len(record_id) > 256:
                raise ValueError("id must be nonblank and at most 256 characters")
            if type(revision) is not int or revision < 1:
                raise ValueError("revision must be a positive integer")
        acknowledged = 0
        async with self._lock.writer_lock:
            for record_id, revision in acknowledgements:
                result = await self._observability_collection.update_one(
                    {
                        "id": {"$eq": record_id},
                        "revision": {"$eq": revision},
                        "pending": {"$eq": True},
                    },
                    {"pending": False},
                )
                acknowledged += result.modified_count
        return acknowledged

    async def _delete_session_unlocked(self, session_id: SessionId) -> None:
        events = await self._event_collection.find(filters={"session_id": {"$eq": session_id}})
        await async_utils.safe_gather(
            *(self._event_collection.delete_one(filters={"id": {"$eq": e["id"]}}) for e in events)
        )
        await self._session_collection.delete_one({"id": {"$eq": session_id}})

    @override
    async def pin_session(self, session_id: SessionId) -> Callable[[], None]:
        async with self._lock.writer_lock:
            if not await self._session_collection.find_one({"id": {"$eq": session_id}}):
                raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")
            self._cleanup_pins[session_id] = self._cleanup_pins.get(session_id, 0) + 1

        released = False

        def release() -> None:
            # Synchronous and idempotent: task callbacks also release work that
            # was cancelled before its coroutine entered a try/finally block.
            nonlocal released
            if not released:
                released = True
                remaining = self._cleanup_pins[session_id] - 1
                if remaining:
                    self._cleanup_pins[session_id] = remaining
                else:
                    del self._cleanup_pins[session_id]

        return release

    @override
    async def list_empty_session_candidates(
        self,
        *,
        created_before: datetime,
        limit: int,
        cursor: Cursor | None = None,
    ) -> SessionListing:
        cutoff: WhereExpression = {
            "creation_utc": {"$lte": created_before.astimezone(timezone.utc).isoformat()}
        }
        candidate_filter: WhereExpression = dict(cutoff)
        candidate_filter["cleanup_established"] = {"$ne": True}
        async with self._lock.writer_lock:
            try:
                result = await self._session_collection.find(
                    filters=candidate_filter,
                    limit=limit,
                    cursor=cursor,
                    sort_direction=SortDirection.ASC,
                )
            except KeyError as error:
                if error.args != ("cleanup_established",):
                    raise
                # Embedded adapters reject a predicate on a missing field.
                # Discover only one bounded old page, without loader defaults
                # or a startup rewrite. The sweep cursor advances past it.
                result = await self._session_collection.find(
                    filters=cutoff,
                    limit=limit,
                    cursor=cursor,
                    sort_direction=SortDirection.ASC,
                )
                for doc in result:
                    if "cleanup_established" not in doc:
                        await self._session_collection.update_one(
                            {"id": {"$eq": doc["id"]}}, {"cleanup_established": False}
                        )
        return SessionListing(
            items=[self._deserialize_session(doc) for doc in result.items],
            total_count=result.total_count,
            has_more=result.has_more,
            next_cursor=result.next_cursor,
        )

    @staticmethod
    def _event_establishes_session(
        event: _EventDocument, metadata: Mapping[str, JSONSerializable]
    ) -> bool:
        if event["kind"] == EventKind.STATUS.value:
            return False
        if event["source"] == EventSource.CUSTOMER_UI.value:
            data = event["data"]
            # Passive page navigation precedes a greeting in the stock widget;
            # it is context, not customer participation.
            return not (isinstance(data, Mapping) and data.get("type") == "page_view")
        if event["source"] in {
            EventSource.CUSTOMER.value,
            EventSource.HUMAN_AGENT.value,
            EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT.value,
        }:
            return True
        if event["kind"] in {EventKind.MESSAGE.value, EventKind.TOOL.value}:
            greeting_trace = metadata_utils.internal_value(
                metadata, ("empty_session", "initial_greeting_trace_id"), None
            )
            return not greeting_trace or event["trace_id"] != greeting_trace
        return False

    @override
    async def delete_session_if_empty(
        self,
        session_id: SessionId,
        *,
        created_before: datetime,
        include_greetings: bool,
    ) -> bool:
        async with self._lock.writer_lock:
            doc = await self._session_collection.find_one({"id": {"$eq": session_id}})
            if not doc or doc.get("cleanup_established") or self._cleanup_pins.get(session_id):
                return False
            if (
                datetime.fromisoformat(doc["creation_utc"]).astimezone(timezone.utc)
                > created_before
            ):
                return False

            # Include soft-deleted history. Removing messages cannot turn an
            # established conversation back into an empty one. This also
            # classifies legacy records without a schema migration.
            events = await self._event_collection.find({"session_id": {"$eq": session_id}})
            metadata = doc.get("metadata", {})
            if any(self._event_establishes_session(event, metadata) for event in events):
                await self._session_collection.update_one(
                    {"id": {"$eq": session_id}}, {"cleanup_established": True}
                )
                return False
            if not include_greetings and any(
                event["kind"] == EventKind.MESSAGE.value for event in events
            ):
                return False
            try:
                await self._capture_usage_unlocked(doc, self._usage_records(metadata))
                await self._write_observability_unlocked(
                    doc, "session_deleted", None, datetime.now(timezone.utc), {}
                )
            except Exception:
                # Metadata remains the durable retry source if capture fails.
                # Never trade missing accounting for successful cleanup.
                return False
            await self._delete_session_unlocked(session_id)
            return True

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

        previous = await self._session_collection.find_one({"id": {"$eq": session_id}})
        if previous is None:
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")
        if previous.get("metadata_revision", 0) != expected_revision:
            raise MetadataRevisionConflict(expected_revision)
        incoming = self._usage_records(metadata)
        prior = self._usage_records(previous.get("metadata", {}))
        changed = (
            dict(incoming)
            if previous.get("observability_dirty")
            else {
                trace_id: record
                for trace_id, record in incoming.items()
                if prior.get(trace_id) != record
            }
        )
        result = await self._session_collection.update_one(
            filters={
                "id": {"$eq": session_id},
                "metadata_revision": {"$eq": expected_revision},
            },
            params={
                "metadata": metadata,
                "metadata_revision": expected_revision + 1,
                "modified_utc": datetime.now(timezone.utc).isoformat(),
                "observability_dirty": bool(changed),
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
        if changed:
            # Save the existing usage metadata first, so a journal outage
            # leaves durable evidence and the engine does not acknowledge it.
            await self._capture_usage_unlocked(updated_document, changed)
            await self._session_collection.update_one(
                {"id": {"$eq": session_id}}, {"observability_dirty": False}
            )
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
                limit=None if labels else limit,
                cursor=None if labels else cursor,
                sort_direction=sort_direction,
            )

            if labels:
                matching_documents = [
                    d for d in result.items if labels.issubset(set(d.get("labels", [])))
                ]
                total_count = len(matching_documents)
                if cursor:
                    position = (cursor.sort_value, str(cursor.id))
                    matching_documents = [
                        d
                        for d in matching_documents
                        if (
                            (str(d["creation_utc"]), str(d["id"])) < position
                            if sort_direction == SortDirection.DESC
                            else (str(d["creation_utc"]), str(d["id"])) > position
                        )
                    ]
                if limit is not None:
                    result = replace(
                        FindResult.create(
                            items=matching_documents[:limit],
                            total_count=len(matching_documents),
                            limit=limit,
                        ),
                        total_count=total_count,
                    )
                else:
                    result = FindResult(
                        items=matching_documents,
                        total_count=total_count,
                        has_more=False,
                    )

            return SessionListing(
                items=[self._deserialize_session(d) for d in result.items],
                total_count=result.total_count,
                has_more=result.has_more,
                next_cursor=result.next_cursor,
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
            session_document = await self._session_collection.find_one(
                filters={"id": {"$eq": session_id}}
            )
            if not session_document:
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

            event_document = self._serialize_event(event, session_id)
            await self._event_collection.insert_one(document=event_document)
            if not session_document.get("cleanup_established") and self._event_establishes_session(
                event_document, session_document.get("metadata", {})
            ):
                # Publish participation only after the event is durable, under
                # the same writer guard. A failed input write must not create
                # a phantom conversation. If this marker write fails, cleanup
                # still checks the persisted full history before any deletion.
                await self._session_collection.update_one(
                    {"id": {"$eq": session_id}},
                    {
                        "cleanup_established": True,
                        "modified_utc": max(
                            datetime.fromisoformat(session_document["modified_utc"]), creation_utc
                        ).isoformat(),
                    },
                )
            elif kind is EventKind.MESSAGE and source is EventSource.AI_AGENT:
                if not session_document.get("cleanup_has_greeting"):
                    await self._session_collection.update_one(
                        {"id": {"$eq": session_id}}, {"cleanup_has_greeting": True}
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
    async def list_observability_records(
        self, *, limit: int = 100, cursor: Cursor | None = None
    ) -> SessionObservabilityPage:
        return await self._writable_store.list_observability_records(limit=limit, cursor=cursor)

    @override
    async def acknowledge_observability_records(
        self, acknowledgements: Sequence[tuple[str, int]]
    ) -> int:
        return await self._writable_store.acknowledge_observability_records(acknowledgements)

    @override
    async def pin_session(self, session_id: SessionId) -> Callable[[], None]:
        return await self._writable_store.pin_session(session_id)

    @override
    async def list_empty_session_candidates(
        self,
        *,
        created_before: datetime,
        limit: int,
        cursor: Cursor | None = None,
    ) -> SessionListing:
        return await self._writable_store.list_empty_session_candidates(
            created_before=created_before, limit=limit, cursor=cursor
        )

    @override
    async def delete_session_if_empty(
        self,
        session_id: SessionId,
        *,
        created_before: datetime,
        include_greetings: bool,
    ) -> bool:
        return await self._writable_store.delete_session_if_empty(
            session_id, created_before=created_before, include_greetings=include_greetings
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
        customer_represented_by_an_agent: bool = False,
    ) -> Session:
        return await self._writable_store.create_session(
            customer_id=customer_id,
            agent_id=agent_id,
            creation_utc=creation_utc,
            title=title,
            mode=mode,
            metadata=metadata,
            labels=labels,
            customer_represented_by_an_agent=customer_represented_by_an_agent,
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


class EmptySessionCleanup:
    AGE = timedelta(minutes=30)
    INTERVAL_SECONDS = 60
    BATCH_SIZE = 100

    def __init__(self, store_provider: StoreProvider, logger: Logger) -> None:
        self._store_provider = store_provider
        self._logger = ScopedLogger(logger, "EmptySessionCleanup")
        settings = SessionCleanupSettings.from_environment()
        self.enabled = settings.enabled
        self._include_greetings = settings.include_greetings
        self._lock = asyncio.Lock()
        self._cursor: Cursor | None = None

    async def run_once(self, *, now: datetime, limit: int = BATCH_SIZE) -> int:
        if not self.enabled or limit <= 0:
            return 0
        async with self._lock:
            store = self._store_provider.get_store(
                SessionStore, StoreProviderHints(call_site="app")
            )
            cutoff = now.astimezone(timezone.utc) - self.AGE
            candidates = await store.list_empty_session_candidates(
                created_before=cutoff, limit=min(limit, self.BATCH_SIZE), cursor=self._cursor
            )
            removed = 0
            for session in candidates:
                if await store.delete_session_if_empty(
                    session.id,
                    created_before=cutoff,
                    include_greetings=self._include_greetings,
                ):
                    removed += 1
            # Advance past protected candidates too; an old pinned/greeting
            # session must not starve later empty ones. Retry a failed batch
            # from its old cursor; the conditional deletion is idempotent.
            self._cursor = candidates.next_cursor if candidates.has_more else None
            return removed

    async def run(self) -> None:
        while self.enabled:
            try:
                await self.run_once(now=datetime.now(timezone.utc))
            except Exception:
                # Never emit conversation content or a per-session log stream.
                self._logger.warning("Empty-session cleanup pass failed; retrying next minute")
            await asyncio.sleep(self.INTERVAL_SECONDS)
