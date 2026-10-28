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

from datetime import datetime
from decimal import Decimal
from enum import Enum
from copy import deepcopy

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import Field, field_serializer
from typing import Any, Annotated, AsyncIterator, Mapping, Sequence, TypeAlias, Union, cast


from parlant.api.authentication import AdminPrincipal, CustomerPrincipal, GuestPrincipal
from parlant.api.client_address import ClientAddressResolver
from parlant.core.session_clients import SESSION_CLIENT_METADATA_KEY, SessionClientRecorder
from parlant.api.authorization import (
    AuthorizationException,
    AuthorizationPolicy,
    GUEST_TOKEN_HEADER,
    Operation,
)
from parlant.api.common import (
    ExampleJson,
    JSONSerializableDTO,
    SortDirectionDTO,
    apigen_config,
    sort_direction_dto_to_sort_direction,
)
from parlant.core.app_modules.common import decode_cursor, encode_cursor
from parlant.core.app_modules.sessions import (
    EventMetadataUpdateParamsModel,
    EventUpdateParamsModel,
    SessionLabelsUpdateParams,
    SessionUpdateParamsModel,
)
from parlant.core.agents import AgentId
from parlant.core.application import Application
from parlant.core.async_utils import Timeout
from parlant.core.common import (
    DefaultBaseModel,
    ItemNotFoundError,
    JSONSerializable,
    generate_id,
)
from parlant.core.customers import CustomerId, CustomerKind, CustomerStore
from parlant.core.engines.types import UtteranceRationale, UtteranceRequest
from parlant.core.sessions import (
    AttachmentRef,
    Event,
    EventId,
    EventKind,
    EventSource,
    Participant,
    Session,
    SessionId,
    SessionStatus,
)
from parlant.core import metadata as metadata_utils
from parlant.core.pricing import costs_per_provider

API_GROUP = "sessions"


class EventKindDTO(Enum):
    """
    Type of event in a session.

    Represents different types of interactions that can occur within a conversation.
    """

    MESSAGE = "message"
    TOOL = "tool"
    STATUS = "status"
    CUSTOM = "custom"


class EventSourceDTO(Enum):
    """
    Source of an event in the session.

    Identifies who or what generated the event.
    """

    CUSTOMER = "customer"
    CUSTOMER_UI = "customer_ui"
    HUMAN_AGENT = "human_agent"
    HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT = "human_agent_on_behalf_of_ai_agent"
    AI_AGENT = "ai_agent"
    SYSTEM = "system"


class SessionStatusDTO(Enum):
    """
    Type of status in a session.
    """

    ACKNOWLEDGED = "acknowledged"
    CANCELLED = "cancelled"
    PROCESSING = "processing"
    READY = "ready"
    TYPING = "typing"
    ERROR = "error"


SessionIdPath: TypeAlias = Annotated[
    SessionId,
    Path(
        description="Unique identifier for the session",
        examples=["sess_123yz"],
    ),
]

SessionAgentIdPath: TypeAlias = Annotated[
    AgentId,
    Path(
        description="Unique identifier for the agent associated with the session.",
        examples=["ag-123Txyz"],
    ),
]

SessionCustomerIdField: TypeAlias = Annotated[
    CustomerId,
    Field(
        description="ID of the customer associated with this session.",
        examples=["cust_123xy"],
    ),
]

SessionCreationUTCField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp of when the session was created",
        examples=["2024-03-24T12:00:00Z"],
    ),
]

SessionModifiedUTCField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp of when the session was last modified",
        examples=["2024-03-24T12:00:00Z"],
    ),
]

SessionTitleField: TypeAlias = Annotated[
    str,
    Field(
        description="Descriptive title for the session",
        examples=["Support inquiry about product X"],
        max_length=200,
    ),
]


class SessionModeDTO(Enum):
    """Defines the reason for the action"""

    AUTO = "auto"
    MANUAL = "manual"


SessionModeField: TypeAlias = Annotated[
    SessionModeDTO,
    Field(
        description="The mode of the session, either 'auto' or 'manual'. In manual mode, events added to a session will not be responded to automatically by the agent.",
        examples=["auto", "manual"],
    ),
]

SessionMetadataField: TypeAlias = Annotated[
    Mapping[str, JSONSerializableDTO],
    Field(
        description="Metadata for the session",
        examples=[{"simulation": True, "priority": "high"}],
    ),
]

SessionLabelsField: TypeAlias = Annotated[
    set[str],
    Field(
        description="Labels associated with the session",
        examples=[{"vip", "priority"}],
    ),
]


session_example: ExampleJson = {
    "id": "sess_123yz",
    "agent_id": "ag_123xyz",
    "customer_id": "cust_123xy",
    "creation_utc": "2024-03-24T12:00:00Z",
    "modified_utc": "2024-03-24T12:00:00Z",
    "title": "Product inquiry session",
    "mode": "auto",
    "metadata": {"simulation": True, "priority": "high"},
    "labels": ["vip", "priority"],
}


class SessionDTO(
    DefaultBaseModel,
    json_schema_extra={"example": session_example},
):
    """A session represents an ongoing conversation between an agent and a customer."""

    id: SessionIdPath
    agent_id: SessionAgentIdPath
    customer_id: SessionCustomerIdField
    creation_utc: SessionCreationUTCField
    modified_utc: SessionModifiedUTCField
    title: SessionTitleField | None = None
    mode: SessionModeField
    metadata: SessionMetadataField
    labels: SessionLabelsField = set()


class ModelRequestUsageDTO(DefaultBaseModel):
    """Raw requests and provider usage attributed to one model in a trace."""

    request_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    extra: Mapping[str, int | float | str] = Field(default_factory=dict)


TurnRequestsUsageDTO: TypeAlias = Mapping[str, ModelRequestUsageDTO]


class TurnCostsDTO(DefaultBaseModel):
    """Exact server-priced cost snapshot for one trace.

    ``per_model`` is keyed by ``provider:model``; ``per_provider`` sums it by
    provider, and is null for records whose keys predate provider qualification.
    """

    currency: str
    total: Decimal
    per_model: Mapping[str, Decimal]
    per_tool: Mapping[str, Decimal] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    per_provider: Mapping[str, Decimal] | None = None

    @field_serializer("total")
    def serialize_total(self, value: Decimal) -> str:
        return format(value, "f")

    @field_serializer("per_model", "per_tool")
    def serialize_per_model(self, value: Mapping[str, Decimal]) -> Mapping[str, str]:
        return {model: format(cost, "f") for model, cost in value.items()}

    @field_serializer("per_provider")
    def serialize_per_provider(
        self, value: Mapping[str, Decimal] | None
    ) -> Mapping[str, str] | None:
        if value is None:
            return None
        return {provider: format(cost, "f") for provider, cost in value.items()}


class TurnUsageDTO(DefaultBaseModel):
    """Model requests and optional costs collected under one trace ID."""

    timestamp_utc: datetime
    requests: TurnRequestsUsageDTO
    costs: TurnCostsDTO | None


SessionUsageDTO: TypeAlias = Mapping[str, TurnUsageDTO]


def _aggregate_session_costs(usage: Mapping[str, TurnUsageDTO]) -> TurnCostsDTO | None:
    """The session-wide cost: the sum of every trace's costs, by total and per model.

    Returns None (rather than a misleadingly-low partial) when there is no usage,
    when any trace could not be priced, or when traces disagree on currency — so the
    session total is either complete or honestly absent, never silently under-reported.
    """
    if not usage:
        return None

    currency: str | None = None
    total = Decimal(0)
    per_model: dict[str, Decimal] = {}
    per_tool: dict[str, Decimal] = {}

    for turn in usage.values():
        if turn.costs is None:
            return None
        if currency is None:
            currency = turn.costs.currency
        elif turn.costs.currency != currency:
            return None
        total += turn.costs.total
        for category, cost in turn.costs.per_tool.items():
            per_tool[category] = per_tool.get(category, Decimal(0)) + cost
        for model, cost in turn.costs.per_model.items():
            per_model[model] = per_model.get(model, Decimal(0)) + cost

    assert currency is not None
    return TurnCostsDTO(
        currency=currency,
        total=total,
        per_model=per_model,
        per_tool=per_tool,
        per_provider=costs_per_provider(per_model),
    )


session_details_example: ExampleJson = {
    **cast(dict[str, Any], session_example),
    "usage": {},
    "costs": None,
}


class SessionDetailsDTO(
    SessionDTO,
    json_schema_extra={"example": session_details_example},
):
    """A retrieved session with its complete trace-level model usage."""

    usage: SessionUsageDTO
    costs: TurnCostsDTO | None
    """The session-wide cost: every trace's costs summed, or null if incomplete."""


class SessionListingDTO(DefaultBaseModel):
    """Paginated response for sessions"""

    items: Sequence[SessionDTO]
    total_count: int
    has_more: bool
    next_cursor: str | None = None


SessionCreationParamsCustomerIdField: TypeAlias = Annotated[
    CustomerId | None,
    Field(
        description=" ID of the customer this session belongs to. If not provided, a guest customer will be created.",
        examples=[None, "cust_123xy"],
    ),
]


session_creation_params_example: ExampleJson = {
    "agent_id": "ag_123xyz",
    "customer_id": "cust_123xy",
    "title": "Product inquiry session",
    "metadata": {"project": "demo", "priority": "high"},
    "labels": ["vip", "priority"],
}


class SessionCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": session_creation_params_example},
):
    """Parameters for creating a new session."""

    agent_id: SessionAgentIdPath
    customer_id: SessionCreationParamsCustomerIdField = None
    title: SessionTitleField | None = None
    metadata: SessionMetadataField | None = None
    labels: SessionLabelsField | None = None


message_example = "Hello, I need help with my order"


SessionEventCreationParamsMessageField: TypeAlias = Annotated[
    str,
    Field(
        description="Event payload data, format depends on kind",
        examples=[message_example],
    ),
]

AgentMessageRuleActionField: TypeAlias = Annotated[
    str,
    Field(
        description='A single action that explains what to say; i.e. "Tell the customer that you are thinking and will be right back with an answer."',
        examples=[message_example],
    ),
]

event_creation_params_example: ExampleJson = {
    "kind": "message",
    "source": "customer",
    "message": message_example,
}


class AgentMessageRuleRationaleDTO(Enum):
    """Defines the rationale for the rule"""

    UNSPECIFIED = "unspecified"
    BUY_TIME = "buy_time"
    FOLLOW_UP = "follow_up"


class AgentMessageRuleDTO(DefaultBaseModel):
    action: AgentMessageRuleActionField
    rationale: AgentMessageRuleRationaleDTO = AgentMessageRuleRationaleDTO.UNSPECIFIED


ParticipantIdDTO = AgentId | CustomerId | None

ParticipantDisplayNameField: TypeAlias = Annotated[
    str,
    Field(
        description="Name to display for the participant",
        examples=["John Doe", "Alice"],
    ),
]


participant_example = {
    "id": "cust_123xy",
    "display_name": "John Doe",
}


class ParticipantDTO(DefaultBaseModel):
    """
    Represents the participant information in a message event.
    """

    id: ParticipantIdDTO = None
    display_name: ParticipantDisplayNameField


EventMetadataField: TypeAlias = Annotated[
    Mapping[str, JSONSerializableDTO],
    Field(
        description="Metadata associated with the event",
        examples=[{"key1": "value1", "key2": 2}],
    ),
]


attachment_ref_example = {
    "url": "https://cdn.example.com/u/9f3a.jpg",
    "media_type": "image/jpeg",
    "caption": "photo of a leaking sink pipe",
    "filename": "faucet.jpg",
}


class AttachmentRefDTO(
    DefaultBaseModel,
    json_schema_extra={"example": attachment_ref_example},
):
    """A reference to a file that lives in the integrator's own storage.

    Parlant never stores the file; it only carries this reference so tools can act
    on it (usually by forwarding its URL)."""

    url: str = Field(description="Where the file lives, in your storage")
    media_type: str | None = Field(
        default=None, description="IANA media (MIME) type, e.g. image/jpeg"
    )
    caption: str | None = Field(
        default=None, description="A short description; the only field the agent model can read"
    )
    filename: str | None = Field(
        default=None, description="Original filename, for display and downloads"
    )
    metadata: JSONSerializableDTO | None = Field(
        default=None, description="Any additional data you want to carry"
    )


def _attachment_dtos_to_refs(
    dtos: list[AttachmentRefDTO] | None,
) -> Sequence[AttachmentRef] | None:
    if dtos is None:
        return None

    refs: list[AttachmentRef] = []
    for dto in dtos:
        ref: AttachmentRef = {"url": dto.url}
        if dto.media_type is not None:
            ref["media_type"] = dto.media_type
        if dto.caption is not None:
            ref["caption"] = dto.caption
        if dto.filename is not None:
            ref["filename"] = dto.filename
        if dto.metadata is not None:
            ref["metadata"] = cast(Mapping[str, JSONSerializable], dto.metadata)
        refs.append(ref)

    return refs


class EventCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": event_creation_params_example},
):
    """Parameters for creating a new event in a session."""

    kind: EventKindDTO
    source: EventSourceDTO
    message: SessionEventCreationParamsMessageField | None = None
    data: JSONSerializableDTO | None = None
    metadata: EventMetadataField | None = None
    rules: list[AgentMessageRuleDTO] | None = None
    participant: ParticipantDTO | None = None
    status: SessionStatusDTO | None = None
    attachments: list[AttachmentRefDTO] | None = None


EventIdPath: TypeAlias = Annotated[
    EventId,
    Path(
        description="Unique identifier for the event",
        examples=["evt_123xyz"],
    ),
]

EventOffsetField: TypeAlias = Annotated[
    int,
    Field(
        description="Sequential position of the event in the session",
        examples=[0, 1, 2],
        ge=0,
    ),
]

EventCreationUTCField: TypeAlias = Annotated[
    datetime,
    Field(description="UTC timestamp of when the event was created"),
]

EventModifiedUTCField: TypeAlias = Annotated[
    datetime,
    Field(description="UTC timestamp of when the event was last modified"),
]

EventTraceIdField: TypeAlias = Annotated[
    str,
    Field(
        description="ID linking related events together",
        examples=["trace_13xyz"],
    ),
]

event_example: ExampleJson = {
    "id": "evt_123xyz",
    "source": "customer",
    "kind": "message",
    "offset": 0,
    "creation_utc": "2024-03-24T12:00:00Z",
    "modified_utc": "2024-03-24T12:00:00Z",
    "trace_id": "trace_13xyz",
    "data": {
        "message": "Hello, I need help with my account",
        "participant": {"id": "cust_123xy", "display_name": "John Doe"},
    },
}


class EventDTO(
    DefaultBaseModel,
    json_schema_extra={"example": event_example},
):
    """Represents a single event within a session."""

    id: EventIdPath
    source: EventSourceDTO
    kind: EventKindDTO
    offset: EventOffsetField
    creation_utc: EventCreationUTCField
    modified_utc: EventModifiedUTCField
    trace_id: EventTraceIdField
    data: JSONSerializableDTO
    metadata: EventMetadataField
    deleted: bool


SessionMetadataUnsetField: TypeAlias = Annotated[
    Sequence[str],
    Field(
        description="Metadata keys to remove from the session",
        examples=[["simulation", "priority"]],
    ),
]

session_metadata_update_params_example: ExampleJson = {
    "set": {
        "simulation": False,
        "priority": "low",
    },
    "unset": ["simulation", "priority"],
}


class SessionMetadataUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": session_metadata_update_params_example},
):
    """Parameters for updating a session's metadata."""

    set: SessionMetadataField | None = None
    unset: SessionMetadataUnsetField | None = None


event_update_params_example: ExampleJson = {
    "metadata": {
        "set": {
            "priority": "high",
            "category": "support",
            "agent_id": "agent_123",
        },
        "unset": ["old_priority"],
    }
}


class EventUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": event_update_params_example},
):
    """Parameters for updating an event.

    Currently only supports updating metadata, but designed to be extensible
    for future event property updates.
    """

    metadata: SessionMetadataUpdateParamsDTO | None = None


session_update_params_example: ExampleJson = {
    "title": "Updated session title",
    "metadata": {
        "set": {"simulation": True, "priority": "low"},
        "unset": ["old_project"],
    },
    "labels": {
        "upsert": ["vip", "priority"],
        "remove": ["old_label"],
    },
}


session_labels_update_params_example: ExampleJson = {
    "upsert": ["vip", "priority"],
    "remove": ["old_label"],
}


class SessionLabelsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": session_labels_update_params_example},
):
    """Parameters for updating a session's labels."""

    upsert: SessionLabelsField | None = None
    remove: SessionLabelsField | None = None


class SessionUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": session_update_params_example},
):
    """Parameters for updating a session."""

    title: SessionTitleField | None = None
    mode: SessionModeField | None = None
    customer_id: CustomerId | None = None
    agent_id: AgentId | None = None
    metadata: SessionMetadataUpdateParamsDTO | None = None
    labels: SessionLabelsUpdateParamsDTO | None = None


ToolResultDataField: TypeAlias = Annotated[
    JSONSerializableDTO,
    Field(
        description="The json content returned from the tool",
        examples=["yes", '{"answer"="42"}', "[ 1, 1, 2, 3 ]"],
    ),
]


tool_result_metadata_example = {
    "duration_ms": 150,
    "cache_hit": False,
    "rate_limited": False,
}


ToolResultMetadataField: TypeAlias = Annotated[
    Mapping[str, JSONSerializableDTO],
    Field(
        description="A `dict` of the metadata associated with the tool's execution",
        examples=[tool_result_metadata_example],
    ),
]


tool_result_example = {
    "data": {
        "balance": 5000.50,
        "currency": "USD",
        "last_updated": "2024-03-24T12:00:00Z",
    },
    "metadata": tool_result_metadata_example,
}


class ToolResultDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tool_result_example},
):
    """Result from a tool execution."""

    data: ToolResultDataField
    metadata: ToolResultMetadataField
    costs: Mapping[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)


ToolIdField: TypeAlias = Annotated[
    str,
    Field(
        description="Unique identifier for the tool in format 'service_name:tool_name'",
        examples=["email-service:send_email", "payment-service:process_payment"],
    ),
]

tool_call_arguments_example = {"account_id": "acc_123xyz", "currency": "USD"}

ToolCallArgumentsField: TypeAlias = Annotated[
    Mapping[str, JSONSerializableDTO],
    Field(
        description="A `dict` of the arguments to the tool call",
        examples=[tool_call_arguments_example],
    ),
]


# Placeholder swapped in for a deep field the caller isn't authorized to see. A
# marker (rather than a dropped key) keeps the event shape stable and lets a
# client tell "hidden from me" apart from "genuinely empty".
REDACTED_EVENT_DATA: JSONSerializable = {"redacted": True}
REDACTED_MESSAGE = "[redacted]"
REDACTED_REASONING_LABEL = "Thinking"
"""What a caller without ``read_event_reasoning`` sees in place of the model's
own reasoning: the same label the engine emits as its "Thinking" stage, so a
client needs no special case to render it."""

INTERNAL_REASONING_METADATA_PATH = ("internal_reasoning",)
"""The ``__parlant__`` metadata path the engine sets on the status event that
carries the model's streamed reasoning. Tool narrations and the engine's stage
labels ("Running tools", ...) never carry it: they are written for the customer
and every caller may read them."""


def _strip_internal_namespace(value: JSONSerializable) -> JSONSerializable:
    """Return a copy of a metadata mapping without the reserved ``__parlant__``
    namespace. Non-mappings pass through unchanged."""
    if isinstance(value, Mapping):
        return {k: v for k, v in value.items() if k != metadata_utils.PARLANT_METADATA_KEY}
    return value


def _is_internal_reasoning(metadata: Mapping[str, JSONSerializable] | None) -> bool:
    return metadata_utils.internal_value(metadata, INTERNAL_REASONING_METADATA_PATH, False) is True


def _redact_event_data(
    kind: EventKind,
    data: JSONSerializable,
    metadata: Mapping[str, JSONSerializable] | None,
    *,
    redact_reasoning: bool,
    redact_tool_data: bool,
    redact_internal_metadata: bool = False,
) -> JSONSerializable:
    """Stub out the deep fields a caller isn't authorized to read.

    Two independent guards, matching two operations:
    - ``redact_reasoning`` replaces the model's own reasoning — the STATUS
      event the engine flags with ``__parlant__.internal_reasoning`` — with a
      bare "Thinking" label and drops its ``chunks``, so nothing of the chain
      of thought survives in any field. The ``status`` value and any ``data``
      are kept. Every other status text (a tool's narration, the engine's
      stage labels) is written for the customer and passes through untouched.
    - ``redact_tool_data`` hides each tool call's raw ``result.data`` while
      keeping the surrounding rationale, arguments and result metadata.

    The stored event is never mutated — redaction works on a copy, since it
    gates exposure only.
    """
    if kind == EventKind.STATUS and redact_reasoning and _is_internal_reasoning(metadata):
        if isinstance(data, Mapping):
            redacted = deepcopy(dict(data))
            redacted["message"] = REDACTED_REASONING_LABEL
            redacted.pop("chunks", None)
            data = cast(JSONSerializable, redacted)

    if redact_internal_metadata and kind == EventKind.STATUS:
        # A failure status carries the full server traceback under
        # ``data.exception``; only internal callers (admins, the Dashboard) may
        # see it. Everyone else gets a stub.
        if isinstance(data, Mapping) and isinstance(data.get("data"), Mapping):
            inner = cast(Mapping[str, Any], data["data"])
            if "exception" in inner:
                redacted = deepcopy(dict(data))
                cast(dict[str, Any], redacted["data"])["exception"] = REDACTED_MESSAGE
                data = cast(JSONSerializable, redacted)

    if kind == EventKind.TOOL and (redact_tool_data or redact_internal_metadata):
        if isinstance(data, Mapping) and isinstance(data.get("tool_calls"), list):
            redacted = deepcopy(dict(data))
            for tool_call in cast(list[Any], redacted["tool_calls"]):
                if isinstance(tool_call, dict) and isinstance(tool_call.get("result"), dict):
                    if redact_tool_data:
                        tool_call["result"]["data"] = REDACTED_EVENT_DATA
                    if redact_internal_metadata and "metadata" in tool_call["result"]:
                        tool_call["result"]["metadata"] = _strip_internal_namespace(
                            tool_call["result"]["metadata"]
                        )
            return cast(JSONSerializable, redacted)

    return data


def event_to_dto(
    event: Event,
    *,
    redact_reasoning: bool = False,
    redact_tool_data: bool = False,
    redact_internal_metadata: bool = False,
) -> EventDTO:
    data = _redact_event_data(
        event.kind,
        event.data,
        event.metadata,
        redact_reasoning=redact_reasoning,
        redact_tool_data=redact_tool_data,
        redact_internal_metadata=redact_internal_metadata,
    )

    metadata = (
        metadata_utils.public_metadata(event.metadata)
        if redact_internal_metadata
        else event.metadata
    )

    return EventDTO(
        id=event.id,
        source=_event_source_to_event_source_dto(event.source),
        kind=_event_kind_to_event_kind_dto(event.kind),
        offset=event.offset,
        creation_utc=event.creation_utc,
        modified_utc=event.modified_utc,
        trace_id=event.trace_id,
        data=cast(JSONSerializableDTO, data),
        metadata=cast(JSONSerializableDTO, metadata),
        deleted=event.deleted,
    )


async def _event_redaction_flags(
    authorization_policy: AuthorizationPolicy,
    request: Request,
) -> tuple[bool, bool, bool]:
    """Returns ``(redact_reasoning, redact_tool_data, redact_internal_metadata)``
    for this caller.

    A caller authorized to list/read events may still be denied the deeper
    fields: the internal reasoning in a status event's ``data``, a tool call's
    raw ``result.data``, and the Parlant-internal ``__parlant__`` metadata /
    failure tracebacks are each stubbed out unless the caller also holds the
    corresponding fine-grained permission.
    """
    redact_reasoning = not await authorization_policy.check_permission(
        request, Operation.READ_EVENT_REASONING
    )
    redact_tool_data = not await authorization_policy.check_permission(
        request, Operation.READ_TOOL_RESULT_DATA
    )
    redact_internal_metadata = not await authorization_policy.check_permission(
        request, Operation.READ_INTERNAL_EVENT_METADATA
    )
    return redact_reasoning, redact_tool_data, redact_internal_metadata


def session_to_dto(session: Session) -> SessionDTO:
    return SessionDTO(
        id=session.id,
        agent_id=session.agent_id,
        customer_id=session.customer_id,
        creation_utc=session.creation_utc,
        modified_utc=session.modified_utc,
        title=session.title,
        mode=SessionModeDTO(session.mode),
        metadata=metadata_utils.public_metadata(session.metadata),
        labels=session.labels,
    )


def session_to_details_dto(session: Session) -> SessionDetailsDTO:
    """Project internal durable usage into the first-class retrieve response."""

    usage_data = cast(
        Mapping[str, object],
        metadata_utils.internal_value(session.metadata, ("usage",), {}),
    )
    records = cast(Mapping[str, Mapping[str, object]], usage_data.get("records", {}))

    usage: dict[str, TurnUsageDTO] = {}
    for trace_id, record in records.items():
        models = cast(Mapping[str, Mapping[str, object]], record.get("models", {}))
        requests = {
            model: ModelRequestUsageDTO(
                request_count=cast(int, model_usage["requests"]),
                input_tokens=cast(int, model_usage["input_tokens"]),
                cached_input_tokens=cast(int, model_usage["cached_input_tokens"]),
                output_tokens=cast(int, model_usage["output_tokens"]),
                extra=cast(Mapping[str, int | float | str], model_usage.get("extra", {})),
            )
            for model, model_usage in models.items()
        }

        serialized_costs = cast(Mapping[str, object] | None, record.get("costs"))
        costs: TurnCostsDTO | None = None
        if serialized_costs is not None:
            per_model = {
                model: Decimal(str(value))
                for model, value in cast(
                    Mapping[str, object], serialized_costs.get("per_model", {})
                ).items()
            }
            costs = TurnCostsDTO(
                currency=str(serialized_costs["currency"]),
                total=Decimal(str(serialized_costs["total"])),
                per_tool={
                    category: Decimal(str(value))
                    for category, value in cast(
                        Mapping[str, object], serialized_costs.get("per_tool", {})
                    ).items()
                },
                per_model=per_model,
                per_provider=costs_per_provider(per_model),
            )

        usage[str(trace_id)] = TurnUsageDTO(
            timestamp_utc=datetime.fromisoformat(str(record["timestamp_utc"])),
            requests=requests,
            costs=costs,
        )

    session_dto = session_to_dto(session)
    return SessionDetailsDTO(
        **session_dto.model_dump(),
        usage=usage,
        costs=_aggregate_session_costs(usage),
    )


def participant_to_dto(participant: Participant) -> ParticipantDTO:
    return ParticipantDTO(
        id=participant["id"],
        display_name=participant["display_name"],
    )


AllowGreetingQuery: TypeAlias = Annotated[
    bool,
    Query(
        description="Whether to allow the agent to send an initial greeting",
    ),
]

AgentIdQuery: TypeAlias = Annotated[
    AgentId,
    Query(
        description="Unique identifier of the agent",
        examples=["ag_123xyz"],
    ),
]

CustomerIdQuery: TypeAlias = Annotated[
    CustomerId,
    Query(
        description="Unique identifier of the customers",
        examples=["cust_123xy"],
    ),
]

MinOffsetQuery: TypeAlias = Annotated[
    int,
    Query(
        description="Only return events with offset >= this value",
        examples=[0, 42],
    ),
]

TraceIdQuery: TypeAlias = Annotated[
    str,
    Query(
        description="ID linking related events together",
        examples=["trace_13xyz"],
    ),
]


KindsQuery: TypeAlias = Annotated[
    str,
    Query(
        description="If set, only list events of the specified kinds (separated by commas)",
        examples=["message,tool", "message,status"],
    ),
]

LimitQuery: TypeAlias = Annotated[
    int,
    Query(
        description="Maximum number of items to return",
        ge=1,
        le=100,
        examples=[10, 25],
    ),
]

CursorQuery: TypeAlias = Annotated[
    str,
    Query(
        description="Pagination cursor for fetching the next page of results",
        examples=["AAABjnBU9gBl/0BQt1axI0VniQI="],
    ),
]

SortQuery: TypeAlias = Annotated[
    SortDirectionDTO,
    Query(
        description="Sort direction for results",
        examples=["asc", "desc"],
    ),
]

MinModifiedUTCQuery: TypeAlias = Annotated[
    datetime,
    Query(
        description="Only return sessions modified at or after this UTC timestamp",
        examples=["2024-03-24T12:00:00Z"],
    ),
]


def agent_message_rule_dto_to_utterance_request(
    rule: AgentMessageRuleDTO,
) -> UtteranceRequest:
    rationale_to_reason = {
        AgentMessageRuleRationaleDTO.UNSPECIFIED: UtteranceRationale.UNSPECIFIED,
        AgentMessageRuleRationaleDTO.BUY_TIME: UtteranceRationale.BUY_TIME,
        AgentMessageRuleRationaleDTO.FOLLOW_UP: UtteranceRationale.FOLLOW_UP,
    }

    return UtteranceRequest(
        action=rule.action,
        rationale=rationale_to_reason[rule.rationale],
    )


def _event_kind_dto_to_event_kind(dto: EventKindDTO) -> EventKind:
    if kind := {
        EventKindDTO.MESSAGE: EventKind.MESSAGE,
        EventKindDTO.TOOL: EventKind.TOOL,
        EventKindDTO.STATUS: EventKind.STATUS,
        EventKindDTO.CUSTOM: EventKind.CUSTOM,
    }.get(dto):
        return kind

    raise ValueError(f"Invalid event kind: {dto}")


def _event_kind_to_event_kind_dto(kind: EventKind) -> EventKindDTO:
    if dto := {
        EventKind.MESSAGE: EventKindDTO.MESSAGE,
        EventKind.TOOL: EventKindDTO.TOOL,
        EventKind.STATUS: EventKindDTO.STATUS,
        EventKind.CUSTOM: EventKindDTO.CUSTOM,
    }.get(kind):
        return dto

    raise ValueError(f"Invalid event kind: {kind}")


def _event_source_dto_to_event_source(dto: EventSourceDTO) -> EventSource:
    if source := {
        EventSourceDTO.CUSTOMER: EventSource.CUSTOMER,
        EventSourceDTO.CUSTOMER_UI: EventSource.CUSTOMER_UI,
        EventSourceDTO.HUMAN_AGENT: EventSource.HUMAN_AGENT,
        EventSourceDTO.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
        EventSourceDTO.AI_AGENT: EventSource.AI_AGENT,
        EventSourceDTO.SYSTEM: EventSource.SYSTEM,
    }.get(dto):
        return source

    raise ValueError(f"Invalid event source: {dto}")


def _event_source_to_event_source_dto(source: EventSource) -> EventSourceDTO:
    if dto := {
        EventSource.CUSTOMER: EventSourceDTO.CUSTOMER,
        EventSource.CUSTOMER_UI: EventSourceDTO.CUSTOMER_UI,
        EventSource.HUMAN_AGENT: EventSourceDTO.HUMAN_AGENT,
        EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: EventSourceDTO.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
        EventSource.AI_AGENT: EventSourceDTO.AI_AGENT,
        EventSource.SYSTEM: EventSourceDTO.SYSTEM,
    }.get(source):
        return dto

    raise ValueError(f"Invalid event source: {source}")


def _participant_dto_to_participant(dto: ParticipantDTO) -> Participant:
    return Participant(
        id=AgentId(dto.id) if dto.id else None,
        display_name=dto.display_name,
    )


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
    client_address_resolver: ClientAddressResolver | None = None,
    session_client_recorder: SessionClientRecorder | None = None,
) -> APIRouter:
    router = APIRouter()
    client_addresses = client_address_resolver or ClientAddressResolver()
    client_recorder = session_client_recorder or SessionClientRecorder()

    def validate_public_metadata(metadata: Mapping[str, JSONSerializableDTO] | None) -> None:
        """Translate reserved-namespace violations into an API validation error."""

        try:
            metadata_utils.ensure_public_metadata(metadata or {})
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_session",
        response_model=SessionDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Session successfully created. Returns the complete session object.",
                "content": {"application/json": {"example": session_example}},
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_session(
        request: Request,
        response: Response,
        params: SessionCreationParamsDTO,
        allow_greeting: AllowGreetingQuery = False,
    ) -> SessionDTO:
        """Creates a new session between an agent and customer.

        The session's customer identity is derived from the authenticated
        principal when one is present: customer tokens bind the session to the
        token's customer; guest tokens (or anonymous callers, for whom a guest
        token is minted and returned via a response header) bind it to the guest
        customer, tagged with the owning guest instance. Policies without
        authentication (e.g. development) trust the request body as-is.
        """
        principal = await authorization_policy.resolve_principal(request)

        customer_id = params.customer_id
        metadata = dict(params.metadata or {})
        validate_public_metadata(metadata)
        guest_instance_id: str | None = None
        guest_token_to_issue: str | None = None

        match principal:
            case CustomerPrincipal():
                if customer_id is not None and customer_id != principal.customer_id:
                    raise AuthorizationException(
                        request,
                        Operation.CREATE_CUSTOMER_SESSION,
                        message_prefix=(
                            "The requested session customer does not match "
                            "the authenticated customer"
                        ),
                    )

                await authorization_policy.authorize(
                    request=request, operation=Operation.CREATE_CUSTOMER_SESSION
                )

                customer_id = principal.customer_id
            case GuestPrincipal():
                if customer_id is not None and customer_id != CustomerStore.GUEST_ID:
                    # A guest asking for a named customer's session: let the
                    # permission gate reject this as a customer-session attempt.
                    await authorization_policy.authorize(
                        request=request, operation=Operation.CREATE_CUSTOMER_SESSION
                    )

                await authorization_policy.authorize(
                    request=request, operation=Operation.CREATE_GUEST_SESSION
                )

                customer_id = CustomerStore.GUEST_ID
                guest_instance_id = principal.guest_instance_id
            case AdminPrincipal() | None:
                # Full-access callers (and policies without principals, like
                # development) keep the legacy body-driven behavior.
                if customer_id:
                    await authorization_policy.authorize(
                        request=request, operation=Operation.CREATE_CUSTOMER_SESSION
                    )
                else:
                    await authorization_policy.authorize(
                        request=request, operation=Operation.CREATE_GUEST_SESSION
                    )

                customer_id = customer_id or CustomerStore.GUEST_ID
            case _:  # AnonymousPrincipal
                if customer_id is not None and customer_id != CustomerStore.GUEST_ID:
                    await authorization_policy.authorize(
                        request=request, operation=Operation.CREATE_CUSTOMER_SESSION
                    )

                await authorization_policy.authorize(
                    request=request, operation=Operation.CREATE_GUEST_SESSION
                )

                customer_id = CustomerStore.GUEST_ID

                guest_instance_id = str(generate_id())
                guest_token_to_issue = authorization_policy.issue_guest_token(guest_instance_id)

        # Opening a session with a persona is a capability of its own, checked
        # here rather than inside each branch above so that every kind of
        # caller is held to it.
        #
        # An unknown id is a real customer for this purpose: only the agent is
        # validated on this path, so a session may be opened for a customer
        # that was never stored, and an id nobody stored is not a persona.
        # Treating the miss as a persona would turn a lookup failure into a
        # permission failure and break that long-standing behavior.
        try:
            target_customer = await app.customers.read(customer_id)
            target_kind = target_customer.kind
        except ItemNotFoundError:
            target_kind = CustomerKind.REAL

        if target_kind is CustomerKind.PERSONA:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_PERSONA_SESSION
            )

        # Record the browser that opened the session. Full-access callers are
        # often an integrator's own server, so they count as a browser only
        # when the request carries an Origin header.
        parlant_metadata: dict[str, JSONSerializable] | None = None
        full_access = principal is None or isinstance(principal, AdminPrincipal)

        if client_recorder.enabled and (not full_access or "origin" in request.headers):
            # An Origin header, even an opaque one, is the browser's own answer;
            # the Referer is consulted only when there is none.
            page = request.headers.get("origin")
            if page is None:
                page = request.headers.get("referer")

            client = await client_recorder.record(client_addresses.resolve(request), page)

            if client is not None:
                parlant_metadata = {SESSION_CLIENT_METADATA_KEY: dict(client)}

        session = await app.sessions.create(
            customer_id=customer_id,
            agent_id=params.agent_id,
            title=params.title,
            allow_greeting=allow_greeting,
            metadata=metadata,
            labels=params.labels,
            guest_instance_id=guest_instance_id,
            parlant_metadata=parlant_metadata,
        )

        if guest_token_to_issue is not None:
            response.headers[GUEST_TOKEN_HEADER] = guest_token_to_issue

        return session_to_dto(session)

    @router.get(
        "/{session_id}",
        operation_id="read_session",
        response_model=SessionDetailsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Session details successfully retrieved",
                "content": {"application/json": {"example": session_details_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_session(
        request: Request,
        session_id: SessionIdPath,
    ) -> SessionDetailsDTO:
        """Retrieves details of a specific session by ID."""
        await authorization_policy.authorize(request=request, operation=Operation.READ_SESSION)

        session = await app.sessions.read(session_id=session_id)

        await authorization_policy.authorize_session_access(request=request, session=session)

        return session_to_details_dto(session)

    @router.get(
        "",
        operation_id="list_sessions",
        response_model=SessionListingDTO | Sequence[SessionDTO],
        responses={
            status.HTTP_200_OK: {
                "description": (
                    "If a limit is provided, a paginated list of sessions will be returned. "
                    "Otherwise, the full list of sessions will be returned."
                ),
                "content": {
                    "application/json": {
                        "example": {
                            "items": [session_example],
                            "total_count": 1,
                            "has_more": False,
                            "next_cursor": None,
                        }
                    }
                },
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in the request parameters."
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_sessions(
        request: Request,
        agent_id: AgentIdQuery | None = None,
        customer_id: CustomerIdQuery | None = None,
        labels: list[str] = Query(default=[]),
        limit: LimitQuery | None = None,
        cursor: CursorQuery | None = None,
        sort: SortQuery | None = None,
        min_modified_utc: MinModifiedUTCQuery | None = None,
    ) -> SessionListingDTO | Sequence[SessionDTO]:
        """Lists all sessions matching the specified filters with pagination support.

        Can filter by agent_id and/or customer_id. Supports cursor-based pagination
        with configurable sort direction."""
        await authorization_policy.authorize(request=request, operation=Operation.LIST_SESSIONS)

        principal = await authorization_policy.resolve_principal(request)

        if isinstance(principal, CustomerPrincipal):
            # A customer may only ever list its own sessions.
            if customer_id is not None and customer_id != principal.customer_id:
                raise AuthorizationException(
                    request,
                    Operation.LIST_SESSIONS,
                    message_prefix=(
                        "The requested session listing customer does not match "
                        "the authenticated customer"
                    ),
                )

            customer_id = principal.customer_id

        sessions_result = await app.sessions.find(
            agent_id=agent_id,
            customer_id=customer_id,
            limit=limit,
            cursor=decode_cursor(cursor) if cursor else None,
            sort_direction=sort_direction_dto_to_sort_direction(sort) if sort else None,
            labels=set(labels) if labels else None,
            min_modified_utc=min_modified_utc,
        )

        if limit is None:
            return [session_to_dto(s) for s in sessions_result.items]

        return SessionListingDTO(
            items=[session_to_dto(s) for s in sessions_result.items],
            total_count=sessions_result.total_count,
            has_more=sessions_result.has_more,
            next_cursor=encode_cursor(sessions_result.next_cursor)
            if sessions_result.next_cursor
            else None,
        )

    @router.delete(
        "/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_session",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Session successfully deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_session(
        request: Request,
        session_id: SessionIdPath,
    ) -> None:
        """Deletes a session and all its associated events.

        The operation is idempotent - deleting a non-existent session will return 404."""
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_SESSION)

        await app.sessions.delete(session_id=session_id)

    @router.delete(
        "",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_sessions",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "All matching sessions successfully deleted"
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_many"),
    )
    async def delete_sessions(
        request: Request,
        agent_id: AgentIdQuery | None = None,
        customer_id: CustomerIdQuery | None = None,
    ) -> None:
        """Deletes all sessions matching the specified filters.

        Can filter by agent_id and/or customer_id. Will delete all sessions if no
        filters are provided."""
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_SESSIONS)

        sessions_result = await app.sessions.find(
            agent_id=agent_id,
            customer_id=customer_id,
        )

        for s in sessions_result.items:
            await app.sessions.delete(s.id)

    @router.patch(
        "/{session_id}",
        operation_id="update_session",
        responses={
            status.HTTP_200_OK: {"description": "Session successfully updated"},
            status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_session(
        request: Request,
        session_id: SessionIdPath,
        params: SessionUpdateParamsDTO,
    ) -> SessionDTO:
        """Updates an existing session's attributes.

        Only provided attributes will be updated; others remain unchanged."""
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_SESSION)
        if params.metadata:
            validate_public_metadata(params.metadata.set)
            for key in params.metadata.unset or ():
                try:
                    metadata_utils.ensure_public_metadata_key(key)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail=str(exc),
                    ) from exc

        async def from_dto(dto: SessionUpdateParamsDTO) -> SessionUpdateParamsModel:
            params: SessionUpdateParamsModel = {}

            if dto.title is not None:
                params["title"] = dto.title

            if dto.mode is not None:
                params["mode"] = dto.mode.value

            if dto.customer_id is not None:
                params["customer_id"] = dto.customer_id

            if dto.agent_id is not None:
                params["agent_id"] = dto.agent_id

            return params

        session = await app.sessions.update(
            session_id=session_id,
            params=await from_dto(params),
            metadata=(
                EventMetadataUpdateParamsModel(
                    set=params.metadata.set or {},
                    unset=params.metadata.unset or (),
                )
                if params.metadata
                else None
            ),
            labels=SessionLabelsUpdateParams(
                upsert=params.labels.upsert,
                remove=params.labels.remove,
            )
            if params.labels
            else None,
        )

        return session_to_dto(session)

    @router.post(
        "/{session_id}/events",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_event",
        response_model=EventDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Event successfully created",
                "content": {"application/json": {"example": event_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in event parameters"
            },
            status.HTTP_409_CONFLICT: {
                "description": "Cancellation requested, but no processing task is active"
            },
            status.HTTP_504_GATEWAY_TIMEOUT: {
                "description": "Cancellation requested, but no cancellation event was emitted"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create_event"),
    )
    async def create_event(
        request: Request,
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        """Creates a new event in the specified session.

        Currently supports creating message events from customer and human agent sources."""
        validate_public_metadata(params.metadata)

        if params.kind == EventKindDTO.MESSAGE:
            if params.source == EventSourceDTO.CUSTOMER:
                await authorization_policy.authorize(
                    request=request, operation=Operation.CREATE_CUSTOMER_EVENT
                )
                await authorization_policy.authorize_session_access(
                    request=request,
                    session=await app.sessions.read(session_id=session_id),
                )
                if params.participant:
                    await authorization_policy.authorize(
                        request=request, operation=Operation.OVERRIDE_CUSTOMER_PARTICIPANT
                    )
                if params.attachments and not await authorization_policy.check_permission(
                    request, Operation.CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=(
                            "This caller is not authorized to attach files to customer "
                            "messages. Retry without 'attachments', or ask the operator to "
                            "grant the 'create_customer_event_with_attachments' permission."
                        ),
                    )
                return await _add_customer_message(session_id, params)
            elif params.source == EventSourceDTO.AI_AGENT:
                await authorization_policy.authorize(
                    request=request, operation=Operation.CREATE_AGENT_EVENT
                )
                return await _add_agent_message(session_id, params)
            elif params.source == EventSourceDTO.HUMAN_AGENT:
                await authorization_policy.authorize(
                    request=request,
                    operation=Operation.CREATE_HUMAN_AGENT_EVENT,
                )
                return await _add_human_agent_message(session_id, params)
            elif params.source == EventSourceDTO.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT:
                await authorization_policy.authorize(
                    request=request,
                    operation=Operation.CREATE_HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT_EVENT,
                )
                return await _add_human_agent_message_on_behalf_of_ai_agent(session_id, params)
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail='Only "customer", "human_agent", and "human_agent_on_behalf_of_ai_agent" sources are supported for direct posting.',
                )

        elif params.kind == EventKindDTO.CUSTOM:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_CUSTOM_EVENT
            )
            await authorization_policy.authorize_session_access(
                request=request,
                session=await app.sessions.read(session_id=session_id),
            )
            return await _add_custom_event(session_id, params)

        elif params.kind == EventKindDTO.STATUS:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_STATUS_EVENT
            )
            await authorization_policy.authorize_session_access(
                request=request,
                session=await app.sessions.read(session_id=session_id),
            )
            return await _add_status_event(session_id, params)

        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Only message, custom and status events can currently be added manually",
            )

    async def _add_status_event(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        def status_dto_to_status(dto: SessionStatusDTO) -> SessionStatus:
            match dto:
                case SessionStatusDTO.ACKNOWLEDGED:
                    return "acknowledged"
                case SessionStatusDTO.CANCELLED:
                    return "cancelled"
                case SessionStatusDTO.PROCESSING:
                    return "processing"
                case SessionStatusDTO.READY:
                    return "ready"
                case SessionStatusDTO.TYPING:
                    return "typing"
                case SessionStatusDTO.ERROR:
                    return "error"

        if params.status is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Missing "status" field for status event',
            )

        session_status = status_dto_to_status(params.status)

        raw_data = params.data or {}
        if not isinstance(raw_data, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Status event "data" must be a JSON object',
            )

        if session_status == "cancelled":
            try:
                return event_to_dto(await app.sessions.cancel_processing(session_id))
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(exc),
                ) from exc
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=str(exc),
                ) from exc

        event = await app.sessions.create_status_event(
            session_id=session_id,
            status=session_status,
            data=raw_data,
            metadata=params.metadata,
            source=_event_source_dto_to_event_source(params.source),
        )

        return event_to_dto(event)

    async def _add_customer_message(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if not params.message and not params.attachments:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="A customer message needs 'message' text or at least one attachment",
            )

        event = await app.sessions.create_customer_message(
            session_id=session_id,
            message=params.message or "",
            metadata=params.metadata,
            source=EventSource.CUSTOMER,
            trigger_processing=True,
            participant=_participant_dto_to_participant(params.participant)
            if params.participant
            else None,
            attachments=_attachment_dtos_to_refs(params.attachments),
        )

        return event_to_dto(event)

    async def _add_agent_message(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if params.message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="If you add an agent message, you cannot specify what the message will be, as it will be auto-generated by the agent.",
            )

        if params.rules:
            requests = [agent_message_rule_dto_to_utterance_request(a) for a in params.rules]
            event = await app.sessions.utter(session_id, requests)
            return event_to_dto(event)
        else:
            event = await app.sessions.process(session_id)
            return event_to_dto(event)

    async def _add_human_agent_message(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if not params.message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Missing 'message' field for event",
            )
        if not params.participant or not params.participant.display_name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Missing 'participant' with 'display_name' for human agent message",
            )

        event = await app.sessions.create_human_agent_message_event(
            session_id=session_id,
            message=params.message,
            participant=_participant_dto_to_participant(params.participant),
            metadata=params.metadata,
        )

        return event_to_dto(event)

    async def _add_human_agent_message_on_behalf_of_ai_agent(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if not params.message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Missing 'data' field for message",
            )

        event = await app.sessions.create_human_agent_on_behalf_of_ai_agent_message_event(
            session_id=session_id,
            message=params.message,
            metadata=params.metadata,
        )

        return event_to_dto(event)

    async def _add_custom_event(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if not params.data:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Missing 'data' field for custom event",
            )

        event = await app.sessions.create_event(
            session_id=session_id,
            kind=_event_kind_dto_to_event_kind(params.kind),
            data=params.data,
            metadata=params.metadata,
            source=_event_source_dto_to_event_source(params.source),
            trigger_processing=False,
        )

        return event_to_dto(event)

    @router.get(
        "/{session_id}/events",
        operation_id="list_events",
        response_model=Sequence[EventDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of events matching the specified criteria",
                "content": {"application/json": {"example": [event_example]}},
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Session not found",
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
            status.HTTP_504_GATEWAY_TIMEOUT: {
                "description": "Request timeout waiting for new events"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list_events"),
    )
    async def list_events(
        request: Request,
        session_id: SessionIdPath,
        min_offset: MinOffsetQuery | None = None,
        source: EventSourceDTO | None = None,
        trace_id: TraceIdQuery | None = None,
        kinds: KindsQuery | None = None,
        wait_for_data: int = 60,
        sse: bool = False,
    ) -> Union[Sequence[EventDTO], Response]:
        """Lists events from a session with optional filtering and waiting capabilities.

        This endpoint retrieves events from a specified session and can:
        1. Filter events by their offset, source, type, and trace ID
        2. Wait for new events to arrive if requested
        3. Return events in chronological order based on their offset
        4. Stream events via Server-Sent Events (SSE) when sse=true

        Notes:
            Long Polling Behavior (when sse=false):
            - When wait_for_data = 0:
                Returns immediately with any existing events that match the criteria
            - When wait_for_data > 0:
                - If new matching events arrive within the timeout period, returns with those events
                - If no new events arrive before timeout, raises 504 Gateway Timeout
                - If matching events already exist, returns immediately with those events

            SSE Mode (when sse=true):
            - Returns a text/event-stream response
            - Continuously sends events as they arrive
            - wait_for_data is used as the timeout between events before closing the stream
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_EVENTS)

        await authorization_policy.authorize_session_access(
            request=request,
            session=await app.sessions.read(session_id=session_id),
        )

        redact_reasoning, redact_tool_data, redact_internal_metadata = await _event_redaction_flags(
            authorization_policy, request
        )

        kind_list: Sequence[EventKind] = [
            _event_kind_dto_to_event_kind(EventKindDTO(k))
            for k in (kinds.split(",") if kinds else [])
        ]

        event_source = _event_source_dto_to_event_source(source) if source else None

        if sse:
            # Return SSE stream
            async def event_stream() -> AsyncIterator[str]:
                current_offset = min_offset or 0
                try:
                    while True:
                        # Wait for new events
                        has_events = await app.sessions.wait_for_more_events(
                            session_id=session_id,
                            min_offset=current_offset,
                            source=event_source,
                            kinds=kind_list,
                            trace_id=trace_id,
                            timeout=Timeout(wait_for_data),
                        )

                        if not has_events:
                            # Timeout - close the stream
                            break

                        # Get new events
                        events = await app.sessions.find_events(
                            session_id=session_id,
                            min_offset=current_offset,
                            source=event_source,
                            kinds=kind_list,
                            trace_id=trace_id,
                        )

                        for e in events:
                            event_dto = event_to_dto(
                                e,
                                redact_reasoning=redact_reasoning,
                                redact_tool_data=redact_tool_data,
                                redact_internal_metadata=redact_internal_metadata,
                            )
                            yield f"data: {event_dto.model_dump_json()}\n\n"
                            current_offset = max(current_offset, e.offset + 1)
                except ItemNotFoundError:
                    # Session was deleted or doesn't exist - gracefully close the stream
                    return

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Standard long-polling behavior
        if wait_for_data > 0:
            if not await app.sessions.wait_for_more_events(
                session_id=session_id,
                min_offset=min_offset or 0,
                source=event_source,
                kinds=kind_list,
                trace_id=trace_id,
                timeout=Timeout(wait_for_data),
            ):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request timed out",
                )

        events = await app.sessions.find_events(
            session_id=session_id,
            min_offset=min_offset or 0,
            source=event_source,
            kinds=kind_list,
            trace_id=trace_id,
        )

        return [
            event_to_dto(
                e,
                redact_reasoning=redact_reasoning,
                redact_tool_data=redact_tool_data,
                redact_internal_metadata=redact_internal_metadata,
            )
            for e in events
        ]

    @router.get(
        "/{session_id}/events/{event_id}",
        operation_id="read_event",
        response_model=EventDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Event details successfully retrieved",
                "content": {"application/json": {"example": event_example}},
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Session or event not found",
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="read_event"),
    )
    async def read_event(
        request: Request,
        session_id: SessionIdPath,
        event_id: EventIdPath,
        wait_for_completion: bool = False,
        wait_for_data: int = 60,
        sse: bool = False,
    ) -> Union[EventDTO, Response]:
        """Reads a single event from a session.

        This endpoint retrieves a specific event by its ID and optionally waits
        for the event to complete (useful for streaming messages).

        Args:
            wait_for_completion: If true, wait for the event to complete (for streaming events,
                this means waiting until chunks contains None terminator)
            wait_for_data: Timeout in seconds for wait_for_completion
            sse: If true, stream event updates via Server-Sent Events until completion

        Notes:
            For streaming message events (events with 'chunks' property):
            - The event is considered complete when chunks contains a None terminator
            - Use wait_for_completion=true to wait for the full message
            - Use sse=true to stream updates as chunks are added

            SSE Mode (when sse=true):
            - Returns a text/event-stream response
            - Sends the event each time it's updated
            - Closes when the event is complete (chunks ends with None)
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_EVENT)

        await authorization_policy.authorize_session_access(
            request=request,
            session=await app.sessions.read(session_id=session_id),
        )

        redact_reasoning, redact_tool_data, redact_internal_metadata = await _event_redaction_flags(
            authorization_policy, request
        )

        if sse:
            # Return SSE stream that sends event updates until completion
            async def event_stream() -> AsyncIterator[str]:
                chunk_count = 0
                while True:
                    event = await app.sessions.read_event(
                        session_id=session_id,
                        event_id=event_id,
                    )
                    event_dto = event_to_dto(
                        event,
                        redact_reasoning=redact_reasoning,
                        redact_tool_data=redact_tool_data,
                        redact_internal_metadata=redact_internal_metadata,
                    )
                    yield f"data: {event_dto.model_dump_json()}\n\n"

                    # Check if event is complete (for streaming events)
                    data = cast(dict[str, object], event.data)
                    if "chunks" in data:
                        chunks = cast(list[str | None], data["chunks"])
                        chunk_count = len(chunks)
                        if chunks and chunks[-1] is None:
                            break
                    else:
                        # Non-streaming event, just return it once
                        break

                    # Wait for new chunks (not full completion)
                    if not await app.sessions.wait_for_new_streaming_chunks(
                        session_id=session_id,
                        event_id=event_id,
                        last_known_chunk_count=chunk_count,
                        timeout=Timeout(wait_for_data),
                    ):
                        break

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Standard read
        event = await app.sessions.read_event(
            session_id=session_id,
            event_id=event_id,
        )

        if wait_for_completion:
            await app.sessions.wait_for_event_completion(
                session_id=session_id,
                event_id=event_id,
                timeout=Timeout(wait_for_data),
            )
            # Re-read the event after waiting
            event = await app.sessions.read_event(
                session_id=session_id,
                event_id=event_id,
            )

        return event_to_dto(
            event,
            redact_reasoning=redact_reasoning,
            redact_tool_data=redact_tool_data,
            redact_internal_metadata=redact_internal_metadata,
        )

    @router.delete(
        "/{session_id}/events",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_events",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Events successfully deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_events"),
    )
    async def delete_events(
        request: Request,
        session_id: SessionIdPath,
        min_offset: MinOffsetQuery,
    ) -> None:
        """Deletes events from a session with offset >= the specified value.

        This operation is permanent and cannot be undone."""
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_EVENTS)

        try:
            await app.sessions.delete_events(session_id=session_id, min_offset=min_offset)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"{e}")

    @router.patch(
        "/{session_id}/events/{event_id}",
        operation_id="update_event",
        response_model=EventDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Event successfully updated",
                "content": {"application/json": {"example": event_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "Session or event not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update_event"),
    )
    async def update_event(
        request: Request,
        session_id: SessionIdPath,
        event_id: EventIdPath,
        params: EventUpdateParamsDTO,
    ) -> EventDTO:
        """Updates an event's properties.

        Currently only supports updating metadata. Other event properties cannot be modified.
        This API is designed to be extensible for future event property updates.
        """
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_EVENT)
        if params.metadata:
            validate_public_metadata(params.metadata.set)
            for key in params.metadata.unset or ():
                try:
                    metadata_utils.ensure_public_metadata_key(key)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail=str(exc),
                    ) from exc

        update_params: EventUpdateParamsModel = {}
        if params.metadata is not None:
            # Convert API DTO to app_modules model - pass through set/unset operations
            metadata_update: EventMetadataUpdateParamsModel = {}
            if params.metadata.set:
                metadata_update["set"] = params.metadata.set
            if params.metadata.unset:
                metadata_update["unset"] = params.metadata.unset

            update_params["metadata"] = metadata_update

        event = await app.sessions.update_event(
            session_id=session_id,
            event_id=event_id,
            params=update_params,
        )

        return event_to_dto(event)

    return router
