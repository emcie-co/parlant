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

from datetime import datetime
from enum import Enum
from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from itertools import chain
from pydantic import Field
from typing import Annotated, Mapping, Optional, Sequence, Set, TypeAlias, cast


from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import GuidelineIdField, ExampleJson, JSONSerializableDTO, apigen_config
from parlant.api.glossary import TermSynonymsField, TermIdPath, TermNameField, TermDescriptionField
from parlant.core.agents import AgentId, AgentStore
from parlant.core.application import Application
from parlant.core.async_utils import Timeout
from parlant.core.common import DefaultBaseModel
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.engines.types import UtteranceRationale, UtteranceRequest
from parlant.core.loggers import Logger
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.moderation import ModerationService
from parlant.core.nlp.service import NLPService
from parlant.core.sessions import (
    Event,
    EventId,
    EventKind,
    EventSource,
    MessageEventData,
    MessageGenerationInspection,
    Participant,
    PreparationIteration,
    SessionId,
    SessionListener,
    SessionStatus,
    SessionStore,
    SessionUpdateParams,
    StatusEventData,
    ToolEventData,
)
from parlant.core.canned_responses import CannedResponseId

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


class Moderation(Enum):
    """Content moderation settings."""

    AUTO = "auto"
    PARANOID = "paranoid"
    NONE = "none"


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


ConsumptionOffsetClientField: TypeAlias = Annotated[
    int,
    Field(
        description="Latest event offset processed by the client",
        examples=[42, 100],
        ge=0,
    ),
]

consumption_offsets_example = {"client": 42}


class ConsumptionOffsetsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": consumption_offsets_example},
):
    """Tracking for message consumption state."""

    client: Optional[ConsumptionOffsetClientField] = None


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


session_example: ExampleJson = {
    "id": "sess_123yz",
    "agent_id": "ag_123xyz",
    "customer_id": "cust_123xy",
    "creation_utc": "2024-03-24T12:00:00Z",
    "title": "Product inquiry session",
    "mode": "auto",
    "consumption_offsets": consumption_offsets_example,
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
    title: Optional[SessionTitleField] = None
    mode: SessionModeField
    consumption_offsets: ConsumptionOffsetsDTO


SessionCreationParamsCustomerIdField: TypeAlias = Annotated[
    Optional[CustomerId],
    Field(
        default=None,
        description=" ID of the customer this session belongs to. If not provided, a guest customer will be created.",
        examples=[None, "cust_123xy"],
    ),
]


session_creation_params_example: ExampleJson = {
    "agent_id": "ag_123xyz",
    "customer_id": "cust_123xy",
    "title": "Product inquiry session",
}


class SessionCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": session_creation_params_example},
):
    """Parameters for creating a new session."""

    agent_id: SessionAgentIdPath
    customer_id: SessionCreationParamsCustomerIdField = None
    title: Optional[SessionTitleField] = None


message_example = "Hello, I need help with my order"


SessionEventCreationParamsMessageField: TypeAlias = Annotated[
    str,
    Field(
        description="Event payload data, format depends on kind",
        examples=[message_example],
    ),
]

AgentMessageGuidelineActionField: TypeAlias = Annotated[
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


class AgentMessageGuidelineRationaleDTO(Enum):
    """Defines the rationale for the guideline"""

    UNSPECIFIED = "unspecified"
    BUY_TIME = "buy_time"
    FOLLOW_UP = "follow_up"


class AgentMessageGuidelineDTO(DefaultBaseModel):
    action: AgentMessageGuidelineActionField
    rationale: AgentMessageGuidelineRationaleDTO = AgentMessageGuidelineRationaleDTO.UNSPECIFIED


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


class EventCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": event_creation_params_example},
):
    """Parameters for creating a new event in a session."""

    kind: EventKindDTO
    source: EventSourceDTO
    message: Optional[SessionEventCreationParamsMessageField] = None
    data: Optional[JSONSerializableDTO] = None
    guidelines: Optional[list[AgentMessageGuidelineDTO]] = None
    participant: Optional[ParticipantDTO] = None
    status: Optional[SessionStatusDTO] = None


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


EventCorrelationIdField: TypeAlias = Annotated[
    str,
    Field(
        description="ID linking related events together",
        examples=["corr_13xyz"],
    ),
]

event_example: ExampleJson = {
    "id": "evt_123xyz",
    "source": "customer",
    "kind": "message",
    "offset": 0,
    "creation_utc": "2024-03-24T12:00:00Z",
    "correlation_id": "corr_13xyz",
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
    correlation_id: EventCorrelationIdField
    data: JSONSerializableDTO
    deleted: bool


class ConsumptionOffsetsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": consumption_offsets_example},
):
    """Parameters for updating consumption offsets."""

    client: Optional[ConsumptionOffsetClientField] = None


session_update_params_example: ExampleJson = {
    "title": "Updated session title",
    "consumption_offsets": {"client": 42},
}


class SessionUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": session_update_params_example},
):
    """Parameters for updating a session."""

    consumption_offsets: Optional[ConsumptionOffsetsUpdateParamsDTO] = None
    title: Optional[SessionTitleField] = None
    mode: Optional[SessionModeField] = None


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


tool_call_example = {
    "tool_id": "finance_service:check_balance",
    "arguments": tool_call_arguments_example,
    "result": {
        "data": {
            "balance": 5000.50,
            "currency": "USD",
            "last_updated": "2024-03-24T12:00:00Z",
        },
        "metadata": tool_result_metadata_example,
    },
}


class ToolCallDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tool_call_example},
):
    """Information about a tool call."""

    tool_id: ToolIdField
    arguments: ToolCallArgumentsField
    result: ToolResultDTO


GuidelineMatchConditionField: TypeAlias = Annotated[
    str,
    Field(
        description="The condition for the guideline",
        examples=["when customer asks about their balance"],
    ),
]

GuidelineMatchActionField: TypeAlias = Annotated[
    str,
    Field(
        description="The action for the guideline",
        examples=["check their current balance and provide the amount with currency"],
    ),
]

GuidelineMatchScoreField: TypeAlias = Annotated[
    int,
    Field(
        description="The score for the guideline",
        examples=[95],
    ),
]

GuidelineMatchRationaleField: TypeAlias = Annotated[
    str,
    Field(
        description="The rationale for the guideline",
        examples=["This guideline directly addresses balance inquiries with specific actions"],
    ),
]

guideline_match_example = {
    "guideline_id": "guide_123x",
    "condition": "when customer asks about their balance",
    "action": "check their current balance and provide the amount with currency",
    "score": 95,
    "rationale": "This guideline directly addresses balance inquiries with specific actions",
}


class GuidelineMatchDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_match_example},
):
    """A matched guideline."""

    guideline_id: GuidelineIdField
    condition: GuidelineMatchConditionField
    action: GuidelineMatchActionField
    score: GuidelineMatchScoreField
    rationale: GuidelineMatchRationaleField


ContextVariableIdPath: TypeAlias = Annotated[
    str,
    Path(
        description="Unique identifier for the context variable",
        examples=["var_123xyz"],
    ),
]

ContextVariableNameField: TypeAlias = Annotated[
    str,
    Field(
        description="The name of the context variable",
        examples=["user_preferences", "account_status"],
        min_length=1,
        max_length=100,
    ),
]

ContextVariableDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="The description text assigned to this variable",
        examples=["`c` counts the cost of the count cutting costs"],
    ),
]

ContextVariableKeyField: TypeAlias = Annotated[
    str,
    Field(
        description="This is the key which can be used to identify the variable",
        examples=["cool_variable_name", "melupapepkin"],
    ),
]

context_variable_and_value_example = {
    "id": "var_123xyz",
    "name": "AccountBalance",
    "description": "Customer's current account balance and currency",
    "key": "user_123",
    "value": {
        "balance": 5000.50,
        "currency": "USD",
        "last_updated": "2024-03-24T12:00:00Z",
    },
}


class ContextVariableAndValueDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_and_value_example},
):
    """A context variable and its current value."""

    id: ContextVariableIdPath
    name: ContextVariableNameField
    description: ContextVariableDescriptionField
    key: ContextVariableKeyField
    value: JSONSerializableDTO


UsageInfoInputTokensField: TypeAlias = Annotated[
    int,
    Field(
        description="Amount of token received from user over the session",
        examples=[256],
    ),
]

UsageInfoOutputTokensField: TypeAlias = Annotated[
    int,
    Field(
        description="Amount of token sent to user over the session",
        examples=[128],
    ),
]
usage_info_extra_example = {
    "prompt_tokens": 200,
    "completion_tokens": 128,
}

UsageInfoExtraField: TypeAlias = Annotated[
    Mapping[str, int],
    Field(
        description="Extra data associated with the usage information",
        examples=[usage_info_extra_example],
    ),
]

usage_info_example = {
    "input_tokens": 256,
    "output_tokens": 128,
    "extra": usage_info_extra_example,
}


class UsageInfoDTO(
    DefaultBaseModel,
    json_schema_extra={"example": usage_info_example},
):
    """Token usage information."""

    input_tokens: UsageInfoInputTokensField
    output_tokens: UsageInfoOutputTokensField
    extra: Optional[UsageInfoExtraField] = None


GenerationInfoSchemaNameField: TypeAlias = Annotated[
    str,
    Field(
        description="The name of the schema used for the generation",
        examples=["customer_response_v2"],
    ),
]

GenerationInfoModelField: TypeAlias = Annotated[
    str,
    Field(
        description="Id of the model used for the generation",
        examples=["gpt-4-turbo"],
    ),
]

GenerationInfoDurationField: TypeAlias = Annotated[
    float,
    Field(
        description="Amount of time spent generating",
        examples=[2.5],
    ),
]


generation_info_example = {
    "schema_name": "customer_response_v2",
    "model": "gpt-4-turbo",
    "duration": 2.5,
    "usage": usage_info_example,
}


class GenerationInfoDTO(
    DefaultBaseModel,
    json_schema_extra={"example": generation_info_example},
):
    """Information about a text generation."""

    schema_name: GenerationInfoSchemaNameField
    model: GenerationInfoModelField
    duration: GenerationInfoDurationField
    usage: UsageInfoDTO


MessageGenerationInspectionMessagesField: TypeAlias = Annotated[
    Sequence[str | None],
    Field(
        description="The messages that were generated",
    ),
]


MessageEventDataMessageField: TypeAlias = Annotated[
    str,
    Field(
        description="Text content of the message",
        examples=["Hello, I need help with my order"],
    ),
]

MessageEventDataFlaggedField: TypeAlias = Annotated[
    Optional[bool],
    Field(
        description="Indicates whether the message was flagged by moderation",
        examples=[True, False, None],
    ),
]

MessageEventDataTagsField: TypeAlias = Annotated[
    Optional[Sequence[str]],
    Field(
        description="Sequence of tags providing additional context about the message",
        examples=[["greeting", "urgent"], ["support-request"]],
    ),
]

MessageEventDataCannedResponsesField: TypeAlias = Annotated[
    Optional[Sequence[CannedResponseId]],
    Field(
        description="List of associated canned response references, if any",
        examples=[["frag_123xyz", "frag_789abc"]],
    ),
]

message_event_data_example = {
    "message": "Hello, I need help with my order",
    "participant": participant_example,
    "flagged": False,
    "tags": ["greeting", "help-request"],
    "canned_responses": ["frag_123xyz", "frag_789abc"],
}


class MessageEventDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": message_event_data_example},
):
    """
    DTO for data carried by a 'message' event.
    """

    message: MessageEventDataMessageField
    participant: ParticipantDTO
    flagged: MessageEventDataFlaggedField = None
    tags: MessageEventDataTagsField = None
    canned_responses: MessageEventDataCannedResponsesField = None


message_generation_inspection_example = {
    "generation": {
        "schema_name": "customer_response_v2",
        "model": "gpt-4-turbo",
        "duration": 2.5,
        "usage": {
            "input_tokens": 256,
            "output_tokens": 128,
            "extra": {"prompt_tokens": 200, "completion_tokens": 128},
        },
    },
    "messages": [
        message_event_data_example,
        None,
        {
            "message": "Based on your request, I can confirm that your order is being processed.",
            "participant": participant_example,
            "flagged": False,
            "tags": ["order-status"],
            "canned_responses": ["frag_987abc"],
        },
    ],
}


class MessageGenerationInspectionDTO(
    DefaultBaseModel,
    json_schema_extra={"example": message_generation_inspection_example},
):
    """Inspection data for message generation."""

    generations: Mapping[str, GenerationInfoDTO]
    messages: Sequence[Optional[str]]


GuidelineMatchingInspectionTotalDurationField: TypeAlias = Annotated[
    float,
    Field(
        description="Amount of time spent matching guidelines",
        examples=[3.5],
    ),
]


GuidelineMatchingInspectionBatchesField: TypeAlias = Annotated[
    Sequence[GenerationInfoDTO],
    Field(
        description="A list of `GenerationInfoDTO` describing the batches of generation executed",
    ),
]


guideline_matching_inspection_example = {
    "total_duration": 3.5,
    "batches": [generation_info_example],
}


class GuidelineMatchingInspectionDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_matching_inspection_example},
):
    """Inspection data for guideline matching."""

    total_duration: GuidelineMatchingInspectionTotalDurationField
    batches: GuidelineMatchingInspectionBatchesField


PreparationIterationGenerationsToolCallsField: TypeAlias = Annotated[
    Sequence[GenerationInfoDTO],
    Field(
        description="A list of `GenerationInfoDTO` describing the executed tool calls",
    ),
]

preparation_iteration_generations_example = {
    "guideline_matching": guideline_matching_inspection_example,
    "tool_calls": [generation_info_example],
}


class PreparationIterationGenerationsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": preparation_iteration_generations_example},
):
    """Generation information for a preparation iteration."""

    guideline_matching: GuidelineMatchingInspectionDTO
    tool_calls: PreparationIterationGenerationsToolCallsField


PreparationIterationGuidelineMatchField: TypeAlias = Annotated[
    Sequence[GuidelineMatchDTO],
    Field(
        description="List of guideline matches used in preparation for this iteration",
    ),
]


PreparationIterationToolCallsField: TypeAlias = Annotated[
    Sequence[ToolCallDTO],
    Field(
        description="List of tool calls made in preparation for this iteration",
    ),
]

term_example = {
    "id": "term_123xyz",
    "name": "balance",
    "description": "The current amount of money in an account",
    "synonyms": ["funds", "account balance", "available funds"],
}


class PreparationIterationTermDTO(
    DefaultBaseModel,
    json_schema_extra={"example": term_example},
):
    """A term participating in the preparation for an iteration."""

    id: TermIdPath
    name: TermNameField
    description: TermDescriptionField
    synonyms: TermSynonymsField


PreparationIterationTermsField: TypeAlias = Annotated[
    Sequence[PreparationIterationTermDTO],
    Field(
        description="List of terms participating in the preparation for this iteration",
    ),
]


PreparationIterationContextVariablesField: TypeAlias = Annotated[
    Sequence[ContextVariableAndValueDTO],
    Field(
        description="List of context variables (and their values) that participated in the preparation for this iteration",
    ),
]

preparation_iteration_example = {
    "generations": preparation_iteration_generations_example,
    "guideline_matches": [guideline_match_example],
    "tool_calls": [tool_call_example],
    "terms": [
        {
            "id": "term_123xyz",
            "name": "balance",
            "description": "The current amount of money in an account",
            "synonyms": ["funds", "account balance", "available funds"],
        }
    ],
    "context_variables": [context_variable_and_value_example],
}


class PreparationIterationDTO(
    DefaultBaseModel,
    json_schema_extra={"example": preparation_iteration_example},
):
    """Information about a preparation iteration."""

    generations: PreparationIterationGenerationsDTO
    guideline_matches: PreparationIterationGuidelineMatchField
    tool_calls: PreparationIterationToolCallsField
    terms: PreparationIterationTermsField
    context_variables: PreparationIterationContextVariablesField


EventTraceToolCallsField: TypeAlias = Annotated[
    Sequence[ToolCallDTO],
    Field(
        description="List of tool calls made for the traced event",
    ),
]

EventTraceMessageGenerationsField: TypeAlias = Annotated[
    Sequence[MessageGenerationInspectionDTO],
    Field(
        description="List of message generations made for the traced event",
    ),
]

EventTracePreparationIterationsField: TypeAlias = Annotated[
    Sequence[PreparationIterationDTO],
    Field(
        description="List of preparation iterations made for the traced event",
    ),
]

event_trace_example = {
    "tool_calls": [tool_call_example],
    "message_generations": [message_generation_inspection_example],
    "preparation_iterations": [preparation_iteration_example],
}


class EventTraceDTO(
    DefaultBaseModel,
    json_schema_extra={"example": event_trace_example},
):
    """Trace information for an event."""

    tool_calls: EventTraceToolCallsField
    message_generations: EventTraceMessageGenerationsField
    preparation_iterations: EventTracePreparationIterationsField


event_inspection_example = {
    "session_id": "sess_123yz",
    "event": event_example,
    "trace": event_trace_example,
}


class EventInspectionResult(
    DefaultBaseModel,
    json_schema_extra={"example": event_inspection_example},
):
    """Result of inspecting an event."""

    session_id: SessionIdPath
    event: EventDTO
    trace: Optional[EventTraceDTO] = None


def event_to_dto(event: Event) -> EventDTO:
    return EventDTO(
        id=event.id,
        source=_event_source_to_event_source_dto(event.source),
        kind=_event_kind_to_event_kind_dto(event.kind),
        offset=event.offset,
        creation_utc=event.creation_utc,
        correlation_id=event.correlation_id,
        data=cast(JSONSerializableDTO, event.data),
        deleted=event.deleted,
    )


def generation_info_to_dto(gi: GenerationInfo) -> GenerationInfoDTO:
    return GenerationInfoDTO(
        schema_name=gi.schema_name,
        model=gi.model,
        duration=gi.duration,
        usage=UsageInfoDTO(
            input_tokens=gi.usage.input_tokens,
            output_tokens=gi.usage.output_tokens,
            extra=gi.usage.extra,
        ),
    )


def participant_to_dto(participant: Participant) -> ParticipantDTO:
    return ParticipantDTO(
        id=participant["id"],
        display_name=participant["display_name"],
    )


def message_generation_inspection_to_dto(
    m: MessageGenerationInspection,
) -> MessageGenerationInspectionDTO:
    return MessageGenerationInspectionDTO(
        generations={
            name: generation_info_to_dto(generation) for name, generation in m.generations.items()
        },
        messages=[message for message in m.messages if message is not None],
    )


def preparation_iteration_to_dto(iteration: PreparationIteration) -> PreparationIterationDTO:
    return PreparationIterationDTO(
        generations=PreparationIterationGenerationsDTO(
            guideline_matching=GuidelineMatchingInspectionDTO(
                total_duration=iteration.generations.guideline_matching.total_duration,
                batches=[
                    generation_info_to_dto(generation)
                    for generation in iteration.generations.guideline_matching.batches
                ],
            ),
            tool_calls=[
                generation_info_to_dto(generation)
                for generation in iteration.generations.tool_calls
            ],
        ),
        guideline_matches=[
            GuidelineMatchDTO(
                guideline_id=match["guideline_id"],
                condition=match["condition"],
                action=match["action"],
                score=match["score"],
                rationale=match["rationale"],
            )
            for match in iteration.guideline_matches
        ],
        tool_calls=[
            ToolCallDTO(
                tool_id=tool_call["tool_id"],
                arguments=cast(Mapping[str, JSONSerializableDTO], tool_call["arguments"]),
                result=ToolResultDTO(
                    data=cast(JSONSerializableDTO, tool_call["result"]["data"]),
                    metadata=cast(
                        Mapping[str, JSONSerializableDTO], tool_call["result"]["metadata"]
                    ),
                ),
            )
            for tool_call in iteration.tool_calls
        ],
        terms=[
            PreparationIterationTermDTO(
                id=term["id"],
                name=term["name"],
                description=term["description"],
                synonyms=term["synonyms"],
            )
            for term in iteration.terms
        ],
        context_variables=[
            ContextVariableAndValueDTO(
                id=cv["id"],
                name=cv["name"],
                description=cv["description"] or "",
                key=cv["key"],
                value=cast(JSONSerializableDTO, cv["value"]),
            )
            for cv in iteration.context_variables
        ],
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

ModerationQuery: TypeAlias = Annotated[
    Moderation,
    Query(
        description="Content moderation level for the event",
    ),
]

MinOffsetQuery: TypeAlias = Annotated[
    int,
    Query(
        description="Only return events with offset >= this value",
        examples=[0, 42],
    ),
]

CorrelationIdQuery: TypeAlias = Annotated[
    str,
    Query(
        description="ID linking related events together",
        examples=["corr_13xyz"],
    ),
]

KindsQuery: TypeAlias = Annotated[
    str,
    Query(
        description="If set, only list events of the specified kinds (separated by commas)",
        examples=["message,tool", "message,status"],
    ),
]


def _get_jailbreak_moderation_service(logger: Logger) -> ModerationService:
    from parlant.adapters.nlp.lakera import LakeraGuard

    return LakeraGuard(logger)


def agent_message_guideline_dto_to_utterance_request(
    guideline: AgentMessageGuidelineDTO,
) -> UtteranceRequest:
    rationale_to_reason = {
        AgentMessageGuidelineRationaleDTO.UNSPECIFIED: UtteranceRationale.UNSPECIFIED,
        AgentMessageGuidelineRationaleDTO.BUY_TIME: UtteranceRationale.BUY_TIME,
        AgentMessageGuidelineRationaleDTO.FOLLOW_UP: UtteranceRationale.FOLLOW_UP,
    }

    return UtteranceRequest(
        action=guideline.action,
        rationale=rationale_to_reason[guideline.rationale],
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


def create_router(
    authorization_policy: AuthorizationPolicy,
    logger: Logger,
    application: Application,
    agent_store: AgentStore,
    customer_store: CustomerStore,
    session_store: SessionStore,
    session_listener: SessionListener,
    nlp_service: NLPService,
) -> APIRouter:
    router = APIRouter()

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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_session(
        request: Request,
        params: SessionCreationParamsDTO,
        allow_greeting: AllowGreetingQuery = False,
    ) -> SessionDTO:
        """Creates a new session between an agent and customer.

        The session will be initialized with the specified agent and optional customer.
        If no customer_id is provided, a guest customer will be created.
        """
        _ = await agent_store.read_agent(agent_id=params.agent_id)

        if params.customer_id:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_CUSTOMER_SESSION
            )

        else:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_GUEST_SESSION
            )

        session = await application.create_customer_session(
            customer_id=params.customer_id or CustomerStore.GUEST_ID,
            agent_id=params.agent_id,
            title=params.title,
            allow_greeting=allow_greeting,
        )

        return SessionDTO(
            id=session.id,
            agent_id=session.agent_id,
            customer_id=session.customer_id,
            creation_utc=session.creation_utc,
            consumption_offsets=ConsumptionOffsetsDTO(client=session.consumption_offsets["client"]),
            title=session.title,
            mode=SessionModeDTO(session.mode),
        )

    @router.get(
        "/{session_id}",
        operation_id="read_session",
        response_model=SessionDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Session details successfully retrieved",
                "content": {"application/json": {"example": session_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_session(
        request: Request,
        session_id: SessionIdPath,
    ) -> SessionDTO:
        """Retrieves details of a specific session by ID."""
        await authorization_policy.authorize(request=request, operation=Operation.READ_SESSION)

        session = await session_store.read_session(session_id=session_id)

        return SessionDTO(
            id=session.id,
            agent_id=session.agent_id,
            creation_utc=session.creation_utc,
            title=session.title,
            customer_id=session.customer_id,
            consumption_offsets=ConsumptionOffsetsDTO(
                client=session.consumption_offsets["client"],
            ),
            mode=SessionModeDTO(session.mode),
        )

    @router.get(
        "",
        operation_id="list_sessions",
        response_model=Sequence[SessionDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all matching sessions",
                "content": {"application/json": {"example": [session_example]}},
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_sessions(
        request: Request,
        agent_id: Optional[AgentIdQuery] = None,
        customer_id: Optional[CustomerIdQuery] = None,
    ) -> Sequence[SessionDTO]:
        """Lists all sessions matching the specified filters.

        Can filter by agent_id and/or customer_id. Returns all sessions if no
        filters are provided."""
        await authorization_policy.authorize(request=request, operation=Operation.LIST_SESSIONS)

        sessions = await session_store.list_sessions(
            agent_id=agent_id,
            customer_id=customer_id,
        )

        return [
            SessionDTO(
                id=s.id,
                agent_id=s.agent_id,
                creation_utc=s.creation_utc,
                title=s.title,
                customer_id=s.customer_id,
                consumption_offsets=ConsumptionOffsetsDTO(
                    client=s.consumption_offsets["client"],
                ),
                mode=SessionModeDTO(s.mode),
            )
            for s in sessions
        ]

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

        await session_store.read_session(session_id)
        await session_store.delete_session(session_id)

    @router.delete(
        "",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_sessions",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "All matching sessions successfully deleted"
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_many"),
    )
    async def delete_sessions(
        request: Request,
        agent_id: Optional[AgentIdQuery] = None,
        customer_id: Optional[CustomerIdQuery] = None,
    ) -> None:
        """Deletes all sessions matching the specified filters.

        Can filter by agent_id and/or customer_id. Will delete all sessions if no
        filters are provided."""
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_SESSIONS)

        sessions = await session_store.list_sessions(
            agent_id=agent_id,
            customer_id=customer_id,
        )

        for s in sessions:
            await session_store.delete_session(s.id)

    @router.patch(
        "/{session_id}",
        operation_id="update_session",
        responses={
            status.HTTP_200_OK: {"description": "Session successfully updated"},
            status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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

        async def from_dto(dto: SessionUpdateParamsDTO) -> SessionUpdateParams:
            params: SessionUpdateParams = {}

            if dto.consumption_offsets:
                session = await session_store.read_session(session_id)

                if dto.consumption_offsets.client:
                    params["consumption_offsets"] = {
                        **session.consumption_offsets,
                        "client": dto.consumption_offsets.client,
                    }

            if dto.title:
                params["title"] = dto.title

            if dto.mode:
                params["mode"] = dto.mode.value

            return params

        session = await session_store.update_session(
            session_id=session_id,
            params=await from_dto(params),
        )

        return SessionDTO(
            id=session.id,
            agent_id=session.agent_id,
            creation_utc=session.creation_utc,
            title=session.title,
            customer_id=session.customer_id,
            consumption_offsets=ConsumptionOffsetsDTO(
                client=session.consumption_offsets["client"],
            ),
            mode=SessionModeDTO(session.mode),
        )

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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in event parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create_event"),
    )
    async def create_event(
        request: Request,
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
        moderation: ModerationQuery = Moderation.NONE,
    ) -> EventDTO:
        """Creates a new event in the specified session.

        Currently supports creating message events from customer and human agent sources."""

        if params.kind == EventKindDTO.MESSAGE:
            if params.source == EventSourceDTO.CUSTOMER:
                await authorization_policy.authorize(
                    request=request, operation=Operation.CREATE_CUSTOMER_EVENT
                )
                return await _add_customer_message(session_id, params, moderation)
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
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail='Only "customer", "human_agent", and "human_agent_on_behalf_of_ai_agent" sources are supported for direct posting.',
                )

        elif params.kind == EventKindDTO.CUSTOM:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_CUSTOM_EVENT
            )
            return await _add_custom_event(session_id, params)

        elif params.kind == EventKindDTO.STATUS:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_STATUS_EVENT
            )
            return await _add_status_event(session_id, params)

        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail='Missing "status" field for status event',
            )

        raw_data = params.data or {}
        if not isinstance(raw_data, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail='Status event "data" must be a JSON object',
            )

        status_data: StatusEventData = {
            "status": status_dto_to_status(params.status),
            "data": raw_data,
        }

        event = await application.post_event(
            session_id=session_id,
            kind=_event_kind_dto_to_event_kind(params.kind),
            data=status_data,
            source=_event_source_dto_to_event_source(params.source),
            trigger_processing=False,
        )

        return event_to_dto(event)

    async def _add_customer_message(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
        moderation: Moderation = Moderation.NONE,
    ) -> EventDTO:
        if not params.message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing 'message' field for event",
            )

        flagged = False
        tags: Set[str] = set()

        if moderation in [Moderation.AUTO, Moderation.PARANOID]:
            moderation_service = await nlp_service.get_moderation_service()
            check = await moderation_service.check(params.message)
            flagged |= check.flagged
            tags.update(check.tags)

        if moderation == Moderation.PARANOID:
            check = await _get_jailbreak_moderation_service(logger).check(params.message)
            if "jailbreak" in check.tags:
                flagged = True
                tags.update({"jailbreak"})

        session = await session_store.read_session(session_id)

        try:
            customer = await customer_store.read_customer(session.customer_id)
            customer_display_name = customer.name
        except Exception:
            customer_display_name = session.customer_id

        message_data: MessageEventData = {
            "message": params.message,
            "participant": {
                "id": session.customer_id,
                "display_name": customer_display_name,
            },
            "flagged": flagged,
            "tags": list(tags),
        }

        event = await application.post_event(
            session_id=session_id,
            kind=_event_kind_dto_to_event_kind(params.kind),
            data=message_data,
            source=EventSource.CUSTOMER,
            trigger_processing=True,
        )

        return event_to_dto(event)

    async def _add_agent_message(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if params.message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="If you add an agent message, you cannot specify what the message will be, as it will be auto-generated by the agent.",
            )

        session = await session_store.read_session(session_id)

        if params.guidelines:
            requests = [
                agent_message_guideline_dto_to_utterance_request(a) for a in params.guidelines
            ]
            correlation_id = await application.utter(session, requests)
            event, *_ = await session_store.list_events(
                session_id=session_id,
                correlation_id=correlation_id,
                kinds=[EventKind.MESSAGE],
            )
            return event_to_dto(event)
        else:
            correlation_id = await application.dispatch_processing_task(session)

            await session_listener.wait_for_events(
                session_id=session_id,
                correlation_id=correlation_id,
                timeout=Timeout(60),
            )

            event = next(
                iter(
                    await session_store.list_events(
                        session_id=session_id,
                        correlation_id=correlation_id,
                        kinds=[EventKind.STATUS],
                    )
                )
            )

            return event_to_dto(event)

    async def _add_human_agent_message(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if not params.message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing 'message' field for event",
            )
        if not params.participant or not params.participant.display_name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing 'participant' with 'display_name' for human agent message",
            )

        message_data: MessageEventData = {
            "message": params.message,
            "participant": {
                "id": AgentId(params.participant.id) if params.participant.id else None,
                "display_name": params.participant.display_name,
            },
        }

        event = await application.post_event(
            session_id=session_id,
            kind=_event_kind_dto_to_event_kind(params.kind),
            data=message_data,
            source=EventSource.HUMAN_AGENT,
            trigger_processing=False,
        )

        return event_to_dto(event)

    async def _add_human_agent_message_on_behalf_of_ai_agent(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if not params.message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing 'data' field for message",
            )

        session = await session_store.read_session(session_id)
        agent = await agent_store.read_agent(session.agent_id)

        message_data: MessageEventData = {
            "message": params.message,
            "participant": {
                "id": agent.id,
                "display_name": agent.name,
            },
        }

        event = await application.post_event(
            session_id=session_id,
            kind=_event_kind_dto_to_event_kind(params.kind),
            data=message_data,
            source=EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
            trigger_processing=False,
        )

        return EventDTO(
            id=event.id,
            source=_event_source_to_event_source_dto(event.source),
            kind=_event_kind_to_event_kind_dto(event.kind),
            offset=event.offset,
            creation_utc=event.creation_utc,
            correlation_id=event.correlation_id,
            data=cast(JSONSerializableDTO, event.data),
            deleted=event.deleted,
        )

    async def _add_custom_event(
        session_id: SessionIdPath,
        params: EventCreationParamsDTO,
    ) -> EventDTO:
        if not params.data:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing 'data' field for custom event",
            )

        event = await application.post_event(
            session_id=session_id,
            kind=_event_kind_dto_to_event_kind(params.kind),
            data=params.data,
            source=_event_source_dto_to_event_source(params.source),
            trigger_processing=False,
        )

        return EventDTO(
            id=event.id,
            source=_event_source_to_event_source_dto(event.source),
            kind=_event_kind_to_event_kind_dto(event.kind),
            offset=event.offset,
            creation_utc=event.creation_utc,
            correlation_id=event.correlation_id,
            data=cast(JSONSerializableDTO, event.data),
            deleted=event.deleted,
        )

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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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
        min_offset: Optional[MinOffsetQuery] = None,
        source: Optional[EventSourceDTO] = None,
        correlation_id: Optional[CorrelationIdQuery] = None,
        kinds: Optional[KindsQuery] = None,
        wait_for_data: int = 60,
    ) -> Sequence[EventDTO]:
        """Lists events from a session with optional filtering and waiting capabilities.

        This endpoint retrieves events from a specified session and can:
        1. Filter events by their offset, source, type, and correlation ID
        2. Wait for new events to arrive if requested
        3. Return events in chronological order based on their offset

        Notes:
            Long Polling Behavior:
            - When wait_for_data = 0:
                Returns immediately with any existing events that match the criteria
            - When wait_for_data > 0:
                - If new matching events arrive within the timeout period, returns with those events
                - If no new events arrive before timeout, raises 504 Gateway Timeout
                - If matching events already exist, returns immediately with those events
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_EVENTS)

        kind_list: Sequence[EventKind] = [
            _event_kind_dto_to_event_kind(EventKindDTO(k))
            for k in (kinds.split(",") if kinds else [])
        ]

        if wait_for_data > 0:
            if not await session_listener.wait_for_events(
                session_id=session_id,
                min_offset=min_offset or 0,
                source=_event_source_dto_to_event_source(source) if source else None,
                kinds=kind_list,
                correlation_id=correlation_id,
                timeout=Timeout(wait_for_data),
            ):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request timed out",
                )

        events = await session_store.list_events(
            session_id=session_id,
            min_offset=min_offset,
            source=_event_source_dto_to_event_source(source) if source else None,
            kinds=kind_list,
            correlation_id=correlation_id,
        )

        return [
            EventDTO(
                id=e.id,
                source=_event_source_to_event_source_dto(e.source),
                kind=_event_kind_to_event_kind_dto(e.kind),
                offset=e.offset,
                creation_utc=e.creation_utc,
                correlation_id=e.correlation_id,
                data=cast(JSONSerializableDTO, e.data),
                deleted=e.deleted,
            )
            for e in events
        ]

    @router.delete(
        "/{session_id}/events",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_events",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Events successfully deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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

        session = await session_store.read_session(session_id)

        events = await session_store.list_events(
            session_id=session_id,
            min_offset=0,
            exclude_deleted=True,
        )

        events_starting_from_min_offset = [e for e in events if e.offset >= min_offset]

        if not events_starting_from_min_offset:
            return

        event_at_min_offset = events_starting_from_min_offset[0]

        first_event_of_correlation_id = next(
            e for e in events if e.correlation_id == event_at_min_offset.correlation_id
        )

        if event_at_min_offset.id != first_event_of_correlation_id.id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Cannot delete events with offset < min_offset unless they are the first event of their correlation ID",
            )

        for e in events_starting_from_min_offset:
            await session_store.delete_event(e.id)

        if not session.agent_states:
            return

        state_index_offset = next(
            i
            for i, s in enumerate(session.agent_states, start=0)
            if s.correlation_id.startswith(event_at_min_offset.correlation_id)
        )

        agent_states = session.agent_states[:state_index_offset]

        await session_store.update_session(
            session_id=session_id,
            params={"agent_states": agent_states},
        )

    async def _find_correlated_tool_calls(
        session_id: SessionIdPath,
        event: Event,
    ) -> Sequence[ToolCallDTO]:
        """Helper function to find tool calls correlated with an event."""

        tool_events = await session_store.list_events(
            session_id=session_id,
            kinds=[EventKind.TOOL],
            correlation_id=event.correlation_id,
        )

        tool_calls = list(
            chain.from_iterable(cast(ToolEventData, e.data)["tool_calls"] for e in tool_events)
        )

        return [
            ToolCallDTO(
                tool_id=tc["tool_id"],
                arguments=cast(Mapping[str, JSONSerializableDTO], tc["arguments"]),
                result=ToolResultDTO(
                    data=cast(JSONSerializableDTO, tc["result"]["data"]),
                    metadata=cast(Mapping[str, JSONSerializableDTO], tc["result"]["metadata"]),
                ),
            )
            for tc in tool_calls
        ]

    return router
