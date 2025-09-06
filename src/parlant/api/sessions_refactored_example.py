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
from pydantic import Field
from typing import Annotated, Optional, Sequence, TypeAlias, cast

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import ExampleJson, JSONSerializableDTO, apigen_config
from parlant.core.application import Application
from parlant.core.async_utils import Timeout
from parlant.core.common import DefaultBaseModel
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.engines.types import UtteranceRationale, UtteranceRequest
from parlant.core.sessions import (
    Event,
    EventId,
    EventKind,
    EventSource,
    SessionId,
    SessionStatus,
    SessionUpdateParams,
)
from parlant.core.agents import AgentId


class EventKindDTO(Enum):
    MESSAGE = "message"
    TOOL = "tool"
    STATUS = "status"
    CUSTOM = "custom"


class EventSourceDTO(Enum):
    CUSTOMER = "customer"
    CUSTOMER_UI = "customer_ui"
    HUMAN_AGENT = "human_agent"
    HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT = "human_agent_on_behalf_of_ai_agent"
    AI_AGENT = "ai_agent"
    SYSTEM = "system"


class SessionModeDTO(Enum):
    AUTO = "auto"
    MANUAL = "manual"


class SessionStatusDTO(Enum):
    ACKNOWLEDGED = "acknowledged"
    CANCELLED = "cancelled"
    PROCESSING = "processing"
    READY = "ready"
    TYPING = "typing"
    ERROR = "error"


class Moderation(Enum):
    AUTO = "auto"
    PARANOID = "paranoid"
    NONE = "none"


SessionIdPath: TypeAlias = Annotated[
    SessionId,
    Path(description="Unique identifier for the session", examples=["sess_123yz"]),
]

SessionAgentIdPath: TypeAlias = Annotated[
    AgentId,
    Path(description="Unique identifier for the agent", examples=["ag-123Txyz"]),
]

SessionCustomerIdField: TypeAlias = Annotated[
    CustomerId,
    Field(description="ID of the customer", examples=["cust_123xy"]),
]

SessionTitleField: TypeAlias = Annotated[
    str,
    Field(description="Descriptive title for the session", max_length=200),
]

SessionModeField: TypeAlias = Annotated[
    SessionModeDTO,
    Field(description="The mode of the session, either 'auto' or 'manual'"),
]


class ConsumptionOffsetsDTO(DefaultBaseModel):
    client: Optional[int] = None


class SessionDTO(DefaultBaseModel):
    id: SessionIdPath
    agent_id: SessionAgentIdPath
    customer_id: SessionCustomerIdField
    creation_utc: datetime
    title: Optional[SessionTitleField] = None
    mode: SessionModeField
    consumption_offsets: ConsumptionOffsetsDTO


class SessionCreationParamsDTO(DefaultBaseModel):
    agent_id: SessionAgentIdPath
    customer_id: Optional[CustomerId] = None
    title: Optional[SessionTitleField] = None


class EventCreationParamsDTO(DefaultBaseModel):
    kind: EventKindDTO
    source: EventSourceDTO
    message: Optional[str] = None
    data: Optional[JSONSerializableDTO] = None
    status: Optional[SessionStatusDTO] = None


class EventDTO(DefaultBaseModel):
    id: EventId
    source: EventSourceDTO
    kind: EventKindDTO
    offset: int
    creation_utc: datetime
    correlation_id: str
    data: JSONSerializableDTO
    deleted: bool


class SessionUpdateParamsDTO(DefaultBaseModel):
    consumption_offsets: Optional[ConsumptionOffsetsDTO] = None
    title: Optional[SessionTitleField] = None
    mode: Optional[SessionModeField] = None


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


def _event_kind_dto_to_event_kind(dto: EventKindDTO) -> EventKind:
    mapping = {
        EventKindDTO.MESSAGE: EventKind.MESSAGE,
        EventKindDTO.TOOL: EventKind.TOOL,
        EventKindDTO.STATUS: EventKind.STATUS,
        EventKindDTO.CUSTOM: EventKind.CUSTOM,
    }
    if kind := mapping.get(dto):
        return kind
    raise ValueError(f"Invalid event kind: {dto}")


def _event_kind_to_event_kind_dto(kind: EventKind) -> EventKindDTO:
    mapping = {
        EventKind.MESSAGE: EventKindDTO.MESSAGE,
        EventKind.TOOL: EventKindDTO.TOOL,
        EventKind.STATUS: EventKindDTO.STATUS,
        EventKind.CUSTOM: EventKindDTO.CUSTOM,
    }
    if dto := mapping.get(kind):
        return dto
    raise ValueError(f"Invalid event kind: {kind}")


def _event_source_dto_to_event_source(dto: EventSourceDTO) -> EventSource:
    mapping = {
        EventSourceDTO.CUSTOMER: EventSource.CUSTOMER,
        EventSourceDTO.CUSTOMER_UI: EventSource.CUSTOMER_UI,
        EventSourceDTO.HUMAN_AGENT: EventSource.HUMAN_AGENT,
        EventSourceDTO.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
        EventSourceDTO.AI_AGENT: EventSource.AI_AGENT,
        EventSourceDTO.SYSTEM: EventSource.SYSTEM,
    }
    if source := mapping.get(dto):
        return source
    raise ValueError(f"Invalid event source: {dto}")


def _event_source_to_event_source_dto(source: EventSource) -> EventSourceDTO:
    mapping = {
        EventSource.CUSTOMER: EventSourceDTO.CUSTOMER,
        EventSource.CUSTOMER_UI: EventSourceDTO.CUSTOMER_UI,
        EventSource.HUMAN_AGENT: EventSourceDTO.HUMAN_AGENT,
        EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: EventSourceDTO.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
        EventSource.AI_AGENT: EventSourceDTO.AI_AGENT,
        EventSource.SYSTEM: EventSourceDTO.SYSTEM,
    }
    if dto := mapping.get(source):
        return dto
    raise ValueError(f"Invalid event source: {source}")


def create_router(
    authorization_policy: AuthorizationPolicy,
    application: Application,
) -> APIRouter:
    """
    REFACTORED VERSION: Notice how this router now only takes:
    - authorization_policy (for HTTP-level concerns)
    - application (single entry point to business logic)
    
    No more direct core dependencies!
    """
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_session",
        response_model=SessionDTO,
    )
    async def create_session(
        request: Request,
        params: SessionCreationParamsDTO,
        allow_greeting: bool = Query(False),
    ) -> SessionDTO:
        """Creates a new session between an agent and customer."""
        
        # Authorization (HTTP-level concern)
        if params.customer_id:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_CUSTOMER_SESSION
            )
        else:
            await authorization_policy.authorize(
                request=request, operation=Operation.CREATE_GUEST_SESSION
            )

        # Business logic delegated to app module
        session = await application.sessions.create_session(
            agent_id=params.agent_id,
            customer_id=params.customer_id,
            title=params.title,
            allow_greeting=allow_greeting,
        )

        # DTO conversion (API-level concern)
        return SessionDTO(
            id=session.id,
            agent_id=session.agent_id,
            customer_id=session.customer_id,
            creation_utc=session.creation_utc,
            consumption_offsets=ConsumptionOffsetsDTO(
                client=session.consumption_offsets["client"]
            ),
            title=session.title,
            mode=SessionModeDTO(session.mode),
        )

    @router.get("/{session_id}", response_model=SessionDTO)
    async def read_session(
        request: Request,
        session_id: SessionIdPath,
    ) -> SessionDTO:
        """Retrieves details of a specific session by ID."""
        await authorization_policy.authorize(request=request, operation=Operation.READ_SESSION)

        # Business logic delegated to app module
        session = await application.sessions.get_session(session_id=session_id)

        # DTO conversion
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

    @router.get("", response_model=Sequence[SessionDTO])
    async def list_sessions(
        request: Request,
        agent_id: Optional[AgentId] = Query(None),
        customer_id: Optional[CustomerId] = Query(None),
    ) -> Sequence[SessionDTO]:
        """Lists all sessions matching the specified filters."""
        await authorization_policy.authorize(request=request, operation=Operation.LIST_SESSIONS)

        # Business logic delegated to app module
        sessions = await application.sessions.list_sessions(
            agent_id=agent_id,
            customer_id=customer_id,
        )

        # DTO conversion
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

    @router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_session(
        request: Request,
        session_id: SessionIdPath,
    ) -> None:
        """Deletes a session and all its associated events."""
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_SESSION)

        # Business logic delegated to app module
        await application.sessions.delete_session(session_id)

    @router.patch("/{session_id}", response_model=SessionDTO)
    async def update_session(
        request: Request,
        session_id: SessionIdPath,
        params: SessionUpdateParamsDTO,
    ) -> SessionDTO:
        """Updates an existing session's attributes."""
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_SESSION)

        # Convert DTO to core params
        update_params: SessionUpdateParams = {}
        if params.consumption_offsets and params.consumption_offsets.client:
            session = await application.sessions.get_session(session_id)
            update_params["consumption_offsets"] = {
                **session.consumption_offsets,
                "client": params.consumption_offsets.client,
            }
        if params.title:
            update_params["title"] = params.title
        if params.mode:
            update_params["mode"] = params.mode.value

        # Business logic delegated to app module
        session = await application.sessions.update_session(
            session_id=session_id,
            params=update_params,
        )

        # DTO conversion
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

    return router
