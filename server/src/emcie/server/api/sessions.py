from typing import Any, Optional, Union
from fastapi import APIRouter, HTTPException, Response, status
from datetime import datetime

from pydantic import Field

from emcie.server.async_utils import Timeout
from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.agents import AgentId
from emcie.server.core.end_users import EndUserId
from emcie.server.core.sessions import (
    EventId,
    EventKind,
    EventSource,
    SessionId,
    SessionListener,
    SessionStore,
)
from emcie.server.mc import MC


class CreateSessionRequest(DefaultBaseModel):
    end_user_id: EndUserId
    agent_id: AgentId
    title: Optional[str] = None


class CreateSessionResponse(DefaultBaseModel):
    session_id: SessionId
    title: Optional[str] = None


class CreateMessageRequest(DefaultBaseModel):
    kind: EventKind = Field("message", description='Internal (leave as "message")')
    content: str


class CreateEventResponse(DefaultBaseModel):
    event_id: EventId
    event_offset: int


class ConsumptionOffsetsDTO(DefaultBaseModel):
    client: int


class SessionDTO(DefaultBaseModel):
    session_id: SessionId
    end_user_id: EndUserId
    consumption_offsets: ConsumptionOffsetsDTO
    title: Optional[str] = None


class ConsumptionOffsetsPatchDTO(DefaultBaseModel):
    client: Optional[int] = None


class PatchSessionRequest(DefaultBaseModel):
    consumption_offsets: Optional[ConsumptionOffsetsPatchDTO] = None


class EventDTO(DefaultBaseModel):
    id: EventId
    source: EventSource
    kind: str
    offset: int
    creation_utc: datetime
    correlation_id: str
    data: Any


class ListEventsResponse(DefaultBaseModel):
    events: list[EventDTO]


class ListSessionsResponse(DefaultBaseModel):
    sessions: list[SessionDTO]


def create_router(
    mc: MC,
    session_store: SessionStore,
    session_listener: SessionListener,
) -> APIRouter:
    router = APIRouter()

    @router.post("/")
    async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
        session = await mc.create_end_user_session(
            end_user_id=request.end_user_id,
            agent_id=request.agent_id,
            title=request.title,
        )

        return CreateSessionResponse(session_id=session.id, title=session.title)

    @router.get("/{session_id}")
    async def read_session(session_id: SessionId) -> SessionDTO:
        session = await session_store.read_session(session_id=session_id)

        return SessionDTO(
            session_id=session.id,
            title=session.title,
            end_user_id=session.end_user_id,
            consumption_offsets=ConsumptionOffsetsDTO(
                client=session.consumption_offsets["client"],
            ),
        )

    @router.get("/")
    async def list_sessions(agent_id: Optional[AgentId] = None) -> ListSessionsResponse:
        sessions = await session_store.list_sessions(agent_id=agent_id)

        return ListSessionsResponse(
            sessions=[
                SessionDTO(
                    session_id=s.id,
                    title=s.title,
                    end_user_id=s.end_user_id,
                    consumption_offsets=ConsumptionOffsetsDTO(
                        client=s.consumption_offsets["client"],
                    ),
                )
                for s in sessions
            ]
        )

    @router.patch("/{session_id}")
    async def patch_session(
        session_id: SessionId,
        request: PatchSessionRequest,
    ) -> Response:
        if request.consumption_offsets:
            if request.consumption_offsets.client:
                session = await session_store.read_session(session_id)

                await mc.update_consumption_offset(
                    session=session,
                    new_offset=request.consumption_offsets.client,
                )

        return Response(content=None, status_code=status.HTTP_204_NO_CONTENT)

    @router.delete("/{session_id}")
    async def delete_session(
        session_id: SessionId,
    ) -> Response:
        await session_store.delete_session(session_id)
        return Response(content=None, status_code=status.HTTP_204_NO_CONTENT)

    @router.post("/{session_id}/events")
    async def create_event(
        session_id: SessionId,
        request: Union[CreateMessageRequest],
    ) -> CreateEventResponse:
        event = await mc.post_client_event(
            session_id=session_id,
            kind=request.kind,
            data={"message": request.content},
        )

        return CreateEventResponse(
            event_id=event.id,
            event_offset=event.offset,
        )

    @router.get("/{session_id}/events")
    async def list_events(
        session_id: SessionId,
        min_offset: Optional[int] = None,
        wait: Optional[bool] = None,
    ) -> ListEventsResponse:
        if wait:
            if not await session_listener.wait_for_events(
                session_id=session_id,
                min_offset=min_offset or 0,
                timeout=Timeout(60),
            ):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request timed out",
                )

        events = await session_store.list_events(
            session_id=session_id,
            source=None,
            min_offset=min_offset,
        )

        return ListEventsResponse(
            events=[
                EventDTO(
                    id=e.id,
                    source=e.source,
                    kind=e.kind,
                    offset=e.offset,
                    creation_utc=e.creation_utc,
                    correlation_id=e.correlation_id,
                    data=e.data,
                )
                for e in events
            ],
        )

    return router
