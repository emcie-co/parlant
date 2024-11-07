from datetime import datetime
from enum import Enum
from itertools import chain, groupby
from typing import Any, Literal, Mapping, Optional, cast

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import Field

from parlant.api.glossary import TermDTO
from parlant.core.async_utils import Timeout
from parlant.core.common import DefaultBaseModel
from parlant.core.context_variables import ContextVariableId
from parlant.core.agents import AgentId, AgentStore
from parlant.core.end_users import EndUserId, EndUserStore
from parlant.core.guidelines import GuidelineId
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import (
    EventId,
    EventKind,
    MessageEventData,
    SessionId,
    SessionListener,
    SessionStore,
    SessionUpdateParams,
    ToolEventData,
)

from parlant.core.application import Application


class EventKindDTO(Enum):
    MESSAGE = "message"
    TOOL = "tool"
    STATUS = "status"
    CUSTOM = "custom"


class EventSourceDTO(Enum):
    END_USER = "end_user"
    END_USER_UI = "end_user_ui"
    HUMAN_AGENT = "human_agent"
    HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT = "human_agent_on_behalf_of_ai_agent"
    AI_AGENT = "ai_agent"


class ConsumptionOffsetsDTO(DefaultBaseModel):
    client: int


class SessionDTO(DefaultBaseModel):
    id: SessionId
    agent_id: AgentId
    end_user_id: EndUserId
    creation_utc: datetime
    title: Optional[str] = None
    consumption_offsets: ConsumptionOffsetsDTO


class CreateSessionRequest(DefaultBaseModel):
    end_user_id: EndUserId
    agent_id: AgentId
    title: Optional[str] = None


class CreateSessionResponse(DefaultBaseModel):
    session: SessionDTO


class CreateEventRequest(DefaultBaseModel):
    kind: EventKindDTO
    source: EventSourceDTO
    content: str


class EventDTO(DefaultBaseModel):
    id: EventId
    source: EventSourceDTO
    kind: str
    offset: int
    creation_utc: datetime
    correlation_id: str
    data: Any


class CreateEventResponse(DefaultBaseModel):
    event: EventDTO


class ConsumptionOffsetsPatchDTO(DefaultBaseModel):
    client: Optional[int] = None


class PatchSessionRequest(DefaultBaseModel):
    consumption_offsets: Optional[ConsumptionOffsetsPatchDTO] = None
    title: Optional[str] = None


class ListEventsResponse(DefaultBaseModel):
    session_id: SessionId
    events: list[EventDTO]


class ListSessionsResponse(DefaultBaseModel):
    sessions: list[SessionDTO]


class DeleteSessionResponse(DefaultBaseModel):
    session_id: SessionId


class ToolResultDTO(DefaultBaseModel):
    data: Any
    metadata: Mapping[str, Any]


class ToolCallDTO(DefaultBaseModel):
    tool_id: str
    arguments: Mapping[str, Any]
    result: ToolResultDTO


class InteractionDTO(DefaultBaseModel):
    kind: Literal["message"]
    source: EventSourceDTO
    correlation_id: str
    data: Any = Field(
        description="The data associated with this interaction's kind. "
        "If kind is 'message', this is the message string."
    )
    tool_calls: list[ToolCallDTO]


class ListInteractionsResponse(DefaultBaseModel):
    session_id: SessionId
    interactions: list[InteractionDTO]


class DeleteEventsResponse(DefaultBaseModel):
    event_ids: list[EventId]


class GuidelinePropositionDTO(DefaultBaseModel):
    guideline_id: GuidelineId
    predicate: str
    action: str
    score: int
    rationale: str


class ContextVariableAndValueDTO(DefaultBaseModel):
    id: ContextVariableId
    name: str
    description: str
    key: str
    value: Any


class PreparationIterationDTO(DefaultBaseModel):
    guideline_propositions: list[GuidelinePropositionDTO]
    tool_calls: list[ToolCallDTO]
    terms: list[TermDTO]
    context_variables: list[ContextVariableAndValueDTO]


class ReadInteractionResponse(DefaultBaseModel):
    session_id: SessionId
    correlation_id: str
    preparation_iterations: list[PreparationIterationDTO]


def create_router(
    application: Application,
    agent_store: AgentStore,
    end_user_store: EndUserStore,
    session_store: SessionStore,
    session_listener: SessionListener,
    service_registry: ServiceRegistry,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_session",
    )
    async def create_session(
        request: CreateSessionRequest,
        allow_greeting: bool = Query(default=True),
    ) -> CreateSessionResponse:
        _ = await agent_store.read_agent(agent_id=request.agent_id)

        session = await application.create_end_user_session(
            end_user_id=request.end_user_id,
            agent_id=request.agent_id,
            title=request.title,
            allow_greeting=allow_greeting,
        )

        return CreateSessionResponse(
            session=SessionDTO(
                id=session.id,
                agent_id=session.agent_id,
                end_user_id=session.end_user_id,
                creation_utc=session.creation_utc,
                consumption_offsets=ConsumptionOffsetsDTO(
                    client=session.consumption_offsets["client"]
                ),
                title=session.title,
            )
        )

    @router.get(
        "/{session_id}",
        operation_id="read_session",
    )
    async def read_session(session_id: SessionId) -> SessionDTO:
        session = await session_store.read_session(session_id=session_id)

        return SessionDTO(
            id=session.id,
            agent_id=session.agent_id,
            creation_utc=session.creation_utc,
            title=session.title,
            end_user_id=session.end_user_id,
            consumption_offsets=ConsumptionOffsetsDTO(
                client=session.consumption_offsets["client"],
            ),
        )

    @router.get(
        "/",
        operation_id="list_sessions",
    )
    async def list_sessions(
        agent_id: Optional[AgentId] = None,
        end_user_id: Optional[EndUserId] = None,
    ) -> ListSessionsResponse:
        sessions = await session_store.list_sessions(
            agent_id=agent_id,
            end_user_id=end_user_id,
        )

        return ListSessionsResponse(
            sessions=[
                SessionDTO(
                    id=s.id,
                    agent_id=s.agent_id,
                    creation_utc=s.creation_utc,
                    title=s.title,
                    end_user_id=s.end_user_id,
                    consumption_offsets=ConsumptionOffsetsDTO(
                        client=s.consumption_offsets["client"],
                    ),
                )
                for s in sessions
            ]
        )

    @router.delete(
        "/",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_sessions",
    )
    async def delete_sessions(
        agent_id: Optional[AgentId] = None,
        end_user_id: Optional[EndUserId] = None,
    ) -> None:
        sessions = await session_store.list_sessions(
            agent_id=agent_id,
            end_user_id=end_user_id,
        )

        for s in sessions:
            await session_store.delete_session(s.id)

    @router.patch(
        "/{session_id}",
        operation_id="patch_session",
    )
    async def patch_session(
        session_id: SessionId,
        request: PatchSessionRequest,
    ) -> Response:
        params: SessionUpdateParams = {}

        if request.consumption_offsets:
            session = await session_store.read_session(session_id)

            if request.consumption_offsets.client:
                params["consumption_offsets"] = {
                    **session.consumption_offsets,
                    "client": request.consumption_offsets.client,
                }

        if request.title:
            params["title"] = request.title

        await session_store.update_session(session_id=session_id, params=params)

        return Response(content=None, status_code=status.HTTP_204_NO_CONTENT)

    @router.delete(
        "/{session_id}",
        operation_id="delete_session",
    )
    async def delete_session(
        session_id: SessionId,
    ) -> DeleteSessionResponse:
        if await session_store.delete_session(session_id):
            return DeleteSessionResponse(session_id=session_id)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    @router.post(
        "/{session_id}/events",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_event",
    )
    async def create_event(
        session_id: SessionId,
        request: CreateEventRequest,
        moderation: Literal["none", "auto"] = "none",
    ) -> CreateEventResponse:
        if request.source == EventSourceDTO.END_USER:
            return await _add_end_user_message(session_id, request, moderation)
        elif request.source == EventSourceDTO.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT:
            return await _add_agent_message(session_id, request)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail='Only "end_user" and "human_agent_on_behalf_of_ai_agent" sources are supported for direct posting.',
            )

    async def _add_end_user_message(
        session_id: SessionId,
        request: CreateEventRequest,
        moderation: Literal["none", "auto"] = "none",
    ) -> CreateEventResponse:
        flagged = False
        tags = set()

        if moderation == "auto":
            for _, moderation_service in await service_registry.list_moderation_services():
                check = await moderation_service.check(request.content)
                flagged |= check.flagged
                tags.update(check.tags)

        session = await session_store.read_session(session_id)

        try:
            end_user = await end_user_store.read_end_user(session.end_user_id)
            end_user_display_name = end_user.name
        except Exception:
            end_user_display_name = session.end_user_id

        message_data: MessageEventData = {
            "message": request.content,
            "participant": {
                "id": session.end_user_id,
                "display_name": end_user_display_name,
            },
            "flagged": flagged,
            "tags": list(tags),
        }

        event = await application.post_event(
            session_id=session_id,
            kind=request.kind.value,
            data=message_data,
            source="end_user",
            trigger_processing=True,
        )

        return CreateEventResponse(
            event=EventDTO(
                id=event.id,
                source=EventSourceDTO(event.source),
                kind=event.kind,
                offset=event.offset,
                creation_utc=event.creation_utc,
                correlation_id=event.correlation_id,
                data=event.data,
            )
        )

    async def _add_agent_message(
        session_id: SessionId,
        request: CreateEventRequest,
    ) -> CreateEventResponse:
        session = await session_store.read_session(session_id)
        agent = await agent_store.read_agent(session.agent_id)

        message_data: MessageEventData = {
            "message": request.content,
            "participant": {
                "id": agent.id,
                "display_name": agent.name,
            },
        }

        event = await application.post_event(
            session_id=session_id,
            kind=request.kind.value,
            data=message_data,
            source="human_agent_on_behalf_of_ai_agent",
            trigger_processing=False,
        )

        return CreateEventResponse(
            event=EventDTO(
                id=event.id,
                source=EventSourceDTO(event.source),
                kind=event.kind,
                offset=event.offset,
                creation_utc=event.creation_utc,
                correlation_id=event.correlation_id,
                data=event.data,
            )
        )

    @router.get(
        "/{session_id}/events",
        operation_id="list_events",
    )
    async def list_events(
        session_id: SessionId,
        min_offset: Optional[int] = None,
        kinds: Optional[str] = Query(
            default=None,
            description="If set, only list events of the specified kinds (separated by commas)",
        ),
        wait: Optional[bool] = None,
    ) -> ListEventsResponse:
        kind_list: list[EventKind] = kinds.split(",") if kinds else []  # type: ignore
        assert all(k in EventKind.__args__ for k in kind_list)  # type: ignore

        if wait:
            if not await session_listener.wait_for_events(
                session_id=session_id,
                min_offset=min_offset or 0,
                kinds=kind_list,
                timeout=Timeout(60),
            ):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request timed out",
                )

        events = await session_store.list_events(
            session_id=session_id,
            source=None,
            kinds=kind_list,
            min_offset=min_offset,
        )

        return ListEventsResponse(
            session_id=session_id,
            events=[
                EventDTO(
                    id=e.id,
                    source=EventSourceDTO(e.source),
                    kind=e.kind,
                    offset=e.offset,
                    creation_utc=e.creation_utc,
                    correlation_id=e.correlation_id,
                    data=e.data,
                )
                for e in events
            ],
        )

    @router.get(
        "/{session_id}/interactions",
        operation_id="list_interactions",
    )
    async def list_interactions(
        session_id: SessionId,
        min_event_offset: int,
        source: EventSourceDTO,
        wait: bool = False,
    ) -> ListInteractionsResponse:
        if wait:
            await application.wait_for_update(
                session_id=session_id,
                min_offset=min_event_offset,
                kinds=["message"],
                source=source.value,
                timeout=Timeout(300),
            )

        events = await session_store.list_events(
            session_id=session_id,
            kinds=["message", "tool"],
            source=source.value,
            min_offset=min_event_offset,
        )

        interactions = []

        for correlation_id, correlated_events in groupby(events, key=lambda e: e.correlation_id):
            message_events = [e for e in correlated_events if e.kind == "message"]
            tool_events = [e for e in correlated_events if e.kind == "tool"]

            tool_calls = list(
                chain.from_iterable(cast(ToolEventData, e.data)["tool_calls"] for e in tool_events)
            )

            for e in message_events:
                interactions.append(
                    InteractionDTO(
                        kind="message",
                        source=EventSourceDTO(e.source),
                        correlation_id=correlation_id,
                        data=cast(MessageEventData, e.data)["message"],
                        tool_calls=[
                            ToolCallDTO(
                                tool_id=tc["tool_id"],
                                arguments=tc["arguments"],
                                result=ToolResultDTO(
                                    data=tc["result"]["data"],
                                    metadata=tc["result"]["metadata"],
                                ),
                            )
                            for tc in tool_calls
                        ],
                    )
                )

        return ListInteractionsResponse(
            session_id=session_id,
            interactions=interactions,
        )

    @router.delete(
        "/{session_id}/events",
        operation_id="delete_events",
    )
    async def delete_events(
        session_id: SessionId,
        min_offset: int,
    ) -> DeleteEventsResponse:
        events = await session_store.list_events(
            session_id=session_id,
            min_offset=min_offset,
            exclude_deleted=True,
        )

        deleted_event_ids = [await session_store.delete_event(e.id) for e in events]

        return DeleteEventsResponse(event_ids=[id for id in deleted_event_ids if id is not None])

    @router.get(
        "/{session_id}/interactions/{correlation_id}",
        operation_id="read_interaction",
    )
    async def read_interaction(
        session_id: SessionId,
        correlation_id: str,
    ) -> ReadInteractionResponse:
        inspection = await session_store.read_inspection(
            session_id=session_id,
            correlation_id=correlation_id,
        )

        return ReadInteractionResponse(
            session_id=session_id,
            correlation_id=correlation_id,
            preparation_iterations=[
                PreparationIterationDTO(
                    guideline_propositions=[
                        GuidelinePropositionDTO(
                            guideline_id=proposition["guideline_id"],
                            predicate=proposition["predicate"],
                            action=proposition["action"],
                            score=proposition["score"],
                            rationale=proposition["rationale"],
                        )
                        for proposition in iteration.guideline_propositions
                    ],
                    tool_calls=[
                        ToolCallDTO(
                            tool_id=tool_call["tool_id"],
                            arguments=tool_call["arguments"],
                            result=ToolResultDTO(
                                data=tool_call["result"]["data"],
                                metadata=tool_call["result"]["metadata"],
                            ),
                        )
                        for tool_call in iteration.tool_calls
                    ],
                    terms=[
                        TermDTO(
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
                            value=cv["value"],
                        )
                        for cv in iteration.context_variables
                    ],
                )
                for iteration in inspection.preparation_iterations
            ],
        )

    return router
