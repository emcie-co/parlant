import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence, Set, cast

from parlant.core.agents import AgentId, AgentStore
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.async_utils import Timeout
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import JSONSerializable
from parlant.core.health import HealthReporter
from parlant.core.meter import Meter
from parlant.core.persistence.common import Cursor, SortDirection
from parlant.core.tracer import Tracer
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.emissions import EventEmitterFactory
from parlant.core.engines.types import Context, Engine, EngineRegistry, UtteranceRequest
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import CustomerModerationContext, ModerationService
from parlant.core.nlp.service import NLPService
from parlant.core.sessions import (
    AgentState,
    ConsumerId,
    Event,
    EventId,
    EventKind,
    EventSource,
    EventUpdateParams,
    MessageEventData,
    Participant,
    Session,
    SessionId,
    SessionListener,
    SessionMode,
    SessionStatus,
    SessionStore,
    StatusEventData,
)
from parlant.core.store_provider import StoreProvider, StoreProviderHints
from dataclasses import dataclass
from typing_extensions import TypedDict


class SessionUpdateParamsModel(TypedDict, total=False):
    """Parameters for updating a session."""

    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: str | None
    consumption_offsets: Mapping[ConsumerId, int]
    agent_states: Sequence[AgentState]
    metadata: Mapping[str, JSONSerializable]


class EventMetadataUpdateParamsModel(TypedDict, total=False):
    """Parameters for updating event metadata with granular control."""

    set: Mapping[str, JSONSerializable]
    unset: Sequence[str]


class EventUpdateParamsModel(TypedDict, total=False):
    """Parameters for updating an event."""

    metadata: EventMetadataUpdateParamsModel


@dataclass(frozen=True)
class SessionLabelsUpdateParams:
    """Parameters for updating session labels."""

    upsert: Set[str] | None = None
    remove: Set[str] | None = None


@dataclass(frozen=True)
class SessionListingModel:
    """Paginated result model for sessions at the application layer"""

    items: Sequence[Session]
    total_count: int
    has_more: bool
    next_cursor: Cursor | None = None


@dataclass(frozen=True)
class _ActiveProcessingTask:
    trace_id: str
    tag: str


class Moderation(Enum):
    """Content moderation settings."""

    AUTO = "auto"
    PARANOID = "paranoid"
    NONE = "none"


def _get_jailbreak_moderation_service(
    logger: Logger,
    meter: Meter,
    health_reporter: HealthReporter,
) -> ModerationService:
    from parlant.adapters.nlp.lakera import LakeraGuard

    return LakeraGuard(logger, meter, health_reporter)


class SessionModule:
    def __init__(
        self,
        request_context: RequestContext,
        logger: Logger,
        meter: Meter,
        tracer: Tracer,
        session_listener: SessionListener,
        nlp_service: NLPService,
        engine_registry: EngineRegistry,
        event_emitter_factory: EventEmitterFactory,
        background_task_service: BackgroundTaskService,
        health_reporter: HealthReporter,
        store_provider: StoreProvider,
    ):
        self._request_context = request_context
        self._logger = logger
        self._store_provider = store_provider
        self._meter = meter
        self._tracer = tracer
        self._session_listener = session_listener
        self._nlp_service = nlp_service
        self._health_reporter = health_reporter

        self._engine_registry = engine_registry
        self._event_emitter_factory = event_emitter_factory
        self._background_task_service = background_task_service

        self._lock = asyncio.Lock()
        self._active_processing_tasks: dict[SessionId, _ActiveProcessingTask] = {}

    @property
    def _agent_store(self) -> AgentStore:
        return self._store_provider.get_store(
            AgentStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    @property
    def _session_store(self) -> SessionStore:
        return self._store_provider.get_store(
            SessionStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    @property
    def _customer_store(self) -> CustomerStore:
        return self._store_provider.get_store(
            CustomerStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    async def wait_for_more_events(
        self,
        session_id: SessionId,
        min_offset: int,
        kinds: Sequence[EventKind] = [],
        source: EventSource | None = None,
        trace_id: str | None = None,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        return await self._session_listener.wait_for_more_events(
            session_id=session_id,
            min_offset=min_offset,
            kinds=kinds,
            source=source,
            trace_id=trace_id,
            timeout=timeout,
        )

    async def wait_for_event_completion(
        self,
        session_id: SessionId,
        event_id: EventId,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        return await self._session_listener.wait_for_event_completion(
            session_id=session_id,
            event_id=event_id,
            timeout=timeout,
        )

    async def wait_for_new_streaming_chunks(
        self,
        session_id: SessionId,
        event_id: EventId,
        last_known_chunk_count: int,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        return await self._session_listener.wait_for_new_streaming_chunks(
            session_id=session_id,
            event_id=event_id,
            last_known_chunk_count=last_known_chunk_count,
            timeout=timeout,
        )

    async def create(
        self,
        customer_id: CustomerId,
        agent_id: AgentId,
        title: str | None = None,
        allow_greeting: bool = False,
        metadata: Mapping[str, JSONSerializable] | None = None,
        labels: Set[str] | None = None,
    ) -> Session:
        agent = await self._agent_store.read_agent(agent_id=agent_id)

        session = await self._session_store.create_session(
            creation_utc=datetime.now(timezone.utc),
            customer_id=customer_id,
            agent_id=agent_id,
            title=title,
            metadata=metadata or {},
            labels=labels,
        )

        # Warm the engine's caches in the background as soon as the session
        # exists, so the first response reads them instead of building them.
        await self._initialize_session(session)

        # TODO: For now, Compass engine doesn't support dynamic greetings.
        # User the utter() method manually after creation to send a greeting, if needed.
        if allow_greeting and agent.engine != "compass":
            await self.dispatch_processing_task(session)

        return session

    async def read(self, session_id: SessionId) -> Session:
        session = await self._session_store.read_session(session_id=session_id)
        return session

    async def find(
        self,
        agent_id: AgentId | None,
        customer_id: CustomerId | None,
        limit: int | None = None,
        cursor: Cursor | None = None,
        sort_direction: SortDirection | None = None,
        labels: Set[str] | None = None,
        min_modified_utc: datetime | None = None,
    ) -> SessionListingModel:
        result = await self._session_store.list_sessions(
            agent_id=agent_id,
            customer_id=customer_id,
            limit=limit,
            cursor=cursor,
            sort_direction=sort_direction,
            labels=labels,
            min_modified_utc=min_modified_utc,
        )

        return SessionListingModel(
            items=result.items,
            total_count=result.total_count,
            has_more=result.has_more,
            next_cursor=result.next_cursor,
        )

    async def update(
        self,
        session_id: SessionId,
        params: SessionUpdateParamsModel,
        labels: SessionLabelsUpdateParams | None = None,
    ) -> Session:
        session = (
            await self._session_store.update_session(
                session_id=session_id,
                params=params,
            )
            if params
            else await self._session_store.read_session(session_id)
        )

        if labels:
            if labels.upsert:
                session = await self._session_store.upsert_labels(
                    session_id=session_id,
                    labels=labels.upsert,
                )

            if labels.remove:
                session = await self._session_store.remove_labels(
                    session_id=session_id,
                    labels=labels.remove,
                )

        return session

    async def delete(
        self,
        session_id: SessionId,
    ) -> None:
        await self._session_store.read_session(session_id)
        await self._session_store.delete_session(session_id)

    async def create_event(
        self,
        session_id: SessionId,
        kind: EventKind,
        data: Mapping[str, Any],
        metadata: Mapping[str, JSONSerializable] | None,
        source: EventSource = EventSource.CUSTOMER,
        trigger_processing: bool = True,
    ) -> Event:
        with self._tracer.span(
            "session.create_event",
            {
                "session_id": session_id,
                "event.kind": kind.value,
                "event.source": source.value,
            },
        ):
            return await self._create_event(
                session_id=session_id,
                kind=kind,
                data=data,
                metadata=metadata,
                source=source,
                trigger_processing=trigger_processing,
            )

    async def _create_event(
        self,
        session_id: SessionId,
        kind: EventKind,
        data: Mapping[str, Any],
        metadata: Mapping[str, JSONSerializable] | None,
        source: EventSource,
        trigger_processing: bool,
    ) -> Event:
        event = await self._session_store.create_event(
            session_id=session_id,
            source=source,
            kind=kind,
            trace_id=self._tracer.trace_id,
            data=data,
            metadata=metadata or {},
        )

        if trigger_processing:
            session = await self._session_store.read_session(session_id)
            await self.dispatch_processing_task(session)

        return event

    async def create_status_event(
        self,
        session_id: SessionId,
        source: EventSource,
        status: SessionStatus,
        data: JSONSerializable,
        metadata: Mapping[str, JSONSerializable] | None,
    ) -> Event:
        status_data: StatusEventData = {
            "status": status,
            "data": data,
        }

        return await self.create_event(
            session_id=session_id,
            kind=EventKind.STATUS,
            data=status_data,
            metadata=metadata,
            source=source,
            trigger_processing=False,
        )

    async def create_customer_message(
        self,
        session_id: SessionId,
        moderation: Moderation,
        message: str,
        source: EventSource,
        trigger_processing: bool,
        metadata: Mapping[str, JSONSerializable] | None,
        participant: Participant | None = None,
    ) -> Event:
        flagged = False
        groups: Set[str] = set()

        session = await self._session_store.read_session(session_id)

        if moderation in [Moderation.AUTO, Moderation.PARANOID]:
            moderation_service = await self._nlp_service.get_moderation_service()
            context = CustomerModerationContext(session=session, message=message)
            check = await moderation_service.moderate_customer(context)
            flagged |= check.flagged
            groups.update(check.tags)

        if moderation == Moderation.PARANOID:
            check = await _get_jailbreak_moderation_service(
                self._logger, self._meter, self._health_reporter
            ).moderate_customer(context)
            if "jailbreak" in check.tags:
                flagged = True
                groups.update({"jailbreak"})

        if participant is None:
            try:
                customer = await self._customer_store.read_customer(session.customer_id)
                customer_display_name = customer.name
            except Exception:
                customer_display_name = session.customer_id

            participant = {
                "id": session.customer_id,
                "display_name": customer_display_name,
            }

        message_data: MessageEventData = {
            "message": message,
            "participant": participant,
            "flagged": flagged,
            "groups": list(groups),
        }

        return await self.create_event(
            session_id=session.id,
            kind=EventKind.MESSAGE,
            data=message_data,
            source=source,
            trigger_processing=trigger_processing,
            metadata=metadata,
        )

    async def create_human_agent_message_event(
        self,
        session_id: SessionId,
        message: str,
        participant: Participant,
        metadata: Mapping[str, JSONSerializable] | None,
    ) -> Event:
        message_data: MessageEventData = {
            "message": message,
            "participant": {
                "id": AgentId(participant["id"])
                if "id" in participant and participant["id"]
                else None,
                "display_name": participant["display_name"],
            },
        }

        event = await self.create_event(
            session_id=session_id,
            kind=EventKind.MESSAGE,
            data=message_data,
            source=EventSource.HUMAN_AGENT,
            trigger_processing=False,
            metadata=metadata,
        )

        return event

    async def create_human_agent_on_behalf_of_ai_agent_message_event(
        self,
        session_id: SessionId,
        message: str,
        metadata: Mapping[str, JSONSerializable] | None,
    ) -> Event:
        session = await self._session_store.read_session(session_id)
        agent = await self._agent_store.read_agent(session.agent_id)

        message_data: MessageEventData = {
            "message": message,
            "participant": {
                "id": agent.id,
                "display_name": agent.name,
            },
        }

        event = await self.create_event(
            session_id=session_id,
            kind=EventKind.MESSAGE,
            data=message_data,
            source=EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
            trigger_processing=False,
            metadata=metadata,
        )

        return event

    def _processing_task_tag(self, session_id: SessionId) -> str:
        return f"process-session({session_id})"

    async def dispatch_processing_task(self, session: Session) -> str:
        tag = self._processing_task_tag(session.id)
        trace_id = self._tracer.trace_id

        async with self._lock:
            self._active_processing_tasks[session.id] = _ActiveProcessingTask(
                trace_id=trace_id,
                tag=tag,
            )

        await self._background_task_service.restart(
            self._process_session(session),
            tag=tag,
        )

        return trace_id

    async def _engine_for_agent(self, agent_id: AgentId) -> Engine:
        agent = await self._agent_store.read_agent(agent_id)

        return self._engine_registry.get_engine(agent.engine)

    async def _initialize_session(self, session: Session) -> None:
        event_emitter = await self._event_emitter_factory.create_event_emitter(
            emitting_agent_id=session.agent_id,
            session_id=session.id,
        )

        engine = await self._engine_for_agent(session.agent_id)

        await engine.initialize(
            Context(
                session_id=session.id,
                agent_id=session.agent_id,
            ),
            event_emitter=event_emitter,
        )

    async def _process_session(self, session: Session) -> None:
        try:
            event_emitter = await self._event_emitter_factory.create_event_emitter(
                emitting_agent_id=session.agent_id,
                session_id=session.id,
            )

            engine = await self._engine_for_agent(session.agent_id)

            await engine.process(
                Context(
                    session_id=session.id,
                    agent_id=session.agent_id,
                ),
                event_emitter=event_emitter,
            )
        finally:
            async with self._lock:
                active_task = self._active_processing_tasks.get(session.id)
                if active_task and active_task.trace_id == self._tracer.trace_id:
                    del self._active_processing_tasks[session.id]

    async def cancel_processing(
        self,
        session_id: SessionId,
    ) -> Event:
        _ = await self._session_store.read_session(session_id)

        async with self._lock:
            active_task = self._active_processing_tasks.get(session_id)

        if not active_task:
            raise ValueError(f"No active processing task for session '{session_id}'")

        cancelled = await self._background_task_service.cancel(
            tag=active_task.tag,
            reason="cancelled_by_api",
        )

        if not cancelled:
            raise ValueError(f"No active processing task for session '{session_id}'")

        timeout = Timeout(60)
        min_offset: int | None = None

        while True:
            status_events = await self._session_store.list_events(
                session_id=session_id,
                trace_id=active_task.trace_id,
                kinds=[EventKind.STATUS],
            )

            cancellation_events = [
                event
                for event in status_events
                if cast(dict[str, Any], event.data).get("status") == "cancelled"
            ]

            if cancellation_events:
                return cancellation_events[-1]

            if timeout.expired():
                raise TimeoutError(
                    f"Cancelled processing task for session '{session_id}', "
                    "but no cancellation event was emitted"
                )

            if status_events:
                min_offset = max(event.offset for event in status_events) + 1

            await self._session_listener.wait_for_more_events(
                session_id=session_id,
                min_offset=min_offset,
                trace_id=active_task.trace_id,
                kinds=[EventKind.STATUS],
                timeout=timeout,
            )

    async def process(
        self,
        session_id: SessionId,
    ) -> Event:
        session = await self._session_store.read_session(session_id)

        trace_id = await self.dispatch_processing_task(session)

        await self._session_listener.wait_for_more_events(
            session_id=session_id,
            trace_id=trace_id,
            timeout=Timeout(60),
        )

        event = next(
            iter(
                await self._session_store.list_events(
                    session_id=session_id,
                    trace_id=trace_id,
                    kinds=[EventKind.STATUS],
                )
            )
        )

        return event

    async def utter(
        self,
        session_id: SessionId,
        requests: Sequence[UtteranceRequest],
    ) -> Event:
        session = await self._session_store.read_session(session_id)

        with self._tracer.span("utter", {"session_id": session_id}):
            event_emitter = await self._event_emitter_factory.create_event_emitter(
                emitting_agent_id=session.agent_id,
                session_id=session.id,
            )

            engine = await self._engine_for_agent(session.agent_id)

            await engine.utter(
                context=Context(session_id=session.id, agent_id=session.agent_id),
                event_emitter=event_emitter,
                requests=requests,
            )

            event, *_ = await self._session_store.list_events(
                session_id=session_id,
                trace_id=self._tracer.trace_id,
                kinds=[EventKind.MESSAGE],
            )

            return event

    async def find_events(
        self,
        session_id: SessionId,
        min_offset: int,
        source: EventSource | None,
        kinds: Sequence[EventKind],
        trace_id: str | None,
    ) -> Sequence[Event]:
        events = await self._session_store.list_events(
            session_id=session_id,
            min_offset=min_offset,
            source=source,
            kinds=kinds,
            trace_id=trace_id,
        )

        return events

    async def delete_events(
        self,
        session_id: SessionId,
        min_offset: int,
    ) -> None:
        session = await self._session_store.read_session(session_id)

        events = sorted(
            await self._session_store.list_events(
                session_id=session_id,
                min_offset=0,
                exclude_deleted=True,
            ),
            key=lambda event: event.offset,
        )

        events_starting_from_min_offset = [e for e in events if e.offset >= min_offset]

        if not events_starting_from_min_offset:
            return

        event_at_min_offset = events_starting_from_min_offset[0]

        first_event_of_trace_id = next(
            e for e in events if e.trace_id == event_at_min_offset.trace_id
        )

        if event_at_min_offset.id != first_event_of_trace_id.id:
            raise ValueError(
                "Cannot delete events with offset < min_offset unless they are the first event of their trace ID"
            )

        for e in events_starting_from_min_offset:
            await self._session_store.delete_event(e.id)

        if not session.agent_states:
            return

        state_index_offset = next(
            (
                i
                for i, s in enumerate(session.agent_states, start=0)
                if s.trace_id.startswith(event_at_min_offset.trace_id)
            ),
            None,
        )

        if state_index_offset is None:
            return

        agent_states = session.agent_states[:state_index_offset]

        await self._session_store.update_session(
            session_id=session_id,
            params={"agent_states": agent_states},
        )

    async def read_event(
        self,
        session_id: SessionId,
        event_id: EventId,
    ) -> Event:
        """Reads a single event by ID."""
        return await self._session_store.read_event(
            session_id=session_id,
            event_id=event_id,
        )

    async def update_event(
        self,
        session_id: SessionId,
        event_id: EventId,
        params: EventUpdateParamsModel,
    ) -> Event:
        """Updates an event. Currently supports updating metadata, but extensible for future properties."""
        # Convert from app_modules EventUpdateParamsModel to store EventUpdateParams
        store_params: EventUpdateParams = {}

        if "metadata" in params and params["metadata"]:
            # For metadata updates, we need to get current event and apply set/unset operations
            current_event = await self.read_event(session_id, event_id)
            current_metadata = dict(current_event.metadata)

            metadata_params = params["metadata"]

            # Apply set operations
            if "set" in metadata_params and metadata_params["set"]:
                current_metadata.update(metadata_params["set"])

            # Apply unset operations
            if "unset" in metadata_params and metadata_params["unset"]:
                for key in metadata_params["unset"]:
                    current_metadata.pop(key, None)

            store_params["metadata"] = current_metadata

        return await self._session_store.update_event(
            session_id=session_id,
            event_id=event_id,
            params=store_params,
        )
