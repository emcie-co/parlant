import asyncio
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Any, Coroutine, Mapping, Sequence, Set, cast

from parlant.core.agents import Moderation, AgentId, AgentStore
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.application_context import ApplicationContext
from parlant.core.async_utils import Timeout
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.diagnostic_reports import (
    DiagnosticReportService,
    get_diagnostic_report_service,
)
from parlant.core.common import ItemNotFoundError, JSONSerializable
from parlant.core import metadata as metadata_utils
from parlant.core.health import HealthReporter
from parlant.core.meter import Meter
from parlant.core.persistence.common import Cursor, SortDirection
from parlant.core.tracer import Tracer, session_operation_span
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.emissions import EventEmitterFactory
from parlant.core.engines.types import (
    Context,
    Engine,
    EngineRegistry,
    UtteranceRationale,
    UtteranceRequest,
)
from parlant.core.loggers import ScopedLogger, Logger
from parlant.core.nlp.moderation import CustomerModerationContext
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.response_coordination import (
    CoordinatedEventEmitter,
    Inputs,
    LocalResponseCoordinator,
    ResponseRun,
    response_inputs,
    response_run_scope,
)
from parlant.core.sessions import (
    AgentState,
    AttachmentRef,
    Event,
    EventId,
    EventKind,
    EventSource,
    MetadataRevisionConflict,
    MessageEventData,
    Participant,
    Session,
    SessionId,
    SessionListener,
    SessionMode,
    SessionStatus,
    SessionStore,
    StatusEventData,
    patch_internal_session_metadata,
    patch_public_session_metadata,
)
from parlant.core.store_provider import StoreProvider, StoreProviderHints
from dataclasses import dataclass
from typing_extensions import TypedDict


class SessionNotOwnedByGuestError(Exception):
    """The guest claiming a session is not the guest who owns it."""


class SessionAlreadyClaimedError(Exception):
    """The session already belongs to a customer, or another claim won."""


class ClaimSecretRejectedError(Exception):
    """The claim secret is wrong, has expired, or no claim was prepared."""


class SessionNotOwnedByCustomerError(Exception):
    """The customer allowing more is not the customer the session belongs to."""


class SessionUpdateParamsModel(TypedDict, total=False):
    """Parameters for updating a session."""

    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: str | None
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


class SessionModule:
    def __init__(
        self,
        request_context: RequestContext,
        logger: Logger,
        meter: Meter,
        tracer: Tracer,
        session_listener: SessionListener,
        model_router: ModelRouter,
        engine_registry: EngineRegistry,
        event_emitter_factory: EventEmitterFactory,
        background_task_service: BackgroundTaskService,
        health_reporter: HealthReporter,
        store_provider: StoreProvider,
        application_context: ApplicationContext,
    ):
        self._application_context = application_context
        self._request_context = request_context
        self._logger = ScopedLogger(logger, "SessionModule")
        self._store_provider = store_provider
        self._meter = meter
        self._tracer = tracer
        self._session_listener = session_listener
        self._model_router = model_router
        self._health_reporter = health_reporter

        self._engine_registry = engine_registry
        self._event_emitter_factory = event_emitter_factory
        self._background_task_service = background_task_service

        self._lock = asyncio.Lock()
        self._active_processing_tasks: dict[SessionId, _ActiveProcessingTask] = {}
        self.response_coordinator = LocalResponseCoordinator(
            background_tasks=background_task_service,
            logger=self._logger,
            read_inputs=self._read_response_inputs,
            render=self._render_queued_response,
        )

    async def _read_response_inputs(self, session_id: SessionId) -> Inputs:
        # Emission and engine history use the runtime store, which can differ
        # from the SDK declaration store on managed servers.
        store = self._store_provider.get_store(SessionStore, StoreProviderHints(call_site="engine"))
        return response_inputs(await store.list_events(session_id))

    async def queue_response(
        self, session_id: SessionId, *, after: Coroutine[Any, Any, None] | None = None
    ) -> None:
        try:
            await self._session_store.read_session(session_id)
        except BaseException:
            if after is not None:
                after.close()
            raise
        await self.response_coordinator.request(session_id, after=after)

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
        guest_instance_id: str | None = None,
        # External plugins use this to write trusted values under __parlant__; do not remove.
        parlant_metadata: Mapping[str, JSONSerializable] | None = None,
        customer_represented_by_an_agent: bool = False,
    ) -> Session:
        # Validates the agent exists before the session is created.
        _ = await self._agent_store.read_agent(agent_id=agent_id)
        public_metadata = dict(metadata or {})
        metadata_utils.ensure_public_metadata(public_metadata)
        trusted_values = dict(parlant_metadata or {})
        if guest_instance_id is not None:
            auth = dict(
                cast(
                    Mapping[str, JSONSerializable],
                    trusted_values.get("auth", {}),
                )
            )
            auth["guest_instance_id"] = guest_instance_id
            trusted_values["auth"] = auth

        # Which server produced this session. Recorded in the reserved namespace
        # so a caller cannot forge it, and so observability can tell production
        # traffic apart from a developer trying things out.
        server_values = dict(cast(Mapping[str, JSONSerializable], trusted_values.get("server", {})))
        server_values["environment"] = self._application_context.environment.value
        trusted_values["server"] = server_values

        persisted_metadata = metadata_utils.add_internal_metadata(public_metadata, trusted_values)

        session = await self._session_store.create_session(
            creation_utc=datetime.now(timezone.utc),
            customer_id=customer_id,
            agent_id=agent_id,
            title=title,
            metadata=persisted_metadata,
            labels=labels,
            customer_represented_by_an_agent=customer_represented_by_an_agent,
        )

        # Warm the engine's caches in the background as soon as the session
        # exists, so the first response reads them instead of building them.
        await self._initialize_session(session)

        if allow_greeting:
            await self.dispatch_processing_task(session)

        return await self._session_store.read_session(session.id)

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
        metadata: EventMetadataUpdateParamsModel | None = None,
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

        if metadata:
            session = await patch_public_session_metadata(
                self._session_store,
                session_id,
                metadata.get("set") or {},
                metadata.get("unset") or (),
            )

        return session

    async def prepare_claim(
        self,
        session_id: SessionId,
        secret_sha256: str,
        expires_at: datetime,
    ) -> None:
        """Lets whoever holds a one-time secret claim a guest's session, until
        ``expires_at``.

        For a session with no guest owner to prove ownership with, such as one
        a platform created for an AI agent acting for the customer: the
        platform hands the secret to the person through the store's sign-in
        page. Only the secret's SHA-256 hex digest is kept, under Parlant's own
        metadata, where no public metadata write can reach it. Preparing again
        replaces the earlier secret."""
        session = await self._session_store.read_session(session_id)

        if session.customer_id != CustomerStore.GUEST_ID:
            raise SessionAlreadyClaimedError(session_id)

        prepared = metadata_utils.set_internal_value(
            session.metadata,
            ("auth", "claim"),
            {"secret_sha256": secret_sha256, "expires_at": expires_at.isoformat()},
        )

        try:
            await self._session_store.commit_session_metadata(
                session_id, session.metadata_revision, prepared
            )
        except MetadataRevisionConflict:
            raise SessionAlreadyClaimedError(session_id)

    async def claim(
        self,
        session_id: SessionId,
        customer_id: CustomerId,
        guest_instance_id: str | None = None,
        scopes: Sequence[str] = (),
        scopes_expire_at: int | None = None,
        claim_secret: str | None = None,
    ) -> Session:
        """Hands a guest's session to the customer the guest signed in as.

        Ownership is proven by exactly one of the owning guest's instance id or
        the secret of a prepared claim (see ``prepare_claim``). Both are dropped
        in one conditional write against the revision read here, so of two
        concurrent claims exactly one wins. The claim then records
        ``customer.signed_in`` (carrying the token's scopes, so the agent can
        act on them right away) and triggers the agent's response: there is no
        customer message, the event is what it answers.
        """
        if (guest_instance_id is None) == (claim_secret is None):
            raise ValueError("Claiming needs exactly one of a guest instance id and a claim secret")

        session = await self._session_store.read_session(session_id)

        if session.customer_id != CustomerStore.GUEST_ID:
            raise SessionAlreadyClaimedError(session_id)

        if claim_secret is not None:
            if not _claim_secret_matches(session.metadata, claim_secret):
                raise ClaimSecretRejectedError(session_id)
        else:
            owner = metadata_utils.internal_value(session.metadata, ("auth", "guest_instance_id"))

            if owner != guest_instance_id:
                raise SessionNotOwnedByGuestError(session_id)

        claimed_metadata = metadata_utils.set_internal_value(
            session.metadata, ("auth", "guest_instance_id"), None
        )
        claimed_metadata = metadata_utils.set_internal_value(
            claimed_metadata, ("auth", "claim"), None
        )
        claimed_metadata = metadata_utils.set_internal_value(
            claimed_metadata, ("auth", "claimed_by"), customer_id
        )

        try:
            await self._session_store.commit_session_metadata(
                session_id, session.metadata_revision, claimed_metadata
            )
        except MetadataRevisionConflict:
            raise SessionAlreadyClaimedError(session_id)

        session = await self._session_store.update_session(session_id, {"customer_id": customer_id})
        customer = await self._customer_store.read_customer(customer_id)

        await self._create_event(
            session_id=session_id,
            kind=EventKind.CUSTOM,
            data={"type": "customer.signed_in", "customer_id": customer_id, "name": customer.name},
            metadata=None,
            source=EventSource.SYSTEM,
            trigger_processing=True,
            parlant_metadata=(
                {"auth": {"scopes": list(scopes), "expires_at": scopes_expire_at}}
                if scopes
                else None
            ),
        )

        return session

    async def authorize(
        self,
        session_id: SessionId,
        customer_id: CustomerId,
        scopes: Sequence[str],
        scopes_expire_at: int | None,
    ) -> Session:
        """Records that the session's own customer, already signed in, allowed
        more (a step-up), and triggers the agent's response so it can carry on
        with the new scopes. Like a sign-in, it is a moment the agent answers
        rather than a message."""
        session = await self._session_store.read_session(session_id)

        if session.customer_id != customer_id or customer_id == CustomerStore.GUEST_ID:
            raise SessionNotOwnedByCustomerError(session_id)

        await self._create_event(
            session_id=session_id,
            kind=EventKind.CUSTOM,
            data={"type": "customer.authorized", "customer_id": customer_id},
            metadata=None,
            source=EventSource.SYSTEM,
            trigger_processing=True,
            parlant_metadata={"auth": {"scopes": list(scopes), "expires_at": scopes_expire_at}},
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
        # External plugins use this to write trusted values under __parlant__; do not remove.
        parlant_metadata: Mapping[str, JSONSerializable] | None = None,
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
                parlant_metadata=parlant_metadata,
            )

    async def _create_event(
        self,
        session_id: SessionId,
        kind: EventKind,
        data: Mapping[str, Any],
        metadata: Mapping[str, JSONSerializable] | None,
        source: EventSource,
        trigger_processing: bool,
        # Forwarded from plugin-facing entry points; do not remove.
        parlant_metadata: Mapping[str, JSONSerializable] | None,
    ) -> Event:
        metadata_utils.ensure_public_metadata(metadata or {})
        persisted_metadata = metadata_utils.add_internal_metadata(
            metadata or {},
            parlant_metadata or {},
        )
        event = await self._session_store.create_event(
            session_id=session_id,
            source=source,
            kind=kind,
            trace_id=self._tracer.trace_id,
            data=data,
            metadata=persisted_metadata,
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
        message: str,
        source: EventSource,
        trigger_processing: bool,
        metadata: Mapping[str, JSONSerializable] | None,
        participant: Participant | None = None,
        # External plugins use this to write trusted values under __parlant__; do not remove.
        parlant_metadata: Mapping[str, JSONSerializable] | None = None,
        attachments: Sequence[AttachmentRef] | None = None,
    ) -> Event:
        flagged = False
        groups: Set[str] = set()

        session = await self._session_store.read_session(session_id)
        agent = await self._agent_store.read_agent(session.agent_id)

        # Moderation is an agent property, not a per-message choice:
        # OFF skips regardless of routing; AUTO runs iff the route is
        # configured (silently skipping otherwise); ON requires the route -
        # enforced at startup and on agent creation, so a failure here is a
        # mid-run configuration regression and is logged as an error.
        if agent.moderation is not Moderation.OFF:
            try:
                moderation_service = await self._model_router.get_moderation_service(
                    context=None,
                    agent_id=session.agent_id,
                    route="moderation.customer",
                )
            except Exception as exc:
                if agent.moderation is Moderation.ON:
                    self._logger.error(
                        "Agent moderation is ON but no model resolves for the "
                        f"'moderation.customer' route; message NOT moderated ({exc})"
                    )
            else:
                context = CustomerModerationContext(session=session, message=message)
                check = await moderation_service.moderate_customer(context)
                flagged |= check.flagged
                groups.update(check.tags)

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

        if attachments is not None:
            message_data["attachments"] = list(attachments)

        return await self.create_event(
            session_id=session.id,
            kind=EventKind.MESSAGE,
            data=message_data,
            source=source,
            trigger_processing=trigger_processing,
            metadata=metadata,
            parlant_metadata=parlant_metadata,
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
        response_run = await self.response_coordinator.begin_foreground(session.id)
        tag = self._processing_task_tag(session.id)
        trace_id = self._tracer.trace_id
        report_service = get_diagnostic_report_service(self._tracer)
        report_service.mark_engine_turn(trace_id, str(session.id))
        retained_trace = report_service.retain(trace_id)

        self._logger.debug(f"Dispatching processing for session {session.id}")

        async with self._lock:
            self._active_processing_tasks[session.id] = _ActiveProcessingTask(
                trace_id=trace_id,
                tag=tag,
            )

        processing = self._process_session(
            session,
            trace_id=trace_id,
            release_retained_trace=retained_trace,
            response_run=response_run,
        )
        try:
            await self._background_task_service.restart(
                processing,
                tag=tag,
            )
        except BaseException:
            processing.close()
            await self.response_coordinator.finish_foreground(response_run, succeeded=False)
            if retained_trace:
                report_service.release(trace_id)
            async with self._lock:
                active_task = self._active_processing_tasks.get(session.id)
                if active_task and active_task.trace_id == trace_id:
                    del self._active_processing_tasks[session.id]
            raise

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

        try:
            await engine.initialize(
                Context(
                    session_id=session.id,
                    agent_id=session.agent_id,
                ),
                event_emitter=event_emitter,
            )
        except ItemNotFoundError as exc:
            # Initialization only warms the engine's caches; a session may
            # legitimately reference entities the engine can't load yet (e.g.
            # a customer id the application hasn't registered). Creation must
            # not fail on a warming miss.
            self._logger.warning(f"Session {session.id} cache warming skipped: {exc}")

    async def _process_session(
        self,
        session: Session,
        trace_id: str,
        release_retained_trace: bool,
        response_run: ResponseRun,
    ) -> None:
        succeeded = False
        coordinated_emitter: CoordinatedEventEmitter | None = None
        try:
            self._logger.debug(f"Starting background processing for session {session.id}")
            event_emitter = await self._event_emitter_factory.create_event_emitter(
                emitting_agent_id=session.agent_id,
                session_id=session.id,
            )

            engine = await self._engine_for_agent(session.agent_id)
            coordinated_emitter = CoordinatedEventEmitter(event_emitter, response_run)

            with response_run_scope(response_run):
                succeeded = await engine.process(
                    Context(
                        session_id=session.id,
                        agent_id=session.agent_id,
                    ),
                    event_emitter=coordinated_emitter,
                )
        finally:
            if coordinated_emitter is not None:
                await self._close_response_streams(coordinated_emitter)
            await self.response_coordinator.finish_foreground(response_run, succeeded=succeeded)
            report_service = get_diagnostic_report_service(self._tracer)

            if report_service.should_attach_session_data(trace_id):
                # Attach the session's persisted state (full event stream +
                # metadata) to the pending report before the release below
                # finalizes it. Best-effort: a store failure must not mask
                # the turn's outcome.
                try:
                    await self._attach_session_data_to_diagnostics(
                        report_service, trace_id, session.id
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"Diagnostic session export failed for session {session.id}: {exc}"
                    )

            if release_retained_trace:
                report_service.release(trace_id)

            async with self._lock:
                active_task = self._active_processing_tasks.get(session.id)
                if active_task and active_task.trace_id == trace_id:
                    del self._active_processing_tasks[session.id]

    async def _close_response_streams(self, emitter: CoordinatedEventEmitter) -> None:
        try:
            await emitter.close_interrupted_messages()
        except ItemNotFoundError:
            # History/session deletion already removed the old stream.
            pass
        except Exception as error:
            self._logger.warning(f"Closing interrupted response streams failed: {error}")

    async def _attach_session_data_to_diagnostics(
        self,
        report_service: DiagnosticReportService,
        trace_id: str,
        session_id: SessionId,
    ) -> None:
        """Export persisted session state into a pending report or trace archive.

        Runs after processing ends, so the stream read here includes a
        failure's own error event and the session's freshest metadata.
        Deleted events are included: an archive is a diagnostic artifact, and
        what was deleted can matter to the diagnosis.
        """

        session = await self._session_store.read_session(session_id=session_id)
        events = await self._session_store.list_events(
            session_id=session_id,
            exclude_deleted=False,
        )

        report_service.attach_session_data(
            trace_id,
            session={
                "id": session.id,
                "creation_utc": session.creation_utc.isoformat(),
                "customer_id": session.customer_id,
                "agent_id": session.agent_id,
                "mode": session.mode,
                "title": session.title,
                "metadata": dict(session.metadata),
                "labels": sorted(session.labels),
            },
            events=[
                {
                    "id": event.id,
                    "offset": event.offset,
                    "source": event.source.value,
                    "kind": event.kind.value,
                    "creation_utc": event.creation_utc.isoformat(),
                    "trace_id": event.trace_id,
                    "data": event.data,
                    "metadata": dict(event.metadata),
                    "deleted": event.deleted,
                }
                for event in events
            ],
        )

    async def cancel_processing(
        self,
        session_id: SessionId,
        trace_id: str | None = None,
    ) -> Event:
        """Cancel the session's active processing task.

        When ``trace_id`` is given, cancellation applies only if the active
        task belongs to that trace — a stale cancellation (e.g. for a turn that
        has already been superseded) must never kill the successor's run."""
        _ = await self._session_store.read_session(session_id)

        async with self._lock:
            active_task = self._active_processing_tasks.get(session_id)

        if not active_task:
            raise ValueError(f"No active processing task for session '{session_id}'")

        if trace_id is not None and active_task.trace_id != trace_id:
            raise ValueError(
                f"The active processing task for session '{session_id}' belongs to "
                f"another trace ('{active_task.trace_id}', not '{trace_id}')"
            )

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
        # Explicit utter() remains an independent operation, even when called
        # from inside a coordinated turn or its inherited background context.
        with response_run_scope(None):
            return await self._utter(session_id, requests)

    async def _render_queued_response(self, run: ResponseRun) -> bool:
        session = await self._session_store.read_session(run.session_id)
        if session.mode == "manual":
            return False
        with response_run_scope(run):
            event = await self._utter(
                run.session_id,
                requests=[
                    UtteranceRequest(
                        action="Respond to the customer's pending requests using the completed background results in the conversation.",
                        rationale=UtteranceRationale.FOLLOW_UP,
                    )
                ],
                response_run=run,
            )
        return event.kind == EventKind.MESSAGE

    async def _utter(
        self,
        session_id: SessionId,
        requests: Sequence[UtteranceRequest],
        response_run: ResponseRun | None = None,
    ) -> Event:
        session = await self._session_store.read_session(session_id)
        report_service = get_diagnostic_report_service(self._tracer)

        # Outside a turn this starts a trace of its own, which observability
        # must export as it does a request's, or the message shows no activity.
        with session_operation_span(
            self._tracer,
            "sessions.queue_response" if response_run else "sessions.utter",
            session_id,
            "queue_response" if response_run else "utter",
        ):
            trace_id = self._tracer.trace_id
            report_service.mark_engine_turn(trace_id, str(session.id))
            event_emitter = await self._event_emitter_factory.create_event_emitter(
                emitting_agent_id=session.agent_id,
                session_id=session.id,
            )

            engine = await self._engine_for_agent(session.agent_id)

            coordinated_emitter: CoordinatedEventEmitter | None = None
            if response_run is not None:

                async def may_publish() -> bool:
                    current = await self._session_store.read_session(session_id)
                    return current.mode != "manual"

                coordinated_emitter = CoordinatedEventEmitter(
                    event_emitter, response_run, may_publish=may_publish
                )
                event_emitter = coordinated_emitter
                async with self._lock:
                    self._active_processing_tasks[session_id] = _ActiveProcessingTask(
                        trace_id=trace_id,
                        tag=self.response_coordinator.task_tag(session_id),
                    )

            # Engine cleanup clears its EntityContext. Keep that mutation in
            # an awaited child task so the caller can continue using its own
            # Session.current / Agent.current after this follow-up completes.
            try:
                succeeded = await asyncio.create_task(
                    engine.utter(
                        context=Context(session_id=session.id, agent_id=session.agent_id),
                        event_emitter=event_emitter,
                        requests=requests,
                    )
                )
            except asyncio.CancelledError:
                if response_run is not None:
                    await event_emitter.emit_status_event(
                        trace_id=trace_id,
                        data=StatusEventData(status="cancelled", data={}),
                    )
                    await event_emitter.emit_status_event(
                        trace_id=trace_id,
                        data=StatusEventData(status="ready", data={"stage": "completed"}),
                    )
                raise
            finally:
                if coordinated_emitter is not None:
                    await self._close_response_streams(coordinated_emitter)
                if response_run is not None:
                    async with self._lock:
                        active = self._active_processing_tasks.get(session_id)
                        if active is not None and active.trace_id == trace_id:
                            del self._active_processing_tasks[session_id]

            event_kinds = [EventKind.MESSAGE] if succeeded else [EventKind.STATUS]
            events = await self._session_store.list_events(
                session_id=session_id,
                trace_id=self._tracer.trace_id,
                kinds=event_kinds,
            )

            if not events:
                raise RuntimeError(
                    f"Engine utter produced no {'message' if succeeded else 'status'} event"
                )

            if report_service.should_attach_session_data(trace_id):
                try:
                    await self._attach_session_data_to_diagnostics(
                        report_service, trace_id, session.id
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"Diagnostic report session data failed for session {session.id}: {exc}"
                    )

            return events[-1]

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

        await patch_internal_session_metadata(
            self._session_store,
            session_id,
            ("compass", "needs_rebuild"),
            True,
        )

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
        if "metadata" in params and params["metadata"]:
            metadata_params = params["metadata"]
            set_values = dict(metadata_params.get("set", {}))
            unset_keys = tuple(metadata_params.get("unset", ()))
            for key in (*set_values, *unset_keys):
                metadata_utils.ensure_public_metadata_key(key)

            for _ in range(8):
                current_event = await self.read_event(session_id, event_id)
                try:
                    return await self._session_store.patch_event_metadata(
                        session_id,
                        event_id,
                        current_event.metadata_revision,
                        set_values,
                        unset_keys,
                    )
                except MetadataRevisionConflict:
                    continue
            raise MetadataRevisionConflict(current_event.metadata_revision)

        return await self.read_event(session_id, event_id)


def _claim_secret_matches(metadata: Mapping[str, JSONSerializable], claim_secret: str) -> bool:
    """Whether ``claim_secret`` is the session's prepared one, and still valid."""
    prepared = metadata_utils.internal_value(metadata, ("auth", "claim"))

    if not isinstance(prepared, Mapping):
        return False

    stored = prepared.get("secret_sha256")
    expires_at = prepared.get("expires_at")

    if not isinstance(stored, str) or not isinstance(expires_at, str):
        return False

    try:
        if datetime.fromisoformat(expires_at) <= datetime.now(timezone.utc):
            return False
    except ValueError:
        return False

    return hmac.compare_digest(hashlib.sha256(claim_secret.encode()).hexdigest(), stored)
