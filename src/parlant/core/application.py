from __future__ import annotations
import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
import time
import traceback
from typing import Any, Iterable, Mapping, Optional, TypeAlias, cast
from lagom import Container

from parlant.core.async_utils import Timeout
from parlant.core.common import generate_id
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.agents import AgentId
from parlant.core.emissions import EventEmitterFactory
from parlant.core.end_users import EndUserId
from parlant.core.evaluations import ConnectionProposition, Invoice
from parlant.core.guideline_connections import (
    GuidelineConnectionStore,
)
from parlant.core.guidelines import GuidelineId, GuidelineStore
from parlant.core.sessions import (
    Event,
    EventKind,
    EventSource,
    Session,
    SessionId,
    SessionListener,
    SessionStore,
)
from parlant.core.engines.types import Context, Engine
from parlant.core.logging import Logger


TaskQueue: TypeAlias = list[asyncio.Task[None]]


class Application:
    def __init__(self, container: Container) -> None:
        self._logger = container[Logger]
        self._correlator = container[ContextualCorrelator]
        self._session_store = container[SessionStore]
        self._session_listener = container[SessionListener]
        self._guideline_store = container[GuidelineStore]
        self._guideline_connection_store = container[GuidelineConnectionStore]
        self._engine = container[Engine]
        self._event_emitter_factory = container[EventEmitterFactory]

        self._tasks_by_session = dict[SessionId, TaskQueue]()
        self._lock = asyncio.Lock()
        self._last_garbage_collection = 0.0
        self._garbage_collection_interval = 5.0

    async def __aenter__(self) -> Application:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        await self._collect_garbage(force=True)
        return False

    async def _collect_garbage(self, force: bool = False) -> None:
        now = time.time()

        if not force:
            if (now - self._last_garbage_collection) < self._garbage_collection_interval:
                return

        async with self._lock:
            session_ids = list(self._tasks_by_session.keys())

            for session_id in session_ids:
                task_queue = self._tasks_by_session[session_id]
                re_enqueued_tasks = []

                for task in task_queue:
                    if task.done() or force:
                        try:
                            await task
                        except Exception as exc:
                            self._logger.warning(
                                f"Awaited session task raised an exception: {traceback.format_exception(exc)}"
                            )
                    else:
                        re_enqueued_tasks.append(task)

                if re_enqueued_tasks:
                    self._tasks_by_session[session_id] = re_enqueued_tasks
                else:
                    del self._tasks_by_session[session_id]

            self._last_garbage_collection = now

    async def wait_for_update(
        self,
        session_id: SessionId,
        min_offset: int,
        kinds: Sequence[EventKind] = [],
        source: Optional[EventSource] = None,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        await self._collect_garbage()

        return await self._session_listener.wait_for_events(
            session_id=session_id,
            min_offset=min_offset,
            kinds=kinds,
            source=source,
            timeout=timeout,
        )

    async def create_end_user_session(
        self,
        end_user_id: EndUserId,
        agent_id: AgentId,
        title: Optional[str] = None,
        allow_greeting: bool = False,
    ) -> Session:
        await self._collect_garbage()

        session = await self._session_store.create_session(
            creation_utc=datetime.now(timezone.utc),
            end_user_id=end_user_id,
            agent_id=agent_id,
            title=title,
        )

        if allow_greeting:
            await self.dispatch_processing_task(session)

        return session

    async def post_event(
        self,
        session_id: SessionId,
        kind: EventKind,
        data: Mapping[str, Any],
        source: EventSource = "end_user",
        trigger_processing: bool = True,
    ) -> Event:
        await self._collect_garbage()

        event = await self._session_store.create_event(
            session_id=session_id,
            source=source,
            kind=kind,
            correlation_id=self._correlator.correlation_id,
            data=data,
        )

        if trigger_processing:
            session = await self._session_store.read_session(session_id)
            await self.dispatch_processing_task(session)

        return event

    async def dispatch_processing_task(self, session: Session) -> str:
        async with self._lock:
            if session.id not in self._tasks_by_session:
                self._tasks_by_session[session.id] = TaskQueue()

            for task in self._tasks_by_session[session.id]:
                task.cancel()
            with self._correlator.correlation_scope(generate_id()):
                self._tasks_by_session[session.id].append(
                    asyncio.create_task(self._process_session(session))
                )

                return self._correlator.correlation_id

    async def _process_session(self, session: Session) -> None:
        event_emitter = await self._event_emitter_factory.create_event_emitter(
            emitting_agent_id=session.agent_id,
            session_id=session.id,
        )

        await self._engine.process(
            Context(
                session_id=session.id,
                agent_id=session.agent_id,
            ),
            event_emitter=event_emitter,
        )

    async def create_guidelines(
        self,
        guideline_set: str,
        invoices: Sequence[Invoice],
    ) -> Iterable[GuidelineId]:
        async def _create_connection_with_existing_guideline(
            source_key: str,
            target_key: str,
            content_guidelines: dict[str, GuidelineId],
            guideline_set: str,
            proposition: ConnectionProposition,
        ) -> None:
            if source_key in content_guidelines:
                source_guideline_id = content_guidelines[source_key]
                target_guideline_id = (
                    await self._guideline_store.find_guideline(
                        guideline_set=guideline_set,
                        guideline_content=proposition.target,
                    )
                ).id
            else:
                source_guideline_id = (
                    await self._guideline_store.find_guideline(
                        guideline_set=guideline_set,
                        guideline_content=proposition.source,
                    )
                ).id
                target_guideline_id = content_guidelines[target_key]

            await self._guideline_connection_store.create_connection(
                source=source_guideline_id,
                target=target_guideline_id,
                kind=proposition.connection_kind,
            )

        content_guidelines: dict[str, GuidelineId] = {
            f"{i.payload.content.predicate}_{i.payload.content.action}": (
                await self._guideline_store.create_guideline(
                    guideline_set=guideline_set,
                    predicate=i.payload.content.predicate,
                    action=i.payload.content.action,
                )
                if i.payload.operation == "add"
                else await self._guideline_store.update_guideline(
                    guideline_id=cast(GuidelineId, i.payload.updated_id),
                    params={
                        "predicate": i.payload.content.predicate,
                        "action": i.payload.content.action,
                    },
                )
            ).id
            for i in invoices
        }

        for i in invoices:
            if i.payload.operation == "update" and i.payload.connection_proposition:
                guideline_id = cast(GuidelineId, i.payload.updated_id)

                connections_to_delete = list(
                    await self._guideline_connection_store.list_connections(
                        indirect=False,
                        source=guideline_id,
                    )
                )
                connections_to_delete.extend(
                    await self._guideline_connection_store.list_connections(
                        indirect=False,
                        target=guideline_id,
                    )
                )
                await asyncio.gather(
                    *(
                        self._guideline_connection_store.delete_connection(c.id)
                        for c in connections_to_delete
                    )
                )

        connections: set[ConnectionProposition] = set([])

        for i in invoices:
            assert i.data

            if not i.data.connection_propositions:
                continue

            for c in i.data.connection_propositions:
                source_key = f"{c.source.predicate}_{c.source.action}"
                target_key = f"{c.target.predicate}_{c.target.action}"

                if c not in connections:
                    if c.check_kind == "connection_with_another_evaluated_guideline":
                        await self._guideline_connection_store.create_connection(
                            source=content_guidelines[source_key],
                            target=content_guidelines[target_key],
                            kind=c.connection_kind,
                        )
                    else:
                        await _create_connection_with_existing_guideline(
                            source_key,
                            target_key,
                            content_guidelines,
                            guideline_set,
                            c,
                        )
                    connections.add(c)

        return content_guidelines.values()
