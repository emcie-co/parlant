from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import time
import traceback
from typing import Any, Mapping, Optional, Sequence, Type, TypeAlias
from lagom import Container

from emcie.server.async_utils import Timeout
from emcie.server.core.agents import AgentId
from emcie.server.core.end_users import EndUserId
from emcie.server.core.sessions import (
    Event,
    Session,
    SessionId,
    SessionListener,
    SessionStore,
    ToolCall,
)
from emcie.server.engines.common import Context, Engine
from emcie.server.engines.event_emitter import EventEmitter, EmittedEvent
from emcie.server.logger import Logger


class EventBuffer(EventEmitter):
    def __init__(self) -> None:
        self.events: list[EmittedEvent] = []

    async def emit_message(
        self,
        correlation_id: str,
        message: str,
    ) -> EmittedEvent:
        event = EmittedEvent(
            source="server",
            kind=Event.MESSAGE_KIND,
            correlation_id=correlation_id,
            data={"message": message},
        )

        self.events.append(event)

        return event

    async def emit_tool_results(
        self,
        correlation_id: str,
        results: Sequence[ToolCall],
    ) -> EmittedEvent:
        event = EmittedEvent(
            source="server",
            kind=Event.TOOL_KIND,
            correlation_id=correlation_id,
            data={"tool_calls": list(results)},  # type: ignore
        )

        self.events.append(event)

        return event


TaskQueue: TypeAlias = list[asyncio.Task[None]]


class MC:
    def __init__(self, container: Container) -> None:
        self._logger = container[Logger]
        self._session_store = container[SessionStore]
        self._session_listener = container[SessionListener]
        self._engine = container[Engine]
        self._tasks_by_session = dict[SessionId, TaskQueue]()
        self._lock = asyncio.Lock()
        self._last_garbage_collection = 0.0
        self._garbage_collection_interval = 5.0

    async def __aenter__(self) -> MC:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
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
        timeout: Timeout,
    ) -> bool:
        await self._collect_garbage()

        return await self._session_listener.wait_for_events(
            session_id=session_id,
            min_offset=min_offset,
            timeout=timeout,
        )

    async def create_end_user_session(
        self,
        end_user_id: EndUserId,
        agent_id: AgentId,
        title: Optional[str] = None,
    ) -> Session:
        await self._collect_garbage()

        session = await self._session_store.create_session(
            end_user_id=end_user_id,
            agent_id=agent_id,
            title=title,
        )

        await self._dispatch_processing_task(session)

        return session

    async def post_client_event(
        self,
        session_id: SessionId,
        kind: str,
        data: Mapping[str, Any],
    ) -> Event:
        await self._collect_garbage()

        event = await self._session_store.create_event(
            session_id=session_id,
            source="client",
            kind=kind,
            data=data,
        )

        session = await self._session_store.read_session(session_id)

        await self._dispatch_processing_task(session)

        return event

    async def update_consumption_offset(
        self,
        session: Session,
        new_offset: int,
    ) -> None:
        await self._session_store.update_consumption_offset(
            session.id,
            consumer_id="client",
            new_offset=new_offset,
        )

    async def _dispatch_processing_task(self, session: Session) -> None:
        async with self._lock:
            if session.id not in self._tasks_by_session:
                self._tasks_by_session[session.id] = TaskQueue()

            for task in self._tasks_by_session[session.id]:
                task.cancel()

            self._tasks_by_session[session.id].append(
                asyncio.create_task(self._process_session(session))
            )

    async def _process_session(self, session: Session) -> None:
        emitter = EventBuffer()

        await self._engine.process(
            Context(
                session_id=session.id,
                agent_id=session.agent_id,
            ),
            emitter,
        )

        await self._add_events_to_session(
            session_id=session.id,
            events=emitter.events,
        )

    async def _add_events_to_session(
        self,
        session_id: SessionId,
        events: Sequence[EmittedEvent],
    ) -> None:
        utc_now = datetime.now(timezone.utc)

        for e in events:
            await self._session_store.create_event(
                session_id=session_id,
                source=e.source,
                kind=e.kind,
                data=e.data,
                creation_utc=utc_now,
            )
