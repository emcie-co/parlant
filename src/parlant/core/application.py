# Copyright 2024 Emcie Co Ltd.
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
from collections.abc import Sequence
from datetime import datetime, timezone
from lagom import Container
from typing import Any, Iterable, Mapping, Optional, TypeAlias, cast

from parlant.core.async_utils import Timeout
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import (
    AgentId,
    CustomerId,
    EventKind,
    EventSource,
    GuidelineId,
    SessionId,
    generate_id,
)
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.emissions import EventEmitterFactory
from parlant.core.engines.types import Context, Engine, UtteranceRequest
from parlant.core.evaluations import ConnectionProposition, Invoice
from parlant.core.guideline_connections import (
    GuidelineConnectionStore,
)
from parlant.core.guidelines import GuidelineStore
from parlant.core.logging import Logger
from parlant.core.sessions import (
    Event,
    Session,
    SessionListener,
    SessionStore,
)

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
        self._background_task_service = container[BackgroundTaskService]

        self._lock = asyncio.Lock()

    async def wait_for_update(
        self,
        session_id: SessionId,
        min_offset: int,
        kinds: Sequence[EventKind] = [],
        source: Optional[EventSource] = None,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        return await self._session_listener.wait_for_events(
            session_id=session_id,
            min_offset=min_offset,
            kinds=kinds,
            source=source,
            timeout=timeout,
        )

    async def create_customer_session(
        self,
        customer_id: CustomerId,
        agent_id: AgentId,
        title: Optional[str] = None,
        allow_greeting: bool = False,
    ) -> Session:
        session = await self._session_store.create_session(
            creation_utc=datetime.now(timezone.utc),
            customer_id=customer_id,
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
        source: EventSource = "customer",
        trigger_processing: bool = True,
    ) -> Event:
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
        with self._correlator.correlation_scope(generate_id()):
            await self._background_task_service.restart(
                self._process_session(session),
                tag=f"process-session({session.id})",
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

    async def utter(
        self,
        session: Session,
        requests: Sequence[UtteranceRequest],
    ) -> str:
        with self._correlator.correlation_scope(generate_id()):
            event_emitter = await self._event_emitter_factory.create_event_emitter(
                emitting_agent_id=session.agent_id,
                session_id=session.id,
            )

            await self._engine.utter(
                context=Context(session_id=session.id, agent_id=session.agent_id),
                event_emitter=event_emitter,
                requests=requests,
            )

            return self._correlator.correlation_id

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
            )

        content_guidelines: dict[str, GuidelineId] = {
            f"{invoice.payload.content.condition}_{invoice.payload.content.action}": (
                await self._guideline_store.create_guideline(
                    guideline_set=guideline_set,
                    condition=invoice.payload.content.condition,
                    action=invoice.payload.content.action,
                )
                if invoice.payload.operation == "add"
                else await self._guideline_store.update_guideline(
                    guideline_id=cast(GuidelineId, invoice.payload.updated_id),
                    params={
                        "condition": invoice.payload.content.condition,
                        "action": invoice.payload.content.action,
                    },
                )
            ).id
            for invoice in invoices
        }

        for invoice in invoices:
            if invoice.payload.operation == "update" and invoice.payload.connection_proposition:
                guideline_id = cast(GuidelineId, invoice.payload.updated_id)

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

                for conn in connections_to_delete:
                    await self._guideline_connection_store.delete_connection(conn.id)

        connections: set[ConnectionProposition] = set([])

        for invoice in invoices:
            assert invoice.data

            if not invoice.data.connection_propositions:
                continue

            for proposition in invoice.data.connection_propositions:
                source_key = f"{proposition.source.condition}_{proposition.source.action}"
                target_key = f"{proposition.target.condition}_{proposition.target.action}"

                if proposition not in connections:
                    if proposition.check_kind == "connection_with_another_evaluated_guideline":
                        await self._guideline_connection_store.create_connection(
                            source=content_guidelines[source_key],
                            target=content_guidelines[target_key],
                        )
                    else:
                        await _create_connection_with_existing_guideline(
                            source_key,
                            target_key,
                            content_guidelines,
                            guideline_set,
                            proposition,
                        )
                    connections.add(proposition)

        return content_guidelines.values()
