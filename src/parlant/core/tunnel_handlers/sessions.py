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

from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from enum import Enum
from typing import Any, cast

from parlant.core.agents import AgentId
from parlant.core.app_modules.common import decode_cursor, encode_cursor
from parlant.core.app_modules.sessions import (
    EventMetadataUpdateParamsModel,
    SessionModule,
    SessionUpdateParamsModel,
)
from parlant.core.async_utils import Timeout
from parlant.core.common import ItemNotFoundError, UniqueId
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core import metadata
from parlant.core.persistence.common import SortDirection
from parlant.core.session_clients import SESSION_CLIENT_METADATA_KEY, SessionClientRecorder
from parlant.core.sessions import EventId, EventKind, EventSource, Session, SessionId
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


def _parse_sort_direction(value: str | None) -> SortDirection | None:
    if value is None:
        return None
    match value:
        case "asc":
            return SortDirection.ASC
        case "desc":
            return SortDirection.DESC
        case _:
            raise ValueError(f"Unsupported sort direction: {value}")


def _parse_event_kinds(value: Any) -> list[EventKind]:
    if value is None:
        return []
    if isinstance(value, str):
        return [EventKind(k) for k in value.split(",") if k]
    return [EventKind(k) for k in value]


# The tunnel wire protocol uses the API's source vocabulary ("customer"),
# which is deliberately decoupled from the core enum's values.
_WIRE_EVENT_SOURCES: dict[str, EventSource] = {
    "customer": EventSource.CUSTOMER,
    "customer_ui": EventSource.CUSTOMER_UI,
    "human_agent": EventSource.HUMAN_AGENT,
    "human_agent_on_behalf_of_ai_agent": EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
    "ai_agent": EventSource.AI_AGENT,
    "system": EventSource.SYSTEM,
}


_EVENT_SOURCE_WIRE_NAMES: dict[EventSource, str] = {
    source: name for name, source in _WIRE_EVENT_SOURCES.items()
}


def _parse_event_source(value: str) -> EventSource:
    if source := _WIRE_EVENT_SOURCES.get(value):
        return source
    raise ValueError(f"'{value}' is not a valid event source")


class SessionTunnelHandlers:
    def __init__(
        self,
        session_module: SessionModule,
        session_client_recorder: SessionClientRecorder | None = None,
    ) -> None:
        self._session_module = session_module
        self._session_client_recorder = session_client_recorder or SessionClientRecorder()

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "sessions.create": self.create_session,
                "sessions.read": self.read_session,
                "sessions.list": self.list_sessions,
                "sessions.update": self.update_session,
                "sessions.delete": self.delete_session,
                "sessions.delete_many": self.delete_many_sessions,
                "sessions.create_event": self.create_event,
                "sessions.list_events": self.list_events,
                "sessions.read_event": self.read_event,
                "sessions.update_event": self.update_event,
                "sessions.delete_events": self.delete_events,
            },
            streams={
                "sessions.stream_events": self.stream_events,
                "sessions.stream_event": self.stream_event,
            },
        )

    async def create_event(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = SessionId(params["session_id"])
        kind = params.get("kind", "message")
        message = params.get("message", "")
        source_str = params.get("source", "customer")

        source = _parse_event_source(source_str)

        if kind == "message" and source in (EventSource.CUSTOMER, EventSource.CUSTOMER_UI):
            event = await self._session_module.create_customer_message(
                session_id=session_id,
                message=message,
                source=source,
                trigger_processing=True,
                metadata=params.get("metadata"),
            )
        elif kind == "message" and source == EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT:
            # The participant here is the agent, which only the server knows — construct it
            # rather than trusting the caller's data (which would otherwise be stored raw).
            event = (
                await self._session_module.create_human_agent_on_behalf_of_ai_agent_message_event(
                    session_id=session_id,
                    message=message,
                    metadata=params.get("metadata"),
                )
            )
        elif kind == "message":
            # Any other message source must carry its own participant in the data. The engine
            # reads participant on every turn to build the session history, so a message
            # without one corrupts the session — reject it here rather than store it.
            data = params.get("data", {})
            participant = data.get("participant") if isinstance(data, Mapping) else None
            if not isinstance(participant, Mapping) or not participant.get("display_name"):
                raise ValueError("A message event requires a 'participant' with a 'display_name'")
            event = await self._session_module.create_event(
                session_id=session_id,
                kind=EventKind.MESSAGE,
                data=data,
                source=source,
                metadata=params.get("metadata"),
            )
        else:
            event = await self._session_module.create_event(
                session_id=session_id,
                kind=EventKind(kind),
                data=params.get("data", {}),
                source=source,
                metadata=params.get("metadata"),
            )

        return {"event_id": event.id, "offset": event.offset}

    async def list_events(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = SessionId(params["session_id"])
        min_offset = params.get("min_offset", 0)
        source = _parse_event_source(params["source"]) if params.get("source") else None
        kinds = _parse_event_kinds(params.get("kinds"))
        trace_id = params.get("trace_id")
        wait_for_data = params.get("wait_for_data", 60)

        await self._read_parent_session(session_id)

        if wait_for_data > 0:
            has_events = await self._session_module.wait_for_more_events(
                session_id=session_id,
                min_offset=min_offset,
                source=source,
                kinds=kinds,
                trace_id=trace_id,
                timeout=Timeout(wait_for_data),
            )
            if not has_events:
                await self._read_parent_session(session_id)
                return {"events": []}

        try:
            events = await self._session_module.find_events(
                session_id=session_id,
                min_offset=min_offset,
                source=source,
                kinds=kinds,
                trace_id=trace_id,
            )
        except ItemNotFoundError:
            await self._read_parent_session(session_id)
            raise

        if not events:
            await self._read_parent_session(session_id)

        return {"events": [self._serialize_event(e) for e in events]}

    async def stream_events(self, params: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        session_id = SessionId(params["session_id"])
        current_offset = params.get("min_offset", 0)
        source = _parse_event_source(params["source"]) if params.get("source") else None
        kinds = _parse_event_kinds(params.get("kinds"))
        trace_id = params.get("trace_id")
        wait_for_data = params.get("wait_for_data", 60)

        while True:
            has_events = await self._session_module.wait_for_more_events(
                session_id=session_id,
                min_offset=current_offset,
                source=source,
                kinds=kinds,
                trace_id=trace_id,
                timeout=Timeout(wait_for_data),
            )
            if not has_events:
                break

            events = await self._session_module.find_events(
                session_id=session_id,
                min_offset=current_offset,
                source=source,
                kinds=kinds,
                trace_id=trace_id,
            )

            for event in events:
                yield self._serialize_event(event)
                current_offset = max(current_offset, event.offset + 1)

    async def stream_event(self, params: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        session_id = SessionId(params["session_id"])
        event_id = EventId(params["event_id"])
        wait_for_data = params.get("wait_for_data", 60)
        chunk_count = 0

        while True:
            event = await self._session_module.read_event(
                session_id=session_id,
                event_id=event_id,
            )
            yield self._serialize_event(event)

            data = cast(dict[str, Any], event.data)
            if "chunks" not in data:
                break

            chunks = cast(list[str | None], data["chunks"])
            chunk_count = len(chunks)
            if chunks and chunks[-1] is None:
                break

            has_new_chunks = await self._session_module.wait_for_new_streaming_chunks(
                session_id=session_id,
                event_id=event_id,
                last_known_chunk_count=chunk_count,
                timeout=Timeout(wait_for_data),
            )
            if not has_new_chunks:
                break

    async def create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        # The platform creates sessions here on a browser's behalf and passes the
        # browser it resolved; the tunnel is authenticated, so it is trusted.
        parlant_metadata: dict[str, Any] | None = None
        supplied_client = params.get("client")

        if isinstance(supplied_client, dict):
            ip = supplied_client.get("ip")
            origin = supplied_client.get("origin")
            client = await self._session_client_recorder.record(
                ip if isinstance(ip, str) else None,
                origin if isinstance(origin, str) else None,
            )
            if client is not None:
                parlant_metadata = {SESSION_CLIENT_METADATA_KEY: dict(client)}

        session = await self._session_module.create(
            customer_id=CustomerId(params["customer_id"])
            if params.get("customer_id")
            else CustomerStore.GUEST_ID,
            agent_id=AgentId(params["agent_id"]),
            title=params.get("title"),
            allow_greeting=params.get("allow_greeting", False),
            metadata=params.get("metadata"),
            parlant_metadata=parlant_metadata,
        )
        return self._serialize_session(session)

    async def read_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session = await self._read_parent_session(SessionId(params["session_id"]))
        return self._serialize_session(session)

    async def _read_parent_session(self, session_id: SessionId) -> Session:
        # This operation reads only the parent, so unrelated missing resources
        # (including an event sharing its ID) cannot imply session deletion.
        try:
            return await self._session_module.read(session_id=session_id)
        except ItemNotFoundError as exc:
            if exc.item_id == UniqueId(session_id) and exc.item_type in (None, "session"):
                raise ItemNotFoundError(UniqueId(session_id), item_type="session") from exc
            raise

    async def list_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        cursor = decode_cursor(params["cursor"]) if params.get("cursor") else None
        sort_direction = _parse_sort_direction(params.get("sort_direction"))
        labels = set(params["labels"]) if params.get("labels") else None
        min_modified_utc = (
            datetime.fromisoformat(params["min_modified_utc"])
            if params.get("min_modified_utc")
            else None
        )

        result = await self._session_module.find(
            agent_id=AgentId(params["agent_id"]) if params.get("agent_id") else None,
            customer_id=CustomerId(params["customer_id"]) if params.get("customer_id") else None,
            limit=params.get("limit"),
            cursor=cursor,
            sort_direction=sort_direction,
            labels=labels,
            min_modified_utc=min_modified_utc,
        )
        return {
            "sessions": [self._serialize_session(s) for s in result.items],
            "total_count": result.total_count,
            "has_more": result.has_more,
            "next_cursor": encode_cursor(result.next_cursor) if result.next_cursor else None,
        }

    async def update_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = SessionId(params.pop("session_id"))
        update_params: SessionUpdateParamsModel = {}
        if "customer_id" in params:
            update_params["customer_id"] = CustomerId(params["customer_id"])
        if "agent_id" in params:
            update_params["agent_id"] = AgentId(params["agent_id"])
        if "mode" in params:
            update_params["mode"] = params["mode"]
        if "title" in params:
            update_params["title"] = params["title"]
        if "agent_states" in params:
            update_params["agent_states"] = params["agent_states"]
        metadata_update: EventMetadataUpdateParamsModel | None = None
        if "metadata" in params:
            requested_metadata = dict(params["metadata"])
            metadata.ensure_public_metadata(requested_metadata)
            current = await self._session_module.read(session_id)
            metadata_update = {
                "set": requested_metadata,
                "unset": [
                    key
                    for key in current.metadata
                    if key != metadata.PARLANT_METADATA_KEY and key not in requested_metadata
                ],
            }
        session = await self._session_module.update(
            session_id=session_id,
            params=update_params,
            metadata=metadata_update,
        )
        return self._serialize_session(session)

    async def delete_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = SessionId(params["session_id"])
        try:
            await self._session_module.delete(session_id=session_id)
        except ItemNotFoundError:
            # A delete can fail on related data; confirm the parent independently.
            await self._read_parent_session(session_id)
            raise
        return {}

    async def delete_many_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        result = await self._session_module.find(
            agent_id=AgentId(params["agent_id"]) if params.get("agent_id") else None,
            customer_id=CustomerId(params["customer_id"]) if params.get("customer_id") else None,
            limit=None,
        )

        session_ids = [session.id for session in result.items]
        if any(not isinstance(session_id, str) or not session_id for session_id in session_ids):
            raise ValueError("Invalid session ID returned by source")

        deleted_session_ids: list[str] = []
        failed_session_ids: list[str] = []
        # Federated stores can return the same session more than once.
        for session_id in dict.fromkeys(session_ids):
            try:
                await self.delete_session({"session_id": session_id})
            except ItemNotFoundError as exc:
                if exc.item_type == "session" and exc.item_id == UniqueId(session_id):
                    # Removed since listing: the parent read confirmed absence.
                    deleted_session_ids.append(session_id)
                else:
                    failed_session_ids.append(session_id)
            except Exception:
                failed_session_ids.append(session_id)
            else:
                deleted_session_ids.append(session_id)

        return {
            "deleted_session_ids": deleted_session_ids,
            "failed_session_ids": failed_session_ids,
        }

    async def read_event(self, params: dict[str, Any]) -> dict[str, Any]:
        event = await self._session_module.read_event(
            session_id=SessionId(params["session_id"]),
            event_id=EventId(params["event_id"]),
        )
        return self._serialize_event(event)

    async def update_event(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = SessionId(params["session_id"])
        event_id = EventId(params["event_id"])
        requested_metadata = dict(params.get("metadata", {}))
        metadata.ensure_public_metadata(requested_metadata)
        current = await self._session_module.read_event(
            session_id=session_id,
            event_id=event_id,
        )
        event = await self._session_module.update_event(
            session_id=session_id,
            event_id=event_id,
            params={
                "metadata": {
                    "set": requested_metadata,
                    "unset": [
                        key
                        for key in current.metadata
                        if key != metadata.PARLANT_METADATA_KEY and key not in requested_metadata
                    ],
                }
            },
        )
        return self._serialize_event(event)

    async def delete_events(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._session_module.delete_events(
            session_id=SessionId(params["session_id"]),
            min_offset=params["min_offset"],
        )
        return {}

    @staticmethod
    def _serialize_session(session: Any) -> dict[str, Any]:
        return {
            "session_id": session.id,
            "creation_utc": session.creation_utc.isoformat(),
            "modified_utc": session.modified_utc.isoformat(),
            "customer_id": session.customer_id,
            "agent_id": session.agent_id,
            "mode": session.mode,
            "title": session.title,
            "metadata": dict(session.metadata) if session.metadata else {},
            "labels": list(session.labels),
        }

    @staticmethod
    def _serialize_event(event: Any) -> dict[str, Any]:
        return {
            "id": event.id,
            "offset": event.offset,
            "source": (
                _EVENT_SOURCE_WIRE_NAMES[event.source]
                if isinstance(event.source, EventSource)
                else SessionTunnelHandlers._serialize_scalar(event.source)
            ),
            "kind": SessionTunnelHandlers._serialize_scalar(event.kind),
            "creation_utc": event.creation_utc.isoformat(),
            "modified_utc": event.modified_utc.isoformat(),
            "trace_id": event.trace_id,
            "data": event.data,
            "metadata": dict(event.metadata) if event.metadata else {},
            "deleted": event.deleted,
        }

    @staticmethod
    def _serialize_scalar(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value

        return value
