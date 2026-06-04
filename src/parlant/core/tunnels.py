from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping

from parlant.core.agents import AgentId
from parlant.core.app_modules.agents import AgentModule
from parlant.core.app_modules.common import decode_cursor, encode_cursor
from parlant.core.app_modules.customers import CustomerModule
from parlant.core.app_modules.sessions import Moderation, SessionModule, SessionUpdateParamsModel
from parlant.core.app_modules.tags import TagModule
from parlant.core.async_utils import Timeout
from parlant.core.common import ItemNotFoundError
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.loggers import Logger
from parlant.core.persistence.common import SortDirection
from parlant.core.sessions import EventId, EventKind, EventSource, SessionId
from parlant.core.tags import TagId


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


class TunnelRequest:
    """A request received from the platform through the tunnel."""

    def __init__(
        self,
        request_id: str,
        method: str,
        params: Mapping[str, Any],
    ) -> None:
        self.request_id = request_id
        self.method = method
        self.params = params


class TunnelResponse:
    """A response to send back to the platform through the tunnel."""

    def __init__(
        self,
        request_id: str,
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        self.request_id = request_id
        self.result = result
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"request_id": self.request_id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


class TunnelService(ABC):
    """Interface for a persistent tunnel connection to the platform."""

    @abstractmethod
    async def start(self) -> None:
        """Start the tunnel connection loop (blocking — run as a background task)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the tunnel."""
        ...


class TunnelRequestDispatcher:
    """Routes tunnel requests to the appropriate app_module methods."""

    def __init__(
        self,
        session_module: SessionModule,
        agent_module: AgentModule | None = None,
        customer_module: CustomerModule | None = None,
        tag_module: TagModule | None = None,
        logger: Logger | None = None,
    ) -> None:
        self._session_module = session_module
        self._agent_module = agent_module
        self._customer_module = customer_module
        self._tag_module = tag_module
        self._logger = logger

    async def dispatch(self, request: TunnelRequest) -> TunnelResponse:
        try:
            handler = self._get_handler(request.method)
            if handler is None:
                return TunnelResponse(
                    request_id=request.request_id,
                    error=f"Unknown method: {request.method}",
                )
            result = await handler(dict(request.params))
            return TunnelResponse(request_id=request.request_id, result=result)
        except Exception as e:
            return TunnelResponse(
                request_id=request.request_id,
                error=str(e),
            )

    def _get_handler(
        self, method: str
    ) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None:
        handlers: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
            "sessions.create": self._handle_create_session,
            "sessions.read": self._handle_read_session,
            "sessions.list": self._handle_list_sessions,
            "sessions.update": self._handle_update_session,
            "sessions.delete": self._handle_delete_session,
            "sessions.delete_many": self._handle_delete_many_sessions,
            "sessions.create_event": self._handle_create_event,
            "sessions.list_events": self._handle_list_events,
            "sessions.read_event": self._handle_read_event,
            "sessions.update_event": self._handle_update_event,
            "sessions.delete_events": self._handle_delete_events,
            "agents.list": self._handle_list_agents,
            "agents.retrieve": self._handle_retrieve_agent,
            "customers.retrieve": self._handle_retrieve_customer,
            "customers.list": self._handle_list_customers,
            "tags.list": self._handle_list_tags,
            "tags.retrieve": self._handle_retrieve_tag,
        }
        return handlers.get(method)

    async def _handle_list_tags(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._tag_module is None:
            raise RuntimeError("Tag module is not available")

        tags = await self._tag_module.find(name=params.get("name"))
        return {"tags": [self._serialize_tag(t) for t in tags]}

    async def _handle_retrieve_tag(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._tag_module is None:
            raise RuntimeError("Tag module is not available")

        try:
            tag = await self._tag_module.read(tag_id=TagId(params["tag_id"]))
        except ItemNotFoundError:
            return {"tag": None}
        return {"tag": self._serialize_tag(tag)}

    async def _handle_list_agents(self, params: dict[str, Any]) -> dict[str, Any]:
        del params
        if self._agent_module is None:
            raise RuntimeError("Agent module is not available")

        agents = await self._agent_module.find()
        return {"agents": [self._serialize_agent(a) for a in agents]}

    async def _handle_retrieve_agent(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._agent_module is None:
            raise RuntimeError("Agent module is not available")

        try:
            agent = await self._agent_module.read(agent_id=AgentId(params["agent_id"]))
        except ItemNotFoundError:
            return {"agent": None}
        return {"agent": self._serialize_agent(agent)}

    async def _handle_retrieve_customer(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._customer_module is None:
            raise RuntimeError("Customer module is not available")

        try:
            customer = await self._customer_module.read(
                customer_id=CustomerId(params["customer_id"]),
            )
        except ItemNotFoundError:
            return {"customer": None}
        return {"customer": self._serialize_customer(customer)}

    async def _handle_list_customers(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._customer_module is None:
            raise RuntimeError("Customer module is not available")

        cursor = decode_cursor(params["cursor"]) if params.get("cursor") else None
        sort_direction = _parse_sort_direction(params.get("sort_direction"))

        result = await self._customer_module.find(
            limit=params.get("limit"),
            cursor=cursor,
            sort_direction=sort_direction,
        )
        return {
            "customers": [self._serialize_customer(c) for c in result.items],
            "total_count": result.total_count,
            "has_more": result.has_more,
            "next_cursor": encode_cursor(result.next_cursor) if result.next_cursor else None,
        }

    async def _handle_create_event(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = SessionId(params["session_id"])
        kind = params.get("kind", "message")
        message = params.get("message", "")
        source_str = params.get("source", "customer")

        source = EventSource(source_str)

        if kind == "message" and source in (EventSource.CUSTOMER, EventSource.CUSTOMER_UI):
            event = await self._session_module.create_customer_message(
                session_id=session_id,
                moderation=Moderation.AUTO,
                message=message,
                source=source,
                trigger_processing=True,
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

    async def _handle_list_events(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = SessionId(params["session_id"])
        min_offset = params.get("min_offset", 0)
        source = EventSource(params["source"]) if params.get("source") else None
        kinds = _parse_event_kinds(params.get("kinds"))
        trace_id = params.get("trace_id")
        wait_for_data = params.get("wait_for_data", 0)

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
                return {"events": []}

        events = await self._session_module.find_events(
            session_id=session_id,
            min_offset=min_offset,
            source=source,
            kinds=kinds,
            trace_id=trace_id,
        )

        return {"events": [self._serialize_event(e) for e in events]}

    async def _handle_create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session = await self._session_module.create(
            customer_id=CustomerId(params["customer_id"])
            if params.get("customer_id")
            else CustomerStore.GUEST_ID,
            agent_id=AgentId(params["agent_id"]),
            title=params.get("title"),
            allow_greeting=params.get("allow_greeting", False),
            metadata=params.get("metadata"),
        )
        return self._serialize_session(session)

    async def _handle_read_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session = await self._session_module.read(
            session_id=SessionId(params["session_id"]),
        )
        return self._serialize_session(session)

    async def _handle_list_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        cursor = decode_cursor(params["cursor"]) if params.get("cursor") else None
        sort_direction = _parse_sort_direction(params.get("sort_direction"))
        labels = set(params["labels"]) if params.get("labels") else None

        result = await self._session_module.find(
            agent_id=AgentId(params["agent_id"]) if params.get("agent_id") else None,
            customer_id=CustomerId(params["customer_id"]) if params.get("customer_id") else None,
            limit=params.get("limit"),
            cursor=cursor,
            sort_direction=sort_direction,
            labels=labels,
        )
        return {
            "sessions": [self._serialize_session(s) for s in result.items],
            "total_count": result.total_count,
            "has_more": result.has_more,
            "next_cursor": encode_cursor(result.next_cursor) if result.next_cursor else None,
        }

    async def _handle_update_session(self, params: dict[str, Any]) -> dict[str, Any]:
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
        if "consumption_offsets" in params:
            update_params["consumption_offsets"] = params["consumption_offsets"]
        if "agent_states" in params:
            update_params["agent_states"] = params["agent_states"]
        if "metadata" in params:
            update_params["metadata"] = params["metadata"]
        session = await self._session_module.update(
            session_id=session_id,
            params=update_params,
        )
        return self._serialize_session(session)

    async def _handle_delete_session(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._session_module.delete(
            session_id=SessionId(params["session_id"]),
        )
        return {}

    async def _handle_delete_many_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        result = await self._session_module.find(
            agent_id=AgentId(params["agent_id"]) if params.get("agent_id") else None,
            customer_id=CustomerId(params["customer_id"]) if params.get("customer_id") else None,
            limit=None,
        )

        deleted_session_ids: list[str] = []
        failed_session_ids: list[str] = []
        for session in result.items:
            try:
                await self._session_module.delete(session_id=session.id)
                deleted_session_ids.append(session.id)
            except Exception:
                failed_session_ids.append(session.id)

        if failed_session_ids:
            raise RuntimeError("Failed to delete sessions: " + ", ".join(failed_session_ids))

        return {"deleted_session_ids": deleted_session_ids}

    async def _handle_read_event(self, params: dict[str, Any]) -> dict[str, Any]:
        event = await self._session_module.read_event(
            session_id=SessionId(params["session_id"]),
            event_id=EventId(params["event_id"]),
        )
        return self._serialize_event(event)

    async def _handle_update_event(self, params: dict[str, Any]) -> dict[str, Any]:
        event = await self._session_module.update_event(
            session_id=SessionId(params["session_id"]),
            event_id=EventId(params["event_id"]),
            params={"metadata": params.get("metadata", {})},
        )
        return self._serialize_event(event)

    async def _handle_delete_events(self, params: dict[str, Any]) -> dict[str, Any]:
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
            "source": TunnelRequestDispatcher._serialize_scalar(event.source),
            "kind": TunnelRequestDispatcher._serialize_scalar(event.kind),
            "creation_utc": event.creation_utc.isoformat(),
            "data": event.data,
            "metadata": dict(event.metadata) if event.metadata else {},
        }

    @staticmethod
    def _serialize_agent(agent: Any) -> dict[str, Any]:
        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "max_engine_iterations": agent.max_engine_iterations,
            "composition_mode": TunnelRequestDispatcher._serialize_scalar(agent.composition_mode),
            "message_output_mode": TunnelRequestDispatcher._serialize_scalar(
                agent.message_output_mode
            ),
            "tags": list(agent.tags),
        }

    @staticmethod
    def _serialize_scalar(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value

        return value

    @staticmethod
    def _serialize_customer(customer: Any) -> dict[str, Any]:
        return {
            "id": customer.id,
            "name": customer.name,
            "metadata": dict(customer.extra) if customer.extra else {},
            "tags": list(customer.tags),
            "creation_utc": customer.creation_utc.isoformat(),
        }

    @staticmethod
    def _serialize_tag(tag: Any) -> dict[str, Any]:
        return {
            "id": tag.id,
            "name": tag.name,
        }
