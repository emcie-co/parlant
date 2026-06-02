from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Mapping

from parlant.core.agents import AgentId
from parlant.core.app_modules.sessions import Moderation, SessionModule, SessionUpdateParamsModel
from parlant.core.customers import CustomerId
from parlant.core.loggers import Logger
from parlant.core.sessions import EventId, EventKind, EventSource, SessionId


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
        logger: Logger | None = None,
    ) -> None:
        self._session_module = session_module
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
            "sessions.create_event": self._handle_create_event,
            "sessions.list_events": self._handle_list_events,
            "sessions.read_event": self._handle_read_event,
            "sessions.update_event": self._handle_update_event,
            "sessions.delete_events": self._handle_delete_events,
        }
        return handlers.get(method)

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
        events = await self._session_module.find_events(
            session_id=session_id,
            min_offset=params.get("min_offset", 0),
            source=EventSource(params["source"]) if params.get("source") else None,
            kinds=[EventKind(k) for k in params.get("kinds", [])],
            trace_id=params.get("trace_id"),
        )

        return {"events": [self._serialize_event(e) for e in events]}

    async def _handle_create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session = await self._session_module.create(
            customer_id=CustomerId(params["customer_id"]),
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
        result = await self._session_module.find(
            agent_id=AgentId(params["agent_id"]) if params.get("agent_id") else None,
            customer_id=CustomerId(params["customer_id"]) if params.get("customer_id") else None,
            limit=params.get("limit"),
        )
        return {
            "sessions": [self._serialize_session(s) for s in result.items],
            "total_count": result.total_count,
            "has_more": result.has_more,
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
        }

    @staticmethod
    def _serialize_event(event: Any) -> dict[str, Any]:
        return {
            "id": event.id,
            "offset": event.offset,
            "source": event.source,
            "kind": event.kind,
            "creation_utc": event.creation_utc.isoformat(),
            "data": event.data,
            "metadata": dict(event.metadata) if event.metadata else {},
        }
