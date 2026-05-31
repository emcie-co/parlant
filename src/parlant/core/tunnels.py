from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Mapping

from parlant.core.app_modules.sessions import Moderation, SessionModule
from parlant.core.loggers import Logger
from parlant.core.sessions import EventKind, EventSource, SessionId


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
            "sessions.create_event": self._handle_create_event,
            "sessions.list_events": self._handle_list_events,
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

        return {
            "events": [
                {
                    "id": e.id,
                    "offset": e.offset,
                    "source": e.source,
                    "kind": e.kind,
                    "creation_utc": e.creation_utc.isoformat(),
                }
                for e in events
            ]
        }
