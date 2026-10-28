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

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Mapping

from parlant.core.session_clients import SessionClientRecorder
from parlant.core.application import Application
from parlant.core.health import HealthReporter, TunnelState
from parlant.core.loggers import Logger
from parlant.core.tunnel_handlers import create_tunnel_handler_registry
from parlant.core.common import ItemNotFoundError


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
        error: str | dict[str, str] | None = None,
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

    @abstractmethod
    def tunnel_state(self) -> TunnelState:
        """Report the tunnel's own condition for the `tunnel` health view.

        Part of the interface rather than something the view duck-types,
        because a tunnel that cannot say whether it is connected is the exact
        blind spot the view was added to close: a managed project nobody can
        reach otherwise reports itself perfectly healthy.
        """
        ...


class TunnelRequestDispatcher:
    """Routes tunnel requests to the per-domain handler registry."""

    def __init__(
        self,
        application: Application,
        health_reporter: HealthReporter,
        logger: Logger | None = None,
        session_client_recorder: SessionClientRecorder | None = None,
    ) -> None:
        self._handlers = create_tunnel_handler_registry(
            application, health_reporter, logger, session_client_recorder
        )
        self._logger = logger

    async def dispatch(self, request: TunnelRequest) -> TunnelResponse:
        try:
            handler = self._handlers.unary.get(request.method)
            if handler is None:
                return TunnelResponse(
                    request_id=request.request_id,
                    error=f"Unknown method: {request.method}",
                )
            result = await handler(dict(request.params))
            return TunnelResponse(request_id=request.request_id, result=result)
        except ItemNotFoundError as e:
            if e.item_type == "session" and e.item_id == request.params.get("session_id"):
                return TunnelResponse(
                    request_id=request.request_id,
                    error={
                        "code": "session_not_found",
                        "message": "Session not found",
                        "session_id": e.item_id,
                    },
                )
            return TunnelResponse(request_id=request.request_id, error=str(e))
        except Exception as e:
            return TunnelResponse(
                request_id=request.request_id,
                error=str(e),
            )

    async def dispatch_stream(self, request: TunnelRequest) -> AsyncIterator[dict[str, Any]]:
        handler = self._handlers.streams.get(request.method)
        if handler is None:
            raise ValueError(f"Unknown stream method: {request.method}")

        async for item in handler(dict(request.params)):
            yield item
