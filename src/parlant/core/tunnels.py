from abc import ABC, abstractmethod
from typing import Any, Mapping


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
