import asyncio
from typing import Any
from fastapi import WebSocket
from typing_extensions import override

from parlant.core.common import UniqueId, generate_id
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.logging import CorrelationalLogger, LogLevel


class WebSocketLogger(CorrelationalLogger):
    def __init__(
        self,
        correlator: ContextualCorrelator,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        super().__init__(correlator, log_level, logger_id)

        self._messages: asyncio.Queue[Any] = asyncio.Queue()
        self._web_sockets: dict[UniqueId, WebSocket] = {}

    def message_queue(self) -> asyncio.Queue[Any]:
        return self._messages

    def _enqueue_message(self, level: str, message: str) -> None:
        payload = {
            "level": level,
            "correlation_id": self._correlator.correlation_id,
            "message": message,
        }
        self._messages.put_nowait(payload)

    def append(self, web_socket: WebSocket) -> UniqueId:
        uid = generate_id()
        self._web_sockets[uid] = web_socket
        return uid

    def remove(self, web_socket_id: UniqueId) -> WebSocket:
        return self._web_sockets.pop(web_socket_id)

    @override
    def debug(self, message: str) -> None:
        self._enqueue_message("DEBUG", message)

    @override
    def info(self, message: str) -> None:
        self._enqueue_message("INFO", message)

    @override
    def warning(self, message: str) -> None:
        self._enqueue_message("WARNING", message)

    @override
    def error(self, message: str) -> None:
        self._enqueue_message("ERROR", message)

    @override
    def critical(self, message: str) -> None:
        self._enqueue_message("CRITICAL", message)

    async def flush(self) -> None:
        while True:
            try:
                payload = await self._messages.get()

                for ws in self._web_sockets.values():
                    await ws.send_json(payload)

                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
