import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any
from fastapi import WebSocket
from typing_extensions import override

from parlant.core.common import UniqueId, generate_id
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.logging import CorrelationalLogger, LogLevel


@dataclass(frozen=True)
class WebSocketSubscription:
    socket: WebSocket
    expiration: asyncio.Event


class WebSocketLogger(CorrelationalLogger):
    def __init__(
        self,
        correlator: ContextualCorrelator,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        super().__init__(correlator, log_level, logger_id)

        self._message_queue = deque[Any]()
        self._messages_in_queue = asyncio.Semaphore(0)
        self._socket_subscriptions: dict[UniqueId, WebSocketSubscription] = {}
        self._lock = asyncio.Lock()

    def _enqueue_message(self, level: str, message: str) -> None:
        payload = {
            "level": level,
            "correlation_id": self._correlator.correlation_id,
            "message": message,
        }

        self._message_queue.append(payload)
        self._messages_in_queue.release()

    async def subscribe(self, web_socket: WebSocket) -> WebSocketSubscription:
        socket_id = generate_id()

        subscription = WebSocketSubscription(web_socket, asyncio.Event())

        async with self._lock:
            self._socket_subscriptions[socket_id] = subscription

        return subscription

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

    async def start(self) -> None:
        try:
            while True:
                try:
                    await self._messages_in_queue.acquire()
                    payload = self._message_queue.popleft()

                    async with self._lock:
                        socket_subscriptions = dict(self._socket_subscriptions)

                    expired_ids = set()

                    for socket_id, subscription in socket_subscriptions.items():
                        try:
                            await subscription.socket.send_json(payload)
                        except Exception:
                            expired_ids.add(socket_id)

                    async with self._lock:
                        for socket_id in expired_ids:
                            subscription = self._socket_subscriptions.pop(socket_id)
                            subscription.expiration.set()
                except asyncio.CancelledError:
                    return
        finally:
            async with self._lock:
                for socket_id, subscription in self._socket_subscriptions.items():
                    subscription.expiration.set()
