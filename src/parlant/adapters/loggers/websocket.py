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

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any
from fastapi import WebSocket
from typing_extensions import override

from parlant.core.engines.alpha.entity_context import EntityContext
from parlant.core.common import UniqueId, generate_id
from parlant.core.tracer import Tracer
from parlant.core.loggers import TracingLogger, LogLevel


@dataclass(frozen=True)
class WebSocketSubscription:
    socket: WebSocket
    expiration: asyncio.Event


class WebSocketLogger(TracingLogger):
    def __init__(
        self,
        tracer: Tracer,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        super().__init__(tracer, log_level, logger_id)

        self._message_queue = deque[Any]()
        self._messages_in_queue = asyncio.Semaphore(0)
        self._socket_subscriptions: dict[UniqueId, WebSocketSubscription] = {}
        self._lock = asyncio.Lock()

    def _enqueue_message(self, timestamp: str, level: str, message: str) -> None:
        payload = {
            "level": level,
            "trace_id": self._tracer.trace_id,
            "message": message,
        }

        if context_creation := EntityContext.get_context_creation():
            payload["message"] = f"[T+{round(context_creation.elapsed, 3)}s]{message}"

        self._message_queue.append(payload)
        self._messages_in_queue.release()

    async def subscribe(self, web_socket: WebSocket) -> WebSocketSubscription:
        socket_id = generate_id()

        subscription = WebSocketSubscription(web_socket, asyncio.Event())

        async with self._lock:
            self._socket_subscriptions[socket_id] = subscription

        return subscription

    def _timestamp(self) -> str:
        return round(asyncio.get_event_loop().time(), 3).__str__()

    @override
    def trace(self, message: str) -> None:
        self._enqueue_message(self._timestamp(), "TRACE", f"{self.current_scope} {message}")

    @override
    def debug(self, message: str) -> None:
        self._enqueue_message(self._timestamp(), "DEBUG", f"{self.current_scope} {message}")

    @override
    def info(self, message: str) -> None:
        self._enqueue_message(self._timestamp(), "INFO", f"{self.current_scope} {message}")

    @override
    def warning(self, message: str) -> None:
        self._enqueue_message(self._timestamp(), "WARNING", f"{self.current_scope} {message}")

    @override
    def error(self, message: str) -> None:
        self._enqueue_message(self._timestamp(), "ERROR", f"{self.current_scope} {message}")

    @override
    def critical(self, message: str) -> None:
        self._enqueue_message(self._timestamp(), "CRITICAL", f"{self.current_scope} {message}")

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
