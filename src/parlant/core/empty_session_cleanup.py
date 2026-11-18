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

"""Bounded cleanup of unused runtime sessions after thirty minutes.

Stores opt into atomic eligibility checks; declaration/read-only stores and
custom stores without that capability are never swept. A live-work pin protects
initialization and queued/running greetings, rather than a persisted status
that could be left behind after a crash.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from parlant.core.loggers import Logger, ScopedLogger
from parlant.core.persistence.common import Cursor
from parlant.core.sessions import SessionStore
from parlant.core.session_observability import SessionCleanupSettings
from parlant.core.store_provider import StoreProvider, StoreProviderHints


class EmptySessionCleanup:
    AGE = timedelta(minutes=30)
    INTERVAL_SECONDS = 60
    BATCH_SIZE = 100

    def __init__(self, store_provider: StoreProvider, logger: Logger) -> None:
        self._store_provider = store_provider
        self._logger = ScopedLogger(logger, "EmptySessionCleanup")
        settings = SessionCleanupSettings.from_environment()
        self.enabled = settings.enabled
        self._include_greetings = settings.include_greetings
        self._lock = asyncio.Lock()
        self._cursor: Cursor | None = None

    async def run_once(self, *, now: datetime, limit: int = BATCH_SIZE) -> int:
        if not self.enabled or limit <= 0:
            return 0
        async with self._lock:
            store = self._store_provider.get_store(
                SessionStore, StoreProviderHints(call_site="app")
            )
            cutoff = now.astimezone(timezone.utc) - self.AGE
            candidates = await store.list_empty_session_candidates(
                created_before=cutoff, limit=min(limit, self.BATCH_SIZE), cursor=self._cursor
            )
            removed = 0
            for session in candidates:
                if await store.delete_session_if_empty(
                    session.id,
                    created_before=cutoff,
                    include_greetings=self._include_greetings,
                ):
                    removed += 1
            # Advance past protected candidates too; an old pinned/greeting
            # session must not starve later empty ones. Retry a failed batch
            # from its old cursor; the conditional deletion is idempotent.
            self._cursor = candidates.next_cursor if candidates.has_more else None
            return removed

    async def run(self) -> None:
        while self.enabled:
            try:
                await self.run_once(now=datetime.now(timezone.utc))
            except Exception:
                # Never emit conversation content or a per-session log stream.
                self._logger.warning("Empty-session cleanup pass failed; retrying next minute")
            await asyncio.sleep(self.INTERVAL_SECONDS)
