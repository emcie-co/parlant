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

"""Observing how many queries a store issues, and whether they overlap.

Both properties are invisible from a store's return value: an N+1 read and a
batched one produce identical entities, and a concurrent fan-out produces the
same results as a sequential one. So these tests watch the only boundary where
the difference is real — the document collection.

The instrumentation also yields inside each query. ``TransientDocumentCollection``
never awaits internally, so a gathered coroutine would otherwise run from start
to finish without suspending and genuinely concurrent code would look
sequential. The yield makes overlap *observable*; it cannot manufacture it,
since sequential code can never have two queries in flight however often the
collection suspends.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from parlant.adapters.db.transient import TransientDocumentCollection


@dataclass
class QueryLog:
    """Every query issued against an instrumented collection, in order."""

    collections: list[str] = field(default_factory=list)
    max_queries_in_flight: int = 0

    def count_for(self, collection_name: str) -> int:
        return sum(1 for name in self.collections if name == collection_name)

    def count_matching(self, name_suffix: str) -> int:
        return sum(1 for name in self.collections if name.endswith(name_suffix))

    def reset(self) -> None:
        self.collections.clear()
        self.max_queries_in_flight = 0


def instrument_queries(monkeypatch: pytest.MonkeyPatch) -> QueryLog:
    """Record every ``find``/``find_one`` on transient collections for one test.

    ``monkeypatch`` restores the originals when the test ends, so the patch
    never leaks into another test.
    """
    log = QueryLog()
    in_flight = 0

    original_find = TransientDocumentCollection.find
    original_find_one = TransientDocumentCollection.find_one

    def instrumented(original: Any) -> Any:
        async def query(
            self: TransientDocumentCollection[Any],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            nonlocal in_flight

            log.collections.append(self._name)

            in_flight += 1
            log.max_queries_in_flight = max(log.max_queries_in_flight, in_flight)

            try:
                # The suspension point: see the module docstring.
                await asyncio.sleep(0)
                return await original(self, *args, **kwargs)
            finally:
                in_flight -= 1

        return query

    monkeypatch.setattr(TransientDocumentCollection, "find", instrumented(original_find))
    monkeypatch.setattr(TransientDocumentCollection, "find_one", instrumented(original_find_one))

    return log
