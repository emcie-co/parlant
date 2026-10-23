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
from collections import OrderedDict
from typing import Awaitable, Callable, Mapping

from parlant.core.nlp.common import UsageInfo, add_usage_extra_value
from parlant.core.tracer import Tracer


_USAGE_CACHE_MAX_SIZE = 512


UsageListener = Callable[[str, str, UsageInfo], None]
"""(trace_id, usage_key, usage) — notified on every reported usage.

The usage key is ``provider:model`` (see :func:`usage_key`)."""


def usage_key(provider_name: str, model_name: str) -> str:
    """The key one model served by one provider is accounted under.

    Spelled like a model route — ``provider:model`` — so the same model id
    reached through two providers (say, a direct account and a gateway, each
    at its own rate) is counted and priced apart."""

    return f"{provider_name}:{model_name}"


class UsageReporter:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._cache: OrderedDict[str, dict[str, UsageInfo]] = OrderedDict()
        self._listeners: list[UsageListener] = []
        self._pending: dict[str, set[asyncio.Task[None]]] = {}

    def add_listener(self, listener: UsageListener) -> None:
        """Register an observer for reported usage (e.g. the cost-control policy).

        The reporter stays pure accounting; listeners observe it. Listener
        exceptions are swallowed — report_usage runs on the hot path and
        accounting must never be affected by an observer."""
        self._listeners.append(listener)

    def report_usage(
        self,
        model_name: str,
        usage: UsageInfo,
        *,
        provider_name: str,
        trace_id: str | None = None,
    ) -> None:
        """Account one request's usage under ``provider_name:model_name``."""

        key = usage_key(provider_name, model_name)
        trace_id = trace_id if trace_id is not None else self._tracer.trace_id
        if trace_id not in self._cache:
            self._cache[trace_id] = {}
            self._cache.move_to_end(trace_id)

            if len(self._cache) > _USAGE_CACHE_MAX_SIZE:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(trace_id)

        usage_by_key = self._cache[trace_id]
        usage_by_key[key] = self._add_usage(usage_by_key.get(key), usage)

        for listener in self._listeners:
            try:
                listener(trace_id, key, usage)
            except Exception:
                pass  # see add_listener: observers must never affect accounting

    def track(self, work: Awaitable[None], *, trace_id: str | None = None) -> None:
        """Track deferred accounting work that must settle before a trace is persisted."""
        trace_id = trace_id if trace_id is not None else self._tracer.trace_id

        async def run() -> None:
            try:
                await work
            except asyncio.CancelledError:
                pass
            except Exception:
                pass  # deferred accounting must not affect the model hot path

        task = asyncio.create_task(run())
        pending = self._pending.setdefault(trace_id, set())
        pending.add(task)

        def discard(completed: asyncio.Task[None]) -> None:
            current = self._pending.get(trace_id)
            if current is None:
                return
            current.discard(completed)
            if not current:
                self._pending.pop(trace_id, None)

        task.add_done_callback(discard)

    async def wait_for_pending(self, trace_id: str | None = None) -> None:
        """Wait for deferred accounting associated with one trace."""

        trace_id = trace_id or self._tracer.trace_id
        while pending := tuple(self._pending.get(trace_id, ())):
            await asyncio.gather(*pending, return_exceptions=True)

    def get_usage(self) -> Mapping[str, UsageInfo]:
        """Return a detached snapshot for the current trace."""

        return self.snapshot_trace(self._tracer.trace_id)

    def snapshot_trace(self, trace_id: str) -> Mapping[str, UsageInfo]:
        """Return a detached usage snapshot for an explicit trace."""

        usage = self._cache.get(trace_id, {})
        if trace_id in self._cache:
            self._cache.move_to_end(trace_id)
        return dict(usage)

    def acknowledge_trace(self, trace_id: str) -> None:
        """Discard a trace only after its usage record is durable."""

        self._cache.pop(trace_id, None)
        self._pending.pop(trace_id, None)

    def _add_usage(self, existing: UsageInfo | None, usage: UsageInfo) -> UsageInfo:
        if existing is None:
            return usage

        return UsageInfo(
            input_tokens=existing.input_tokens + usage.input_tokens,
            output_tokens=existing.output_tokens + usage.output_tokens,
            cached_input_tokens=existing.cached_input_tokens + usage.cached_input_tokens,
            extra=self._add_extra(existing.extra, usage.extra),
            requests=existing.requests + usage.requests,
        )

    def _add_extra(
        self,
        left: Mapping[str, int | float | str],
        right: Mapping[str, int | float | str],
    ) -> Mapping[str, int | float | str]:
        result: dict[str, int | float | str] = dict(left)

        for key, value in right.items():
            existing = result.get(key)
            result[key] = add_usage_extra_value(key, existing, value)

        return result
