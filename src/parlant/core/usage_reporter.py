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

from collections import OrderedDict
from typing import Callable, Mapping

from parlant.core.nlp.common import UsageInfo
from parlant.core.tracer import Tracer


_USAGE_CACHE_MAX_SIZE = 512


UsageListener = Callable[[str, str, UsageInfo], None]
"""(trace_id, model_name, usage) — notified on every reported usage."""


class UsageReporter:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._cache: OrderedDict[str, dict[str, UsageInfo]] = OrderedDict()
        self._listeners: list[UsageListener] = []

    def add_listener(self, listener: UsageListener) -> None:
        """Register an observer for reported usage (e.g. the cost-control policy).

        The reporter stays pure accounting; listeners observe it. Listener
        exceptions are swallowed — report_usage runs on the hot path and
        accounting must never be affected by an observer."""
        self._listeners.append(listener)

    def report_usage(self, model_name: str, usage: UsageInfo) -> None:
        trace_id = self._tracer.trace_id
        if trace_id not in self._cache:
            self._cache[trace_id] = {}
            self._cache.move_to_end(trace_id)

            if len(self._cache) > _USAGE_CACHE_MAX_SIZE:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(trace_id)

        usage_by_model = self._cache[trace_id]
        usage_by_model[model_name] = self._add_usage(
            usage_by_model.get(model_name),
            usage,
        )

        for listener in self._listeners:
            try:
                listener(trace_id, model_name, usage)
            except Exception:
                pass  # see add_listener: observers must never affect accounting

    def get_usage(self) -> Mapping[str, UsageInfo]:
        trace_id = self._tracer.trace_id
        usage = self._cache.get(trace_id, {})
        if trace_id in self._cache:
            self._cache.move_to_end(trace_id)
        return dict(usage)

    def _add_usage(self, existing: UsageInfo | None, usage: UsageInfo) -> UsageInfo:
        if existing is None:
            return usage

        return UsageInfo(
            input_tokens=existing.input_tokens + usage.input_tokens,
            output_tokens=existing.output_tokens + usage.output_tokens,
            cached_input_tokens=existing.cached_input_tokens + usage.cached_input_tokens,
            extra=self._add_extra(existing.extra, usage.extra),
        )

    def _add_extra(
        self,
        left: Mapping[str, int | float | str],
        right: Mapping[str, int | float | str],
    ) -> Mapping[str, int | float | str]:
        result: dict[str, int | float | str] = dict(left)

        for key, value in right.items():
            existing = result.get(key)
            if isinstance(existing, (int, float)) and isinstance(value, (int, float)):
                result[key] = existing + value
            else:
                result[key] = value

        return result
