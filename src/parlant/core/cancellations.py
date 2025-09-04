# Copyright 2025 Emcie Co Ltd.
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

from __future__ import annotations
import contextvars


_CURRENT_TASK_ID = contextvars.ContextVar[str | None]("_cancellations_current_task", default=None)
_SUPPRESSION_LATCHES = dict[str, bool]()


def initialize_contextual_suppression_latch(key: str) -> None:
    _SUPPRESSION_LATCHES[key] = False
    _CURRENT_TASK_ID.set(key)


def is_contextual_suppression_latch_enabled(key: str) -> bool:
    return _SUPPRESSION_LATCHES.get(key, False)


class CancellationSuppressionLatch:
    def __enter__(self) -> "CancellationSuppressionLatch":
        if task_id := _CURRENT_TASK_ID.get():
            self._task_id = task_id
        else:
            raise RuntimeError(
                "CancellationSuppressionLatch must be used within a supported context"
            )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        del _SUPPRESSION_LATCHES[self._task_id]

    def enable(self) -> None:
        _SUPPRESSION_LATCHES[self._task_id] = True
