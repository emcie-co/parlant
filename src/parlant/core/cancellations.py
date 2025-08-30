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
import asyncio
from typing import Any

_TASK_FLAG = "_cancel_suppressed"


def task_suppression_enabled(task: asyncio.Task[object]) -> bool:
    return bool(getattr(task, _TASK_FLAG, False))


class CancellationSuppressionLatch:
    def __init__(self) -> None:
        self._suppressed = False

    @staticmethod
    def enabled_for_task(task: asyncio.Task[Any]) -> bool:
        return task_suppression_enabled(task)

    def __enter__(self) -> "CancellationSuppressionLatch":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        if self._suppressed:
            t = asyncio.current_task()

            if t is not None and hasattr(t, _TASK_FLAG):
                delattr(t, _TASK_FLAG)

            if exc_type is not None and issubclass(exc_type, asyncio.CancelledError):
                return True

        return False

    def enable(self) -> None:
        if self._suppressed:
            return

        t = asyncio.current_task()
        if t is None:
            return

        setattr(t, _TASK_FLAG, True)
        self._suppressed = True
