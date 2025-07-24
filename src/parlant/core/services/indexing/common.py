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

import asyncio
from typing import Awaitable, Callable


class EvaluationError(Exception):
    def __init__(self, message: str = "Evaluation failed") -> None:
        super().__init__(message)


class ProgressReport:
    def __init__(self, progress_callback: Callable[[float], Awaitable[None]]) -> None:
        self._total = 0
        self._current = 0
        self._lock = asyncio.Lock()
        self._progress_callback = progress_callback

    @property
    def percentage(self) -> float:
        if self._total == 0:
            return 0.0
        return self._current / self._total * 100

    async def stretch(self, amount: int) -> None:
        async with self._lock:
            self._total += amount
            await self._progress_callback(self.percentage)

    async def increment(self, amount: int = 1) -> None:
        async with self._lock:
            self._current += amount
            await self._progress_callback(self.percentage)
