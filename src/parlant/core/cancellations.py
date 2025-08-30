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


class _LatchShim:
    def __init__(self) -> None:
        self.enabled = False


_CONTEXTUAL_LATCH = contextvars.ContextVar[_LatchShim | None](
    "_cancellation_suppression_latch",
    default=None,
)


def initialize_contextual_suppression_latch() -> None:
    _CONTEXTUAL_LATCH.set(_LatchShim())


def is_contextual_suppression_latch_enabled() -> bool:
    if latch := _CONTEXTUAL_LATCH.get():
        return latch.enabled
    return False


class CancellationSuppressionLatch:
    def __enter__(self) -> "CancellationSuppressionLatch":
        if latch := _CONTEXTUAL_LATCH.get():
            self._latch = latch
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
        self._latch.enabled = False

    def enable(self) -> None:
        self._latch.enabled = True
