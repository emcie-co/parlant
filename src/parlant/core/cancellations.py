from __future__ import annotations
import contextvars
from typing import Optional


LATCH = contextvars.ContextVar[bool](
    "cancellation_suppression_enabled",
    default=False,
)


class CancellationSuppressionLatch:
    @staticmethod
    def enabled_for_context() -> bool:
        return LATCH.get()

    def __init__(self) -> None:
        self._reset_token: Optional[contextvars.Token[bool]] = None

    def __enter__(self) -> CancellationSuppressionLatch:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        if self._reset_token is not None:
            LATCH.reset(self._reset_token)

    def enable(self) -> None:
        self._reset_token = LATCH.set(True)
