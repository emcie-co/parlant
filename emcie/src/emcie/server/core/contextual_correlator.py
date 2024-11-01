from contextlib import contextmanager
import contextvars
from typing import Iterator

from emcie.server.core.common import generate_id

_UNINITIALIZED = 0xC0FFEE


class ContextualCorrelator:
    def __init__(self) -> None:
        self._instance_id = generate_id()

        self._scopes = contextvars.ContextVar[str](
            f"correlator_{self._instance_id}_scopes",
            default="",
        )

    @contextmanager
    def correlation_scope(self, scope_id: str) -> Iterator[None]:
        current_scopes = self._scopes.get()

        if current_scopes:
            new_scopes = current_scopes + f".{scope_id}"
        else:
            new_scopes = scope_id

        reset_token = self._scopes.set(new_scopes)

        yield

        self._scopes.reset(reset_token)

    @property
    def correlation_id(self) -> str:
        if scopes := self._scopes.get():
            return scopes
        return "<main>"
