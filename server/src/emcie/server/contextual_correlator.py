from contextlib import contextmanager
import contextvars
from typing import Iterator

from emcie.server.core.common import generate_id

_UNINITIALIZED = 0xC0FFEE


class ContextualCorrelator:
    def __init__(self) -> None:
        self._instance_id = generate_id()
        self.scopes = contextvars.ContextVar[list[str]](f"scope_ids_{self._instance_id}")

    @contextmanager
    def correlation_scope(self, scope_id: str) -> Iterator[None]:
        scopes: int | list[str] = self.scopes.get(_UNINITIALIZED)

        if scopes == _UNINITIALIZED:
            self.scopes.set([])
            scopes = self.scopes.get()

        assert isinstance(scopes, list)

        scopes.append(scope_id)
        yield
        scopes.pop()

    @property
    def correlation_id(self) -> str:
        if scopes := self.scopes.get([]):
            chained_scopes = ".".join(scopes)
            return chained_scopes
        return "<main>"
