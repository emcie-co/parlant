from contextlib import contextmanager
import contextvars
from typing import Iterator

from emcie.server.core.common import generate_id


class ContextualCorrelator:
    def __init__(self) -> None:
        self._instance_id = generate_id()
        self.scopes = contextvars.ContextVar[list[str]](
            f"correlation_ids_{self._instance_id}",
            default=[],
        )

    @contextmanager
    def correlation_scope(self, scope_id: str) -> Iterator[None]:
        self.scopes.get().append(scope_id)
        yield
        self.scopes.get().pop()

    @property
    def correlation_id(self) -> str:
        if scopes := self.scopes.get():
            chained_scopes = ".".join(scopes)
            return chained_scopes
        return "<main>"
