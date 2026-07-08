import contextvars
from dataclasses import dataclass


@dataclass
class _ContextData:
    origin: str | None = None


class RequestContext:
    def __init__(
        self,
    ) -> None:
        self._context_var = contextvars.ContextVar[_ContextData](
            "_parlant_request_context",
            default=_ContextData(),
        )

    def reset(self) -> None:
        self._context_var.set(_ContextData())

    def set_origin(self, origin: str) -> None:
        data = self._context_var.get()
        data.origin = origin

    def get_origin(self) -> str | None:
        return self._context_var.get().origin
