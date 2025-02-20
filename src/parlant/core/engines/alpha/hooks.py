from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Sequence

from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.engines.types import Context
from parlant.core.guidelines import Guideline


class LifecycleHookResult(Enum):
    CALL_NEXT = auto()
    """Runs the next hook in the chain, if any"""

    RESOLVE = auto()
    """Returns without running the next hooks in the chain"""

    BAIL = auto()
    """Returns without running the next hooks in the chain, and quietly discards the current execution.

    For most hooks, this completely bails out of the processing execution, dropping the response to the customer.
    Specifically for preparation iterations, this immediately signals that preparation is complete.
    """


@dataclass(frozen=False)
class LifecycleHooks:
    on_error: list[Callable[[Context, EventEmitter, Exception], Awaitable[LifecycleHookResult]]] = (
        field(default_factory=list)
    )
    """Called when the engine has encountered a runtime error"""

    on_acknowledging: list[Callable[[Context, EventEmitter], Awaitable[LifecycleHookResult]]] = (
        field(default_factory=list)
    )
    """Called just before emitting an acknowledgement status event"""

    on_acknowledged: list[Callable[[Context, EventEmitter], Awaitable[LifecycleHookResult]]] = (
        field(default_factory=list)
    )
    """Called right after emitting an acknowledgement status event"""

    on_preparing: list[Callable[[Context, EventEmitter], Awaitable[LifecycleHookResult]]] = field(
        default_factory=list
    )
    """Called just before beginning the preparation iterations"""

    on_preparation_iteration_start: list[
        Callable[
            [Context, EventEmitter, list[EmittedEvent]],
            Awaitable[LifecycleHookResult],
        ]
    ] = field(default_factory=list)
    """Called just before beginning a preparation iteration"""

    on_preparation_iteration_end: list[
        Callable[
            [Context, EventEmitter, list[EmittedEvent], Sequence[Guideline]],
            Awaitable[LifecycleHookResult],
        ]
    ] = field(default_factory=list)
    """Called right after finishing a preparation iteration"""

    on_generating_messages: list[
        Callable[
            [Context, EventEmitter, list[EmittedEvent], Sequence[Guideline]],
            Awaitable[LifecycleHookResult],
        ]
    ] = field(default_factory=list)
    """Called just before generating messages"""

    on_generated_messages: list[
        Callable[
            [Context, EventEmitter, Sequence[EmittedEvent], Sequence[Guideline]],
            Awaitable[LifecycleHookResult],
        ]
    ] = field(default_factory=list)
    """Called right after generating messages"""

    async def call_on_error(
        self, context: Context, emitter: EventEmitter, exception: Exception
    ) -> bool:
        return await self._call_hook(self.on_error, context, emitter, exception)

    async def call_on_acknowledging(self, context: Context, emitter: EventEmitter) -> bool:
        return await self._call_hook(self.on_acknowledging, context, emitter)

    async def call_on_acknowledged(self, context: Context, emitter: EventEmitter) -> bool:
        return await self._call_hook(self.on_acknowledged, context, emitter)

    async def call_on_preparing(self, context: Context, emitter: EventEmitter) -> bool:
        return await self._call_hook(self.on_preparing, context, emitter)

    async def call_on_preparation_iteration_start(
        self,
        context: Context,
        emitter: EventEmitter,
        events: list[EmittedEvent],
    ) -> bool:
        return await self._call_hook(self.on_preparation_iteration_start, context, emitter, events)

    async def call_on_preparation_iteration_end(
        self,
        context: Context,
        emitter: EventEmitter,
        events: list[EmittedEvent],
        guidelines: Sequence[Guideline],
    ) -> bool:
        return await self._call_hook(
            self.on_preparation_iteration_end, context, emitter, events, guidelines
        )

    async def call_on_generating_messages(
        self,
        context: Context,
        emitter: EventEmitter,
        events: list[EmittedEvent],
        guidelines: Sequence[Guideline],
    ) -> bool:
        return await self._call_hook(
            self.on_generating_messages, context, emitter, events, guidelines
        )

    async def call_on_generated_messages(
        self,
        context: Context,
        emitter: EventEmitter,
        events: Sequence[EmittedEvent],
        guidelines: Sequence[Guideline],
    ) -> bool:
        return await self._call_hook(
            self.on_generated_messages, context, emitter, events, guidelines
        )

    async def _call_hook(self, hook: Any, *args: Any) -> bool:
        for callable in hook:
            match await callable(*args):
                case LifecycleHookResult.CALL_NEXT:
                    continue
                case LifecycleHookResult.RESOLVE:
                    return True
                case LifecycleHookResult.BAIL:
                    return False
        return True
