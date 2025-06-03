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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Awaitable, Callable, Optional, Sequence, TypeAlias

from parlant.core.engines.alpha.loaded_context import LoadedContext


class EngineHookResult(Enum):
    CALL_NEXT = auto()
    """Runs the next hook in the chain, if any"""

    RESOLVE = auto()
    """Returns without running the next hooks in the chain"""

    BAIL = auto()
    """Returns without running the next hooks in the chain, and quietly discards the current execution.

    For most hooks, this completely bails out of the processing execution, dropping the response to the customer.
    Specifically for preparation iterations, this immediately signals that preparation is complete.
    """


EngineHook: TypeAlias = Callable[[LoadedContext, Optional[Exception]], Awaitable[EngineHookResult]]


@dataclass(frozen=False)
class EngineHooks:
    on_error: list[EngineHook] = field(default_factory=list)
    """Called when the engine has encountered a runtime error"""

    on_acknowledging: list[EngineHook] = field(default_factory=list)
    """Called just before emitting an acknowledgement status event"""

    on_acknowledged: list[EngineHook] = field(default_factory=list)
    """Called right after emitting an acknowledgement status event"""

    on_generating_preamble: list[EngineHook] = field(default_factory=list)
    """Called just before generating the preamble message"""

    on_generated_preamble: list[EngineHook] = field(default_factory=list)
    """Called right after generating the preamble message"""

    on_preparing: list[EngineHook] = field(default_factory=list)
    """Called just before beginning the preparation iterations"""

    on_preparation_iteration_start: list[EngineHook] = field(default_factory=list)
    """Called just before beginning a preparation iteration"""

    on_preparation_iteration_end: list[EngineHook] = field(default_factory=list)
    """Called right after finishing a preparation iteration"""

    on_generating_messages: list[EngineHook] = field(default_factory=list)
    """Called just before generating messages"""

    on_generated_messages: list[EngineHook] = field(default_factory=list)
    """Called right after generating messages"""

    async def call_on_error(self, context: LoadedContext, exception: Exception) -> bool:
        return await self.call_hooks(self.on_error, context, exception)

    async def call_on_acknowledging(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_acknowledging, context)

    async def call_on_acknowledged(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_acknowledged, context)

    async def call_on_preparing(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_preparing, context)

    async def call_on_preparation_iteration_start(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_preparation_iteration_start, context)

    async def call_on_preparation_iteration_end(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_preparation_iteration_end, context)

    async def call_on_generating_preamble(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_generating_preamble, context)

    async def call_on_generated_preamble(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_generated_preamble, context)

    async def call_on_generating_messages(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_generating_messages, context)

    async def call_on_generated_messages(self, context: LoadedContext) -> bool:
        return await self.call_hooks(self.on_generated_messages, context)

    async def call_hooks(
        self,
        hooks: Sequence[EngineHook],
        context: LoadedContext,
        exc: Optional[Exception] = None,
    ) -> bool:
        for callable in hooks:
            match await callable(context, exc):
                case EngineHookResult.CALL_NEXT:
                    continue
                case EngineHookResult.RESOLVE:
                    return True
                case EngineHookResult.BAIL:
                    return False
        return True
