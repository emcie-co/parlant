# Copyright 2026 Emcie Co Ltd.
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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING, TypeAlias
from typing_extensions import override

from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.sessions import Session

if TYPE_CHECKING:
    from parlant.core.engines.engine_context import EngineContext


ModerationTag: TypeAlias = Literal[
    "jailbreak",
    "harassment",
    "hate",
    "illicit",
    "self-harm",
    "sexual",
    "violence",
]


@dataclass(frozen=True)
class CustomerModerationContext:
    session: Session
    message: str
    # The turn's generic engine context. Customer-message moderation runs at
    # the API boundary, BEFORE a turn starts, so it is None there; services
    # that want conversational context can read the session's history instead.
    engine_context: "EngineContext[Any, Any] | None" = None


@dataclass(frozen=True)
class AgentModerationContext:
    session: Session
    message: str
    # Agent-message moderation runs inside the response turn, so the full
    # engine context of that turn is available.
    engine_context: "EngineContext[Any, Any] | None" = None


ModerationContext: TypeAlias = "CustomerModerationContext | AgentModerationContext"


@dataclass(frozen=True)
class ModerationCheck:
    flagged: bool
    tags: list[ModerationTag]


class ModerationService(ABC):
    @abstractmethod
    async def moderate_customer(
        self,
        context: CustomerModerationContext,
    ) -> ModerationCheck: ...

    async def moderate_agent(
        self,
        context: AgentModerationContext,
    ) -> ModerationCheck:
        """Moderate an agent-authored message. Defaults to the same text
        moderation as customer messages; services with asymmetric behavior
        override this."""
        return await self.moderate_customer(
            CustomerModerationContext(
                session=context.session,
                message=context.message,
                engine_context=context.engine_context,
            )
        )


class BaseModerationService(ModerationService):
    def __init__(
        self,
        logger: Logger,
        meter: Meter,
    ) -> None:
        self.logger = logger
        self.meter = meter

        self._hist_moderation_request_duration = meter.create_duration_histogram(
            name="moderation",
            description="Duration of moderation requests",
        )

    @override
    async def moderate_customer(
        self,
        context: CustomerModerationContext,
    ) -> ModerationCheck:
        async with self._hist_moderation_request_duration.measure(
            attributes={"class.name": self.__class__.__qualname__}
        ):
            return await self.do_moderate(context)

    @abstractmethod
    async def do_moderate(
        self,
        context: ModerationContext,
    ) -> ModerationCheck: ...


class NoModeration(ModerationService):
    @override
    async def moderate_customer(
        self,
        context: CustomerModerationContext,
    ) -> ModerationCheck:
        return ModerationCheck(flagged=False, tags=[])
