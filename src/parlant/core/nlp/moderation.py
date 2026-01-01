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
from typing import Literal, TypeAlias
from typing_extensions import override

from parlant.core.sessions import Session
from parlant.core.loggers import Logger
from parlant.core.meter import Meter


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


class BaseModerationService(ModerationService):
    def __init__(self, logger: Logger, meter: Meter) -> None:
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
        context: CustomerModerationContext,
    ) -> ModerationCheck: ...


class NoModeration(ModerationService):
    @override
    async def moderate_customer(
        self,
        context: CustomerModerationContext,
    ) -> ModerationCheck:
        return ModerationCheck(flagged=False, tags=[])
