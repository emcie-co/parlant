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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

from parlant.core.agents import AgentId
from parlant.core.sessions import SessionId
from parlant.core.emissions import EventEmitter


@dataclass(frozen=True)
class Context:
    session_id: SessionId
    agent_id: AgentId


class UtteranceRationale(Enum):
    UNSPECIFIED = auto()
    BUY_TIME = auto()
    FOLLOW_UP = auto()


@dataclass(frozen=True)
class UtteranceRequest:
    action: str
    rationale: UtteranceRationale


class Engine(ABC):
    @abstractmethod
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool: ...

    @abstractmethod
    async def utter(
        self,
        context: Context,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> bool: ...
