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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from parlant.core.common import CancellationSuppressionLatch
from parlant.core.engines.alpha.loaded_context import LoadedContext
from parlant.core.emissions import EmittedEvent
from parlant.core.nlp.generation_info import GenerationInfo


@dataclass(frozen=True)
class MessageEventComposition:
    generation_info: Mapping[str, GenerationInfo]
    events: Sequence[Optional[EmittedEvent]]


class MessageCompositionError(Exception):
    def __init__(self, message: str = "Message composition failed") -> None:
        super().__init__(message)


class MessageEventComposer:
    @abstractmethod
    async def generate_preamble(
        self,
        context: LoadedContext,
    ) -> Sequence[MessageEventComposition]: ...

    @abstractmethod
    async def generate_response(
        self,
        context: LoadedContext,
        latch: Optional[CancellationSuppressionLatch] = None,
    ) -> Sequence[MessageEventComposition]: ...
