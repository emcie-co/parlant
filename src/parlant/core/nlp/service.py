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

from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import T, SchematicGenerator
from parlant.core.nlp.moderation import ModerationService


class NLPService(ABC):
    @abstractmethod
    async def get_schematic_generator(self, t: type[T]) -> SchematicGenerator[T]: ...

    @abstractmethod
    async def get_embedder(self) -> Embedder: ...

    @abstractmethod
    async def get_moderation_service(self) -> ModerationService: ...
