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
from typing_extensions import TypedDict, NotRequired

from parlant.core.nlp.common import ModelGeneration, ModelSize, ModelType
from parlant.core.nlp.embedding import Embedder, EmbedderHints
from parlant.core.nlp.generation import T, SchematicGenerator, StreamingTextGenerator
from parlant.core.nlp.moderation import ModerationService
from parlant.core.nlp.react import ReactGenerator

# Re-exports for backward compatibility — these symbols moved to dedicated
# modules but historically lived here.
__all__ = [
    "EmbedderHints",
    "ModelGeneration",
    "ModelSize",
    "ModelType",
    "NLPService",
    "SchematicGeneratorHints",
    "StreamingTextGeneratorHints",
]


class SchematicGeneratorHints(TypedDict, total=False):
    model_size: NotRequired[ModelSize]
    model_generation: NotRequired[ModelGeneration]
    model_type: NotRequired[ModelType]


class StreamingTextGeneratorHints(TypedDict, total=False):
    model_size: NotRequired[ModelSize]
    model_generation: NotRequired[ModelGeneration]


class NLPService(ABC):
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Return whether this NLP service supports streaming text generation."""
        ...

    @property
    def supports_react(self) -> bool:
        """Return whether this NLP service supports ReAct-style generation.

        Defaults to ``False``; services that back a ReactGenerator override this
        (and :meth:`get_react_generator`).
        """
        return False

    @abstractmethod
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> SchematicGenerator[T]: ...

    @abstractmethod
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        """Return a streaming text generator.

        Raises:
            NotImplementedError: If streaming is not supported (supports_streaming is False).
                Callers should check supports_streaming before calling this method.
        """
        ...

    async def get_react_generator(self) -> ReactGenerator:
        """Return a ReAct-style generator.

        Per-call model selection is done via the ``hints`` argument on
        :meth:`ReactGenerator.step` / :meth:`ReactGenerator.stream_step` /
        :meth:`ReactGenerator.run`; the generator itself is not bound to a size
        at construction time.

        Raises:
            NotImplementedError: If ReAct is not supported (supports_react is False).
                Callers should check supports_react before calling this method.
        """
        raise NotImplementedError("ReAct is not supported. Check supports_react first.")

    @abstractmethod
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder: ...

    @abstractmethod
    async def get_moderation_service(self) -> ModerationService: ...
