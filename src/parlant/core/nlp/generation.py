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
from functools import cached_property
from typing import Any, Generic, Mapping, TypeVar, cast, get_args
from typing_extensions import override

from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.tokenization import EstimatingTokenizer

T = TypeVar("T", bound=DefaultBaseModel)


@dataclass(frozen=True)
class SchematicGenerationResult(Generic[T]):
    """Result of a schematic generation operation."""

    content: T
    info: GenerationInfo


class SchematicGenerator(ABC, Generic[T]):
    """An interface for generating structured content based on a prompt."""

    @cached_property
    def schema(self) -> type[T]:
        """Return the schema type for the generated content."""

        orig_class = getattr(self, "__orig_class__")
        generic_args = get_args(orig_class)
        return cast(type[T], generic_args[0])

    @abstractmethod
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        """Generate content based on the provided prompt and hints."""
        ...

    @property
    @abstractmethod
    def id(self) -> str:
        """Return a unique identifier for the generator."""
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return the maximum number of tokens in the underlying model's context window."""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> EstimatingTokenizer:
        """Return a tokenizer that approximates that of the underlying model."""
        ...


class FallbackSchematicGenerator(SchematicGenerator[T]):
    """A generator that tries multiple generators in sequence until one succeeds."""

    def __init__(
        self,
        *generators: SchematicGenerator[T],
        logger: Logger,
    ) -> None:
        assert generators, "Fallback generator must be instantiated with at least 1 generator"

        self._generators = generators
        self._logger = logger

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        last_exception: Exception

        for index, generator in enumerate(self._generators):
            try:
                result = await generator.generate(prompt=prompt, hints=hints)
                return result
            except Exception as e:
                self._logger.warning(
                    f"Generator {index + 1}/{len(self._generators)} failed: {type(generator).__name__}: {e}"
                )
                last_exception = e

        raise last_exception

    @property
    @override
    def id(self) -> str:
        ids = ", ".join(g.id for g in self._generators)
        return f"fallback({ids})"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._generators[0].tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return min(*(g.max_tokens for g in self._generators))
