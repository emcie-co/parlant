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
from functools import cached_property
from typing import Any, AsyncIterator, Callable, Generic, Mapping, TypeVar, cast, get_args
from typing_extensions import override

from parlant.core.async_utils import Stopwatch
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.meter import DurationHistogram, Meter
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.tracer import Tracer

T = TypeVar("T", bound=DefaultBaseModel)


# ============================================================================
# Streaming Text Generator
# ============================================================================


class StreamingTextGenerationResult:
    """Result of a streaming text generation operation.

    Provides access to both the chunk stream and the generation info.
    The info property raises RuntimeError if accessed before the stream is fully consumed.
    """

    def __init__(
        self,
        stream: AsyncIterator[str | None],
        info_getter: Callable[[], GenerationInfo],
    ) -> None:
        self._stream = stream
        self._info_getter = info_getter

    @property
    def stream(self) -> AsyncIterator[str | None]:
        """The async iterator that yields text chunks, terminated by None."""
        return self._stream

    @property
    def info(self) -> GenerationInfo:
        """Generation info including usage statistics.

        Raises RuntimeError if accessed before the stream is fully consumed.
        """
        return self._info_getter()


class StreamingTextGenerator(ABC):
    """An interface for generating streaming text content based on a prompt.

    Unlike SchematicGenerator which returns structured content, this generator
    yields plain text chunks progressively, terminated by None to signal completion.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> StreamingTextGenerationResult:
        """Generate text content based on the provided prompt and hints.

        Returns a StreamingTextGenerationResult containing:
        - stream: AsyncIterator yielding text chunks, followed by None to signal completion
        - info: GenerationInfo with usage statistics (available after stream completes)
        """
        ...

    @property
    @abstractmethod
    def id(self) -> str:
        """Return a unique identifier for the generator."""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> EstimatingTokenizer:
        """Return a tokenizer that approximates that of the underlying model."""
        ...


_STREAMING_REQUEST_DURATION_HISTOGRAM: DurationHistogram | None = None


class BaseStreamingTextGenerator(StreamingTextGenerator):
    """Base class for streaming text generators with tracing and metrics."""

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter, model_name: str) -> None:
        self.logger = logger
        self.tracer = tracer
        self.meter = meter
        self.model_name = model_name

        global _STREAMING_REQUEST_DURATION_HISTOGRAM
        if _STREAMING_REQUEST_DURATION_HISTOGRAM is None:
            _STREAMING_REQUEST_DURATION_HISTOGRAM = meter.create_duration_histogram(
                name="stream",
                description="Duration of streaming generation requests in milliseconds",
            )

    @abstractmethod
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> tuple[AsyncIterator[str | None], Callable[[], UsageInfo]]:
        """Subclasses implement this to perform the actual generation.

        Returns:
            A tuple of:
            - AsyncIterator yielding text chunks, terminated by None
            - A callable that returns UsageInfo (may raise if called before stream completes)
        """
        ...

    @override
    def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> StreamingTextGenerationResult:
        assert _STREAMING_REQUEST_DURATION_HISTOGRAM is not None

        start = Stopwatch.start()
        stream_complete = False
        duration: float = 0.0
        usage_getter: Callable[[], UsageInfo] | None = None

        async def wrapped_stream() -> AsyncIterator[str | None]:
            nonlocal stream_complete, duration, usage_getter

            try:
                self.tracer.add_event(
                    "stream.request_started",
                    attributes={
                        "model.name": self.model_name,
                    },
                )

                inner_stream, usage_getter = await self.do_generate(prompt, hints)

                async for chunk in inner_stream:
                    yield chunk

                duration = start.elapsed
                stream_complete = True

                self.tracer.add_event(
                    "stream.request_completed",
                    attributes={
                        "model.name": self.model_name,
                        "duration": duration,
                    },
                )
            except Exception:
                duration = start.elapsed
                self.tracer.add_event(
                    "stream.request_failed",
                    attributes={
                        "model.name": self.model_name,
                        "duration": duration,
                    },
                )
                raise

        def info_getter() -> GenerationInfo:
            if not stream_complete:
                raise RuntimeError("Cannot access generation info before stream is fully consumed")
            assert usage_getter is not None
            return GenerationInfo(
                schema_name="streaming",
                model=self.id,
                duration=duration,
                usage=usage_getter(),
            )

        return StreamingTextGenerationResult(
            stream=wrapped_stream(),
            info_getter=info_getter,
        )


# ============================================================================
# Schematic Generator
# ============================================================================


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


_REQUEST_DURATION_HISTOGRAM: DurationHistogram | None = None


class BaseSchematicGenerator(SchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter, model_name: str) -> None:
        self.logger = logger
        self.tracer = tracer
        self.meter = meter
        self.model_name = model_name

        global _REQUEST_DURATION_HISTOGRAM
        if _REQUEST_DURATION_HISTOGRAM is None:
            _REQUEST_DURATION_HISTOGRAM = meter.create_duration_histogram(
                name="gen",
                description="Duration of generation requests in milliseconds",
            )

    @abstractmethod
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]: ...

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        assert _REQUEST_DURATION_HISTOGRAM is not None

        async with _REQUEST_DURATION_HISTOGRAM.measure(
            {
                "class.name": self.__class__.__qualname__,
                "model.name": self.model_name,
                "schema.name": self.schema.__name__,
            }
        ):
            start = Stopwatch.start()

            try:
                result = await self.do_generate(prompt, hints)
            except Exception:
                self.tracer.add_event(
                    "gen.request_failed",
                    attributes={
                        "model.name": self.model_name,
                        "schema.name": self.schema.__name__,
                        "duration": start.elapsed,
                    },
                )
                raise
            else:
                self.tracer.add_event(
                    "gen.request_completed",
                    attributes={
                        "model.name": self.model_name,
                        "schema.name": self.schema.__name__,
                        "duration": start.elapsed,
                    },
                )

            return result


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
