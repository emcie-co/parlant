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

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Generic, Literal, Mapping, TypeVar, cast, get_args
from typing_extensions import override

from parlant.core.async_utils import Stopwatch
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.meter import DurationHistogram, Meter
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.tracer import Tracer
from parlant.core.usage_reporter import UsageReporter

T = TypeVar("T", bound=DefaultBaseModel)


# Normalized reasoning-depth hint accepted by every schematic generator under the
# ``"reasoning_effort"`` key in ``generate``'s ``hints``. Each provider maps it to
# its own model-specific reasoning configuration; models without reasoning support
# ignore it. ``"minimal"`` is the floor ("as little reasoning as possible") and
# resolves to *off* where the provider/model allows it.
ReasoningEffort = Literal["minimal", "low", "medium", "high"]

REASONING_EFFORT_HINT = "reasoning_effort"


# When present in ``generate``'s ``hints`` with a positive float value, the request
# is hedged: if the first attempt hasn't returned within this many seconds, an
# identical second request is fired and whichever completes first wins. When usage
# reporting is enabled, the loser finishes in the background so its usage can be
# accounted for; otherwise it is cancelled. Safe because schematic generation is
# a pure read with no side effects. Absent / ``None`` / ``<= 0`` disables hedging.
HEDGE_TIMEOUT_HINT = "hedge_timeout"


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
    # Warn when a single schematic generation takes longer than this (seconds).
    _SLOW_GENERATION_WARNING_SECONDS = 15.0
    # Warn when a single schematic generation emits more output tokens than this
    # (a lot, and expensive).
    _LARGE_OUTPUT_WARNING_TOKENS = 4000

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_name: str,
        usage_reporter: UsageReporter | None = None,
        *,
        provider_name: str,
    ) -> None:
        self.logger = logger
        self.tracer = tracer
        self.meter = meter
        self.model_name = model_name
        self.usage_reporter = usage_reporter
        self._provider_name = provider_name

        global _REQUEST_DURATION_HISTOGRAM
        if _REQUEST_DURATION_HISTOGRAM is None:
            _REQUEST_DURATION_HISTOGRAM = meter.create_duration_histogram(
                name="gen",
                description="Duration of generation requests in milliseconds",
            )

    @property
    def provider_name(self) -> str:
        """The provider this generator reports usage under (``provider:model``).

        Defaults to the adapter's own provider id; a hand-built instance may be
        given another, so one model reached through two providers (say, a direct
        account and a gateway at a different rate) is accounted and priced apart.
        """
        return self._provider_name

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
                result = await self._generate_maybe_hedged(prompt, hints)
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
                elapsed = start.elapsed
                if self.usage_reporter is not None:
                    self.usage_reporter.report_usage(
                        self.model_name,
                        result.info.usage,
                        provider_name=self.provider_name,
                    )

                if elapsed > self._SLOW_GENERATION_WARNING_SECONDS:
                    self.logger.warning(
                        f"Schematic generation took exceptionally long: {elapsed:.1f}s "
                        f"({self.__class__.__qualname__} / {self.model_name} / "
                        f"{self.schema.__name__})"
                    )

                output_tokens = result.info.usage.output_tokens
                if output_tokens > self._LARGE_OUTPUT_WARNING_TOKENS:
                    self.logger.warning(
                        f"Schematic generation produced a large (expensive) output: "
                        f"{output_tokens} output tokens ({self.__class__.__qualname__} / "
                        f"{self.model_name} / {self.schema.__name__})"
                    )

            return result

    async def _generate_maybe_hedged(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any],
    ) -> SchematicGenerationResult[T]:
        """Run ``do_generate``, optionally hedging on latency.

        When ``hints`` carries a positive ``HEDGE_TIMEOUT_HINT`` and the
        first attempt hasn't returned within that window, fire an identical
        second request and return whichever finishes first. When usage reporting
        is configured, a losing attempt finishes in the background and its token
        usage is reported against the originating trace; otherwise it is
        cancelled. Hedging only reacts to *slowness* — if the first attempt raises
        before the window elapses, that error propagates (errors are the
        provider's retry policy's job). Safe to double-fire because schematic
        generation is a pure read with no side effects.
        """
        delay = hints.get(HEDGE_TIMEOUT_HINT)
        should_hedge = isinstance(delay, (int, float)) and not isinstance(delay, bool) and delay > 0

        first: asyncio.Task[SchematicGenerationResult[T]] = asyncio.ensure_future(
            self.do_generate(prompt, hints)
        )

        if not should_hedge:
            return await first

        done, _ = await asyncio.wait({first}, timeout=float(cast(float, delay)))
        if first in done:
            return first.result()

        self.tracer.add_event(
            "gen.request_hedged",
            attributes={
                "model.name": self.model_name,
                "schema.name": self.schema.__name__,
                "hedge_timeout": float(cast(float, delay)),
            },
        )

        second: asyncio.Task[SchematicGenerationResult[T]] = asyncio.ensure_future(
            self.do_generate(prompt, hints)
        )
        attempts = [first, second]
        tracked_attempts: set[asyncio.Task[SchematicGenerationResult[T]]] = set()

        try:
            pending: set[asyncio.Task[SchematicGenerationResult[T]]] = {first, second}
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.exception() is None:
                        result = task.result()

                        if self.usage_reporter is not None:
                            trace_id = self.tracer.trace_id
                            for other in attempts:
                                if other is task:
                                    continue
                                tracked_attempts.add(other)
                                self.usage_reporter.track(
                                    self._report_hedge_attempt_usage(other, trace_id),
                                    trace_id=trace_id,
                                )

                        usage = replace(
                            result.info.usage,
                            requests=result.info.usage.requests + len(attempts) - 1,
                        )
                        return replace(result, info=replace(result.info, usage=usage))

            # Both attempts failed — surface the first attempt's error.
            error = first.exception()
            assert error is not None
            raise error
        finally:
            for task in attempts:
                if task not in tracked_attempts and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in attempts if task not in tracked_attempts),
                return_exceptions=True,
            )

    async def _report_hedge_attempt_usage(
        self,
        attempt: asyncio.Task[SchematicGenerationResult[T]],
        trace_id: str,
    ) -> None:
        """Report a losing hedge's tokens without counting its request twice."""
        assert self.usage_reporter is not None

        try:
            result = await attempt
        except asyncio.CancelledError:
            return
        except Exception:
            return

        self.usage_reporter.report_usage(
            self.model_name,
            replace(result.info.usage, requests=0),
            provider_name=self.provider_name,
            trace_id=trace_id,
        )


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
