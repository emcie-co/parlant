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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from parlant.core.engines.compass.preambles import PreambleConfiguration
from parlant.core.engines.compass.context import (
    DEFAULT_AGENT_MODERATION_FALLBACK,
    EngineContext,
)
from parlant.core.nlp.react import (
    ReasoningConfig,
    StepResult,
    Usage,
)


@dataclass(frozen=True)
class LoopJob:
    context: EngineContext
    system_instructions: str
    step_instructions: Callable[[EngineContext], Awaitable[str]] | None = None
    reasoning_config: ReasoningConfig | None = None
    preamble_config: PreambleConfiguration = PreambleConfiguration.default()
    # Last-resort text for a message blocked by agent moderation (see the Responder's
    # DI-configurable `moderation_fallback`).
    moderation_fallback: str = DEFAULT_AGENT_MODERATION_FALLBACK


@dataclass(frozen=True)
class LoopResult:
    job: LoopJob
    steps: Sequence[StepResult]

    @property
    def total_usage(self) -> Usage:
        return Usage(
            input_tokens=sum(step.usage.input_tokens for step in self.steps),
            cached_input_tokens=sum(step.usage.cached_input_tokens for step in self.steps),
            output_tokens=sum(step.usage.output_tokens for step in self.steps),
            reasoning_tokens=sum(step.usage.reasoning_tokens for step in self.steps),
            model_name=self.steps[-1].usage.model_name if self.steps else "N/A",
            ttft=self.steps[0].usage.ttft if self.steps else 0.0,
            requests=sum(step.usage.requests for step in self.steps),
        )


class Loop(ABC):
    """Base class for the agentic generation loop, which generates a response message
    based on the provided system prompt."""

    @abstractmethod
    async def warm_up(self, job: LoopJob) -> Usage:
        """Warm the generator's cache for ``job``'s stable prefix — the system
        instructions plus the conversation so far, without the dynamic per-turn
        instructions. A later :meth:`run` of an equivalent job then reads the
        provider's (content-addressed) cache instead of re-creating it.

        The job is not retained: the caller rebuilds an equivalent job for the
        actual turn. Returns the prefill :class:`Usage`.
        """
        ...

    @abstractmethod
    async def run(self, job: LoopJob) -> LoopResult: ...
