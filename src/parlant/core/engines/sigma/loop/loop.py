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

from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.engine_context import EngineContext
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.common import ModelSize
from parlant.core.nlp.react import (
    ReactGenerator,
    ReasoningConfig,
    StepResult,
    Usage,
)
from parlant.core.tracer import Tracer


@dataclass(frozen=True)
class LoopJob:
    context: EngineContext
    system_instructions: str
    turn_instructions: Callable[[EngineContext], Awaitable[str]] | None = None
    model_size: ModelSize = ModelSize.MEDIUM
    reasoning_config: ReasoningConfig | None = None


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
        )


class Loop(ABC):
    """Base class for the agentic generation loop, which generates a response message
    based on the provided system prompt."""

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        optimization_policy: OptimizationPolicy,
        react: ReactGenerator,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._optimization_policy = optimization_policy
        self._react = react

    @abstractmethod
    async def run(self, job: LoopJob) -> LoopResult: ...
