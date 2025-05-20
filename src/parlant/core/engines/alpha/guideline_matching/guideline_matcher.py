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
from itertools import chain
import time
from typing import Sequence

from parlant.core import async_utils
from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
)
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline
from parlant.core.sessions import Event
from parlant.core.loggers import Logger


@dataclass(frozen=True)
class GuidelineMatchingContext:
    agent: Agent
    customer: Customer
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    staged_events: Sequence[EmittedEvent]


@dataclass(frozen=True)
class GuidelineMatchingResult:
    total_duration: float
    batch_count: int
    batch_generations: Sequence[GenerationInfo]
    batches: Sequence[Sequence[GuidelineMatch]]

    @cached_property
    def matches(self) -> Sequence[GuidelineMatch]:
        return list(chain.from_iterable(self.batches))


@dataclass(frozen=True)
class GuidelineMatchingBatchResult:
    matches: Sequence[GuidelineMatch]
    generation_info: GenerationInfo


class GuidelineMatchingBatch(ABC):
    @abstractmethod
    async def process(self) -> GuidelineMatchingBatchResult: ...


class GuidelineMatchingStrategy(ABC):
    @abstractmethod
    async def create_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]: ...


class GuidelineMatchingStrategyResolver(ABC):
    @abstractmethod
    async def resolve(self, guideline: Guideline) -> GuidelineMatchingStrategy: ...


class GuidelineMatcher:
    def __init__(
        self,
        logger: Logger,
        strategy_resolver: GuidelineMatchingStrategyResolver,
    ) -> None:
        self._logger = logger
        self.strategy_resolver = strategy_resolver

    async def match_guidelines(
        self,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
        guidelines: Sequence[Guideline],
    ) -> GuidelineMatchingResult:
        if not guidelines:
            return GuidelineMatchingResult(
                total_duration=0.0,
                batch_count=0,
                batch_generations=[],
                batches=[],
            )

        t_start = time.time()

        with self._logger.scope("GuidelineMatcher"):
            with self._logger.operation("Creating batches"):
                guideline_strategies: dict[
                    str, tuple[GuidelineMatchingStrategy, list[Guideline]]
                ] = {}
                for guideline in guidelines:
                    strategy = await self.strategy_resolver.resolve(guideline)
                    if strategy.__class__.__name__ not in guideline_strategies:
                        guideline_strategies[strategy.__class__.__name__] = (strategy, [])
                    guideline_strategies[strategy.__class__.__name__][1].append(guideline)

                batches = await async_utils.safe_gather(
                    *[
                        strategy.create_batches(
                            guidelines,
                            context=GuidelineMatchingContext(
                                agent,
                                customer,
                                context_variables,
                                interaction_history,
                                terms,
                                staged_events,
                            ),
                        )
                        for _, (strategy, guidelines) in guideline_strategies.items()
                    ]
                )

            with self._logger.operation("Processing batches"):
                batch_tasks = [
                    batch.process() for strategy_batches in batches for batch in strategy_batches
                ]
                batch_results = await async_utils.safe_gather(*batch_tasks)

        t_end = time.time()

        return GuidelineMatchingResult(
            total_duration=t_end - t_start,
            batch_count=len(batches[0]),
            batch_generations=[result.generation_info for result in batch_results],
            batches=[result.matches for result in batch_results],
        )
