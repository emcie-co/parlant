__all__ = [
    "GuidelineMatcher",
    "GuidelineMatchingBatch",
    "GuidelineMatchingBatchResult",
    "GuidelineMatchingBatchContext",
    "ResponseAnalysisBatch",
    "ResponseAnalysisBatchResult",
    "ReportAnalysisContext",
    "GuidelineMatchingStrategy",
    "GuidelineMatchingStrategyResolver",
]
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
from parlant.core.capabilities import Capability
from parlant.core.journeys import Journey
from parlant.core.nlp.policies import policy, retry
from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.nlp.generation_info import GenerationInfo


from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
    AnalyzedGuideline,
)
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline
from parlant.core.sessions import Event, Session
from parlant.core.loggers import Logger


@dataclass(frozen=True)
class GuidelineMatchingStrategyContext:
    agent: Agent
    session: Session
    customer: Customer
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    capabilities: Sequence[Capability]
    staged_events: Sequence[EmittedEvent]


@dataclass(frozen=True)
class GuidelineMatchingBatchContext:
    agent: Agent
    session: Session
    customer: Customer
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]]
    interaction_history: Sequence[Event]
    terms: Sequence[Term]
    capabilities: Sequence[Capability]
    staged_events: Sequence[EmittedEvent]
    relevant_journeys: Sequence[Journey]


@dataclass(frozen=True)
class ReportAnalysisContext:
    agent: Agent
    session: Session
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
    matches: Sequence[GuidelineMatch]


@dataclass(frozen=True)
class ResponseAnalysisResult:
    total_duration: float
    batch_count: int
    batch_generations: Sequence[GenerationInfo]
    batches: Sequence[Sequence[AnalyzedGuideline]]

    @cached_property
    def analyzed_guidelines(self) -> Sequence[AnalyzedGuideline]:
        return list(chain.from_iterable(self.batches))


@dataclass(frozen=True)
class GuidelineMatchingBatchResult:
    matches: Sequence[GuidelineMatch]
    generation_info: GenerationInfo


@dataclass(frozen=True)
class ResponseAnalysisBatchResult:
    analyzed_guidelines: Sequence[AnalyzedGuideline]
    generation_info: GenerationInfo


class GuidelineMatchingBatch(ABC):
    @abstractmethod
    async def process(self) -> GuidelineMatchingBatchResult: ...


class ResponseAnalysisBatch(ABC):
    @abstractmethod
    async def process(self) -> ResponseAnalysisBatchResult: ...


class GuidelineMatchingStrategy(ABC):
    @abstractmethod
    async def create_matching_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingStrategyContext,
    ) -> Sequence[GuidelineMatchingBatch]: ...

    @abstractmethod
    async def create_report_analysis_batches(
        self,
        guideline_matches: Sequence[GuidelineMatch],
        context: ReportAnalysisContext,
    ) -> Sequence[ResponseAnalysisBatch]: ...

    @abstractmethod
    async def transform_matches(
        self,
        matches: Sequence[GuidelineMatch],
    ) -> Sequence[GuidelineMatch]: ...


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

    @policy(
        [
            retry(
                exceptions=Exception,
                max_attempts=3,
            )
        ]
    )
    async def _process_batch_with_retry(
        self, batch: GuidelineMatchingBatch
    ) -> GuidelineMatchingBatchResult:
        return await batch.process()

    @policy(
        [
            retry(
                exceptions=Exception,
                max_attempts=3,
            )
        ]
    )
    async def _process_report_analysis_batch_with_retry(
        self, batch: ResponseAnalysisBatch
    ) -> ResponseAnalysisBatchResult:
        return await batch.process()

    async def match_guidelines(
        self,
        agent: Agent,
        session: Session,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        capabilities: Sequence[Capability],
        staged_events: Sequence[EmittedEvent],
        guidelines: Sequence[Guideline],
    ) -> GuidelineMatchingResult:
        if not guidelines:
            return GuidelineMatchingResult(
                total_duration=0.0,
                batch_count=0,
                batch_generations=[],
                batches=[],
                matches=[],
            )

        t_start = time.time()

        with self._logger.scope("GuidelineMatcher"):
            with self._logger.operation("Creating guideline matching batches"):
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
                        strategy.create_matching_batches(
                            guidelines,
                            context=GuidelineMatchingStrategyContext(
                                agent,
                                session,
                                customer,
                                context_variables,
                                interaction_history,
                                terms,
                                capabilities,
                                staged_events,
                            ),
                        )
                        for _, (strategy, guidelines) in guideline_strategies.items()
                    ]
                )

            with self._logger.operation("Processing guideline matching batches"):
                batch_tasks = [
                    self._process_batch_with_retry(batch)
                    for strategy_batches in batches
                    for batch in strategy_batches
                ]
                batch_results = await async_utils.safe_gather(*batch_tasks)

        t_end = time.time()

        result_batches = [result.matches for result in batch_results]
        matches: Sequence[GuidelineMatch] = list(chain.from_iterable(result_batches))

        for strategy, _ in guideline_strategies.values():
            matches = await strategy.transform_matches(matches)

        return GuidelineMatchingResult(
            total_duration=t_end - t_start,
            batch_count=len(batches[0]),
            batch_generations=[result.generation_info for result in batch_results],
            batches=result_batches,
            matches=matches,
        )

    async def analyze_response(
        self,
        agent: Agent,
        session: Session,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
        guideline_matches: Sequence[GuidelineMatch],
    ) -> ResponseAnalysisResult:
        if not guideline_matches:
            return ResponseAnalysisResult(
                total_duration=0.0,
                batch_count=0,
                batch_generations=[],
                batches=[],
            )

        t_start = time.time()

        with self._logger.scope("GuidelineMatcher"):
            with self._logger.operation("Creating report analysis batches"):
                guideline_strategies: dict[
                    str, tuple[GuidelineMatchingStrategy, list[GuidelineMatch]]
                ] = {}
                for match in guideline_matches:
                    strategy = await self.strategy_resolver.resolve(match.guideline)
                    key = strategy.__class__.__name__
                    if key not in guideline_strategies:
                        guideline_strategies[key] = (strategy, [])
                    guideline_strategies[key][1].append(match)

                batches = await async_utils.safe_gather(
                    *[
                        strategy.create_report_analysis_batches(
                            guideline_matches,
                            context=ReportAnalysisContext(
                                agent,
                                session,
                                customer,
                                context_variables,
                                interaction_history,
                                terms,
                                staged_events,
                            ),
                        )
                        for _, (strategy, guideline_matches) in guideline_strategies.items()
                    ]
                )

            with self._logger.operation("Processing response analysis batches"):
                batch_tasks = [
                    self._process_report_analysis_batch_with_retry(batch)
                    for strategy_batches in batches
                    for batch in strategy_batches
                ]
                batch_results = await async_utils.safe_gather(*batch_tasks)

        t_end = time.time()

        return ResponseAnalysisResult(
            total_duration=t_end - t_start,
            batch_count=len(batch_results),
            batch_generations=[result.generation_info for result in batch_results],
            batches=[result.analyzed_guidelines for result in batch_results],
        )
