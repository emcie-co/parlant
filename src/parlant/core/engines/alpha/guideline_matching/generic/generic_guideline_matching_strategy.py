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

from datetime import datetime
from itertools import chain
import math
from typing import Mapping, Optional, Sequence, cast
from typing_extensions import override

from parlant.core.common import JSONSerializable, generate_id
from parlant.core.engines.alpha.guideline_matching.generic.disambiguation_batch import (
    DisambiguationGuidelineMatchesSchema,
    GenericDisambiguationGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_actionable_batch import (
    GenericActionableGuidelineMatchesSchema,
    GenericActionableGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_customer_dependent_batch import (
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic.observational_batch import (
    GenericObservationalGuidelineMatchesSchema,
    GenericObservationalGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchContext,
    GuidelineMatchingStrategy,
    GuidelineMatchingStrategyContext,
    ReportAnalysisContext,
)
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.relationships import GuidelineRelationshipKind, RelationshipStore


class GenericGuidelineMatchingStrategy(GuidelineMatchingStrategy):
    def __init__(
        self,
        logger: Logger,
        relationship_store: RelationshipStore,
        entity_queries: EntityQueries,
        observational_guideline_schematic_generator: SchematicGenerator[
            GenericObservationalGuidelineMatchesSchema
        ],
        previously_applied_actionable_guideline_schematic_generator: SchematicGenerator[
            GenericPreviouslyAppliedActionableGuidelineMatchesSchema
        ],
        previously_applied_actionable_customer_dependent_guideline_schematic_generator: SchematicGenerator[
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
        ],
        actionable_guideline_schematic_generator: SchematicGenerator[
            GenericActionableGuidelineMatchesSchema
        ],
        disambiguation_guidelines_schematic_generator: SchematicGenerator[
            DisambiguationGuidelineMatchesSchema
        ],
        report_analysis_schematic_generator: SchematicGenerator[GenericResponseAnalysisSchema],
    ) -> None:
        self._logger = logger
        self._relationship_store = relationship_store
        self._entity_queries = entity_queries

        self._observational_guideline_schematic_generator = (
            observational_guideline_schematic_generator
        )
        self._actionable_guideline_schematic_generator = actionable_guideline_schematic_generator
        self._previously_applied_actionable_guideline_schematic_generator = (
            previously_applied_actionable_guideline_schematic_generator
        )
        self._previously_applied_actionable_customer_dependent_guideline_schematic_generator = (
            previously_applied_actionable_customer_dependent_guideline_schematic_generator
        )
        self._disambiguation_guidelines_schematic_generator = (
            disambiguation_guidelines_schematic_generator
        )
        self._report_analysis_schematic_generator = report_analysis_schematic_generator

    @override
    async def create_matching_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingStrategyContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        observational_guidelines: list[Guideline] = []
        previously_applied_actionable_guidelines: list[Guideline] = []
        previously_applied_actionable_customer_dependent_guidelines: list[Guideline] = []
        actionable_guidelines: list[Guideline] = []
        disambiguation_groups: list[tuple[Guideline, list[Guideline]]] = []

        for g in guidelines:
            if not g.content.action:
                if targets := await self._try_get_disambiguation_group_targets(g, guidelines):
                    disambiguation_groups.append((g, targets))
                else:
                    observational_guidelines.append(g)
            else:
                if g.metadata.get("continuous", False):
                    actionable_guidelines.append(g)
                else:
                    if g.id in context.session.agent_state["applied_guideline_ids"]:
                        data = g.metadata.get("customer_dependent_action_data", False)
                        if isinstance(data, Mapping) and data.get("is_customer_dependent", False):
                            previously_applied_actionable_customer_dependent_guidelines.append(g)
                        else:
                            previously_applied_actionable_guidelines.append(g)
                    else:
                        actionable_guidelines.append(g)

        guideline_batches: list[GuidelineMatchingBatch] = []
        if observational_guidelines:
            guideline_batches.extend(
                self._create_batches_observational_guideline(observational_guidelines, context)
            )
        if previously_applied_actionable_guidelines:
            guideline_batches.extend(
                self._create_batches_previously_applied_actionable_guideline(
                    previously_applied_actionable_guidelines, context
                )
            )
        if previously_applied_actionable_customer_dependent_guidelines:
            guideline_batches.extend(
                self._create_batches_previously_applied_actionable_customer_dependent_guideline(
                    previously_applied_actionable_customer_dependent_guidelines, context
                )
            )
        if actionable_guidelines:
            guideline_batches.extend(
                self._create_batches_actionable_guideline(actionable_guidelines, context)
            )
        if disambiguation_groups:
            guideline_batches.extend(
                [
                    self._create_batch_disambiguation_guideline(source, targets, context)
                    for source, targets in disambiguation_groups
                ]
            )

        return guideline_batches

    @override
    async def create_report_analysis_batches(
        self,
        guideline_matches: Sequence[GuidelineMatch],
        context: ReportAnalysisContext,
    ) -> Sequence[GenericResponseAnalysisBatch]:
        if not guideline_matches:
            return []

        return [
            GenericResponseAnalysisBatch(
                logger=self._logger,
                schematic_generator=self._report_analysis_schematic_generator,
                context=context,
                guideline_matches=guideline_matches,
            )
        ]

    @override
    async def transform_matches(
        self,
        matches: Sequence[GuidelineMatch],
    ) -> Sequence[GuidelineMatch]:
        result: list[GuidelineMatch] = []
        guidelines_to_skip: list[GuidelineId] = []

        for m in matches:
            if disambiguation := m.metadata.get("disambiguation"):
                guidelines_to_skip.extend(
                    cast(
                        list[GuidelineId],
                        cast(dict[str, JSONSerializable], disambiguation).get("targets"),
                    )
                )

                result.append(
                    GuidelineMatch(
                        guideline=Guideline(
                            id=cast(GuidelineId, f"<transient_{generate_id()}>"),
                            creation_utc=datetime.now(),
                            content=GuidelineContent(
                                condition=m.guideline.content.condition,
                                action=cast(
                                    str,
                                    cast(dict[str, JSONSerializable], disambiguation)[
                                        "enriched_action"
                                    ],
                                ),
                            ),
                            enabled=True,
                            tags=[],
                            metadata={},
                        ),
                        score=10,
                        rationale=m.rationale,
                        metadata=m.metadata,
                    )
                )

        for m in matches:
            if m.metadata.get("disambiguation") or m.guideline.id in guidelines_to_skip:
                continue

            result.append(m)

        return result

    def _create_batches_observational_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingStrategyContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        journeys = list(
            chain.from_iterable(
                self._entity_queries.find_journeys_on_which_this_guideline_depends.get(g.id, [])
                for g in guidelines
            )
        )

        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_batch_observational_guideline(
                    guidelines=list(batch.values()),
                    journeys=journeys,
                    context=GuidelineMatchingBatchContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        relevant_journeys=journeys,
                    ),
                )
            )

        return batches

    def _create_batch_observational_guideline(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingBatchContext,
    ) -> GenericObservationalGuidelineMatchingBatch:
        return GenericObservationalGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._observational_guideline_schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )

    def _create_batches_previously_applied_actionable_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingStrategyContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        journeys = list(
            chain.from_iterable(
                self._entity_queries.find_journeys_on_which_this_guideline_depends.get(g.id, [])
                for g in guidelines
            )
        )

        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_batch_previously_applied_actionable_guideline(
                    guidelines=list(batch.values()),
                    journeys=journeys,
                    context=GuidelineMatchingBatchContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        relevant_journeys=journeys,
                    ),
                )
            )

        return batches

    def _create_batch_previously_applied_actionable_guideline(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingBatchContext,
    ) -> GenericPreviouslyAppliedActionableGuidelineMatchingBatch:
        return GenericPreviouslyAppliedActionableGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._previously_applied_actionable_guideline_schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )

    def _create_batches_previously_applied_actionable_customer_dependent_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingStrategyContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        journeys = list(
            chain.from_iterable(
                self._entity_queries.find_journeys_on_which_this_guideline_depends.get(g.id, [])
                for g in guidelines
            )
        )

        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_batch_previously_applied_actionable_customer_dependent_guideline(
                    guidelines=list(batch.values()),
                    journeys=journeys,
                    context=GuidelineMatchingBatchContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        relevant_journeys=journeys,
                    ),
                )
            )

        return batches

    def _create_batch_previously_applied_actionable_customer_dependent_guideline(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingBatchContext,
    ) -> GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch:
        return GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._previously_applied_actionable_customer_dependent_guideline_schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )

    def _create_batches_actionable_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingStrategyContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        journeys = list(
            chain.from_iterable(
                self._entity_queries.find_journeys_on_which_this_guideline_depends.get(g.id, [])
                for g in guidelines
            )
        )

        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_batch_actionable_guideline(
                    guidelines=list(batch.values()),
                    journeys=journeys,
                    context=GuidelineMatchingBatchContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        relevant_journeys=journeys,
                    ),
                )
            )

        return batches

    def _create_batch_actionable_guideline(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingBatchContext,
    ) -> GenericActionableGuidelineMatchingBatch:
        return GenericActionableGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._actionable_guideline_schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )

    async def _try_get_disambiguation_group_targets(
        self,
        candidate: Guideline,
        guidelines: Sequence[Guideline],
    ) -> Optional[list[Guideline]]:
        guidelines_dict = {g.id: g for g in guidelines}

        if relationships := await self._relationship_store.list_relationships(
            kind=GuidelineRelationshipKind.DISAMBIGUATION,
            source_id=candidate.id,
        ):
            targets = [guidelines_dict[cast(GuidelineId, r.target.id)] for r in relationships]

            if len(targets) > 1:
                return targets

        return None

    def _create_batch_disambiguation_guideline(
        self,
        disambiguation_guideline: Guideline,
        disambiguation_targets: list[Guideline],
        context: GuidelineMatchingStrategyContext,
    ) -> GenericDisambiguationGuidelineMatchingBatch:
        journeys = list(
            chain.from_iterable(
                self._entity_queries.find_journeys_on_which_this_guideline_depends.get(g.id, [])
                for g in [disambiguation_guideline, *disambiguation_targets]
            )
        )

        return GenericDisambiguationGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._disambiguation_guidelines_schematic_generator,
            disambiguation_guideline=disambiguation_guideline,
            disambiguation_targets=disambiguation_targets,
            context=GuidelineMatchingBatchContext(
                agent=context.agent,
                session=context.session,
                customer=context.customer,
                context_variables=context.context_variables,
                interaction_history=context.interaction_history,
                terms=context.terms,
                capabilities=context.capabilities,
                staged_events=context.staged_events,
                relevant_journeys=journeys,
            ),
        )

    def _get_optimal_batch_size(self, guidelines: dict[GuidelineId, Guideline]) -> int:
        guideline_n = len(guidelines)

        if guideline_n <= 10:
            return 1
        elif guideline_n <= 20:
            return 2
        elif guideline_n <= 30:
            return 3
        else:
            return 5
