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

from collections import defaultdict
from datetime import datetime
from itertools import chain
import math
from typing import Mapping, Optional, Sequence, cast
from typing_extensions import override

from parlant.core import async_utils
from parlant.core.common import JSONSerializable, generate_id
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
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
from parlant.core.engines.alpha.guideline_matching.generic.journey_node_selection_batch import (
    GenericJourneyNodeSelectionBatch,
    JourneyNodeSelectionSchema,
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
    GuidelineMatchingStrategy,
    GuidelineMatchingContext,
    ResponseAnalysisContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId, GuidelineStore
from parlant.core.journeys import Journey, JourneyId, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.relationships import RelationshipKind, RelationshipStore


class GenericGuidelineMatchingStrategy(GuidelineMatchingStrategy):
    """
    A guideline matching strategy that categorizes guidelines into different types
    and creates appropriate processing batches for each category.
    
    This strategy handles observational guidelines, actionable guidelines,
    previously applied guidelines, disambiguation guidelines, and journey node selection.
    """
    
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        guideline_store: GuidelineStore,
        journey_store: JourneyStore,
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
        journey_step_selection_schematic_generator: SchematicGenerator[JourneyNodeSelectionSchema],
        response_analysis_schematic_generator: SchematicGenerator[GenericResponseAnalysisSchema],
    ) -> None:
        """
        Initialize the generic guideline matching strategy.
        
        Args:
            logger: Logger instance for recording events
            optimization_policy: Policy for optimization decisions
            guideline_store: Store for accessing guidelines
            journey_store: Store for accessing journeys
            relationship_store: Store for accessing relationships between entities
            entity_queries: Queries for finding entity relationships
            observational_guideline_schematic_generator: Generator for observational guideline schemas
            previously_applied_actionable_guideline_schematic_generator: Generator for previously applied actionable guideline schemas
            previously_applied_actionable_customer_dependent_guideline_schematic_generator: Generator for customer-dependent previously applied guideline schemas
            actionable_guideline_schematic_generator: Generator for actionable guideline schemas
            disambiguation_guidelines_schematic_generator: Generator for disambiguation guideline schemas
            journey_step_selection_schematic_generator: Generator for journey step selection schemas
            response_analysis_schematic_generator: Generator for response analysis schemas
        """
        self._logger = logger

        self._guideline_store = guideline_store
        self._journey_store = journey_store
        self._relationship_store = relationship_store

        self._optimization_policy = optimization_policy
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
        self._journey_step_selection_schematic_generator = (
            journey_step_selection_schematic_generator
        )
        self._response_analysis_schematic_generator = response_analysis_schematic_generator

    @override
    async def create_matching_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        """
        Create matching batches by categorizing guidelines into different types.
        
        Guidelines are categorized into observational, actionable, previously applied,
        disambiguation, and journey node selection types, then processed into appropriate batches.
        
        Args:
            guidelines: Sequence of guidelines to categorize and batch
            context: Context containing session, agent, and interaction information
            
        Returns:
            Sequence of guideline matching batches ready for processing
        """
        observational_guidelines: list[Guideline] = []
        previously_applied_actionable_guidelines: list[Guideline] = []
        previously_applied_actionable_customer_dependent_guidelines: list[Guideline] = []
        actionable_guidelines: list[Guideline] = []
        disambiguation_groups: list[tuple[Guideline, list[Guideline]]] = []
        journey_step_selection_journeys: dict[Journey, list[Guideline]] = defaultdict(list)

        active_journeys_mapping = {journey.id: journey for journey in context.active_journeys}

        for g in guidelines:
            if g.metadata.get("journey_node") is not None:
                # If the guideline is associated with a journey node, we add the journey steps
                # to the list of journeys that need reevaluation.
                if journey_id := cast(
                    Mapping[str, JSONSerializable], g.metadata.get("journey_node", {})
                ).get("journey_id"):
                    journey_id = cast(JourneyId, journey_id)

                    if journey_id in active_journeys_mapping:
                        journey_step_selection_journeys[active_journeys_mapping[journey_id]].append(
                            g
                        )

            elif not g.content.action:
                if targets := await self._try_get_disambiguation_group_targets(g, guidelines):
                    disambiguation_groups.append((g, targets))
                else:
                    observational_guidelines.append(g)
            else:
                if g.metadata.get("continuous", False):
                    actionable_guidelines.append(g)
                else:
                    if (
                        context.session.agent_states
                        and g.id in context.session.agent_states[-1].applied_guideline_ids
                    ):
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
        if journey_step_selection_journeys:
            guideline_batches.extend(
                await async_utils.safe_gather(
                    *[
                        self._create_batch_journey_step_selection(examined_journey, steps, context)
                        for examined_journey, steps in journey_step_selection_journeys.items()
                    ]
                )
            )

        return guideline_batches

    @override
    async def create_response_analysis_batches(
        self,
        guideline_matches: Sequence[GuidelineMatch],
        context: ResponseAnalysisContext,
    ) -> Sequence[GenericResponseAnalysisBatch]:
        """
        Create response analysis batches from guideline matches.
        
        Args:
            guideline_matches: Sequence of guideline matches to analyze
            context: Context for response analysis
            
        Returns:
            Sequence of response analysis batches, empty if no matches provided
        """
        if not guideline_matches:
            return []

        return [
            GenericResponseAnalysisBatch(
                logger=self._logger,
                optimization_policy=self._optimization_policy,
                schematic_generator=self._response_analysis_schematic_generator,
                context=context,
                guideline_matches=guideline_matches,
            )
        ]

    @override
    async def transform_matches(
        self,
        matches: Sequence[GuidelineMatch],
    ) -> Sequence[GuidelineMatch]:
        """
        Transform guideline matches by processing disambiguation matches.
        
        Creates transient guidelines for disambiguation matches and filters out
        guidelines that should be skipped due to disambiguation.
        
        Args:
            matches: Sequence of guideline matches to transform
            
        Returns:
            Transformed sequence of guideline matches
        """
        result: list[GuidelineMatch] = []
        guidelines_to_skip: set[GuidelineId] = set()

        for m in matches:
            if disambiguation := m.metadata.get("disambiguation"):
                guidelines_to_skip.update(
                    cast(
                        list[GuidelineId],
                        cast(dict[str, JSONSerializable], disambiguation).get("targets"),
                    )
                )

                guidelines_to_skip.add(m.guideline.id)

                result.append(
                    GuidelineMatch(
                        guideline=Guideline(
                            id=cast(GuidelineId, f"<transient_{generate_id()}>"),
                            creation_utc=datetime.now(),
                            content=GuidelineContent(
                                condition=internal_representation(m.guideline).condition,
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

        result.extend(m for m in matches if m.guideline.id not in guidelines_to_skip)

        return result

    def _create_batches_observational_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        """
        Create batches for observational guidelines.
        
        Splits guidelines into optimally sized batches and creates observational
        guideline matching batches with associated journeys.
        
        Args:
            guidelines: Sequence of observational guidelines to batch
            context: Context for guideline matching
            
        Returns:
            Sequence of observational guideline matching batches
        """
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
                    context=GuidelineMatchingContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        active_journeys=journeys,
                        journey_paths=context.journey_paths,
                    ),
                )
            )

        return batches

    def _create_batch_observational_guideline(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingContext,
    ) -> GenericObservationalGuidelineMatchingBatch:
        """
        Create a single observational guideline matching batch.
        
        Args:
            guidelines: Guidelines to include in the batch
            journeys: Associated journeys for the guidelines
            context: Context for guideline matching
            
        Returns:
            Observational guideline matching batch
        """
        return GenericObservationalGuidelineMatchingBatch(
            logger=self._logger,
            optimization_policy=self._optimization_policy,
            schematic_generator=self._observational_guideline_schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )

    def _create_batches_previously_applied_actionable_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        """
        Create batches for previously applied actionable guidelines.
        
        Args:
            guidelines: Sequence of previously applied actionable guidelines
            context: Context for guideline matching
            
        Returns:
            Sequence of previously applied actionable guideline matching batches
        """
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
                    context=GuidelineMatchingContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        active_journeys=journeys,
                        journey_paths=context.journey_paths,
                    ),
                )
            )

        return batches

    def _create_batch_previously_applied_actionable_guideline(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingContext,
    ) -> GenericPreviouslyAppliedActionableGuidelineMatchingBatch:
        """
        Create a single previously applied actionable guideline matching batch.
        
        Args:
            guidelines: Guidelines to include in the batch
            journeys: Associated journeys for the guidelines
            context: Context for guideline matching
            
        Returns:
            Previously applied actionable guideline matching batch
        """
        return GenericPreviouslyAppliedActionableGuidelineMatchingBatch(
            logger=self._logger,
            optimization_policy=self._optimization_policy,
            schematic_generator=self._previously_applied_actionable_guideline_schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )

    def _create_batches_previously_applied_actionable_customer_dependent_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        """
        Create batches for previously applied customer-dependent actionable guidelines.
        
        Args:
            guidelines: Sequence of customer-dependent previously applied guidelines
            context: Context for guideline matching
            
        Returns:
            Sequence of customer-dependent previously applied guideline matching batches
        """
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
                    context=GuidelineMatchingContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        active_journeys=journeys,
                        journey_paths=context.journey_paths,
                    ),
                )
            )

        return batches

    def _create_batch_previously_applied_actionable_customer_dependent_guideline(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingContext,
    ) -> GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch:
        """
        Create a single customer-dependent previously applied actionable guideline batch.
        
        Args:
            guidelines: Guidelines to include in the batch
            journeys: Associated journeys for the guidelines
            context: Context for guideline matching
            
        Returns:
            Customer-dependent previously applied actionable guideline matching batch
        """
        return GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch(
            logger=self._logger,
            optimization_policy=self._optimization_policy,
            schematic_generator=self._previously_applied_actionable_customer_dependent_guideline_schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )

    def _create_batches_actionable_guideline(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        """
        Create batches for actionable guidelines.
        
        Args:
            guidelines: Sequence of actionable guidelines to batch
            context: Context for guideline matching
            
        Returns:
            Sequence of actionable guideline matching batches
        """
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
                    context=GuidelineMatchingContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        active_journeys=journeys,
                        journey_paths=context.journey_paths,
                    ),
                )
            )

        return batches

    def _create_batch_actionable_guideline(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingContext,
    ) -> GenericActionableGuidelineMatchingBatch:
        """
        Create a single actionable guideline matching batch.
        
        Args:
            guidelines: Guidelines to include in the batch
            journeys: Associated journeys for the guidelines
            context: Context for guideline matching
            
        Returns:
            Actionable guideline matching batch
        """
        return GenericActionableGuidelineMatchingBatch(
            logger=self._logger,
            optimization_policy=self._optimization_policy,
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
        """
        Attempt to find disambiguation targets for a candidate guideline.
        
        Args:
            candidate: Guideline to check for disambiguation relationships
            guidelines: Available guidelines to search for targets
            
        Returns:
            List of target guidelines if disambiguation group found, None otherwise
        """
        guidelines_dict = {g.id: g for g in guidelines}

        if relationships := await self._relationship_store.list_relationships(
            kind=RelationshipKind.DISAMBIGUATION,
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
        context: GuidelineMatchingContext,
    ) -> GenericDisambiguationGuidelineMatchingBatch:
        """
        Create a disambiguation guideline matching batch.
        
        Args:
            disambiguation_guideline: Source guideline for disambiguation
            disambiguation_targets: Target guidelines for disambiguation
            context: Context for guideline matching
            
        Returns:
            Disambiguation guideline matching batch
        """
        journeys = list(
            chain.from_iterable(
                self._entity_queries.find_journeys_on_which_this_guideline_depends.get(g.id, [])
                for g in [disambiguation_guideline, *disambiguation_targets]
            )
        )

        return GenericDisambiguationGuidelineMatchingBatch(
            logger=self._logger,
            journey_store=self._journey_store,
            optimization_policy=self._optimization_policy,
            schematic_generator=self._disambiguation_guidelines_schematic_generator,
            disambiguation_guideline=disambiguation_guideline,
            disambiguation_targets=disambiguation_targets,
            context=GuidelineMatchingContext(
                agent=context.agent,
                session=context.session,
                customer=context.customer,
                context_variables=context.context_variables,
                interaction_history=context.interaction_history,
                terms=context.terms,
                capabilities=context.capabilities,
                staged_events=context.staged_events,
                active_journeys=journeys,
                journey_paths=context.journey_paths,
            ),
        )

    async def _create_batch_journey_step_selection(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> GenericJourneyNodeSelectionBatch:
        """
        Create a journey node selection batch.
        
        Args:
            examined_journey: Journey being examined for node selection
            step_guidelines: Guidelines associated with journey steps
            context: Context for guideline matching
            
        Returns:
            Journey node selection batch
        """
        return GenericJourneyNodeSelectionBatch(
            logger=self._logger,
            guideline_store=self._guideline_store,
            optimization_policy=self._optimization_policy,
            schematic_generator=self._journey_step_selection_schematic_generator,
            examined_journey=examined_journey,
            context=GuidelineMatchingContext(
                agent=context.agent,
                session=context.session,
                customer=context.customer,
                context_variables=context.context_variables,
                interaction_history=context.interaction_history,
                terms=context.terms,
                capabilities=context.capabilities,
                staged_events=context.staged_events,
                active_journeys=context.active_journeys,
                journey_paths=context.journey_paths,
            ),
            node_guidelines=step_guidelines,
            journey_path=context.journey_paths.get(examined_journey.id, []),
        )

    def _get_optimal_batch_size(self, guidelines: dict[GuidelineId, Guideline]) -> int:
        """
        Get the optimal batch size for processing guidelines.
        
        Args:
            guidelines: Dictionary of guidelines to determine batch size for
            
        Returns:
            Optimal batch size based on optimization policy
        """
        return self._optimization_policy.get_guideline_matching_batch_size(len(guidelines))
