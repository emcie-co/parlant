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

import asyncio
import traceback
from typing import Any, Iterable, Optional, OrderedDict, Sequence, cast

from parlant.core import async_utils
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import JSONSerializable, md5_checksum
from parlant.core.evaluations import (
    CoherenceCheck,
    CoherenceCheckKind,
    EntailmentRelationshipProposition,
    EntailmentRelationshipPropositionKind,
    Evaluation,
    EvaluationStatus,
    EvaluationId,
    GuidelinePayload,
    InvoiceData,
    InvoiceJourneyData,
    JourneyPayload,
    PayloadOperation,
    Invoice,
    InvoiceGuidelineData,
    Payload,
    EvaluationStore,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId, GuidelineStore
from parlant.core.journey_guideline_projection import (
    JourneyGuidelineProjection,
    extract_node_id_from_journey_node_guideline_id,
)
from parlant.core.journeys import Journey, JourneyId, JourneyStore
from parlant.core.services.indexing.coherence_checker import (
    CoherenceChecker,
)
from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.services.indexing.customer_dependent_action_detector import (
    CustomerDependentActionDetector,
    CustomerDependentActionProposition,
)
from parlant.core.services.indexing.guideline_action_proposer import (
    GuidelineActionProposer,
    GuidelineActionProposition,
)
from parlant.core.services.indexing.guideline_agent_intention_proposer import (
    AgentIntentionProposer,
    AgentIntentionProposition,
)
from parlant.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
)
from parlant.core.services.indexing.guideline_continuous_proposer import (
    GuidelineContinuousProposer,
    GuidelineContinuousProposition,
)
from parlant.core.loggers import Logger
from parlant.core.entity_cq import EntityQueries
from parlant.core.services.indexing.relative_action_proposer import (
    RelativeActionProposer,
    RelativeActionProposition,
)
from parlant.core.services.indexing.tool_running_action_detector import (
    ToolRunningActionDetector,
    ToolRunningActionProposition,
)
from parlant.core.tags import Tag


class EvaluationValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class LegacyGuidelineEvaluator:
    def __init__(
        self,
        logger: Logger,
        entity_queries: EntityQueries,
        guideline_connection_proposer: GuidelineConnectionProposer,
        coherence_checker: CoherenceChecker,
    ) -> None:
        self._logger = logger
        self._entity_queries = entity_queries
        self._guideline_connection_proposer = guideline_connection_proposer
        self._coherence_checker = coherence_checker

    async def evaluate(
        self,
        agent: Agent,
        payloads: Sequence[Payload],
        progress_report: ProgressReport,
    ) -> Sequence[InvoiceGuidelineData]:
        journeys = await self._entity_queries.finds_journeys_for_context(agent.id)
        existing_guidelines = await self._entity_queries.find_guidelines_for_context(
            agent.id, journeys
        )

        tasks: list[asyncio.Task[Any]] = []
        coherence_checks_task: Optional[
            asyncio.Task[Optional[Iterable[Sequence[CoherenceCheck]]]]
        ] = None

        connection_propositions_task: Optional[
            asyncio.Task[Optional[Iterable[Sequence[EntailmentRelationshipProposition]]]]
        ] = None

        coherence_checks_task = asyncio.create_task(
            self._check_guideline_payloads_coherence(
                agent,
                cast(Sequence[GuidelinePayload], payloads),
                existing_guidelines,
                progress_report,
            )
        )
        tasks.append(coherence_checks_task)

        connection_propositions_task = asyncio.create_task(
            self._propose_guideline_payloads_connections(
                agent,
                cast(Sequence[GuidelinePayload], payloads),
                existing_guidelines,
                progress_report,
            )
        )
        tasks.append(connection_propositions_task)

        if tasks:
            await async_utils.safe_gather(*tasks)

        coherence_checks: Optional[Iterable[Sequence[CoherenceCheck]]] = []
        if coherence_checks_task:
            coherence_checks = coherence_checks_task.result()

        connection_propositions: Optional[Iterable[Sequence[EntailmentRelationshipProposition]]] = (
            None
        )
        if connection_propositions_task:
            connection_propositions = connection_propositions_task.result()

        if coherence_checks:
            return [
                InvoiceGuidelineData(
                    coherence_checks=payload_coherence_checks,
                    entailment_propositions=None,
                    properties_proposition=None,
                )
                for payload_coherence_checks in coherence_checks
            ]

        elif connection_propositions:
            return [
                InvoiceGuidelineData(
                    coherence_checks=[],
                    entailment_propositions=payload_connection_propositions,
                    properties_proposition=None,
                )
                for payload_connection_propositions in connection_propositions
            ]

        else:
            return [
                InvoiceGuidelineData(
                    coherence_checks=[],
                    entailment_propositions=None,
                    properties_proposition=None,
                )
                for _ in payloads
            ]

    async def _check_guideline_payloads_coherence(
        self,
        agent: Agent,
        payloads: Sequence[GuidelinePayload],
        existing_guidelines: Sequence[Guideline],
        progress_report: ProgressReport,
    ) -> Optional[Iterable[Sequence[CoherenceCheck]]]:
        guidelines_to_evaluate = [p.content for p in payloads if p.coherence_check]

        guidelines_to_skip = [(p.content, False) for p in payloads if not p.coherence_check]

        updated_ids = {
            cast(GuidelineId, p.updated_id)
            for p in payloads
            if p.operation == PayloadOperation.UPDATE
        }

        remaining_existing_guidelines = []

        for g in existing_guidelines:
            if g.id not in updated_ids:
                remaining_existing_guidelines.append(
                    (GuidelineContent(condition=g.content.condition, action=g.content.action), True)
                )
            else:
                updated_ids.remove(g.id)

        if len(updated_ids) > 0:
            raise EvaluationError(
                f"Guideline ID(s): {', '.join(list(updated_ids))} in '{agent.id}' agent do not exist."
            )

        comparison_guidelines = guidelines_to_skip + remaining_existing_guidelines

        incoherences = await self._coherence_checker.propose_incoherencies(
            agent=agent,
            guidelines_to_evaluate=guidelines_to_evaluate,
            comparison_guidelines=[g for g, _ in comparison_guidelines],
            progress_report=progress_report,
        )

        if not incoherences:
            return None

        coherence_checks_by_guideline_payload: OrderedDict[str, list[CoherenceCheck]] = OrderedDict(
            {f"{p.content.condition}{p.content.action}": [] for p in payloads}
        )

        guideline_payload_is_skipped_pairs = {
            f"{p.content.condition}{p.content.action}": p.coherence_check for p in payloads
        }

        for c in incoherences:
            if (
                f"{c.guideline_a.condition}{c.guideline_a.action}"
                in coherence_checks_by_guideline_payload
                and guideline_payload_is_skipped_pairs[
                    f"{c.guideline_a.condition}{c.guideline_a.action}"
                ]
            ):
                coherence_checks_by_guideline_payload[
                    f"{c.guideline_a.condition}{c.guideline_a.action}"
                ].append(
                    CoherenceCheck(
                        kind=CoherenceCheckKind.CONTRADICTION_WITH_ANOTHER_EVALUATED_GUIDELINE
                        if f"{c.guideline_b.condition}{c.guideline_b.action}"
                        in coherence_checks_by_guideline_payload
                        else CoherenceCheckKind.CONTRADICTION_WITH_EXISTING_GUIDELINE,
                        first=c.guideline_a,
                        second=c.guideline_b,
                        issue=c.actions_contradiction_rationale,
                        severity=c.actions_contradiction_severity,
                    )
                )

            if (
                f"{c.guideline_b.condition}{c.guideline_b.action}"
                in coherence_checks_by_guideline_payload
                and guideline_payload_is_skipped_pairs[
                    f"{c.guideline_b.condition}{c.guideline_b.action}"
                ]
            ):
                coherence_checks_by_guideline_payload[
                    f"{c.guideline_b.condition}{c.guideline_b.action}"
                ].append(
                    CoherenceCheck(
                        kind=CoherenceCheckKind.CONTRADICTION_WITH_ANOTHER_EVALUATED_GUIDELINE,
                        first=c.guideline_a,
                        second=c.guideline_b,
                        issue=c.actions_contradiction_rationale,
                        severity=c.actions_contradiction_severity,
                    )
                )

        return coherence_checks_by_guideline_payload.values()

    async def _propose_guideline_payloads_connections(
        self,
        agent: Agent,
        payloads: Sequence[GuidelinePayload],
        existing_guidelines: Sequence[Guideline],
        progress_report: ProgressReport,
    ) -> Optional[Iterable[Sequence[EntailmentRelationshipProposition]]]:
        proposed_guidelines = [p.content for p in payloads if p.connection_proposition]

        guidelines_to_skip = [(p.content, False) for p in payloads if not p.connection_proposition]

        updated_ids = {p.updated_id for p in payloads if p.operation == PayloadOperation.UPDATE}

        remaining_existing_guidelines = [
            (GuidelineContent(condition=g.content.condition, action=g.content.action), True)
            for g in existing_guidelines
            if g.id not in updated_ids
        ]

        comparison_guidelines = guidelines_to_skip + remaining_existing_guidelines

        connection_propositions = [
            p
            for p in await self._guideline_connection_proposer.propose_connections(
                agent,
                introduced_guidelines=proposed_guidelines,
                existing_guidelines=[g for g, _ in comparison_guidelines],
                progress_report=progress_report,
            )
            if p.score >= 6
        ]

        if not connection_propositions:
            return None

        connection_results_by_guideline_payload: OrderedDict[
            str, list[EntailmentRelationshipProposition]
        ] = OrderedDict({f"{p.content.condition}{p.content.action}": [] for p in payloads})
        guideline_payload_is_skipped_pairs = {
            f"{p.content.condition}{p.content.action}": p.connection_proposition for p in payloads
        }

        for c in connection_propositions:
            if (
                f"{c.source.condition}{c.source.action}" in connection_results_by_guideline_payload
                and guideline_payload_is_skipped_pairs[f"{c.source.condition}{c.source.action}"]
            ):
                connection_results_by_guideline_payload[
                    f"{c.source.condition}{c.source.action}"
                ].append(
                    EntailmentRelationshipProposition(
                        check_kind=EntailmentRelationshipPropositionKind.CONNECTION_WITH_ANOTHER_EVALUATED_GUIDELINE
                        if f"{c.target.condition}{c.target.action}"
                        in connection_results_by_guideline_payload
                        else EntailmentRelationshipPropositionKind.CONNECTION_WITH_EXISTING_GUIDELINE,
                        source=c.source,
                        target=c.target,
                    )
                )

            if (
                f"{c.target.condition}{c.target.action}" in connection_results_by_guideline_payload
                and guideline_payload_is_skipped_pairs[f"{c.target.condition}{c.target.action}"]
            ):
                connection_results_by_guideline_payload[
                    f"{c.target.condition}{c.target.action}"
                ].append(
                    EntailmentRelationshipProposition(
                        check_kind=EntailmentRelationshipPropositionKind.CONNECTION_WITH_ANOTHER_EVALUATED_GUIDELINE
                        if f"{c.source.condition}{c.source.action}"
                        in connection_results_by_guideline_payload
                        else EntailmentRelationshipPropositionKind.CONNECTION_WITH_EXISTING_GUIDELINE,
                        source=c.source,
                        target=c.target,
                    )
                )

        return connection_results_by_guideline_payload.values()


class LegacyBehavioralChangeEvaluator:
    def __init__(
        self,
        logger: Logger,
        background_task_service: BackgroundTaskService,
        agent_store: AgentStore,
        evaluation_store: EvaluationStore,
        entity_queries: EntityQueries,
        guideline_connection_proposer: GuidelineConnectionProposer,
        coherence_checker: CoherenceChecker,
    ) -> None:
        self._logger = logger
        self._background_task_service = background_task_service
        self._agent_store = agent_store
        self._evaluation_store = evaluation_store
        self._entity_queries = entity_queries
        self._guideline_evaluator = LegacyGuidelineEvaluator(
            logger=logger,
            entity_queries=entity_queries,
            guideline_connection_proposer=guideline_connection_proposer,
            coherence_checker=coherence_checker,
        )

    async def validate_payloads(
        self,
        agent: Agent,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> None:
        if not payload_descriptors:
            raise EvaluationValidationError("No payloads provided for the evaluation task.")

        guideline_payloads = [
            cast(GuidelinePayload, p) for k, p in payload_descriptors if k == PayloadKind.GUIDELINE
        ]

        if guideline_payloads:

            async def _check_for_duplications() -> None:
                seen_guidelines = set((g.content) for g in guideline_payloads)
                if len(seen_guidelines) < len(guideline_payloads):
                    raise EvaluationValidationError(
                        "Duplicate guideline found among the provided guidelines."
                    )

                journeys = await self._entity_queries.finds_journeys_for_context(agent.id)
                existing_guidelines = await self._entity_queries.find_guidelines_for_context(
                    agent.id, journeys
                )

                if guideline := next(
                    iter(g for g in existing_guidelines if (g.content) in seen_guidelines),
                    None,
                ):
                    raise EvaluationValidationError(
                        f"Duplicate guideline found against existing guidelines: {str(guideline)} in {agent.id} guideline_set"
                    )

            await _check_for_duplications()

    async def create_evaluation_task(
        self,
        agent: Agent,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> EvaluationId:
        await self.validate_payloads(agent, payload_descriptors)

        evaluation = await self._evaluation_store.create_evaluation(
            payload_descriptors,
            tags=[Tag.for_agent_id(agent.id)],
        )

        await self._background_task_service.start(
            self.run_evaluation(evaluation),
            tag=f"evaluation({evaluation.id})",
        )

        return evaluation.id

    async def run_evaluation(
        self,
        evaluation: Evaluation,
    ) -> None:
        async def _update_progress(percentage: float) -> None:
            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"progress": percentage},
            )

        progress_report = ProgressReport(_update_progress)

        try:
            if running_task := next(
                iter(
                    e
                    for e in await self._evaluation_store.list_evaluations()
                    if e.status == EvaluationStatus.RUNNING and e.id != evaluation.id
                ),
                None,
            ):
                raise EvaluationError(f"An evaluation task '{running_task.id}' is already running.")

            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"status": EvaluationStatus.RUNNING},
            )

            agent = await self._agent_store.read_agent(
                agent_id=AgentId(cast(str, Tag.extract_agent_id(evaluation.tags[0])))
            )

            guideline_evaluation_data = await self._guideline_evaluator.evaluate(
                agent=agent,
                payloads=[
                    invoice.payload
                    for invoice in evaluation.invoices
                    if invoice.kind == PayloadKind.GUIDELINE
                ],
                progress_report=progress_report,
            )

            invoices: list[Invoice] = []
            for i, result in enumerate(guideline_evaluation_data):
                invoice_checksum = md5_checksum(str(evaluation.invoices[i].payload))
                state_version = str(hash("Temporarily"))

                invoices.append(
                    Invoice(
                        kind=evaluation.invoices[i].kind,
                        payload=evaluation.invoices[i].payload,
                        checksum=invoice_checksum,
                        state_version=state_version,
                        approved=True if not result.coherence_checks else False,
                        data=result,
                        error=None,
                    )
                )

            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"invoices": invoices},
            )

            self._logger.trace(f"evaluation task '{evaluation.id}' completed")

            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"status": EvaluationStatus.COMPLETED},
            )

        except Exception as exc:
            logger_level = "info" if isinstance(exc, EvaluationError) else "error"
            getattr(self._logger, logger_level)(
                f"Evaluation task '{evaluation.id}' failed due to the following error: '{str(exc)}'"
            )

            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={
                    "status": EvaluationStatus.FAILED,
                    "error": str(exc),
                },
            )

            raise


class JourneyEvaluator:
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        journey_store: JourneyStore,
        journey_guideline_projection: JourneyGuidelineProjection,
        relative_action_proposer: RelativeActionProposer,
    ) -> None:
        self._logger = logger

        self._guideline_store = guideline_store
        self._journey_store = journey_store
        self._journey_guideline_projection = journey_guideline_projection

        self._relative_action_proposer = relative_action_proposer

    async def _build_invoice_data(
        self,
        relative_action_propositions: Sequence[RelativeActionProposition],
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Guideline], tuple[Guideline]]],
    ) -> Sequence[InvoiceJourneyData]:
        index_to_node_ids = {
            journey_id: {
                cast(dict[str, JSONSerializable], g.metadata["journey_node"])[
                    "index"
                ]: extract_node_id_from_journey_node_guideline_id(g.id)
                for g in journey_projections[journey_id][1]
            }
            for journey_id in journey_projections
        }

        result = []

        for proposition, journey_id in zip(
            relative_action_propositions, journey_projections.keys()
        ):
            invoice_data = InvoiceJourneyData(
                node_properties_proposition={
                    index_to_node_ids[journey_id][r.index]: {
                        "internal_action": r.rewritten_actions,
                    }
                    for r in proposition.actions
                },
                edge_properties_proposition={},
            )

            result.append(invoice_data)

        return result

    async def evaluate(
        self,
        payloads: Sequence[JourneyPayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[InvoiceJourneyData]:
        journeys: dict[JourneyId, Journey] = {
            j.id: j
            for j in await async_utils.safe_gather(
                *[
                    self._journey_store.read_journey(journey_id=payload.journey_id)
                    for payload in payloads
                ]
            )
        }

        journey_conditions = [
            await async_utils.safe_gather(
                *[
                    self._guideline_store.read_guideline(guideline_id=condition)
                    for condition in journey.conditions
                ]
            )
            for journey in journeys.values()
        ]

        journey_projections = {
            payload.journey_id: (journeys[payload.journey_id], projection, conditions)
            for payload, projection, conditions in zip(
                payloads,
                await async_utils.safe_gather(
                    *[
                        self._journey_guideline_projection.project_journey_to_guidelines(
                            journey_id=payload.journey_id
                        )
                        for payload in payloads
                    ]
                ),
                journey_conditions,
            )
        }

        relative_action_propositions = await self._propose_relative_actions(
            journey_projections,
            progress_report,
        )

        invoices = await self._build_invoice_data(
            relative_action_propositions,
            journey_projections,
        )

        return invoices

    async def _propose_relative_actions(
        self,
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Guideline], tuple[Guideline]]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[RelativeActionProposition]:
        tasks: list[asyncio.Task[RelativeActionProposition]] = []

        for journey_id, (
            journey,
            step_guidelines,
            journey_conditions,
        ) in journey_projections.items():
            if not step_guidelines:
                continue

            tasks.append(
                asyncio.create_task(
                    self._relative_action_proposer.propose_relative_action(
                        examined_journey=journey,
                        step_guidelines=step_guidelines,
                        journey_conditions=journey_conditions,
                        progress_report=progress_report,
                    )
                )
            )

        sparse_results = list(await async_utils.safe_gather(*tasks))

        return sparse_results


class GuidelineEvaluator:
    def __init__(
        self,
        logger: Logger,
        entity_queries: EntityQueries,
        guideline_action_proposer: GuidelineActionProposer,
        guideline_continuous_proposer: GuidelineContinuousProposer,
        customer_dependent_action_detector: CustomerDependentActionDetector,
        agent_intention_proposer: AgentIntentionProposer,
        tool_running_action_detector: ToolRunningActionDetector,
    ) -> None:
        self._logger = logger
        self._entity_queries = entity_queries
        self._guideline_action_proposer = guideline_action_proposer
        self._guideline_continuous_proposer = guideline_continuous_proposer
        self._customer_dependent_action_detector = customer_dependent_action_detector
        self._agent_intention_proposer = agent_intention_proposer
        self._tool_running_action_detector = tool_running_action_detector

    def _build_invoice_data(
        self,
        action_propositions: Sequence[Optional[GuidelineActionProposition]],
        continuous_propositions: Sequence[Optional[GuidelineContinuousProposition]],
        customer_dependant_action_detections: Sequence[
            Optional[CustomerDependentActionProposition]
        ],
        agent_intention_propositions: Sequence[Optional[AgentIntentionProposition]],
        tool_running_action_propositions: Sequence[Optional[ToolRunningActionProposition]],
    ) -> Sequence[InvoiceGuidelineData]:
        results = []
        for (
            payload_action,
            payload_continuous,
            payload_customer_dependent,
            agent_intention,
            tool_running_action,
        ) in zip(
            action_propositions,
            continuous_propositions,
            customer_dependant_action_detections,
            agent_intention_propositions,
            tool_running_action_propositions,
        ):
            properties_prop: dict[str, JSONSerializable] = {
                **{
                    "continuous": payload_continuous.is_continuous if payload_continuous else None,
                    "customer_dependent_action_data": payload_customer_dependent.model_dump()
                    if payload_customer_dependent
                    else None,
                    "agent_intention_condition": agent_intention.rewritten_condition
                    if agent_intention
                    and agent_intention.rewritten_condition
                    and agent_intention.is_agent_intention
                    else None,
                    "internal_action": payload_action.content.action if payload_action else None,
                },
                **(
                    {"tool_running_only": tool_running_action.is_tool_running_only}
                    if tool_running_action
                    else {}
                ),
            }

            invoice_data = InvoiceGuidelineData(
                coherence_checks=None,
                entailment_propositions=None,
                properties_proposition=properties_prop,
            )

            results.append(invoice_data)

        return results

    async def evaluate(
        self,
        payloads: Sequence[GuidelinePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[InvoiceGuidelineData]:
        action_propositions = await self._propose_actions(
            payloads,
            progress_report,
        )

        continuous_propositions = await self._propose_continuous(
            payloads,
            action_propositions,
            progress_report,
        )

        customer_dependant_action_detections = await self._detect_customer_dependant_actions(
            payloads, action_propositions, progress_report
        )

        agent_intention_propositions = await self._propose_agent_intention(
            payloads, progress_report
        )

        tool_running_action_propositions = await self._detect_tool_running_actions(
            payloads, progress_report
        )

        return self._build_invoice_data(
            action_propositions,
            continuous_propositions,
            customer_dependant_action_detections,
            agent_intention_propositions,
            tool_running_action_propositions,
        )

    async def _propose_actions(
        self,
        payloads: Sequence[GuidelinePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[GuidelineActionProposition]]:
        tasks: list[asyncio.Task[Optional[GuidelineActionProposition]]] = []
        indices: list[int] = []

        for i, p in enumerate(payloads):
            if p.action_proposition:
                indices.append(i)
                tasks.append(
                    asyncio.create_task(
                        self._guideline_action_proposer.propose_action(
                            guideline=p.content,
                            tool_ids=p.tool_ids or [],
                            progress_report=progress_report,
                        )
                    )
                )

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[GuidelineActionProposition]] = [None] * len(payloads)
        for i, res in zip(indices, sparse_results):
            results[i] = res

        return results

    async def _detect_customer_dependant_actions(
        self,
        payloads: Sequence[GuidelinePayload],
        proposed_actions: Sequence[Optional[GuidelineActionProposition]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[CustomerDependentActionProposition]]:
        tasks: list[asyncio.Task[CustomerDependentActionProposition]] = []
        indices: list[int] = []
        for i, (p, action_prop) in enumerate(zip(payloads, proposed_actions)):
            if not p.properties_proposition and not p.journey_node_proposition:
                continue
            action_to_use = (
                action_prop.content.action if action_prop is not None else p.content.action
            )
            guideline_content = GuidelineContent(
                condition=p.content.condition,
                action=action_to_use,
            )
            indices.append(i)
            tasks.append(
                asyncio.create_task(
                    self._customer_dependent_action_detector.detect_if_customer_dependent(
                        guideline=guideline_content,
                        progress_report=progress_report,
                    )
                )
            )
        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[CustomerDependentActionProposition]] = [None] * len(payloads)
        for i, res in zip(indices, sparse_results):
            results[i] = res
        return results

    async def _propose_continuous(
        self,
        payloads: Sequence[GuidelinePayload],
        proposed_actions: Sequence[Optional[GuidelineActionProposition]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[GuidelineContinuousProposition]]:
        tasks: list[asyncio.Task[GuidelineContinuousProposition]] = []
        indices: list[int] = []

        for i, (p, action_prop) in enumerate(zip(payloads, proposed_actions)):
            if not p.properties_proposition:
                continue

            action_to_use = (
                action_prop.content.action if action_prop is not None else p.content.action
            )
            guideline_content = GuidelineContent(
                condition=p.content.condition,
                action=action_to_use,
            )

            indices.append(i)
            tasks.append(
                asyncio.create_task(
                    self._guideline_continuous_proposer.propose_continuous(
                        guideline=guideline_content,
                        progress_report=progress_report,
                    )
                )
            )

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[GuidelineContinuousProposition]] = [None] * len(payloads)
        for i, res in zip(indices, sparse_results):
            results[i] = res
        return results

    async def _propose_agent_intention(
        self,
        payloads: Sequence[GuidelinePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[AgentIntentionProposition]]:
        tasks: list[asyncio.Task[AgentIntentionProposition]] = []
        indices: list[int] = []

        for i, p in enumerate(payloads):
            if not p.properties_proposition:
                continue

            guideline_content = GuidelineContent(
                condition=p.content.condition,
                action=p.content.action,
            )

            indices.append(i)
            tasks.append(
                asyncio.create_task(
                    self._agent_intention_proposer.propose_agent_intention(
                        guideline=guideline_content,
                        progress_report=progress_report,
                    )
                )
            )

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[AgentIntentionProposition]] = [None] * len(payloads)
        for i, res in zip(indices, sparse_results):
            results[i] = res
        return results

    async def _detect_tool_running_actions(
        self,
        payloads: Sequence[GuidelinePayload],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[Optional[ToolRunningActionProposition]]:
        tasks: list[asyncio.Task[ToolRunningActionProposition]] = []
        indices: list[int] = []

        for i, p in enumerate(payloads):
            if not p.journey_node_proposition:
                continue

            tasks.append(
                asyncio.create_task(
                    self._tool_running_action_detector.detect_if_tool_running(
                        guideline=p.content,
                        tool_ids=p.tool_ids,
                        progress_report=progress_report,
                    )
                )
            )
            indices.append(i)

        sparse_results = await async_utils.safe_gather(*tasks)
        results: list[Optional[ToolRunningActionProposition]] = [None] * len(payloads)

        for i, res in zip(indices, sparse_results):
            results[i] = res

        return results


class BehavioralChangeEvaluator:
    def __init__(
        self,
        logger: Logger,
        background_task_service: BackgroundTaskService,
        agent_store: AgentStore,
        guideline_store: GuidelineStore,
        journey_store: JourneyStore,
        evaluation_store: EvaluationStore,
        entity_queries: EntityQueries,
        journey_guideline_projection: JourneyGuidelineProjection,
        guideline_action_proposer: GuidelineActionProposer,
        guideline_continuous_proposer: GuidelineContinuousProposer,
        customer_dependent_action_detector: CustomerDependentActionDetector,
        agent_intention_proposer: AgentIntentionProposer,
        tool_running_action_detector: ToolRunningActionDetector,
        relative_action_proposer: RelativeActionProposer,
    ) -> None:
        self._logger = logger
        self._background_task_service = background_task_service

        self._agent_store = agent_store

        self._evaluation_store = evaluation_store
        self._entity_queries = entity_queries

        self._guideline_evaluator = GuidelineEvaluator(
            logger=logger,
            entity_queries=entity_queries,
            guideline_action_proposer=guideline_action_proposer,
            guideline_continuous_proposer=guideline_continuous_proposer,
            customer_dependent_action_detector=customer_dependent_action_detector,
            agent_intention_proposer=agent_intention_proposer,
            tool_running_action_detector=tool_running_action_detector,
        )

        self._journey_evaluator = JourneyEvaluator(
            logger=logger,
            guideline_store=guideline_store,
            journey_store=journey_store,
            journey_guideline_projection=journey_guideline_projection,
            relative_action_proposer=relative_action_proposer,
        )

    async def validate_payloads(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> None:
        if not payload_descriptors:
            raise EvaluationValidationError("No payloads provided for the evaluation task.")

    async def create_evaluation_task(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> EvaluationId:
        await self.validate_payloads(payload_descriptors)

        evaluation = await self._evaluation_store.create_evaluation(
            payload_descriptors,
        )

        await self._background_task_service.start(
            self._run_evaluation(evaluation),
            tag=f"evaluation({evaluation.id})",
        )

        return evaluation.id

    async def _run_evaluation(
        self,
        evaluation: Evaluation,
    ) -> None:
        async def _update_progress(percentage: float) -> None:
            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"progress": percentage},
            )

        progress_report = ProgressReport(_update_progress)

        try:
            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"status": EvaluationStatus.RUNNING},
            )

            guideline_evaluation_data, journey_evaluation_data = await async_utils.safe_gather(
                self._guideline_evaluator.evaluate(
                    payloads=[
                        cast(GuidelinePayload, invoice.payload)
                        for invoice in evaluation.invoices
                        if invoice.kind == PayloadKind.GUIDELINE
                    ],
                    progress_report=progress_report,
                ),
                self._journey_evaluator.evaluate(
                    payloads=[
                        cast(JourneyPayload, invoice.payload)
                        for invoice in evaluation.invoices
                        if invoice.kind == PayloadKind.JOURNEY
                    ],
                    progress_report=progress_report,
                ),
            )

            evaluation_data: Sequence[InvoiceData] = list(guideline_evaluation_data) + list(
                journey_evaluation_data
            )

            invoices: list[Invoice] = []
            for i, result in enumerate(evaluation_data):
                invoice_checksum = md5_checksum(str(evaluation.invoices[i].payload))
                state_version = str(hash("Temporarily"))

                invoices.append(
                    Invoice(
                        kind=evaluation.invoices[i].kind,
                        payload=evaluation.invoices[i].payload,
                        checksum=invoice_checksum,
                        state_version=state_version,
                        approved=True,
                        data=result,
                        error=None,
                    )
                )

            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"invoices": invoices},
            )

            self._logger.trace(f"evaluation task '{evaluation.id}' completed")

            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"status": EvaluationStatus.COMPLETED},
            )

        except Exception as exc:
            logger_level = "info" if isinstance(exc, EvaluationError) else "error"
            getattr(self._logger, logger_level)(
                f"Evaluation task '{evaluation.id}' failed due to the following error: '{str(exc)}'"
            )

            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={
                    "status": EvaluationStatus.FAILED,
                    "error": str(exc) + str(traceback.format_exception(exc)),
                },
            )

            raise
