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
import traceback
from dataclasses import replace
from typing import Optional, Sequence, cast

from parlant.core import async_utils
from parlant.core.agents import AgentStore
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import JSONSerializable, md5_checksum
from parlant.core.evaluations import (
    Evaluation,
    EvaluationStatus,
    EvaluationId,
    GuidelinePayload,
    InvoiceData,
    InvoiceJourneyData,
    JourneyPayload,
    Invoice,
    InvoiceGuidelineData,
    EvaluationStore,
    PayloadDescriptor,
    PayloadKind,
    PayloadOperation,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineStore
from parlant.core.journey_guideline_projection import (
    JourneyGuidelineProjection,
    extract_node_id_from_journey_node_guideline_id,
)
from parlant.core.journeys import Journey, JourneyId, JourneyNodeId, JourneyStore
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
from parlant.core.services.indexing.guideline_continuous_proposer import (
    GuidelineContinuousProposer,
    GuidelineContinuousProposition,
)
from parlant.core.loggers import Logger
from parlant.core.entity_cq import EntityQueries
from parlant.core.services.indexing.journey_reachable_nodes_evaluation import (
    JourneyReachableNodesEvaluator,
    ReachableNodesEvaluation,
)
from parlant.core.services.indexing.relative_action_proposer import (
    RelativeActionProposer,
    RelativeActionProposition,
)
from parlant.core.services.indexing.tool_running_action_detector import (
    ToolRunningActionDetector,
    ToolRunningActionProposition,
)


class EvaluationValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


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


class JourneyEvaluator:
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        journey_store: JourneyStore,
        journey_guideline_projection: JourneyGuidelineProjection,
        guideline_evaluator: GuidelineEvaluator,
        relative_action_proposer: RelativeActionProposer,
        journey_reachable_node_evaluator: JourneyReachableNodesEvaluator,
    ) -> None:
        self._logger = logger

        self._guideline_store = guideline_store
        self._journey_store = journey_store
        self._journey_guideline_projection = journey_guideline_projection
        self._guideline_evaluator = guideline_evaluator
        self._journey_reachable_node_evaluator = journey_reachable_node_evaluator

        self._relative_action_proposer = relative_action_proposer

    async def _build_invoice_data(
        self,
        relative_action_propositions: Sequence[RelativeActionProposition],
        reachable_nodes_evaluations: Sequence[ReachableNodesEvaluation],
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Guideline], tuple[Guideline]]],
    ) -> Sequence[InvoiceJourneyData]:
        # Create mapping from index to node_id for relative actions and reachable nodes
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

        for action_proposition, reachable_node_evaluation, journey_id in zip(
            relative_action_propositions, reachable_nodes_evaluations, journey_projections.keys()
        ):
            node_properties_proposition: dict[JourneyNodeId, dict[str, JSONSerializable]] = {}

            # Add guideline evaluation properties for each node
            _, step_guidelines, __ = journey_projections[journey_id]
            for guideline in step_guidelines:
                # Extract node ID directly from the guideline ID
                node_id = extract_node_id_from_journey_node_guideline_id(guideline.id)

                if node_id not in node_properties_proposition:
                    node_properties_proposition[node_id] = {}

                # Extract guideline evaluation metadata
                for key, value in cast(
                    dict[str, JSONSerializable], guideline.metadata.get("guideline_evaluation", {})
                ).items():
                    node_properties_proposition[node_id][key] = value

            for a in action_proposition.actions:
                node_id = index_to_node_ids[journey_id][a.index]
                if node_id not in node_properties_proposition:
                    node_properties_proposition[node_id] = {}
                node_properties_proposition[node_id]["internal_action"] = a.rewritten_actions

            for index, r in reachable_node_evaluation.node_to_reachable_follow_ups.items():
                node_id = index_to_node_ids[journey_id][index]
                if node_id not in node_properties_proposition:
                    node_properties_proposition[node_id] = {}
                if "journey_node" not in node_properties_proposition[node_id]:
                    node_properties_proposition[node_id]["journey_node"] = {}
                node_properties_proposition[node_id]["journey_node"] = {
                    **cast(
                        dict[str, JSONSerializable],
                        node_properties_proposition[node_id]["journey_node"],
                    ),
                    "reachable_follow_ups": [{"condition": c, "path": p} for c, p in r],
                }

            invoice_data = InvoiceJourneyData(
                node_properties_proposition=node_properties_proposition,
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

        # Evaluate guidelines to get metadata for journey nodes
        journey_projections_with_metadata = await self._add_guideline_metadata_to_projections(
            journey_projections, progress_report
        )

        relative_action_propositions = await self._propose_relative_actions(
            journey_projections_with_metadata,
            progress_report,
        )

        reachable_nodes_evaluations = await self._evaluate_reachable_nodes(
            journey_projections_with_metadata,
            progress_report,
        )

        invoices = await self._build_invoice_data(
            relative_action_propositions,
            reachable_nodes_evaluations,
            journey_projections_with_metadata,
        )

        return invoices

    async def _add_guideline_metadata_to_projections(
        self,
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Guideline], tuple[Guideline]]],
        progress_report: Optional[ProgressReport] = None,
    ) -> dict[JourneyId, tuple[Journey, Sequence[Guideline], tuple[Guideline]]]:
        """Add guideline evaluation metadata to journey node guidelines."""
        guideline_payloads: list[GuidelinePayload] = []
        journey_to_node_guidelines: dict[JourneyId, dict[JourneyNodeId, Guideline]] = {}

        # Collect all nodes and create payloads
        for journey_id, (
            journey,
            step_guidelines,
            journey_conditions,
        ) in journey_projections.items():
            journey_to_node_guidelines[journey_id] = {}
            for guideline in step_guidelines:
                node_id = extract_node_id_from_journey_node_guideline_id(guideline.id)
                if node_id == JourneyStore.END_NODE_ID:
                    continue
                node = await self._journey_store.read_node(node_id=node_id)

                # Store the guideline by node_id for later mapping
                journey_to_node_guidelines[journey_id][node_id] = guideline

                # Create GuidelinePayload for journey node guidelines
                guideline_payload = GuidelinePayload(
                    content=guideline.content,
                    tool_ids=node.tools,
                    operation=PayloadOperation.ADD,
                    action_proposition=True,
                    properties_proposition=False,
                    journey_node_proposition=True,
                )
                guideline_payloads.append(guideline_payload)

        if not guideline_payloads:
            return journey_projections

        # Evaluate each guideline payload individually using async gather
        guideline_evaluation_tasks = [
            self._guideline_evaluator.evaluate(
                [payload],  # Pass each payload as a single-item list
                progress_report=progress_report,
            )
            for payload in guideline_payloads
        ]

        guideline_evaluation_results = await async_utils.safe_gather(*guideline_evaluation_tasks)

        guideline_evaluations = [result[0] for result in guideline_evaluation_results]

        # Add metadata back to the guidelines
        updated_projections: dict[
            JourneyId, tuple[Journey, Sequence[Guideline], tuple[Guideline]]
        ] = {}

        # Create a mapping from guideline payloads to evaluations
        evaluation_index = 0
        for journey_id, (
            journey,
            step_guidelines,
            journey_conditions,
        ) in journey_projections.items():
            updated_step_guidelines: list[Guideline] = []
            node_guidelines = journey_to_node_guidelines[journey_id]

            for guideline in step_guidelines:
                node_id = extract_node_id_from_journey_node_guideline_id(guideline.id)

                if node_id in node_guidelines and evaluation_index < len(guideline_evaluations):
                    evaluation_data = guideline_evaluations[evaluation_index]
                    evaluation_index += 1

                    updated_metadata = {
                        **(
                            evaluation_data.properties_proposition
                            if evaluation_data.properties_proposition
                            else {}
                        ),
                        **guideline.metadata,
                        **(
                            {"guideline_evaluation": evaluation_data.properties_proposition}
                            if evaluation_data.properties_proposition
                            else {}
                        ),
                    }

                    updated_guideline = replace(
                        guideline,
                        metadata=updated_metadata,
                    )
                    updated_step_guidelines.append(updated_guideline)
                else:
                    updated_step_guidelines.append(guideline)

            updated_projections[journey_id] = (journey, updated_step_guidelines, journey_conditions)

        return updated_projections

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

    async def _evaluate_reachable_nodes(
        self,
        journey_projections: dict[JourneyId, tuple[Journey, Sequence[Guideline], tuple[Guideline]]],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[ReachableNodesEvaluation]:
        tasks: list[asyncio.Task[ReachableNodesEvaluation]] = []

        for journey_id, (
            journey,
            step_guidelines,
            journey_conditions,
        ) in journey_projections.items():
            if not step_guidelines:
                continue

            tasks.append(
                asyncio.create_task(
                    self._journey_reachable_node_evaluator.evaluate_reachable_follow_ups(
                        node_guidelines=step_guidelines,
                        progress_report=progress_report,
                    )
                )
            )

        sparse_results = list(await async_utils.safe_gather(*tasks))

        return sparse_results


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
        journey_reachable_node_evaluator: JourneyReachableNodesEvaluator,
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
            guideline_evaluator=self._guideline_evaluator,
            relative_action_proposer=relative_action_proposer,
            journey_reachable_node_evaluator=journey_reachable_node_evaluator,
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
            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"status": EvaluationStatus.RUNNING},
            )

            evaluation_invoices = list(evaluation.invoices)

            guideline_evaluation_data, journey_evaluation_data = await async_utils.safe_gather(
                self._guideline_evaluator.evaluate(
                    payloads=[
                        cast(GuidelinePayload, invoice.payload)
                        for invoice in evaluation_invoices
                        if invoice.kind == PayloadKind.GUIDELINE
                    ],
                    progress_report=progress_report,
                ),
                self._journey_evaluator.evaluate(
                    payloads=[
                        cast(JourneyPayload, invoice.payload)
                        for invoice in evaluation_invoices
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
