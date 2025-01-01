# Copyright 2024 Emcie Co Ltd.
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
import hashlib
from itertools import chain
from typing import Any, Iterable, Optional, OrderedDict, Sequence, cast

from parlant.core import async_utils
from parlant.core.agents import Agent, AgentStore
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.evaluations import (
    GuidelineCoherenceCheck,
    GuidelineConnectionProposition,
    Evaluation,
    EvaluationStatus,
    EvaluationId,
    GuidelinePayload,
    Invoice,
    GuidelineInvoiceData,
    EvaluationStore,
    PayloadDescriptor,
    PayloadKind,
    StyleGuideCoherenceCheck,
    StyleGuideInvoiceData,
    StyleGuidePayload,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineStore, GuidelineId
from parlant.core.style_guides import StyleGuide, StyleGuideContent, StyleGuideStore, StyleGuideId
from parlant.core.services.indexing.guideline_coherence_checker import (
    GuidelineCoherenceChecker,
)
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
)
from parlant.core.logging import Logger
from parlant.core.services.indexing.style_guide_coherence_checker import StyleGuideCoherenceChecker


class EvaluationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class EvaluationValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def md5_checksum(input: str) -> str:
    md5_hash = hashlib.md5()
    md5_hash.update(input.encode("utf-8"))

    return md5_hash.hexdigest()


class GuidelineEvaluator:
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        guideline_connection_proposer: GuidelineConnectionProposer,
        coherence_checker: GuidelineCoherenceChecker,
    ) -> None:
        self._logger = logger
        self._guideline_store = guideline_store
        self._guideline_connection_proposer = guideline_connection_proposer
        self._coherence_checker = coherence_checker

    async def evaluate(
        self,
        agent: Agent,
        payloads: Sequence[GuidelinePayload],
        progress_report: ProgressReport,
    ) -> Sequence[GuidelineInvoiceData]:
        existing_guidelines = await self._guideline_store.list_guidelines(guideline_set=agent.id)

        tasks: list[asyncio.Task[Any]] = []
        coherence_checks_task: Optional[
            asyncio.Task[Optional[Iterable[Sequence[GuidelineCoherenceCheck]]]]
        ] = None

        connection_propositions_task: Optional[
            asyncio.Task[Optional[Iterable[Sequence[GuidelineConnectionProposition]]]]
        ] = None

        coherence_checks_task = asyncio.create_task(
            self._check_payloads_coherence(
                agent,
                payloads,
                existing_guidelines,
                progress_report,
            )
        )
        tasks.append(coherence_checks_task)

        connection_propositions_task = asyncio.create_task(
            self._propose_payloads_connections(
                agent,
                payloads,
                existing_guidelines,
                progress_report,
            )
        )
        tasks.append(connection_propositions_task)

        if tasks:
            await async_utils.safe_gather(*tasks)

        coherence_checks: Optional[Iterable[Sequence[GuidelineCoherenceCheck]]] = []
        if coherence_checks_task:
            coherence_checks = coherence_checks_task.result()

        connection_propositions: Optional[Iterable[Sequence[GuidelineConnectionProposition]]] = None
        if connection_propositions_task:
            connection_propositions = connection_propositions_task.result()

        if coherence_checks:
            return [
                GuidelineInvoiceData(
                    coherence_checks=payload_coherence_checks,
                    connection_propositions=None,
                )
                for payload_coherence_checks in coherence_checks
            ]

        elif connection_propositions:
            return [
                GuidelineInvoiceData(
                    coherence_checks=[],
                    connection_propositions=payload_connection_propositions,
                )
                for payload_connection_propositions in connection_propositions
            ]

        else:
            return [
                GuidelineInvoiceData(
                    coherence_checks=[],
                    connection_propositions=None,
                )
                for _ in payloads
            ]

    async def _check_payloads_coherence(
        self,
        agent: Agent,
        payloads: Sequence[GuidelinePayload],
        existing_guidelines: Sequence[Guideline],
        progress_report: ProgressReport,
    ) -> Optional[Iterable[Sequence[GuidelineCoherenceCheck]]]:
        guidelines_to_evaluate = [p.content for p in payloads if p.coherence_check]

        guidelines_to_skip = [(p.content, False) for p in payloads if not p.coherence_check]

        updated_ids = {cast(GuidelineId, p.updated_id) for p in payloads if p.operation == "update"}

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

        coherence_checks_by_guideline_payload: OrderedDict[str, list[GuidelineCoherenceCheck]] = (
            OrderedDict({f"{p.content.condition}{p.content.action}": [] for p in payloads})
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
                    GuidelineCoherenceCheck(
                        kind="contradiction_with_another_evaluated_guideline"
                        if f"{c.guideline_b.condition}{c.guideline_b.action}"
                        in coherence_checks_by_guideline_payload
                        else "contradiction_with_existing_guideline",
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
                    GuidelineCoherenceCheck(
                        kind="contradiction_with_another_evaluated_guideline",
                        first=c.guideline_a,
                        second=c.guideline_b,
                        issue=c.actions_contradiction_rationale,
                        severity=c.actions_contradiction_severity,
                    )
                )

        return coherence_checks_by_guideline_payload.values()

    async def _propose_payloads_connections(
        self,
        agent: Agent,
        payloads: Sequence[GuidelinePayload],
        existing_guidelines: Sequence[Guideline],
        progress_report: ProgressReport,
    ) -> Optional[Iterable[Sequence[GuidelineConnectionProposition]]]:
        proposed_guidelines = [p.content for p in payloads if p.connection_proposition]

        guidelines_to_skip = [(p.content, False) for p in payloads if not p.connection_proposition]

        updated_ids = {p.updated_id for p in payloads if p.operation == "update"}

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
            str, list[GuidelineConnectionProposition]
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
                    GuidelineConnectionProposition(
                        check_kind="connection_with_another_evaluated_guideline"
                        if f"{c.target.condition}{c.target.action}"
                        in connection_results_by_guideline_payload
                        else "connection_with_existing_guideline",
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
                    GuidelineConnectionProposition(
                        check_kind="connection_with_another_evaluated_guideline"
                        if f"{c.source.condition}{c.source.action}"
                        in connection_results_by_guideline_payload
                        else "connection_with_existing_guideline",
                        source=c.source,
                        target=c.target,
                    )
                )

        return connection_results_by_guideline_payload.values()


class StyleGuideEvaluator:
    def __init__(
        self,
        logger: Logger,
        style_guide_store: StyleGuideStore,
        coherence_checker: StyleGuideCoherenceChecker,
    ) -> None:
        self._logger = logger
        self._style_guide_store = style_guide_store
        self._coherence_checker = coherence_checker

    async def evaluate(
        self,
        agent: Agent,
        payloads: Sequence[StyleGuidePayload],
        progress_report: ProgressReport,
    ) -> Sequence[StyleGuideInvoiceData]:
        existing_style_guides = await self._style_guide_store.list_style_guides(
            style_guide_set=agent.id
        )

        tasks: list[asyncio.Task[Any]] = []
        coherence_checks_task: Optional[
            asyncio.Task[Optional[Iterable[Sequence[StyleGuideCoherenceCheck]]]]
        ] = None

        coherence_checks_task = asyncio.create_task(
            self._check_payloads_coherence(
                agent,
                payloads,
                existing_style_guides,
                progress_report,
            )
        )
        tasks.append(coherence_checks_task)

        if tasks:
            await async_utils.safe_gather(*tasks)

        coherence_checks: Optional[Iterable[Sequence[StyleGuideCoherenceCheck]]] = []
        if coherence_checks_task:
            coherence_checks = coherence_checks_task.result()

        if coherence_checks:
            return [
                StyleGuideInvoiceData(
                    coherence_checks=payload_coherence_checks,
                )
                for payload_coherence_checks in coherence_checks
            ]

        else:
            return [
                StyleGuideInvoiceData(
                    coherence_checks=[],
                )
                for _ in payloads
            ]

    async def _check_payloads_coherence(
        self,
        agent: Agent,
        payloads: Sequence[StyleGuidePayload],
        existing_style_guides: Sequence[StyleGuide],
        progress_report: ProgressReport,
    ) -> Optional[Iterable[Sequence[StyleGuideCoherenceCheck]]]:
        style_guides_to_evaluate = [p.content for p in payloads if p.coherence_check]

        style_guides_to_skip = [(p.content, False) for p in payloads if not p.coherence_check]

        updated_ids = {
            cast(StyleGuideId, p.updated_id) for p in payloads if p.operation == "update"
        }

        remaining_existing_style_guides = []

        for s in existing_style_guides:
            if s.id not in updated_ids:
                remaining_existing_style_guides.append(
                    (
                        StyleGuideContent(
                            principle=s.content.principle,
                            examples=s.content.examples,
                        ),
                        True,
                    )
                )
            else:
                updated_ids.remove(s.id)

        if len(updated_ids) > 0:
            raise EvaluationError(
                f"StyleGuide ID(s): {', '.join(list(updated_ids))} in '{agent.id}' agent do not exist."
            )

        comparison_style_guides = style_guides_to_skip + remaining_existing_style_guides

        incoherences = await self._coherence_checker.propose_incoherencies(
            agent=agent,
            style_guides_to_evaluate=style_guides_to_evaluate,
            comparison_style_guides=[s for s, _ in comparison_style_guides],
            progress_report=progress_report,
        )

        if not incoherences:
            return None

        coherence_checks_by_guideline_payload: OrderedDict[str, list[StyleGuideCoherenceCheck]] = (
            OrderedDict({f"{p.content.principle}{p.content.examples}": [] for p in payloads})
        )

        guideline_payload_is_skipped_pairs = {
            f"{p.content.principle}{p.content.examples}": p.coherence_check for p in payloads
        }

        for c in incoherences:
            if (
                f"{c.style_guide_a.principle}{c.style_guide_a.examples}"
                in coherence_checks_by_guideline_payload
                and guideline_payload_is_skipped_pairs[
                    f"{c.style_guide_a.principle}{c.style_guide_a.examples}"
                ]
            ):
                coherence_checks_by_guideline_payload[
                    f"{c.style_guide_a.principle}{c.style_guide_a.examples}"
                ].append(
                    StyleGuideCoherenceCheck(
                        kind="contradiction_with_another_evaluated_style_guide"
                        if f"{c.style_guide_b.principle}{c.style_guide_b.examples}"
                        in coherence_checks_by_guideline_payload
                        else "contradiction_with_existing_style_guide",
                        first=c.style_guide_a,
                        second=c.style_guide_b,
                        issue=c.principles_rationale,
                        severity=c.principles_severity,
                    )
                )

            if (
                f"{c.style_guide_b.principle}{c.style_guide_b.examples}"
                in coherence_checks_by_guideline_payload
                and guideline_payload_is_skipped_pairs[
                    f"{c.style_guide_b.principle}{c.style_guide_b.examples}"
                ]
            ):
                coherence_checks_by_guideline_payload[
                    f"{c.style_guide_b.principle}{c.style_guide_b.examples}"
                ].append(
                    StyleGuideCoherenceCheck(
                        kind="contradiction_with_another_evaluated_style_guide",
                        first=c.style_guide_a,
                        second=c.style_guide_b,
                        issue=c.principles_rationale,
                        severity=c.principles_severity,
                    )
                )

        return coherence_checks_by_guideline_payload.values()


class BehavioralChangeEvaluator:
    def __init__(
        self,
        logger: Logger,
        background_task_service: BackgroundTaskService,
        agent_store: AgentStore,
        evaluation_store: EvaluationStore,
        guideline_store: GuidelineStore,
        guideline_connection_proposer: GuidelineConnectionProposer,
        style_guide_store: StyleGuideStore,
        guideline_coherence_checker: GuidelineCoherenceChecker,
        style_guide_coherence_checker: StyleGuideCoherenceChecker,
    ) -> None:
        self._logger = logger
        self._background_task_service = background_task_service
        self._agent_store = agent_store
        self._evaluation_store = evaluation_store

        self._guideline_store = guideline_store
        self._guideline_evaluator = GuidelineEvaluator(
            logger=logger,
            guideline_store=guideline_store,
            guideline_connection_proposer=guideline_connection_proposer,
            coherence_checker=guideline_coherence_checker,
        )

        self._style_guide_store = style_guide_store
        self._style_guide_evaluator = StyleGuideEvaluator(
            logger=logger,
            style_guide_store=style_guide_store,
            coherence_checker=style_guide_coherence_checker,
        )

    async def validate_payloads(
        self,
        agent: Agent,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> None:
        if not payload_descriptors:
            raise EvaluationValidationError("No payloads provided for the evaluation task.")

        guideline_payloads = [p for k, p in payload_descriptors if k == PayloadKind.GUIDELINE]

        if guideline_payloads:
            seen_guidelines = set((g.content) for g in guideline_payloads)
            if len(seen_guidelines) < len(guideline_payloads):
                raise EvaluationValidationError(
                    "Duplicate guideline found among the provided guidelines."
                )

            existing_guidelines = await self._guideline_store.list_guidelines(
                guideline_set=agent.id,
            )

            if guideline := next(
                iter(g for g in existing_guidelines if (g.content) in seen_guidelines),
                None,
            ):
                raise EvaluationValidationError(
                    f"Duplicate guideline found against existing guideline: {str(guideline)} in {agent.id} guideline_set"
                )

        style_guide_payloads = [
            cast(StyleGuidePayload, p)
            for k, p in payload_descriptors
            if k == PayloadKind.STYLE_GUIDE
        ]

        if style_guide_payloads:
            seen_style_guides = set((s.content.principle) for s in style_guide_payloads)
            if len(seen_style_guides) < len(style_guide_payloads):
                raise EvaluationValidationError(
                    "Duplicate style guide found among the provided style guides."
                )

            existing_style_guides = await self._style_guide_store.list_style_guides(
                style_guide_set=agent.id,
            )

            if style_guide := next(
                iter(
                    s for s in existing_style_guides if (s.content.principle) in seen_style_guides
                ),
                None,
            ):
                raise EvaluationValidationError(
                    f"Duplicate style guide found against existing style guide: {str(style_guide.content.principle)} in {agent.id} style_guide_set"
                )

    async def create_evaluation_task(
        self,
        agent: Agent,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> EvaluationId:
        await self.validate_payloads(agent, payload_descriptors)

        evaluation = await self._evaluation_store.create_evaluation(
            agent.id,
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

            agent = await self._agent_store.read_agent(agent_id=evaluation.agent_id)

            invoices: list[Invoice] = []

            guideline_payloads = [
                cast(GuidelinePayload, invoice.payload)
                for invoice in evaluation.invoices
                if invoice.kind == PayloadKind.GUIDELINE
            ]
            guideline_evaluation_task = self._guideline_evaluator.evaluate(
                agent=agent,
                payloads=guideline_payloads,
                progress_report=progress_report,
            )

            style_guide_payloads = [
                cast(StyleGuidePayload, invoice.payload)
                for invoice in evaluation.invoices
                if invoice.kind == PayloadKind.STYLE_GUIDE
            ]
            style_guide_evaluation_task = self._style_guide_evaluator.evaluate(
                agent=agent,
                payloads=style_guide_payloads,
                progress_report=progress_report,
            )

            guideline_evaluation_data, style_guide_evaluation_data = await async_utils.safe_gather(
                guideline_evaluation_task, style_guide_evaluation_task
            )

            for payload, result in chain(
                zip(guideline_payloads, guideline_evaluation_data),
                zip(style_guide_payloads, style_guide_evaluation_data),
            ):
                invoice_checksum = md5_checksum(str(payload))
                state_version = str(hash("Temporarily"))
                is_approved = not bool(
                    cast(GuidelineInvoiceData | StyleGuideInvoiceData, result).coherence_checks
                )

                invoices.append(
                    Invoice(
                        kind=PayloadKind.GUIDELINE
                        if isinstance(payload, GuidelinePayload)
                        else PayloadKind.STYLE_GUIDE,
                        payload=cast(GuidelinePayload | StyleGuidePayload, payload),
                        checksum=invoice_checksum,
                        state_version=state_version,
                        approved=is_approved,
                        data=cast(GuidelineInvoiceData | StyleGuideInvoiceData, result),
                        error=None,
                    )
                )

            await self._evaluation_store.update_evaluation(
                evaluation_id=evaluation.id,
                params={"invoices": invoices},
            )

            self._logger.info(f"evaluation task '{evaluation.id}' completed")

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
