import asyncio
from typing import Any, Iterable, Optional, OrderedDict, Sequence, cast

from emcie.server.core.agents import Agent, AgentStore
from emcie.server.core.evaluations import (
    CoherenceCheck,
    ConnectionProposition,
    Evaluation,
    EvaluationStatus,
    EvaluationId,
    Invoice,
    InvoiceGuidelineData,
    Payload,
    EvaluationStore,
    PayloadDescriptor,
    PayloadKind,
)
from emcie.server.core.guidelines import Guideline, GuidelineContent, GuidelineStore, GuidelineId
from emcie.server.core.services.indexing.coherence_checker import (
    CoherenceChecker,
)
from emcie.server.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
)
from emcie.server.core.logging import Logger
from emcie.server.core.common import ProgressReport, md5_checksum


class EvaluationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class EvaluationValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GuidelineEvaluator:
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        guideline_connection_proposer: GuidelineConnectionProposer,
        coherence_checker: CoherenceChecker,
    ) -> None:
        self._logger = logger
        self._guideline_store = guideline_store
        self._guideline_connection_proposer = guideline_connection_proposer
        self._coherence_checker = coherence_checker

    async def evaluate(
        self,
        agent: Agent,
        payloads: Sequence[Payload],
        progress_report: ProgressReport,
    ) -> Sequence[InvoiceGuidelineData]:
        existing_guidelines = await self._guideline_store.list_guidelines(guideline_set=agent.id)

        tasks: list[asyncio.Task[Any]] = []
        coherence_checks_task: Optional[
            asyncio.Task[Optional[Iterable[Sequence[CoherenceCheck]]]]
        ] = None

        connection_propositions_task: Optional[
            asyncio.Task[Optional[Iterable[Sequence[ConnectionProposition]]]]
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
            await asyncio.gather(*tasks)

        coherence_checks: Optional[Iterable[Sequence[CoherenceCheck]]] = []
        if coherence_checks_task:
            coherence_checks = coherence_checks_task.result()

        connection_propositions: Optional[Iterable[Sequence[ConnectionProposition]]] = None
        if connection_propositions_task:
            connection_propositions = connection_propositions_task.result()

        if coherence_checks:
            return [
                InvoiceGuidelineData(
                    coherence_checks=payload_coherence_checks,
                    connection_propositions=None,
                )
                for payload_coherence_checks in coherence_checks
            ]

        elif connection_propositions:
            return [
                InvoiceGuidelineData(
                    coherence_checks=[],
                    connection_propositions=payload_connection_propositions,
                )
                for payload_connection_propositions in connection_propositions
            ]

        else:
            return [
                InvoiceGuidelineData(
                    coherence_checks=[],
                    connection_propositions=None,
                )
                for _ in payloads
            ]

    async def _check_payloads_coherence(
        self,
        agent: Agent,
        payloads: Sequence[Payload],
        existing_guidelines: Sequence[Guideline],
        progress_report: ProgressReport,
    ) -> Optional[Iterable[Sequence[CoherenceCheck]]]:
        guidelines_to_evaluate = [p.content for p in payloads if p.coherence_check]

        guidelines_to_skip = [(p.content, False) for p in payloads if not p.coherence_check]

        updated_ids = {cast(GuidelineId, p.updated_id) for p in payloads if p.action == "update"}

        remaining_existing_guidelines = []

        for g in existing_guidelines:
            if g.id not in updated_ids:
                remaining_existing_guidelines.append(
                    (GuidelineContent(predicate=g.content.predicate, action=g.content.action), True)
                )
            else:
                updated_ids.remove(g.id)

        if len(updated_ids) > 0:
            raise EvaluationError(
                f"Guideline ID(s): {', '.join(updated_ids)} in {agent.name} agent do not exist."
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
            {f"{p.content.predicate}{p.content.action}": [] for p in payloads}
        )

        guideline_payload_is_skipped_pairs = {
            f"{p.content.predicate}{p.content.action}": p.coherence_check for p in payloads
        }

        for c in incoherences:
            if (
                f"{c.guideline_a.predicate}{c.guideline_a.action}"
                in coherence_checks_by_guideline_payload
                and guideline_payload_is_skipped_pairs[
                    f"{c.guideline_a.predicate}{c.guideline_a.action}"
                ]
            ):
                coherence_checks_by_guideline_payload[
                    f"{c.guideline_a.predicate}{c.guideline_a.action}"
                ].append(
                    CoherenceCheck(
                        kind="contradiction_with_another_evaluated_guideline"
                        if f"{c.guideline_b.predicate}{c.guideline_b.action}"
                        in coherence_checks_by_guideline_payload
                        else "contradiction_with_existing_guideline",
                        first=c.guideline_a,
                        second=c.guideline_b,
                        issue=c.actions_contradiction_rationale,
                        severity=c.actions_contradiction_severity,
                    )
                )

            if (
                f"{c.guideline_b.predicate}{c.guideline_b.action}"
                in coherence_checks_by_guideline_payload
                and guideline_payload_is_skipped_pairs[
                    f"{c.guideline_b.predicate}{c.guideline_b.action}"
                ]
            ):
                coherence_checks_by_guideline_payload[
                    f"{c.guideline_b.predicate}{c.guideline_b.action}"
                ].append(
                    CoherenceCheck(
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
        payloads: Sequence[Payload],
        existing_guidelines: Sequence[Guideline],
        progress_report: ProgressReport,
    ) -> Optional[Iterable[Sequence[ConnectionProposition]]]:
        proposed_guidelines = [p.content for p in payloads if p.connection_proposition]

        guidelines_to_skip = [(p.content, False) for p in payloads if not p.connection_proposition]

        updated_ids = {p.updated_id for p in payloads if p.action == "update"}

        remaining_existing_guidelines = [
            (GuidelineContent(predicate=g.content.predicate, action=g.content.action), True)
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

        connection_results_by_guideline_payload: OrderedDict[str, list[ConnectionProposition]] = (
            OrderedDict({f"{p.content.predicate}{p.content.action}": [] for p in payloads})
        )
        guideline_payload_is_skipped_pairs = {
            f"{p.content.predicate}{p.content.action}": p.connection_proposition for p in payloads
        }

        for c in connection_propositions:
            if (
                f"{c.source.predicate}{c.source.action}" in connection_results_by_guideline_payload
                and guideline_payload_is_skipped_pairs[f"{c.source.predicate}{c.source.action}"]
            ):
                connection_results_by_guideline_payload[
                    f"{c.source.predicate}{c.source.action}"
                ].append(
                    ConnectionProposition(
                        check_kind="connection_with_another_evaluated_guideline"
                        if f"{c.target.predicate}{c.target.action}"
                        in connection_results_by_guideline_payload
                        else "connection_with_existing_guideline",
                        source=c.source,
                        target=c.target,
                        connection_kind=c.kind,
                    )
                )

            if (
                f"{c.target.predicate}{c.target.action}" in connection_results_by_guideline_payload
                and guideline_payload_is_skipped_pairs[f"{c.target.predicate}{c.target.action}"]
            ):
                connection_results_by_guideline_payload[
                    f"{c.target.predicate}{c.target.action}"
                ].append(
                    ConnectionProposition(
                        check_kind="connection_with_another_evaluated_guideline"
                        if f"{c.source.predicate}{c.source.action}"
                        in connection_results_by_guideline_payload
                        else "connection_with_existing_guideline",
                        source=c.source,
                        target=c.target,
                        connection_kind=c.kind,
                    )
                )

        return connection_results_by_guideline_payload.values()


class BehavioralChangeEvaluator:
    def __init__(
        self,
        logger: Logger,
        agent_store: AgentStore,
        evaluation_store: EvaluationStore,
        guideline_store: GuidelineStore,
        guideline_connection_proposer: GuidelineConnectionProposer,
        coherence_checker: CoherenceChecker,
    ) -> None:
        self._logger = logger
        self._agent_store = agent_store
        self._evaluation_store = evaluation_store
        self._guideline_store = guideline_store
        self._guideline_evaluator = GuidelineEvaluator(
            logger=logger,
            guideline_store=guideline_store,
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

        guideline_payloads = [p for k, p in payload_descriptors if k == PayloadKind.GUIDELINE]

        if guideline_payloads:

            async def _check_for_duplications() -> None:
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
            agent.id,
            payload_descriptors,
        )

        asyncio.create_task(self.run_evaluation(evaluation))

        return evaluation.id

    async def run_evaluation(
        self,
        evaluation: Evaluation,
    ) -> None:
        self._logger.info(f"Starting evaluation task '{evaluation.id}'")

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
