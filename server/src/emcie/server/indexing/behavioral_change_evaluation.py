import asyncio
from typing import Iterable, Optional, OrderedDict, Sequence

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
from emcie.server.core.guidelines import Guideline, GuidelineContent, GuidelineStore
from emcie.server.indexing.coherence_checker import CoherenceChecker, ContradictionTest
from emcie.server.indexing.guideline_connection_proposer import GuidelineConnectionProposer
from emcie.server.logger import Logger
from emcie.server.utils import md5_checksum


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
        self.logger = logger
        self._guideline_store = guideline_store
        self._guideline_connection_proposer = guideline_connection_proposer
        self._coherence_checker = coherence_checker

    async def evaluate(
        self,
        payloads: Sequence[Payload],
    ) -> Sequence[InvoiceGuidelineData]:
        guideline_set = payloads[0].guideline_set

        guideline_to_evaluate = [
            GuidelineContent(
                predicate=p.predicate,
                action=p.action,
            )
            for p in payloads
        ]

        existing_guidelines = await self._guideline_store.list_guidelines(
            guideline_set=guideline_set
        )

        coherence_checks = await self._check_payloads_coherence(
            guideline_to_evaluate,
            existing_guidelines,
        )

        if not coherence_checks:
            connection_propositions = await self._propose_payloads_connections(
                guideline_to_evaluate, existing_guidelines
            )

            if connection_propositions:
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
                    for _ in range(len(payloads))
                ]

        return [
            InvoiceGuidelineData(
                coherence_checks=payload_coherence_checks,
                connection_propositions=None,
            )
            for payload_coherence_checks in coherence_checks
        ]

    async def _check_payloads_coherence(
        self,
        guidelines_to_evaluate: Sequence[GuidelineContent],
        existing_guidelines: Sequence[Guideline],
    ) -> Optional[Iterable[Sequence[CoherenceCheck]]]:
        coherence_checks = await self._coherence_checker.evaluate_coherence(
            guidelines_to_evaluate=guidelines_to_evaluate,
            comparison_guidelines=[
                GuidelineContent(predicate=g.content.predicate, action=g.content.action)
                for g in existing_guidelines
            ],
        )

        contradictions: dict[str, ContradictionTest] = {}

        for c in coherence_checks:
            key = f"{c.guideline_a.predicate}{c.guideline_a.action}"
            if (c.severity >= 6) and (
                key not in contradictions or c.severity > contradictions[key].severity
            ):
                contradictions[key] = c

        if not contradictions:
            return None

        coherence_checks_by_guideline_payload: OrderedDict[str, list[CoherenceCheck]] = OrderedDict(
            {f"{g.predicate}{g.action}": [] for g in guidelines_to_evaluate}
        )

        for c in contradictions.values():
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
                    issue=c.rationale,
                    severity=c.severity,
                )
            )

            if (
                f"{c.guideline_b.predicate}{c.guideline_b.action}"
                in coherence_checks_by_guideline_payload
            ):
                coherence_checks_by_guideline_payload[
                    f"{c.guideline_b.predicate}{c.guideline_b.action}"
                ].append(
                    CoherenceCheck(
                        kind="contradiction_with_another_evaluated_guideline",
                        first=c.guideline_a,
                        second=c.guideline_b,
                        issue=c.rationale,
                        severity=c.severity,
                    )
                )

        return coherence_checks_by_guideline_payload.values()

    async def _propose_payloads_connections(
        self,
        proposed_guidelines: Sequence[GuidelineContent],
        existing_guidelines: Sequence[Guideline],
    ) -> Optional[Iterable[Sequence[ConnectionProposition]]]:
        connection_propositions = [
            p
            for p in await self._guideline_connection_proposer.propose_connections(
                introduced_guidelines=proposed_guidelines,
                existing_guidelines=[
                    GuidelineContent(predicate=g.content.predicate, action=g.content.action)
                    for g in existing_guidelines
                ],
            )
            if p.score >= 6
        ]

        if not connection_propositions:
            return None

        connection_results: OrderedDict[str, list[ConnectionProposition]] = OrderedDict(
            {f"{g.predicate}{g.action}": [] for g in proposed_guidelines}
        )

        for c in connection_propositions:
            if f"{c.source.predicate}{c.source.action}" in connection_results:
                connection_results[f"{c.source.predicate}{c.source.action}"].append(
                    ConnectionProposition(
                        check_kind="connection_with_another_evaluated_guideline"
                        if f"{c.target.predicate}{c.target.action}" in connection_results
                        else "connection_with_existing_guideline",
                        source=c.source,
                        target=c.target,
                        connection_kind=c.kind,
                    )
                )

            if f"{c.target.predicate}{c.target.action}" in connection_results:
                connection_results[f"{c.target.predicate}{c.target.action}"].append(
                    ConnectionProposition(
                        check_kind="connection_with_another_evaluated_guideline"
                        if f"{c.source.predicate}{c.source.action}" in connection_results
                        else "connection_with_existing_guideline",
                        source=c.source,
                        target=c.target,
                        connection_kind=c.kind,
                    )
                )

        return connection_results.values()


class BehavioralChangeEvaluator:
    def __init__(
        self,
        logger: Logger,
        evaluation_store: EvaluationStore,
        guideline_store: GuidelineStore,
        guideline_connection_proposer: GuidelineConnectionProposer,
        coherence_checker: CoherenceChecker,
    ) -> None:
        self.logger = logger
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
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> None:
        if not payload_descriptors:
            raise EvaluationValidationError("No payloads provided for the evaluation task.")

        guideline_payloads = [p for k, p in payload_descriptors if k == PayloadKind.GUIDELINE]

        if guideline_payloads:
            guideline_set = guideline_payloads[0].guideline_set

            if len({p.guideline_set for p in guideline_payloads}) > 1:
                raise EvaluationValidationError(
                    "Evaluation task must be processed for a single guideline_set."
                )

            async def _check_for_duplications() -> None:
                seen_guidelines = set(
                    (
                        g.guideline_set,
                        g.predicate,
                        g.action,
                    )
                    for g in guideline_payloads
                )
                if len(seen_guidelines) < len(guideline_payloads):
                    raise EvaluationValidationError(
                        "Duplicate guideline found among the provided guidelines."
                    )

                existing_guidelines = await self._guideline_store.list_guidelines(
                    guideline_set=guideline_set
                )

                if guideline := next(
                    iter(
                        g
                        for g in existing_guidelines
                        if (
                            guideline_set,
                            g.content.predicate,
                            g.content.action,
                        )
                        in seen_guidelines
                    ),
                    None,
                ):
                    raise EvaluationValidationError(
                        f"Duplicate guideline found against existing guidelines: {str(guideline)} in {guideline_set} guideline_set"
                    )

            await _check_for_duplications()

    async def create_evaluation_task(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
    ) -> EvaluationId:
        await self.validate_payloads(payload_descriptors)

        evaluation = await self._evaluation_store.create_evaluation(payload_descriptors)

        asyncio.create_task(self.run_evaluation(evaluation))

        return evaluation.id

    async def run_evaluation(
        self,
        evaluation: Evaluation,
    ) -> None:
        self.logger.info(f"Starting evaluation task '{evaluation.id}'")

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

            await self._evaluation_store.update_evaluation_status(
                evaluation_id=evaluation.id,
                status=EvaluationStatus.RUNNING,
            )

            guideline_evaluation_data = await self._guideline_evaluator.evaluate(
                payloads=[
                    invoice.payload
                    for invoice in evaluation.invoices
                    if invoice.kind == PayloadKind.GUIDELINE
                ],
            )

            for i, result in enumerate(guideline_evaluation_data):
                invoice_checksum = md5_checksum(str(evaluation.invoices[i].payload))
                state_version = str(hash("Temporarily"))

                invoice = Invoice(
                    kind=evaluation.invoices[i].kind,
                    payload=evaluation.invoices[i].payload,
                    checksum=invoice_checksum,
                    state_version=state_version,
                    approved=True if not result.coherence_checks else False,
                    data=result,
                    error=None,
                )

                await self._evaluation_store.update_evaluation_invoice(
                    evaluation.id,
                    invoice_index=i,
                    updated_invoice=invoice,
                )

            self.logger.info(f"evaluation task '{evaluation.id}' completed")

            await self._evaluation_store.update_evaluation_status(
                evaluation_id=evaluation.id,
                status=EvaluationStatus.COMPLETED,
            )

        except Exception as exc:
            logger_level = "info" if isinstance(exc, EvaluationError) else "error"
            getattr(self.logger, logger_level)(
                f"Evaluation task '{evaluation.id}' failed due to the following error: '{str(exc)}'"
            )
            await self._evaluation_store.update_evaluation_status(
                evaluation_id=evaluation.id,
                status=EvaluationStatus.FAILED,
                error=str(exc),
            )
