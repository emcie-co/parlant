from datetime import datetime
from typing import Optional, Sequence, cast
from fastapi import APIRouter, HTTPException, status

from parlant.api.common import (
    CoherenceCheckDTO,
    ConnectionPropositionDTO,
    EvaluationStatusDTO,
    GuidelineContentDTO,
    InvoiceDataDTO,
    GuidelineInvoiceDataDTO,
    PayloadDTO,
    PayloadKindDTO,
    connection_kind_to_dto,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.agents import AgentId, AgentStore
from parlant.core.evaluations import (
    EvaluationId,
    EvaluationStatus,
    EvaluationStore,
    GuidelinePayload,
    InvoiceData,
    Payload,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
    EvaluationValidationError,
)


def _evaluation_status_to_dto(
    status: EvaluationStatus,
) -> EvaluationStatusDTO:
    return cast(
        EvaluationStatusDTO,
        {
            EvaluationStatus.PENDING: "pending",
            EvaluationStatus.RUNNING: "running",
            EvaluationStatus.COMPLETED: "completed",
            EvaluationStatus.FAILED: "failed",
        }[status],
    )


def _payload_from_dto(dto: PayloadDTO) -> Payload:
    return {
        PayloadKindDTO.GUIDELINE: GuidelinePayload(
            content=GuidelineContent(
                predicate=dto.content.predicate,
                action=dto.content.action,
            ),
            operation=dto.operation,
            updated_id=dto.updated_id,
            coherence_check=dto.coherence_check,
            connection_proposition=dto.connection_proposition,
        )
    }[dto.kind]


def _payload_descriptor_to_dto(descriptor: PayloadDescriptor) -> PayloadDTO:
    return {
        PayloadKind.GUIDELINE: PayloadDTO(
            kind=PayloadKindDTO.GUIDELINE,
            content=GuidelineContentDTO(
                predicate=descriptor.payload.content.predicate,
                action=descriptor.payload.content.action,
            ),
            operation=descriptor.payload.operation,
            updated_id=descriptor.payload.updated_id,
            coherence_check=descriptor.payload.coherence_check,
            connection_proposition=descriptor.payload.connection_proposition,
        )
    }[descriptor.kind]


def _invoice_data_to_dto(kind: PayloadKind, invoice_data: InvoiceData) -> InvoiceDataDTO:
    return {
        PayloadKind.GUIDELINE: GuidelineInvoiceDataDTO(
            coherence_checks=[
                CoherenceCheckDTO(
                    kind=c.kind,
                    first=GuidelineContent(
                        predicate=c.first.predicate,
                        action=c.first.action,
                    ),
                    second=GuidelineContent(
                        predicate=c.second.predicate,
                        action=c.second.action,
                    ),
                    issue=c.issue,
                    severity=c.severity,
                )
                for c in invoice_data.coherence_checks
            ],
            connection_propositions=[
                ConnectionPropositionDTO(
                    check_kind=c.check_kind,
                    source=GuidelineContent(
                        predicate=c.source.predicate,
                        action=c.source.action,
                    ),
                    target=GuidelineContent(
                        predicate=c.target.predicate,
                        action=c.target.action,
                    ),
                    connection_kind=connection_kind_to_dto(c.connection_kind),
                )
                for c in invoice_data.connection_propositions
            ]
            if invoice_data.connection_propositions
            else None,
        )
    }[kind]


class InvoiceDTO(DefaultBaseModel):
    payload: PayloadDTO
    checksum: str
    approved: bool
    data: Optional[InvoiceDataDTO]
    error: Optional[str]


class CreateEvaluationRequest(DefaultBaseModel):
    payloads: Sequence[PayloadDTO]


class CreateEvaluationResponse(DefaultBaseModel):
    evaluation_id: EvaluationId


class ReadEvaluationResponse(DefaultBaseModel):
    evaluation_id: EvaluationId
    status: EvaluationStatusDTO
    progress: float
    creation_utc: datetime
    error: Optional[str]
    invoices: list[InvoiceDTO]


def create_router(
    evaluation_service: BehavioralChangeEvaluator,
    evaluation_store: EvaluationStore,
    agent_store: AgentStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/{agent_id}/index/evaluations",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_evaluation",
    )
    async def create_evaluation(
        agent_id: AgentId, request: CreateEvaluationRequest
    ) -> CreateEvaluationResponse:
        try:
            agent = await agent_store.read_agent(agent_id=agent_id)
            evaluation_id = await evaluation_service.create_evaluation_task(
                agent=agent,
                payload_descriptors=[
                    PayloadDescriptor(PayloadKind.GUIDELINE, p)
                    for p in [_payload_from_dto(p) for p in request.payloads]
                ],
            )
        except EvaluationValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )

        return CreateEvaluationResponse(evaluation_id=evaluation_id)

    @router.get("/index/evaluations/{evaluation_id}", operation_id="read_evaluation")
    async def read_evaluation(evaluation_id: EvaluationId) -> ReadEvaluationResponse:
        evaluation = await evaluation_store.read_evaluation(evaluation_id=evaluation_id)

        return ReadEvaluationResponse(
            evaluation_id=evaluation.id,
            status=_evaluation_status_to_dto(evaluation.status),
            progress=evaluation.progress,
            creation_utc=evaluation.creation_utc,
            invoices=[
                InvoiceDTO(
                    payload=_payload_descriptor_to_dto(
                        PayloadDescriptor(kind=invoice.kind, payload=invoice.payload)
                    ),
                    checksum=invoice.checksum,
                    approved=invoice.approved,
                    data=_invoice_data_to_dto(invoice.kind, invoice.data) if invoice.data else None,
                    error=invoice.error,
                )
                for invoice in evaluation.invoices
            ],
            error=evaluation.error,
        )

    return router
