from datetime import datetime
from typing import Literal, Optional, Sequence, TypeAlias, TypedDict, Union, cast
from fastapi import APIRouter, HTTPException, status


from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.evaluations import (
    CoherenceCheckKind,
    ConnectionPropositionKind,
    EvaluationId,
    EvaluationStatus,
    EvaluationStore,
    GuidelinePayload,
    InvoiceData,
    Payload,
    PayloadDescriptor,
    PayloadKind,
)
from emcie.server.core.guideline_connections import ConnectionKind
from emcie.server.core.guidelines import GuidelineContent
from emcie.server.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
    EvaluationValidationError,
)

EvaluationStatusDTO = Literal["pending", "running", "completed", "failed"]


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


ConnectionKindDTO = Literal["entails", "suggests"]


def _connection_kind_to_dto(kind: ConnectionKind) -> ConnectionKindDTO:
    return cast(
        ConnectionKindDTO,
        {
            ConnectionKind.ENTAILS: "entails",
            ConnectionKind.SUGGESTS: "suggests",
        }[kind],
    )


PayloadKindDTO = Literal["guideline"]


class GuidelinePayloadDTO(TypedDict):
    kind: PayloadKindDTO
    guideline_set: str
    predicate: str
    action: str


PayloadDTO: TypeAlias = Union[GuidelinePayloadDTO]


def _payload_from_dto(dto: PayloadDTO) -> Payload:
    return {
        "guideline": GuidelinePayload(
            guideline_set=dto["guideline_set"],
            predicate=dto["predicate"],
            action=dto["action"],
        )
    }[dto["kind"]]


def _payload_descriptor_to_dto(descriptor: PayloadDescriptor) -> PayloadDTO:
    return {
        PayloadKind.GUIDELINE: PayloadDTO(
            kind="guideline",
            guideline_set=descriptor.payload.guideline_set,
            predicate=descriptor.payload.predicate,
            action=descriptor.payload.action,
        )
    }[descriptor.kind]


class CoherenceCheckDTO(DefaultBaseModel):
    kind: CoherenceCheckKind
    first: GuidelineContent
    second: GuidelineContent
    issue: str
    severity: int


class ConnectionPropositionDTO(DefaultBaseModel):
    check_kind: ConnectionPropositionKind
    source: GuidelineContent
    target: GuidelineContent
    connection_kind: ConnectionKindDTO


class InvoiceGuidelineDataDTO(DefaultBaseModel):
    coherence_checks: list[CoherenceCheckDTO]
    connection_propositions: Optional[list[ConnectionPropositionDTO]]


InvoiceDataDTO: TypeAlias = Union[InvoiceGuidelineDataDTO]


class CreateEvaluationRequest(DefaultBaseModel):
    payloads: Sequence[PayloadDTO]


class CreateEvaluationResponse(DefaultBaseModel):
    evaluation_id: EvaluationId


class InvoiceDTO(DefaultBaseModel):
    payload: PayloadDTO
    checksum: str
    approved: bool
    data: Optional[InvoiceDataDTO]
    error: Optional[str]


def _invoice_data_to_dto(kind: PayloadKind, invoice_data: InvoiceData) -> InvoiceDataDTO:
    return {
        PayloadKind.GUIDELINE: InvoiceGuidelineDataDTO(
            coherence_checks=[
                CoherenceCheckDTO(
                    kind=c.kind,
                    first=c.first,
                    second=c.second,
                    issue=c.issue,
                    severity=c.severity,
                )
                for c in invoice_data.coherence_checks
            ],
            connection_propositions=[
                ConnectionPropositionDTO(
                    check_kind=c.check_kind,
                    source=c.source,
                    target=c.target,
                    connection_kind=_connection_kind_to_dto(c.connection_kind),
                )
                for c in invoice_data.connection_propositions
            ]
            if invoice_data.connection_propositions
            else None,
        )
    }[kind]


class ReadEvaluationResponse(DefaultBaseModel):
    evaluation_id: EvaluationId
    status: EvaluationStatusDTO
    creation_utc: datetime
    error: Optional[str]
    invoices: list[InvoiceDTO]


def create_router(
    evaluation_service: BehavioralChangeEvaluator,
    evaluation_store: EvaluationStore,
) -> APIRouter:
    router = APIRouter()

    @router.post("/evaluations")
    async def create_evaluation(request: CreateEvaluationRequest) -> CreateEvaluationResponse:
        try:
            evaluation_id = await evaluation_service.create_evaluation_task(
                payload_descriptors=[
                    PayloadDescriptor(PayloadKind.GUIDELINE, p)
                    for p in [_payload_from_dto(p) for p in request.payloads]
                ]
            )
        except EvaluationValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )

        return CreateEvaluationResponse(evaluation_id=evaluation_id)

    @router.get("/evaluations/{evaluation_id}")
    async def get_evaluation(evaluation_id: EvaluationId) -> ReadEvaluationResponse:
        evaluation = await evaluation_store.read_evaluation(evaluation_id=evaluation_id)

        return ReadEvaluationResponse(
            evaluation_id=evaluation.id,
            status=_evaluation_status_to_dto(evaluation.status),
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
