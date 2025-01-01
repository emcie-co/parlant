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

from datetime import datetime
from typing import Annotated, Optional, Sequence, TypeAlias, cast
from fastapi import APIRouter, HTTPException, Path, Query, status
from pydantic import Field

from parlant.api import common
from parlant.api.common import (
    GuidelinesCoherenceCheckDTO,
    GuidelineCoherenceCheckKindDTO,
    ConnectionPropositionDTO,
    ConnectionPropositionKindDTO,
    EvaluationStatusDTO,
    GuidelineContentDTO,
    GuidelinePayloadDTO,
    GuidelinePayloadOperationDTO,
    GuidelineInvoiceDataDTO,
    InvoiceDTO,
    InvoiceDataDTO,
    PayloadDTO,
    PayloadKindDTO,
    StyleGuideCoherenceCheckDTO,
    StyleGuideCoherenceCheckKindDTO,
    StyleGuideContentDTO,
    StyleGuideInvoiceDataDTO,
    StyleGuidePayloadDTO,
    StyleGuidePayloadOperationDTO,
    apigen_config,
    ExampleJson,
    ErrorField,
    style_guide_content_dto_to_content,
    style_guide_content_to_dto,
)
from parlant.core.async_utils import Timeout
from parlant.core.common import DefaultBaseModel
from parlant.core.agents import AgentId, AgentStore
from parlant.core.evaluations import (
    Evaluation,
    EvaluationId,
    EvaluationListener,
    EvaluationStatus,
    EvaluationStore,
    GuidelinePayload,
    InvoiceData,
    GuidelineInvoiceData,
    StyleGuideInvoiceData,
    PayloadDescriptor,
    PayloadKind,
    StyleGuidePayload,
)
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
    EvaluationValidationError,
)

API_GROUP = "evaluations"


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


def _payload_descriptor_from_dto(dto: PayloadDTO) -> PayloadDescriptor:
    if dto.kind == PayloadKindDTO.GUIDELINE:
        if not dto.guideline:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing Guideline payload",
            )

        return PayloadDescriptor(
            PayloadKind.GUIDELINE,
            GuidelinePayload(
                content=GuidelineContent(
                    condition=dto.guideline.content.condition,
                    action=dto.guideline.content.action,
                ),
                operation=dto.guideline.operation.value,
                updated_id=dto.guideline.updated_id,
                coherence_check=dto.guideline.coherence_check,
                connection_proposition=dto.guideline.connection_proposition,
            ),
        )

    if dto.kind == PayloadKindDTO.STYLE_GUIDE:
        if not dto.style_guide:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing StyleGuide payload",
            )

        return PayloadDescriptor(
            PayloadKind.STYLE_GUIDE,
            StyleGuidePayload(
                content=style_guide_content_dto_to_content(dto.style_guide.content),
                operation=dto.style_guide.operation.value,
                updated_id=dto.style_guide.updated_id,
                coherence_check=dto.style_guide.coherence_check,
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported DTO kind",
    )


def _guideline_payload_to_dto(payload: GuidelinePayload) -> GuidelinePayloadDTO:
    return GuidelinePayloadDTO(
        content=GuidelineContentDTO(
            condition=payload.content.condition,
            action=payload.content.action,
        ),
        operation=GuidelinePayloadOperationDTO(payload.operation),
        updated_id=payload.updated_id,
        coherence_check=payload.coherence_check,
        connection_proposition=payload.connection_proposition,
    )


def _style_guide_payload_to_dto(payload: StyleGuidePayload) -> StyleGuidePayloadDTO:
    return StyleGuidePayloadDTO(
        content=StyleGuideContentDTO(
            principle=payload.content.principle,
            examples=payload.content.examples,
        ),
        operation=StyleGuidePayloadOperationDTO(payload.operation),
        updated_id=payload.updated_id,
        coherence_check=payload.coherence_check,
    )


def _payload_descriptor_to_dto(descriptor: PayloadDescriptor) -> PayloadDTO:
    if descriptor.kind == PayloadKind.GUIDELINE:
        return PayloadDTO(
            kind=PayloadKindDTO.GUIDELINE,
            guideline=_guideline_payload_to_dto(cast(GuidelinePayload, descriptor.payload)),
        )
    elif descriptor.kind == PayloadKind.STYLE_GUIDE:
        return PayloadDTO(
            kind=PayloadKindDTO.STYLE_GUIDE,
            style_guide=_style_guide_payload_to_dto(cast(StyleGuidePayload, descriptor.payload)),
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported descriptor kind",
    )


def _invoice_guideline_data_to_dto(invoice_data: GuidelineInvoiceData) -> GuidelineInvoiceDataDTO:
    return GuidelineInvoiceDataDTO(
        coherence_checks=[
            GuidelinesCoherenceCheckDTO(
                kind=GuidelineCoherenceCheckKindDTO(c.kind),
                first=GuidelineContentDTO(
                    condition=c.first.condition,
                    action=c.first.action,
                ),
                second=GuidelineContentDTO(
                    condition=c.second.condition,
                    action=c.second.action,
                ),
                issue=c.issue,
                severity=c.severity,
            )
            for c in invoice_data.coherence_checks
        ],
        connection_propositions=[
            ConnectionPropositionDTO(
                check_kind=ConnectionPropositionKindDTO(c.check_kind),
                source=GuidelineContentDTO(
                    condition=c.source.condition,
                    action=c.source.action,
                ),
                target=GuidelineContentDTO(
                    condition=c.target.condition,
                    action=c.target.action,
                ),
            )
            for c in invoice_data.connection_propositions
        ]
        if invoice_data.connection_propositions
        else None,
    )


def _invoice_style_guide_data_to_dto(
    invoice_data: StyleGuideInvoiceData,
) -> StyleGuideInvoiceDataDTO:
    return StyleGuideInvoiceDataDTO(
        coherence_checks=[
            StyleGuideCoherenceCheckDTO(
                kind=StyleGuideCoherenceCheckKindDTO(c.kind),
                first=style_guide_content_to_dto(c.first),
                second=style_guide_content_to_dto(c.second),
                issue=c.issue,
                severity=c.severity,
            )
            for c in invoice_data.coherence_checks
        ]
    )


def _invoice_data_to_dto(kind: PayloadKind, invoice_data: InvoiceData) -> InvoiceDataDTO:
    if kind == PayloadKind.GUIDELINE:
        return InvoiceDataDTO(
            guideline=_invoice_guideline_data_to_dto(cast(GuidelineInvoiceData, invoice_data))
        )

    if kind == PayloadKind.STYLE_GUIDE:
        return InvoiceDataDTO(
            style_guide=_invoice_style_guide_data_to_dto(cast(StyleGuideInvoiceData, invoice_data))
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported descriptor kind",
    )


AgentIdField: TypeAlias = Annotated[
    AgentId,
    Field(
        description="Unique identifier for the agent",
        examples=["a1g2e3n4t5"],
    ),
]


evaluation_creation_params_example: ExampleJson = {
    "agent_id": "a1g2e3n4t5",
    "payloads": [
        {
            "kind": "guideline",
            "guideline": {
                "content": {
                    "condition": "when customer asks about pricing",
                    "action": "provide current pricing information",
                },
                "operation": "add",
                "coherence_check": True,
                "connection_proposition": True,
            },
        }
    ],
}


class EvaluationCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": evaluation_creation_params_example},
):
    """Parameters for creating a new evaluation task"""

    agent_id: AgentIdField
    payloads: Sequence[PayloadDTO]


EvaluationIdPath: TypeAlias = Annotated[
    EvaluationId,
    Path(
        description="Unique identifier of the evaluation to retrieve",
        examples=["eval_123xz"],
    ),
]

EvaluationProgressField: TypeAlias = Annotated[
    float,
    Field(
        description="Progress of the evaluation from 0.0 to 100.0",
        ge=0.0,
        le=100.0,
        examples=[75.0],
    ),
]

CreationUtcField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp when the evaluation was created",
    ),
]


evaluation_example: ExampleJson = {
    "id": "eval_123xz",
    "status": "completed",
    "progress": 100.0,
    "creation_utc": "2024-03-24T12:00:00Z",
    "error": None,
    "invoices": [
        {
            "payload": {
                "kind": "guideline",
                "guideline": {
                    "content": {
                        "condition": "when customer asks about pricing",
                        "action": "provide current pricing information",
                    },
                    "operation": "add",
                    "updated_id": None,
                    "coherence_check": True,
                    "connection_proposition": True,
                },
            },
            "checksum": "abc123def456",
            "approved": True,
            "data": {
                "guideline": {
                    "coherence_checks": [
                        {
                            "kind": "semantic_overlap",
                            "first": {
                                "condition": "when customer asks about pricing",
                                "action": "provide current pricing information",
                            },
                            "second": {
                                "condition": "if customer inquires about cost",
                                "action": "share the latest pricing details",
                            },
                            "issue": "These guidelines handle similar scenarios",
                            "severity": "warning",
                        }
                    ],
                    "connection_propositions": [
                        {
                            "check_kind": "semantic_similarity",
                            "source": {
                                "condition": "when customer asks about pricing",
                                "action": "provide current pricing information",
                            },
                            "target": {
                                "condition": "if customer inquires about cost",
                                "action": "share the latest pricing details",
                            },
                        }
                    ],
                }
            },
            "error": None,
        }
    ],
}


class EvaluationDTO(
    DefaultBaseModel,
    json_schema_extra={"example": evaluation_example},
):
    """An evaluation task information tracking analysis of payloads."""

    id: EvaluationIdPath
    status: EvaluationStatusDTO
    progress: EvaluationProgressField
    creation_utc: CreationUtcField
    error: Optional[ErrorField] = None
    invoices: Sequence[InvoiceDTO]


WaitForCompletionQuery: TypeAlias = Annotated[
    int,
    Query(
        description="Maximum time in seconds to wait for evaluation completion",
        ge=0,
    ),
]


def create_router(
    evaluation_service: BehavioralChangeEvaluator,
    evaluation_store: EvaluationStore,
    evaluation_listener: EvaluationListener,
    agent_store: AgentStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/evaluations",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_evaluation",
        response_model=EvaluationDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Evaluation successfully created. Returns the initial evaluation state.",
                "content": common.example_json_content(evaluation_example),
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in evaluation parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_evaluation(
        params: EvaluationCreationParamsDTO,
    ) -> EvaluationDTO:
        """
        Creates a new evaluation task for the specified agent.

        An evaluation analyzes proposed changes (payloads) to an agent's guidelines
        to ensure coherence and consistency with existing guidelines and the agent's
        configuration. This helps maintain predictable agent behavior by detecting
        potential conflicts and unintended consequences before applying changes.

        Returns immediately with the created evaluation's initial state.
        """
        try:
            agent = await agent_store.read_agent(agent_id=params.agent_id)
            evaluation_id = await evaluation_service.create_evaluation_task(
                agent=agent,
                payload_descriptors=[_payload_descriptor_from_dto(p) for p in params.payloads],
            )
        except EvaluationValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )

        evaluation = await evaluation_store.read_evaluation(evaluation_id)
        return _evaluation_to_dto(evaluation)

    @router.get(
        "/evaluations/{evaluation_id}",
        operation_id="read_evaluation",
        response_model=EvaluationDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Evaluation details successfully retrieved.",
                "content": common.example_json_content(evaluation_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Evaluation not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in evaluation parameters"
            },
            status.HTTP_504_GATEWAY_TIMEOUT: {
                "description": "Timeout waiting for evaluation completion"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_evaluation(
        evaluation_id: EvaluationIdPath,
        wait_for_completion: WaitForCompletionQuery = 60,
    ) -> EvaluationDTO:
        """Retrieves the current state of an evaluation.

        * If wait_for_completion == 0, returns current state immediately.
        * If wait_for_completion > 0, waits for completion/failure or timeout. Defaults to 60.

        Notes:
        When wait_for_completion > 0:
        - Returns final state if evaluation completes within timeout
        - Raises 504 if timeout is reached before completion
        """
        if wait_for_completion > 0:
            if not await evaluation_listener.wait_for_completion(
                evaluation_id=evaluation_id,
                timeout=Timeout(wait_for_completion),
            ):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request timed out",
                )

        evaluation = await evaluation_store.read_evaluation(evaluation_id=evaluation_id)
        return _evaluation_to_dto(evaluation)

    def _evaluation_to_dto(evaluation: Evaluation) -> EvaluationDTO:
        return EvaluationDTO(
            id=evaluation.id,
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
