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
from typing import Annotated, Optional, Sequence, TypeAlias, cast
from fastapi import APIRouter, HTTPException, Path, Query, status
from pydantic import Field

from parlant.api import common
from parlant.api.common import (
    EvaluationStatusDTO,
    GuidelineContentDTO,
    GuidelineIdField,
    GuidelinePayloadOperationDTO,
    JSONSerializableDTO,
    PayloadKindDTO,
    ExampleJson,
    ToolIdDTO,
    apigen_config,
    operation_dto_to_operation,
)
from parlant.core.async_utils import Timeout
from parlant.core.common import DefaultBaseModel
from parlant.core.evaluations import (
    Evaluation,
    EvaluationId,
    EvaluationListener,
    EvaluationStatus,
    EvaluationStore,
    GuidelinePayload,
    GuidelinePayloadOperation,
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
from parlant.core.tools import ToolId

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


GuidelinePayloadActionPropositionField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether the action proposition is enabled",
        examples=[True],
    ),
]

GuidelinePayloadPropertiesPropositionField: TypeAlias = Annotated[
    bool,
    Field(
        description="Properties proposition",
        examples=[{"action_proposition": True}],
    ),
]

guideline_payload_example: ExampleJson = {
    "content": {
        "condition": "User asks about product pricing",
        "action": "Provide current price list and any active discounts",
    },
    "tool_ids": ["google_calendar:get_events"],
    "operation": "add",
    "updated_id": None,
    "action_proposition": True,
    "properties_proposition": {"continuous": True},
}


class GuidelinePayloadDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_payload_example},
):
    """Payload data for a Guideline operation"""

    content: GuidelineContentDTO
    tool_ids: Sequence[ToolIdDTO]
    operation: GuidelinePayloadOperationDTO
    updated_id: Optional[GuidelineIdField] = None
    action_proposition: GuidelinePayloadActionPropositionField
    properties_proposition: GuidelinePayloadPropertiesPropositionField


payload_example: ExampleJson = {
    "kind": "guideline",
    "guideline": {
        "content": {
            "condition": "User asks about product pricing",
            "action": None,
        },
        "operation": "add",
        "updated_id": None,
        "action_proposition": True,
        "properties_proposition": True,
    },
}


class PayloadDTO(
    DefaultBaseModel,
    json_schema_extra={"example": payload_example},
):
    kind: PayloadKindDTO
    guideline: Optional[GuidelinePayloadDTO] = None


action_proposition_example: ExampleJson = {
    "content": {
        "condition": "User asks about product pricing",
        "action": "Provide current price list and any active discounts",
    },
}

properties_proposition_example: ExampleJson = {
    "continious": True,
}


ChecksumField: TypeAlias = Annotated[
    str,
    Field(
        description="Checksum of the invoice content",
        examples=["abc123def456"],
    ),
]

ApprovedField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether the evaluation task the invoice represents has been approved",
        examples=[True],
    ),
]


ErrorField: TypeAlias = Annotated[
    str,
    Field(
        description="Error message if the evaluation failed",
        examples=["Failed to process evaluation due to invalid payload"],
    ),
]


ActionPropositionField: TypeAlias = Annotated[
    str,
    Field(
        description="Proposed action proposition",
        examples=["provide current pricing information"],
    ),
]

PropertiesPropositionField: TypeAlias = Annotated[
    Optional[dict[str, JSONSerializableDTO]],
    Field(
        description="Properties proposition",
        examples=[{"continious": True}],
    ),
]

invoice_example: ExampleJson = {
    "payload": {
        "kind": "guideline",
        "guideline": {
            "content": {
                "condition": "when customer asks about pricing",
                "action": "provide current pricing information",
            },
            "operation": "add",
            "updated_id": None,
            "action_proposition": True,
            "properties_proposition": True,
        },
    },
    "checksum": "abc123def456",
    "approved": True,
    "data": {
        "guideline": {
            "action_proposition": {
                "content": {
                    "condition": "when customer asks about pricing",
                    "action": "provide current pricing information",
                },
                "properties_proposition": {
                    "continious": True,
                },
            },
        }
    },
    "error": None,
}

guideline_invoice_data_example: ExampleJson = {
    "action_proposition": action_proposition_example,
    "properties_proposition": properties_proposition_example,
}


class GuidelineInvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_invoice_data_example},
):
    """Evaluation results for a Guideline, including action propositions"""

    action_proposition: ActionPropositionField
    properties_proposition: Optional[PropertiesPropositionField] = None


invoice_data_example: ExampleJson = {"guideline": guideline_invoice_data_example}


class InvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": invoice_data_example},
):
    """
    Contains the relevant invoice data.

    At this point only `guideline` is suppoerted.
    """

    guideline: Optional[GuidelineInvoiceDataDTO] = None


class InvoiceDTO(
    DefaultBaseModel,
    json_schema_extra={"example": invoice_example},
):
    """Represents the result of evaluating a single payload in an evaluation task.

    An invoice is a comprehensive record of the evaluation results for a single payload.
    """

    payload: PayloadDTO
    checksum: ChecksumField
    approved: ApprovedField
    data: Optional[InvoiceDataDTO] = None
    error: Optional[ErrorField] = None


def _payload_from_dto(dto: PayloadDTO) -> Payload:
    if dto.kind == PayloadKindDTO.GUIDELINE:
        if not dto.guideline:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing Guideline payload",
            )

        return GuidelinePayload(
            content=GuidelineContent(
                condition=dto.guideline.content.condition,
                action=dto.guideline.content.action,
            ),
            tool_ids=[
                ToolId(service_name=t.service_name, tool_name=t.tool_name)
                for t in dto.guideline.tool_ids
            ],
            operation=operation_dto_to_operation(dto.guideline.operation),
            updated_id=dto.guideline.updated_id,
            coherence_check=False,  # Legacy and will be removed in the future
            connection_proposition=False,  # Legacy and will be removed in the future
            action_proposition=dto.guideline.action_proposition,
            properties_proposition=dto.guideline.properties_proposition,
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported DTO kind",
    )


def _operation_to_operation_dto(
    operation: GuidelinePayloadOperation,
) -> GuidelinePayloadOperationDTO:
    if dto := {
        GuidelinePayloadOperation.ADD: GuidelinePayloadOperationDTO.ADD,
        GuidelinePayloadOperation.UPDATE: GuidelinePayloadOperationDTO.UPDATE,
    }.get(operation):
        return dto

    raise ValueError(f"Unsupported operation: {operation}")


def _payload_descriptor_to_dto(descriptor: PayloadDescriptor) -> PayloadDTO:
    if descriptor.kind == PayloadKind.GUIDELINE:
        return PayloadDTO(
            kind=PayloadKindDTO.GUIDELINE,
            guideline=GuidelinePayloadDTO(
                content=GuidelineContentDTO(
                    condition=descriptor.payload.content.condition,
                    action=descriptor.payload.content.action,
                ),
                tool_ids=[
                    ToolIdDTO(service_name=t.service_name, tool_name=t.tool_name)
                    for t in descriptor.payload.tool_ids
                ],
                operation=_operation_to_operation_dto(descriptor.payload.operation),
                updated_id=descriptor.payload.updated_id,
                action_proposition=descriptor.payload.properties_proposition,
                properties_proposition=descriptor.payload.properties_proposition,
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported descriptor kind",
    )


def _invoice_data_to_dto(
    kind: PayloadKind,
    invoice_data: InvoiceData,
) -> InvoiceDataDTO:
    if kind == PayloadKind.GUIDELINE:
        return InvoiceDataDTO(
            guideline=GuidelineInvoiceDataDTO(
                action_proposition=invoice_data.action_proposition,
                properties_proposition=invoice_data.properties_proposition,
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported descriptor kind",
    )


evaluation_creation_params_example: ExampleJson = {
    "agent_id": "a1g2e3n4t5",
    "payloads": [
        {
            "kind": "guideline",
            "guideline": {
                "content": {
                    "condition": "when customer asks about pricing",
                    "action": None,
                },
                "operation": "add",
                "action_proposition": True,
            },
        }
    ],
}


class EvaluationCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": evaluation_creation_params_example},
):
    """Parameters for creating a new evaluation task"""

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
                    "action_proposition": True,
                    "properties_proposition": True,
                },
            },
            "checksum": "abc123def456",
            "approved": True,
            "data": {
                "guideline": {
                    "action_proposition": "provide current pricing information",
                    "properties_proposition": {"continious": True},
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
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
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
        Creates a new evaluation task for the specified payloads.

        Returns immediately with the created evaluation's initial state.
        """
        try:
            evaluation_id = await evaluation_service.create_evaluation_task(
                payload_descriptors=[
                    PayloadDescriptor(PayloadKind.GUIDELINE, p)
                    for p in [_payload_from_dto(p) for p in params.payloads]
                ],
            )
        except EvaluationValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )

        evaluation = await evaluation_store.read_evaluation(evaluation_id)
        return _evaluation_to_dto(evaluation)

    @router.get(
        "/{evaluation_id}",
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
