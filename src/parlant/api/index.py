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
    CoherenceCheckDTO,
    CoherenceCheckKindDTO,
    ConnectionPropositionDTO,
    ConnectionPropositionKindDTO,
    EvaluationStatusDTO,
    GuidelineContentDTO,
    LegacyGuidelinePayloadDTO,
    GuidelinePayloadOperationDTO,
    LegacyGuidelineInvoiceDataDTO,
    LegacyInvoiceDataDTO,
    LegacyPayloadDTO,
    PayloadKindDTO,
    ExampleJson,
    apigen_skip_config,
    operation_dto_to_operation,
)
from parlant.core.async_utils import Timeout
from parlant.core.common import DefaultBaseModel
from parlant.core.agents import AgentId, AgentStore
from parlant.core.evaluations import (
    CoherenceCheckKind,
    EntailmentRelationshipPropositionKind,
    Evaluation,
    EvaluationId,
    EvaluationListener,
    EvaluationStatus,
    EvaluationStore,
    GuidelinePayload,
    InvoiceGuidelineData,
    PayloadOperation,
    Payload,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.behavioral_change_evaluation import (
    LegacyBehavioralChangeEvaluator,
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


def _payload_from_dto(dto: LegacyPayloadDTO) -> Payload:
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
            tool_ids=[],
            operation=operation_dto_to_operation(dto.guideline.operation),
            updated_id=dto.guideline.updated_id,
            coherence_check=dto.guideline.coherence_check,
            connection_proposition=dto.guideline.connection_proposition,
            action_proposition=False,
            properties_proposition=False,
            journey_node_proposition=False,
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported DTO kind",
    )


def _operation_to_operation_dto(
    operation: PayloadOperation,
) -> GuidelinePayloadOperationDTO:
    if dto := {
        PayloadOperation.ADD: GuidelinePayloadOperationDTO.ADD,
        PayloadOperation.UPDATE: GuidelinePayloadOperationDTO.UPDATE,
    }.get(operation):
        return dto

    raise ValueError(f"Unsupported operation: {operation}")


def _payload_descriptor_to_dto(descriptor: PayloadDescriptor) -> LegacyPayloadDTO:
    if descriptor.kind == PayloadKind.GUIDELINE:
        return LegacyPayloadDTO(
            kind=PayloadKindDTO.GUIDELINE,
            guideline=LegacyGuidelinePayloadDTO(
                content=GuidelineContentDTO(
                    condition=cast(GuidelinePayload, descriptor.payload).content.condition,
                    action=cast(GuidelinePayload, descriptor.payload).content.action,
                ),
                operation=_operation_to_operation_dto(
                    cast(GuidelinePayload, descriptor.payload).operation
                ),
                updated_id=cast(GuidelinePayload, descriptor.payload).updated_id,
                coherence_check=cast(GuidelinePayload, descriptor.payload).coherence_check,
                connection_proposition=cast(
                    GuidelinePayload, descriptor.payload
                ).connection_proposition,
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported descriptor kind",
    )


def _coherence_check_kind_to_dto(
    kind: CoherenceCheckKind,
) -> CoherenceCheckKindDTO:
    match kind:
        case CoherenceCheckKind.CONTRADICTION_WITH_EXISTING_GUIDELINE:
            return CoherenceCheckKindDTO.CONTRADICTION_WITH_EXISTING_GUIDELINE
        case CoherenceCheckKind.CONTRADICTION_WITH_ANOTHER_EVALUATED_GUIDELINE:
            return CoherenceCheckKindDTO.CONTRADICTION_WITH_ANOTHER_EVALUATED_GUIDELINE


def _connection_proposition_kind_to_dto(
    kind: EntailmentRelationshipPropositionKind,
) -> ConnectionPropositionKindDTO:
    match kind:
        case EntailmentRelationshipPropositionKind.CONNECTION_WITH_EXISTING_GUIDELINE:
            return ConnectionPropositionKindDTO.CONNECTION_WITH_EXISTING_GUIDELINE
        case EntailmentRelationshipPropositionKind.CONNECTION_WITH_ANOTHER_EVALUATED_GUIDELINE:
            return ConnectionPropositionKindDTO.CONNECTION_WITH_ANOTHER_EVALUATED_GUIDELINE


def _invoice_data_to_dto(
    kind: PayloadKind, invoice_data: InvoiceGuidelineData
) -> LegacyInvoiceDataDTO:
    if kind == PayloadKind.GUIDELINE:
        return LegacyInvoiceDataDTO(
            guideline=LegacyGuidelineInvoiceDataDTO(
                coherence_checks=[
                    CoherenceCheckDTO(
                        kind=_coherence_check_kind_to_dto(c.kind),
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
                ]
                if invoice_data.coherence_checks
                else [],
                connection_propositions=[
                    ConnectionPropositionDTO(
                        check_kind=_connection_proposition_kind_to_dto(c.check_kind),
                        source=GuidelineContentDTO(
                            condition=c.source.condition,
                            action=c.source.action,
                        ),
                        target=GuidelineContentDTO(
                            condition=c.target.condition,
                            action=c.target.action,
                        ),
                    )
                    for c in invoice_data.entailment_propositions
                ]
                if invoice_data.entailment_propositions
                else None,
            )
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Unsupported descriptor kind",
    )


LegacyChecksumField: TypeAlias = Annotated[
    str,
    Field(
        description="Checksum of the invoice content",
        examples=["abc123def456"],
    ),
]

LegacyApprovedField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether the evaluation task the invoice represents has been approved",
        examples=[True],
    ),
]


LegacyErrorField: TypeAlias = Annotated[
    str,
    Field(
        description="Error message if the evaluation failed",
        examples=["Failed to process evaluation due to invalid payload"],
    ),
]


legacy_invoice_example: ExampleJson = {
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


class LegacyInvoiceDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_invoice_example},
):
    """Represents the result of evaluating a single payload in an evaluation task.

    An invoice is a comprehensive record of the evaluation results for a single payload.
    """

    payload: LegacyPayloadDTO
    checksum: LegacyChecksumField
    approved: LegacyApprovedField
    data: Optional[LegacyInvoiceDataDTO] = None
    error: Optional[LegacyErrorField] = None


LegacyAgentIdField: TypeAlias = Annotated[
    AgentId,
    Field(
        description="Unique identifier for the agent",
        examples=["a1g2e3n4t5"],
    ),
]


legacy_evaluation_creation_params_example: ExampleJson = {
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


class LegacyEvaluationCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_evaluation_creation_params_example},
):
    """Parameters for creating a new evaluation task"""

    agent_id: LegacyAgentIdField
    payloads: Sequence[LegacyPayloadDTO]


LegacyEvaluationIdPath: TypeAlias = Annotated[
    EvaluationId,
    Path(
        description="Unique identifier of the evaluation to retrieve",
        examples=["eval_123xz"],
    ),
]

LegacyEvaluationProgressField: TypeAlias = Annotated[
    float,
    Field(
        description="Progress of the evaluation from 0.0 to 100.0",
        ge=0.0,
        le=100.0,
        examples=[75.0],
    ),
]

LegacyCreationUtcField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp when the evaluation was created",
    ),
]


legacy_evaluation_example: ExampleJson = {
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


class LegacyEvaluationDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_evaluation_example},
):
    """An evaluation task information tracking analysis of payloads."""

    id: LegacyEvaluationIdPath
    status: EvaluationStatusDTO
    progress: LegacyEvaluationProgressField
    creation_utc: LegacyCreationUtcField
    error: Optional[LegacyErrorField] = None
    invoices: Sequence[LegacyInvoiceDTO]


LegacyWaitForCompletionQuery: TypeAlias = Annotated[
    int,
    Query(
        description="Maximum time in seconds to wait for evaluation completion",
        ge=0,
    ),
]


def legacy_create_router(
    evaluation_service: LegacyBehavioralChangeEvaluator,
    evaluation_store: EvaluationStore,
    evaluation_listener: EvaluationListener,
    agent_store: AgentStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/evaluations",
        status_code=status.HTTP_201_CREATED,
        operation_id="legacy_create_evaluation",
        response_model=LegacyEvaluationDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Evaluation successfully created. Returns the initial evaluation state.",
                "content": common.example_json_content(legacy_evaluation_example),
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in evaluation parameters"
            },
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def create_evaluation(
        params: LegacyEvaluationCreationParamsDTO,
    ) -> LegacyEvaluationDTO:
        """
        DEPRECATED AND WILL REMOVED IN A FUTURE RELEASE.

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
        "/evaluations/{evaluation_id}",
        operation_id="legacy_read_evaluation",
        response_model=LegacyEvaluationDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Evaluation details successfully retrieved.",
                "content": common.example_json_content(legacy_evaluation_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Evaluation not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in evaluation parameters"
            },
            status.HTTP_504_GATEWAY_TIMEOUT: {
                "description": "Timeout waiting for evaluation completion"
            },
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def read_evaluation(
        evaluation_id: LegacyEvaluationIdPath,
        wait_for_completion: LegacyWaitForCompletionQuery = 60,
    ) -> LegacyEvaluationDTO:
        """DEPRECATED AND WILL REMOVED IN A FUTURE RELEASE:
        Retrieves the current state of an evaluation.

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

    def _evaluation_to_dto(evaluation: Evaluation) -> LegacyEvaluationDTO:
        return LegacyEvaluationDTO(
            id=evaluation.id,
            status=_evaluation_status_to_dto(evaluation.status),
            progress=evaluation.progress,
            creation_utc=evaluation.creation_utc,
            invoices=[
                LegacyInvoiceDTO(
                    payload=_payload_descriptor_to_dto(
                        PayloadDescriptor(kind=invoice.kind, payload=invoice.payload)
                    ),
                    checksum=invoice.checksum,
                    approved=invoice.approved,
                    data=_invoice_data_to_dto(
                        invoice.kind, cast(InvoiceGuidelineData, invoice.data)
                    )
                    if invoice.data
                    else None,
                    error=invoice.error,
                )
                for invoice in evaluation.invoices
            ],
            error=evaluation.error,
        )

    return router
